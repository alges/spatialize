#ifndef _SPTLZ_DECODERS_IDW_
#define _SPTLZ_DECODERS_IDW_

#include <stdexcept>
#include <cmath>
#include "spatialize/decoder.hpp"
#include "spatialize/utils.hpp"

namespace sptlz{
  // Inverse-distance prediction with weights relative to the nearest datum,
  //   w_j = b_j (d_min / d_j)^p,
  // the same normalised weights as b_j / d_j^p, with the largest equal to 1: no power under- or
  // overflows, so the prediction does not depend on the units of the coordinates (1/(eps + d^p)
  // vanished for large coordinates and fitted exponents). Data at distance 0 take all the weight
  // (their mean). `each(f)` calls f(distance, value, boost) for every datum; NaN without data.
  template <typename Each>
  inline float relative_idw(Each each, float p){
    float dmin = INFINITY;
    int n = 0;
    each([&](float d, float, float){ if(d < dmin) dmin = d; n++; });
    if(n == 0) return(NAN);
    float s = 0.0f, sv = 0.0f;
    if(dmin == 0.0f){
      each([&](float d, float v, float){ if(d == 0.0f){ s += 1.0f; sv += v; } });
      return(sv/s);
    }
    each([&](float d, float v, float b){
      float w = b*std::pow(dmin/d, p);
      s += w;
      sv += w*v;
    });
    return((s > 0.0f) ? sv/s : NAN);
  }

  // the weights themselves (one per datum), with the same convention
  inline std::vector<float> relative_idw_weights(const std::vector<float> &d, const std::vector<float> &boost, float p){
    std::vector<float> w(d.size(), 0.0f);
    float dmin = INFINITY;
    for(float x: d) if(x < dmin) dmin = x;
    for(size_t j=0; j<d.size(); j++){
      w[j] = (dmin == 0.0f) ? (d[j] == 0.0f ? 1.0f : 0.0f) : boost[j]*std::pow(dmin/d[j], p);
    }
    return(w);
  }

  // the weights without boosts, (d_min / d_j)^p
  inline std::vector<float> relative_idw_weights(const std::vector<float> &d, float p){
    std::vector<float> w(d.size(), 0.0f);
    float dmin = INFINITY;
    for(float x: d) if(x < dmin) dmin = x;
    for(size_t j=0; j<d.size(); j++){
      w[j] = (dmin == 0.0f) ? (d[j] == 0.0f ? 1.0f : 0.0f) : std::pow(dmin/d[j], p);
    }
    return(w);
  }

  // IDW prediction at `point` from the samples `ids`, with the weights of relative_idw: 1/d^p
  // relative to the nearest datum, (d_min/d)^p, so no power under- or overflows whatever the units of
  // the coordinates and the exponent (1/d^p was infinite for small coordinates and 0 for large ones).
  // Data at distance 0 take all the weight (a query on a datum gets its value; duplicated locations,
  // their mean).
  inline float idw_predict(std::vector<float> *point, std::vector<std::vector<float>> *coords,
                           std::vector<float> *values, const std::vector<int> &ids, float exponent){
    std::vector<float> distances(ids.size());
    for(size_t j=0; j<ids.size(); j++) distances[j] = distance(point, &(coords->at(ids[j])));
    return(relative_idw([&](auto f){
      for(size_t j=0; j<ids.size(); j++) f(distances[j], values->at(ids[j]), 1.0f);
    }, exponent));
  }

  // Inverse distance weighting inside a cell, with the same weights 1/d^p in estimation,
  // leave-one-out and k-fold (see idw_predict).
  class IDWDecoder: public Decoder {
    protected:
      float exponent;

    public:
      IDWDecoder(float _exponent){
        this->exponent = _exponent;
      }

      std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params, const CellContext &cell){
        std::vector<float> result;

        if(samples_id->size()==0){
          result.assign(locations_id->size(), NAN);
          return(result);
        }

        for(size_t i=0; i<locations_id->size(); i++){
          result.push_back(idw_predict(&(locations->at(locations_id->at(i))), coords, values, *samples_id, exponent));
        }
        return(result);
      }

      std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params, const CellContext &cell){
        std::vector<float> result;

        if((samples_id->size()==0) || (samples_id->size()==1)){
          result.assign(samples_id->size(), NAN);
          return(result);
        }

        // every sample is predicted from the others of its cell
        for(size_t i=0; i<samples_id->size(); i++){
          std::vector<int> others;
          for(size_t j=0; j<samples_id->size(); j++){
            if(i!=j){
              others.push_back(samples_id->at(j));
            }
          }
          result.push_back(idw_predict(&(coords->at(samples_id->at(i))), coords, values, others, exponent));
        }
        return(result);
      }

      std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *folds, std::vector<int> *samples_id, std::vector<float> *params, const CellContext &cell){
        std::vector<float> result(samples_id->size());

        if((samples_id->size()==0) || (samples_id->size()==1)){
          // nothing to train on: every sample of the cell gets NaN (result already has their size,
          // so the NaN must be assigned, not appended — appending left 0.0 in place)
          result.assign(samples_id->size(), NAN);
          return(result);
        }

        auto sl_folds = slice(folds, samples_id);

        for(int i=0; i<k; i++){
          auto test_train = indexes_by_predicate<int>(&sl_folds, [i](int *j){return(*j==i);});
          if(test_train.first.size()!=0){ // if is 0, then there's nothing to estimate
            if(test_train.second.size()==0){
              for(int j: test_train.first){
                result.at(j) = NAN;
              }
            }else{
              // the samples of fold i are predicted from the samples of the other folds in the cell
              std::vector<int> train;
              for(int l: test_train.second){
                train.push_back(samples_id->at(l));
              }
              for(int j: test_train.first){
                result.at(j) = idw_predict(&(coords->at(samples_id->at(j))), coords, values, train, exponent);
              }
            }
          }
        }
        return(result);
      }

      float get_exponent(){
        return(this->exponent);
      }
  };
}

#endif

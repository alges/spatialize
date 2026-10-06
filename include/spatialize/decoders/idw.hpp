#ifndef _SPTLZ_DECODERS_IDW_
#define _SPTLZ_DECODERS_IDW_

#include <stdexcept>
#include <cmath>
#include "spatialize/decoder.hpp"
#include "spatialize/utils.hpp"

namespace sptlz{
  // IDW prediction at `point` from the samples `ids`: weights 1/d^p, except that data at distance 0
  // take all the weight (a query on a datum gets its value; duplicated locations, their mean).
  inline float idw_predict(std::vector<float> *point, std::vector<std::vector<float>> *coords,
                           std::vector<float> *values, const std::vector<int> &ids, float exponent){
    float w, w_sum = 0.0, w_v_sum = 0.0;
    int exact_location = 0;
    std::vector<float> distances;

    for(size_t j=0; j<ids.size(); j++){
      float dist = distance(point, &(coords->at(ids.at(j))));
      distances.push_back(dist);
      if(dist==0){
        exact_location = 1;
      }
    }

    for(size_t j=0; j<ids.size(); j++){
      float dist = distances.at(j);
      if(exact_location==1){
        w = (dist==0) ? 1 : 0;
      }else{
        w = 1/std::pow(dist, exponent);
      }
      // keep sum of weighted values and sum of weights
      w_sum += w;
      w_v_sum += w*values->at(ids.at(j));
    }
    // weighted values sum normalized (divided by weights sum)
    return(w_v_sum/w_sum);
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

      std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params){
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

      std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params){
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

      std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *folds, std::vector<int> *samples_id, std::vector<float> *params){
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

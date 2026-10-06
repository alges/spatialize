#ifndef _SPTLZ_DECODERS_IDW_
#define _SPTLZ_DECODERS_IDW_

#include <stdexcept>
#include <cmath>
#include "spatialize/decoder.hpp"
#include "spatialize/utils.hpp"

namespace sptlz{
  // Inverse distance weighting inside a cell: weights 1/d^p; a query on a datum takes its value.
  class IDWDecoder: public Decoder {
    protected:
      float exponent;

    public:
      IDWDecoder(float _exponent){
        this->exponent = _exponent;
      }

      std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params){
        std::vector<float> result;
        float w, w_sum, w_v_sum;
        int exact_location;

        if(samples_id->size()==0){
          for(auto l: *locations_id){
            std::ignore = l;
            result.push_back(NAN);
          }
          return(result);
        }

        // for every location
        for(size_t i=0; i<locations_id->size(); i++){
          w_sum = 0.0;
          w_v_sum = 0.0;
          exact_location = 0;
          std::vector<float> distances;

          for(size_t j=0; j<samples_id->size(); j++){
            // calculate weight
            float dist = distance(&(locations->at(locations_id->at(i))), &(coords->at(samples_id->at(j))));
            distances.push_back(dist);
            if(dist==0){
              exact_location = 1;
            }
          }

          if(exact_location==1){
            for(size_t j=0; j<samples_id->size(); j++){
              // calculate weight
              float dist = distances.at(j);
              if(dist==0)
                w = 1;
              else
                w = 0;
              // keep sum of weighted values and sum of weights
              w_sum += w;
              w_v_sum += w*values->at(samples_id->at(j));
            }
          }
          else {
            for(size_t j=0; j<samples_id->size(); j++){
              // calculate weight
              float dist = distances.at(j);
              w = 1/std::pow(dist, exponent);
              // keep sum of weighted values and sum of weights
              w_sum += w;
              w_v_sum += w*values->at(samples_id->at(j));
            }
          }


          // return weighted values sum normalized (divided by weights sum)
          result.push_back(w_v_sum/w_sum);
        }
        return(result);
      }

      std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params){
        std::vector<float> result;

        if((samples_id->size()==0) || (samples_id->size()==1)){
          for(auto l: *samples_id){
            std::ignore = l;
            result.push_back(NAN);
          }
          return(result);
        }

        float w, w_sum, w_v_sum;

        // for every location
        for(size_t i=0; i<samples_id->size(); i++){
          w_sum = 0.0;
          w_v_sum = 0.0;

          for(size_t j=0; j<samples_id->size(); j++){
            if(i!=j){
              // calculate weight
              w = 1/(1+std::pow(distance(&(coords->at(samples_id->at(i))), &(coords->at(samples_id->at(j)))), exponent));
              // keep sum of weighted values and sum of weights
              w_sum += w;
              w_v_sum += w*values->at(samples_id->at(j));}
          }
          // return weighted values sum normalized (divided by weights sum)
          result.push_back(w_v_sum/w_sum);
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

        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        auto sl_folds = slice(folds, samples_id);
        float w, w_sum, w_v_sum;

        for(int i=0; i<k; i++){
          auto test_train = indexes_by_predicate<int>(&sl_folds, [i](int *j){return(*j==i);});
          if(test_train.first.size()!=0){ // if is 0, then there's nothing to estimate
            if(test_train.second.size()==0){
              for(int j: test_train.first){
                result.at(j) = NAN;
              }
            }else{
              for(int j: test_train.first){
                w_sum = 0.0;
                w_v_sum = 0.0;
                for(int l: test_train.second){
                  w = 1/(1+std::pow(distance(&(sl_coords.at(j)), &(sl_coords.at(l))), exponent));
                  w_sum += w;
                  w_v_sum += w*values->at(samples_id->at(l));
                }
                result.at(j) = w_v_sum/w_sum;
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

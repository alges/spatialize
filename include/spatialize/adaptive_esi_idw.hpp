#ifndef _SPTLZ_ADTV_ESI_IDW_
#define _SPTLZ_ADTV_ESI_IDW_

#include <stdexcept>
#include <cmath>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "spatialize/abstract_esi.hpp"
#include "spatialize/utils.hpp"
#include "spatialize/grad_descent.hpp"

namespace sptlz{
  // Constants for numerical stability and algorithm parameters
  constexpr float EPSILON = 1e-10f;  // Small value to prevent division by zero
  constexpr float DEFAULT_EXPONENT = 2.0f;
  constexpr float DEFAULT_ANISOTROPY = 1.0f;

  class LOO2D{
    protected:
      std::vector<std::vector<float>> *coords;
      std::vector<float> *values;
      std::vector<float> centroid;
      bool use_mse;  // true for MSE, false for MAE

    public:
      LOO2D(std::vector<std::vector<float>> *_coords, std::vector<float> *_values, std::string metric = "mae"){
        values = _values;
        centroid = sptlz::get_centroid(_coords);
        coords = _coords;
        use_mse = (metric == "mse");
      }

      float eval(std::vector<float> X){
        int n = static_cast<int>(values->size());
        float r = 0.0;
        std::vector<float> params = {X.at(1), X.at(2)};

        // Transform coordinates once
        auto tr_coords = sptlz::transform(coords, &params, &centroid);

        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:r) schedule(static)
        #endif
        for(int i=0; i<n; i++){
          float sum_w = 0.0;
          float est = 0.0;
          float wj, dist;
          for(int j=0;j<n;j++){
            if(j!=i){
              // Use transformed coordinates for distance calculation
              dist = sptlz::distance(&(tr_coords[i]), &(tr_coords[j]));
              wj = 1.0f/(EPSILON + std::pow(dist, X.at(0)));
              sum_w += wj;
              est += wj*values->at(j);
            }
          }
          // Use MAE (Mean Absolute Error) or MSE (Mean Squared Error)
          if(sum_w > EPSILON){
            float error = values->at(i) - est/sum_w;
            r += use_mse ? (error * error) : std::abs(error);
          }else{
            float error = values->at(i);
            r += use_mse ? (error * error) : std::abs(error);
          }
        }
        return(r/n);
      }
  };

  class LOO3D{
    protected:
      std::vector<std::vector<float>> *coords;
      std::vector<float> *values;
      std::vector<float> centroid;
      bool use_mse;  // true for MSE, false for MAE

    public:
      LOO3D(std::vector<std::vector<float>> *_coords, std::vector<float> *_values, std::string metric = "mae"){
        values = _values;
        centroid = sptlz::get_centroid(_coords);
        coords = _coords;
        use_mse = (metric == "mse");
      }

      float eval(std::vector<float> X){
        int n = static_cast<int>(values->size());
        float r = 0.0;
        std::vector<float> params = {X.at(1), X.at(2), X.at(3), X.at(4), X.at(5)};

        // Transform coordinates once
        auto tr_coords = sptlz::transform(coords, &params, &centroid);

        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:r) schedule(static)
        #endif
        for(int i=0; i<n; i++){
          float sum_w = 0.0;
          float est = 0.0;
          float wj, dist;
          for(int j=0;j<n;j++){
            if(j!=i){
              // Calculate distance directly
              dist = sptlz::distance(&(tr_coords[i]), &(tr_coords[j]));
              wj = 1.0f/(EPSILON + std::pow(dist, X.at(0)));
              sum_w += wj;
              est += wj*values->at(j);
            }
          }
          // Use MAE (Mean Absolute Error) or MSE (Mean Squared Error)
          if(sum_w > EPSILON){
            float error = values->at(i) - est/sum_w;
            r += use_mse ? (error * error) : std::abs(error);
          }else{
            float error = values->at(i);
            r += use_mse ? (error * error) : std::abs(error);
          }
        }
        return(r/n);
      }
  };

  class ADAPTIVE_ESI_IDW: public ESI {
    protected:
      int d, k;
      std::string metric;  // "mae" or "mse"
      std::vector<std::vector<float>> param_ranges;
      std::vector<float> steps;
      std::vector<int> ns;

      std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params){
        std::vector<float> result;

        if(locations_id->size()==0){
          return(result);
        }

        if(samples_id->size()==0){
          for([[maybe_unused]] auto l: *locations_id){
            result.push_back(NAN);
          }
          return(result);
        }

        if(samples_id->size()==1){
          // Return the single sample's value for all locations
          float single_value = values->at(samples_id->at(0));
          for([[maybe_unused]] auto l: *locations_id){
            result.push_back(single_value);
          }
          return(result);
        }

        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        auto sl_locations = slice(locations, locations_id);
        std::vector<float> centroid;
        int i_params = 0;

        centroid.push_back(params->at(i_params++));
        centroid.push_back(params->at(i_params++));

        if(coords->at(0).size()==3){
          centroid.push_back(params->at(i_params++));
        }

        float exponent = params->at(i_params++);
        std::vector<float> rot_params = slice_from(params, i_params);

        auto tr_coords = transform(&sl_coords, &rot_params, &centroid);
        auto tr_locations = transform(&sl_locations, &rot_params, &centroid);

        result.resize(locations_id->size());

        // Parallelize over locations (each location is independent)
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for(int i=0; i<locations_id->size(); i++){
          float w_sum = 0.0;
          float w_v_sum = 0.0;

          for(int j=0; j<samples_id->size(); j++){
            // calculate weight
            float w = 1.0f/(EPSILON + std::pow(distance(&(tr_locations.at(i)), &(tr_coords.at(j))), exponent));
            // keep sum of weighted values and sum of weights
            w_sum += w;
            w_v_sum += w*sl_values.at(j);
          }
          // return weighted values sum normalized (divided by weights sum)
          if(w_sum > EPSILON){
            result[i] = w_v_sum/w_sum;
          }else{
            result[i] = NAN;
          }
        }

        return(result);
      }

      std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params){
        std::vector<float> result;

        if((samples_id->size()==0) || (samples_id->size()==1)){
          for([[maybe_unused]] auto l: *samples_id){
            result.push_back(NAN);
          }
          return(result);
        }

        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        std::vector<float> centroid;
        int i_params = 0;
        centroid.push_back(params->at(i_params++));
        centroid.push_back(params->at(i_params++));
        if(coords->at(0).size()==3){
          centroid.push_back(params->at(i_params++));
        }

        float exponent = params->at(i_params++);
        std::vector<float> rot_params = slice_from(params, i_params);

        auto tr_coords = transform(&sl_coords, &rot_params, &centroid);

        result.resize(samples_id->size());

        // Parallelize LOO evaluation (each sample is independent)
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for(int i=0; i<samples_id->size(); i++){
          float w_sum = 0.0;
          float w_v_sum = 0.0;

          for(int j=0; j<samples_id->size(); j++){
            if(i!=j){
              // calculate weight
              float w = 1.0f/(EPSILON + std::pow(distance(&(tr_coords.at(i)), &(tr_coords.at(j))), exponent));
              // keep sum of weighted values and sum of weights
              w_sum += w;
              w_v_sum += w*sl_values.at(j);
            }
          }
          // return weighted values sum normalized (divided by weights sum)
          if(w_sum > EPSILON){
            result[i] = w_v_sum/w_sum;
          }else{
            result[i] = NAN;
          }
        }
        return(result);
      }

      std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *folds, std::vector<int> *samples_id, std::vector<float> *params){
        std::vector<float> result(samples_id->size());
        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        auto sl_folds = slice(folds, samples_id);

        if((samples_id->size()==0) || (samples_id->size()==1)){
          for([[maybe_unused]] auto l: *samples_id){
            result.push_back(NAN);
          }
          return(result);
        }

        std::vector<float> centroid;
        int i_params = 0;
        centroid.push_back(params->at(i_params++));
        centroid.push_back(params->at(i_params++));
        if(coords->at(0).size()==3){
          centroid.push_back(params->at(i_params++));
        }

        float exponent = params->at(i_params++);
        std::vector<float> rot_params = slice_from(params, i_params);
        auto tr_coords = transform(&sl_coords, &rot_params, &centroid);

        for(int i=0; i<k; i++){
          auto test_train = indexes_by_predicate<int>(&sl_folds, [i](int *j){return(*j==i);});
          if(test_train.first.size()!=0){ // if is 0, then there's nothing to estimate
            if(test_train.second.size()==0){
              for(int j: test_train.first){
                result.at(j) = NAN;
              }
            }else{
              // Parallelize over test samples within each fold
              std::vector<int> test_indices(test_train.first.begin(), test_train.first.end());
              #ifdef _OPENMP
              #pragma omp parallel for schedule(static)
              #endif
              for(int idx=0; idx<test_indices.size(); idx++){
                int j = test_indices[idx];
                float w_sum = 0.0;
                float w_v_sum = 0.0;
                for(int l: test_train.second){
                  float w = 1.0f/(EPSILON + std::pow(distance(&(tr_coords.at(j)), &(tr_coords.at(l))), exponent));
                  w_sum += w;
                  w_v_sum += w*values->at(samples_id->at(l));
                }
                if(w_sum > EPSILON){
                  result.at(j) = w_v_sum/w_sum;
                }else{
                  result.at(j) = NAN;
                }
              }
            }
          }
        }
        return(result);
      }

      void post_process(){
        sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
        sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

        logger->info("computing optimal parameters");

        progress->init(static_cast<int>(mondrian_forest.size()), 1);

        bool interrupted = false;

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) shared(interrupted)
        #endif
        for(int i=0; i<mondrian_forest.size(); i++){
          #ifdef _OPENMP
          if(interrupted) continue;  // Skip remaining work if interrupted
          #endif

          auto mt = mondrian_forest.at(i);
          for(int j=0; j<mt->samples_by_leaf.size(); j++){
            std::vector<std::vector<float>> leaf_coords;
            std::vector<float> leaf_values;
            for(int k=0; k<mt->samples_by_leaf.at(j).size(); k++){
              leaf_coords.push_back(coords.at(mt->samples_by_leaf.at(j).at(k)));
              leaf_values.push_back(values.at(mt->samples_by_leaf.at(j).at(k)));
            }

            mt->leaf_params.at(j) = get_params2(&leaf_coords, &leaf_values);
          }

          #ifdef _OPENMP
          #pragma omp critical
          #endif
          {
            if (PyErr_CheckSignals() != 0) {  // to allow ctrl-c from user
              interrupted = true;
            }
            progress->inform(i + 1);
          }
        }

        if(interrupted){
          delete logger;
          delete progress;
          throw std::runtime_error("Computation interrupted by user");
        }

        progress->stop();

        delete logger;
        delete progress;
      }

      std::vector<float> get_params(std::vector<std::vector<float>> *coords, std::vector<float> *values){
        if(coords->size()==0){
          return(std::vector<float>());
        }else if(coords->size()==1){
          // Return default parameters with centroid
          auto centroid = sptlz::get_centroid(coords);
          if(coords->at(0).size()==2){
            // 2D: centroid_x, centroid_y, exponent, azimuth, anisotropy_ratio
            return(std::vector<float>({centroid[0], centroid[1], DEFAULT_EXPONENT, 0.0f, DEFAULT_ANISOTROPY}));
          }else{
            // 3D: centroid_x, centroid_y, centroid_z, exponent, azim, dip, plunge, ratio1, ratio2
            return(std::vector<float>({centroid[0], centroid[1], centroid[2], DEFAULT_EXPONENT, 0.0f, 0.0f, 0.0f, DEFAULT_ANISOTROPY, DEFAULT_ANISOTROPY}));
          }
        }

        LOOND *fn;
        if(this->d==2){
          fn = new LOO_2D(coords, values, 0.01f);
        }else{
          fn = new LOO_3D(coords, values, 0.01f);
        }
        GradDesc *opt = new GridNBRndDesc(fn, this->param_ranges, this->steps, this->ns, this->k, std::rand());
        std::vector<float> m = get_minimum(opt, &(this->param_ranges), 100);

        delete opt;
        delete fn;

        // Safety check: if optimization failed, return default parameters
        if(m.size() == 0){
          auto centroid = sptlz::get_centroid(coords);
          if(coords->at(0).size()==2){
            return(std::vector<float>({centroid[0], centroid[1], DEFAULT_EXPONENT, 0.0f, DEFAULT_ANISOTROPY}));
          }else{
            return(std::vector<float>({centroid[0], centroid[1], centroid[2], DEFAULT_EXPONENT, 0.0f, 0.0f, 0.0f, DEFAULT_ANISOTROPY, DEFAULT_ANISOTROPY}));
          }
        }

        // Add centroid to the beginning of optimized parameters
        auto centroid = sptlz::get_centroid(coords);
        for(auto v: m){
          centroid.push_back(v);
        }
        return(centroid);
      }

      std::vector<float> get_params2(std::vector<std::vector<float>> *coords, std::vector<float> *values){
        std::uniform_real_distribution<float> uni_float(0, 1);
        int best_of = 3;
        if(coords->size()==0){
          return(std::vector<float>());
        }else if(coords->size()==1){
          // Return default parameters with centroid
          auto centroid = sptlz::get_centroid(coords);
          if(coords->at(0).size()==2){
            return(std::vector<float>({centroid[0], centroid[1], DEFAULT_EXPONENT, 0.0f, DEFAULT_ANISOTROPY}));
          }else{
            return(std::vector<float>({centroid[0], centroid[1], centroid[2], DEFAULT_EXPONENT, 0.0f, 0.0f, 0.0f, DEFAULT_ANISOTROPY, DEFAULT_ANISOTROPY}));
          }
        }

        std::vector<float> min_coords;
        if(coords->at(0).size()==2){
          std::vector<float> starting_point, candidate;
          float min_value=1e20f, aux;
          LOO2D *func = new LOO2D(coords, values, this->metric);
          std::vector<std::vector<float>> ranges = {
            {0.1f, 8.0f, 0.2f},    // exponent p
            {0.0f, 180.0f, 1.0f},  // azimuth φ
            {0.1f, 5.0f, 0.2f}    // anisotropy factor a_f
          };
          for(int i=0; i<best_of; i++){
            starting_point = {};
            for(int j=0; j<ranges.size(); j++){
              starting_point.push_back(ranges.at(j).at(0)+uni_float(this->my_rand)*(ranges.at(j).at(1)-ranges.at(j).at(0)));
            }
            candidate = sptlz::grid_search<LOO2D>(func, &ranges, starting_point);
            aux = func->eval(candidate);
            if(aux<min_value){
              min_coords = candidate;
              min_value = aux;
            }
          }
          delete func;
        }else if(coords->at(0).size()==3){
          std::vector<float> starting_point, candidate;
          float min_value=1e20f, aux;
          LOO3D *func = new LOO3D(coords, values, this->metric);
          std::vector<std::vector<float>> ranges = {
            {0.1f, 8.0f, 0.2f},    // exponent p
            {0.0f, 180.0f, 1.0f},  // azimuth
            {0.0f, 180.0f, 1.0f},  // dip
            {0.0f, 180.0f, 1.0f},  // plunge
            {0.1f, 5.0f, 0.2f},   // anisotropy ratio 1
            {0.1f, 5.0f, 0.2f}    // anisotropy ratio 2
          };
          for(int i=0; i<best_of; i++){
            starting_point = {};
            for(int j=0; j<ranges.size(); j++){
              starting_point.push_back(ranges.at(j).at(0)+uni_float(this->my_rand)*(ranges.at(j).at(1)-ranges.at(j).at(0)));
            }
            candidate = sptlz::grid_search<LOO3D>(func, &ranges, starting_point);
            aux = func->eval(candidate);
            if(aux<min_value){
              min_coords = candidate;
              min_value = aux;
            }
          }
          delete func;
        }

        if(min_coords.size()==0){ // Fallback if optimization failed
          if (coords->at(0).size()==2){
            min_coords = {DEFAULT_EXPONENT, 0.0f, DEFAULT_ANISOTROPY};
          }else if(coords->at(0).size()==3){
            min_coords = {DEFAULT_EXPONENT, 0.0f, 0.0f, 0.0f, DEFAULT_ANISOTROPY, DEFAULT_ANISOTROPY};
          }
        }
        auto centroid = sptlz::get_centroid(coords);
        for(auto v: min_coords){
          centroid.push_back(v);
        }

        return(centroid);
      }

    public:
      ADAPTIVE_ESI_IDW(std::vector<std::vector<float>> _coords,
                       std::vector<float> _values,
                       float lambda,
                       int forest_size,
                       std::vector<std::vector<float>> bbox,
                       std::function<int(std::string)> visitor,
                       int seed=206936,
                       std::string _metric="mae"):
      ESI(_coords, _values, lambda, forest_size, bbox, visitor, seed){
        this->class_name = __func__;
        this->metric = _metric;
        if(_coords.at(0).size()==2){
          this->d = 2;
          this->param_ranges = {
            {  0.5f, 10.0f},  // exponent p
            {-90.0f, 90.0f},  // azimuth φ
            {  0.1f, 5.0f}   // anisotropy factor a_f
          };
          this->steps = {0.5f, 10.0f, 0.2f};
          this->ns = {19, 18, 25};
        }else if(_coords.at(0).size()==3){
          this->d = 3;
          this->param_ranges = {
            {  0.5f, 10.0f},  // exponent p
            {-90.0f, 90.0f},  // azimuth
            {-90.0f, 90.0f},  // dip
            {-90.0f, 90.0f},  // plunge
            {  0.1f, 5.0f},  // anisotropy ratio 1
            {  0.1f, 5.0f}   // anisotropy ratio 2
          };
          this->steps = {0.5f, 10.0f, 10.0f, 10.0f, 0.2f, 0.2f};
          this->ns = {19, 18, 18, 18, 25, 25};
        }else{
          throw std::runtime_error("ADAPTIVE_ESI_IDW available just for 2D and 3D");
        }
        this->k = (int)std::ceil(0.1*std::pow(3, this->ns.size()));
        post_process();
      }

      ADAPTIVE_ESI_IDW(std::vector<sptlz::MondrianTree*> _mondrian_forest,
                       std::vector<std::vector<float>> _coords,
                       std::vector<float> _values,
                       std::function<int(std::string)> visitor):
      ESI(_mondrian_forest, _coords, _values, visitor){
        this->class_name = __func__;
      }
  };
}

#endif

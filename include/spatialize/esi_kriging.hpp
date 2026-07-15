#ifndef _SPTLZ_ESI_KRIGING_
#define _SPTLZ_ESI_KRIGING_

#include <cmath>
#include <functional>
#include <utility>
#include <Eigen/Dense>
#include "spatialize/abstract_esi.hpp"
#include "spatialize/utils.hpp"

namespace sptlz{
  std::pair<std::vector<float>, std::vector<float>> split_cov_matrix(std::vector<float> *mat, int n, int idx){
    std::vector<float> left, right;

    for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){
        if(i==idx){
          if(j!=idx){
            right.push_back(mat->at(i*n+j));
          }
        }else{
          if(j!=idx){
            left.push_back(mat->at(i*n+j));
          }
        }
      }
    }
    return(std::make_pair(left, right));
  }

  std::vector<float> distances(std::vector<std::vector<float>> *coords){
    int n = static_cast<int>(coords->size());
    std::vector<float> result;

    for(int i=0; i<n; i++){
      for(int j=0; j<i; j++){
        result.push_back(result.at(j*n+i));
      }
      for(int j=i; j<n; j++){
        result.push_back(distance(&(coords->at(i)), &(coords->at(j))));
      }
    }
    return(result);
  }

  std::vector<float> kriging_right_matrix(std::vector<std::vector<float>> *coords, std::vector<std::vector<float>> *locations, std::function<float(float)> gamma){
    int n = static_cast<int>(coords->size()), m = static_cast<int>(locations->size());
    std::vector<float> result;

    for(int i=0; i<m; i++){
      for(int j=0; j<n; j++){
        result.push_back(gamma(distance(&(locations->at(i)), &(coords->at(j)))));
      }
      result.push_back(1.0);
    }
    return(result);
  }

  std::vector<float> kriging_left_matrix(std::vector<std::vector<float>> *coords, std::function<float(float)> gamma){
    int n = static_cast<int>(coords->size());
    std::vector<float> result;

    for(int i=0; i<n; i++){
      for(int j=0; j<i; j++){
        result.push_back(result.at(j*(n+1)+i));
      }
      for(int j=i; j<n; j++){
        result.push_back(gamma(distance(&(coords->at(i)), &(coords->at(j)))));
      }
      result.push_back(1.0);
    }
    for(int i=0; i<n; i++){
      result.push_back(1.0);
    }
    result.push_back(0.0);
    return(result);
  }

  class ESI_Kriging: public ESI {
    protected:
      int variogram_model; // 1:Spherical 2:Exponetial 3:Cubic 4:Gaussian
      float nugget, range, sill;

      std::function<float(float)> variogram(int m, float n, float r, float s){
        float c = static_cast<float>(1.0-nugget);

        if(m==1){ // Spherical
          return([n,c,r,s](float d){return(s*std::min(1.0, std::max(0.0, 1.0 - n - c*(1.5*d/r - 0.5*pow(d/r, 3.0)))));});
        }else if(m==2){ // Exponential
          return([n,c,r,s](float d){return(s*std::min(1.0, std::max(0.0, 1.0 - n - c*(1.0-exp(-3.0*d/r)))));});
        }else if(m==3){ // Cubic
          return([n,c,r,s](float d){return(s*std::min(1.0, std::max(0.0, 1.0 - n - c*(7.0*pow(d/r, 2.0) - 35.0*pow(d/r, 3.0)/4.0 + 3.5*pow(d/r, 5.0) - 0.75*pow(d/r, 7.0)))));});
        }else if(m==4){ // Gaussian
          return([n,c,r,s](float d){return(s*std::min(1.0, std::max(0.0, 1.0 - n - c*(1-exp(-3.0*pow(d/r, 2))))));});
        }else{
          return([](float d){return(d);});
        }
      }

      std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params){
        std::vector<float> result;
        int n = static_cast<int>(samples_id->size());
        int m = static_cast<int>(locations_id->size());

        if(n==0){
          for(auto l: *locations_id){
            std::ignore = l;
            result.push_back(NAN);
          }
          return(result);
        }

        auto sl_values = slice(values, samples_id);
        if(n==1){
          for(auto l: *locations_id){
            std::ignore = l;
            result.push_back(sl_values.at(0));
          }
          return(result);
        }

        auto gamma = variogram(this->variogram_model, this->nugget, this->range, this->sill);
        auto sl_coords = slice(coords, samples_id);
        auto sl_locations = slice(locations, locations_id);
        auto left_cov = kriging_left_matrix(&sl_coords, gamma);
        auto right_cov = kriging_right_matrix(&sl_coords, &sl_locations, gamma);
        sl_values.push_back(0.0); // to anulate the mu coeffcicient
        Eigen::Map<Eigen::MatrixXf> v = Eigen::Map<Eigen::MatrixXf>(sl_values.data(), 1, n+1);
        Eigen::Map<Eigen::MatrixXf> b = Eigen::Map<Eigen::MatrixXf>(right_cov.data(), n+1, m);
        Eigen::Map<Eigen::MatrixXf> A = Eigen::Map<Eigen::MatrixXf>(left_cov.data(), n+1, n+1);
        auto A_1 = A.completeOrthogonalDecomposition().pseudoInverse();
        auto weights = A_1*b;
        auto vals = v*weights;

        result.resize(m);
        Eigen::Map<Eigen::MatrixXf>(&result[0], 1, m) = vals;

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
        int n = static_cast<int>(samples_id->size());
        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        sl_values.push_back(0.0); // to anulate the mu coeffcicient

        auto left_cov = kriging_left_matrix(&sl_coords, variogram(variogram_model, nugget, range, sill));
        for(int i=0; i<n; i++){
          auto aux = split_cov_matrix(&left_cov, n+1, i);
          auto right_cov = aux.second;
          auto new_values = slice_drop_idx(&sl_values, i);

          Eigen::Map<Eigen::MatrixXf> A = Eigen::Map<Eigen::MatrixXf>(aux.first.data(), n, n);
          auto inv = A.completeOrthogonalDecomposition().pseudoInverse();
          Eigen::Map<Eigen::MatrixXf> v = Eigen::Map<Eigen::MatrixXf>(sl_values.data(), 1, n);
          Eigen::Map<Eigen::MatrixXf> b = Eigen::Map<Eigen::MatrixXf>(right_cov.data(), n, 1);
          auto weights = inv*b;
          auto est = v*weights;

          result.push_back(est(0));
        }

        //for(auto l: *samples_id){result.push_back(NAN);}
        return(result);
      }

      std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *folds, std::vector<int> *samples_id, std::vector<float> *params){
        std::vector<float> result(samples_id->size());

        if((samples_id->size()==0) || (samples_id->size()==1)){
          for(auto l: *samples_id){
            std::ignore = l;
            result.push_back(NAN);
          }
          return(result);
        }

        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        auto sl_folds = slice(folds, samples_id);

        for(int i=0; i<k; i++){
          auto test_train = indexes_by_predicate<int>(&sl_folds, [i](int *j){return(*j==i);});
          if(test_train.first.size()!=0){ // if is 0, then there's nothing to estimate
            if(test_train.second.size()==0){
              for(int j: test_train.first){
                result.at(j) = NAN;
              }
            }else{
              int m = static_cast<int>(test_train.first.size()), n = static_cast<int>(test_train.second.size());
              auto sl_coords_train = slice(&sl_coords, &(test_train.second));
              auto sl_coords_test = slice(&sl_coords, &(test_train.first));
              auto sl_values_train = slice(&sl_values, &(test_train.second));
              sl_values_train.push_back(0.0);

              auto left_cov = kriging_left_matrix(&sl_coords_train, variogram(variogram_model, nugget, range, sill));
              auto right_cov = kriging_right_matrix(&sl_coords_train, &sl_coords_test, variogram(variogram_model, nugget, range, sill));
              Eigen::Map<Eigen::MatrixXf> A = Eigen::Map<Eigen::MatrixXf>(left_cov.data(), n+1, n+1);
              auto inv = A.completeOrthogonalDecomposition().pseudoInverse();
              Eigen::Map<Eigen::MatrixXf> v = Eigen::Map<Eigen::MatrixXf>(sl_values_train.data(), 1, n+1);
              Eigen::Map<Eigen::MatrixXf> b = Eigen::Map<Eigen::MatrixXf>(right_cov.data(), n+1, m);
              auto weights = inv*b;
              auto est = v*weights;
              for(size_t j=0; j<test_train.first.size(); j++){
                result.at(test_train.first.at(j)) = est(j);
              }
            }
          }
        }
        return(result);
      }

    public:
      ESI_Kriging(std::vector<std::vector<float>> _coords,
                  std::vector<float> _values,
                  float lambda,
                  int forest_size,
                  std::vector<std::vector<float>> bbox,
                  int _model,
                  float _nugget,
                  float _range,
                  float _sill,
                  std::function<int(std::string)> visitor,
                  int seed=206936):
      ESI(_coords, _values, lambda, forest_size, bbox, visitor, seed){
        class_name = __func__;
        variogram_model = _model;
        nugget = _nugget;
        range = _range;
        sill = _sill;
      }

      ESI_Kriging(std::vector<sptlz::MondrianTree*> _mondrian_forest,
                  std::vector<std::vector<float>> _coords,
                  std::vector<float> _values,
                  int _model,
                  float _nugget,
                  float _range,
                  float _sill,
                  std::function<int(std::string)> visitor):
      ESI(_mondrian_forest, _coords, _values, visitor){
        class_name = __func__;
        variogram_model = _model;
        nugget = _nugget;
        range = _range;
        sill = _sill;
      }


      ~ESI_Kriging() {}

      int get_variogram_model(){
        return(this->variogram_model);
      }

      float get_nugget(){
        return(this->nugget);
      }

      float get_range(){
        return(this->range);
      }

      float get_sill(){
        return(this->sill);
      }


  };
}

#endif

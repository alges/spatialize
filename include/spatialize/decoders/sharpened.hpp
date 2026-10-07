#ifndef _SPTLZ_DECODERS_SHARPENED_
#define _SPTLZ_DECODERS_SHARPENED_

#include <algorithm>
#include <cmath>
#include <vector>
#ifndef EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_PARALLELIZE
#endif
#include <Eigen/Dense>
#include "spatialize/decoder.hpp"
#include "spatialize/utils.hpp"
#include "spatialize/decoders/adaptive_idw.hpp"
#include "spatialize/decoders/draw.hpp"

namespace sptlz{
  // The sharpened adaptive decoder of the theory. In each cell, after the adaptive fit
  // (exponent p, rotation, anisotropy) and in its transformed coordinates:
  //   r_j = leave-one-out residual of the adaptive weights, s = median |r_j|,
  //   rho = |slope of the least-squares plane| * mean distance to the centre / median |z_j - mean z|,
  //   p_C = p (1 + kappa_g min(rho, rho_max)),
  //   weight of datum j at v = (1 + kappa_r |r_j| / s) / (EPSILON + d(v_j, v)^p_C).
  // The weights are positive, so the prediction is a convex combination of the cell's values.
  // With kappa_r = kappa_g = 0 it is the adaptive decoder, bit for bit.

  inline float median_of(std::vector<float> x){
    if(x.empty()) return(0.0f);
    size_t n = x.size(), h = n/2;
    std::nth_element(x.begin(), x.begin()+h, x.end());
    float hi = x[h];
    if(n % 2 == 1) return(hi);
    float lo = *std::max_element(x.begin(), x.begin()+h);
    return(0.5f*(lo+hi));
  }

  struct SharpFactors {
    std::vector<float> boost;   // 1 + kappa_r |r_j| / s, per datum
    float exponent;             // p_C
  };

  // the per-datum boosts and the raised exponent of a set of data, in transformed coordinates
  inline SharpFactors sharp_factors(std::vector<std::vector<float>> &tr, std::vector<float> &z, float p,
                                    float kappa_r, float kappa_g, float rho_max){
    size_t n = z.size();
    SharpFactors f{std::vector<float>(n, 1.0f), p};
    if(n < 2) return(f);
    int d = static_cast<int>(tr.at(0).size());

    // leave-one-out residuals of the adaptive weights
    if(kappa_r != 0.0f){
      std::vector<float> r(n), abs_r(n);
      for(size_t j=0; j<n; j++){
        float w_sum = 0.0f, w_v_sum = 0.0f;
        for(size_t l=0; l<n; l++){
          if(l == j) continue;
          float w = 1.0f/(EPSILON + std::pow(distance(&(tr[l]), &(tr[j])), p));
          w_sum += w;
          w_v_sum += w*z[l];
        }
        r[j] = (w_sum > EPSILON) ? z[j] - w_v_sum/w_sum : 0.0f;
        abs_r[j] = std::fabs(r[j]);
      }
      float s = median_of(abs_r);
      if(s > 0){
        for(size_t j=0; j<n; j++) f.boost[j] = 1.0f + kappa_r*abs_r[j]/s;
      }
    }

    // the cell's gradient, made dimensionless by its dispersion and extent
    if(kappa_g != 0.0f && static_cast<int>(n) > d){
      Eigen::VectorXd centre = Eigen::VectorXd::Zero(d);
      double z_mean = 0.0;
      for(size_t j=0; j<n; j++){
        for(int c=0; c<d; c++) centre(c) += tr[j][c];
        z_mean += z[j];
      }
      centre /= static_cast<double>(n);
      z_mean /= static_cast<double>(n);
      Eigen::MatrixXd X(n, d);
      Eigen::VectorXd y(n);
      double extent = 0.0;
      std::vector<float> dev(n);
      for(size_t j=0; j<n; j++){
        for(int c=0; c<d; c++) X(j, c) = tr[j][c] - centre(c);
        y(j) = z[j] - z_mean;
        extent += X.row(j).norm();
        dev[j] = static_cast<float>(std::fabs(z[j] - z_mean));
      }
      extent /= static_cast<double>(n);
      float mad = median_of(dev);
      Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X);
      if(mad > 0 && qr.rank() == d){
        Eigen::VectorXd beta = qr.solve(y);
        double rho = beta.norm()*extent/mad;
        if(std::isfinite(rho)){
          f.exponent = p*(1.0f + kappa_g*static_cast<float>(std::min(rho, static_cast<double>(rho_max))));
        }
      }
    }
    return(f);
  }

  class SharpenedIDWDecoder: public AdaptiveIDWDecoder {
    protected:
      float kappa_r, kappa_g, rho_max;

    public:
      SharpenedIDWDecoder(int dim, std::string _metric, float _kappa_r, float _kappa_g, float _rho_max):
        AdaptiveIDWDecoder(dim, _metric), kappa_r(_kappa_r), kappa_g(_kappa_g), rho_max(_rho_max){}

      // the transformed coordinates of `points` under the cell's fitted parameters, and its exponent
      std::vector<std::vector<float>> to_cell(std::vector<std::vector<float>> *points, std::vector<float> *params, float &exponent){
        int dim = static_cast<int>(points->at(0).size());
        std::vector<float> centroid(params->begin(), params->begin()+dim);
        exponent = params->at(dim);
        std::vector<float> rot = slice_from(params, dim+1);
        return(transform(points, &rot, &centroid));
      }

      SharpFactors factors(std::vector<std::vector<float>> &tr, std::vector<float> &z, float p){
        return(sharp_factors(tr, z, p, kappa_r, kappa_g, rho_max));
      }

      // as AdaptiveIDWDecoder::leaf_estimation, with the boosted weights and the raised exponent
      std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params, const CellContext &cell){
        std::vector<float> result;
        if(locations_id->size()==0) return(result);
        if(samples_id->size()==0){
          result.assign(locations_id->size(), NAN);
          return(result);
        }
        if(samples_id->size()==1){
          result.assign(locations_id->size(), values->at(samples_id->at(0)));
          return(result);
        }
        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        auto sl_locations = slice(locations, locations_id);
        float p;
        auto tr_coords = to_cell(&sl_coords, params, p);
        auto tr_locations = to_cell(&sl_locations, params, p);
        auto f = factors(tr_coords, sl_values, p);

        result.resize(locations_id->size());
        for(size_t i=0; i<locations_id->size(); i++){
          float w_sum = 0.0, w_v_sum = 0.0, exact_sum = 0.0f;
          int n_exact = 0;
          for(size_t j=0; j<samples_id->size(); j++){
            float dist = distance(&(tr_locations.at(i)), &(tr_coords.at(j)));
            if(dist == 0.0f){
              exact_sum += sl_values.at(j);
              n_exact++;
              continue;
            }
            float w = f.boost[j]*(1.0f/(EPSILON + std::pow(dist, f.exponent)));
            w_sum += w;
            w_v_sum += w*sl_values.at(j);
          }
          if(n_exact > 0) result[i] = exact_sum/n_exact;
          else if(w_sum > EPSILON) result[i] = w_v_sum/w_sum;
          else result[i] = NAN;
        }
        return(result);
      }

      // a sample predicted from the others: the cell's adaptive fit, with the boosts and the raised
      // exponent of the other data
      float predict_from(std::vector<std::vector<float>> &tr, std::vector<float> &z, std::vector<int> &others, int target, float p){
        std::vector<std::vector<float>> tr_o;
        std::vector<float> z_o;
        for(int l: others){ tr_o.push_back(tr[l]); z_o.push_back(z[l]); }
        auto f = factors(tr_o, z_o, p);
        float w_sum = 0.0, w_v_sum = 0.0;
        for(size_t j=0; j<others.size(); j++){
          float w = f.boost[j]*(1.0f/(EPSILON + std::pow(distance(&(tr[target]), &(tr_o[j])), f.exponent)));
          w_sum += w;
          w_v_sum += w*z_o[j];
        }
        return((w_sum > EPSILON) ? w_v_sum/w_sum : NAN);
      }

      std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params, const CellContext &cell){
        std::vector<float> result(samples_id->size(), NAN);
        if(samples_id->size() <= 1) return(result);
        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        float p;
        auto tr = to_cell(&sl_coords, params, p);
        for(size_t i=0; i<samples_id->size(); i++){
          std::vector<int> others;
          for(size_t j=0; j<samples_id->size(); j++) if(i != j) others.push_back(static_cast<int>(j));
          result[i] = predict_from(tr, sl_values, others, static_cast<int>(i), p);
        }
        return(result);
      }

      std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *folds, std::vector<int> *samples_id, std::vector<float> *params, const CellContext &cell){
        std::vector<float> result(samples_id->size(), NAN);
        if(samples_id->size() <= 1) return(result);
        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        auto sl_folds = slice(folds, samples_id);
        float p;
        auto tr = to_cell(&sl_coords, params, p);
        for(int fo=0; fo<k; fo++){
          auto test_train = indexes_by_predicate<int>(&sl_folds, [fo](int *j){return(*j==fo);});
          if(test_train.first.empty() || test_train.second.empty()) continue;
          for(int j: test_train.first){
            result.at(j) = predict_from(tr, sl_values, test_train.second, j, p);
          }
        }
        return(result);
      }
  };

  // ----------------------------------------------------------------------------------------------
  // weights for the draws, from the adaptive fit (kappa_r = kappa_g = 0) or the sharpened one

  class SharpenedWeights: public CellWeights {
    protected:
      SharpenedIDWDecoder decoder;   // holds the adaptive fit and the sharpening constants
    public:
      SharpenedWeights(int dim, std::string metric, float kappa_r, float kappa_g, float rho_max):
        decoder(dim, metric, kappa_r, kappa_g, rho_max){}

      void fit(std::vector<Partition*> *forest, std::vector<std::vector<float>> *coords, std::vector<float> *values,
               std::mt19937 &rng, std::function<int(std::string)> visitor, std::string class_name){
        decoder.fit(forest, coords, values, rng, visitor, class_name);
      }

      std::vector<std::vector<float>> at(std::vector<std::vector<float>> *cell_coords, std::vector<float> *cell_values,
                                         std::vector<std::vector<float>> *points, std::vector<float> *params){
        size_t n = cell_coords->size();
        if(n <= 1 || params->empty()){
          return(std::vector<std::vector<float>>(points->size(), std::vector<float>(n, 1.0f)));
        }
        float p;
        auto tr = decoder.to_cell(cell_coords, params, p);
        auto tr_points = decoder.to_cell(points, params, p);
        auto f = decoder.factors(tr, *cell_values, p);
        std::vector<std::vector<float>> result;
        for(auto &v: tr_points){
          std::vector<float> d(n), w(n);
          bool exact = false;
          for(size_t j=0; j<n; j++){
            d[j] = distance(&v, &(tr[j]));
            if(d[j] == 0.0f) exact = true;
          }
          for(size_t j=0; j<n; j++){
            w[j] = exact ? (d[j] == 0.0f ? 1.0f : 0.0f) : f.boost[j]*(1.0f/(EPSILON + std::pow(d[j], f.exponent)));
          }
          result.push_back(w);
        }
        return(result);
      }
  };
}

#endif

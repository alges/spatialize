#ifndef _SPTLZ_DECODERS_ADAPTIVE_IDW_
#define _SPTLZ_DECODERS_ADAPTIVE_IDW_

#include <stdexcept>
#include <atomic>
#include <thread>
#include <cmath>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "spatialize/decoder.hpp"
#include "spatialize/utils.hpp"
#include "spatialize/grad_descent.hpp"

namespace sptlz{
  // Constants for numerical stability and algorithm parameters
  constexpr float EPSILON = 1e-10f;  // Small value to prevent division by zero
  constexpr float DEFAULT_EXPONENT = 2.0f;
  constexpr float DEFAULT_ANISOTROPY = 1.0f;

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

  // Leave-one-out error of IDW on a line, as a function of the exponent alone.
  class LOO1D{
    protected:
      std::vector<std::vector<float>> *coords;
      std::vector<float> *values;
      bool use_mse;  // true for MSE, false for MAE

    public:
      LOO1D(std::vector<std::vector<float>> *_coords, std::vector<float> *_values, std::string metric = "mae"){
        values = _values;
        coords = _coords;
        use_mse = (metric == "mse");
      }

      float eval(std::vector<float> X){
        int n = static_cast<int>(values->size());
        float r = 0.0;

        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:r) schedule(static)
        #endif
        for(int i=0; i<n; i++){
          float pred = relative_idw([&](auto f){
            for(int j=0; j<n; j++){
              if(j!=i) f(std::abs(coords->at(i).at(0) - coords->at(j).at(0)), values->at(j), 1.0f);
            }
          }, X.at(0));
          float error = std::isnan(pred) ? values->at(i) : values->at(i) - pred;
          r += use_mse ? (error * error) : std::abs(error);
        }
        return(r/n);
      }
  };

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
          float pred = relative_idw([&](auto f){
            for(int j=0; j<n; j++){
              if(j!=i) f(sptlz::distance(&(tr_coords[i]), &(tr_coords[j])), values->at(j), 1.0f);
            }
          }, X.at(0));
          // Use MAE (Mean Absolute Error) or MSE (Mean Squared Error)
          float error = std::isnan(pred) ? values->at(i) : values->at(i) - pred;
          r += use_mse ? (error * error) : std::abs(error);
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
          float pred = relative_idw([&](auto f){
            for(int j=0; j<n; j++){
              if(j!=i) f(sptlz::distance(&(tr_coords[i]), &(tr_coords[j])), values->at(j), 1.0f);
            }
          }, X.at(0));
          // Use MAE (Mean Absolute Error) or MSE (Mean Squared Error)
          float error = std::isnan(pred) ? values->at(i) : values->at(i) - pred;
          r += use_mse ? (error * error) : std::abs(error);
        }
        return(r/n);
      }
  };

  // Adaptive IDW: per-cell exponent and anisotropy fitted by leave-one-out (1D: the exponent alone).
  class AdaptiveIDWDecoder: public Decoder {
    protected:
      int d, k;
      std::string metric;  // "mae" or "mse"
      std::vector<std::vector<float>> param_ranges;
      std::vector<float> steps;
      std::vector<int> ns;


    public:
      AdaptiveIDWDecoder(int dim, std::string _metric="mae"){
        this->metric = _metric;
        if(dim==1){
          this->d = 1;
          this->param_ranges = {{0.5f, 10.0f}};  // exponent p
          this->steps = {0.5f};
          this->ns = {19};
        }else if(dim==2){
          this->d = 2;
          this->param_ranges = {
            {  0.5f, 10.0f},  // exponent p
            {-90.0f, 90.0f},  // azimuth φ
            {  0.1f, 5.0f}   // anisotropy factor a_f
          };
          this->steps = {0.5f, 10.0f, 0.2f};
          this->ns = {19, 18, 25};
        }else if(dim==3){
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
          throw std::runtime_error("adaptive IDW is available in 1D, 2D and 3D");
        }
        this->k = (int)std::ceil(0.1*std::pow(3, this->ns.size()));
      }

      std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params, const CellContext &cell){
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

        for(size_t c=0; c<coords->at(0).size(); c++){
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
          // a query on a datum takes its value (the mean, for duplicated locations)
          result[i] = relative_idw([&](auto f){
            for(size_t j=0; j<samples_id->size(); j++){
              f(distance(&(tr_locations.at(i)), &(tr_coords.at(j))), sl_values.at(j), 1.0f);
            }
          }, exponent);
        }

        return(result);
      }

      std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params, const CellContext &cell){
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
        for(size_t c=0; c<coords->at(0).size(); c++){
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
          result[i] = relative_idw([&](auto f){
            for(int j=0; j<static_cast<int>(samples_id->size()); j++){
              if(i!=j) f(distance(&(tr_coords.at(i)), &(tr_coords.at(j))), sl_values.at(j), 1.0f);
            }
          }, exponent);
        }
        return(result);
      }

      std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *folds, std::vector<int> *samples_id, std::vector<float> *params, const CellContext &cell){
        std::vector<float> result(samples_id->size());
        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        auto sl_folds = slice(folds, samples_id);

        if((samples_id->size()==0) || (samples_id->size()==1)){
          // nothing to train on: every sample of the cell gets NaN (result already has their size,
          // so the NaN must be assigned, not appended — appending left 0.0 in place)
          result.assign(samples_id->size(), NAN);
          return(result);
        }

        std::vector<float> centroid;
        int i_params = 0;
        for(size_t c=0; c<coords->at(0).size(); c++){
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
                result.at(j) = relative_idw([&](auto f){
                  for(int l: test_train.second){
                    f(distance(&(tr_coords.at(j)), &(tr_coords.at(l))), values->at(samples_id->at(l)), 1.0f);
                  }
                }, exponent);
              }
            }
          }
        }
        return(result);
      }

      std::vector<float> fit_cell(std::vector<std::vector<float>> *coords, std::vector<float> *values,
                                  const std::vector<int> &samples, unsigned int seed){
        std::vector<std::vector<float>> c;
        std::vector<float> v;
        for(int d: samples){ c.push_back(coords->at(d)); v.push_back(values->at(d)); }
        std::mt19937 rng(seed);
        return(get_params2(&c, &v, rng));
      }

      // Fits the per-cell parameters (exponent, anisotropy) of every cell of every tree.
      void fit(std::vector<Partition*> *forest,
               std::vector<std::vector<float>> *coords,
               std::vector<float> *values,
               std::mt19937 &rng,
               std::function<int(std::string)> visitor,
               std::string class_name){
        auto &mondrian_forest = *forest;
        sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(visitor, class_name);
        sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(visitor);

        logger->info("computing optimal parameters");

        progress->init(static_cast<int>(mondrian_forest.size()), 1);

        // Python (signals, progress) may only be touched by the thread that holds the GIL, i.e.
        // the calling thread. It is identified by its OS thread id, not by omp_get_thread_num():
        // the loop body opens nested OpenMP regions (LOO2D::eval), so OpenMP thread numbers do not
        // identify it reliably. Worker threads only update these atomics.
        const std::thread::id caller = std::this_thread::get_id();
        std::atomic<bool> interrupted(false);
        std::atomic<int> n_done(0);

        // Pre-draw one seed per tree sequentially from the shared engine
        // *before* entering the parallel region. Concurrent threads must
        // never read/write the same std::mt19937 instance (that would be a
        // data race, silently corrupting the sequence and making results
        // depend on thread scheduling instead of `seed`), so each iteration
        // below gets its own independent, deterministically-seeded engine.
        std::uniform_int_distribution<unsigned int> uni_int;
        std::vector<unsigned int> tree_seeds(mondrian_forest.size());
        for(size_t i=0; i<mondrian_forest.size(); i++){
          tree_seeds[i] = uni_int(rng);
        }

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) shared(interrupted, n_done)
        #endif
        for(int i=0; i<mondrian_forest.size(); i++){
          if(interrupted.load()) continue;  // Skip remaining work if interrupted

          std::mt19937 leaf_rand(tree_seeds[i]);

          auto mt = mondrian_forest.at(i);
          for(int j=0; j<mt->samples_by_leaf.size(); j++){
            std::vector<std::vector<float>> leaf_coords;
            std::vector<float> leaf_values;
            for(int k=0; k<mt->samples_by_leaf.at(j).size(); k++){
              leaf_coords.push_back(coords->at(mt->samples_by_leaf.at(j).at(k)));
              leaf_values.push_back(values->at(mt->samples_by_leaf.at(j).at(k)));
            }

            mt->leaf_params.at(j) = get_params2(&leaf_coords, &leaf_values, leaf_rand);
          }

          int done = ++n_done;
          if(std::this_thread::get_id() == caller){
            if (PyErr_CheckSignals() != 0) {  // to allow ctrl-c from user
              interrupted = true;
            }
            progress->inform(done);
          }
        }

        if(interrupted.load()){
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

      std::vector<float> get_params2(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::mt19937 &rng){
        std::uniform_real_distribution<float> uni_float(0, 1);
        int best_of = 3;
        if(coords->size()==0){
          return(std::vector<float>());
        }else if(coords->size()==1){
          // Return default parameters with centroid
          auto centroid = sptlz::get_centroid(coords);
          if(coords->at(0).size()==1){
            return(std::vector<float>({centroid[0], DEFAULT_EXPONENT}));
          }else if(coords->at(0).size()==2){
            return(std::vector<float>({centroid[0], centroid[1], DEFAULT_EXPONENT, 0.0f, DEFAULT_ANISOTROPY}));
          }else{
            return(std::vector<float>({centroid[0], centroid[1], centroid[2], DEFAULT_EXPONENT, 0.0f, 0.0f, 0.0f, DEFAULT_ANISOTROPY, DEFAULT_ANISOTROPY}));
          }
        }

        std::vector<float> min_coords;
        if(coords->at(0).size()==1){
          std::vector<float> starting_point, candidate;
          float min_value=1e20f, aux;
          LOO1D *func = new LOO1D(coords, values, this->metric);
          std::vector<std::vector<float>> ranges = {
            {0.1f, 8.0f, 0.2f}     // exponent p
          };
          for(int i=0; i<best_of; i++){
            starting_point = {ranges.at(0).at(0)+uni_float(rng)*(ranges.at(0).at(1)-ranges.at(0).at(0))};
            candidate = sptlz::grid_search<LOO1D>(func, &ranges, starting_point);
            aux = func->eval(candidate);
            if(aux<min_value){
              min_coords = candidate;
              min_value = aux;
            }
          }
          delete func;
        }else if(coords->at(0).size()==2){
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
              starting_point.push_back(ranges.at(j).at(0)+uni_float(rng)*(ranges.at(j).at(1)-ranges.at(j).at(0)));
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
              starting_point.push_back(ranges.at(j).at(0)+uni_float(rng)*(ranges.at(j).at(1)-ranges.at(j).at(0)));
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
          if (coords->at(0).size()==1){
            min_coords = {DEFAULT_EXPONENT};
          }else if (coords->at(0).size()==2){
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
  };

}

#endif

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
      std::vector<float> *values;
      bool use_mse;  // true for MSE, false for MAE
      int n;
      // Per-instance working storage. grid_search calls eval() thousands of
      // times against one instance, and the coordinate differences from the
      // centroid do NOT depend on the parameters eval() varies -- so they are
      // computed once in the constructor instead of rebuilt (with a fresh
      // vector-of-vectors allocation) by sptlz::transform on every call.
      std::vector<float> raw_dx, raw_dy;   // coords - centroid (invariant)
      std::vector<float> tr_x, tr_y;       // rotated coords (per eval)
      // Pairwise pow(dist, exponent), n*n floats. Filled exploiting symmetry
      // (dist(i,j) == dist(j,i) bit-for-bit, since only the sign of each diff
      // flips and it is squared), halving the std::pow calls per eval.
      std::vector<float> dist_pow;

    public:
      LOO2D(std::vector<std::vector<float>> *_coords, std::vector<float> *_values, std::string metric = "mae"){
        values = _values;
        use_mse = (metric == "mse");
        n = static_cast<int>(_values->size());
        auto centroid = sptlz::get_centroid(_coords);
        raw_dx.resize(n);
        raw_dy.resize(n);
        for(int i = 0; i < n; i++){
          raw_dx[i] = (*_coords)[i][0] - centroid[0];
          raw_dy[i] = (*_coords)[i][1] - centroid[1];
        }
        tr_x.resize(n);
        tr_y.resize(n);
        dist_pow.resize((size_t)n * (size_t)n);
      }

      float eval(const std::vector<float>& X){
        float r = 0.0;

        // 2D rotation inlined, coefficients computed in the IDENTICAL
        // expression order as utils.hpp transform(): r1 = a_f*cos_phi,
        // r2 = -a_f*sin_phi, r3 = sin_phi, r4 = cos_phi, applied to the same
        // (coord - centroid) differences.
        const float phi = sptlz::deg_to_rad(X[1]);
        const float cos_phi = std::cos(phi);
        const float sin_phi = std::sin(phi);
        const float a_f = X[2];
        const float m00 = a_f * cos_phi, m01 = -a_f * sin_phi;
        const float m10 = sin_phi, m11 = cos_phi;

        for(int i = 0; i < n; i++){
          tr_x[i] = m00 * raw_dx[i] + m01 * raw_dy[i];
          tr_y[i] = m10 * raw_dx[i] + m11 * raw_dy[i];
        }

        // Same sum-of-squares + sqrt + std::pow chain the old
        // distance()/std::pow pair produced, just computed once per unordered
        // pair instead of twice.
        const float exponent = X[0];
        for(int i=0; i<n; i++){
          for(int j=i+1; j<n; j++){
            float dx = tr_x[i] - tr_x[j];
            float dy = tr_y[i] - tr_y[j];
            float dist = std::sqrt(dx*dx + dy*dy);
            float dp = std::pow(dist, exponent);
            dist_pow[(size_t)i*n + j] = dp;
            dist_pow[(size_t)j*n + i] = dp;
          }
        }

        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:r) schedule(static)
        #endif
        for(int i=0; i<n; i++){
          float sum_w = 0.0;
          float est = 0.0;
          for(int j=0;j<n;j++){
            if(j!=i){
              float wj = 1.0f/(EPSILON + dist_pow[(size_t)i*n + j]);
              sum_w += wj;
              est += wj*(*values)[j];
            }
          }
          // Use MAE (Mean Absolute Error) or MSE (Mean Squared Error)
          if(sum_w > EPSILON){
            float error = (*values)[i] - est/sum_w;
            r += use_mse ? (error * error) : std::abs(error);
          }else{
            float error = (*values)[i];
            r += use_mse ? (error * error) : std::abs(error);
          }
        }
        return(r/n);
      }
  };

  class LOO3D{
    protected:
      std::vector<float> *values;
      bool use_mse;  // true for MSE, false for MAE
      int n;
      // Same rationale as LOO2D: centroid differences are invariant across the
      // eval() calls grid_search makes, so they are computed once here rather
      // than rebuilt by sptlz::transform per call.
      std::vector<float> raw_dx, raw_dy, raw_dz;
      std::vector<float> tr_x, tr_y, tr_z;
      std::vector<float> dist_pow;

    public:
      LOO3D(std::vector<std::vector<float>> *_coords, std::vector<float> *_values, std::string metric = "mae"){
        values = _values;
        use_mse = (metric == "mse");
        n = static_cast<int>(_values->size());
        auto centroid = sptlz::get_centroid(_coords);
        raw_dx.resize(n);
        raw_dy.resize(n);
        raw_dz.resize(n);
        for(int i = 0; i < n; i++){
          raw_dx[i] = (*_coords)[i][0] - centroid[0];
          raw_dy[i] = (*_coords)[i][1] - centroid[1];
          raw_dz[i] = (*_coords)[i][2] - centroid[2];
        }
        tr_x.resize(n);
        tr_y.resize(n);
        tr_z.resize(n);
        dist_pow.resize((size_t)n * (size_t)n);
      }

      float eval(const std::vector<float>& X){
        float r = 0.0;

        // 3D rotation inlined in the IDENTICAL expression order as
        // utils.hpp transform(): the nine coefficients are built first, then
        // rows 2 and 3 are scaled by the anisotropy ratios (params[3],
        // params[4]) -- exactly as transform() does with r4..r9 *= ...
        const float azim = sptlz::deg_to_rad(X[1]);
        const float dip = sptlz::deg_to_rad(X[2]);
        const float plunge = sptlz::deg_to_rad(X[3]);
        float ca = std::cos(azim), sa = std::sin(azim);
        float cb = std::cos(dip), sb = std::sin(dip);
        float cc = std::cos(plunge), sc_ = std::sin(plunge);
        float m00 = ca*cb, m01 = ca*sb*sc_-sa*cc, m02 = ca*sb*cc+sa*sc_;
        float m10 = sa*cb, m11 = sa*sb*sc_+ca*cc, m12 = sa*sb*cc-ca*sc_;
        float m20 = -sb, m21 = cb*sc_, m22 = cb*cc;
        m10 *= X[4]; m11 *= X[4]; m12 *= X[4];
        m20 *= X[5]; m21 *= X[5]; m22 *= X[5];

        for(int i = 0; i < n; i++){
          tr_x[i] = m00*raw_dx[i] + m01*raw_dy[i] + m02*raw_dz[i];
          tr_y[i] = m10*raw_dx[i] + m11*raw_dy[i] + m12*raw_dz[i];
          tr_z[i] = m20*raw_dx[i] + m21*raw_dy[i] + m22*raw_dz[i];
        }

        const float exponent = X[0];
        for(int i=0; i<n; i++){
          for(int j=i+1; j<n; j++){
            float dx = tr_x[i] - tr_x[j];
            float dy = tr_y[i] - tr_y[j];
            float dz = tr_z[i] - tr_z[j];
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            float dp = std::pow(dist, exponent);
            dist_pow[(size_t)i*n + j] = dp;
            dist_pow[(size_t)j*n + i] = dp;
          }
        }

        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:r) schedule(static)
        #endif
        for(int i=0; i<n; i++){
          float sum_w = 0.0;
          float est = 0.0;
          for(int j=0;j<n;j++){
            if(j!=i){
              float wj = 1.0f/(EPSILON + dist_pow[(size_t)i*n + j]);
              sum_w += wj;
              est += wj*(*values)[j];
            }
          }
          // Use MAE (Mean Absolute Error) or MSE (Mean Squared Error)
          if(sum_w > EPSILON){
            float error = (*values)[i] - est/sum_w;
            r += use_mse ? (error * error) : std::abs(error);
          }else{
            float error = (*values)[i];
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

        std::vector<float> centroid;
        int i_params = 0;

        centroid.push_back(params->at(i_params++));
        centroid.push_back(params->at(i_params++));

        const bool is_3d = (coords->at(0).size()==3);
        if(is_3d){
          centroid.push_back(params->at(i_params++));
        }

        float exponent = params->at(i_params++);
        std::vector<float> rot_params = slice_from(params, i_params);

        // Per-leaf constants hoisted out of the loops: the centroid components
        // and the rotation coefficients, built in the IDENTICAL expression
        // order as utils.hpp transform() (2D: r1..r4; 3D: the nine
        // coefficients with rows 2 and 3 scaled by the anisotropy ratios).
        // This replaces the three per-leaf slice() calls and the two
        // transform() calls -- and their intermediate vector-of-vectors
        // allocations -- with flat arrays, without changing any arithmetic.
        const float cx = centroid.at(0);
        const float cy = centroid.at(1);
        const float cz = is_3d ? centroid.at(2) : 0.0f;
        float r1 = 0.0f, r2 = 0.0f, r3 = 0.0f, r4 = 0.0f;
        float m00 = 0.0f, m01 = 0.0f, m02 = 0.0f;
        float m10 = 0.0f, m11 = 0.0f, m12 = 0.0f;
        float m20 = 0.0f, m21 = 0.0f, m22 = 0.0f;
        if(is_3d){
          const float azim = sptlz::deg_to_rad(rot_params.at(0));
          const float dip = sptlz::deg_to_rad(rot_params.at(1));
          const float plunge = sptlz::deg_to_rad(rot_params.at(2));
          const float ca = std::cos(azim); const float sa = std::sin(azim);
          const float cb = std::cos(dip); const float sb = std::sin(dip);
          const float cc = std::cos(plunge); const float sc = std::sin(plunge);
          const float an1 = rot_params.at(3);
          const float an2 = rot_params.at(4);
          m00 = ca*cb; m01 = ca*sb*sc-sa*cc; m02 = ca*sb*cc+sa*sc;
          m10 = sa*cb; m11 = sa*sb*sc+ca*cc; m12 = sa*sb*cc-ca*sc;
          m20 = -sb;   m21 = cb*sc;          m22 = cb*cc;
          m10 *= an1; m11 *= an1; m12 *= an1;
          m20 *= an2; m21 *= an2; m22 *= an2;
        }else{
          const float phi = sptlz::deg_to_rad(rot_params.at(0));
          const float cos_phi = std::cos(phi);
          const float sin_phi = std::sin(phi);
          const float a_f = rot_params.at(1);
          r1 = a_f*cos_phi; r2 = -a_f*sin_phi; r3 = sin_phi; r4 = cos_phi;
        }

        // Transformed sample coordinates and values, extracted once into flat
        // contiguous arrays. The inner (location x sample) loop then reads
        // adjacent floats instead of chasing vector-of-vectors pointers for
        // every pair.
        const int ns = (int)samples_id->size();
        std::vector<float> tcx(ns), tcy(ns), sv(ns);
        std::vector<float> tcz(is_3d ? ns : 0);
        for(int j=0; j<ns; j++){
          const std::vector<float>& c = coords->at(samples_id->at(j));
          if(is_3d){
            const float x0 = c[0] - cx, x1 = c[1] - cy, x2 = c[2] - cz;
            tcx[j] = m00*x0 + m01*x1 + m02*x2;
            tcy[j] = m10*x0 + m11*x1 + m12*x2;
            tcz[j] = m20*x0 + m21*x1 + m22*x2;
          }else{
            tcx[j] = r1*(c[0] - cx) + r2*(c[1] - cy);
            tcy[j] = r3*(c[0] - cx) + r4*(c[1] - cy);
          }
          sv[j] = values->at(samples_id->at(j));
        }

        result.resize(locations_id->size());

        // Parallelize over locations (each location is independent)
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for(int i=0; i<(int)locations_id->size(); i++){
          float w_sum = 0.0;
          float w_v_sum = 0.0;

          // location coordinates and their transform, hoisted out of the
          // sample loop (they were recomputed per sample before)
          const std::vector<float>& loc = locations->at(locations_id->at(i));
          const float loc_x = loc[0];
          const float loc_y = loc[1];
          const float loc_z = is_3d ? loc[2] : 0.0f;
          float lx, ly, lz = 0.0f;
          if(is_3d){
            const float x0 = loc_x - cx, x1 = loc_y - cy, x2 = loc_z - cz;
            lx = m00*x0 + m01*x1 + m02*x2;
            ly = m10*x0 + m11*x1 + m12*x2;
            lz = m20*x0 + m21*x1 + m22*x2;
          }else{
            lx = r1*(loc_x - cx) + r2*(loc_y - cy);
            ly = r3*(loc_x - cx) + r4*(loc_y - cy);
          }

          for(int j=0; j<ns; j++){
            // distance() inlined on the flat arrays: same squared diffs in the
            // same order, same sqrt, same std::pow
            float dist;
            if(is_3d){
              float dx = lx - tcx[j], dy = ly - tcy[j], dz = lz - tcz[j];
              dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            }else{
              float dx = lx - tcx[j], dy = ly - tcy[j];
              dist = std::sqrt(dx*dx + dy*dy);
            }
            // calculate weight
            float w = 1.0f/(EPSILON + std::pow(dist, exponent));
            // keep sum of weighted values and sum of weights
            w_sum += w;
            w_v_sum += w*sv[j];
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
        const int n_trees = static_cast<int>(mondrian_forest.size());

        // Pre-draw one seed per tree sequentially from the shared engine
        // *before* entering the parallel region. Concurrent threads must
        // never read/write the same std::mt19937 instance (that would be a
        // data race, silently corrupting the sequence and making results
        // depend on thread scheduling instead of `seed`), so each iteration
        // below gets its own independent, deterministically-seeded engine.
        std::uniform_int_distribution<unsigned int> uni_int;
        std::vector<unsigned int> tree_seeds(n_trees);
        for(int i=0; i<n_trees; i++){
          tree_seeds[i] = uni_int(this->my_rand);
        }

        // Pre-draw every leaf's grid_search starting values in a sequential
        // pass, so the parallel loop below can process leaves in ANY order
        // without touching an RNG.
        //
        // The draws come from a PER-TREE engine seeded with tree_seeds[i] and
        // are consumed in ascending leaf order -- exactly the engine and
        // exactly the order the previous per-tree `leaf_rand` used. get_params2
        // consumes best_of(3) * n_params values, and only for leaves holding at
        // least 2 samples (the 0- and 1-sample branches return before drawing),
        // so the sequence produced here is identical value-for-value to the
        // sequence the serial-per-tree version consumed.
        const int n_params = (coords.at(0).size() == 2) ? 3 : 6;
        const int draws_per_leaf = 3 * n_params;
        std::vector<std::vector<std::vector<float>>> pre_randoms(n_trees);
        {
          std::uniform_real_distribution<float> pre_uni(0, 1);
          for(int i = 0; i < n_trees; i++){
            auto mt = mondrian_forest.at(i);
            std::mt19937 leaf_rand(tree_seeds[i]);
            pre_randoms[i].resize(mt->samples_by_leaf.size());
            for(size_t j = 0; j < mt->samples_by_leaf.size(); j++){
              if(mt->samples_by_leaf.at(j).size() >= 2){
                pre_randoms[i][j].reserve(draws_per_leaf);
                for(int r = 0; r < draws_per_leaf; r++){
                  pre_randoms[i][j].push_back(pre_uni(leaf_rand));
                }
              }
            }
          }
        }

        // Flatten every (tree, leaf) pair into one work list and schedule the
        // largest leaves first. Leaf cost grows with leaf size, so without this
        // a single huge leaf can be picked up last and leave every other thread
        // idle waiting on it. Reordering is safe precisely because the random
        // starting values are pre-drawn and addressed by (tree, leaf).
        struct LeafWork { int tree_idx, leaf_idx, leaf_size; };
        std::vector<LeafWork> work_items;
        for(int i = 0; i < n_trees; i++){
          auto mt = mondrian_forest.at(i);
          for(int j = 0; j < (int)mt->samples_by_leaf.size(); j++){
            work_items.push_back({i, j, (int)mt->samples_by_leaf.at(j).size()});
          }
        }
        std::sort(work_items.begin(), work_items.end(),
                  [](const LeafWork& a, const LeafWork& b){ return a.leaf_size > b.leaf_size; });

        // One progress tick per tree is still emitted (the Python handler
        // counts inform() calls and expects exactly `init` of them), fired when
        // a tree's last leaf lands rather than when the tree's serial loop ends.
        std::vector<int> tree_remaining(n_trees);
        for(int i = 0; i < n_trees; i++){
          tree_remaining[i] = (int)mondrian_forest.at(i)->samples_by_leaf.size();
        }
        int trees_done = 0;
        int pending_informs = 0;   // trees finished but not yet reported to Python

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) shared(interrupted)
        #endif
        for(int w = 0; w < (int)work_items.size(); w++){
          #ifdef _OPENMP
          if(interrupted) continue;  // Skip remaining work if interrupted
          #endif

          const int i = work_items[w].tree_idx;
          const int j = work_items[w].leaf_idx;
          auto mt = mondrian_forest.at(i);

          const std::vector<int>& sbl = mt->samples_by_leaf.at(j);
          std::vector<std::vector<float>> leaf_coords;
          std::vector<float> leaf_values;
          leaf_coords.reserve(sbl.size());
          leaf_values.reserve(sbl.size());
          for(size_t k=0; k<sbl.size(); k++){
            leaf_coords.push_back(coords.at(sbl.at(k)));
            leaf_values.push_back(values.at(sbl.at(k)));
          }

          mt->leaf_params.at(j) = get_params2(&leaf_coords, &leaf_values, &pre_randoms[i][j]);

          // NO Python C API here. The binding never releases the GIL, so the
          // calling (OMP master) thread holds it for the whole call; a worker
          // thread doing gil_scoped_acquire would block forever on a GIL that
          // is only released when this function returns -- an unconditional
          // deadlock as soon as more than one thread runs. Workers therefore
          // only tally completed trees; the master, which already owns the
          // GIL, is the sole thread that talks to Python.
          bool report_here = false;
          int to_report = 0;
          #ifdef _OPENMP
          #pragma omp critical
          #endif
          {
            if(--tree_remaining[i] == 0){
              pending_informs++;
            }
            #ifdef _OPENMP
            const bool is_master = (omp_get_thread_num() == 0);
            #else
            const bool is_master = true;
            #endif
            if(is_master && pending_informs > 0){
              to_report = pending_informs;
              pending_informs = 0;
              report_here = true;
            }
          }
          if(report_here){
            if (PyErr_CheckSignals() != 0) {  // to allow ctrl-c from user
              interrupted = true;
            }
            for(int t=0; t<to_report; t++){
              progress->inform(++trees_done);
            }
          }
        }

        if(interrupted){
          delete logger;
          delete progress;
          throw std::runtime_error("Computation interrupted by user");
        }

        // Flush the ticks for trees that finished on worker threads after the
        // master's last visit, so the callback still sees exactly one inform
        // per tree (the Python handler counts calls, not values).
        for(int t=0; t<pending_informs; t++){
          progress->inform(++trees_done);
        }
        pending_informs = 0;

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

      // `_pre_randoms` holds the best_of * n_params uniform(0,1) draws for this
      // leaf, pre-drawn by post_process from the leaf's own tree engine in leaf
      // order (see post_process) -- identical values to consuming the per-tree
      // engine here, but leaving this function free of shared RNG state so the
      // caller can process leaves in any order.
      std::vector<float> get_params2(std::vector<std::vector<float>> *coords, std::vector<float> *values, const std::vector<float> *_pre_randoms){
        int best_of = 3;
        int _ri = 0;
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
              starting_point.push_back(ranges.at(j).at(0)+(*_pre_randoms)[_ri++]*(ranges.at(j).at(1)-ranges.at(j).at(0)));
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
              starting_point.push_back(ranges.at(j).at(0)+(*_pre_randoms)[_ri++]*(ranges.at(j).at(1)-ranges.at(j).at(0)));
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

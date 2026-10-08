#ifndef _SPTLZ_DECODERS_DRAW_
#define _SPTLZ_DECODERS_DRAW_

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "spatialize/decoder.hpp"
#include "spatialize/utils.hpp"
#include "spatialize/decoders/kriging.hpp"

namespace sptlz{
  // Decoders that return a draw (the theory's distance-weighted draw and its relatives): at each
  // query a datum of the cell, chosen with probability proportional to its weight. The mean of the
  // draws is the weighted-mean decoder with the same weights, their variance the weighted
  // dispersion of the cell, and every member is an observed value.

  // ----------------------------------------------------------------------------------------------
  // counter-based random numbers: a uniform number from (seed, tree, item, role) alone, so a draw
  // depends neither on the number of threads nor on the order or set of the other queries

  inline uint64_t splitmix64(uint64_t x){
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return(x ^ (x >> 31));
  }

  enum class DrawRole : uint64_t { ESTIMATE = 1, LOO = 2, KFOLD = 3 };

  // uniform in [0, 1) for one draw; `item` is the query's (or held-out sample's) global index
  inline double draw_uniform(const CellContext &cell, int item, DrawRole role){
    uint64_t h = splitmix64(0x5D7A11ULL ^ static_cast<uint64_t>(cell.seed));
    h = splitmix64(h ^ static_cast<uint64_t>(static_cast<uint32_t>(cell.tree)));
    h = splitmix64(h ^ ((static_cast<uint64_t>(static_cast<uint32_t>(item)) << 2) | static_cast<uint64_t>(role)));
    return(static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0));
  }

  // an index drawn with probability proportional to the (non-negative) weights
  inline int draw_index(const std::vector<float> &w, double u){
    double total = 0.0;
    for(float x: w) total += x;
    double target = u * total, acc = 0.0;
    int last = -1;
    for(size_t j=0; j<w.size(); j++){
      if(w[j] <= 0) continue;
      acc += w[j];
      last = static_cast<int>(j);
      if(target < acc) return(last);
    }
    return(last);  // rounding at the top end
  }

  // ----------------------------------------------------------------------------------------------
  // weights of a cell's data at a batch of points: one row per point, one column per datum

  // `cell_values` and `params` (the cell's fitted parameters, see Decoder::fit) serve the weights
  // that depend on them (adaptive and sharpened IDW); the others ignore them.
  class CellWeights {
    public:
      virtual ~CellWeights(){}
      virtual std::vector<std::vector<float>> at(std::vector<std::vector<float>> *cell_coords, std::vector<float> *cell_values,
                                                 std::vector<std::vector<float>> *points, std::vector<float> *params) = 0;
      // per-cell fitting, run once after the forest is drawn (as Decoder::fit)
      virtual void fit(std::vector<Partition*> *forest, std::vector<std::vector<float>> *coords, std::vector<float> *values,
                       std::mt19937 &rng, std::function<int(std::string)> visitor, std::string class_name){}
      // the parameters of one region (see Decoder::fit_cell)
      virtual std::vector<float> fit_cell(std::vector<std::vector<float>> *coords, std::vector<float> *values,
                                          const std::vector<int> &samples, unsigned int seed){ return(std::vector<float>()); }
  };

  // every datum equally likely (the block-mark decoder)
  class UniformWeights: public CellWeights {
    public:
      std::vector<std::vector<float>> at(std::vector<std::vector<float>> *cell_coords, std::vector<float> *cell_values, std::vector<std::vector<float>> *points, std::vector<float> *params){
        return(std::vector<std::vector<float>>(points->size(), std::vector<float>(cell_coords->size(), 1.0f)));
      }
  };

  // 1/d^p, as IDWDecoder; data at distance 0 take all the weight
  class IDWWeights: public CellWeights {
    protected:
      float exponent;
    public:
      IDWWeights(float _exponent): exponent(_exponent){}
      std::vector<std::vector<float>> at(std::vector<std::vector<float>> *cell_coords, std::vector<float> *cell_values, std::vector<std::vector<float>> *points, std::vector<float> *params){
        std::vector<std::vector<float>> result;
        for(auto &p: *points){
          std::vector<float> d, w;
          bool exact = false;
          for(auto &c: *cell_coords){
            float dist = distance(&p, &c);
            d.push_back(dist);
            if(dist == 0) exact = true;
          }
          for(float dist: d){
            w.push_back(exact ? (dist == 0 ? 1.0f : 0.0f) : 1/std::pow(dist, exponent));
          }
          result.push_back(w);
        }
        return(result);
      }
  };

  // ordinary-kriging weights, made non-negative: "clip" sets the negative ones to 0, "abs" takes
  // their absolute value; with nothing positive left, the nearest datum takes all the weight
  class KrigingWeights: public CellWeights {
    protected:
      KrigingDecoder kriging;
      bool use_abs;
    public:
      KrigingWeights(int model, float nugget, float range, float sill, const std::string &negative_weights):
        kriging(model, nugget, range, sill){
        if(negative_weights == "clip") use_abs = false;
        else if(negative_weights == "abs") use_abs = true;
        else throw std::runtime_error("negative_weights must be 'clip' or 'abs', not '" + negative_weights + "'");
      }
      std::vector<std::vector<float>> at(std::vector<std::vector<float>> *cell_coords, std::vector<float> *cell_values, std::vector<std::vector<float>> *points, std::vector<float> *params){
        auto result = kriging.weights(cell_coords, points);
        for(size_t q=0; q<result.size(); q++){
          double total = 0.0;
          for(auto &w: result[q]){
            w = use_abs ? std::fabs(w) : std::max(w, 0.0f);
            if(!std::isfinite(w)) w = 0.0f;
            total += w;
          }
          if(!(total > 0)){
            size_t nearest = 0;
            float best = INFINITY;
            for(size_t j=0; j<cell_coords->size(); j++){
              float dist = distance(&(points->at(q)), &(cell_coords->at(j)));
              if(dist < best){ best = dist; nearest = j; }
            }
            std::fill(result[q].begin(), result[q].end(), 0.0f);
            result[q][nearest] = 1.0f;
          }
        }
        return(result);
      }
  };

  // ----------------------------------------------------------------------------------------------

  // The weighted mean of the cell's data; with uniform weights, the cell mean, the linear decoder
  // for which the theory's co-occurrence identity is exact.
  class WeightedMeanDecoder: public Decoder {
    protected:
      std::unique_ptr<CellWeights> weights;

      float mean_of(std::vector<float> &w, std::vector<float> &v){
        double sw = 0.0, swv = 0.0;
        for(size_t j=0; j<w.size(); j++){ sw += w[j]; swv += w[j]*v[j]; }
        return(sw > 0 ? static_cast<float>(swv/sw) : NAN);
      }

      float mean_from(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> &ids, int target, std::vector<float> *params){
        auto sl_coords = slice(coords, &ids);
        auto sl_values = slice(values, &ids);
        std::vector<std::vector<float>> point = {coords->at(target)};
        auto w = weights->at(&sl_coords, &sl_values, &point, params);
        return(mean_of(w[0], sl_values));
      }

    public:
      WeightedMeanDecoder(CellWeights *_weights): weights(_weights){}

      std::vector<float> fit_cell(std::vector<std::vector<float>> *coords, std::vector<float> *values,
                                  const std::vector<int> &samples, unsigned int seed){
        return(weights->fit_cell(coords, values, samples, seed));
      }

      void fit(std::vector<Partition*> *forest, std::vector<std::vector<float>> *coords, std::vector<float> *values,
               std::mt19937 &rng, std::function<int(std::string)> visitor, std::string class_name){
        weights->fit(forest, coords, values, rng, visitor, class_name);
      }

      std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params, const CellContext &cell){
        std::vector<float> result(locations_id->size(), NAN);
        if(samples_id->empty() || locations_id->empty()){
          return(result);
        }
        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        auto sl_points = slice(locations, locations_id);
        auto w = weights->at(&sl_coords, &sl_values, &sl_points, params);
        for(size_t q=0; q<locations_id->size(); q++){
          result[q] = mean_of(w[q], sl_values);
        }
        return(result);
      }

      std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params, const CellContext &cell){
        std::vector<float> result(samples_id->size(), NAN);
        if(samples_id->size() <= 1){
          return(result);
        }
        for(size_t i=0; i<samples_id->size(); i++){
          std::vector<int> others;
          for(size_t j=0; j<samples_id->size(); j++){
            if(i != j) others.push_back(samples_id->at(j));
          }
          result[i] = mean_from(coords, values, others, samples_id->at(i), params);
        }
        return(result);
      }

      std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *folds, std::vector<int> *samples_id, std::vector<float> *params, const CellContext &cell){
        std::vector<float> result(samples_id->size(), NAN);
        if(samples_id->size() <= 1){
          return(result);
        }
        auto sl_folds = slice(folds, samples_id);
        for(int f=0; f<k; f++){
          auto test_train = indexes_by_predicate<int>(&sl_folds, [f](int *j){return(*j==f);});
          if(test_train.first.empty() || test_train.second.empty()) continue;
          std::vector<int> train;
          for(int l: test_train.second) train.push_back(samples_id->at(l));
          for(int j: test_train.first){
            result.at(j) = mean_from(coords, values, train, samples_id->at(j), params);
          }
        }
        return(result);
      }
  };

  class DrawDecoder: public Decoder {
    protected:
      std::unique_ptr<CellWeights> weights;

    public:
      DrawDecoder(CellWeights *_weights): weights(_weights){}

      std::vector<float> fit_cell(std::vector<std::vector<float>> *coords, std::vector<float> *values,
                                  const std::vector<int> &samples, unsigned int seed){
        return(weights->fit_cell(coords, values, samples, seed));
      }

      void fit(std::vector<Partition*> *forest, std::vector<std::vector<float>> *coords, std::vector<float> *values,
               std::mt19937 &rng, std::function<int(std::string)> visitor, std::string class_name){
        weights->fit(forest, coords, values, rng, visitor, class_name);
      }

      std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params, const CellContext &cell){
        std::vector<float> result(locations_id->size(), NAN);
        if(samples_id->empty() || locations_id->empty()){
          return(result);
        }
        auto sl_coords = slice(coords, samples_id);
        auto sl_values = slice(values, samples_id);
        auto sl_points = slice(locations, locations_id);
        auto w = weights->at(&sl_coords, &sl_values, &sl_points, params);
        for(size_t q=0; q<locations_id->size(); q++){
          int j = draw_index(w[q], draw_uniform(cell, locations_id->at(q), DrawRole::ESTIMATE));
          if(j >= 0) result[q] = sl_values[j];
        }
        return(result);
      }

      // every sample is drawn among the others of its cell
      std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params, const CellContext &cell){
        std::vector<float> result(samples_id->size(), NAN);
        if(samples_id->size() <= 1){
          return(result);
        }
        for(size_t i=0; i<samples_id->size(); i++){
          std::vector<int> others;
          for(size_t j=0; j<samples_id->size(); j++){
            if(i != j) others.push_back(samples_id->at(j));
          }
          result[i] = draw_from(coords, values, others, samples_id->at(i), params, cell, DrawRole::LOO);
        }
        return(result);
      }

      // the samples of each fold are drawn among the samples of the other folds in the cell
      std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *folds, std::vector<int> *samples_id, std::vector<float> *params, const CellContext &cell){
        std::vector<float> result(samples_id->size(), NAN);
        if(samples_id->size() <= 1){
          return(result);
        }
        auto sl_folds = slice(folds, samples_id);
        for(int f=0; f<k; f++){
          auto test_train = indexes_by_predicate<int>(&sl_folds, [f](int *j){return(*j==f);});
          if(test_train.first.empty() || test_train.second.empty()) continue;
          std::vector<int> train;
          for(int l: test_train.second) train.push_back(samples_id->at(l));
          for(int j: test_train.first){
            result.at(j) = draw_from(coords, values, train, samples_id->at(j), params, cell, DrawRole::KFOLD);
          }
        }
        return(result);
      }

    protected:
      float draw_from(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> &ids, int target, std::vector<float> *params, const CellContext &cell, DrawRole role){
        auto sl_coords = slice(coords, &ids);
        auto sl_values = slice(values, &ids);
        std::vector<std::vector<float>> point = {coords->at(target)};
        auto w = weights->at(&sl_coords, &sl_values, &point, params);
        int j = draw_index(w[0], draw_uniform(cell, target, role));
        return(j < 0 ? NAN : values->at(ids[j]));
      }
  };
}

#endif

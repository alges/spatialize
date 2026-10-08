#ifndef _SPTLZ_DECODERS_CUSTOM_
#define _SPTLZ_DECODERS_CUSTOM_

#include <stdexcept>
#include <cmath>
#include <functional>
#include "spatialize/decoder.hpp"
#include "spatialize/utils.hpp"

namespace sptlz{
  // Decoder defined by user callbacks (per-cell post-creation, estimation, LOO, k-fold).
  class CustomDecoder: public Decoder {
    public:
      typedef std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*)> Post;
      typedef std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*, std::vector<std::vector<float>>*, std::vector<float> *)> Estimation;
      typedef std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*, std::vector<float> *)> Loo;
      typedef std::function<std::vector<float>(int, std::vector<std::vector<float>>*, std::vector<float>*, std::vector<int> *, std::vector<float> *)> Kfold;

    protected:
      Post post_creation;
      Estimation estimation_by_leaf;
      Loo loo_by_leaf;
      Kfold kfold_by_leaf;

    public:
      CustomDecoder(Post _post, Estimation _est, Loo _loo, Kfold _kfold){
        post_creation = _post;
        estimation_by_leaf = _est;
        loo_by_leaf = _loo;
        kfold_by_leaf = _kfold;
      }

      // the callbacks are Python functions, which need the GIL: one cell at a time
      bool thread_safe(){ return false; }

      std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params, const CellContext &cell){
        auto _coords = slice(coords, samples_id);
        auto _values = slice(values, samples_id);
        auto _locations = slice(locations, locations_id);
        auto result = estimation_by_leaf(&_coords, &_values, &_locations, params);
        return(result);
      }

      std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params, const CellContext &cell){
        auto _coords = slice(coords, samples_id);
        auto _values = slice(values, samples_id);
        auto result = loo_by_leaf(&_coords, &_values, params);
        return(result);
      }

      std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *folds, std::vector<int> *samples_id, std::vector<float> *params, const CellContext &cell){
        auto _coords = slice(coords, samples_id);
        auto _values = slice(values, samples_id);
        auto _folds = slice(folds, samples_id);
        auto result = kfold_by_leaf(k, &_coords, &_values, &_folds, params);
        return(result);
      }

      std::vector<float> fit_cell(std::vector<std::vector<float>> *coords, std::vector<float> *values,
                                  const std::vector<int> &samples, unsigned int seed){
        if(post_creation == NULL) return(std::vector<float>());
        std::vector<std::vector<float>> c;
        std::vector<float> v;
        for(int d: samples){ c.push_back(coords->at(d)); v.push_back(values->at(d)); }
        return(post_creation(&c, &v));
      }

      // Computes per-cell parameters with the user's post-creation callback, if any.
      void fit(std::vector<Partition*> *forest,
               std::vector<std::vector<float>> *coords,
               std::vector<float> *values,
               std::mt19937 &rng,
               std::function<int(std::string)> visitor,
               std::string class_name){

        if(post_creation == NULL){
            return;
        }
        auto &mondrian_forest = *forest;

        std::vector<std::vector<float>> leaf_coords;
        std::vector<float> leaf_values;
        sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(visitor, class_name);
        sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(visitor);

        logger->info("computing cell parameters");

        progress->init(static_cast<int>(mondrian_forest.size()), 1);

        for(int i=0; i<mondrian_forest.size(); i++){
          auto mt = mondrian_forest.at(i);
          for(int j=0; j<mt->samples_by_leaf.size(); j++){
            leaf_coords.clear();
            leaf_values.clear();
            for(int k=0; k<mt->samples_by_leaf.at(j).size(); k++){
              leaf_coords.push_back(coords->at(mt->samples_by_leaf.at(j).at(k)));
              leaf_values.push_back(values->at(mt->samples_by_leaf.at(j).at(k)));
            }
            mt->leaf_params.at(j) = post_creation(&leaf_coords, &leaf_values);
            if (PyErr_CheckSignals() != 0)  // check after each leaf (Python callback may be slow)
              throw pybind11::error_already_set();
          }
          if (PyErr_CheckSignals() != 0)  // to allow ctrl-c from user
            throw pybind11::error_already_set();
		  progress->inform(i + 1);
        }

        progress->stop();

        delete logger;
        delete progress;
      }

  };

}

#endif

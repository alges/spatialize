#ifndef _SPTLZ_NN_IDW_
#define _SPTLZ_NN_IDW_

#include <vector>
#include <random>
#include "spatialize/nn/abstract_nn.hpp"
#include "spatialize/kdtree.hpp"
#include "spatialize/utils.hpp"
#include "spatialize/decoders/idw.hpp"

namespace sptlz{
	// IDW over the data within a radius, with the weights of the ensemble's IDW decoder: 1/d^p
	// relative to the nearest datum, (d_min/d)^p, data at distance 0 taking all the weight
	// (relative_idw). Until 1.3 the weights were 1/(1+d^p), which depend on the units of the
	// coordinates (close to a plain mean on a unit square).
	class NN_IDW: public NN {
		protected:
			float exponent;

      float estimate_point(std::pair<std::vector<float>, std::vector<int>> *nbs, std::vector<float> *pt){
        return(relative_idw([&](auto f){
          for(size_t j=0; j<nbs->first.size(); j++) f(nbs->first.at(j), values.at(nbs->second.at(j)), 1.0f);
        }, this->exponent));
      }

      // the neighbours of datum i other than i itself (nbs->second holds sample ids, not positions)
      float estimate_loo(std::pair<std::vector<float>, std::vector<int>> *nbs, size_t i){
        return(relative_idw([&](auto f){
          for(size_t j=0; j<nbs->first.size(); j++){
            if(nbs->second.at(j) != static_cast<int>(i)) f(nbs->first.at(j), values.at(nbs->second.at(j)), 1.0f);
          }
        }, this->exponent));
      }

      // the neighbours of datum i in other folds than i's, each read by its own sample id
      float estimate_kfold(std::pair<std::vector<float>, std::vector<int>> *nbs, int i, std::vector<int> *folds){
        int this_class = folds->at(i);
        return(relative_idw([&](auto f){
          for(size_t j=0; j<nbs->first.size(); j++){
            int id = nbs->second.at(j);
            if(folds->at(id) != this_class) f(nbs->first.at(j), values.at(id), 1.0f);
          }
        }, this->exponent));
      }

		public:
			NN_IDW(std::vector<std::vector<float>> _coords,
			       std::vector<float> _values,
			       std::vector<float> _search_params,
			       float exponent,
			       std::function<int(std::string)> visitor):
			NN(_coords, _values, _search_params, visitor){
			    class_name = __func__;
                this->exponent = exponent;
			}
	};
}

#endif

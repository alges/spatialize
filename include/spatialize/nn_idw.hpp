#ifndef _SPTLZ_NN_IDW_
#define _SPTLZ_NN_IDW_

#include <vector>
#include <random>
#include "spatialize/abstract_nn.hpp"
#include "spatialize/kdtree.hpp"
#include "spatialize/utils.hpp"

namespace sptlz{
	class NN_IDW: public NN {
		protected:
			float exponent;

      float estimate_point(std::pair<std::vector<float>, std::vector<int>> *nbs, std::vector<float> *pt){
        float w, w_sum = 0.0, w_v_sum = 0.0;

        for(size_t j=0; j<nbs->first.size(); j++){
          // calculate weight
          w = static_cast<float>(1.0/(1.0+std::pow(nbs->first.at(j), this->exponent)));
          // keep sum of weighted values and sum of weights
          w_sum += w;
          w_v_sum += w*values.at(nbs->second.at(j));
        }
        // return weighted values sum normalized (divided by weights sum)
        return(w_v_sum/w_sum);
      }

      float estimate_loo(std::pair<std::vector<float>, std::vector<int>> *nbs, size_t i){
        float w, w_sum = 0.0, w_v_sum = 0.0;

        for(size_t j=0; j<nbs->first.size(); j++){
            // do not use the same point
            if(i!=j){
            // calculate weight
            w = static_cast<float>(1.0/(1.0+std::pow(nbs->first.at(j), this->exponent)));
            // keep sum of weighted values and sum of weights
            w_sum += w;
            w_v_sum += w*values.at(nbs->second.at(j));
          }
        }
        // return weighted values sum normalized (divided by weights sum)
        return(w_v_sum/w_sum);
      }

      float estimate_kfold(std::pair<std::vector<float>, std::vector<int>> *nbs, int i, std::vector<int> *folds){
        float w, w_sum = 0.0, w_v_sum = 0.0;
        int this_class = folds->at(i);

        for(size_t j=0; j<nbs->first.size(); j++){
            if(this_class!=folds->at(j)){
            // calculate weight
            w = static_cast<float>(1.0/(1.0+std::pow(nbs->first.at(j), this->exponent)));
            // keep sum of weighted values and sum of weights
            w_sum += w;
            w_v_sum += w*values.at(nbs->second.at(j));
          }
        }
        // return weighted values sum normalized (divided by weights sum)
        return(w_v_sum/w_sum);
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

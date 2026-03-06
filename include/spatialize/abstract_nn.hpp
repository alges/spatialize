#ifndef _SPTLZ_NN_
#define _SPTLZ_NN_

#include <vector>
#include <random>
#include <functional>
#include "kdtree.hpp"
#include "utils.hpp"
#include "callback_logging.hpp"

namespace sptlz{

	class NN{
		protected:
		    std::string class_name;
		    std::function<int(std::string)> callback_visitor;
			int n_samples, n_dims;
			float radius;
			std::vector<std::vector<float>> coords;
			std::vector<float> values;
			std::vector<float> search_params;
            sptlz::KDTree<float> *kdt;

      virtual float estimate_point(std::pair<std::vector<float>, std::vector<int>> *nbs, std::vector<float> *pt){
				throw std::runtime_error("must override");
      }

      virtual float estimate_loo(std::pair<std::vector<float>, std::vector<int>> *nbs, size_t i){
				throw std::runtime_error("must override");
      }

      virtual float estimate_kfold(std::pair<std::vector<float>, std::vector<int>> *nbs, int i, std::vector<int> *folds){
				throw std::runtime_error("must override");
      }

		public:
			NN(std::vector<std::vector<float>> _coords,
			               std::vector<float> _values,
			               std::vector<float> _search_params,
			               std::function<int(std::string)> visitor){
			    this->class_name = __func__;
			    this->callback_visitor = visitor;
				this->n_samples = _coords.size();
				this->n_dims = _coords.at(0).size();
				this->coords = _coords;
				this->values = _values;

                this->search_params = _search_params; // TODO: for anisotropic searches, scale, rotate and set radius=1
                this->radius = search_params.at(0);

                this->kdt = new sptlz::KDTree<float>(&(this->coords));
			}

			~NN() {
                if(this->kdt != NULL){
                    delete(this->kdt);
                }
            }


			std::vector<float> estimate(std::vector<std::vector<float>> *locations){
				std::stringstream json;
				std::vector<float> result;
				float value;
				int n = locations->size();

				sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
                sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

                logger->debug("computing estimates");

				progress->init(n, 1);

                for(int i=0; i<n; i++){
                  if (PyErr_CheckSignals() != 0)  // to allow ctrl-c from user
                      throw pybind11::error_already_set();
                  progress->inform(100.0*(i+1.0)/n);

                  auto nbs = this->kdt->query_ball(&(locations->at(i)), radius, 2.0);
                  value = this->estimate_point(&nbs, &(locations->at(i)));
                  result.push_back(value);
                }

                progress->stop();

                delete logger;
                delete progress;
				return(result);
			}

			std::vector<float> leave_one_out(){
				std::stringstream json;
				std::vector<float> result;
				float value;
				int n = coords.size();

				sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
                sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

                logger->debug("computing leave-one-out");

				progress->init(n, 1);

                for(int i=0; i<n; i++){
                  if (PyErr_CheckSignals() != 0)  // to allow ctrl-c from user
                      throw pybind11::error_already_set();
                  progress->inform(100.0*(i+1.0)/n);

                  auto nbs = this->kdt->query_ball(&(coords.at(i)), radius, 2.0);
                  value = this->estimate_loo(&nbs, i);
                  result.push_back(value);
                }

                progress->stop();

                delete logger;
                delete progress;
				return(result);
			}

			std::vector<float> k_fold(int k, int seed=206936){
				std::stringstream json;
				std::uniform_real_distribution<float> uni_float;
				std::mt19937 my_rand(seed);
				std::vector<float> result;
				auto folds = get_folds(values.size(), k, uni_float(my_rand));
				float value;
				int n = coords.size();

				sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
                sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

                logger->debug("computing k-fold");

				progress->init(n, 1);

                for(int i=0; i<n; i++){
                  if (PyErr_CheckSignals() != 0)  // to allow ctrl-c from user
                      throw pybind11::error_already_set();
                  progress->inform(100.0*(i+1.0)/n);

                  auto nbs = this->kdt->query_ball(&(coords.at(i)), radius, 2.0);
                  value = this->estimate_kfold(&nbs, i, &folds);
                  result.push_back(value);
                }

                progress->stop();

                delete logger;
                delete progress;
				return(result);
			}

	};
}

#endif

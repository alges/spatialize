#ifndef _SPTLZ_CUSTOM_COESI_
#define _SPTLZ_CUSTOM_COESI_

#include <sstream>
#include <random>
#include <queue>
#include <string>
#include <functional>
#include <stdexcept>
#include "utils.hpp"
#include "spatialize/abstract_esi.hpp"
#include "spatialize/custom_esi.hpp"
#include "callback_logging.hpp"

namespace sptlz{
	class CUSTOM_COESI {
		protected:
			std::string class_name;
			std::function<int(std::string)> callback_visitor;
			std::vector<std::vector<float>> coords;
			std::vector<std::vector<float>> values;
			std::vector<std::vector<float>> bbox;
			std::vector<sptlz::MondrianTree*> mondrian_forest;
			std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<std::vector<float>>*)> post_creation;
			std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<std::vector<float>>*, std::vector<std::vector<float>>*, std::vector<float> *)> estimation_by_leaf;
			std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<std::vector<float>>*, std::vector<float> *)> loo_by_leaf;
			std::function<std::vector<float>(int, std::vector<std::vector<float>>*, std::vector<std::vector<float>>*, std::vector<int> *, std::vector<float> *)> kfold_by_leaf;
			std::vector<std::function<float(std::vector<float>*)>> esis_agg;

			std::vector<CUSTOM_ESI*> esis;
			std::mt19937 my_rand;
			bool debug;

			std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<std::vector<float>> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params){
				auto _coords = slice(coords, samples_id);
				auto _values = slice(values, samples_id);
				auto _locations = slice(locations, locations_id);
				auto result = estimation_by_leaf(&_coords, &_values, &_locations, params);
				return(result);
			}

/*			virtual std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params){
				throw std::runtime_error("must override");
			}

			virtual std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *fold, std::vector<int> *samples_id, std::vector<float> *params){
				throw std::runtime_error("must override");
			}

			virtual void post_process(){}
*/
		public:
			CUSTOM_COESI(float lambda,
				int forest_size,
				int n_temp_samples,
				std::vector<std::vector<float>> _bbox,
				std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<std::vector<float>>*)> post,
				std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<std::vector<float>>*, std::vector<std::vector<float>>*, std::vector<float> *)> est,
				std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<std::vector<float>>*, std::vector<float> *)> loo,
				std::function<std::vector<float>(int, std::vector<std::vector<float>>*, std::vector<std::vector<float>>*, std::vector<int> *, std::vector<float> *)> kfold,
				std::function<int(std::string)> visitor,
				int seed=206936){
					this->class_name = __func__;
					this->callback_visitor = visitor;
					this->my_rand = std::mt19937(seed);
					this->post_creation = post;
					this->estimation_by_leaf = est;
					this->loo_by_leaf = loo;
					this->kfold_by_leaf = kfold;
					this->bbox = _bbox;
					// create positions for this esi (locations for esis)
					std::uniform_int_distribution<int> uni_int;
					this->coords = get_random_values(&bbox, n_temp_samples, uni_int(my_rand));
					for(int i=0; i<n_temp_samples; i++){
						this->values.push_back({});
					}

					for(int i=0; i<forest_size; i++){
						mondrian_forest.push_back(new sptlz::MondrianTree(&coords, lambda, bbox, uni_int(my_rand)));
					}
			}

            /* Needs to be defined as 'virtual' because this an abstract class
               and to avoid the warning:
               "delete called on non-final that has virtual functions but non-virtual destructor"
            */
			virtual ~CUSTOM_COESI(){
				for(int i=0; i<esis.size(); i++){
					delete(esis.at(i));
				}
				for(int i=0; i<this->mondrian_forest.size(); i++){
					delete(this->mondrian_forest.at(i));
				}
			}

			void add_variable(std::vector<std::vector<float>> _coords,
				std::vector<float> _values,
				float lambda,
				int forest_size,
				std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*)> _post,
            std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*, std::vector<std::vector<float>>*, std::vector<float> *)> _est,
            std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*, std::vector<float> *)> _loo,
            std::function<std::vector<float>(int, std::vector<std::vector<float>>*, std::vector<float>*, std::vector<int> *, std::vector<float> *)> _kfold,
            std::function<float(std::vector<float>*)> agg){
				// starts the function body
				std::uniform_int_distribution<int> uni_int;

				CUSTOM_ESI *esi = new CUSTOM_ESI(_coords, _values, lambda, forest_size, this->bbox, _post, _est, _loo, _kfold, this->callback_visitor, uni_int(my_rand));
				this->esis.push_back(esi);
				this->esis_agg.push_back(agg);
			}

			int forest_size(){
				return(static_cast<int>(this->mondrian_forest.size()));
			}

			MondrianTree *get_tree(int i){
				return(this->mondrian_forest.at(i));
			}

			std::vector<std::vector<float>> *get_coords(){
				return(&(this->coords));
			}

			std::vector<std::vector<float>> *get_values(){
				return(&(this->values));
			}

			std::vector<std::vector<int>> get_partitions(){
				int n = static_cast<int>(mondrian_forest.size());
				std::vector<std::vector<int>> results(this->coords.size());

				for(int i=0; i<n; i++){
					// get tree
					auto lfs = mondrian_forest.at(i)->leaf_for_sample;
					for(size_t j=0; j<this->coords.size(); j++){
						results.at(j).push_back(lfs.at(j));
					}
				}
				return(results);
			}

			std::vector<std::vector<float>> estimate(std::vector<std::vector<float>> *locations){
				size_t i, j;
				std::stringstream json;
				std::vector<std::vector<float>> results(locations->size());
				std::vector<std::vector<int>> locations_by_leaf;
				int aux, n = static_cast<int>(mondrian_forest.size());
				sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
				sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

				logger->debug("computing estimates");

				progress->init(n, 1);

				for(i=0; i<this->esis.size(); i++){
					auto ind_est = this->esis.at(i)->estimate(&coords);
					for(j=0; j<ind_est.size(); j++){
						this->values.at(j).push_back(esis_agg.at(0)(&(ind_est.at(i))));
					}
				}

				for(int i=0; i<n; i++){
					// get tree
					auto mt = mondrian_forest.at(i);
					locations_by_leaf = std::vector<std::vector<int>>(mt->leaves.size());

					// join all locations for same leaf
					for(int j=0; j<locations->size(); j++){
						aux = mt->search_leaf(locations->at(j));
						locations_by_leaf.at(aux).push_back(j);
					}

					// make estimation by leaf
					for(j=0; j<locations_by_leaf.size(); j++){
						if(mt->samples_by_leaf.at(j).size()==0){
							for(size_t k=0; k<locations_by_leaf.at(j).size(); k++){
								results.at(locations_by_leaf.at(j).at(k)).push_back(NAN);
							}
						}else{
							auto predictions = leaf_estimation(&coords, &values, &(mt->samples_by_leaf.at(j)), locations, &(locations_by_leaf.at(j)), &(mt->leaf_params.at(j)));
							for(size_t k=0; k<locations_by_leaf.at(j).size(); k++){
								results.at(locations_by_leaf.at(j).at(k)).push_back(predictions.at(k));
							}
						}
					}

					if (PyErr_CheckSignals() != 0)  // to allow ctrl-c from user
                      throw pybind11::error_already_set();
					progress->inform(static_cast<int>(100.0*(i+1.0)/n));
				}

				progress->stop();

				delete logger;
				delete progress;
				return(results);
			}

			std::vector<std::vector<std::vector<float>>> marginal_leave_one_out(){
				std::vector<std::vector<std::vector<float>>> results;
				int i;

				sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
                sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

				logger->debug("computing marginal leave-one-out");

				for(i=0; i<this->esis.size(); i++){
					results.push_back(this->esis.at(i)->leave_one_out());
				}

				delete logger;
				return(results);
			}

			std::vector<std::vector<std::vector<float>>> marginal_k_fold(int k, int seed=206936){
				std::vector<std::vector<std::vector<float>>> results;
				auto rand = std::mt19937(seed);
				std::uniform_int_distribution<int> uni_int;
				int i;

				sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
                sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

				logger->debug("computing marginal k-fold");

				for(i=0; i<this->esis.size(); i++){
					results.push_back(this->esis.at(i)->k_fold(k, uni_int(rand)));
				}

				delete logger;
				return(results);
			}

/*
			std::vector<std::vector<float>> leave_one_out(){
				std::stringstream json;
				std::vector<std::vector<float>> results(coords.size());
				int n = mondrian_forest.size();

				sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
                sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

				logger->debug("computing leave-one-out");

				progress->init(n, 1);

				for(int i=0; i<n; i++){
					// get tree
					auto mt = mondrian_forest.at(i);

					// make loo by leaf
					for(size_t j=0; j<mt->samples_by_leaf.size(); j++){
						if(mt->samples_by_leaf.at(j).size()!=0){
							auto predictions = leaf_loo(&coords, &values, &(mt->samples_by_leaf.at(j)), &(mt->leaf_params.at(j)));
							for(size_t k=0; k<mt->samples_by_leaf.at(j).size(); k++){
								results.at(mt->samples_by_leaf.at(j).at(k)).push_back(predictions.at(k));
							}
						}
					}

					if (PyErr_CheckSignals() != 0)  // to allow ctrl-c from user
                      throw pybind11::error_already_set();
					progress->inform(100.0*(i+1.0)/n);
				}

				progress->stop();

				delete logger;
				delete progress;
				return(results);
			}

			std::vector<std::vector<float>> k_fold(int k, int seed=206936){
				std::stringstream json;
				auto fold_rand = std::mt19937(seed);
				std::uniform_real_distribution<float> uni_float;
				auto folds = get_folds(values.size(), k, uni_float(fold_rand));
				std::vector<std::vector<float>> results(coords.size());
				int n = mondrian_forest.size();

				sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
                sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

				logger->debug("computing k-fold");

				progress->init(n, 1);

				for(int i=0; i<n; i++){
					// get tree
					auto mt = mondrian_forest.at(i);
					// make kfold by leaf
					for(size_t j=0; j<mt->samples_by_leaf.size(); j++){
						if(mt->samples_by_leaf.at(j).size()!=0){
							auto predictions = leaf_kfold(k, &coords, &values, &folds, &(mt->samples_by_leaf.at(j)), &(mt->leaf_params.at(j)));
							for(size_t l=0; l<mt->samples_by_leaf.at(j).size(); l++){
								results.at(mt->samples_by_leaf.at(j).at(l)).push_back(predictions.at(l));
							}
						}
					}


					if (PyErr_CheckSignals() != 0)  // to allow ctrl-c from user
                      throw pybind11::error_already_set();
					progress->inform(100.0*(i+1.0)/n);
				}

				progress->stop();

				delete logger;
				delete progress;
				return(results);
			}
*/
	};
}

#endif

#ifndef _SPTLZ_ENSEMBLE_
#define _SPTLZ_ENSEMBLE_

#include <sstream>
#include <random>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include "spatialize/utils.hpp"
#include "spatialize/callback_logging.hpp"
#include "spatialize/partition.hpp"
#include "spatialize/decoder.hpp"

namespace sptlz{
	// Ensemble of random partitions (encoder) with a local interpolator (decoder): one loop for
	// estimation, leave-one-out and k-fold, whatever the partition process and the decoder.
	// Subclasses draw the forest (ESI: Mondrian, VORONOI: Voronoi); set_decoder() then fits the
	// decoder with the same generator, continuing its sequence.
	class Ensemble {
		protected:
			std::string class_name;
			std::function<int(std::string)> callback_visitor;
			std::vector<sptlz::Partition*> forest;
			std::vector<std::vector<float>> coords;
			std::vector<float> values;
			std::mt19937 my_rand;
			sptlz::Decoder *decoder;
			bool estimate_log_debug;  // log "computing estimates" at debug level (Voronoi) instead of info

		public:
			Ensemble(std::vector<std::vector<float>> _coords,
			         std::vector<float> _values,
			         std::function<int(std::string)> visitor,
			         int seed=206936){
				this->class_name = __func__;
				this->callback_visitor = visitor;
				this->my_rand = std::mt19937(seed);
				this->coords = _coords;
				this->values = _values;
				this->decoder = NULL;
				this->estimate_log_debug = false;
			}

			virtual ~Ensemble(){
				for(size_t i=0; i<this->forest.size(); i++){
					delete(this->forest.at(i));
				}
				std::vector<sptlz::Partition*>().swap(forest);
				if(this->decoder != NULL){
					delete(this->decoder);
				}
			}

			// Takes ownership of the decoder and fits it on the forest.
			void set_decoder(sptlz::Decoder *_decoder){
				if(this->decoder != NULL){
					delete(this->decoder);
				}
				this->decoder = _decoder;
				this->decoder->fit(&(this->forest), &(this->coords), &(this->values), this->my_rand, this->callback_visitor, this->class_name);
			}

			int forest_size(){
				return(static_cast<int>(this->forest.size()));
			}

			// name used in log messages
			void set_class_name(std::string _class_name){
				this->class_name = _class_name;
			}

			std::vector<std::vector<int>> get_leaf_for_samples(){
				int n = static_cast<int>(forest.size());
				std::vector<std::vector<int>> results(this->coords.size());

				for(int i=0; i<n; i++){
					// get tree
					auto lfs = forest.at(i)->leaf_for_sample;
					for(size_t j=0; j<this->coords.size(); j++){
						results.at(j).push_back(lfs.at(j));
					}
				}
				return(results);
			}

			std::vector<std::vector<float>> estimate(std::vector<std::vector<float>> *locations){
				std::vector<std::vector<float>> results(locations->size());
				std::vector<std::vector<int>> locations_by_leaf;
				int aux, n = static_cast<int>(forest.size());
				sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
				sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

				if(this->estimate_log_debug){
					logger->debug("computing estimates");
				}else{
					logger->info("computing estimates");
				}

				progress->init(n, 1);

				for(int i=0; i<n; i++){
					// get tree
					auto mt = forest.at(i);
					locations_by_leaf = std::vector<std::vector<int>>(mt->n_leaves());

					// join all locations for same leaf
					for(size_t j=0; j<locations->size(); j++){
						aux = mt->search_leaf(locations->at(j));
						locations_by_leaf.at(aux).push_back(static_cast<int>(j));
					}

					// make estimation by leaf
					for(size_t j=0; j<locations_by_leaf.size(); j++){
						if(mt->samples_by_leaf.at(j).size()==0){
							for(size_t k=0; k<locations_by_leaf.at(j).size(); k++){
								results.at(locations_by_leaf.at(j).at(k)).push_back(NAN);
							}
						}else{
							auto predictions = decoder->leaf_estimation(&coords, &values, &(mt->samples_by_leaf.at(j)), locations, &(locations_by_leaf.at(j)), &(mt->leaf_params.at(j)));
							for(size_t k=0; k<locations_by_leaf.at(j).size(); k++){
								results.at(locations_by_leaf.at(j).at(k)).push_back(predictions.at(k));
							}
							if (PyErr_CheckSignals() != 0)  // check after each leaf callback
								throw pybind11::error_already_set();
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

			std::vector<std::vector<float>> leave_one_out(){
				std::vector<std::vector<float>> results(coords.size());
				int n = static_cast<int>(forest.size());

				sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
				sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

				logger->debug("computing leave-one-out");

				progress->init(n, 1);

				for(int i=0; i<n; i++){
					// get tree
					auto mt = forest.at(i);

					// make loo by leaf
					for(size_t j=0; j<mt->samples_by_leaf.size(); j++){
						if(mt->samples_by_leaf.at(j).size()!=0){
							auto predictions = decoder->leaf_loo(&coords, &values, &(mt->samples_by_leaf.at(j)), &(mt->leaf_params.at(j)));
							for(size_t k=0; k<mt->samples_by_leaf.at(j).size(); k++){
								results.at(mt->samples_by_leaf.at(j).at(k)).push_back(predictions.at(k));
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

			std::vector<std::vector<float>> k_fold(int k, int seed=206936){
				auto fold_rand = std::mt19937(seed);
				std::uniform_real_distribution<float> uni_float;
				auto folds = get_folds(static_cast<int>(values.size()), k, uni_float(fold_rand));
				std::vector<std::vector<float>> results(coords.size());
				int n = static_cast<int>(forest.size());

				sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
				sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

				logger->debug("computing k-fold");

				progress->init(n, 1);

				for(int i=0; i<n; i++){
					// get tree
					auto mt = forest.at(i);
					// make kfold by leaf
					for(size_t j=0; j<mt->samples_by_leaf.size(); j++){
						if(mt->samples_by_leaf.at(j).size()!=0){
							auto predictions = decoder->leaf_kfold(k, &coords, &values, &folds, &(mt->samples_by_leaf.at(j)), &(mt->leaf_params.at(j)));
							for(size_t l=0; l<mt->samples_by_leaf.at(j).size(); l++){
								results.at(mt->samples_by_leaf.at(j).at(l)).push_back(predictions.at(l));
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
	};
}

#endif

#ifndef _SPTLZ_VORONOI_
#define _SPTLZ_VORONOI_

#include <sstream>
#include <random>
#include <queue>
#include <string>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include "kdtree.hpp"
#include "utils.hpp"

namespace sptlz{


	class VoronoiNode {
		protected:
			int n_dims;
		public:
			std::vector<std::vector<float>> nuclei_coords;

			VoronoiNode(){}

			VoronoiNode(std::vector<std::vector<float>> _nuclei_coords, std::vector<std::vector<float>> *_coords){
				this->nuclei_coords = _nuclei_coords;
				this->n_dims = _coords->at(0).size();
			}

			~VoronoiNode(){
			}

	};

	class VoronoiTree {
		public:
			sptlz::KDTree<float> *kdt;
			std::vector<VoronoiNode*> leaves;
			std::vector<int> leaf_for_sample;
			std::vector<std::vector<int>> samples_by_leaf;
			std::vector<std::vector<float>> leaf_params;
			std::vector<std::vector<float>> nuclei_coords;

			int vsize, ndim;

			VoronoiTree(std::vector<std::vector<float>> *coords, float alpha, std::vector<std::vector<float>> bbox, int _vsize, int seed=206936){
				this->vsize = _vsize;
				this->ndim = (int) bbox.size();
				std::mt19937 my_rand(seed);
				std::uniform_int_distribution<int> samples_choice(0, (int)coords->size()-1);

				if (alpha < 0) {
					for (int i=0; i<vsize; ++i) {
						std::vector<float> rand_coord(coords->at(0).size());
						for(int j=0; j<(int)rand_coord.size(); j++){
							for (int k=0; k<ndim; ++k) {
								std::uniform_real_distribution<float> uni_float(bbox[k][0], bbox[k][1]);
								rand_coord.at(k) = uni_float(my_rand);
							}
						}
						this->nuclei_coords.push_back(rand_coord);
						this->samples_by_leaf.push_back({});
					}
				}
				else {
					// generate as many choices as voronoi size
					for (int i=0; i<vsize; ++i) {
						int sampleid = samples_choice(my_rand);
						this->nuclei_coords.push_back(coords->at(sampleid));
						this->samples_by_leaf.push_back({});
					}

				}
				this->kdt = new sptlz::KDTree<float>(&(this->nuclei_coords));

				int aux;
				for(size_t i=0; i< coords->size(); i++){
					aux = search_leaf(coords->at(i)); // we search one nearest nuclei
					// assign samples to leafs and inverse too
					this->samples_by_leaf.at(aux).push_back(i);
					this->leaf_for_sample.push_back(aux);
					this->leaf_params.push_back({});
				}
				// build voronoi node using nuclei and all sample locations
				for (int i=0; i<vsize; ++i) {
					VoronoiNode* cur_node = new VoronoiNode(this->nuclei_coords, coords);
					this->leaves.push_back(cur_node);
				}
			}

			VoronoiTree(){}

			~VoronoiTree(){
				for(int i=0; i<this->leaves.size(); i++){
					delete(this->leaves.at(i));
				}
				std::vector<VoronoiNode*>().swap(this->leaves);
				std::vector<std::vector<int>>().swap(this->samples_by_leaf);
				std::vector<std::vector<float>>().swap(this->nuclei_coords);
				if(this->kdt != NULL){
					delete(this->kdt);
				}
			}

			int search_leaf(std::vector<float> point){
				auto nbs = this->kdt->query_nn(&(point), 1, 2.0);
				return nbs.second.front();
			}

	};

	class VORONOI {
		protected:
		    std::string class_name;
		    std::function<int(std::string)> callback_visitor;
			std::vector<sptlz::VoronoiTree*> voronoi_forest;
			std::vector<std::vector<float>> coords;
			std::vector<float> values;
			std::mt19937 my_rand;
			bool debug;

			virtual std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params){
				throw std::runtime_error("must override");
			}

			virtual std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params){
				throw std::runtime_error("must override");
			}

			virtual std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *fold, std::vector<int> *samples_id, std::vector<float> *params){
				throw std::runtime_error("must override");
			}

			virtual void post_process(){}

		public:
			VORONOI(std::vector<std::vector<float>> _coords,
			        std::vector<float> _values,
			        float alpha,
			        int forest_size,
			        std::vector<std::vector<float>> bbox,
			        std::function<int(std::string)> visitor,
			        int seed=206936){
			    this->class_name = __func__;
			    this->callback_visitor = visitor;
				this->my_rand = std::mt19937(seed);
				this->coords = _coords;
				this->values = _values;
				std::uniform_int_distribution<int> uni_int;

				std::poisson_distribution<int> pdistribution(coords.size()*0.5*std::abs(alpha)); // Poisson distribution with a mean of half the sample size

				for(int i=0; i<forest_size; i++){
					int vsize = std::max(1,pdistribution(my_rand));
					vsize = std::min(vsize, (int)coords.size());
					voronoi_forest.push_back(new sptlz::VoronoiTree(&coords, alpha, bbox, vsize, uni_int(my_rand)));
				}
			}

			VORONOI(std::vector<sptlz::VoronoiTree*> _voronoi_forest,
			        std::vector<std::vector<float>> _coords,
			        std::vector<float> _values,
			        std::function<int(std::string)> visitor) {
			    this->class_name = __func__;
			    this->callback_visitor = visitor;
				this->voronoi_forest = _voronoi_forest;
				this->coords = _coords;
				this->values = _values;
			}

            /* Needs to be defined as 'virtual' because this an abstract class
               and to avoid the warning:
               "delete called on non-final that has virtual functions but non-virtual destructor"
            */
		    virtual ~VORONOI(){
		    	for(int i=0; i<this->voronoi_forest.size(); i++){
		    		delete(this->voronoi_forest.at(i));
		    	}
				std::vector<sptlz::VoronoiTree*>().swap(voronoi_forest);
		    }

			int forest_size(){
				return(this->voronoi_forest.size());
			}

			VoronoiTree *get_tree(int i){
				return(this->voronoi_forest.at(i));
			}

			std::vector<std::vector<float>> *get_coords(){
				return(&(this->coords));
			}

			std::vector<float> *get_values(){
				return(&(this->values));
			}

			std::vector<std::vector<float>> estimate(std::vector<std::vector<float>> *locations){
				std::stringstream json;
				std::vector<std::vector<float>> results(locations->size());
				std::vector<std::vector<int>> locations_by_leaf;
				int aux, n = voronoi_forest.size();

                sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
                sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

                logger->debug("computing estimates");

				progress->init(n, 1);

				for(int i=0; i<n; i++){
					// get tree
					auto vt = voronoi_forest.at(i);
					locations_by_leaf = std::vector<std::vector<int>>(vt->leaves.size());

					// join all locations for same leaf
					for(size_t j=0; j<locations->size(); j++){
						aux = vt->search_leaf(locations->at(j));
						locations_by_leaf.at(aux).push_back(j);
					}

					// make estimation by leaf
					for(size_t j=0; j<locations_by_leaf.size(); j++){
						if(vt->samples_by_leaf.at(j).size()==0){
							for(size_t k=0; k<locations_by_leaf.at(j).size(); k++){
								results.at(locations_by_leaf.at(j).at(k)).push_back(NAN);
							}
						}else{
							auto predictions = leaf_estimation(&coords, &values, &(vt->samples_by_leaf.at(j)), locations, &(locations_by_leaf.at(j)), &(vt->leaf_params.at(j)));
							for(size_t k=0; k<locations_by_leaf.at(j).size(); k++){
								results.at(locations_by_leaf.at(j).at(k)).push_back(predictions.at(k));
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

			std::vector<std::vector<float>> leave_one_out(){
				std::stringstream json;
				std::vector<std::vector<float>> results(coords.size());
				int n = voronoi_forest.size();

				sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
                sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

				logger->debug("computing leave-one-out");

				progress->init(n, 1);

				for(int i=0; i<n; i++){
					// get tree
					auto vt = voronoi_forest.at(i);

					// make loo by leaf
					for(size_t j=0; j<vt->samples_by_leaf.size(); j++){
						if(vt->samples_by_leaf.at(j).size()!=0){
							auto predictions = leaf_loo(&coords, &values, &(vt->samples_by_leaf.at(j)), &(vt->leaf_params.at(j)));
							for(size_t k=0; k<vt->samples_by_leaf.at(j).size(); k++){
								results.at(vt->samples_by_leaf.at(j).at(k)).push_back(predictions.at(k));
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
				int n = voronoi_forest.size();

				sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
                sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

				logger->debug("computing k-fold");

				for(int i=0; i<n; i++){
					// get tree
					auto vt = voronoi_forest.at(i);
					// make kfold by leaf
					for(size_t j=0; j<vt->samples_by_leaf.size(); j++){
						if(vt->samples_by_leaf.at(j).size()!=0){
							auto predictions = leaf_kfold(k, &coords, &values, &folds, &(vt->samples_by_leaf.at(j)), &(vt->leaf_params.at(j)));
							for(size_t k=0; k<vt->samples_by_leaf.at(j).size(); k++){
								results.at(vt->samples_by_leaf.at(j).at(k)).push_back(predictions.at(k));
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
	};
}

#endif

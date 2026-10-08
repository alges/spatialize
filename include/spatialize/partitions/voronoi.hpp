#ifndef _SPTLZ_PARTITIONS_VORONOI_
#define _SPTLZ_PARTITIONS_VORONOI_

#include <sstream>
#include <random>
#include <queue>
#include <string>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include "spatialize/kdtree.hpp"
#include "spatialize/utils.hpp"
#include "spatialize/partition.hpp"
#include "spatialize/ensemble.hpp"

namespace sptlz{


	class VoronoiTree: public Partition {
		public:
			sptlz::KDTree<float> *kdt = nullptr;
			std::vector<std::vector<float>> nuclei_coords;   // one nucleus per cell

			int vsize, ndim;

			VoronoiTree(std::vector<std::vector<float>> *coords, float alpha, std::vector<std::vector<float>> bbox, int _vsize, int seed=206936){
				this->vsize = _vsize;
				this->ndim = (int) bbox.size();
				std::mt19937 my_rand(seed);
				std::uniform_int_distribution<int> samples_choice(0, (int)coords->size()-1);

				if (alpha < 0) {
					for (int i=0; i<vsize; ++i) {
						std::vector<float> rand_coord(ndim);
						for (int k=0; k<ndim; ++k) {
							std::uniform_real_distribution<float> uni_float(bbox[k][0], bbox[k][1]);
							rand_coord.at(k) = uni_float(my_rand);
						}
						this->nuclei_coords.push_back(rand_coord);
						this->samples_by_leaf.push_back({});
						this->leaf_params.push_back({});
					}
				}
				else {
					// generate as many choices as voronoi size
					for (int i=0; i<vsize; ++i) {
						int sampleid = samples_choice(my_rand);
						this->nuclei_coords.push_back(coords->at(sampleid));
						this->samples_by_leaf.push_back({});
						this->leaf_params.push_back({});
					}

				}
				this->kdt = new sptlz::KDTree<float>(&(this->nuclei_coords));

				int aux;
				for(int i=0; i< static_cast<int>(coords->size()); i++){
					aux = search_leaf(coords->at(i)); // we search one nearest nuclei
					// assign samples to leafs and inverse too
					this->samples_by_leaf.at(aux).push_back(i);
					this->leaf_for_sample.push_back(aux);
				}
			}

			VoronoiTree(){}

			~VoronoiTree(){
				std::vector<std::vector<int>>().swap(this->samples_by_leaf);
				std::vector<std::vector<float>>().swap(this->nuclei_coords);
				if(this->kdt != NULL){
					delete(this->kdt);
				}
			}

			// the nearest nucleus, other than the leaf's own, whose cell holds usable data: the Voronoi
			// tessellation of the nuclei with data, read at the point
			Coarse coarser(int leaf, const std::vector<float> &point, const std::function<bool(int)> &usable){
				Coarse out;
				float best = INFINITY;
				for(int c=0; c<static_cast<int>(nuclei_coords.size()); c++){
					if(c == leaf) continue;
					bool any = false;
					for(int d: samples_by_leaf.at(c)) if(usable(d)){ any = true; break; }
					if(!any) continue;
					float dist = distance(&nuclei_coords.at(c), const_cast<std::vector<float>*>(&point));
					if(dist < best){ best = dist; out.id = c; }
				}
				if(out.id >= 0){
					out.fitted = true;
					for(int d: samples_by_leaf.at(out.id)) if(usable(d)) out.samples.push_back(d);
				}
				return(out);
			}

			std::vector<float> leaf_point(int leaf){
				return(this->nuclei_coords.at(leaf));
			}

			int n_leaves(){
				return(static_cast<int>(this->nuclei_coords.size()));
			}

			int search_leaf(std::vector<float> point){
				auto nbs = this->kdt->query_nn(&(point), 1, 2.0);
				return nbs.second.front();
			}

	};

	// Ensemble over Voronoi partitions: each tree has max(1, Poisson(0.5·n·|alpha|)) nuclei (at
	// most n), drawn among the samples for alpha >= 0 or uniformly in `bbox` for alpha < 0.
	class VORONOI: public Ensemble {
		public:
			VORONOI(std::vector<std::vector<float>> _coords,
			        std::vector<float> _values,
			        float alpha,
			        int forest_size,
			        std::vector<std::vector<float>> bbox,
			        std::function<int(std::string)> visitor,
			        int seed=206936):
			Ensemble(_coords, _values, visitor, seed){
				this->class_name = __func__;
				this->estimate_log_debug = true;
				std::uniform_int_distribution<int> uni_int;

				std::poisson_distribution<int> pdistribution(coords.size()*0.5*std::abs(alpha)); // Poisson distribution with a mean of half the sample size

				for(int i=0; i<forest_size; i++){
					int vsize = std::max(1,pdistribution(my_rand));
					vsize = std::min(vsize, (int)coords.size());
					forest.push_back(new sptlz::VoronoiTree(&coords, alpha, bbox, vsize, uni_int(my_rand)));
				}
			}

			VoronoiTree *get_tree(int i){
				return(static_cast<VoronoiTree*>(this->forest.at(i)));
			}
	};
}

#endif

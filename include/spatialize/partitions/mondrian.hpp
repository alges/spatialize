#ifndef _SPTLZ_PARTITIONS_MONDRIAN_
#define _SPTLZ_PARTITIONS_MONDRIAN_

#include <sstream>
#include <random>
#include <queue>
#include <string>
#include <functional>
#include <stdexcept>
#include "spatialize/utils.hpp"
#include "spatialize/callback_logging.hpp"
#include "spatialize/partition.hpp"
#include "spatialize/ensemble.hpp"

namespace sptlz{
	std::string IND_STEP = "    ";

	std::string bbox_to_json(std::vector<std::vector<float>> bbox, std::string indent){
		std::vector<std::string> axis = {"X", "Y", "Z"};
		std::stringstream s;
		for(size_t i=0; i<bbox.size(); i++){
			if(i==0){
				s << "{" << std::endl;
			}
			s << indent << IND_STEP << "\"" << axis.at(i) << "\": [" << bbox.at(i).at(0) << ", " << bbox.at(i).at(1) << "]";

			if(i==bbox.size()-1){
				s  << std::endl << indent << "}";
			}else{
				s << "," << std::endl;
			}
		}
		return(s.str());
	}

	std::vector<std::vector<float>> samples_coords_bbox(std::vector<std::vector<std::vector<float>>> *coords, std::vector<std::vector<float>> *queries=NULL){
		std::vector<std::vector<float>> bbox;

		for(size_t i=0; i<coords->at(0).at(0).size(); i++){
			bbox.push_back({coords->at(0).at(0).at(i), coords->at(0).at(0).at(i)});
		}

		for(size_t i=0; i<coords->size(); i++){
			for(size_t j=0; j<coords->at(i).size(); j++){
				for(size_t k=0; k<coords->at(i).at(j).size(); k++){

					if(coords->at(i).at(j).at(k) < bbox.at(k).at(0)){
						bbox.at(k).at(0) = coords->at(i).at(j).at(k);
					}
					if(bbox.at(k).at(1) < coords->at(i).at(j).at(k)){
						bbox.at(k).at(1) = coords->at(i).at(j).at(k);
					}
				}
			}
		}

		if(queries != NULL){
			for(size_t i=0; i<queries->size(); i++){
				for(size_t j=0; j<queries->at(i).size(); j++){
					if(queries->at(i).at(j) < bbox.at(j).at(0)){
						bbox.at(j).at(0) = queries->at(i).at(j);
					}
					if(bbox.at(j).at(1) < queries->at(i).at(j)){
						bbox.at(j).at(1) = queries->at(i).at(j);
					}
				}
			}
		}

		return(bbox);
	}

	std::vector<std::vector<float>> samples_coords_bbox(std::vector<std::vector<float>> *coords, std::vector<std::vector<float>> *queries=NULL){
		std::vector<std::vector<float>> bbox;

		for(size_t i=0; i<coords->at(0).size(); i++){
			bbox.push_back({coords->at(0)[i], coords->at(0)[i]});
		}

		for(size_t i=0; i<coords->size(); i++){
			for(size_t j=0; j<coords->at(i).size(); j++){
				if(coords->at(i).at(j) < bbox.at(j).at(0)){
					bbox.at(j).at(0) = coords->at(i).at(j);
				}
				if(bbox.at(j).at(1) < coords->at(i).at(j)){
					bbox.at(j).at(1) = coords->at(i).at(j);
				}
			}
		}

		if(queries != NULL){
			for(size_t i=0; i<queries->size(); i++){
				for(size_t j=0; j<queries->at(i).size(); j++){
					if(queries->at(i).at(j) < bbox.at(j).at(0)){
						bbox.at(j).at(0) = queries->at(i).at(j);
					}
					if(bbox.at(j).at(1) < queries->at(i).at(j)){
						bbox.at(j).at(1) = queries->at(i).at(j);
					}
				}
			}
		}

		return(bbox);
	}

	float bbox_sum_interval(std::vector<std::vector<float>> bbox){
		float c = 0.0;
		for(std::vector<float> axis : bbox){
			c += (axis[1]-axis[0]);
		}
		return(c);
	}

	class MondrianNode {
		public:
			int leaf_id;
			std::vector<std::vector<float>> bbox;
			float tau, cut;
			int height, axis;
			MondrianNode* left;
			MondrianNode* right;

			MondrianNode(){}

			MondrianNode(std::vector<std::vector<float>> _bbox, float _tau, int _height){
				bbox = _bbox;
				tau = _tau;
				height = _height;
				left = NULL;
				right = NULL;
				leaf_id = -1;
			}

			~MondrianNode(){
				if(this->left != NULL){
					delete(this->left);
				}
				if(this->right != NULL){
					delete(this->right);
				}
			}

			int search_leaf(std::vector<float> point){
				if(leaf_id<0){
					if(point.at(axis) < cut){
						return(left->search_leaf(point));
					}else{
						return(right->search_leaf(point));
					}
				}else{
					return(leaf_id);
				}
			}

			std::string to_json(std::string indent){
				std::stringstream s;
				s << "{" << std::endl;
				s << indent << IND_STEP << "\"bbox\": " <<  bbox_to_json(bbox, indent+IND_STEP);
				if(leaf_id<0){
					s << ",";
				}
				s << std::endl;
				if(left != NULL){
					s << indent << IND_STEP << "\"left\": " << left->to_json(indent+IND_STEP);
					if(right != NULL){
						s << ",";
					}
					s << std::endl;
				}
				if(right != NULL){
					s << indent << IND_STEP << "\"right\": " << right->to_json(indent+IND_STEP) << std::endl;
				}
				s << indent << "}";
				return(s.str());
			}
	};

	class MondrianTree: public Partition {
		public:
			MondrianNode* root;
			std::vector<MondrianNode*> leaves;
			int ndim;

			// With `raw`, the tree is the Mondrian process of the theory: the root also draws its split
			// time Exp(mu(H)) against the lifetime, and the cut axis is chosen with probability
			// proportional to its side. Without it (the default), the root always splits and the axis is
			// uniform among the dimensions.
			MondrianTree(std::vector<std::vector<float>> *coords, float lambda, std::vector<std::vector<float>> bbox, int seed=206936, bool raw=false){
				ndim = (int) bbox.size();
				std::mt19937 my_rand(seed);
				std::uniform_int_distribution<int> uni_int(0, ndim-1);
				std::uniform_real_distribution<float> uni_float(0, 1);
				std::exponential_distribution<float> exp_float;
				std::queue<MondrianNode*> bft;

				// assign root node
				MondrianNode* cur_node = new MondrianNode(bbox, 0, 0);
				MondrianNode* aux_node;
				std::vector<std::vector<float>> cur_bbox, aux_bbox;
				float aux_value;

				root = cur_node;

				if(raw){
					exp_float.param(std::exponential_distribution<float>::param_type(bbox_sum_interval(bbox)));
					root->tau = exp_float(my_rand);
				}

				if(lambda>0 && root->tau < lambda){
					// there is lifetime so it COULD be splitted
					bft.push(cur_node);
				}

				// when there are nodes to split
				while(!bft.empty()){
					// get the node
					cur_node = bft.front();
					bft.pop();

					// if node should be splitted, then do it
					if (cur_node->tau < lambda){
						// get the bounding box for the node
						cur_bbox = cur_node->bbox;
						// select the component (axis) to make the cut
						if(raw){
							// side lengths as weights: the first axis whose cumulative side exceeds u*mu(H)
							float u = uni_float(my_rand) * bbox_sum_interval(cur_bbox), acc = 0;
							cur_node->axis = ndim-1;
							for(int a=0; a<ndim; a++){
								acc += cur_bbox[a][1] - cur_bbox[a][0];
								if(u < acc){ cur_node->axis = a; break; }
							}
						}else{
							cur_node->axis = uni_int(my_rand);
						}
						// the cut will MIN + (MAX-MIN) * random(0,1)
						cur_node->cut = cur_bbox[cur_node->axis][0] + (cur_bbox[cur_node->axis][1] - cur_bbox[cur_node->axis][0]) * uni_float(my_rand); // CHANGE FOR RANDOM_UNI_FLOAT(0,1)

						// LOWER THAN CHILD
						// bbox for child (just changes the upper limit for selected axis)
						aux_bbox = cur_bbox;
						aux_bbox[cur_node->axis][1] = cur_node->cut;
						// get interval for bbox
						aux_value = bbox_sum_interval(aux_bbox);
						// set the lambda parameter for exponential distribution
						exp_float.param(std::exponential_distribution<float>::param_type(aux_value));
						// get the tau for the child
						aux_value = cur_node->tau + exp_float(my_rand);
						// create the new child
						aux_node = new MondrianNode(aux_bbox, aux_value, cur_node->height+1);
						// set a left child
						cur_node->left = aux_node;
						// if it must be splitted, put it in the queue
						if(aux_value < lambda){
							bft.push(aux_node);
						}

						// GREATER THAN CHILD
						// bbox for child (just changes the lower limit for selected axis)
						aux_bbox = cur_bbox;
						aux_bbox[cur_node->axis][0] = cur_node->cut;
						// get interval for bbox
						aux_value = bbox_sum_interval(aux_bbox);
						// set the lambda parameter for exponential distribution
						exp_float.param(std::exponential_distribution<float>::param_type(aux_value));
						// get the tau for the child
						aux_value = cur_node->tau + exp_float(my_rand);
						// create the new child
						aux_node = new MondrianNode(aux_bbox, aux_value, cur_node->height+1);
						// set a right child
						cur_node->right = aux_node;
						// if it must be splitted, put it in the queue

						if(aux_value < lambda){
							bft.push(aux_node);
						}
					}
				}

				// put the root in the queue (it should be empty)
				bft.push(root);
				// visit all nodes
				while(!bft.empty()){
					// get the node
					cur_node = bft.front();
					bft.pop();

					if((cur_node->left==NULL) && (cur_node->right==NULL)){
						cur_node->leaf_id = (int) this->leaves.size();
						this->leaves.push_back(cur_node);
						this->samples_by_leaf.push_back({});
						this->leaf_params.push_back({});
					}else{
						bft.push(cur_node->left);
						bft.push(cur_node->right);
					}
				}

				// assign samples to leafs and inverse too
				int aux;
				for(size_t i=0; i<coords->size(); i++){
					aux = search_leaf(coords->at(i));
					this->samples_by_leaf.at(aux).push_back(static_cast<int>(i));
					this->leaf_for_sample.push_back(aux);
				}
  			}

         MondrianTree(){}

			~MondrianTree(){
				if(this->root != NULL){
					delete(this->root);
				}
				std::vector<MondrianNode*>().swap(this->leaves);
		   	std::vector<int>().swap(this->leaf_for_sample);
			   std::vector<std::vector<int>>().swap(this->samples_by_leaf);
			   std::vector<std::vector<float>>().swap(this->leaf_params);
			}

  		int n_leaves(){
  			return(static_cast<int>(this->leaves.size()));
  		}

  		int search_leaf(std::vector<float> point){
  			auto bbox = root->bbox;
  			for(size_t i=0; i<point.size(); i++){
  				if((point.at(i) < bbox.at(i).at(0)) || (bbox.at(i).at(1) < point.at(i))){
  					return(-1);
  				}
  			}
  			return(root->search_leaf(point));
  		}

  		std::string to_json(){
  			return(root->to_json(""));
  		}
	};

	// Ensemble over Mondrian partitions: draws `forest_size` Mondrian trees of lifetime `lambda`
	// on `bbox`. Without a decoder it only exposes the partitions (get_partitions...).
	class ESI: public Ensemble {
		public:
			ESI(std::vector<std::vector<float>> _coords,
			    std::vector<float> _values,
			    float lambda,
			    int forest_size,
			    std::vector<std::vector<float>> bbox,
			    std::function<int(std::string)> visitor,
			    int seed=206936,
			    bool raw=false):
			Ensemble(_coords, _values, visitor, seed){
				this->class_name = __func__;
				std::uniform_int_distribution<int> uni_int;

				for(int i=0; i<forest_size; i++){
					forest.push_back(new sptlz::MondrianTree(&coords, lambda, bbox, uni_int(my_rand), raw));
				}
			}

			MondrianTree *get_tree(int i){
				return(static_cast<MondrianTree*>(this->forest.at(i)));
			}

			std::vector<std::vector<std::vector<float>>> get_partitions(){
				std::vector<std::vector<std::vector<float>>> partitions;
				size_t n = forest.size();

				for(size_t i=0; i<n; i++){

					std::vector<std::vector<float>> this_partition;
					// get leaves for i-th tree
					for(auto leaf: get_tree(static_cast<int>(i))->leaves){
						std::vector<float> this_leaf;
						this_leaf.push_back(static_cast<float>(leaf->leaf_id));
						for(auto ax_lims: leaf->bbox){
							this_leaf.push_back(ax_lims[0]);
							this_leaf.push_back(ax_lims[1]);
						}
						this_partition.push_back(this_leaf);
					}
					partitions.push_back(this_partition);
				}
				return(partitions);
			}
	};
}

#endif

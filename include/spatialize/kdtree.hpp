#ifndef _SPTLZ_KDTREE_
#define _SPTLZ_KDTREE_

#include <stdexcept>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <string>
#include <utility>

namespace sptlz{
	template <class T>
	T __distance(T *p1, T *p2, int n, T p){
		T c=0;
		for(int i=0;i<n;i++){
			c += std::pow(p1[i]-p2[i], p);
		}
		return(std::pow(c,1/p));
	}

	template <class T>
	class Leaf{
		protected:
			int idx_point, idx_division;
			T min, med, max;
			Leaf<T> *left, *right;
			int most_probable(T* point, T radius){
				T l_len = std::min(point[this->idx_division]+radius, med) - std::max(point[this->idx_division]-radius, this->min);
				T r_len = std::min(point[this->idx_division]+radius, max) - std::max(point[this->idx_division]-radius, this->med);
				if (l_len>r_len){
					return(1); // binary 01
				}else{
					return(2); // binary 10
				}
			}

		public:
			Leaf<T>(){}

			Leaf<T>(int _idx_point, int _idx_division, T _min, T _med, T _max, Leaf<T> *_left, Leaf<T> *_right){
				this->idx_point = _idx_point;
				this->idx_division = _idx_division;
				this->min = _min;
				this->med = _med;
				this->max = _max;
				this->left = _left;
				this->right = _right;
			}

			~Leaf<T>(){
				if(this->left != NULL){
					delete(this->left);
				}
				if(this->right != NULL){
					delete(this->right);
				}
			}

			static Leaf<T> *divide_space(T *coords, int n_coords, int n_dims, int *idxs, int n_idxs, int i_var){
				/* border cases */
				if (n_idxs==0){
					return(NULL);
				}
				if (n_idxs==1){
					return(new Leaf<T>(idxs[0], i_var, coords[idxs[0]*n_dims+i_var], coords[idxs[0]*n_dims+i_var], coords[idxs[0]*n_dims+i_var], NULL, NULL));
				}

				/* general case */
				// sort the idxs by its value in coords
				std::sort(idxs, idxs+n_idxs, [coords, n_dims, i_var](int i, int j){
					return(coords[i*n_dims+i_var] < coords[j*n_dims+i_var]);
				});
				int m = (int)n_idxs/2;

				return(new Leaf<T>(idxs[m], i_var, coords[idxs[0]*n_dims+i_var], coords[idxs[m]*n_dims+i_var], coords[idxs[n_idxs-1]*n_dims+i_var],
					Leaf<T>::divide_space(coords, n_coords, n_dims, idxs, m, (i_var+1)%n_dims),
					Leaf<T>::divide_space(coords, n_coords, n_dims, idxs+m+1, n_idxs-m-1, (i_var+1)%n_dims)
				));
			}

			std::vector<int> query_ball(T *coords, int n_coords, int n_dims, T *point, T radius, T p){
				std::vector<int> result;

				// this leaf point is inside the ball
				if (__distance<T>(coords+(this->idx_point*n_dims), point, n_dims, p) < radius){
					result.push_back(this->idx_point);
				}

				/* recursive calls */
				// left leaf is touched by the ball so visit it
				if ((this->left != NULL) && (point[this->idx_division]-radius <= this->med) && (point[this->idx_division]+radius >= this->min)){
					auto lresult = this->left->query_ball(coords, n_coords, n_dims, point, radius, p);
					std::copy(lresult.begin(), lresult.end(), std::back_inserter(result));
				}
				// right leaf is touched by the ball so visit it
				if ((this->right != NULL) && (point[this->idx_division]-radius <= this->max) && (point[this->idx_division]+radius >= this->med)){
					auto rresult = this->right->query_ball(coords, n_coords, n_dims, point, radius, p);
					std::copy(rresult.begin(), rresult.end(), std::back_inserter(result));
				}

				return(result);
			}

			std::vector<std::pair<T,int>> query_nn(T *coords, int n_coords, int n_dims, T *point, int k, T p, T max_distance){
				std::vector<std::pair<T,int>> result;
				// this leaf point is closer than max_distance, so it could be a candidate
				T this_dist = __distance<T>(coords+(this->idx_point*n_dims), point, n_dims, p);
				if (this_dist < max_distance){
					result.push_back(std::make_pair(this_dist, this->idx_point));
				}

				int prob = this->most_probable(point, max_distance);
				/* recursive calls:
				if left is more probable, visit the first and the second if, il right is more probable visit the second and the third  */

				// left leaf is touched by the max_distance ball so visit it
				if ((prob==1) && (this->left != NULL) && (point[this->idx_division]-max_distance <= this->med) && (point[this->idx_division]+max_distance >= this->min)){
					auto lresult = this->left->query_nn(coords, n_coords, n_dims, point, k, p, max_distance);
					std::copy(lresult.begin(), lresult.end(), std::back_inserter(result));
					if ((int)result.size()>=k){
						std::sort(result.begin(), result.end(), [](std::pair<T,int> p1, std::pair<T,int> p2){
							return(p1.first < p2.first);
						});
						for(size_t i=k;i<result.size();i++){
							result.pop_back();
						}
						max_distance = result[k-1].first;
					}
				}

				// right leaf is touched by the ball so visit it
				if ((this->right != NULL) && (point[this->idx_division]-max_distance <= this->max) && (point[this->idx_division]+max_distance >= this->med)){
					auto rresult = this->right->query_nn(coords, n_coords, n_dims, point, k, p, max_distance);
					std::copy(rresult.begin(), rresult.end(), std::back_inserter(result));
					if ((int)result.size()>=k){
						std::sort(result.begin(), result.end(), [](std::pair<T,int> p1, std::pair<T,int> p2){
							return(p1.first < p2.first);
						});
						for(size_t i=k;i<result.size();i++){
							result.pop_back();
						}
						max_distance = result[k-1].first;
					}
				}

				// left leaf is touched by the max_distance ball so visit it
				if ((prob==2) && (this->left != NULL) && (point[this->idx_division]-max_distance <= this->med) && (point[this->idx_division]+max_distance >= this->min)){
					auto lresult = this->left->query_nn(coords, n_coords, n_dims, point, k, p, max_distance);
					std::copy(lresult.begin(), lresult.end(), std::back_inserter(result));
					if ((int)result.size()>=k){
						std::sort(result.begin(), result.end(), [](std::pair<T,int> p1, std::pair<T,int> p2){
							return(p1.first < p2.first);
						});
						for(int i=(int)result.size();i>k;i--){
							result.pop_back();
						}
						max_distance = result[k-1].first;
					}
				}

				return(result);
			}

			std::string pprint(){
				std::string s = "(";
				s += std::to_string(this->idx_point);
				if (this->left==NULL){
					s += "()";
				}else{
					s += this->left->pprint();
				}
				if (this->right==NULL){
					s += "()";
				}else{
					s += this->right->pprint();
				}
				s += ")";
				return(s);
			}
	};

	template <class T>
	class KDTree{
		protected:
			T *coords;
			int n_coords, n_dims;
			Leaf<T> *root;

		public:
			KDTree(){
				this->coords = NULL;
				this->n_coords = 0;
				this->n_dims = 0;
				this->root = NULL;
			}

			KDTree(std::vector<std::vector<T>> *_coords){
				this->n_coords = static_cast<int>(_coords->size());
				this->n_dims = static_cast<int>(_coords->at(0).size());
				this->coords = (T *)malloc(this->n_coords*this->n_dims*sizeof(T));
				int *idxs = (int *)malloc(this->n_coords*sizeof(int));

				int pos = 0;
				for(int i=0; i<this->n_coords; i++){
					idxs[i] = i;
					for(int j=0; j<this->n_dims; j++){
						this->coords[pos] = _coords->at(i).at(j);
						pos++;
					}
				}

				this->root = Leaf<T>::divide_space(this->coords, this->n_coords, this->n_dims, idxs, this->n_coords, 0);
				free(idxs);
			}

			~KDTree(){
				if(this->root != NULL){
					delete(this->root);
					free(this->coords);
				}
			}

			int size(){
				return(this->n_coords);
			}

			T* get_coords(){
				return(this->coords);
			}

			std::pair<std::vector<T>, std::vector<int>> query_ball(std::vector<T> *point, T radius, T p){
				if(this->n_coords==0){
					return(std::make_pair(std::vector<T>({}), std::vector<int>({})));
				}
				if ((int)point->size()!=this->n_dims){
					throw std::runtime_error("<kdtree>[KDTree.query_ball] query point and kdtree coords have different dimensions.");
				}
				if (this->root == NULL){
					return(std::make_pair(std::vector<T>(),std::vector<int>()));
				}

				std::vector<int> idxs;
				std::vector<T> distances;
				std::vector<std::pair<T, int>> dists;

				for(auto r: this->root->query_ball(this->coords, this->n_coords, this->n_dims, point->data(), radius, p)){
					auto tup = std::make_pair(__distance<T>(this->coords+r*this->n_dims, point->data(), this->n_dims, p), r);
					dists.push_back(tup);
				}

				std::sort(dists.begin(), dists.end(), [](std::pair<T, int> p1, std::pair<T, int> p2){
					return(p1.first < p2.first);
				});

				for(auto dist: dists){
					idxs.push_back(dist.second);
					distances.push_back(dist.first);
			  	}

				return(std::make_pair(distances, idxs));
			}

			std::pair<std::vector<T>, std::vector<int>> query_nn(std::vector<T> *point, int k, T p){
				if(this->n_coords==0){
					return(std::make_pair(std::vector<T>({}), std::vector<int>({})));
				}
				if ((int)point->size()!=this->n_dims){
					throw std::runtime_error("<kdtree>[KDTree.query_nn] query point and kdtree coords have different dimensions.");
				}
				if (this->root == NULL){
					return(std::make_pair(std::vector<T>(),std::vector<int>()));
				}

				std::vector<int> idxs;
				std::vector<T> distances;
				std::vector<std::pair<T, int>> dists;

				// when there are less elements than k just return them all
				if (this->n_coords<=k){
					for(int r=0;r<n_coords;r++){
						auto tup = std::make_pair(__distance<T>(this->coords+r*this->n_dims, point->data(), this->n_dims, p), r);
						dists.push_back(tup);
					}
				}else{
					dists = this->root->query_nn(this->coords, this->n_coords, this->n_dims, point->data(), k, p, std::numeric_limits<T>::max());
				}


				for(auto dist: dists){
					idxs.push_back(dist.second);
					distances.push_back(dist.first);
			  	}

				return(std::make_pair(distances, idxs));
			}

			std::string pprint(){
				return(this->root->pprint());
			}
	};
}

#endif

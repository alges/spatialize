#ifndef _SPTLZ_GRAD_DESC_
#define _SPTLZ_GRAD_DESC_

#include <cstdlib>
#include <limits>
#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <cmath>
#include <ctime>
#include <algorithm>
#include "spatialize/utils.hpp"

namespace sptlz{
  // Optimization constants (shared with adaptive_esi_idw.hpp via utils)
  constexpr float OPT_EPSILON = 1e-10f;  // Numerical stability for optimization

  std::vector<std::vector<int>> neighbors(std::vector<int> elements, int d){
    std::vector<std::vector<int>> nbs, nb;

    if(d==1){
      for(int e: elements){
        nbs.push_back({e});
      }
    }else{
      std::vector<int> aux;
      nb = neighbors(elements, d-1);
      for(auto _n:nb){
        for(int e:elements){
          aux = std::vector<int>(_n.begin(), _n.end());
          aux.push_back(e);
          nbs.push_back(aux);
        }
      }
    }
    return(nbs);
  }

  std::vector<float> get_random_tuple(std::vector<std::vector<float>> *ranges){
      int j;
      std::vector<float> result;
      float r;
      
      for(j=0; j<ranges->size(); j++){
        r = 1.0*std::rand()/RAND_MAX;
        result.push_back(ranges->at(j).at(0)+r*(ranges->at(j).at(1)-ranges->at(j).at(0)));
      }
      
      return(result);
  }

  std::vector<float> get_random_tuple_in_orthant(std::vector<std::vector<float>> *ranges, std::vector<int> *orthant){
      int j;
      float r, aux;
      std::vector<float> result;


      for(j=0; j<ranges->size(); j++){
          aux = 0.5*ranges->at(j).at(0) + 0.5*ranges->at(j).at(1);
          r = 1.0*std::rand()/RAND_MAX;
          if(orthant->at(j)==0){ //first half
            aux = ranges->at(j).at(0)+r*(aux-ranges->at(j).at(0));
          }else{ // second half
            aux = aux+r*(ranges->at(j).at(1)-aux);
          }
          result.push_back(aux);
      }

      return(result);
  }

  std::vector<std::vector<float>> get_random_tuples(std::vector<std::vector<float>> *ranges, int n){
    std::vector<std::vector<float>> starts;
    std::vector<float> start;
    int i,j;

    for(i=0; i<n; i++){
      starts.push_back(get_random_tuple(ranges));
    }

    return(starts);
  }

  std::vector<std::vector<float>> get_random_tuples_by_orthant(std::vector<std::vector<float>> *ranges, int n){
      int i;
      auto orths = neighbors({0,1}, ranges->size());
      std::vector<std::vector<float>> results;
      std::vector<float> result;


      for(auto orth: orths){
          for(i=0; i<n; i++){
              result = get_random_tuple_in_orthant(ranges, &orth);
          }
          results.push_back(result);
      }

      return(results);
  }

  void add(float alpha, std::vector<float> *v1, float beta, std::vector<float> *v2){
    for(int i=0; i<v1->size(); i++){
      v1->at(i) = alpha*v1->at(i) + beta*v2->at(i);
    }
  }

  void mult(std::vector<float> *v1, std::vector<float> *v2){
    for(int i=0; i<v1->size(); i++){
      v1->at(i) = v1->at(i)*v2->at(i);
    }
  }

  float dot(std::vector<float> *v1, std::vector<float> *v2){
    float v = 0.0;
    for(int i=0; i<v1->size(); i++){
      v += v1->at(i)*v2->at(i);
    }
    return(v);
  }

  class LOOND {
    protected:
        float h;
        std::vector<std::vector<float>> *coords;
        std::vector<float> *values;
        std::vector<float> centroid;

    public:
      LOOND(std::vector<std::vector<float>> *_coords, std::vector<float> *_values, float _h){
        this->h = _h;
        this->values = _values;
        this->centroid = get_centroid(_coords);
        this->coords = _coords;
      }

      virtual float eval(std::vector<float> *X){
        throw std::runtime_error("must override");
      }

      std::vector<float> gradient(std::vector<float> *X){
        float v;
        std::vector<float> gr, aux;

        for(int i=0; i<X->size(); i++){
          aux = std::vector<float>(X->begin(), X->end());
          aux.at(i)  = aux.at(i)+h;
          v = this->eval(&aux);
          aux.at(i) = aux.at(i)-2*h;
          v = v-this->eval(&aux);
          gr.push_back(0.5*v/h);
        }

        return(gr);
      }
  };

  class LOO_2D: public LOOND {
    public:
      LOO_2D(std::vector<std::vector<float>> *_coords, std::vector<float> *_values, float _h): LOOND(_coords, _values, _h){}

      float eval(std::vector<float> *X){
          int n = values->size();

          float r = 0.0;
          std::vector<float> params = {X->at(1), X->at(2)};

          auto tr_coords = transform(this->coords, &params, &(this->centroid));

          for(int i=0; i<n; i++){
            auto ds = distances(&tr_coords, i);
            float sum_w = 0.0;
            float est = 0.0;
            float wj;
            for(int j=0;j<n;j++){
              if(j!=i){
                wj = 1.0f/(OPT_EPSILON + std::pow(ds.at(j), X->at(0)));
                sum_w += wj;
                est += wj*values->at(j);
              }
            }
            r += std::abs(values->at(i)-est/sum_w);
          }

          return(r);
        }
  };

  class LOO_3D: public LOOND{
    public:
      LOO_3D(std::vector<std::vector<float>> *_coords, std::vector<float> *_values, float _h): LOOND(_coords, _values, _h){}

      float eval(std::vector<float> *X){
          int n = values->size();

          float r = 0.0;
          std::vector<float> params = {X->at(1), X->at(2), X->at(3), X->at(4), X->at(5)};

          auto tr_coords = transform(this->coords, &params, &(this->centroid));

          for(int i=0; i<n; i++){
            auto ds = distances(&tr_coords, i);
            float sum_w = 0.0;
            float est = 0.0;
            float wj;
            for(int j=0;j<n;j++){
              if(j!=i){
                wj = 1.0f/(OPT_EPSILON + std::pow(ds.at(j), X->at(0)));
                sum_w += wj;
                est += wj*values->at(j);
              }
            }
            r += std::abs(values->at(i)-est/sum_w);
          }

          return(r);
        }
  };

  class GradDesc{
    protected:
      LOOND *func;
      std::vector<std::vector<float>> rs;

      void fix(std::vector<float> *c){
        if (c->size() == 3){
          for(int i=0; i<c->size(); i++){
            if(i==1){
              if(c->at(i) < this->rs.at(i).at(0)){
                c->at(i) = fmod(90.0-c->at(i), 180.0)-90.0;
              }
              if(this->rs.at(i).at(1) < c->at(i)){
                c->at(i) = fmod(90.0+c->at(i), 180.0)-90.0;
              }
            }else{
              if(c->at(i) < this->rs.at(i).at(0)){
                c->at(i) = this->rs.at(i).at(0);
              }
              if(this->rs.at(i).at(1) < c->at(i)){
                c->at(i) = this->rs.at(i).at(1);
              }
            }
          }
        }else{
          for(int i=0; i<c->size(); i++){
            if((i==1) || (i==2) || (i==3)){
              if(c->at(i) < this->rs.at(i).at(0)){
                c->at(i) = fmod(90.0-c->at(i), 180.0)-90.0;
              }
              if(this->rs.at(i).at(1) < c->at(i)){
                c->at(i) = fmod(90.0+c->at(i), 180.0)-90.0;
              }
            }else{
              if(c->at(i) < this->rs.at(i).at(0)){
                c->at(i) = this->rs.at(i).at(0);
              }
              if(this->rs.at(i).at(1) < c->at(i)){
                c->at(i) = this->rs.at(i).at(1);
              }
            }
          }
        }
      }

      virtual std::vector<float> step(std::vector<float> *x){
        std::vector<float> r;
        return(r);
      }

      virtual void reset(){}

    public:
      GradDesc(LOOND *fn, std::vector<std::vector<float>> ranges){
        this->func = fn;
        this->rs = ranges;
      }

      virtual std::pair<std::vector<float>, int> get_min(std::vector<float> start, int max_it=100, float eps=1e-10){
        int i=0;
        this->reset();
        std::vector<float> x_min=start, curr=start, st;
        float prev = this->func->eval(&curr), new_v;

        while(true){
          st = this->step(&curr);
          add(1.0, &curr, 1.0, &st);
          this->fix(&curr);
          new_v = this->func->eval(&curr);
          if((i>=max_it) || (std::abs(prev-new_v)<eps)){
            if(prev < new_v){
              curr = x_min;
            }
            break;
          }
          prev = new_v;
          x_min = curr;
          i++;
        }

        return(std::make_pair(curr, i));
      }

      float eval_fn(std::vector<float> *x){
        return(func->eval(x));
      }
  };

  class Adagrad: public GradDesc{
    protected:
      float a, e, s;

      std::vector<float> step(std::vector<float> *x){
        std::vector<float> r;

        std::vector<float> grad = this->func->gradient(x);
        this->s = this->s + dot(&grad, &grad);
        add(-this->a/sqrt(this->s+this->e), &grad, 0.0, &grad);
        return(grad);
      }

      void reset(){
        this->s = 0.0;
      }

    public:
      Adagrad(LOOND *fn, std::vector<std::vector<float>> ranges, float alpha, float eps): GradDesc(fn, ranges){
        this->a = alpha;
        this->e = eps;
        this->s = 0.0;
      }
  };

  class RMSprop: public GradDesc{
    protected:
      float a, b, e, s;

      std::vector<float> step(std::vector<float> *x){
        std::vector<float> r;

        std::vector<float> grad = this->func->gradient(x);
        this->s = this->b*this->s + (1-this->b)*dot(&grad, &grad);

        add(-this->a/sqrt(this->s+this->e), &grad, 0.0, &grad);
        return(grad);
      }

      void reset(){
        this->s = 0.0;
      }

    public:
      RMSprop(LOOND *fn, std::vector<std::vector<float>> ranges, float alpha, float beta, float eps): GradDesc(fn, ranges){
        this->a = alpha;
        this->b = beta;
        this->e = eps;
        this->s = 0.0;
      }
  };

  class Adam: public GradDesc{
    protected:
      float a, b1, b1t, b2, b2t, e, s;
      std::vector<float> v;

      std::vector<float> step(std::vector<float> *x){
        std::vector<float> r;

        std::vector<float> grad = this->func->gradient(x);
        add(this->b1, &(this->v), (1-this->b1), &grad);
        this->s = this->b2*this->s + (1-this->b2)*dot(&grad, &grad);
        this->b1t *= this->b1;
        this->b2t *= this->b2;
        add(0.0, &grad, -std::sqrt(this->a*(1-this->b2t)+this->e)/(1-this->b1t)/s, &(this->v));
        return(grad);
      }

      void reset(){
        this->b1t = 1.0;
        this->b2t = 1.0;
        this->s = 0.0;
        for(int i=0; i<this->rs.size(); i++){
          this->v.at(i) = 0.0;
        }
      }

    public:
      Adam(LOOND *fn, std::vector<std::vector<float>> ranges, float alpha, float beta1, float beta2, float eps): GradDesc(fn, ranges){
        this->a = alpha;
        this->b1 = beta1;
        this->b1t = 1.0;
        this->b2 = beta2;
        this->b2t = 1.0;
        this->e = eps;
        this->s = 0.0;
        for(int i=0; i<this->rs.size(); i++){
          this->v.push_back(0.0);
        }
      }
  };

  class Adamax: public GradDesc{
    protected:
      float a, b1, b1t, b2, b2t, e, s;
      std::vector<float> v;

      std::vector<float> step(std::vector<float> *x){
        std::vector<float> r;

        std::vector<float> grad = this->func->gradient(x);
        this->b1t *= this->b1;
        add(this->b1, &(this->v), (1-this->b1), &grad);
        this->s = std::max(this->b2*this->s, std::sqrt(dot(&grad, &grad)));
        add(0.0, &grad, -this->a/(1-this->b1t)/s, &(this->v));
        return(grad);
      }

      void reset(){
        this->b1t = 1.0;
        this->s = 0.0;
        for(int i=0; i<this->rs.size(); i++){
          this->v.at(i) = 0.0;
        }
      }

    public:
      Adamax(LOOND *fn, std::vector<std::vector<float>> ranges, float alpha, float beta1, float beta2): GradDesc(fn, ranges){
        this->a = alpha;
        this->b1 = beta1;
        this->b1t = 1.0;
        this->b2 = beta2;
        this->s = 0.0;
        for(int i=0; i<this->rs.size(); i++){
          this->v.push_back(0.0);
        }
      }
  };

  class AMSGrad: public GradDesc{
    protected:
      float a, b1, b2, e, s;
      std::vector<float> v;

      std::vector<float> step(std::vector<float> *x){
        float aux = this->s;
        std::vector<float> r;
        std::vector<float> grad = this->func->gradient(x);
        add(this->b1, &(this->v), (1-this->b1), &grad);
        this->s = this->b2*this->s + (1-this->b2)*dot(&grad, &grad);
        this->s = std::max(aux, this->s);
        add(0.0, &grad, -this->a/std::sqrt(this->s+this->e), &(this->v));
        return(grad);
      }

      void reset(){
        this->s = 0.0;
        for(int i=0; i<this->rs.size(); i++){
          this->v.at(i) = 0.0;
        }
      }

    public:
      AMSGrad(LOOND *fn, std::vector<std::vector<float>> ranges, float alpha, float beta1, float beta2, float eps): GradDesc(fn, ranges){
        this->a = alpha;
        this->b1 = beta1;
        this->b2 = beta2;
        this->e = eps;
        this->s = 0.0;
        for(int i=0; i<this->rs.size(); i++){
          this->v.push_back(0.0);
        }
      }
  };

  class FLPNBDescent: public GradDesc{
    protected:
      std::vector<float> st;
      std::vector<int> n;
      std::vector<std::vector<int>> nbs;

    float eval(std::vector<int> idxs){
      std::vector<float> vec;

      for(int j=0; j<this->st.size(); j++){
        vec.push_back(this->rs.at(j).at(0) + idxs.at(j)*this->st.at(j));
      }

      return(this->func->eval(&vec));
    }

    std::pair<std::vector<float>, int> get_min(std::vector<float> start, int max_it=100, float eps=1e-10){
      int i;
      float prev, min_val, aux;
      std::vector<int> curr, new_curr;
      std::vector<int> min_coord;
      std::vector<float> result;
      for(int i=0; i<start.size(); i++){
        curr.push_back(0.5+(start.at(i)-this->rs.at(i).at(0))/this->st.at(i));
      } 

      prev = this->eval(curr);
      for(int j=0; j<curr.size(); j++){
        new_curr.push_back(0);
      }

      for(i=0; i<max_it; i++){
        min_val = std::numeric_limits<float>::max();
        min_coord = {};
        for(auto nb: this->nbs){
          for(int j=0; j<curr.size(); j++){
            new_curr.at(j) = curr.at(j)+nb.at(j);
            if (new_curr.at(j)<0){
              new_curr.at(j) = (int)std::ceil(new_curr.at(j)/this->n.at(j))*this->n.at(j) - new_curr.at(j);
            }
            if (this->n.at(j)<new_curr.at(j)){
              new_curr.at(j) = new_curr.at(j) % this->n.at(j);
            }
          }

          aux = this->eval(new_curr);
          if (aux<min_val){
            min_val = aux;
            min_coord = new_curr;
          }
        }

        if (min_val<prev){
          if (std::abs(min_val-prev)<eps){
            curr = min_coord;
            prev = min_val;
            break;
          }
          curr = min_coord;
          prev = min_val;
        }
      }

      for(int j=0; j<curr.size(); j++){
        result.push_back(this->rs.at(j).at(0) + curr.at(j)*this->st.at(j));
      }

      return(std::make_pair(result, i));    
    }

    public:
      FLPNBDescent(LOOND *fn, std::vector<std::vector<float>> ranges, std::vector<float> steps, std::vector<int> ns): GradDesc(fn, ranges){
        this->st = steps;
        this->n = ns;
        this->nbs = neighbors({-1,0,1}, (int)steps.size());
      }
  };

  class GridNBRndDesc: public GradDesc{
    protected:
      int k;
      std::vector<float> st;
      std::vector<int> n;
      std::vector<std::vector<int>> nbs;
      std::map<std::vector<int>, float> cache;
      std::mt19937 my_rand;

    float eval(std::vector<int> *idxs){
      auto search = this->cache.find(*idxs);

      if (search != this->cache.end()){
        return(search->second);
      }

      std::vector<float> vec;

      for(int j=0; j<this->st.size(); j++){
        vec.push_back(this->rs.at(j).at(0) + idxs->at(j)*this->st.at(j));
      }

      float value = this->func->eval(&vec);
      this->cache[*idxs] = value;

      return(value);
    }

    std::pair<std::vector<float>, int> get_min(std::vector<float> start, int max_it=100, float eps=1e-10){
      int i, j, k, count;
      float prev, min_val, aux;
      std::vector<int> curr, new_curr, min_coord, nb;
      std::vector<float> result;
      for(int i=0; i<start.size(); i++){
        curr.push_back(0.5+(start.at(i)-this->rs.at(i).at(0))/this->st.at(i));
      } 

      prev = this->eval(&curr);
      for(int j=0; j<curr.size(); j++){
        new_curr.push_back(0);
      }

      for(i=0; i<max_it; i++){
        min_val = std::numeric_limits<float>::max();
        min_coord = {};
        count = 0;
        std::shuffle(this->nbs.begin(), this->nbs.end(), this->my_rand);
        for(j=0;j<this->nbs.size(); j++){
          if ((this->k < j) && (0<min_coord.size())){ // try k directions and find a minimum, I stop, otherwise keep until I find one or run out of directions
            break;
          }
          nb = this->nbs.at(j);
          for(k=0; k<curr.size(); k++){
            new_curr.at(k) = curr.at(k)+nb.at(k);
            if (new_curr.at(k)<0){
              new_curr.at(k) = (int)std::ceil(new_curr.at(k)/this->n.at(k))*this->n.at(k) - new_curr.at(k);
            }
            if (this->n.at(k)<new_curr.at(k)){
              new_curr.at(k) = new_curr.at(k) % this->n.at(k);
            }
          }

          aux = this->eval(&new_curr);
          if (aux<min_val){
            min_val = aux;
            min_coord = new_curr;
          }
        }

        if (min_val<prev){
          if (std::abs(min_val-prev)<eps){
            curr = min_coord;
            prev = min_val;
            break;
          }
          curr = min_coord;
          prev = min_val;
        }
      }

      for(int j=0; j<curr.size(); j++){
        result.push_back(this->rs.at(j).at(0) + curr.at(j)*this->st.at(j));
      }

      return(std::make_pair(result, i));    
    }

    public:
      GridNBRndDesc(LOOND *fn, std::vector<std::vector<float>> ranges, std::vector<float> steps, std::vector<int> ns, int k_nb, int seed): GradDesc(fn, ranges){
        this->st = steps;
        this->n = ns;
        this->nbs = neighbors({-1,0,1}, (int)steps.size());
        this->k = k_nb;
        this->my_rand = std::mt19937(seed);
      }

      void reset(){
        this->cache.clear();
      }
  };

  std::vector<float> get_minimum(GradDesc *opt, std::vector<std::vector<float>> *ranges, int max_it){
    std::vector<std::vector<float>> starting_points = get_random_tuples_by_orthant(ranges, 1);
    int nreps = starting_points.size();
    int i_min;
    float v_min = std::numeric_limits<float>::max();
    std::vector<float> c_min;

    for(int i=0; i<nreps; i++){
      auto result = opt->get_min(starting_points.at(i), max_it);
      auto curr_v = opt->eval_fn(&(result.first));
      if (curr_v<v_min){
        v_min = curr_v;
        c_min = result.first;
        i_min = result.second;
      }
    }

    return(c_min);
  }

/*  int main(int argc, char* argv[]){
    int i_opt = atoi(argv[1]);
    int maxit = atoi(argv[2]);
    int nreps = atoi(argv[3]);
    float alpha = atof(argv[4]);

    int t = std::time(0);
    std::srand(t);
    std::cerr << "Go!" << std::endl;
    std::vector<std::vector<float>> parmas_ranges = {
      {  0.5, 10.0},
      {-90.0, 90.0},
      {-90.0, 90.0},
      {-90.0, 90.0},
      {  0.1,  1.0},
      {  0.1,  1.0}
    }, coords_ranges = {
      {0.0, 2000.0},
      {0.0, 1500.0},
      {0.0, 500.0},
      {0.0, 100.0}
    }, alphas_ranges = {
      {0.01, 0.5}
    };
    std::vector<int> coords_idx = {0,1,2};
    std::vector<int> values_idx = {3};

    // simulated leaf data
    std::vector<std::vector<float>> pool = get_random_values(&coords_ranges, 10);
  	std::vector<std::vector<float>> coords = slice_columns(&pool, &coords_idx);
  	std::vector<float> values = as_1d_array(&pool, values_idx);

    // Gradient descent stuff
    LOOND *fn = new LOOND(&coords, &values, 0.01);

    // get the minimum in a grid, just to get an aproximation of real minimum
    std::vector<float> aux = {0, 0, 0, 0, 0, 0};
    float curr, grid_min = std::numeric_limits<float>::max();
    for(aux.at(0) = 1; aux.at(0)<10.1; aux.at(0)+=1){
      for(aux.at(1) = -90; aux.at(1)<90.1; aux.at(1)+=30){
        for(aux.at(2) = -90; aux.at(2)<90.1; aux.at(2)+=30){
          for(aux.at(3) = -90; aux.at(3)<90.1; aux.at(3)+=30){
            for(aux.at(4) = 0.1; aux.at(4)<1.1; aux.at(4)+=0.1){
              for(aux.at(5) = 0.1; aux.at(5)<1.1; aux.at(5)+=0.1){
                curr = fn->eval(&aux);
                if (curr<grid_min){
                  grid_min = curr;
                }
              }
            }
          }
        }
      }
    }
    std::cerr << "Grid Minimum : " << grid_min << std::endl;

    // initial setup
    std::vector<std::vector<std::vector<float>>> cand_coords;
    std::vector<std::vector<float>> cand_values;
    std::vector<std::vector<float>> cand_it;

    std::vector<int> i_min;
    std::vector<float> v_min;
    std::vector<std::vector<float>> c_min;

    std::vector<std::vector<float>> starting_points = get_random_values(&parmas_ranges, 100);
    std::vector<std::vector<float>> alphas = get_random_values(&alphas_ranges, 100);
    std::vector<GradDesc*> opts;

    std::vector<std::string> names = {"Adagrad", "RMSprop", "Adam", "Adamax", "AMSGrad", "FLPNBDescent"};
    int n = names.size();
    for(int j=0; j<n; j++){
      i_min.push_back(0);
      v_min.push_back(std::numeric_limits<float>::max());
      c_min.push_back({0.0});
    }

    int j= i_opt;
    // get several minimums
    // for(std::vector<float> stpt : starting_points){
    for(int i=0; i<nreps; i++){
      opts.clear();
      opts.push_back(new Adagrad(fn, parmas_ranges, alpha, 1e-3));
      opts.push_back(new RMSprop(fn, parmas_ranges, alpha, 0.1, 1e-3));
      opts.push_back(new Adam(fn, parmas_ranges, alpha, 0.1, 0.1, 1e-3));
      opts.push_back(new Adamax(fn, parmas_ranges, alpha, 0.1, 0.1));
      opts.push_back(new AMSGrad(fn, parmas_ranges, alpha, 0.1, 0.1, 1e-3));
      opts.push_back(new FLPNBDescent(fn, parmas_ranges, {0.5, 10.0, 10.0, 10.0, 0.1, 0.1}, {19, 18, 18, 18, 9, 9}));

      std::cerr << "\rrepetition : " << (i+1) << std::flush;
      //for(int j=0; j<n; j++){
        auto result = opts.at(j)->get_min(starting_points.at(i), maxit);
        auto curr_v = fn->eval(&(result.first));
        if (curr_v<v_min.at(j)){
          v_min.at(j) = curr_v;
          c_min.at(j) = result.first;
          i_min.at(j) = result.second;
        }
      //}
    }

    // show trhe minimum of the candidates
    //for(int j=0; j<n; j++){
      std::cerr << std::endl;
      std::cerr << "[data]," << names.at(j) << "," << maxit << "," << nreps << "," << alpha << "," << grid_min << "," << v_min.at(j) << std::endl;
      std::cout << "[data]," << names.at(j) << "," << maxit << "," << nreps << "," << alpha << "," << grid_min << "," << v_min.at(j) << std::endl;
    //}
  }*/
}

#endif
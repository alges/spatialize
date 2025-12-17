#ifndef _SPTLZ_UTILS_
#define _SPTLZ_UTILS_ 

#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace sptlz{
  // Mathematical constants
  constexpr float PI = 3.14159265358979323846f;

  // Inline utility functions
  inline float deg_to_rad(float degrees) { return degrees * PI / 180.0f; }
  template <class T>
  void pprint(std::vector<T> v, int nl=1){
      for(T i: v){
        std::cout << " " << i;
      }
      for(int j=0; j<nl; j++){
        std::cout << std::endl;
      }
  }

  std::vector<std::vector<float>> get_random_values(std::vector<std::vector<float>> *ranges, int n, int seed){
    std::srand(seed);
    std::vector<std::vector<float>> result;
    std::vector<float> values;
    int i,j;

    for(i=0; i<n; i++){
      values = {};
      for(j=0; j<ranges->size(); j++){
        float r = 1.0*std::rand()/RAND_MAX;
        values.push_back(ranges->at(j).at(0)+r*(ranges->at(j).at(1)-ranges->at(j).at(0)));
      }
      result.push_back(values);
    }

    return(result);
  }

  std::vector<std::vector<int>> get_full_neighboorhood(int d){
    if(d==0){
      std::vector<std::vector<int>> r;
      r.push_back({});
      return(r);
    }else{
      std::vector<std::vector<int>> result = get_full_neighboorhood(d-1);
      std::vector<int> aux;
      int n = result.size();
      for(int i=0; i<n; i++){
        aux = result.at(i);
        aux.push_back(-1);
        result.push_back(aux);
        aux = result.at(i);
        aux.push_back(0);
        result.push_back(aux);
        aux = result.at(i);
        aux.push_back(1);
        result.push_back(aux);
      }
      auto cur = result.begin();
      for(int i=0; i<n; i++){
        cur = result.erase(cur);
      }
      return(result);
    }
  }

  template <class T>
  std::vector<float> grid_search(T *func, std::vector<std::vector<float>> *ranges, std::vector<float> cur, float tol=1e-5){
    int n = ranges->size();
    for(int i=0; i<n; i++){
      if((cur.at(i)<ranges->at(i).at(0))||(ranges->at(i).at(1)<cur.at(i))){
        throw std::runtime_error("starting point outside the bounds");
      }
    }
    float minimum = func->eval(cur), value, diff;
    std::vector<float> new_pos(n), best_candidate;
    auto neighbors = get_full_neighboorhood(n);

    while(true){
      for(auto nb: neighbors){
        bool from_top=false, all_zero=true;
        // refresh new_pos
        for(int i=0; i<n; i++){
          if(nb.at(i)!=0){
            all_zero=false;
          }
          new_pos.at(i) = cur.at(i)+nb.at(i)*ranges->at(i).at(2);
          if((new_pos.at(i) < ranges->at(i).at(0)) || (ranges->at(i).at(1) < new_pos.at(i))){
            from_top = true;
            break;
          }
        }
        // if new position is in place and not all zero (no movement, same point) calculate and refresh best candidate
        if(!from_top && !all_zero){
          value = func->eval(new_pos);
          if(best_candidate.size()==0 || value<best_candidate.at(3)){
            best_candidate = {};
            for(int i=0; i<n; i++){
              best_candidate.push_back(new_pos.at(i));
            }
            best_candidate.push_back(value);
          }
        }
      }
      // difference
      if(best_candidate.size()==0){
        // no valid neighbors found
        break;
      }
      diff = minimum - best_candidate.at(n);
      if(diff>0){
        // if lower, refresh
        minimum = best_candidate.at(n);
        cur = {};
        for(int i=0; i<n; i++){
          cur.push_back(best_candidate.at(i));
        }
        // reset best_candidate for next iteration
        best_candidate = {};
        // if close enough, leave
        if(std::abs(diff)<tol){
          break;
        }
      }else{
        // not better than current, so local minimum
        break;
      }
    }

    return(cur);
  }

  template <class T>
  std::vector<T> as_1d_array(std::vector<std::vector<T>> *arr, std::vector<int> idxs){
    std::vector<T> result;

    if (arr->size()==0){
      return(result);
    }

    for(auto &record : *arr){
      for(size_t i=0;i<idxs.size();i++){
        result.push_back(record[idxs[i]]);
      }
    }

    return(result);
  }

  template <class T>
  std::vector<T> as_1d_array(std::vector<std::vector<T>> *arr){
    std::vector<int> idxs;

    if (arr->size()==0){
      std::vector<T> result;
      return(result);
    }

    for(size_t i=0;i<arr->at(0).size();i++){
      idxs.push_back(i);
    }

    return(as_1d_array(arr, idxs));
  }

  float distance(std::vector<float> *p1, std::vector<float> *p2){
    float c = 0.0;
    for(size_t i=0; i<p1->size(); i++){
      c += std::pow(p1->at(i)-p2->at(i), 2.0);
    }
    return(std::pow(c, 0.5));
  }

  std::vector<float> distances(std::vector<std::vector<float>> *coords, size_t j){
    std::vector<float> result;

    for(size_t i=0; i<coords->size(); i++){
      if(i!=j){
        result.push_back(distance(&(coords->at(j)), &(coords->at(i))));
      }else{
        result.push_back(0.0);
      }
    }
    return(result);
  }

  std::vector<float> get_centroid(std::vector<std::vector<float>> *coords){
    std::vector<float> result(coords->at(0).size());
    int n = 0;

    for(size_t i=0; i<coords->size(); i++){
      for(size_t j=0; j<result.size(); j++){
        result.at(j) = result.at(j)+coords->at(i).at(j);
      }
      n++;
    }

    for(size_t j=0; j<result.size(); j++){
      result.at(j) = result.at(j)/n;
    }

    return(result);
  }

  std::vector<std::vector<float>> transform(const float *coords, const float *params, const float *centroid, int n, int d){
    std::vector<std::vector<float>> tr_coords;

    if (d==2){
      // Rotation matrix R_φ with anisotropy factor a_f applied to x-component
      // According to paper: R_φ = [cos(φ) -sin(φ); sin(φ) cos(φ)], then multiply by [a_f; 1]
      // Result: [a_f*cos(φ) -a_f*sin(φ); sin(φ) cos(φ)]
      const float phi = deg_to_rad(params[0]);
      const float cos_phi = std::cos(phi);
      const float sin_phi = std::sin(phi);
      const float a_f = params[1];  // anisotropy factor
      float r1 = a_f*cos_phi, r2 = -a_f*sin_phi, r3 = sin_phi, r4 = cos_phi;
      for(int i=0; i<n; i++){
        tr_coords.push_back({
          r1*(coords[2*i]-centroid[0])+r2*(coords[2*i+1]-centroid[1]),
          r3*(coords[2*i]-centroid[0])+r4*(coords[2*i+1]-centroid[1])
        });
      }
    }else if (d==3){
      // 3D rotation: azimuth, dip, plunge (Euler angles in degrees)
      const float azim = deg_to_rad(params[0]);
      const float dip = deg_to_rad(params[1]);
      const float plunge = deg_to_rad(params[2]);
      float ca = std::cos(azim), sa = std::sin(azim);
      float cb = std::cos(dip), sb = std::sin(dip);
      float cc = std::cos(plunge), sc = std::sin(plunge);
      float r1 = ca*cb, r2 = ca*sb*sc-sa*cc, r3 = ca*sb*cc+sa*sc, r4 = sa*cb, r5 = sa*sb*sc+ca*cc, r6 = sa*sb*cc-ca*sc, r7 = -sb, r8 = cb*sc, r9 = cb*cc;
      r4 *= params[3]; r5 *= params[3]; r6 *= params[3];
      r7 *= params[4]; r8 *= params[4]; r9 *= params[4];
      for(int i=0; i<n; i++){
        tr_coords.push_back({
          r1*(coords[3*i]-centroid[0])+r2*(coords[3*i+1]-centroid[1])+r3*(coords[3*i+2]-centroid[2]),
          r4*(coords[3*i]-centroid[0])+r5*(coords[3*i+1]-centroid[1])+r6*(coords[3*i+2]-centroid[2]),
          r7*(coords[3*i]-centroid[0])+r8*(coords[3*i+1]-centroid[1])+r9*(coords[3*i+2]-centroid[2])
        });
      }
    }
    return(tr_coords);
  }

  std::vector<std::vector<float>> transform(std::vector<std::vector<float>> *coords, std::vector<float> *params, std::vector<float> *centroid){
    auto coords_1d = as_1d_array(coords);

    return(transform(coords_1d.data(), params->data(), centroid->data(), coords->size(), centroid->size()));
  }

  template <class T>
  std::vector<T> slice(std::vector<T> *arr, std::vector<int> *idxs){
    std::vector<T> result; 
    for(int i : *idxs){
      result.push_back(arr->at(i));
    }
    return(result);
  }

  template <class T>
  std::vector<std::vector<T>> slice_columns(std::vector<std::vector<T>> *arr, std::vector<int> *idxs){
    std::vector<std::vector<T>> result;

    for(auto rows: *arr){
      result.push_back(slice(&rows, idxs));
    }

    return(result);
  }

  template <class T>
  std::vector<T> slice_from(std::vector<T> *arr, int idx){
    std::vector<T> result;
    for(int i=idx; i<arr->size(); i++){
      result.push_back(arr->at(i));
    }
    return(result);
  }

  template <class T>
  std::vector<T> slice_to(std::vector<T> *arr, int idx){
    std::vector<T> result;
    for(int i=0; i<idx; i++){
      result.push_back(arr->at(i));
    }
    return(result);
  }

  template <class T>
  std::vector<T> slice_from_to(std::vector<T> *arr, int from, int to){
    std::vector<T> result;
    for(int i=from; i<to; i++){
      result.push_back(arr->at(i));
    }
    return(result);
  }

  template <class T>
  std::vector<T> slice_drop_idx(std::vector<T> *arr, size_t idx){
    std::vector<T> result;
    for(size_t i=0; i<arr->size(); i++){
      if(i!=idx){
        result.push_back(arr->at(i));
      }
    }
    return(result);
  }

  std::vector<int> get_folds(int n, int k, float seed){
    std::vector<int> result(n);
    std::mt19937 my_rand(seed);
    std::uniform_real_distribution<float> uni_float(0, 1);
    std::vector<std::pair<int, float>> permutation;

    for(int i=0; i<n; i++){
      permutation.push_back(std::make_pair(i,uni_float(my_rand)));
    }
    std::sort(permutation.begin(), permutation.end(), [](auto a, auto b){return(a.second<b.second);});
    for(int i=0; i<n; i++){
      result.at(i) = permutation.at(i).first*k/n;
    }

    return(result);
  }

  template <class T>
  std::pair<std::vector<T>, std::vector<T>> divide_by_predicate(std::vector<T> *arr, std::function<bool(T *)> pred){
    std::vector<T> result1;
    std::vector<T> result2;

    for(auto record: *arr){
      if(pred(&record)){
        result1.push_back(record);
      }else{
        result2.push_back(record);
      }
    }
    return(std::make_pair(result1, result2));
  }

  template <class T>
  std::pair<std::vector<int>, std::vector<int>> indexes_by_predicate(std::vector<T> *arr, std::function<bool(T *)> pred){
    std::vector<int> result1;
    std::vector<int> result2;

    for(size_t i=0; i<arr->size(); i++){
      if(pred(&(arr->at(i)))){
        result1.push_back(i);
      }else{
        result2.push_back(i);
      }
    }
    return(std::make_pair(result1, result2));
  }

  template<typename T>
  std::string get_class_name(T) {
    return typeid(T).name();
  }

  template <class T>
  std::vector<T> ndarray_to_vector_1d(py::array_t<T> *arr) {
      py::buffer_info info = arr->request();
      std::vector<T> v;

      for (py::ssize_t i = 0; i < arr->shape()[0]; i++){
          v.push_back(*(arr->data(i)));
      }
      return v;
  }

  template <class T>
  std::vector<std::vector<T>> ndarray_to_vector_2d(py::array_t<T> *arr) {
      py::buffer_info info = arr->request();
      std::vector<std::vector<T>> v;
      std::vector<T> aux;

      for (py::ssize_t i = 0; i < arr->shape()[0]; i++){
        aux.clear();
        for (py::ssize_t j = 0; j < arr->shape()[1]; j++){
          aux.push_back(*(arr->data(i,j)));
        }
        v.push_back(aux);
      }
      return v;
  }

  template <class T>
  std::vector<std::vector<std::vector<T>>> ndarray_to_vector_3d(py::array_t<T> *arr) {
      py::buffer_info info = arr->request();
      std::vector<std::vector<std::vector<T>>> r;
      std::vector<std::vector<T>> v;
      std::vector<T> aux;

      for (py::ssize_t i = 0; i < arr->shape()[0]; i++){
        for (py::ssize_t j = 0; j < arr->shape()[1]; j++){
          aux.clear();
          for (py::ssize_t k = 0; k < arr->shape()[2]; k++){
            aux.push_back(*(arr->data(i,j,k)));
          }
          v.push_back(aux);
        }
        r.push_back(v);
      }
      return(r);
  }

  template <class T>
  py::array_t<T> vector_1d_to_ndarray(std::vector<T> *arr) {
    py::ssize_t ndim = 1;
    std::vector<py::ssize_t> shape = {(py::ssize_t)arr->size()};
    std::vector<py::ssize_t> strides = {sizeof(T)};

    // return 1-D NumPy array
    return py::array(py::buffer_info(
      arr->data(),                          /* data as contiguous array  */
      sizeof(T),                              /* size of one scalar        */
      py::format_descriptor<T>::format(),     /* data type                 */
      ndim,                                   /* number of dimensions      */
      shape,                                  /* shape of the matrix       */
      strides                                 /* strides for each axis     */
    ));
  }

  template <class T>
  py::array_t<T> vector_2d_to_ndarray(std::vector<std::vector<T>> *arr, int n=0) {
    if (arr->size()==0){
      py::ssize_t ndim = 2;
      std::vector<py::ssize_t> shape = {(py::ssize_t) 0, (py::ssize_t) n};
      std::vector<py::ssize_t> strides = {sizeof(T)*shape.at(1), sizeof(T)};
      std::vector<T> arr1d;

      // return 2-D NumPy array
      return py::array(py::buffer_info(
        arr1d.data(),                          /* data as contiguous array  */
        sizeof(T),                              /* size of one scalar        */
        py::format_descriptor<T>::format(),     /* data type                 */
        ndim,                                   /* number of dimensions      */
        shape,                                  /* shape of the matrix       */
        strides                                 /* strides for each axis     */
      ));
    }else{
      py::ssize_t ndim = 2;
      std::vector<py::ssize_t> shape = {(py::ssize_t)arr->size(), (py::ssize_t)arr->at(0).size()};
      std::vector<py::ssize_t> strides = {sizeof(T)*shape.at(1), sizeof(T)};
      std::vector<T> arr1d = as_1d_array(arr);

      // return 2-D NumPy array
      return py::array(py::buffer_info(
        arr1d.data(),                          /* data as contiguous array  */
        sizeof(T),                              /* size of one scalar        */
        py::format_descriptor<T>::format(),     /* data type                 */
        ndim,                                   /* number of dimensions      */
        shape,                                  /* shape of the matrix       */
        strides                                 /* strides for each axis     */
      ));
    }
  }

  template <class T>
  py::array_t<T> vector_3d_to_ndarray(std::vector<std::vector<std::vector<T>>> *arr, int n=0, int m=0) {
    if (arr->size()==0){
      py::ssize_t ndim = 3;
      std::vector<py::ssize_t> shape = {(py::ssize_t) 0, (py::ssize_t) n, (py::ssize_t) m};
      std::vector<py::ssize_t> strides = {sizeof(T)*shape.at(1)*shape.at(2), sizeof(T)*shape.at(2), sizeof(T)};
      std::vector<T> arr1d;

      // return 2-D NumPy array
      return py::array(py::buffer_info(
        arr1d.data(),                          /* data as contiguous array  */
        sizeof(T),                              /* size of one scalar        */
        py::format_descriptor<T>::format(),     /* data type                 */
        ndim,                                   /* number of dimensions      */
        shape,                                  /* shape of the matrix       */
        strides                                 /* strides for each axis     */
      ));
    }else{
      py::ssize_t ndim = 3;
      std::vector<py::ssize_t> shape = {(py::ssize_t)arr->size(), (py::ssize_t)arr->at(0).size(), (py::ssize_t)arr->at(0).at(0).size()};
      std::vector<py::ssize_t> strides = {sizeof(T)*shape.at(1)*shape.at(2), sizeof(T)*shape.at(2), sizeof(T)};
      std::vector<std::vector<T>> arr2d = as_1d_array(arr);
      std::vector<T> arr1d = as_1d_array(&arr2d);

      // return 2-D NumPy array
      return py::array(py::buffer_info(
        arr1d.data(),                          /* data as contiguous array  */
        sizeof(T),                              /* size of one scalar        */
        py::format_descriptor<T>::format(),     /* data type                 */
        ndim,                                   /* number of dimensions      */
        shape,                                  /* shape of the matrix       */
        strides                                 /* strides for each axis     */
      ));
    }
  }
}

#endif

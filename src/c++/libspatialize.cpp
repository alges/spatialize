#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <tuple>
#include <optional>
#include "spatialize/nn_idw.hpp"
#include "spatialize/esi_idw.hpp"
#include "spatialize/esi_kriging.hpp"
#include "spatialize/voronoi_idw.hpp"
#include "spatialize/adaptive_esi_idw.hpp"
#include "spatialize/custom_esi.hpp"
#include "spatialize/custom_coesi.hpp"

namespace py = pybind11;

std::vector<std::vector<std::vector<float>>> get_partitions_using_esi(py::array_t<float> samples, int forest_size, float alpha, std::optional<py::function> visitor, int seed){
    py::buffer_info smp_info = samples.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);

    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);

    sptlz::ESI* esi = new sptlz::ESI(smp, {}, lambda, forest_size, bbox, _visitor, seed);
    auto r = esi->get_partitions();

    delete esi;

//    return(sptlz::vector_3d_to_ndarray(&r));
    return(r);
}

py::array_t<int> get_leaf_for_samples_using_esi(py::array_t<float> samples, int forest_size, float alpha, std::optional<py::function> visitor, int seed){
    py::buffer_info smp_info = samples.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);

    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);

    sptlz::ESI* esi = new sptlz::ESI(smp, {}, lambda, forest_size, bbox, _visitor, seed);
    auto r = esi->get_leaf_for_samples();

    delete esi;

    return(sptlz::vector_2d_to_ndarray(&r));
}

py::array_t<float> estimation_nn_idw(py::array_t<float> samples, py::array_t<float> values, float radius, float exp, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (val_info.ndim != 1)
        throw std::runtime_error("[2] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[3] queries must be a 2 dimensions array");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    std::vector<float> search_params = {radius, radius, radius, 0.0, 0.0, 0.0};
    sptlz::NN_IDW* myIDW = new sptlz::NN_IDW(smp, val, search_params, exp, _visitor);

    auto r = myIDW->estimate(&qry);

    delete myIDW;
    
    return(sptlz::vector_1d_to_ndarray(&r));
}

py::array_t<float> loo_nn_idw(py::array_t<float> samples, py::array_t<float> values, float radius, float exp, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (val_info.ndim != 1)
        throw std::runtime_error("[2] values must be a 1 dimension array");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    std::vector<float> search_params = {radius, radius, radius, 0.0, 0.0, 0.0};
    sptlz::NN_IDW* myIDW = new sptlz::NN_IDW(smp, val, search_params, exp, _visitor);
    
    auto r = myIDW->leave_one_out();

    delete myIDW;
    
    return(sptlz::vector_1d_to_ndarray(&r));
}

py::array_t<float> kfold_nn_idw(py::array_t<float> samples, py::array_t<float> values, float radius, float exp, int k, int seed, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (val_info.ndim != 1)
        throw std::runtime_error("[2] values must be a 1 dimension array");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    std::vector<float> search_params = {radius, radius, radius, 0.0, 0.0, 0.0};
    sptlz::NN_IDW* myIDW = new sptlz::NN_IDW(smp, val, search_params, exp, _visitor);

    auto r = myIDW->k_fold(k, seed);

    delete myIDW;
    
    return(sptlz::vector_1d_to_ndarray(&r));
}

std::tuple<py::object, py::array_t<float>> estimation_esi_idw(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, float exp, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (val_info.ndim != 1)
        throw std::runtime_error("[2] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[3] queries must be a 2 dimensions array");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);
        
    sptlz::ESI_IDW* esi = new sptlz::ESI_IDW(smp, val, lambda, forest_size, bbox, exp, _visitor, seed);
    
    auto r = esi->estimate(&qry);

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> loo_esi_idw(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, float exp, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (val_info.ndim != 1)
        throw std::runtime_error("[2] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[3] queries must be a 2 dimensions array");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);
        
    sptlz::ESI_IDW* esi = new sptlz::ESI_IDW(smp, val, lambda, forest_size, bbox, exp, _visitor, seed);
    
    auto r = esi->leave_one_out();

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> kfold_esi_idw(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, float exp, int creation_seed, int k, int folding_seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (val_info.ndim != 1)
        throw std::runtime_error("[2] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[3] queries must be a 2 dimensions array");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);
        
    sptlz::ESI_IDW* esi = new sptlz::ESI_IDW(smp, val, lambda, forest_size, bbox, exp, _visitor, creation_seed);

    auto r = esi->k_fold(k, folding_seed);

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> estimation_esi_kriging_2d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int model, float nugget, float range, float sill, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (smp_info.shape[1] != 2)
        throw std::runtime_error("[2] samples must have 2 coordinates");
    if (val_info.ndim != 1)
        throw std::runtime_error("[3] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[4] queries must be a 2 dimensions array");
    if (qry_info.shape[1] != 2)
        throw std::runtime_error("[5] queries must have 2 coordinates");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);

    sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(smp, val, lambda, forest_size, bbox, model, nugget, range, sill, _visitor, seed);

    auto r = esi->estimate(&qry);

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> loo_esi_kriging_2d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int model, float nugget, float range, float sill, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (smp_info.shape[1] != 2)
        throw std::runtime_error("[2] samples must have 2 coordinates");
    if (val_info.ndim != 1)
        throw std::runtime_error("[3] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[4] queries must be a 2 dimensions array");
    if (qry_info.shape[1] != 2)
        throw std::runtime_error("[5] queries must have 2 coordinates");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);

    sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(smp, val, lambda, forest_size, bbox, model, nugget, range, sill, _visitor, seed);

    auto r = esi->leave_one_out();

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> kfold_esi_kriging_2d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int model, float nugget, float range, float sill, int creation_seed, int k, int folding_seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (smp_info.shape[1] != 2)
        throw std::runtime_error("[2] samples must have 2 coordinates");
    if (val_info.ndim != 1)
        throw std::runtime_error("[3] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[4] queries must be a 2 dimensions array");
    if (qry_info.shape[1] != 2)
        throw std::runtime_error("[5] queries must have 2 coordinates");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);

    sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(smp, val, lambda, forest_size, bbox, model, nugget, range, sill, _visitor, creation_seed);

    auto r = esi->k_fold(k, folding_seed);

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> estimation_esi_kriging_3d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int model, float nugget, float range, float sill, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (smp_info.shape[1] != 3)
        throw std::runtime_error("[2] samples must have 3 coordinates");
    if (val_info.ndim != 1)
        throw std::runtime_error("[3] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[4] queries must be a 2 dimensions array");
    if (qry_info.shape[1] != 2)
        throw std::runtime_error("[5] queries must have 2 coordinates");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);

    sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(smp, val, lambda, forest_size, bbox, model, nugget, range, sill, _visitor, seed);

    auto r = esi->estimate(&qry);

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> loo_esi_kriging_3d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int model, float nugget, float range, float sill, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (smp_info.shape[1] != 3)
        throw std::runtime_error("[2] samples must have 3 coordinates");
    if (val_info.ndim != 1)
        throw std::runtime_error("[3] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[4] queries must be a 2 dimensions array");
    if (qry_info.shape[1] != 2)
        throw std::runtime_error("[5] queries must have 2 coordinates");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);

    sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(smp, val, lambda, forest_size, bbox, model, nugget, range, sill, _visitor, seed);

    auto r = esi->leave_one_out();

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> kfold_esi_kriging_3d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int model, float nugget, float range, float sill, int creation_seed, int k, int folding_seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (smp_info.shape[1] != 3)
        throw std::runtime_error("[2] samples must have 3 coordinates");
    if (val_info.ndim != 1)
        throw std::runtime_error("[3] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[4] queries must be a 2 dimensions array");
    if (qry_info.shape[1] != 2)
        throw std::runtime_error("[5] queries must have 2 coordinates");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);

    sptlz::ESI_Kriging* esi = new sptlz::ESI_Kriging(smp, val, lambda, forest_size, bbox, model, nugget, range, sill, _visitor, creation_seed);

    auto r = esi->k_fold(k, folding_seed);

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> estimation_voronoi_idw(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, float exp, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (val_info.ndim != 1)
        throw std::runtime_error("[2] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[3] queries must be a 2 dimensions array");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
        
    sptlz::VORONOI_IDW* voronoi = new sptlz::VORONOI_IDW(smp, val, alpha, forest_size, bbox, exp, _visitor, seed);
    
    auto r = voronoi->estimate(&qry);

    delete voronoi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> loo_voronoi_idw(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, float exp, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (val_info.ndim != 1)
        throw std::runtime_error("[2] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[3] queries must be a 2 dimensions array");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
        
    sptlz::VORONOI_IDW* voronoi = new sptlz::VORONOI_IDW(smp, val, alpha, forest_size, bbox, exp, _visitor, seed);
    
    auto r = voronoi->leave_one_out();

    delete voronoi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> kfold_voronoi_idw(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, float exp, int creation_seed, int k, int folding_seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (val_info.ndim != 1)
        throw std::runtime_error("[2] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[3] queries must be a 2 dimensions array");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
        
    sptlz::VORONOI_IDW* voronoi = new sptlz::VORONOI_IDW(smp, val, alpha, forest_size, bbox, exp, _visitor, creation_seed);
    
    auto r = voronoi->k_fold(k, folding_seed);

    delete voronoi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> estimation_adaptive_esi_idw_2d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (smp_info.shape[1] != 2)
        throw std::runtime_error("[2] samples must have 2 coordinates");
    if (val_info.ndim != 1)
        throw std::runtime_error("[3] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[4] queries must be a 2 dimensions array");
    if (qry_info.shape[1] != 2)
        throw std::runtime_error("[5] queries must have 2 coordinates");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);
        
    sptlz::ADAPTIVE_ESI_IDW* esi = new sptlz::ADAPTIVE_ESI_IDW(smp, val, lambda, forest_size, bbox, _visitor, seed);
    
    auto r = esi->estimate(&qry);

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> loo_adaptive_esi_idw_2d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (smp_info.shape[1] != 2)
        throw std::runtime_error("[2] samples must have 2 coordinates");
    if (val_info.ndim != 1)
        throw std::runtime_error("[3] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[4] queries must be a 2 dimensions array");
    if (qry_info.shape[1] != 2)
        throw std::runtime_error("[5] queries must have 2 coordinates");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);
        
    sptlz::ADAPTIVE_ESI_IDW* esi = new sptlz::ADAPTIVE_ESI_IDW(smp, val, lambda, forest_size, bbox, _visitor, seed);
    
    auto r = esi->leave_one_out();

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> kfold_adaptive_esi_idw_2d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int creation_seed, int k, int folding_seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (smp_info.shape[1] != 2)
        throw std::runtime_error("[2] samples must have 2 coordinates");
    if (val_info.ndim != 1)
        throw std::runtime_error("[3] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[4] queries must be a 2 dimensions array");
    if (qry_info.shape[1] != 2)
        throw std::runtime_error("[5] queries must have 2 coordinates");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);
        
    sptlz::ADAPTIVE_ESI_IDW* esi = new sptlz::ADAPTIVE_ESI_IDW(smp, val, lambda, forest_size, bbox, _visitor, creation_seed);
    
    auto r = esi->k_fold(k, folding_seed);

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> estimation_adaptive_esi_idw_3d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (smp_info.shape[1] != 3)
        throw std::runtime_error("[2] samples must have 3 coordinates");
    if (val_info.ndim != 1)
        throw std::runtime_error("[3] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[4] queries must be a 2 dimensions array");
    if (qry_info.shape[1] != 3)
        throw std::runtime_error("[5] queries must have 3 coordinates");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);
        
    sptlz::ADAPTIVE_ESI_IDW* esi = new sptlz::ADAPTIVE_ESI_IDW(smp, val, lambda, forest_size, bbox, _visitor, seed);
    
    auto r = esi->estimate(&qry);

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> loo_adaptive_esi_idw_3d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (smp_info.shape[1] != 3)
        throw std::runtime_error("[2] samples must have 3 coordinates");
    if (val_info.ndim != 1)
        throw std::runtime_error("[3] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[4] queries must be a 2 dimensions array");
    if (qry_info.shape[1] != 3)
        throw std::runtime_error("[5] queries must have 3 coordinates");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);
        
    sptlz::ADAPTIVE_ESI_IDW* esi = new sptlz::ADAPTIVE_ESI_IDW(smp, val, lambda, forest_size, bbox, _visitor, seed);
    
    auto r = esi->leave_one_out();

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> kfold_adaptive_esi_idw_3d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int creation_seed, int k, int folding_seed, py::array_t<float> queries, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (smp_info.shape[1] != 3)
        throw std::runtime_error("[2] samples must have 3 coordinates");
    if (val_info.ndim != 1)
        throw std::runtime_error("[3] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[4] queries must be a 2 dimensions array");
    if (qry_info.shape[1] != 3)
        throw std::runtime_error("[5] queries must have 3 coordinates");

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);
        
    sptlz::ADAPTIVE_ESI_IDW* esi = new sptlz::ADAPTIVE_ESI_IDW(smp, val, lambda, forest_size, bbox, _visitor, creation_seed);
    
    auto r = esi->k_fold(k, folding_seed);

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> estimation_custom_esi(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int seed, py::array_t<float> queries, std::optional<py::function> post_creation, py::function estimation, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();

    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (val_info.ndim != 1)
        throw std::runtime_error("[2] values must be a 1 dimensions array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[3] queries must be a 2 dimensions array");
    int n = smp_info.shape[1];

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*)> _post = NULL;
    if (post_creation.has_value()){
      _post = [post_creation, n](std::vector<std::vector<float>>* pos, std::vector<float>* val)->std::vector<float>{
        auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
        auto _val = sptlz::vector_1d_to_ndarray(val);
        auto res = (py::array_t<float>) post_creation.value().call(_pos, _val);
        return(sptlz::ndarray_to_vector_1d(&res));
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);
    
    sptlz::CUSTOM_ESI* esi = new sptlz::CUSTOM_ESI(smp, val, lambda, forest_size, bbox, _post, [estimation, n](std::vector<std::vector<float>>* pos, std::vector<float>* val, std::vector<std::vector<float>>* loc, std::vector<float>* params){
      auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
      auto _val = sptlz::vector_1d_to_ndarray(val);
      auto _loc = sptlz::vector_2d_to_ndarray(loc, n);
      auto _params = sptlz::vector_1d_to_ndarray(params);
      auto res = (py::array_t<float>) estimation.call(_pos, _val, _loc, _params);
      return(sptlz::ndarray_to_vector_1d(&res));
    }, NULL, NULL, _visitor, seed);
    auto r = esi->estimate(&qry);

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> loo_custom_esi(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int seed, py::array_t<float> queries, std::optional<py::function> post_creation, py::function loo, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();

    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (val_info.ndim != 1)
        throw std::runtime_error("[2] values must be a 1 dimensions array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[3] queries must be a 2 dimensions array");
    int n = smp_info.shape[1];

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*)> _post = NULL;
    if (post_creation.has_value()){
      _post = [post_creation, n](std::vector<std::vector<float>>* pos, std::vector<float>* val)->std::vector<float>{
        auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
        auto _val = sptlz::vector_1d_to_ndarray(val);
        auto res = (py::array_t<float>) post_creation.value().call(_pos, _val);
        return(sptlz::ndarray_to_vector_1d(&res));
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);
    
    sptlz::CUSTOM_ESI* esi = new sptlz::CUSTOM_ESI(smp, val, lambda, forest_size, bbox, _post, NULL, [loo, n](std::vector<std::vector<float>>* pos, std::vector<float>* val, std::vector<float>* params){
      auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
      auto _val = sptlz::vector_1d_to_ndarray(val);
      auto _params = sptlz::vector_1d_to_ndarray(params);
      auto res = (py::array_t<float>) loo.call(_pos, _val, _params);
      return(sptlz::ndarray_to_vector_1d(&res));
    }, NULL, _visitor, seed);
    auto r = esi->leave_one_out();

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> kfold_custom_esi(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int creation_seed, int k, int folding_seed, py::array_t<float> queries, std::optional<py::function> post_creation, py::function kfold, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();

    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (val_info.ndim != 1)
        throw std::runtime_error("[2] values must be a 1 dimensions array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[3] queries must be a 2 dimensions array");
    int n = smp_info.shape[1];

    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*)> _post = NULL;
    if (post_creation.has_value()){
      _post = [post_creation, n](std::vector<std::vector<float>>* pos, std::vector<float>* val)->std::vector<float>{
        auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
        auto _val = sptlz::vector_1d_to_ndarray(val);
        auto res = (py::array_t<float>) post_creation.value().call(_pos, _val);
        return(sptlz::ndarray_to_vector_1d(&res));
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);
    
    sptlz::CUSTOM_ESI* esi = new sptlz::CUSTOM_ESI(smp, val, lambda, forest_size, bbox, _post, NULL, NULL, [kfold, n](int _k, std::vector<std::vector<float>>* pos, std::vector<float>* val, std::vector<int>* fld, std::vector<float>* params){
      auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
      auto _val = sptlz::vector_1d_to_ndarray(val);
      auto _fld = sptlz::vector_1d_to_ndarray(fld);
      auto _params = sptlz::vector_1d_to_ndarray(params);
      auto res = (py::array_t<float>) kfold.call(_k, _pos, _val, _fld, _params);
      return(sptlz::ndarray_to_vector_1d(&res));
    }, _visitor, creation_seed);
    auto r = esi->k_fold(k, folding_seed);

    delete esi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> estimation_custom_coesi(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int seed, py::array_t<float> queries, std::optional<py::function> co_post_creation, py::function co_estimation, std::optional<py::function> ind_post_creation, py::function ind_estimation, py::function ind_aggregation, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();

    if (smp_info.ndim != 3)
        throw std::runtime_error("[1] samples must be a 3 dimensions array");
    if (val_info.ndim != 2)
        throw std::runtime_error("[2] values must be a 2 dimensions array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[3] queries must be a 2 dimensions array");
    int n = smp_info.shape[1], m = smp_info.shape[0];

    auto smp = sptlz::ndarray_to_vector_3d(&samples);
    auto val = sptlz::ndarray_to_vector_2d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*)> _ind_post = NULL;
    if (ind_post_creation.has_value()){
      _ind_post = [ind_post_creation, n](std::vector<std::vector<float>>* pos, std::vector<float>* val)->std::vector<float>{
        auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
        auto _val = sptlz::vector_1d_to_ndarray(val);
        auto res = (py::array_t<float>) ind_post_creation.value().call(_pos, _val);
        return(sptlz::ndarray_to_vector_1d(&res));
      };
    }

    std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<std::vector<float>>*)> _co_post = NULL;
    if (co_post_creation.has_value()){
      _co_post = [co_post_creation, n, m](std::vector<std::vector<float>>* pos, std::vector<std::vector<float>>* val)->std::vector<float>{
        auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
        auto _val = sptlz::vector_2d_to_ndarray(val, m);
        auto res = (py::array_t<float>) co_post_creation.value().call(_pos, _val);
        return(sptlz::ndarray_to_vector_1d(&res));
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);
    
    sptlz::CUSTOM_COESI* coesi = new sptlz::CUSTOM_COESI(lambda, forest_size, 100, bbox, _co_post, 
        [co_estimation, n, m](std::vector<std::vector<float>>* pos, std::vector<std::vector<float>>* val, std::vector<std::vector<float>>* loc, std::vector<float>* params){
            auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
            auto _val = sptlz::vector_2d_to_ndarray(val, m);
            auto _loc = sptlz::vector_2d_to_ndarray(loc, n);
            auto _params = sptlz::vector_1d_to_ndarray(params);
            auto res = (py::array_t<float>) co_estimation.call(_pos, _val, _loc, _params);
            return(sptlz::ndarray_to_vector_1d(&res));
        }, NULL, NULL, _visitor, seed);
    for(int i=0; i<m; i++){
        coesi->add_variable(smp.at(0), val.at(0), lambda, forest_size, _ind_post, 
            [ind_estimation, n](std::vector<std::vector<float>>* pos, std::vector<float>* val, std::vector<std::vector<float>>* loc, std::vector<float>* params){
                auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
                auto _val = sptlz::vector_1d_to_ndarray(val);
                auto _loc = sptlz::vector_2d_to_ndarray(loc, n);
                auto _params = sptlz::vector_1d_to_ndarray(params);
                auto res = (py::array_t<float>) ind_estimation.call(_pos, _val, _loc, _params);
                return(sptlz::ndarray_to_vector_1d(&res));
            }, NULL, NULL,
            [ind_aggregation](std::vector<float>* v)->float{
                auto _val = sptlz::vector_1d_to_ndarray(v);
                auto res =  ind_aggregation.call(_val);
                return(res.cast<float>());
            });
    }
    auto r = coesi->estimate(&qry);

    delete coesi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> marginal_loo_custom_coesi(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int seed, py::array_t<float> queries, std::optional<py::function> post_creation, py::function loo, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();

    if (smp_info.ndim != 3)
        throw std::runtime_error("[1] samples must be a 3 dimensions array");
    if (val_info.ndim != 2)
        throw std::runtime_error("[2] values must be a 2 dimensions array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[3] queries must be a 2 dimensions array");
    int n = smp_info.shape[1], m = smp_info.shape[0];

    auto smp = sptlz::ndarray_to_vector_3d(&samples);
    auto val = sptlz::ndarray_to_vector_2d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*)> _ind_post = NULL;
    if (post_creation.has_value()){
      _ind_post = [post_creation, n](std::vector<std::vector<float>>* pos, std::vector<float>* val)->std::vector<float>{
        auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
        auto _val = sptlz::vector_1d_to_ndarray(val);
        auto res = (py::array_t<float>) post_creation.value().call(_pos, _val);
        return(sptlz::ndarray_to_vector_1d(&res));
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);
    
    sptlz::CUSTOM_COESI* coesi = new sptlz::CUSTOM_COESI(lambda, forest_size, 100, bbox, NULL, NULL, NULL, NULL, _visitor, seed);
    for(int i=0; i<m; i++){
        coesi->add_variable(smp.at(0), val.at(0), lambda, forest_size, _ind_post, NULL, 
            [loo, n](std::vector<std::vector<float>>* pos, std::vector<float>* val, std::vector<float>* params){
                auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
                auto _val = sptlz::vector_1d_to_ndarray(val);
                auto _params = sptlz::vector_1d_to_ndarray(params);
                auto res = (py::array_t<float>) loo.call(_pos, _val, _params);
                return(sptlz::ndarray_to_vector_1d(&res));
            }, NULL, NULL);
    }
    auto r = coesi->marginal_leave_one_out();

    delete coesi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_3d_to_ndarray(&r));
    return(out);
}

std::tuple<py::object, py::array_t<float>> marginal_kfold_custom_coesi(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int creation_seed, int k, int folding_seed, py::array_t<float> queries, std::optional<py::function> post_creation, py::function kfold, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();

    if (smp_info.ndim != 3)
        throw std::runtime_error("[1] samples must be a 3 dimensions array");
    if (val_info.ndim != 2)
        throw std::runtime_error("[2] values must be a 2 dimensions array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[3] queries must be a 2 dimensions array");
    int n = smp_info.shape[1], m = smp_info.shape[0];

    auto smp = sptlz::ndarray_to_vector_3d(&samples);
    auto val = sptlz::ndarray_to_vector_2d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value().call(s);
        return(0);
      };
    }

    std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*)> _ind_post = NULL;
    if (post_creation.has_value()){
      _ind_post = [post_creation, n](std::vector<std::vector<float>>* pos, std::vector<float>* val)->std::vector<float>{
        auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
        auto _val = sptlz::vector_1d_to_ndarray(val);
        auto res = (py::array_t<float>) post_creation.value().call(_pos, _val);
        return(sptlz::ndarray_to_vector_1d(&res));
      };
    }

    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float lambda = sptlz::bbox_sum_interval(bbox);
    lambda = 1/(lambda-alpha*lambda);
    
    sptlz::CUSTOM_COESI* coesi = new sptlz::CUSTOM_COESI(lambda, forest_size, 100, bbox, NULL, NULL, NULL, NULL, _visitor, creation_seed);
    for(int i=0; i<m; i++){
        coesi->add_variable(smp.at(0), val.at(0), lambda, forest_size, _ind_post, NULL, NULL,
            [kfold, n](int _k, std::vector<std::vector<float>>* pos, std::vector<float>* val, std::vector<int>* fld, std::vector<float>* params){
                auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
                auto _val = sptlz::vector_1d_to_ndarray(val);
                auto _fld = sptlz::vector_1d_to_ndarray(fld);
                auto _params = sptlz::vector_1d_to_ndarray(params);
                auto res = (py::array_t<float>) kfold.call(_k, _pos, _val, _fld, _params);
                return(sptlz::ndarray_to_vector_1d(&res));
            }, NULL);
    }
    auto r = coesi->marginal_k_fold(k, folding_seed);

    delete coesi;

    std::tuple<py::object, py::array_t<float>> out = std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_3d_to_ndarray(&r));
    return(out);
}

PYBIND11_MODULE(libspatialize, m) {
    /* Partition */
    m.def(
      "get_partitions_using_esi", 
      &get_partitions_using_esi, 
      "get leaves bbox for every partition"
    );
    m.def(
      "get_leaf_for_samples_using_esi", 
      &get_leaf_for_samples_using_esi, 
      "get several partitions using MondrianTree"
    );
    /* plain NN IDW*/
    m.def(
      "estimation_nn_idw", 
      &estimation_nn_idw, 
      "IDW using nearest neighbors to estimate"
    );
    m.def(
      "loo_nn_idw", 
      &loo_nn_idw, 
      "Leave-one-out validation for IDW using nearest neighbors"
    );
    m.def(
      "kfold_nn_idw", 
      &kfold_nn_idw, 
      "K-fold validation for IDW using nearest neighbors"
    );
    /* ESI IDW*/
    m.def(
      "estimation_esi_idw", 
      &estimation_esi_idw, 
      "IDW using ESI to estimate"
    );
    m.def(
      "loo_esi_idw", 
      &loo_esi_idw, 
      "Leave-one-out validation for IDW using ESI"
    );
    m.def(
      "kfold_esi_idw", 
      &kfold_esi_idw, 
      "K-fold validation for IDW using ESI"
    );
    /* ESI Kriging*/
    m.def(
      "estimation_esi_kriging_2d",
      &estimation_esi_kriging_2d,
      "Esi using Kriging on 2 dimensions to estimate"
    );
    m.def(
      "loo_esi_kriging_2d",
      &loo_esi_kriging_2d,
      "Leave-one-out validation for Esi using Kriging on 2 dimensions"
    );
    m.def(
      "kfold_esi_kriging_2d",
      &kfold_esi_kriging_2d,
      "K-fold validation for Esi using Kriging on 2 dimensions"
    );
    m.def(
      "estimation_esi_kriging_3d",
      &estimation_esi_kriging_3d,
      "Esi using Kriging on 3 dimensions to estimate"
    );
    m.def(
      "loo_esi_kriging_3d",
      &loo_esi_kriging_3d,
      "Leave-one-out validation for Esi using Kriging on 3 dimensions"
    );
    m.def(
      "kfold_esi_kriging_3d",
      &kfold_esi_kriging_3d,
      "K-fold validation for Esi using Kriging on 3 dimensions"
    );
    /* ESI voronoi IDW */
    m.def(
      "estimation_voronoi_idw",
      &estimation_voronoi_idw,
      "IDW using VORONOI ESI to estimate"
    );
    m.def(
      "loo_voronoi_idw",
      &loo_voronoi_idw,
      "Leave-one-out validation for IDW using VORONOI ESI"
    );
    m.def(
      "kfold_voronoi_idw",
      &kfold_voronoi_idw,
      "K-fold validation for IDW using VORONOI ESI"
    );
    /* Adaptive ESI */
    m.def(
      "estimation_adaptive_esi_idw_2d",
      &estimation_adaptive_esi_idw_2d,
      "ADAPTIVE IDW using ESI in 2D to estimate"
    );
    m.def(
      "loo_adaptive_esi_idw_2d",
      &loo_adaptive_esi_idw_2d,
      "Leave-one-out validation for ADAPTIVE IDW using Esi in 2D"
    );
    m.def(
      "kfold_adaptive_esi_idw_2d",
      &kfold_adaptive_esi_idw_2d,
      "K-fold validation for ADAPTIVE IDW using Esi in 2D"
    );
    m.def(
      "estimation_adaptive_esi_idw_3d",
      &estimation_adaptive_esi_idw_3d,
      "ADAPTIVE IDW using ESI in 3D to estimate"
    );
    m.def(
      "loo_adaptive_esi_idw_3d",
      &loo_adaptive_esi_idw_3d,
      "Leave-one-out validation for ADAPTIVE IDW using Esi in 3D"
    );
    m.def(
      "kfold_adaptive_esi_idw_3d",
      &kfold_adaptive_esi_idw_3d,
      "K-fold validation for ADAPTIVE IDW using Esi in 3D"
    );
    /* Custom ESI */
    m.def(
      "estimation_custom_esi", 
      &estimation_custom_esi, 
      "Custom ESI to estimate"
    );
    m.def(
      "loo_custom_esi", 
      &loo_custom_esi, 
      "Leave-one-out validation for Custom ESI"
    );
    m.def(
      "kfold_custom_esi", 
      &kfold_custom_esi, 
      "K-fold validation for Custom ESI"
    );
    /* Custom COESI */
    m.def(
      "estimation_custom_coesi", 
      &estimation_custom_coesi, 
      "Custom COESI to estimate"
    );
    m.def(
      "marginal_loo_custom_coesi", 
      &marginal_loo_custom_coesi, 
      "Custom COESI marginal distribution leave one out"
    );
    m.def(
      "marginal_kfold_custom_coesi", 
      &marginal_kfold_custom_coesi, 
      "Custom COESI marginal distribution kfold validation"
    );
}

/*
signatures of argument FUNCTIONS for Custom ESI

def post_creation(leaf_coords, leaf_values):
  '''
  leaf_coords: coordinates for elements in the leaf
  leaf_values: values for elements in the leaf

  return: extra parameters for every leaf (example: [exponent, ratio, angle] for adaptive esi 2d)
  '''
  pass

def estimation(leaf_coords, leaf_values, queries, leaf_params):
  '''
  leaf_coords: coordinates for elements in the leaf
  leaf_values: values for elements in the leaf
  queries: coordinates for elements to estimate
  leaf_params: parameters for elements in the leaf (those calculated in post_creation)

  return: estimation in queries positions
  '''
  pass

def loo(leaf_coords, leaf_values, leaf_params):
  '''
  leaf_coords: coordinates for elements in the leaf
  leaf_values: values for elements in the leaf
  leaf_params: parameters for elements in the leaf (those calculated in post_creation)

  return: leave one out validation for the elements of the leaf
  '''
  pass

def kfold(k, leaf_coords, leaf_values, leaf_folds, leaf_params):
  '''
  k: number of classes
  leaf_coords: coordinates for elements in the leaf
  leaf_values: values for elements in the leaf
  leaf_folds: category betwwen 1 and k for elements in the leaf
  leaf_params: parameters for elements in the leaf (those calculated in post_creation)

  return: kfold validation for the elements of the leaf
  '''
  pass

*/  

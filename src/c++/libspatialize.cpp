#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <tuple>
#include <optional>
#include <memory>
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
        visitor.value()(s);
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
        visitor.value()(s);
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
        visitor.value()(s);
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
        visitor.value()(s);
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
        visitor.value()(s);
        return(0);
      };
    }

    std::vector<float> search_params = {radius, radius, radius, 0.0, 0.0, 0.0};
    sptlz::NN_IDW* myIDW = new sptlz::NN_IDW(smp, val, search_params, exp, _visitor);

    auto r = myIDW->k_fold(k, seed);

    delete myIDW;

    return(sptlz::vector_1d_to_ndarray(&r));
}

// ----------------------------------------------------------------------------------------------
// Ensemble estimation: one internal engine (any partition × any decoder × estimate/loo/kfold) and
// thin entry points. The legacy functions keep their names, signatures, validation messages and
// outputs; `run` exposes the same engine generically.
// ----------------------------------------------------------------------------------------------

typedef std::tuple<py::object, py::array_t<float>> EsiOutput;

static std::function<int(std::string)> make_visitor(std::optional<py::function> visitor){
    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value()(s);
        return(0);
      };
    }
    return(_visitor);
}

static EsiOutput esi_output(std::vector<std::vector<float>> *r){
    return(std::make_tuple(py::cast<py::none>(Py_None), sptlz::vector_2d_to_ndarray(r)));
}

// samples (n×d), values (n) and queries (m×d) of any dimension d
static void check_arrays(py::array_t<float> &samples, py::array_t<float> &values, py::array_t<float> &queries, const char *values_msg="[2] values must be a 1 dimension array"){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (val_info.ndim != 1)
        throw std::runtime_error(values_msg);
    if (qry_info.ndim != 2)
        throw std::runtime_error("[3] queries must be a 2 dimensions array");
}

// samples (n×d), values (n) and queries (m×d) of a fixed dimension d
static void check_arrays_dim(py::array_t<float> &samples, py::array_t<float> &values, py::array_t<float> &queries, int d){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();
    std::string coords = std::to_string(d) + " coordinates";
    if (smp_info.ndim != 2)
        throw std::runtime_error("[1] samples must be a 2 dimensions array");
    if (smp_info.shape[1] != d)
        throw std::runtime_error("[2] samples must have " + coords);
    if (val_info.ndim != 1)
        throw std::runtime_error("[3] values must be a 1 dimension array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[4] queries must be a 2 dimensions array");
    if (qry_info.shape[1] != d)
        throw std::runtime_error("[5] queries must have " + coords);
}

// Restricts OpenMP to one thread while alive unless `parallelize` (restores the count after).
class OmpThreads {
    #ifdef _OPENMP
    int original_threads;
    #endif
    bool parallelize;
  public:
    OmpThreads(bool _parallelize): parallelize(_parallelize){
        #ifdef _OPENMP
        original_threads = omp_get_max_threads();
        if (!parallelize) {
            omp_set_num_threads(1);
        }
        #endif
    }
    ~OmpThreads(){
        #ifdef _OPENMP
        if (!parallelize) {
            omp_set_num_threads(original_threads);
        }
        #endif
    }
};

enum class EsiMethod { ESTIMATE, LOO, KFOLD };

// The engine. Draws the forest ("mondrian": lifetime from alpha and the box of samples ∪ queries;
// "voronoi": alpha as nuclei rate, sign = data conditioning), fits the decoder (takes ownership)
// and runs the method. `class_name` names the computation in log messages.
static std::vector<std::vector<float>> run_ensemble(std::vector<std::vector<float>> &smp,
                                                    std::vector<float> &val,
                                                    std::vector<std::vector<float>> &qry,
                                                    const std::string &partition,
                                                    float alpha, int forest_size, int seed,
                                                    sptlz::Decoder *decoder,
                                                    EsiMethod method, int k, int folding_seed,
                                                    std::function<int(std::string)> visitor,
                                                    const std::string &class_name){
    std::unique_ptr<sptlz::Decoder> owned(decoder);
    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    std::unique_ptr<sptlz::Ensemble> ensemble;
    if (partition == "mondrian"){
        float lambda = sptlz::bbox_sum_interval(bbox);
        lambda = 1/(lambda-alpha*lambda);
        ensemble.reset(new sptlz::ESI(smp, val, lambda, forest_size, bbox, visitor, seed));
    }else if (partition == "voronoi"){
        ensemble.reset(new sptlz::VORONOI(smp, val, alpha, forest_size, bbox, visitor, seed));
    }else{
        throw std::runtime_error("unknown partition '" + partition + "' (expected 'mondrian' or 'voronoi')");
    }
    ensemble->set_class_name(class_name);
    ensemble->set_decoder(owned.release());

    if (method == EsiMethod::ESTIMATE){
        return(ensemble->estimate(&qry));
    }else if (method == EsiMethod::LOO){
        return(ensemble->leave_one_out());
    }
    return(ensemble->k_fold(k, folding_seed));
}

static EsiOutput run_ensemble_py(py::array_t<float> &samples, py::array_t<float> &values, py::array_t<float> &queries,
                                 const std::string &partition, float alpha, int forest_size, int seed,
                                 sptlz::Decoder *decoder, EsiMethod method, int k, int folding_seed,
                                 std::optional<py::function> visitor, const std::string &class_name){
    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    auto r = run_ensemble(smp, val, qry, partition, alpha, forest_size, seed, decoder, method, k, folding_seed, make_visitor(visitor), class_name);
    return(esi_output(&r));
}

/* ESI IDW */

EsiOutput estimation_esi_idw(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, float exp, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    check_arrays(samples, values, queries);
    return(run_ensemble_py(samples, values, queries, "mondrian", alpha, forest_size, seed, new sptlz::IDWDecoder(exp), EsiMethod::ESTIMATE, 0, 0, visitor, "ESI_IDW"));
}

EsiOutput loo_esi_idw(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, float exp, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    check_arrays(samples, values, queries);
    return(run_ensemble_py(samples, values, queries, "mondrian", alpha, forest_size, seed, new sptlz::IDWDecoder(exp), EsiMethod::LOO, 0, 0, visitor, "ESI_IDW"));
}

EsiOutput kfold_esi_idw(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, float exp, int creation_seed, int k, int folding_seed, py::array_t<float> queries, std::optional<py::function> visitor){
    check_arrays(samples, values, queries);
    return(run_ensemble_py(samples, values, queries, "mondrian", alpha, forest_size, creation_seed, new sptlz::IDWDecoder(exp), EsiMethod::KFOLD, k, folding_seed, visitor, "ESI_IDW"));
}

/* ESI Kriging */

static EsiOutput esi_kriging(int d, EsiMethod method, py::array_t<float> &samples, py::array_t<float> &values, int forest_size, float alpha, int model, float nugget, float range, float sill, int seed, int k, int folding_seed, py::array_t<float> &queries, std::optional<py::function> visitor){
    check_arrays_dim(samples, values, queries, d);
    return(run_ensemble_py(samples, values, queries, "mondrian", alpha, forest_size, seed, new sptlz::KrigingDecoder(model, nugget, range, sill), method, k, folding_seed, visitor, "ESI_Kriging"));
}

EsiOutput estimation_esi_kriging_2d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int model, float nugget, float range, float sill, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    return(esi_kriging(2, EsiMethod::ESTIMATE, samples, values, forest_size, alpha, model, nugget, range, sill, seed, 0, 0, queries, visitor));
}

EsiOutput loo_esi_kriging_2d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int model, float nugget, float range, float sill, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    return(esi_kriging(2, EsiMethod::LOO, samples, values, forest_size, alpha, model, nugget, range, sill, seed, 0, 0, queries, visitor));
}

EsiOutput kfold_esi_kriging_2d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int model, float nugget, float range, float sill, int creation_seed, int k, int folding_seed, py::array_t<float> queries, std::optional<py::function> visitor){
    return(esi_kriging(2, EsiMethod::KFOLD, samples, values, forest_size, alpha, model, nugget, range, sill, creation_seed, k, folding_seed, queries, visitor));
}

EsiOutput estimation_esi_kriging_3d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int model, float nugget, float range, float sill, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    return(esi_kriging(3, EsiMethod::ESTIMATE, samples, values, forest_size, alpha, model, nugget, range, sill, seed, 0, 0, queries, visitor));
}

EsiOutput loo_esi_kriging_3d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int model, float nugget, float range, float sill, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    return(esi_kriging(3, EsiMethod::LOO, samples, values, forest_size, alpha, model, nugget, range, sill, seed, 0, 0, queries, visitor));
}

EsiOutput kfold_esi_kriging_3d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int model, float nugget, float range, float sill, int creation_seed, int k, int folding_seed, py::array_t<float> queries, std::optional<py::function> visitor){
    return(esi_kriging(3, EsiMethod::KFOLD, samples, values, forest_size, alpha, model, nugget, range, sill, creation_seed, k, folding_seed, queries, visitor));
}

/* Voronoi IDW (its own IDW kernel, see VoronoiIDWDecoder) */

EsiOutput estimation_voronoi_idw(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, float exp, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    check_arrays(samples, values, queries);
    return(run_ensemble_py(samples, values, queries, "voronoi", alpha, forest_size, seed, new sptlz::VoronoiIDWDecoder(exp), EsiMethod::ESTIMATE, 0, 0, visitor, "VORONOI_IDW"));
}

EsiOutput loo_voronoi_idw(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, float exp, int seed, py::array_t<float> queries, std::optional<py::function> visitor){
    check_arrays(samples, values, queries);
    return(run_ensemble_py(samples, values, queries, "voronoi", alpha, forest_size, seed, new sptlz::VoronoiIDWDecoder(exp), EsiMethod::LOO, 0, 0, visitor, "VORONOI_IDW"));
}

EsiOutput kfold_voronoi_idw(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, float exp, int creation_seed, int k, int folding_seed, py::array_t<float> queries, std::optional<py::function> visitor){
    check_arrays(samples, values, queries);
    return(run_ensemble_py(samples, values, queries, "voronoi", alpha, forest_size, creation_seed, new sptlz::VoronoiIDWDecoder(exp), EsiMethod::KFOLD, k, folding_seed, visitor, "VORONOI_IDW"));
}

/* Adaptive ESI IDW */

static EsiOutput adaptive_esi_idw(int d, EsiMethod method, py::array_t<float> &samples, py::array_t<float> &values, int forest_size, float alpha, int seed, std::string metric, bool parallelize, int k, int folding_seed, py::array_t<float> &queries, std::optional<py::function> visitor){
    check_arrays_dim(samples, values, queries, d);
    OmpThreads threads(parallelize);
    return(run_ensemble_py(samples, values, queries, "mondrian", alpha, forest_size, seed, new sptlz::AdaptiveIDWDecoder(d, metric), method, k, folding_seed, visitor, "ADAPTIVE_ESI_IDW"));
}

EsiOutput estimation_adaptive_esi_idw_2d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int seed, std::string metric, bool parallelize, py::array_t<float> queries, std::optional<py::function> visitor){
    return(adaptive_esi_idw(2, EsiMethod::ESTIMATE, samples, values, forest_size, alpha, seed, metric, parallelize, 0, 0, queries, visitor));
}

EsiOutput loo_adaptive_esi_idw_2d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int seed, std::string metric, bool parallelize, py::array_t<float> queries, std::optional<py::function> visitor){
    return(adaptive_esi_idw(2, EsiMethod::LOO, samples, values, forest_size, alpha, seed, metric, parallelize, 0, 0, queries, visitor));
}

EsiOutput kfold_adaptive_esi_idw_2d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int creation_seed, std::string metric, bool parallelize, int k, int folding_seed, py::array_t<float> queries, std::optional<py::function> visitor){
    return(adaptive_esi_idw(2, EsiMethod::KFOLD, samples, values, forest_size, alpha, creation_seed, metric, parallelize, k, folding_seed, queries, visitor));
}

EsiOutput estimation_adaptive_esi_idw_3d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int seed, std::string metric, bool parallelize, py::array_t<float> queries, std::optional<py::function> visitor){
    return(adaptive_esi_idw(3, EsiMethod::ESTIMATE, samples, values, forest_size, alpha, seed, metric, parallelize, 0, 0, queries, visitor));
}

EsiOutput loo_adaptive_esi_idw_3d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int seed, std::string metric, bool parallelize, py::array_t<float> queries, std::optional<py::function> visitor){
    return(adaptive_esi_idw(3, EsiMethod::LOO, samples, values, forest_size, alpha, seed, metric, parallelize, 0, 0, queries, visitor));
}

EsiOutput kfold_adaptive_esi_idw_3d(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int creation_seed, std::string metric, bool parallelize, int k, int folding_seed, py::array_t<float> queries, std::optional<py::function> visitor){
    return(adaptive_esi_idw(3, EsiMethod::KFOLD, samples, values, forest_size, alpha, creation_seed, metric, parallelize, k, folding_seed, queries, visitor));
}

/* Custom ESI (decoder given by Python callbacks) */

typedef std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*)> CustomPost;

static CustomPost custom_post(std::optional<py::function> post_creation, int n){
    CustomPost _post = NULL;
    if (post_creation.has_value()){
      _post = [post_creation, n](std::vector<std::vector<float>>* pos, std::vector<float>* val)->std::vector<float>{
        auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
        auto _val = sptlz::vector_1d_to_ndarray(val);
        auto res = (py::array_t<float>) post_creation.value()(_pos, _val);
        return(sptlz::ndarray_to_vector_1d(&res));
      };
    }
    return(_post);
}

EsiOutput estimation_custom_esi(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int seed, py::array_t<float> queries, std::optional<py::function> post_creation, py::function estimation, std::optional<py::function> visitor){
    check_arrays(samples, values, queries, "[2] values must be a 1 dimensions array");
    int n = static_cast<int>(samples.request().shape[1]);
    auto decoder = new sptlz::CustomDecoder(custom_post(post_creation, n), [estimation, n](std::vector<std::vector<float>>* pos, std::vector<float>* val, std::vector<std::vector<float>>* loc, std::vector<float>* params){
      auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
      auto _val = sptlz::vector_1d_to_ndarray(val);
      auto _loc = sptlz::vector_2d_to_ndarray(loc, n);
      auto _params = sptlz::vector_1d_to_ndarray(params);
      auto res = (py::array_t<float>) estimation(_pos, _val, _loc, _params);
      return(sptlz::ndarray_to_vector_1d(&res));
    }, NULL, NULL);
    return(run_ensemble_py(samples, values, queries, "mondrian", alpha, forest_size, seed, decoder, EsiMethod::ESTIMATE, 0, 0, visitor, "CUSTOM_ESI"));
}

EsiOutput loo_custom_esi(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int seed, py::array_t<float> queries, std::optional<py::function> post_creation, py::function loo, std::optional<py::function> visitor){
    check_arrays(samples, values, queries, "[2] values must be a 1 dimensions array");
    int n = static_cast<int>(samples.request().shape[1]);
    auto decoder = new sptlz::CustomDecoder(custom_post(post_creation, n), NULL, [loo, n](std::vector<std::vector<float>>* pos, std::vector<float>* val, std::vector<float>* params){
      auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
      auto _val = sptlz::vector_1d_to_ndarray(val);
      auto _params = sptlz::vector_1d_to_ndarray(params);
      auto res = (py::array_t<float>) loo(_pos, _val, _params);
      return(sptlz::ndarray_to_vector_1d(&res));
    }, NULL);
    return(run_ensemble_py(samples, values, queries, "mondrian", alpha, forest_size, seed, decoder, EsiMethod::LOO, 0, 0, visitor, "CUSTOM_ESI"));
}

EsiOutput kfold_custom_esi(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int creation_seed, int k, int folding_seed, py::array_t<float> queries, std::optional<py::function> post_creation, py::function kfold, std::optional<py::function> visitor){
    check_arrays(samples, values, queries, "[2] values must be a 1 dimensions array");
    int n = static_cast<int>(samples.request().shape[1]);
    auto decoder = new sptlz::CustomDecoder(custom_post(post_creation, n), NULL, NULL, [kfold, n](int _k, std::vector<std::vector<float>>* pos, std::vector<float>* val, std::vector<int>* fld, std::vector<float>* params){
      auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
      auto _val = sptlz::vector_1d_to_ndarray(val);
      auto _fld = sptlz::vector_1d_to_ndarray(fld);
      auto _params = sptlz::vector_1d_to_ndarray(params);
      auto res = (py::array_t<float>) kfold(_k, _pos, _val, _fld, _params);
      return(sptlz::ndarray_to_vector_1d(&res));
    });
    return(run_ensemble_py(samples, values, queries, "mondrian", alpha, forest_size, creation_seed, decoder, EsiMethod::KFOLD, k, folding_seed, visitor, "CUSTOM_ESI"));
}

/* Generic entry point */

template <typename T>
static T param_or(const py::dict &params, const char *name, T fallback){
    return(params.contains(name) ? params[name].cast<T>() : fallback);
}

template <typename T>
static T required_param(const py::dict &params, const char *name, const std::string &decoder){
    if (!params.contains(name))
        throw std::runtime_error("decoder '" + decoder + "' needs parameter '" + name + "'");
    return(params[name].cast<T>());
}

// run(samples, values, queries, partition, alpha, forest_size, seed, decoder, params, method, k,
//     folding_seed, visitor): any partition ("mondrian", "voronoi") with any decoder ("idw",
// "kriging", "adaptiveidw"). `params` holds the decoder parameters (idw: exponent; kriging: model
// (1 spherical, 2 exponential, 3 cubic, 4 gaussian), nugget, range, sill; adaptiveidw: metric,
// parallelize). method: "estimate" (queries), "loo" or "kfold" (k, folding_seed). Returns
// (None, array) like the other entry points.
EsiOutput run(py::array_t<float> samples, py::array_t<float> values, py::array_t<float> queries,
              std::string partition, float alpha, int forest_size, int seed,
              std::string decoder, py::dict params, std::string method, int k, int folding_seed,
              std::optional<py::function> visitor){
    check_arrays(samples, values, queries);
    int d = static_cast<int>(samples.request().shape[1]);
    if (d != static_cast<int>(queries.request().shape[1]))
        throw std::runtime_error("samples and queries must have the same number of coordinates");

    EsiMethod m;
    if (method == "estimate") m = EsiMethod::ESTIMATE;
    else if (method == "loo") m = EsiMethod::LOO;
    else if (method == "kfold") m = EsiMethod::KFOLD;
    else throw std::runtime_error("unknown method '" + method + "' (expected 'estimate', 'loo' or 'kfold')");

    bool parallelize = false;
    sptlz::Decoder *dec;
    if (decoder == "idw"){
        dec = new sptlz::IDWDecoder(required_param<float>(params, "exponent", decoder));
    }else if (decoder == "kriging"){
        dec = new sptlz::KrigingDecoder(required_param<int>(params, "model", decoder),
                                        required_param<float>(params, "nugget", decoder),
                                        required_param<float>(params, "range", decoder),
                                        required_param<float>(params, "sill", decoder));
    }else if (decoder == "adaptiveidw"){
        if (d != 2 && d != 3)
            throw std::runtime_error("decoder 'adaptiveidw' is available for 2 and 3 dimensions only");
        parallelize = param_or<bool>(params, "parallelize", false);
        dec = new sptlz::AdaptiveIDWDecoder(d, param_or<std::string>(params, "metric", "mae"));
    }else{
        throw std::runtime_error("unknown decoder '" + decoder + "' (expected 'idw', 'kriging' or 'adaptiveidw')");
    }

    OmpThreads threads(parallelize);
    return(run_ensemble_py(samples, values, queries, partition, alpha, forest_size, seed, dec, m, k, folding_seed, visitor, partition + "/" + decoder));
}

std::tuple<py::object, py::array_t<float>> estimation_custom_coesi(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int seed, py::array_t<float> queries, std::optional<py::function> co_post_creation, py::function co_estimation, std::optional<py::function> ind_post_creation, py::function ind_estimation, py::function ind_aggregation, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();

    if (smp_info.ndim != 3)
        throw std::runtime_error("[1] samples must be a 3 dimensions array");
    if (val_info.ndim != 2)
        throw std::runtime_error("[2] values must be a 2 dimensions array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[3] queries must be a 2 dimensions array");
    int n = static_cast<int>(smp_info.shape[1]), m = static_cast<int>(smp_info.shape[0]);

    auto smp = sptlz::ndarray_to_vector_3d(&samples);
    auto val = sptlz::ndarray_to_vector_2d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);

    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value()(s);
        return(0);
      };
    }

    std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*)> _ind_post = NULL;
    if (ind_post_creation.has_value()){
      _ind_post = [ind_post_creation, n](std::vector<std::vector<float>>* pos, std::vector<float>* val)->std::vector<float>{
        auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
        auto _val = sptlz::vector_1d_to_ndarray(val);
        auto res = (py::array_t<float>) ind_post_creation.value()(_pos, _val);
        return(sptlz::ndarray_to_vector_1d(&res));
      };
    }

    std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<std::vector<float>>*)> _co_post = NULL;
    if (co_post_creation.has_value()){
      _co_post = [co_post_creation, n, m](std::vector<std::vector<float>>* pos, std::vector<std::vector<float>>* val)->std::vector<float>{
        auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
        auto _val = sptlz::vector_2d_to_ndarray(val, m);
        auto res = (py::array_t<float>) co_post_creation.value()(_pos, _val);
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
            auto res = (py::array_t<float>) co_estimation(_pos, _val, _loc, _params);
            return(sptlz::ndarray_to_vector_1d(&res));
        }, NULL, NULL, _visitor, seed);
    for(int i=0; i<m; i++){
        coesi->add_variable(smp.at(0), val.at(0), lambda, forest_size, _ind_post,
            [ind_estimation, n](std::vector<std::vector<float>>* pos, std::vector<float>* val, std::vector<std::vector<float>>* loc, std::vector<float>* params){
                auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
                auto _val = sptlz::vector_1d_to_ndarray(val);
                auto _loc = sptlz::vector_2d_to_ndarray(loc, n);
                auto _params = sptlz::vector_1d_to_ndarray(params);
                auto res = (py::array_t<float>) ind_estimation(_pos, _val, _loc, _params);
                return(sptlz::ndarray_to_vector_1d(&res));
            }, NULL, NULL,
            [ind_aggregation](std::vector<float>* v)->float{
                auto _val = sptlz::vector_1d_to_ndarray(v);
                auto res =  ind_aggregation(_val);
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
    int n = static_cast<int>(smp_info.shape[1]), m = static_cast<int>(smp_info.shape[0]);

    auto smp = sptlz::ndarray_to_vector_3d(&samples);
    auto val = sptlz::ndarray_to_vector_2d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);

    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value()(s);
        return(0);
      };
    }

    std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*)> _ind_post = NULL;
    if (post_creation.has_value()){
      _ind_post = [post_creation, n](std::vector<std::vector<float>>* pos, std::vector<float>* val)->std::vector<float>{
        auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
        auto _val = sptlz::vector_1d_to_ndarray(val);
        auto res = (py::array_t<float>) post_creation.value()(_pos, _val);
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
                auto res = (py::array_t<float>) loo(_pos, _val, _params);
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
    int n = static_cast<int>(smp_info.shape[1]), m = static_cast<int>(smp_info.shape[0]);

    auto smp = sptlz::ndarray_to_vector_3d(&samples);
    auto val = sptlz::ndarray_to_vector_2d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);

    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value()(s);
        return(0);
      };
    }

    std::function<std::vector<float>(std::vector<std::vector<float>>*, std::vector<float>*)> _ind_post = NULL;
    if (post_creation.has_value()){
      _ind_post = [post_creation, n](std::vector<std::vector<float>>* pos, std::vector<float>* val)->std::vector<float>{
        auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
        auto _val = sptlz::vector_1d_to_ndarray(val);
        auto res = (py::array_t<float>) post_creation.value()(_pos, _val);
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
                auto res = (py::array_t<float>) kfold(_k, _pos, _val, _fld, _params);
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
    /* Generic ensemble: any partition with any decoder */
    m.def(
      "run",
      &run,
      "Ensemble estimation with any partition ('mondrian', 'voronoi') and decoder ('idw', 'kriging', 'adaptiveidw')",
      py::arg("samples"), py::arg("values"), py::arg("queries"), py::arg("partition"), py::arg("alpha"),
      py::arg("forest_size"), py::arg("seed"), py::arg("decoder"), py::arg("params"),
      py::arg("method") = "estimate", py::arg("k") = 0, py::arg("folding_seed") = 0,
      py::arg("visitor") = py::none()
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

    // On macOS, libomp's atexit cleanup hangs during Jupyter kernel restart because
    // worker threads are sleeping and cannot be joined cleanly.
    // Fix: at module load, set blocktime=0 so threads sleep immediately (not spin-wait).
    // At atexit, call omp_pause_resource_all(omp_pause_hard) which terminates all worker
    // threads (OpenMP 5.0), leaving nothing to join when libomp's own destructor runs.
    #if defined(_OPENMP) && defined(__APPLE__)
    kmp_set_blocktime(0);
    #endif
    py::module_::import("atexit").attr("register")(
        py::cpp_function([]() {
            #ifdef _OPENMP
            #ifdef _MSC_VER
                // MSVC does not support OpenMP 5.0 pause resources
            #else
                omp_pause_resource_all(omp_pause_hard);
            #endif
            omp_set_num_threads(1);
            #endif
        }, py::name("_spatialize_omp_cleanup"))
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

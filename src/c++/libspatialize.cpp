#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <tuple>
#include <optional>
#include <memory>
#include "spatialize/nn/nn_idw.hpp"
#include "spatialize/nn/nn_cokriging.hpp"
#include "spatialize/partitions/mondrian.hpp"
#include "spatialize/partitions/voronoi.hpp"
#include "spatialize/decoders/idw.hpp"
#include "spatialize/decoders/kriging.hpp"
#include "spatialize/decoders/adaptive_idw.hpp"
#include "spatialize/decoders/custom.hpp"
#include "spatialize/coesi/custom_coesi.hpp"
#include "registry.hpp"

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

    sptlz::ESI* esi = new sptlz::ESI(smp, {}, lambda, forest_size, bbox, _visitor, seed, true);
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

    sptlz::ESI* esi = new sptlz::ESI(smp, {}, lambda, forest_size, bbox, _visitor, seed, true);
    auto r = esi->get_leaf_for_samples();

    delete esi;

    return(sptlz::vector_2d_to_ndarray(&r));
}

// ---- plain NN (co)kriging --------------------------------------------------------------------
namespace {
std::vector<double> cok_vec(py::handle h){
    auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(h);
    if (!a) throw std::runtime_error("a numeric array was expected");
    return(std::vector<double>(a.data(), a.data() + a.size()));
}

template <size_t N, class T>
std::array<T, N> cok_arr(py::handle h, const char *name){
    auto v = cok_vec(h);
    if (v.size() != N) throw std::runtime_error(std::string("'") + name + "' must have " + std::to_string(N) + " values");
    std::array<T, N> r;
    for (size_t i = 0; i < N; i++) r[i] = static_cast<T>(v[i]);
    return(r);
}

sptlz::CokrigingParams cok_params(py::dict d, int n_vars){
    sptlz::CokrigingParams p;
    if (!d.contains("kriging_type")) throw std::runtime_error("Parameter 'kriging_type' needed ('simple' or 'ordinary')");
    std::string kt = py::cast<std::string>(d["kriging_type"]);
    std::transform(kt.begin(), kt.end(), kt.begin(), [](unsigned char c){return(std::tolower(c));});
    if (kt == "simple") {
        p.ordinary = false;
        if (!d.contains("kriging_mean")) throw std::runtime_error("Parameter 'kriging_mean' needed (one per variable)");
        p.mean = cok_vec(d["kriging_mean"]);
    } else if (kt != "ordinary") {
        throw std::runtime_error("Parameter 'kriging_type' = '" + kt + "' is not allowed");
    }
    if (d.contains("limits")) {
        auto l = cok_arr<2, double>(d["limits"], "limits");
        p.limit_min = l[0];
        p.limit_max = l[1];
    }
    p.top_cut = d.contains("top_cut") ? cok_vec(d["top_cut"]) : std::vector<double>(n_vars, std::numeric_limits<double>::max());
    if (!d.contains("nb_radius")) throw std::runtime_error("Parameter 'nb_radius' needed (3 array)");
    p.nb_radius = cok_arr<3, double>(d["nb_radius"], "nb_radius");
    if (!d.contains("nb_angles")) throw std::runtime_error("Parameter 'nb_angles' needed (3 array)");
    p.nb_angles = cok_arr<3, double>(d["nb_angles"], "nb_angles");
    if (!d.contains("nb_ndata")) throw std::runtime_error("Parameter 'nb_ndata' needed (2 array)");
    auto nd = cok_arr<2, int>(d["nb_ndata"], "nb_ndata");
    p.nb_min = nd[0];
    p.nb_max = nd[1];
    if (d.contains("nb_octant")) p.nb_octant = py::cast<bool>(d["nb_octant"]);
    if (d.contains("missing_value")) p.missing_value = py::cast<double>(d["missing_value"]);
    if (d.contains("discretization")) p.discretization = cok_arr<3, int>(d["discretization"], "discretization");
    if (d.contains("block_size")) {
        p.block_size = cok_arr<3, double>(d["block_size"], "block_size");
    } else if (p.discretization != std::array<int, 3>({1, 1, 1})) {
        throw std::runtime_error("Parameter 'block_size' needed (3 array)");
    }
    if (!d.contains("variogram")) throw std::runtime_error("Parameter 'variogram' needed");
    py::dict vg = py::cast<py::dict>(d["variogram"]);
    p.nugget = cok_vec(vg["nugget"]);
    for (auto item : py::cast<py::list>(vg["model"])) {
        py::dict m = py::cast<py::dict>(item);
        sptlz::CokrigingStructure s;
        std::string t = py::cast<std::string>(m["type"]);
        if (t == "spherical") s.type = sptlz::COKRIGING_SPHERICAL;
        else if (t == "exponential") s.type = sptlz::COKRIGING_EXPONENTIAL;
        else if (t == "cubic") s.type = sptlz::COKRIGING_CUBIC;
        else if (t == "gaussian") s.type = sptlz::COKRIGING_GAUSSIAN;
        else throw std::runtime_error("variogram model '" + t + "' not supported");
        s.sill = cok_vec(m["sill"]);
        s.ranges = cok_arr<3, double>(m["ranges"], "ranges");
        s.angles = cok_arr<3, double>(m["angles"], "angles");
        p.structures.push_back(s);
    }
    return(p);
}

py::array_t<double> cok_matrix(const std::vector<double> &v, int rows, int cols){
    py::array_t<double> a({rows, cols});
    std::copy(v.begin(), v.end(), a.mutable_data());
    return(a);
}
}

// cokriging(samples_coords (N x 3), samples_values (N x M), params, target, target_coords, samples_dhid,
//           k, seed, visitor): the prediction, the variance and the number of neighbours of every
// target, as N' x M arrays. target: 'scattered' (target_coords), 'grid' (params grid_origin and
// grid_nodes), 'leave_one_out_sample', 'leave_one_out_drillhole' or 'k_fold' (k, seed).
py::tuple cokriging(py::array_t<double> samples, py::array_t<double> values, py::dict params, std::string target,
                    std::optional<py::array_t<double>> target_coords, std::optional<py::array_t<int>> samples_dhid,
                    int k, int seed, std::optional<py::function> visitor){
    py::buffer_info smp_info = samples.request(), val_info = values.request();
    if (smp_info.ndim != 2 || smp_info.shape[1] != 3)
        throw std::runtime_error("[1] samples_coords must be an N x 3 array");
    if (val_info.ndim != 2)
        throw std::runtime_error("[2] samples_values must be an N x M array");
    if (val_info.shape[0] != smp_info.shape[0])
        throw std::runtime_error("Parameters 'samples_coords'(" + std::to_string(smp_info.shape[0]) + ") and 'samples_values'(" + std::to_string(val_info.shape[0]) + ") do not have same length");

    int n_vars = static_cast<int>(val_info.shape[1]);
    std::vector<std::array<double, 3>> coords;
    auto cd = cok_vec(samples);
    for (size_t i = 0; i < cd.size(); i += 3) coords.push_back({cd[i], cd[i+1], cd[i+2]});
    std::vector<int> dhid;
    if (samples_dhid.has_value()) {
        auto a = samples_dhid.value();
        dhid.assign(a.data(), a.data() + a.size());
    }

    std::function<int(std::string)> _visitor = [](std::string s)->int{
      return(0);
    };
    if (visitor.has_value()){
      _visitor = [visitor](std::string s)->int{
        visitor.value()(s);
        return(0);
      };
    }

    auto p = cok_params(params, n_vars);
    sptlz::NN_Cokriging engine(coords, cok_vec(values), n_vars, p, dhid, _visitor);
    sptlz::CokrigingOutput out;

    if (target == "scattered") {
        if (!target_coords.has_value()) throw std::runtime_error("Parameter 'target_coords' needed (N x 3 array)");
        auto tb = target_coords.value().request();
        if (tb.ndim != 2 || tb.shape[1] != 3) throw std::runtime_error("Parameter 'target_coords' must be an N x 3 array");
        auto td = cok_vec(target_coords.value());
        std::vector<std::array<double, 3>> locs;
        for (size_t i = 0; i < td.size(); i += 3) locs.push_back({td[i], td[i+1], td[i+2]});
        out = engine.estimate(locs);
    } else if (target == "grid") {
        if (!params.contains("grid_origin") || !params.contains("grid_nodes"))
            throw std::runtime_error("Parameters 'grid_origin' and 'grid_nodes' needed (3 arrays)");
        out = engine.estimate_grid(cok_arr<3, double>(params["grid_origin"], "grid_origin"), cok_arr<3, int>(params["grid_nodes"], "grid_nodes"));
    } else if (target == "leave_one_out_sample") {
        out = engine.leave_one_out();
    } else if (target == "leave_one_out_drillhole") {
        out = engine.leave_one_drillhole_out();
    } else if (target == "k_fold") {
        if (k < 2) throw std::runtime_error("k must be at least 2");
        out = engine.k_fold(k, seed);
    } else {
        throw std::runtime_error("Parameter 'target' = '" + target + "' is not allowed");
    }
    return(py::make_tuple(cok_matrix(out.prediction, out.n_targets, out.n_vars),
                          cok_matrix(out.variance, out.n_targets, out.n_vars),
                          cok_matrix(out.neighbors, out.n_targets, out.n_vars)));
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

// Sets the number of OpenMP threads of the calling thread while alive (0 leaves the runtime's
// default, all processors unless OMP_NUM_THREADS says otherwise) and restores it after.
class OmpThreads {
    #ifdef _OPENMP
    int original_threads;
    #endif
    int num_threads;
  public:
    OmpThreads(int _num_threads): num_threads(_num_threads){
        #ifdef _OPENMP
        original_threads = omp_get_max_threads();
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }
        #endif
    }
    ~OmpThreads(){
        #ifdef _OPENMP
        if (num_threads > 0) {
            omp_set_num_threads(original_threads);
        }
        #endif
    }
};

// How the extension was built: whether it runs in parallel (OpenMP) and on how many threads.
py::dict build_info(){
    py::dict info;
    #ifdef _OPENMP
    info["openmp"] = true;
    info["openmp_version"] = _OPENMP;
    info["max_threads"] = omp_get_max_threads();
    info["num_procs"] = omp_get_num_procs();
    #else
    info["openmp"] = false;
    info["openmp_version"] = py::none();
    info["max_threads"] = 1;
    info["num_procs"] = 1;
    #endif
    return(info);
}

using EsiMethod = registry::Method;

// The engine. Draws the forest of the named partition (registry.hpp) on the box of samples ∪
// queries, fits the decoder (takes ownership) and runs the method. `class_name` names the
// computation in log messages.
static std::vector<std::vector<float>> run_ensemble(std::vector<std::vector<float>> &smp,
                                                    std::vector<float> &val,
                                                    std::vector<std::vector<float>> &qry,
                                                    const std::string &partition,
                                                    float alpha, int forest_size, int seed,
                                                    sptlz::Decoder *decoder,
                                                    EsiMethod method, int k, int folding_seed,
                                                    std::function<int(std::string)> visitor,
                                                    const std::string &class_name,
                                                    sptlz::EmptyCellPolicy empty_cells = sptlz::EmptyCellPolicy()){
    std::unique_ptr<sptlz::Decoder> owned(decoder);
    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float mu = sptlz::granularity_measure(&smp, bbox);
    auto &spec = registry::find(registry::partitions(), partition, "partition");
    std::unique_ptr<sptlz::Ensemble> ensemble(spec.make(smp, val, bbox, mu, alpha, forest_size, seed, visitor));
    ensemble->set_class_name(class_name);
    ensemble->set_empty_cells(empty_cells);
    // cross-validation fits the cells on demand (Ensemble::params_of)
    ensemble->set_decoder(owned.release(), method != EsiMethod::ESTIMATE);

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
                                 std::optional<py::function> visitor, const std::string &class_name,
                                 sptlz::EmptyCellPolicy empty_cells = sptlz::EmptyCellPolicy()){
    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto val = sptlz::ndarray_to_vector_1d(&values);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    auto r = run_ensemble(smp, val, qry, partition, alpha, forest_size, seed, decoder, method, k, folding_seed, make_visitor(visitor), class_name, empty_cells);
    return(esi_output(&r));
}

/* Generic entry point */

// run(samples, values, queries, partition, alpha, forest_size, seed, decoder, params, method, k,
//     folding_seed, visitor, num_threads): any partition with any decoder of the catalogue
// (registry.hpp), in the dimensions both support. `params` holds the decoder's parameters, checked
// against the catalogue. method: "estimate" (queries), "loo" or "kfold" (k, folding_seed).
// num_threads: OpenMP threads (0: the runtime's default). empty_cells: what a member is when its
// cell holds no datum ("nan", "mark", "coarsen"); mark_source, mark_knn: where the marks come from
// ("local" among the mark_knn nearest cells with data, "cells", "data"); mark_value: what the drawn
// cell gives ("decoder" prediction or a "datum", the block-mark model); see empty_cells.hpp. Returns (None, members).
EsiOutput run(py::array_t<float> samples, py::array_t<float> values, py::array_t<float> queries,
              std::string partition, float alpha, int forest_size, int seed,
              std::string decoder, py::dict params, std::string method, int k, int folding_seed,
              std::optional<py::function> visitor, int num_threads,
              std::string empty_cells, std::string mark_source, int mark_knn, std::string mark_value){
    check_arrays(samples, values, queries);
    int d = static_cast<int>(samples.request().shape[1]);
    if (d != static_cast<int>(queries.request().shape[1]))
        throw std::runtime_error("samples and queries must have the same number of coordinates");

    EsiMethod m = registry::method_from(method);
    auto &ps = registry::find(registry::partitions(), partition, "partition");
    auto &ds = registry::find(registry::decoders(), decoder, "decoder");
    registry::check_dim("partition", partition, ps.min_dim, ps.max_dim, d);
    registry::check_dim("decoder", decoder, ds.min_dim, ds.max_dim, d);
    registry::check_params(ds, params);

    if (num_threads < 0)
        throw std::runtime_error("num_threads must be 0 (the runtime's default) or positive");
    auto policy = sptlz::EmptyCellPolicy::from(empty_cells, mark_source, mark_knn, mark_value);
    sptlz::Decoder *dec = ds.make(d, params, m);
    OmpThreads threads(num_threads);
    return(run_ensemble_py(samples, values, queries, partition, alpha, forest_size, seed, dec, m, k, folding_seed, visitor, partition + "/" + decoder, policy));
}

// cells(samples, queries, partition, alpha, forest_size, seed, num_threads): the cell of every
// query in each of the forest_size partitions that run draws with the same arguments (the box of
// samples ∪ queries, the same seed), as an int array of shape (queries, forest_size). Two queries
// share a cell of partition t exactly when their labels in column t are equal.
py::array_t<int> cells(py::array_t<float> samples, py::array_t<float> queries, std::string partition,
                       float alpha, int forest_size, int seed, int num_threads){
    py::buffer_info smp_info = samples.request(), qry_info = queries.request();
    if (smp_info.ndim != 2 || qry_info.ndim != 2)
        throw std::runtime_error("samples and queries must be 2 dimensions arrays");
    int d = static_cast<int>(smp_info.shape[1]);
    if (d != static_cast<int>(qry_info.shape[1]))
        throw std::runtime_error("samples and queries must have the same number of coordinates");
    auto &ps = registry::find(registry::partitions(), partition, "partition");
    registry::check_dim("partition", partition, ps.min_dim, ps.max_dim, d);
    auto smp = sptlz::ndarray_to_vector_2d(&samples);
    auto qry = sptlz::ndarray_to_vector_2d(&queries);
    std::vector<float> val(smp.size(), 0.0f);
    auto bbox = sptlz::samples_coords_bbox(&smp, &qry);
    float mu = sptlz::granularity_measure(&smp, bbox);
    OmpThreads threads(num_threads);
    std::unique_ptr<sptlz::Ensemble> ensemble(ps.make(smp, val, bbox, mu, alpha, forest_size, seed, make_visitor(std::nullopt)));
    auto r = ensemble->cells_of(&qry);
    return(sptlz::vector_2d_to_ndarray(&r));
}

std::tuple<py::object, py::array_t<float>> estimation_custom_coesi(py::array_t<float> samples, py::array_t<float> values, int forest_size, float alpha, int seed, py::array_t<float> queries, std::optional<py::function> co_post_creation, py::function co_estimation, std::optional<py::function> ind_post_creation, py::function ind_estimation, py::function ind_aggregation, std::optional<py::function> visitor, int n_aux){
    py::buffer_info smp_info = samples.request(), val_info = values.request(), qry_info = queries.request();

    if (n_aux < 1)
        throw std::runtime_error("n_aux must be a positive number of auxiliary locations");
    if (smp_info.ndim != 3)
        throw std::runtime_error("[1] samples must be a 3 dimensions array");
    if (val_info.ndim != 2)
        throw std::runtime_error("[2] values must be a 2 dimensions array");
    if (qry_info.ndim != 2)
        throw std::runtime_error("[3] queries must be a 2 dimensions array");
    int n = static_cast<int>(smp_info.shape[2]), m = static_cast<int>(smp_info.shape[0]);  // coordinates, variables

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
    float lambda = sptlz::granularity_measure(&smp, bbox);
    lambda = 1/(lambda-alpha*lambda);

    sptlz::CUSTOM_COESI* coesi = new sptlz::CUSTOM_COESI(lambda, forest_size, n_aux, bbox, _co_post,
        [co_estimation, n, m](std::vector<std::vector<float>>* pos, std::vector<std::vector<float>>* val, std::vector<std::vector<float>>* loc, std::vector<float>* params){
            auto _pos = sptlz::vector_2d_to_ndarray(pos, n);
            auto _val = sptlz::vector_2d_to_ndarray(val, m);
            auto _loc = sptlz::vector_2d_to_ndarray(loc, n);
            auto _params = sptlz::vector_1d_to_ndarray(params);
            auto res = (py::array_t<float>) co_estimation(_pos, _val, _loc, _params);
            return(sptlz::ndarray_to_vector_1d(&res));
        }, NULL, NULL, _visitor, seed);
    for(int i=0; i<m; i++){
        coesi->add_variable(smp.at(i), val.at(i), lambda, forest_size, _ind_post,
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
    int n = static_cast<int>(smp_info.shape[2]), m = static_cast<int>(smp_info.shape[0]);  // coordinates, variables

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
    float lambda = sptlz::granularity_measure(&smp, bbox);
    lambda = 1/(lambda-alpha*lambda);

    sptlz::CUSTOM_COESI* coesi = new sptlz::CUSTOM_COESI(lambda, forest_size, 100, bbox, NULL, NULL, NULL, NULL, _visitor, seed);
    for(int i=0; i<m; i++){
        coesi->add_variable(smp.at(i), val.at(i), lambda, forest_size, _ind_post, NULL,
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
    int n = static_cast<int>(smp_info.shape[2]), m = static_cast<int>(smp_info.shape[0]);  // coordinates, variables

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
    float lambda = sptlz::granularity_measure(&smp, bbox);
    lambda = 1/(lambda-alpha*lambda);

    sptlz::CUSTOM_COESI* coesi = new sptlz::CUSTOM_COESI(lambda, forest_size, 100, bbox, NULL, NULL, NULL, NULL, _visitor, creation_seed);
    for(int i=0; i<m; i++){
        coesi->add_variable(smp.at(i), val.at(i), lambda, forest_size, _ind_post, NULL, NULL,
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
    /* plain NN cokriging */
    m.def(
      "cokriging",
      &cokriging,
      "Ordinary or simple (co)kriging using the neighbors of a search ellipsoid",
      py::arg("samples_coords"), py::arg("samples_values"), py::arg("params"), py::arg("target"),
      py::arg("target_coords") = py::none(), py::arg("samples_dhid") = py::none(),
      py::arg("k") = 0, py::arg("seed") = 206936, py::arg("visitor") = py::none()
    );
    /* Generic ensemble: any partition with any decoder */
    m.def(
      "run",
      &run,
      "Ensemble estimation with any partition and decoder of catalog()",
      py::arg("samples"), py::arg("values"), py::arg("queries"), py::arg("partition"), py::arg("alpha"),
      py::arg("forest_size"), py::arg("seed"), py::arg("decoder"), py::arg("params"),
      py::arg("method") = "estimate", py::arg("k") = 0, py::arg("folding_seed") = 0,
      py::arg("visitor") = py::none(), py::arg("num_threads") = 0,
      py::arg("empty_cells") = "nan", py::arg("mark_source") = "local", py::arg("mark_knn") = 8,
      py::arg("mark_value") = "decoder"
    );
    m.def(
      "cells",
      &cells,
      "The cell of every query in each partition run draws with the same arguments",
      py::arg("samples"), py::arg("queries"), py::arg("partition"), py::arg("alpha"),
      py::arg("forest_size"), py::arg("seed"), py::arg("num_threads") = 0
    );
    m.def(
      "build_info",
      &build_info,
      "How the extension was built: OpenMP availability, version and threads"
    );
    m.def(
      "catalog",
      &registry::catalog,
      "The partitions and decoders of the extension, with their dimensions and parameters"
    );
    /* Custom COESI */
    m.def(
      "estimation_custom_coesi",
      &estimation_custom_coesi,
      "Custom COESI to estimate (experimental, see spatialize.futures.coesi)",
      py::arg("samples"), py::arg("values"), py::arg("forest_size"), py::arg("alpha"), py::arg("seed"),
      py::arg("queries"), py::arg("co_post_creation"), py::arg("co_estimation"), py::arg("ind_post_creation"),
      py::arg("ind_estimation"), py::arg("ind_aggregation"), py::arg("visitor"), py::arg("n_aux") = 100
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

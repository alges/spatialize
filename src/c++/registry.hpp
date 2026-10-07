#ifndef _SPTLZ_BINDINGS_REGISTRY_
#define _SPTLZ_BINDINGS_REGISTRY_

// The catalogue of partitions (encoders) and decoders the extension offers: the single place where
// each one is registered, with the dimensions it supports and its parameters. `run` dispatches
// through it and `libspatialize.catalog()` exposes it to Python, which builds its facade from it.

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "spatialize/utils.hpp"
#include "spatialize/ensemble.hpp"
#include "spatialize/partitions/mondrian.hpp"
#include "spatialize/partitions/voronoi.hpp"
#include "spatialize/decoders/idw.hpp"
#include "spatialize/decoders/kriging.hpp"
#include "spatialize/decoders/adaptive_idw.hpp"
#include "spatialize/decoders/custom.hpp"

namespace py = pybind11;

namespace registry {

  enum class Method { ESTIMATE, LOO, KFOLD };

  inline Method method_from(const std::string &method){
    if (method == "estimate") return Method::ESTIMATE;
    if (method == "loo") return Method::LOO;
    if (method == "kfold") return Method::KFOLD;
    throw std::runtime_error("unknown method '" + method + "' (expected 'estimate', 'loo' or 'kfold')");
  }

  const int ANY = -1;  // no upper bound on the dimension

  struct Param {
    std::string name, type, doc;      // type: "float", "int", "bool", "str", "choice" or "callable"
    bool required;
    py::object fallback;              // default when not required (None for "no default")
    std::vector<std::string> choices; // for "choice": the accepted names, numbered from 1
  };

  typedef std::function<sptlz::Ensemble*(std::vector<std::vector<float>> &smp, std::vector<float> &val,
                                         std::vector<std::vector<float>> &bbox, float alpha, int forest_size,
                                         int seed, std::function<int(std::string)> visitor)> EnsembleFactory;
  typedef std::function<sptlz::Decoder*(int d, const py::dict &params, Method method)> DecoderFactory;

  struct PartitionSpec {
    std::string name, doc, alpha;
    int min_dim, max_dim;
    EnsembleFactory make;
  };

  struct DecoderSpec {
    std::string name, doc;
    int min_dim, max_dim;
    bool thread_safe;                 // false: cells are decoded one at a time (Python callbacks)
    std::vector<Param> params;
    DecoderFactory make;
  };

  // ---------------------------------------------------------------------------------------------
  // parameters

  inline const Param &param_spec(const DecoderSpec &spec, const std::string &name){
    for (auto &p: spec.params) if (p.name == name) return p;
    throw std::runtime_error("decoder '" + spec.name + "' has no parameter '" + name + "'");
  }

  // the value of a parameter: given, or its default; "choice" accepts a name or its number
  template <typename T>
  T param(const DecoderSpec &spec, const py::dict &params, const std::string &name){
    const Param &p = param_spec(spec, name);
    py::object value;
    if (params.contains(name.c_str()) && !params[name.c_str()].is_none()){
      value = params[name.c_str()];
    }else if (p.required){
      throw std::runtime_error("decoder '" + spec.name + "' needs parameter '" + name + "'");
    }else{
      value = p.fallback;
    }
    if (p.type == "choice" && py::isinstance<py::str>(value)){
      std::string s = value.cast<std::string>();
      for (size_t i=0; i<p.choices.size(); i++){
        if (p.choices[i] == s) return static_cast<T>(i+1);
      }
      throw std::runtime_error("parameter '" + name + "' of decoder '" + spec.name + "' must be one of its choices, not '" + s + "'");
    }
    return value.cast<T>();
  }

  inline void check_params(const DecoderSpec &spec, const py::dict &params){
    for (auto item: params){
      param_spec(spec, item.first.cast<std::string>());  // unknown names are errors
    }
  }

  // ---------------------------------------------------------------------------------------------
  // custom decoder: Python callables on the cells (cat_esi and user decoders)

  inline sptlz::CustomDecoder *custom_decoder(const DecoderSpec &spec, int d, const py::dict &params, Method method){
    auto callable = [&](const char *name, bool needed)->py::object{
      py::object f = params.contains(name) ? py::object(params[name]) : py::object(py::none());
      if (needed && f.is_none())
        throw std::runtime_error(std::string("decoder 'custom' needs parameter '") + name + "' for this method");
      return(f);
    };
    py::object post = callable("post_creation", false);
    sptlz::CustomDecoder::Post _post = NULL;
    if (!post.is_none()){
      _post = [post, d](std::vector<std::vector<float>>* pos, std::vector<float>* val)->std::vector<float>{
        auto res = (py::array_t<float>) post(sptlz::vector_2d_to_ndarray(pos, d), sptlz::vector_1d_to_ndarray(val));
        return(sptlz::ndarray_to_vector_1d(&res));
      };
    }
    sptlz::CustomDecoder::Estimation _est = NULL;
    sptlz::CustomDecoder::Loo _loo = NULL;
    sptlz::CustomDecoder::Kfold _kfold = NULL;
    if (method == Method::ESTIMATE){
      py::object f = callable("estimation", true);
      _est = [f, d](std::vector<std::vector<float>>* pos, std::vector<float>* val, std::vector<std::vector<float>>* loc, std::vector<float>* p){
        auto res = (py::array_t<float>) f(sptlz::vector_2d_to_ndarray(pos, d), sptlz::vector_1d_to_ndarray(val),
                                         sptlz::vector_2d_to_ndarray(loc, d), sptlz::vector_1d_to_ndarray(p));
        return(sptlz::ndarray_to_vector_1d(&res));
      };
    }else if (method == Method::LOO){
      py::object f = callable("loo", true);
      _loo = [f, d](std::vector<std::vector<float>>* pos, std::vector<float>* val, std::vector<float>* p){
        auto res = (py::array_t<float>) f(sptlz::vector_2d_to_ndarray(pos, d), sptlz::vector_1d_to_ndarray(val),
                                         sptlz::vector_1d_to_ndarray(p));
        return(sptlz::ndarray_to_vector_1d(&res));
      };
    }else{
      py::object f = callable("kfold", true);
      _kfold = [f, d](int k, std::vector<std::vector<float>>* pos, std::vector<float>* val, std::vector<int>* fld, std::vector<float>* p){
        auto res = (py::array_t<float>) f(k, sptlz::vector_2d_to_ndarray(pos, d), sptlz::vector_1d_to_ndarray(val),
                                         sptlz::vector_1d_to_ndarray(fld), sptlz::vector_1d_to_ndarray(p));
        return(sptlz::ndarray_to_vector_1d(&res));
      };
    }
    return(new sptlz::CustomDecoder(_post, _est, _loo, _kfold));
  }

  // ---------------------------------------------------------------------------------------------
  // the catalogue

  inline const std::vector<PartitionSpec> &partitions(){
    static const std::vector<PartitionSpec> specs = {
      {"mondrian",
       "Spatialize's Mondrian partition: the root box is always split, the cut axis drawn uniformly.",
       "normalised granularity in [0, 1): lifetime lambda = 1/(mu(H)(1 - alpha)), mu(H) the sum of the box's sides",
       1, ANY,
       [](std::vector<std::vector<float>> &smp, std::vector<float> &val, std::vector<std::vector<float>> &bbox,
          float alpha, int forest_size, int seed, std::function<int(std::string)> visitor)->sptlz::Ensemble*{
         float lambda = sptlz::bbox_sum_interval(bbox);
         lambda = 1/(lambda-alpha*lambda);
         return(new sptlz::ESI(smp, val, lambda, forest_size, bbox, visitor, seed, false));
       }},
      {"mondrian-raw",
       "The theory's Mondrian process: the root draws its split time Exp(mu(H)), the cut axis is chosen with probability proportional to its side.",
       "as for 'mondrian'",
       1, ANY,
       [](std::vector<std::vector<float>> &smp, std::vector<float> &val, std::vector<std::vector<float>> &bbox,
          float alpha, int forest_size, int seed, std::function<int(std::string)> visitor)->sptlz::Ensemble*{
         float lambda = sptlz::bbox_sum_interval(bbox);
         lambda = 1/(lambda-alpha*lambda);
         return(new sptlz::ESI(smp, val, lambda, forest_size, bbox, visitor, seed, true));
       }},
      {"voronoi",
       "Voronoi partition with max(1, Poisson(0.5 n |alpha|)) nuclei, at most n.",
       "nuclei rate in (-1, 1): nuclei among the samples for alpha >= 0, uniform in the box for alpha < 0",
       1, ANY,
       [](std::vector<std::vector<float>> &smp, std::vector<float> &val, std::vector<std::vector<float>> &bbox,
          float alpha, int forest_size, int seed, std::function<int(std::string)> visitor)->sptlz::Ensemble*{
         return(new sptlz::VORONOI(smp, val, alpha, forest_size, bbox, visitor, seed));
       }},
    };
    return(specs);
  }

  inline const std::vector<DecoderSpec> &decoders(){
    // never destroyed: its defaults are Python objects, which must not be released after the
    // interpreter has finalised (that crashed the process at exit)
    static std::vector<DecoderSpec> &specs = *new std::vector<DecoderSpec>();
    if (!specs.empty()) return(specs);
    specs.push_back({"idw", "Inverse distance weighting, weights 1/d^p (a datum at distance 0 takes all the weight).",
       1, ANY, true,
       {{"exponent", "float", "the power p of the distance", true, py::none(), {}}},
       nullptr});
    specs.push_back({"kriging", "Ordinary kriging with a fixed variogram model.",
       1, ANY, true,
       {{"model", "choice", "variogram model", true, py::none(), {"spherical", "exponential", "cubic", "gaussian"}},
        {"nugget", "float", "nugget", true, py::none(), {}},
        {"range", "float", "range", true, py::none(), {}},
        {"sill", "float", "sill", true, py::none(), {}}},
       nullptr});
    specs.push_back({"adaptiveidw", "IDW whose exponent and anisotropy are fitted in each cell.",
       2, 3, true,
       {{"metric", "choice", "error minimised when fitting each cell", false, py::str("mae"), {"mae", "mse"}}},
       nullptr});
    specs.push_back({"custom", "A decoder given by Python callables on the samples of each cell.",
       1, ANY, false,
       {{"post_creation", "callable", "post_creation(coords, values) -> cell parameters", false, py::none(), {}},
        {"estimation", "callable", "estimation(coords, values, queries, params) -> predictions (method 'estimate')", false, py::none(), {}},
        {"loo", "callable", "loo(coords, values, params) -> predictions (method 'loo')", false, py::none(), {}},
        {"kfold", "callable", "kfold(k, coords, values, folds, params) -> predictions (method 'kfold')", false, py::none(), {}}},
       nullptr});
    // factories, which read the parameters through their own spec
    specs[0].make = [](int d, const py::dict &p, Method m)->sptlz::Decoder*{
      return(new sptlz::IDWDecoder(param<float>(decoders()[0], p, "exponent")));
    };
    specs[1].make = [](int d, const py::dict &p, Method m)->sptlz::Decoder*{
      const DecoderSpec &s = decoders()[1];
      return(new sptlz::KrigingDecoder(param<int>(s, p, "model"), param<float>(s, p, "nugget"),
                                       param<float>(s, p, "range"), param<float>(s, p, "sill")));
    };
    specs[2].make = [](int d, const py::dict &p, Method m)->sptlz::Decoder*{
      const DecoderSpec &s = decoders()[2];
      return(new sptlz::AdaptiveIDWDecoder(d, s.params[0].choices[param<int>(s, p, "metric")-1]));
    };
    specs[3].make = [](int d, const py::dict &p, Method m)->sptlz::Decoder*{
      return(custom_decoder(decoders()[3], d, p, m));
    };
    return(specs);
  }

  template <typename Spec>
  const Spec &find(const std::vector<Spec> &specs, const std::string &name, const char *what){
    std::string names;
    for (auto &s: specs){
      if (s.name == name) return(s);
      names += (names.empty() ? "'" : ", '") + s.name + "'";
    }
    throw std::runtime_error(std::string("unknown ") + what + " '" + name + "' (expected " + names + ")");
  }

  inline void check_dim(const std::string &what, const std::string &name, int min_dim, int max_dim, int d){
    if (d < min_dim || (max_dim != ANY && d > max_dim)){
      std::string range = max_dim == ANY ? std::to_string(min_dim) + " or more" :
                          (min_dim == max_dim ? std::to_string(min_dim) : std::to_string(min_dim) + " to " + std::to_string(max_dim));
      throw std::runtime_error(what + " '" + name + "' is available for " + range + " dimensions, not " + std::to_string(d));
    }
  }

  inline py::object dims(int min_dim, int max_dim){
    return(py::make_tuple(min_dim, max_dim == ANY ? py::object(py::none()) : py::object(py::int_(max_dim))));
  }

  // the catalogue as plain Python data
  inline py::dict catalog(){
    py::list parts, decs;
    for (auto &s: partitions()){
      py::dict e;
      e["name"] = s.name; e["doc"] = s.doc; e["alpha"] = s.alpha; e["dims"] = dims(s.min_dim, s.max_dim);
      parts.append(e);
    }
    for (auto &s: decoders()){
      py::list ps;
      for (auto &p: s.params){
        py::dict q;
        q["name"] = p.name; q["type"] = p.type; q["doc"] = p.doc; q["required"] = p.required;
        q["default"] = p.fallback; q["choices"] = py::cast(p.choices);
        ps.append(q);
      }
      py::dict e;
      e["name"] = s.name; e["doc"] = s.doc; e["dims"] = dims(s.min_dim, s.max_dim);
      e["thread_safe"] = s.thread_safe; e["params"] = ps;
      decs.append(e);
    }
    py::dict out;
    out["partitions"] = parts;
    out["decoders"] = decs;
    out["methods"] = py::make_tuple("estimate", "loo", "kfold");
    return(out);
  }
}

#endif

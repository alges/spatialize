#ifndef _SPTLZ_NN_COKRIGING_
#define _SPTLZ_NN_COKRIGING_

#include <vector>
#include <array>
#include <limits>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <algorithm>
// the systems of a neighbourhood are small: Eigen must not parallelise inside one (see decoders/kriging.hpp)
#ifndef EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_PARALLELIZE
#endif
#include <Eigen/Dense>
#include "spatialize/kdtree.hpp"
#include "spatialize/utils.hpp"
#include "spatialize/callback_logging.hpp"

namespace sptlz{

  // Kriging of one structure of the variogram (a covariance, see CokrigingStructure::covariance)
  enum CokrigingModel {COKRIGING_SPHERICAL = 0, COKRIGING_EXPONENTIAL = 1, COKRIGING_CUBIC = 2, COKRIGING_GAUSSIAN = 3};

  // One nested structure of the linear model of coregionalization: a model, the (M x M, row-major)
  // matrix of partial sills, and the ranges and the angles (degrees) of its anisotropy ellipsoid.
  struct CokrigingStructure {
    int type = COKRIGING_SPHERICAL;
    std::vector<double> sill;
    std::array<double, 3> ranges = {1.0, 1.0, 1.0};
    std::array<double, 3> angles = {0.0, 0.0, 0.0};
  };

  struct CokrigingParams {
    bool ordinary = true;
    std::vector<double> mean;                  // simple kriging: one per variable
    double limit_min = std::numeric_limits<double>::lowest();  // values out of [min, max] are missing
    double limit_max = std::numeric_limits<double>::max();
    std::vector<double> top_cut;               // one per variable, values over it are set to it
    std::array<double, 3> nb_radius = {1.0, 1.0, 1.0};  // semi-axes of the search ellipsoid
    std::array<double, 3> nb_angles = {0.0, 0.0, 0.0};
    int nb_min = 1, nb_max = 16;               // neighbours needed / used (per variable, per octant)
    bool nb_octant = false;
    double missing_value = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> nugget;                // M x M, row-major
    std::vector<CokrigingStructure> structures;
    std::array<int, 3> discretization = {1, 1, 1};  // points per axis of the block (1: punctual)
    std::array<double, 3> block_size = {1.0, 1.0, 1.0};
  };

  struct CokrigingOutput {
    int n_targets = 0, n_vars = 0;
    std::vector<double> prediction;  // n_targets x n_vars, row-major
    std::vector<double> variance;
    std::vector<double> neighbors;   // number used per variable; -1 where the system failed
  };

  // Ordinary or simple (co)kriging of M variables with a search neighbourhood given by an ellipsoid:
  // like NN_IDW, the data of a location are the ones a kd-tree finds within the search radius.
  // Coordinates are always 3D (pad with zeros). Computes in double.
  class NN_Cokriging {
    protected:
      std::string class_name;
      std::function<int(std::string)> callback_visitor;
      int n_samples, n_vars;
      CokrigingParams p;
      std::vector<std::array<double, 3>> coords;
      std::vector<double> values;   // n_samples x n_vars, row-major, missing as NaN
      std::vector<int> dhid;        // drillhole of every sample, may be empty
      Eigen::Matrix3d search_t;     // search ellipsoid -> unit ball
      std::vector<Eigen::Matrix3d> vario_t;
      std::vector<std::array<double, 3>> block_grid;
      sptlz::KDTree<float> *kdt = NULL;

      static Eigen::Matrix3d transformation(const std::array<double, 3> &ranges, const std::array<double, 3> &angles){
        const double deg2rad = M_PI/180.0;
        double a = (90.0-angles[0])*deg2rad, b = -angles[1]*deg2rad, c = angles[2]*deg2rad;
        double ca = std::cos(a), sa = std::sin(a), cb = std::cos(b), sb = std::sin(b), cc = std::cos(c), sc = std::sin(c);
        Eigen::Matrix3d rotation;
        rotation << cb*ca,                 cb*sa,                 -sb,
                    -cc*sa + sc*sb*ca,     cc*ca + sc*sb*sa,      sc*cb,
                    sc*sa + cc*sb*ca,      -sc*ca + cc*sb*sa,     cc*cb;
        Eigen::Matrix3d reduction = Eigen::Matrix3d::Zero();
        for(int i=0; i<3; i++){
          reduction(i, i) = 1.0/ranges[i];
        }
        return(reduction*rotation);  // applied to a column point
      }

      static Eigen::Vector3d apply(const Eigen::Matrix3d &t, const double *x){
        return(t*Eigen::Vector3d(x[0], x[1], x[2]));
      }

      // the covariance of a model, h being the distance in units of the range
      static double structure_covariance(int type, double h){
        double hp = std::min(h, 1.0);
        switch(type){
          case COKRIGING_SPHERICAL:   return(1.0 - 1.5*hp + 0.5*std::pow(hp, 3.0));
          case COKRIGING_EXPONENTIAL: return(std::exp(-3.0*h));
          case COKRIGING_CUBIC:       return(1.0 - 7.0*std::pow(hp, 2.0) + (35.0/4.0)*std::pow(hp, 3.0) - 3.5*std::pow(hp, 5.0) + 0.75*std::pow(hp, 7.0));
          case COKRIGING_GAUSSIAN:    return(std::exp(-3.0*h*h));
        }
        return(1.0);
      }

      void validate(){
        const size_t mm = static_cast<size_t>(n_vars)*n_vars;
        if(p.ordinary == false && static_cast<int>(p.mean.size()) != n_vars)
          throw std::runtime_error("'kriging_mean' must have one value per variable");
        if(static_cast<int>(p.top_cut.size()) != n_vars)
          throw std::runtime_error("'top_cut' must have one value per variable");
        if(p.nugget.size() != mm)
          throw std::runtime_error("the nugget of the variogram must be a matrix of variables x variables");
        if(p.structures.empty())
          throw std::runtime_error("the variogram needs at least one structure");
        for(auto &s: p.structures){
          if(s.sill.size() != mm)
            throw std::runtime_error("the sill of a structure must be a matrix of variables x variables");
          for(int i=0; i<3; i++){
            if(!(s.ranges[i] > 0.0))
              throw std::runtime_error("the ranges of a structure must be positive");
          }
        }
        for(int i=0; i<3; i++){
          if(!(p.nb_radius[i] > 0.0))
            throw std::runtime_error("the search radii must be positive");
          if(p.discretization[i] < 1)
            throw std::runtime_error("'discretization' must be positive");
        }
      }

      void filter_data(){
        for(int i=0; i<n_samples; i++){
          for(int j=0; j<n_vars; j++){
            double &v = values[static_cast<size_t>(i)*n_vars+j];
            if(!std::isnan(p.missing_value) && v == p.missing_value) v = std::nan("");
            if(v < p.limit_min || p.limit_max < v) v = std::nan("");
            if(v > p.top_cut[j]) v = p.top_cut[j];
          }
        }
      }

      void make_block_grid(){
        auto lin = [this](int axis){
          double h = (p.block_size[axis]/2.0)*(1.0 + 1.0/p.discretization[axis]);
          int n = p.discretization[axis];
          std::vector<double> r;
          for(int i=1; i<=n; i++){
            r.push_back(-h + 2.0*h*i/(n+1));
          }
          return(r);
        };
        auto dx = lin(0), dy = lin(1), dz = lin(2);
        for(double z: dz) for(double y: dy) for(double x: dx){
          block_grid.push_back({x, y, z});
        }
      }

      // The data of a location, per variable: the nearest ones inside the search ellipsoid that are
      // not missing and are not excluded, at most nb_max (per octant when octants are used).
      std::vector<std::vector<int>> neighbors(const double *x, const std::function<bool(int)> *excluded){
        std::vector<std::vector<int>> result(n_vars);
        Eigen::Vector3d q = apply(search_t, x);
        std::vector<float> qf = {static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2])};
        auto nbs = kdt->query_ball(&qf, 1.0f, 2.0f);  // sorted by distance
        std::vector<std::array<int, 8>> per_octant(n_vars);
        for(auto &a: per_octant) a.fill(0);

        for(int idx: nbs.second){
          if(excluded != NULL && (*excluded)(idx)) continue;
          int oct = 0;
          if(p.nb_octant){
            Eigen::Vector3d d = apply(search_t, coords[idx].data()) - q;
            oct = (d[0]<0 ? 0 : 1) + (d[1]<0 ? 0 : 2) + (d[2]<0 ? 0 : 4);
          }
          for(int j=0; j<n_vars; j++){
            if(std::isnan(values[static_cast<size_t>(idx)*n_vars+j])) continue;
            if(p.nb_octant){
              if(per_octant[j][oct] >= p.nb_max) continue;
              per_octant[j][oct]++;
              result[j].push_back(idx);
            }else if(static_cast<int>(result[j].size()) < p.nb_max){
              result[j].push_back(idx);
            }
          }
        }
        return(result);
      }

      // Kriging of one location (or block, around x). Returns false when the system is singular.
      bool krige(const double *x, const std::vector<std::vector<int>> &nbs, double *pred, double *var){
        const int D = static_cast<int>(block_grid.size());
        int n = 0;
        for(auto &v: nbs) n += static_cast<int>(v.size());

        // points: the neighbours (variable by variable), then the block points for each variable
        const int P = n + n_vars*D;
        std::vector<std::array<double, 3>> pts;
        std::vector<int> var_of;
        pts.reserve(P);
        var_of.reserve(P);
        std::vector<int> nb_all;
        for(int j=0; j<n_vars; j++){
          for(int idx: nbs[j]){
            pts.push_back(coords[idx]);
            var_of.push_back(j);
            nb_all.push_back(idx);
          }
        }
        for(int j=0; j<n_vars; j++){
          for(int d=0; d<D; d++){
            pts.push_back({x[0]+block_grid[d][0], x[1]+block_grid[d][1], x[2]+block_grid[d][2]});
            var_of.push_back(j);
          }
        }

        // the covariance of all the points
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(P, P);
        for(int i=0; i<P; i++){
          for(int j=0; j<=i; j++){
            double c = 0.0;
            if(pts[i] == pts[j]){
              c = p.nugget[static_cast<size_t>(var_of[i])*n_vars+var_of[j]];
            }
            for(size_t s=0; s<p.structures.size(); s++){
              Eigen::Vector3d d = apply(vario_t[s], pts[i].data()) - apply(vario_t[s], pts[j].data());
              c += p.structures[s].sill[static_cast<size_t>(var_of[i])*n_vars+var_of[j]]*structure_covariance(p.structures[s].type, d.norm());
            }
            C(i, j) = c;
            C(j, i) = c;
          }
        }
        Eigen::MatrixXd K = C.topLeftCorner(n, n);
        Eigen::MatrixXd K0(n, n_vars), K1(n_vars, n_vars);
        if(D > 1){  // block kriging: average over the discretization points
          for(int j=0; j<n_vars; j++){
            for(int i=0; i<n; i++){
              K0(i, j) = C.block(i, n+j*D, 1, D).mean();
            }
            for(int i=0; i<n_vars; i++){
              K1(i, j) = C.block(n+i*D, n+j*D, D, D).mean();
            }
          }
        }else{
          K0 = C.block(0, n, n, n_vars);
          K1 = C.block(n, n, n_vars, n_vars);
        }

        Eigen::MatrixXd W;
        if(p.ordinary){
          // the extended system, with a Lagrange multiplier per variable
          Eigen::MatrixXd Ke = Eigen::MatrixXd::Zero(n+n_vars, n+n_vars), Rhs = Eigen::MatrixXd::Zero(n+n_vars, n_vars);
          Ke.topLeftCorner(n, n) = K;
          int pos = 0;
          for(int j=0; j<n_vars; j++){
            for(size_t i=0; i<nbs[j].size(); i++){
              Ke(pos, n+j) = 1.0;
              Ke(n+j, pos) = 1.0;
              pos++;
            }
          }
          Rhs.topRows(n) = K0;
          Rhs.bottomRows(n_vars) = Eigen::MatrixXd::Identity(n_vars, n_vars);
          Eigen::FullPivLU<Eigen::MatrixXd> lu(Ke);
          if(!lu.isInvertible()) return(false);
          W = lu.solve(Rhs).topRows(n);
        }else{
          Eigen::FullPivLU<Eigen::MatrixXd> lu(K);
          if(!lu.isInvertible()) return(false);
          W = lu.solve(K0);
        }
        if(!W.allFinite()) return(false);

        Eigen::MatrixXd sigma = K1 - W.transpose()*K0 - K0.transpose()*W + W.transpose()*K*W;
        // simple kriging estimates the residual of the known means
        Eigen::VectorXd z(n);
        int pos = 0;
        for(int j=0; j<n_vars; j++){
          for(int idx: nbs[j]){
            z[pos] = values[static_cast<size_t>(idx)*n_vars+j] - (p.ordinary ? 0.0 : p.mean[j]);
            pos++;
          }
        }
        Eigen::VectorXd est = W.transpose()*z;
        for(int j=0; j<n_vars; j++){
          pred[j] = est[j] + (p.ordinary ? 0.0 : p.mean[j]);
          var[j] = sigma(j, j);
        }
        return(true);
      }

      // The estimate at `n` locations (row-major, 3 columns); `excluded(i, idx)` tells, for the
      // location i, the data that must not be used (leave-one-out, k-fold).
      CokrigingOutput estimate_at(const std::vector<std::array<double, 3>> &locations,
                                  const std::function<bool(int, int)> *excluded){
        const int n = static_cast<int>(locations.size());
        CokrigingOutput out;
        out.n_targets = n;
        out.n_vars = n_vars;
        out.prediction.assign(static_cast<size_t>(n)*n_vars, p.missing_value);
        out.variance.assign(static_cast<size_t>(n)*n_vars, p.missing_value);
        out.neighbors.assign(static_cast<size_t>(n)*n_vars, 0.0);

        sptlz::CallbackLogger logger(this->callback_visitor, this->class_name);
        sptlz::CallbackProgressSender progress(this->callback_visitor);
        logger.debug("computing estimates");
        progress.init(n, 1);

        for(int i=0; i<n; i++){
          if(PyErr_CheckSignals() != 0)  // to allow ctrl-c from user
            throw pybind11::error_already_set();
          progress.inform(static_cast<int>(100.0*(i+1.0)/n));

          std::function<bool(int)> ex;
          if(excluded != NULL){
            ex = [excluded, i](int idx){return((*excluded)(i, idx));};
          }
          auto nbs = neighbors(locations[i].data(), excluded != NULL ? &ex : NULL);
          size_t total = 0;
          for(int j=0; j<n_vars; j++){
            out.neighbors[static_cast<size_t>(i)*n_vars+j] = static_cast<double>(nbs[j].size());
            total += nbs[j].size();
          }
          // too few neighbours: the location stays missing
          if(total == 0 || static_cast<int>(total) < p.nb_min) continue;
          // a variable without data would leave an empty row in the (ordinary) system
          if(p.ordinary){
            bool empty = false;
            for(int j=0; j<n_vars; j++) empty = empty || nbs[j].empty();
            if(empty){
              for(int j=0; j<n_vars; j++) out.neighbors[static_cast<size_t>(i)*n_vars+j] = -1.0;
              continue;
            }
          }
          if(!krige(locations[i].data(), nbs, &out.prediction[static_cast<size_t>(i)*n_vars], &out.variance[static_cast<size_t>(i)*n_vars])){
            for(int j=0; j<n_vars; j++) out.neighbors[static_cast<size_t>(i)*n_vars+j] = -1.0;
          }
        }
        progress.stop();
        return(out);
      }

    public:
      // coords: n x 3, values: n x n_vars (row-major); dhid: the drillhole of every sample, or empty
      NN_Cokriging(std::vector<std::array<double, 3>> _coords, std::vector<double> _values, int _n_vars,
                   CokrigingParams _params, std::vector<int> _dhid, std::function<int(std::string)> visitor){
        class_name = "NN_Cokriging";
        callback_visitor = visitor;
        n_samples = static_cast<int>(_coords.size());
        n_vars = _n_vars;
        if(n_samples == 0) throw std::runtime_error("no samples");
        if(n_vars < 1) throw std::runtime_error("no variables");
        if(_values.size() != static_cast<size_t>(n_samples)*n_vars)
          throw std::runtime_error("'samples_coords' and 'samples_values' do not have the same length");
        if(!_dhid.empty() && _dhid.size() != static_cast<size_t>(n_samples))
          throw std::runtime_error("'samples_dhid' and 'samples_coords' do not have the same length");
        coords = std::move(_coords);
        values = std::move(_values);
        dhid = std::move(_dhid);
        p = std::move(_params);
        validate();
        filter_data();

        search_t = transformation(p.nb_radius, p.nb_angles);
        for(auto &s: p.structures){
          vario_t.push_back(transformation(s.ranges, s.angles));
        }
        make_block_grid();

        std::vector<std::vector<float>> tc;
        for(auto &c: coords){
          Eigen::Vector3d q = apply(search_t, c.data());
          tc.push_back({static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2])});
        }
        kdt = new sptlz::KDTree<float>(&tc);
      }

      ~NN_Cokriging(){
        if(kdt != NULL) delete(kdt);
      }

      NN_Cokriging(const NN_Cokriging &) = delete;
      NN_Cokriging &operator=(const NN_Cokriging &) = delete;

      CokrigingOutput estimate(const std::vector<std::array<double, 3>> &locations){
        return(estimate_at(locations, NULL));
      }

      // a regular grid of blocks: x varies fastest; the nodes are the centers of the blocks
      CokrigingOutput estimate_grid(const std::array<double, 3> &origin, const std::array<int, 3> &nodes){
        std::vector<std::array<double, 3>> locations;
        for(int k=0; k<nodes[2]; k++) for(int j=0; j<nodes[1]; j++) for(int i=0; i<nodes[0]; i++){
          locations.push_back({origin[0] + (i+0.5)*p.block_size[0], origin[1] + (j+0.5)*p.block_size[1], origin[2] + (k+0.5)*p.block_size[2]});
        }
        return(estimate_at(locations, NULL));
      }

      // every sample estimated without itself
      CokrigingOutput leave_one_out(){
        std::function<bool(int, int)> ex = [](int i, int idx){return(i == idx);};
        return(estimate_at(coords, &ex));
      }

      // every sample estimated without the ones of its drillhole
      CokrigingOutput leave_one_drillhole_out(){
        if(dhid.empty()) throw std::runtime_error("'samples_dhid' must be given for 'leave_one_out_drillhole'");
        std::function<bool(int, int)> ex = [this](int i, int idx){return(dhid[i] == dhid[idx]);};
        return(estimate_at(coords, &ex));
      }

      // every sample estimated without the ones of its fold
      CokrigingOutput k_fold(int k, int seed=206936){
        auto folds = get_folds(n_samples, k, static_cast<unsigned int>(seed));
        std::function<bool(int, int)> ex = [&folds](int i, int idx){return(folds[i] == folds[idx]);};
        return(estimate_at(coords, &ex));
      }
  };
}

#endif

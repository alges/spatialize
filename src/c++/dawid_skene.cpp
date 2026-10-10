// Dawid-Skene EM aggregation of categorical ensembles (Bayesian truth discovery): the partitions
// of an ensemble are treated as annotators of unequal reliability, each with its own confusion
// matrix, and the most probable true category of each location is inferred by EM. A spatial variant
// penalises forbidden adjacencies between categories on a grid. Brought into Spatialize from the
// categorical mini-project (2026-10-10): messages and progress go through Spatialize's visitor
// protocol (Python's warnings module without a visitor), Ctrl-C is checked between iterations, the
// random initialisation is seeded (std::mt19937), the E-step, the log-likelihood and the M-step run
// under OpenMP with sums in a fixed order (results identical for any number of threads), and the
// module is spatialize.gs.cat_esi._dawid_skene.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <functional>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "spatialize/callback_logging.hpp"

namespace py = pybind11;

// A warning through Python's warnings module (an error when the warning filters say so)
static void ds_warn(const std::string &msg) {
    if (PyErr_WarnEx(PyExc_UserWarning, msg.c_str(), 1) < 0) {
        throw py::error_already_set();
    }
}

// --- Helper Functions for Numerical Stability and Math ---

// Numerically stable logsumexp
double logsumexp(const std::vector<double>& log_probs) {
    if (log_probs.empty()) {
        return -std::numeric_limits<double>::infinity();
    }
    double max_log_prob = log_probs[0];
    for (double lp : log_probs) {
        if (lp > max_log_prob) {
            max_log_prob = lp;
        }
    }
    double sum_exp = 0.0;
    for (double lp : log_probs) {
        sum_exp += std::exp(lp - max_log_prob);
    }
    return max_log_prob + std::log(sum_exp);
}

// Clamp probabilities to avoid log(0) and division by zero
const double EPSILON = 1e-300; // Smallest positive double

// --- Base Dawid-Skene EM Implementation ---

class DawidSkeneEM {
protected:
    // Model parameters
    std::vector<double> pi_; // K-vector
    std::vector<std::vector<std::vector<double>>> theta_; // J x K x K tensor

    // Generator of the random initialisation (seeded: results reproducible for a given seed)
    std::mt19937 rng_{0u};

    // Spatialize's visitor (JSON messages and progress), and the number of threads (0: OpenMP's default)
    std::function<int(std::string)> visitor_;
    int num_threads_ = 0;

    // The observations of each item, (source, label), and of each source, (item, label), in the order
    // of the claims: built once per fit, they keep every sum in the serial order under OpenMP
    std::vector<std::vector<std::pair<int, int>>> obs_by_item_;
    std::vector<std::vector<std::pair<int, int>>> obs_by_source_;

    void build_observation_index() {
        obs_by_item_.assign(num_items_, {});
        obs_by_source_.assign(num_sources_, {});
        for (size_t n = 0; n < item_indices_obs_.size(); ++n) {
            obs_by_item_[item_indices_obs_[n]].push_back({annotator_indices_obs_[n], observed_labels_obs_[n]});
            obs_by_source_[annotator_indices_obs_[n]].push_back({item_indices_obs_[n], observed_labels_obs_[n]});
        }
    }

    int threads() const {
        #ifdef _OPENMP
        return num_threads_ > 0 ? num_threads_ : omp_get_max_threads();
        #else
        return 1;
        #endif
    }

    // A warning: through the visitor when there is one, else Python's warnings module
    void ds_warn(const std::string &msg) {
        if (visitor_) {
            std::string clean = msg.rfind("Warning: ", 0) == 0 ? msg.substr(9) : msg;
            for (auto &c : clean) if (c == '"' || c == '\\') c = '\'';
            sptlz::CallbackLogger(visitor_, "dawid-skene").warning(clean);
        } else if (PyErr_WarnEx(PyExc_UserWarning, msg.c_str(), 1) < 0) {
            throw py::error_already_set();
        }
    }

    // Dimensions
    int num_items_;
    int num_sources_;
    int num_categories_;

    // Mappings
    std::map<std::string, int> source_to_idx_;
    std::map<std::string, int> statement_to_idx_;
    std::map<std::string, int> category_to_idx_;
    std::vector<std::string> idx_to_category_;
    std::vector<std::string> idx_to_statement_id_;
    std::vector<std::string> idx_to_source_id_;

    // Observed data (0-based indices)
    std::vector<int> item_indices_obs_;
    std::vector<int> annotator_indices_obs_;
    std::vector<int> observed_labels_obs_;

    // Configuration
    double tolerance_; 
    int max_iter_;     
    std::string category_type_; 
    std::vector<std::string> user_provided_categories_raw_; 
    std::map<int, int> ordinal_category_orders_; 

public:
    void set_seed(unsigned int seed) { rng_.seed(seed); }
    void set_visitor(std::function<int(std::string)> visitor) { visitor_ = visitor; }
    void set_num_threads(int n) { num_threads_ = n; }

    // Constructor
    DawidSkeneEM(py::object categories_obj, double tolerance, int max_iter, const std::string& category_type_str, py::object ordinal_order_map_obj)
        : tolerance_(tolerance), 
          max_iter_(max_iter),   
          category_type_(category_type_str) { 
        if (!categories_obj.is_none()) {
            user_provided_categories_raw_ = categories_obj.cast<std::vector<std::string>>();
        }
    }

    // Add virtual destructor to base class
    virtual ~DawidSkeneEM() = default;

    // Main initialization method (virtual for spatial extension)
    virtual void initialize_all_data(
        const std::vector<std::tuple<std::string, std::string, std::string>>& claims,
        py::object map_dimensions_obj = py::none(), 
        py::dict item_coordinates_obj = py::dict(), // Still needed here in C++
        const std::vector<std::tuple<std::string, std::string>>& spatial_constraints_py = {}, 
        py::object ordinal_order_map_obj = py::none() 
    ) {
        std::set<std::string> all_sources_set;
        std::set<std::string> all_statements_set;
        std::set<std::string> all_categories_in_data_set;

        for (const auto& claim : claims) {
            all_sources_set.insert(std::get<0>(claim));
            all_statements_set.insert(std::get<1>(claim));
            all_categories_in_data_set.insert(std::get<2>(claim));
        }

        std::vector<std::string> effective_categories = user_provided_categories_raw_;
        if (effective_categories.empty()) {
            effective_categories.assign(all_categories_in_data_set.begin(), all_categories_in_data_set.end());
        } else {
            for (const auto& cat : all_categories_in_data_set) {
                if (std::find(effective_categories.begin(), effective_categories.end(), cat) == effective_categories.end()) {
                    { std::ostringstream w_; w_ << "Warning: Category '" << cat << "' found in data but not in provided categories. It will be added."; ds_warn(w_.str()); }
                    effective_categories.push_back(cat);
                }
            }
        }
        std::sort(effective_categories.begin(), effective_categories.end());
        idx_to_category_ = effective_categories;

        idx_to_source_id_.assign(all_sources_set.begin(), all_sources_set.end());
        std::sort(idx_to_source_id_.begin(), idx_to_source_id_.end());
        for(int i = 0; i < idx_to_source_id_.size(); ++i) {
             source_to_idx_[idx_to_source_id_[i]] = i;
        }

        idx_to_statement_id_.assign(all_statements_set.begin(), all_statements_set.end());
        std::sort(idx_to_statement_id_.begin(), idx_to_statement_id_.end());
        for(int i = 0; i < idx_to_statement_id_.size(); ++i) {
            statement_to_idx_[idx_to_statement_id_[i]] = i;
        }

        for (int i = 0; i < idx_to_category_.size(); ++i) {
            category_to_idx_[idx_to_category_[i]] = i;
        }

        num_sources_ = source_to_idx_.size();
        num_items_ = statement_to_idx_.size();
        num_categories_ = idx_to_category_.size();

        if (num_categories_ == 0 || num_sources_ == 0 || num_items_ == 0) {
            { std::ostringstream w_; w_ << "Warning: No valid claims or categories to initialize with. Model will not fit."; ds_warn(w_.str()); }
            return;
        }

        ordinal_category_orders_.clear();
        if (category_type_ == "ordinal") {
            if (!ordinal_order_map_obj.is_none()) {
                py::dict py_ordinal_map = ordinal_order_map_obj.cast<py::dict>();
                for (auto const& [cat_str_obj, order_val_obj] : py_ordinal_map) {
                    std::string cat_str = cat_str_obj.cast<std::string>();
                    int order_val = order_val_obj.cast<int>();
                    if (category_to_idx_.count(cat_str)) {
                        ordinal_category_orders_[category_to_idx_.at(cat_str)] = order_val;
                    } else {
                        { std::ostringstream w_; w_ << "Warning: Ordinal category '" << cat_str << "' in order map not found in actual categories. It will be ignored."; ds_warn(w_.str()); }
                    }
                }
                if (ordinal_category_orders_.size() != num_categories_) {
                    { std::ostringstream w_; w_ << "Warning: Not all categories have an explicit order defined for ordinal type. Treating as nominal."; ds_warn(w_.str()); }
                    category_type_ = "nominal"; 
                }
            } else {
                { std::ostringstream w_; w_ << "Warning: category_type is 'ordinal' but no ordinal_order_map provided. Treating as nominal."; ds_warn(w_.str()); }
                category_type_ = "nominal"; 
            }
        }

        item_indices_obs_.clear();
        annotator_indices_obs_.clear();
        observed_labels_obs_.clear();

        for (const auto& claim : claims) {
            int s_idx = source_to_idx_.at(std::get<0>(claim));
            int st_idx = statement_to_idx_.at(std::get<1>(claim));
            int cat_idx = category_to_idx_.at(std::get<2>(claim));

            item_indices_obs_.push_back(st_idx);
            annotator_indices_obs_.push_back(s_idx);
            observed_labels_obs_.push_back(cat_idx);
        }
    }

    void initialize_parameters_random() {
        pi_.assign(num_categories_, 1.0 / num_categories_);

        theta_.assign(num_sources_, std::vector<std::vector<double>>(num_categories_, std::vector<double>(num_categories_)));
        for (int j = 0; j < num_sources_; ++j) {
            for (int k_true = 0; k_true < num_categories_; ++k_true) {
                double sum_prob = 0.0;
                for (int k_obs = 0; k_obs < num_categories_; ++k_obs) {
                    if (k_true == k_obs) {
                        theta_[j][k_true][k_obs] = 0.7 + std::uniform_real_distribution<double>(0.0, 1.0)(rng_) * 0.1;
                    } else {
                        theta_[j][k_true][k_obs] = 0.1 / (num_categories_ - 1);
                    }
                    theta_[j][k_true][k_obs] = std::max(theta_[j][k_true][k_obs], EPSILON); 
                    sum_prob += theta_[j][k_true][k_obs];
                }
                for (int k_obs = 0; k_obs < num_categories_; ++k_obs) {
                    theta_[j][k_true][k_obs] /= sum_prob;
                }
            }
        }
    }

    // E-Step (virtual for spatial extension)
    virtual std::vector<std::vector<double>> e_step_impl() {
        std::vector<std::vector<double>> responsibilities(num_items_, std::vector<double>(num_categories_));

        // items are independent: one thread per block of items, each writing its own row
        #pragma omp parallel for schedule(static) num_threads(threads())
        for (int i = 0; i < num_items_; ++i) {
            std::vector<double> unnorm_log_probs_for_item_i(num_categories_);

            for (int k_true = 0; k_true < num_categories_; ++k_true) {
                double log_prob_z_prior = std::log(std::max(pi_[k_true], EPSILON));

                double log_likelihood_obs_given_z = 0.0;
                {
                    for (const auto& obs : obs_by_item_[i]) {
                        int j = obs.first;  
                        int k_obs = obs.second; 
                        log_likelihood_obs_given_z += std::log(std::max(theta_[j][k_true][k_obs], EPSILON));
                    }
                }
                
                unnorm_log_probs_for_item_i[k_true] = log_prob_z_prior + log_likelihood_obs_given_z;
            }
            
            double log_norm_const = logsumexp(unnorm_log_probs_for_item_i);
            for (int k_true = 0; k_true < num_categories_; ++k_true) {
                responsibilities[i][k_true] = std::exp(unnorm_log_probs_for_item_i[k_true] - log_norm_const);
            }
        }
        return responsibilities;
    }

    // M-Step: Update pi and theta using responsibilities
    void m_step(const std::vector<std::vector<double>>& responsibilities) {
        std::vector<double> new_pi(num_categories_, 0.0);
        for (int i = 0; i < num_items_; ++i) {
            for (int k = 0; k < num_categories_; ++k) {
                new_pi[k] += responsibilities[i][k];
            }
        }
        double sum_new_pi = std::accumulate(new_pi.begin(), new_pi.end(), 0.0);
        if (sum_new_pi > EPSILON) {
            for (int k = 0; k < num_categories_; ++k) {
                pi_[k] = new_pi[k] / sum_new_pi;
            }
        } else {
             { std::ostringstream w_; w_ << "Warning: Sum of new pi is too small. Pi not updated."; ds_warn(w_.str()); }
        }


        std::vector<std::vector<std::vector<double>>> new_theta_counts(
            num_sources_, std::vector<std::vector<double>>(num_categories_, std::vector<double>(num_categories_, 0.0)));

        const double ORDINAL_SMOOTHING_FACTOR = 0.1; 
        const double DISTANCE_PENALTY_EXPONENT = 2.0; 

        // one source per thread: each sums its own observations in claim order (the serial order)
        #pragma omp parallel for schedule(dynamic) num_threads(threads())
        for (int j = 0; j < num_sources_; ++j) {
          for (const auto &ob : obs_by_source_[j]) {
            int i = ob.first;
            int k_obs = ob.second;

            for (int k_true = 0; k_true < num_categories_; ++k_true) {
                new_theta_counts[j][k_true][k_obs] += responsibilities[i][k_true];
                
                if (category_type_ == "ordinal") {
                    if (ordinal_category_orders_.count(k_true) && ordinal_category_orders_.count(k_obs)) {
                        double distance = std::abs(static_cast<double>(ordinal_category_orders_.at(k_true)) - ordinal_category_orders_.at(k_obs));
                        double ordinal_pseudo_count = ORDINAL_SMOOTHING_FACTOR / (std::pow(distance, DISTANCE_PENALTY_EXPONENT) + 1.0);
                        new_theta_counts[j][k_true][k_obs] += ordinal_pseudo_count;
                    } 
                }
            }
          }
        }

        for (int j = 0; j < num_sources_; ++j) {
            for (int k_true = 0; k_true < num_categories_; ++k_true) {
                double sum_counts_for_jk = 0.0;
                for (int k_obs = 0; k_obs < num_categories_; ++k_obs) {
                    sum_counts_for_jk += new_theta_counts[j][k_true][k_obs];
                }

                if (sum_counts_for_jk > EPSILON) { 
                    for (int k_obs = 0; k_obs < num_categories_; ++k_obs) {
                        theta_[j][k_true][k_obs] = new_theta_counts[j][k_true][k_obs] / sum_counts_for_jk;
                    }
                } 
            }
        }
    }

    // Main fit method
    double fit(const std::vector<std::tuple<std::string, std::string, std::string>>& claims, // Returns double (log-likelihood)
             py::object map_dimensions_obj = py::none(),
             py::dict item_coordinates_obj = py::dict(),
             const std::vector<std::tuple<std::string, std::string>>& spatial_constraints_py = {},
             py::object ordinal_order_map_obj = py::none()) {

        initialize_all_data(claims, map_dimensions_obj, item_coordinates_obj, spatial_constraints_py, ordinal_order_map_obj);

        if (num_items_ == 0 || num_sources_ == 0 || num_categories_ == 0) {
            return -std::numeric_limits<double>::infinity(); // Return neg infinity for invalid setup
        }

        initialize_parameters_random();
        build_observation_index();

        std::function<int(std::string)> cb = visitor_ ? visitor_ : [](std::string) -> int { return 0; };
        {
            std::ostringstream m_;
            m_ << "Dawid-Skene EM on " << num_items_ << " locations, " << num_sources_ << " partitions, "
               << num_categories_ << " categories";
            sptlz::CallbackLogger(cb, "dawid-skene").info(m_.str());
        }
        sptlz::CallbackProgressSender progress(cb);
        progress.init(max_iter_, 1);

        double prev_log_likelihood = -std::numeric_limits<double>::infinity();
        double current_log_likelihood = -std::numeric_limits<double>::infinity(); // Initialize for first iteration

        for (int iter = 0; iter < max_iter_; ++iter) {
            if (PyErr_CheckSignals() != 0) {  // Ctrl-C between iterations
                progress.stop();
                throw py::error_already_set();
            }
            std::vector<std::vector<double>> responsibilities = e_step_impl();

            m_step(responsibilities);

            current_log_likelihood = calculate_marginal_log_likelihood(); // Update for current iter
            

            progress.inform(iter);
            if (iter > 0 && std::abs(current_log_likelihood - prev_log_likelihood) < tolerance_) {
                break;
            }
            prev_log_likelihood = current_log_likelihood;
            if (iter == max_iter_ - 1) {
                { std::ostringstream w_; w_ << "Warning: EM did not converge within max_iter."; ds_warn(w_.str()); }
            }
        }
        progress.stop();
        return current_log_likelihood; // Return the final log-likelihood
    }

    // Calculates the marginal log-likelihood P(y | pi, theta)
    double calculate_marginal_log_likelihood() const {
        double total_log_likelihood = 0.0;


        // one term per item, in parallel, summed afterwards in item order (the serial sum)
        std::vector<double> per_item(num_items_, 0.0);
        #pragma omp parallel for schedule(static) num_threads(threads())
        for (int i = 0; i < num_items_; ++i) {
            std::vector<double> log_terms_for_logsumexp(num_categories_);

            for (int k_true = 0; k_true < num_categories_; ++k_true) {
                double log_prob_z_prior = std::log(std::max(pi_[k_true], EPSILON));
                double log_likelihood_obs_given_z = 0.0;

                {
                    for (const auto& obs : obs_by_item_[i]) {
                        int j = obs.first;
                        int k_obs = obs.second;
                        log_likelihood_obs_given_z += std::log(std::max(theta_[j][k_true][k_obs], EPSILON));
                    }
                }
                log_terms_for_logsumexp[k_true] = log_prob_z_prior + log_likelihood_obs_given_z;
            }
            per_item[i] = logsumexp(log_terms_for_logsumexp);
        }
        for (int i = 0; i < num_items_; ++i) total_log_likelihood += per_item[i];
        return total_log_likelihood;
    }

    // --- Result Getter Methods ---

    py::dict get_source_confusion_matrices() const {
        py::dict results;
        for (int j_idx = 0; j_idx < num_sources_; ++j_idx) {
            std::string source_id = idx_to_source_id_[j_idx];
            py::dict source_matrix;
            for (int true_cat_idx = 0; true_cat_idx < num_categories_; ++true_cat_idx) {
                std::string true_cat = idx_to_category_[true_cat_idx];
                py::dict claimed_probs;
                for (int claimed_cat_idx = 0; claimed_cat_idx < num_categories_; ++claimed_cat_idx) {
                    std::string claimed_cat = idx_to_category_[claimed_cat_idx];
                    claimed_probs[claimed_cat.c_str()] = theta_[j_idx][true_cat_idx][claimed_cat_idx];
                }
                source_matrix[true_cat.c_str()] = claimed_probs;
            }
            results[source_id.c_str()] = source_matrix;
        }
        return results;
    }

    py::dict get_statement_truths() {
        std::vector<std::vector<double>> responsibilities = e_step_impl(); 

        py::dict results;
        for (int i_idx = 0; i_idx < num_items_; ++i_idx) {
            std::string statement_id = idx_to_statement_id_[i_idx];
            py::dict probs;
            for (int cat_idx = 0; cat_idx < num_categories_; ++cat_idx) {
                std::string category = idx_to_category_[cat_idx];
                probs[category.c_str()] = responsibilities[i_idx][cat_idx];
            }
            results[statement_id.c_str()] = probs;
        }
        return results;
    }

    py::dict get_final_predicted_truth() {
        py::dict statement_truths_probs = get_statement_truths();
        py::dict final_truths;
        for (auto const& [statement_id_obj, probs_dict_obj] : statement_truths_probs) {
            std::string statement_id = statement_id_obj.cast<std::string>();
            py::dict probs_dict = probs_dict_obj.cast<py::dict>();

            std::string most_probable_category = "";
            double max_prob = -1.0;

            for (auto const& [cat_obj, prob_obj] : probs_dict) {
                std::string category = cat_obj.cast<std::string>();
                double prob = prob_obj.cast<double>();
                if (prob > max_prob) {
                    max_prob = prob;
                    most_probable_category = category;
                }
            }
            final_truths[statement_id.c_str()] = most_probable_category;
        }
        return final_truths;
    }

    py::dict get_pi() const {
        py::dict pi_dict;
        for (int k_idx = 0; k_idx < num_categories_; ++k_idx) {
            pi_dict[idx_to_category_[k_idx].c_str()] = pi_[k_idx];
        }
        return pi_dict;
    }

    py::dict get_theta() const {
        py::dict theta_dict;
        for (int j_idx = 0; j_idx < num_sources_; ++j_idx) {
            std::string source_id = idx_to_source_id_[j_idx];
            py::dict source_matrix;
            for (int k_true_idx = 0; k_true_idx < num_categories_; ++k_true_idx) {
                std::string true_cat_str = idx_to_category_[k_true_idx];
                py::dict true_cat_row;
                for (int k_obs_idx = 0; k_obs_idx < num_categories_; ++k_obs_idx) {
                    std::string obs_cat_str = idx_to_category_[k_obs_idx];
                    true_cat_row[obs_cat_str.c_str()] = theta_[j_idx][k_true_idx][k_obs_idx];
                }
                source_matrix[true_cat_str.c_str()] = true_cat_row;
            }
            theta_dict[source_id.c_str()] = source_matrix;
        }
        return theta_dict;
    }   
};

// --- Spatial Dawid-Skene EM Implementation ---

class SpatialDawidSkeneEM : public DawidSkeneEM {
private:
    std::map<int, std::vector<int>> item_neighbors_;
    std::vector<std::vector<int>> item_coordinates_matrix_;
    std::set<std::pair<int, int>> spatial_constraints_idx_set_;
    std::vector<int> map_dimensions_;
    bool is_spatial_active_ = false; // Flag to indicate if spatial features are truly active
    double spatial_penalty_factor_; // New member variable for configurable penalty factor

public:
    SpatialDawidSkeneEM(py::object categories_obj, double tolerance, int max_iter, const std::string& category_type_str, py::object ordinal_order_map_obj, double spatial_penalty_factor)
        : DawidSkeneEM(categories_obj, tolerance, max_iter, category_type_str, ordinal_order_map_obj),
          spatial_penalty_factor_(spatial_penalty_factor) { // Initialize new member
    }

    void initialize_all_data(
        const std::vector<std::tuple<std::string, std::string, std::string>>& claims,
        py::object map_dimensions_obj, 
        py::dict item_coordinates_obj, // Still receiving this from Python, even if auto-generated there
        const std::vector<std::tuple<std::string, std::string>>& spatial_constraints_py_raw,
        py::object ordinal_order_map_obj = py::none() 
    ) override {
        DawidSkeneEM::initialize_all_data(claims, map_dimensions_obj, item_coordinates_obj, spatial_constraints_py_raw, ordinal_order_map_obj);

        if (num_items_ == 0 || num_sources_ == 0 || num_categories_ == 0) {
            is_spatial_active_ = false;
            return;
        }

        is_spatial_active_ = true; // Assume active unless a problem is found
        
        // Map Dimensions validation
        if (map_dimensions_obj.is_none()) {
            is_spatial_active_ = false;
            { std::ostringstream w_; w_ << "Warning: map_dimensions not provided. Spatial features will be inactive."; ds_warn(w_.str()); }
        } else {
            map_dimensions_ = map_dimensions_obj.cast<std::vector<int>>();
            if (map_dimensions_.empty() || std::any_of(map_dimensions_.begin(), map_dimensions_.end(), [](int d){ return d <= 0; })) {
                is_spatial_active_ = false;
                { std::ostringstream w_; w_ << "Warning: Invalid map_dimensions (empty or non-positive). Spatial features will be inactive."; ds_warn(w_.str()); }
            }
        }

        // Item Coordinates matrix
        if (is_spatial_active_ && !map_dimensions_.empty()) {
            item_coordinates_matrix_.assign(num_items_, std::vector<int>(map_dimensions_.size())); 
            int num_dims = map_dimensions_.size();
            // This loop populates item_coordinates_matrix_ from the provided item_coordinates_obj
            // which is now generated in Python for `aggregate_with_btd_em` caller.
            for (auto const& [item_id_obj, coords_obj] : item_coordinates_obj) {
                std::string item_id = item_id_obj.cast<std::string>();
                std::vector<int> coords = coords_obj.cast<std::vector<int>>();
                
                if (statement_to_idx_.count(item_id)) {
                    int item_idx = statement_to_idx_.at(item_id);
                    if (coords.size() != num_dims) {
                        { std::ostringstream w_; w_ << "Warning: Coordinate " << py::str(coords_obj).cast<std::string>() << " for item " << item_id << " has incorrect dimensions. Spatial features inactive."; ds_warn(w_.str()); }
                        is_spatial_active_ = false;
                        break;
                    }
                    // Validate coordinates against map dimensions
                    for (int d = 0; d < num_dims; ++d) {
                        if (coords[d] < 0 || coords[d] >= map_dimensions_[d]) {
                            { std::ostringstream w_; w_ << "Warning: Coordinate " << py::str(coords_obj).cast<std::string>() << " for item " << item_id << " out of bounds for dimension " << d << ". Spatial features inactive."; ds_warn(w_.str()); }
                            is_spatial_active_ = false;
                            break;
                        }
                    }
                    if (!is_spatial_active_) break; // Break outer loop if problem found
                    item_coordinates_matrix_[item_idx] = coords;
                } else {
                    { std::ostringstream w_; w_ << "Warning: Item ID '" << item_id << "' in item_coordinates not found in claims. Spatial features inactive."; ds_warn(w_.str()); }
                    is_spatial_active_ = false;
                    break;
                }
            }
        } else {
            is_spatial_active_ = false; 
        }

        // Spatial Constraints set
        spatial_constraints_idx_set_.clear();
        if (!spatial_constraints_py_raw.empty() && is_spatial_active_) {
            for (const auto& constraint : spatial_constraints_py_raw) {
                std::string cat1_str = std::get<0>(constraint);
                std::string cat2_str = std::get<1>(constraint);
                if (category_to_idx_.count(cat1_str) && category_to_idx_.count(cat2_str)) {
                    int cat1_idx = category_to_idx_.at(cat1_str);
                    int cat2_idx = category_to_idx_.at(cat2_str);
                    spatial_constraints_idx_set_.insert({cat1_idx, cat2_idx});
                    spatial_constraints_idx_set_.insert({cat2_idx, cat1_idx}); // Symmetric
                } else {
                    { std::ostringstream w_; w_ << "Warning: Spatial constraint ('" << cat1_str << "', '" << cat2_str << "') involves unknown category. Skipping this constraint."; ds_warn(w_.str()); }
                }
            }
            if (spatial_constraints_idx_set_.empty()) {
                { std::ostringstream w_; w_ << "Warning: No valid spatial constraints after parsing. Spatial features inactive."; ds_warn(w_.str()); }
                is_spatial_active_ = false;
            }
        } else if (is_spatial_active_ && spatial_constraints_py_raw.empty()) {
             { std::ostringstream w_; w_ << "Warning: Spatial constraints list is empty. Spatial features inactive."; ds_warn(w_.str()); }
             is_spatial_active_ = false;
        }


        if (!is_spatial_active_) {
            map_dimensions_.clear();
            item_coordinates_matrix_.clear();
            item_neighbors_.clear(); 
            spatial_constraints_idx_set_.clear();
            { std::ostringstream w_; w_ << "Spatial features are INACTIVE due to invalid/incomplete spatial data or settings."; ds_warn(w_.str()); }
        } else {
            // Build item_neighbors_ map (only if spatial features are truly active)
            item_neighbors_.clear();
            int num_dims = map_dimensions_.size();
            std::map<std::vector<int>, int> coord_to_item_idx_temp; 
            for (int temp_i = 0; temp_i < num_items_; ++temp_i) {
                coord_to_item_idx_temp[item_coordinates_matrix_[temp_i]] = temp_i;
            }

            for (int i_idx_a = 0; i_idx_a < num_items_; ++i_idx_a) {
                std::vector<int> current_item_coord = item_coordinates_matrix_[i_idx_a];
                std::vector<int> neighbors_for_item;

                for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx) {
                    for (int delta = -1; delta <= 1; delta += 2) { 
                        std::vector<int> neighbor_coord = current_item_coord;
                        neighbor_coord[dim_idx] += delta;

                        bool is_valid_coord = true;
                        for (int d = 0; d < num_dims; ++d) {
                            if (neighbor_coord[d] < 0 || neighbor_coord[d] >= map_dimensions_[d]) {
                                is_valid_coord = false;
                                break;
                            }
                        }
                        if (is_valid_coord && coord_to_item_idx_temp.count(neighbor_coord)) {
                            int neighbor_item_idx = coord_to_item_idx_temp.at(neighbor_coord);
                            if (neighbor_item_idx != i_idx_a) { 
                                neighbors_for_item.push_back(neighbor_item_idx);
                            }
                        }
                    }
                }
                item_neighbors_[i_idx_a] = neighbors_for_item;
            }
        }
    }

    // Override E-Step to include spatial penalties
    std::vector<std::vector<double>> e_step_impl() override {
        std::vector<std::vector<double>> responsibilities(num_items_, std::vector<double>(num_categories_));

        #pragma omp parallel for schedule(static) num_threads(threads())
        for (int i = 0; i < num_items_; ++i) {
            std::vector<double> unnorm_log_probs_for_item_i(num_categories_);

            for (int k_true = 0; k_true < num_categories_; ++k_true) {
                double log_prob_z_prior = std::log(std::max(pi_[k_true], EPSILON));
                double log_likelihood_obs_given_z = 0.0;

                {
                    for (const auto& obs : obs_by_item_[i]) {
                        int j = obs.first;
                        int k_obs = obs.second;
                        log_likelihood_obs_given_z += std::log(std::max(theta_[j][k_true][k_obs], EPSILON));
                    }
                }

                double spatial_penalty_log_term = 0.0;
                // Only apply spatial penalty if spatial features are active
                if (is_spatial_active_ && !item_neighbors_.empty() && item_neighbors_.count(i)) {
                    for (int neighbor_item_idx : item_neighbors_.at(i)) { 
                        (void)neighbor_item_idx; // Suppress unused warning if not directly used

                        for (int neighbor_k_true = 0; neighbor_k_true < num_categories_; ++neighbor_k_true) {
                            if (spatial_constraints_idx_set_.count({k_true, neighbor_k_true})) {
                                // spatial_penalty_factor_ is positive.
                                // std::log(std::max(pi_[neighbor_k_true], EPSILON)) is negative (or zero).
                                // Product will be negative or zero. Adding this negative product penalizes.
                                spatial_penalty_log_term += std::log(std::max(pi_[neighbor_k_true], EPSILON)) * spatial_penalty_factor_;

                                // For debugging, uncomment this to see individual penalty contributions:
                                // std::cerr << "Applying penalty for item " << i << " (k_true=" << idx_to_category_[k_true]
                                //           << ") and neighbor " << neighbor_item_idx << " (neighbor_k_true="
                                //           << idx_to_category_[neighbor_k_true] << "). Penalty added: "
                                //           << std::log(std::max(pi_[neighbor_k_true], EPSILON)) * spatial_penalty_factor_ << std::endl;
                            }
                        }
                    }
                }
                
                unnorm_log_probs_for_item_i[k_true] = log_prob_z_prior + log_likelihood_obs_given_z + spatial_penalty_log_term;
            }
            
            double log_norm_const = logsumexp(unnorm_log_probs_for_item_i);
            for (int k_true = 0; k_true < num_categories_; ++k_true) {
                responsibilities[i][k_true] = std::exp(unnorm_log_probs_for_item_i[k_true] - log_norm_const);
            }
        }
        return responsibilities;
    }
};

// --- Pybind11 Module Definition ---

PYBIND11_MODULE(_dawid_skene, m) {
    m.doc() = "C++ Dawid-Skene EM implementation with Python bindings using setuptools.";

    py::class_<DawidSkeneEM>(m, "DawidSkeneEM")
        .def(py::init<py::object, double, int, const std::string&, py::object>(), 
             py::arg("categories") = py::none(), 
             py::arg("tolerance") = 1e-4, 
             py::arg("max_iter") = 100,
             py::arg("category_type") = "nominal", 
             py::arg("ordinal_order_map") = py::none())
        .def("fit", &DawidSkeneEM::fit, // fit method returns double now
             py::arg("claims"),
             py::arg("map_dimensions") = py::none(), 
             py::arg("item_coordinates") = py::dict(), 
             py::arg("spatial_constraints") = std::vector<std::tuple<std::string, std::string>>{}, 
             py::arg("ordinal_order_map") = py::none(),
             "Fits the Dawid-Skene EM model to claims data and returns the final log-likelihood.")
        .def("get_source_confusion_matrices", &DawidSkeneEM::get_source_confusion_matrices, "Returns inferred source confusion matrices.")
        .def("get_statement_truths", &DawidSkeneEM::get_statement_truths, "Returns inferred probabilities for true statements.")
        .def("get_final_predicted_truth", &DawidSkeneEM::get_final_predicted_truth, "Returns the final most probable true statement for each item.")
        .def("get_pi", &DawidSkeneEM::get_pi, "Returns the inferred prior probabilities of true categories (pi).")
        .def("get_theta", &DawidSkeneEM::get_theta, "Returns the inferred annotator confusion matrices (theta).")
        .def("set_seed", &DawidSkeneEM::set_seed, py::arg("seed"), "Seeds the random initialisation."); 


    py::class_<SpatialDawidSkeneEM, DawidSkeneEM>(m, "SpatialDawidSkeneEM")
        .def(py::init<py::object, double, int, const std::string&, py::object, double>(), 
             py::arg("categories") = py::none(),
             py::arg("tolerance") = 1e-4,
             py::arg("max_iter") = 100,
             py::arg("category_type") = "nominal",
             py::arg("ordinal_order_map") = py::none(),
             py::arg("spatial_penalty_factor") = 1.0); 

    m.def("aggregate_with_btd_em", [](
        py::list esi_samples_py, 
        py::object categories_list_obj, 
        double tolerance,
        int max_iter,
        const std::string& category_type_str, 
        py::object ordinal_order_map_obj,     
        py::object map_dimensions_obj, 
        py::dict item_coordinates_obj, // Still need this for passing to C++ model
        const std::vector<std::tuple<std::string, std::string>>& spatial_constraints_py,
        double spatial_penalty_factor, // Parameter for binding
        unsigned int seed,
        py::object visitor,
        int num_threads
    ) -> py::tuple { // Changed return type to py::tuple
        long p_statements = py::len(esi_samples_py);
        if (p_statements == 0) {
            { std::ostringstream w_; w_ << "Warning: Empty esi_samples provided. Returning empty list and neg infinity log-likelihood."; ds_warn(w_.str()); }
            return py::make_tuple(py::list(), -std::numeric_limits<double>::infinity());
        }
        long n_sources = 0;
        if (py::len(esi_samples_py[0]) > 0) {
            n_sources = py::len(esi_samples_py[0]);
        }
        
        if (n_sources == 0) {
            { std::ostringstream w_; w_ << "Warning: Empty esi_samples provided (no sources). Returning empty list and neg infinity log-likelihood."; ds_warn(w_.str()); }
            return py::make_tuple(py::list(), -std::numeric_limits<double>::infinity());
        }

        std::vector<std::tuple<std::string, std::string, std::string>> claims;
        claims.reserve(p_statements * n_sources);

        for (long r_idx = 0; r_idx < p_statements; ++r_idx) {
            py::list row = esi_samples_py[r_idx].cast<py::list>();
            for (long c_idx = 0; c_idx < n_sources; ++c_idx) {
                std::string source_id = "Source_R" + std::to_string(c_idx + 1);
                std::string statement_id = "Item_" + std::to_string(r_idx + 1);
                py::handle value = row[c_idx];
                // a member of an empty cell (None, or a float NaN) is no claim
                if (value.is_none() || (py::isinstance<py::float_>(value) && std::isnan(value.cast<double>()))) {
                    continue;
                }
                std::string claimed_category = py::str(value).cast<std::string>();
                claims.emplace_back(source_id, statement_id, claimed_category);
            }
        }

        // Check if all spatial parameters are provided for activation
        bool use_spatial_model = (!map_dimensions_obj.is_none() && !item_coordinates_obj.empty() && !spatial_constraints_py.empty());

        std::unique_ptr<DawidSkeneEM> model_ptr;
        if (use_spatial_model) {
            model_ptr.reset(new SpatialDawidSkeneEM(categories_list_obj, tolerance, max_iter, category_type_str, ordinal_order_map_obj, spatial_penalty_factor));
        } else {
            model_ptr.reset(new DawidSkeneEM(categories_list_obj, tolerance, max_iter, category_type_str, ordinal_order_map_obj));
        }
        model_ptr->set_seed(seed);
        model_ptr->set_num_threads(num_threads);
        if (!visitor.is_none()) {
            model_ptr->set_visitor([visitor](std::string s) -> int { visitor(s); return 0; });
        }

        double final_log_likelihood = model_ptr->fit(claims, map_dimensions_obj, item_coordinates_obj, spatial_constraints_py, ordinal_order_map_obj);
        
        py::dict inferred_truths = model_ptr->get_final_predicted_truth();
        py::dict statement_probabilities = model_ptr->get_statement_truths(); 

        py::list esi_estimation_list;
        for (long i = 0; i < p_statements; ++i) {
            std::string st_id = "Item_" + std::to_string(i + 1);
            if (inferred_truths.contains(st_id.c_str())) {
                esi_estimation_list.append(inferred_truths[st_id.c_str()]);
            } else {
                esi_estimation_list.append(py::none());
            }
        }


        return py::make_tuple(esi_estimation_list, statement_probabilities, final_log_likelihood); // Return tuple
    },
    py::arg("esi_samples"),
    py::arg("categories_list") = py::none(),
    py::arg("tolerance") = 1e-4,
    py::arg("max_iter") = 100,
    py::arg("category_type") = "nominal", 
    py::arg("ordinal_order_map") = py::none(), 
    py::arg("map_dimensions") = py::none(), 
    py::arg("item_coordinates") = py::dict(), // Still binding this, as it's a param to fit()
    py::arg("spatial_constraints") = std::vector<std::tuple<std::string, std::string>>{},
    py::arg("spatial_penalty_factor") = 1.0,
    py::arg("seed") = 0u,
    py::arg("visitor") = py::none(),
    py::arg("num_threads") = 0
    );
}
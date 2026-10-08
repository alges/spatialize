#ifndef _SPTLZ_ENSEMBLE_
#define _SPTLZ_ENSEMBLE_

#include <sstream>
#include <random>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <atomic>
#include <thread>
#include <exception>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "spatialize/utils.hpp"
#include "spatialize/callback_logging.hpp"
#include "spatialize/partition.hpp"
#include "spatialize/decoder.hpp"
#include "spatialize/empty_cells.hpp"

namespace sptlz{
	// Ensemble of random partitions (encoder) with a local interpolator (decoder): one loop for
	// estimation, leave-one-out and k-fold, whatever the partition process and the decoder.
	// Subclasses draw the forest (ESI: Mondrian, VORONOI: Voronoi); set_decoder() then fits the
	// decoder with the same generator, continuing its sequence.
	class Ensemble {
		protected:
			std::string class_name;
			std::function<int(std::string)> callback_visitor;
			std::vector<sptlz::Partition*> forest;
			std::vector<std::vector<float>> coords;
			std::vector<float> values;
			std::mt19937 my_rand;
			unsigned int seed;        // the run's seed, handed to the decoders (CellContext)
			sptlz::Decoder *decoder;
			bool estimate_log_debug;  // log "computing estimates" at debug level (Voronoi) instead of info
			sptlz::EmptyCellPolicy empty_cells;  // what a member is when its cell holds no datum

			// the cells of a tree holding at least one datum `usable` accepts
			template <typename Usable>
			std::vector<int> cells_with_data(sptlz::Partition *mt, Usable usable){
				std::vector<int> out;
				for(size_t j=0; j<mt->samples_by_leaf.size(); j++){
					for(int d: mt->samples_by_leaf.at(j)){
						if(usable(d)){ out.push_back(static_cast<int>(j)); break; }
					}
				}
				return(out);
			}

			// The marks of one partition (empty_cells.hpp): each target, in the order given, draws a
			// source among the cells holding data that `usable` accepts (the nearest mark_knn of them,
			// or all), without repetition while candidates remain, and receives that cell's decoder
			// prediction at its point (or, with mark_value "datum", one of its data); with mark_source
			// "data", a datum drawn without repetition.
			template <typename Usable>
			void fill_marks(int t, sptlz::Partition *mt, std::vector<sptlz::MarkTarget> &targets, Usable usable,
			                std::vector<std::vector<float>> &results){
				if(targets.empty()) return;
				const auto &policy = this->empty_cells;
				if(policy.source == sptlz::MarkSource::DATA){
					std::vector<int> data;
					for(int d=0; d<static_cast<int>(values.size()); d++) if(usable(d)) data.push_back(d);
					if(data.empty()) return;
					std::vector<bool> used(values.size(), false);
					for(auto &target: targets){
						std::vector<int> free;
						for(int d: data) if(!used.at(d)) free.push_back(d);
						const auto &pool = free.empty() ? data : free;
						int d = pool.at(sptlz::mark_pick(sptlz::mark_uniform(this->seed, t, target.key, 1), static_cast<int>(pool.size())));
						used.at(d) = true;
						for(int r: target.rows) results.at(r).at(t) = values.at(d);
					}
					return;
				}
				std::vector<int> eligible = cells_with_data(mt, usable);
				std::vector<std::vector<float>> points;
				for(int c: eligible) points.push_back(mt->leaf_point(c));
				std::vector<bool> used(mt->samples_by_leaf.size(), false);
				for(auto &target: targets){
					std::vector<int> candidates;   // positions in `eligible`
					for(size_t a=0; a<eligible.size(); a++) if(eligible.at(a) != target.cell) candidates.push_back(static_cast<int>(a));
					if(candidates.empty()) continue;
					if(policy.source == sptlz::MarkSource::LOCAL && static_cast<int>(candidates.size()) > policy.knn){
						std::vector<float> dist(eligible.size());
						for(int a: candidates) dist.at(a) = sptlz::distance(&(target.point), &(points.at(a)));
						std::stable_sort(candidates.begin(), candidates.end(), [&dist](int x, int y){ return(dist.at(x) < dist.at(y)); });
						candidates.resize(policy.knn);
					}
					std::vector<int> free;
					for(int a: candidates) if(!used.at(eligible.at(a))) free.push_back(a);
					const auto &pool = free.empty() ? candidates : free;
					int c = eligible.at(pool.at(sptlz::mark_pick(sptlz::mark_uniform(this->seed, t, target.key, 1), static_cast<int>(pool.size()))));
					used.at(c) = true;
					std::vector<int> source;
					for(int d: mt->samples_by_leaf.at(c)) if(usable(d)) source.push_back(d);
					if(!policy.value_from_decoder){
						// the block-mark model: one of the cell's data, uniformly
						int d = source.at(sptlz::mark_pick(sptlz::mark_uniform(this->seed, t, target.key, 2), static_cast<int>(source.size())));
						for(int r: target.rows) results.at(r).at(t) = values.at(d);
						continue;
					}
					std::vector<std::vector<float>> at{target.point};
					std::vector<int> at_id{0};
					auto value = this->decoder->leaf_estimation(&coords, &values, &source, &at, &at_id, &(mt->leaf_params.at(c)),
					                                            CellContext{t, c, sptlz::mark_seed(this->seed, target.key)});
					for(int r: target.rows) results.at(r).at(t) = value.at(0);
				}
			}

		public:
			Ensemble(std::vector<std::vector<float>> _coords,
			         std::vector<float> _values,
			         std::function<int(std::string)> visitor,
			         int seed=206936){
				this->class_name = __func__;
				this->callback_visitor = visitor;
				this->my_rand = std::mt19937(seed);
				this->seed = static_cast<unsigned int>(seed);
				this->coords = _coords;
				this->values = _values;
				this->decoder = NULL;
				this->estimate_log_debug = false;
			}

			virtual ~Ensemble(){
				for(size_t i=0; i<this->forest.size(); i++){
					delete(this->forest.at(i));
				}
				std::vector<sptlz::Partition*>().swap(forest);
				if(this->decoder != NULL){
					delete(this->decoder);
				}
			}

			// Takes ownership of the decoder and fits it on the forest.
			void set_decoder(sptlz::Decoder *_decoder){
				if(this->decoder != NULL){
					delete(this->decoder);
				}
				this->decoder = _decoder;
				this->decoder->fit(&(this->forest), &(this->coords), &(this->values), this->my_rand, this->callback_visitor, this->class_name);
			}

			int forest_size(){
				return(static_cast<int>(this->forest.size()));
			}

			// The cell of every location in every tree, [location][tree], -1 outside the box. Cell
			// labels are local to a tree, so two locations share a cell of tree t exactly when their
			// labels in column t are equal.
			std::vector<std::vector<int>> cells_of(std::vector<std::vector<float>> *locations){
				int n = static_cast<int>(forest.size());
				std::vector<std::vector<int>> result(locations->size(), std::vector<int>(n, -1));
				#ifdef _OPENMP
				#pragma omp parallel for schedule(dynamic, 1)
				#endif
				for(int t=0; t<n; t++){
					for(size_t j=0; j<locations->size(); j++){
						result[j][t] = forest.at(t)->search_leaf(locations->at(j));
					}
				}
				return(result);
			}

			void set_empty_cells(const sptlz::EmptyCellPolicy &policy){
				this->empty_cells = policy;
			}

			// name used in log messages
			void set_class_name(std::string _class_name){
				this->class_name = _class_name;
			}

			std::vector<std::vector<int>> get_leaf_for_samples(){
				int n = static_cast<int>(forest.size());
				std::vector<std::vector<int>> results(this->coords.size());

				for(int i=0; i<n; i++){
					// get tree
					auto lfs = forest.at(i)->leaf_for_sample;
					for(size_t j=0; j<this->coords.size(); j++){
						results.at(j).push_back(lfs.at(j));
					}
				}
				return(results);
			}

			// Runs body(i) for every tree i, in parallel over the trees when the decoder allows it.
			// Each tree writes only its own column of the results, so the output does not depend on
			// the number of threads. Only the calling thread talks to Python (Ctrl-C checks and
			// progress, one token per tree as before); an exception thrown by a tree is rethrown
			// after the loop, since it cannot leave a parallel region.
			template <typename Body>
			void for_each_tree(int n, sptlz::CallbackProgressSender *progress, Body body){
				const bool parallel = (this->decoder == NULL) || this->decoder->thread_safe();
				const std::thread::id caller = std::this_thread::get_id();
				std::atomic<int> done(0);
				std::atomic<bool> stop(false);
				std::exception_ptr error = nullptr;
				int sent = 0;  // progress tokens sent, touched by the calling thread only

				#ifdef _OPENMP
				#pragma omp parallel for schedule(dynamic, 1) if(parallel)
				#endif
				for(int i=0; i<n; i++){
					if(stop.load()){
						continue;
					}
					try{
						body(i);
					}catch(...){
						#ifdef _OPENMP
						#pragma omp critical(sptlz_ensemble_error)
						#endif
						{
							if(!error){
								error = std::current_exception();
							}
						}
						stop.store(true);
					}
					done++;
					if(std::this_thread::get_id() == caller && !stop.load()){
						if(PyErr_CheckSignals() != 0){  // to allow ctrl-c from user
							#ifdef _OPENMP
							#pragma omp critical(sptlz_ensemble_error)
							#endif
							{
								if(!error){
									error = std::make_exception_ptr(pybind11::error_already_set());
								}
							}
							stop.store(true);
						}else{
							for(int d=done.load(); sent<d; ){
								sent++;
								progress->inform(static_cast<int>(100.0*sent/n));
							}
						}
					}
				}

				if(error){
					std::rethrow_exception(error);
				}
				while(sent<n){
					sent++;
					progress->inform(static_cast<int>(100.0*sent/n));
				}
			}

			std::vector<std::vector<float>> estimate(std::vector<std::vector<float>> *locations){
				int n = static_cast<int>(forest.size());
				std::vector<std::vector<float>> results(locations->size(), std::vector<float>(n, NAN));
				sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
				sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

				if(this->estimate_log_debug){
					logger->debug("computing estimates");
				}else{
					logger->info("computing estimates");
				}

				progress->init(n, 1);

				for_each_tree(n, progress, [&](int i){
					auto mt = forest.at(i);
					// join all locations for same leaf
					std::vector<std::vector<int>> locations_by_leaf(mt->n_leaves());
					for(size_t j=0; j<locations->size(); j++){
						int aux = mt->search_leaf(locations->at(j));
						locations_by_leaf.at(aux).push_back(static_cast<int>(j));
					}
					// make estimation by leaf (empty cells keep NaN, unless the policy fills them)
					bool marking = this->empty_cells.kind == sptlz::EmptyCells::MARK;
					std::vector<sptlz::MarkTarget> targets;
					for(size_t j=0; j<locations_by_leaf.size(); j++){
						if(mt->samples_by_leaf.at(j).size()==0){
							if(marking && !locations_by_leaf.at(j).empty()){
								targets.push_back({static_cast<int>(j), mt->leaf_point(static_cast<int>(j)),
								                   sptlz::mark_key_estimate(static_cast<int>(j)), locations_by_leaf.at(j)});
							}
							continue;
						}
						auto predictions = decoder->leaf_estimation(&coords, &values, &(mt->samples_by_leaf.at(j)), locations, &(locations_by_leaf.at(j)), &(mt->leaf_params.at(j)), CellContext{i, static_cast<int>(j), this->seed});
						for(size_t k=0; k<locations_by_leaf.at(j).size(); k++){
							results.at(locations_by_leaf.at(j).at(k)).at(i) = predictions.at(k);
						}
					}
					fill_marks(i, mt, targets, [](int){ return(true); }, results);
				});

				progress->stop();

				delete logger;
				delete progress;
				return(results);
			}

			std::vector<std::vector<float>> leave_one_out(){
				int n = static_cast<int>(forest.size());
				std::vector<std::vector<float>> results(coords.size(), std::vector<float>(n, NAN));

				sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
				sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

				logger->debug("computing leave-one-out");

				progress->init(n, 1);

				for_each_tree(n, progress, [&](int i){
					auto mt = forest.at(i);
					bool marking = this->empty_cells.kind == sptlz::EmptyCells::MARK;
					std::vector<sptlz::MarkTarget> targets;
					for(size_t j=0; j<mt->samples_by_leaf.size(); j++){
						if(marking && mt->samples_by_leaf.at(j).size()==1){
							// the held-out datum leaves its cell empty: a mark from the other cells, read at
							// the datum's location
							int held = mt->samples_by_leaf.at(j).at(0);
							targets.push_back({static_cast<int>(j), coords.at(held), sptlz::mark_key_loo(held), {held}});
							continue;
						}
						if(mt->samples_by_leaf.at(j).size()!=0){
							auto predictions = decoder->leaf_loo(&coords, &values, &(mt->samples_by_leaf.at(j)), &(mt->leaf_params.at(j)), CellContext{i, static_cast<int>(j), this->seed});
							for(size_t k=0; k<mt->samples_by_leaf.at(j).size(); k++){
								results.at(mt->samples_by_leaf.at(j).at(k)).at(i) = predictions.at(k);
							}
						}
					}
					// a held-out datum serves no other datum's mark, being alone in its own cell, which
					// is never a candidate for its mark
					fill_marks(i, mt, targets, [](int){ return(true); }, results);
				});

				progress->stop();

				delete logger;
				delete progress;
				return(results);
			}

			std::vector<std::vector<float>> k_fold(int k, int seed=206936){
				auto fold_rand = std::mt19937(seed);
				std::uniform_real_distribution<float> uni_float;
				auto folds = get_folds(static_cast<int>(values.size()), k, uni_float(fold_rand));
				int n = static_cast<int>(forest.size());
				std::vector<std::vector<float>> results(coords.size(), std::vector<float>(n, NAN));

				sptlz::CallbackLogger *logger = new sptlz::CallbackLogger(this->callback_visitor, this->class_name);
				sptlz::CallbackProgressSender *progress = new sptlz::CallbackProgressSender(this->callback_visitor);

				logger->debug("computing k-fold");

				progress->init(n, 1);

				for_each_tree(n, progress, [&](int i){
					auto mt = forest.at(i);
					bool marking = this->empty_cells.kind == sptlz::EmptyCells::MARK;
					std::vector<std::vector<sptlz::MarkTarget>> targets_by_fold(marking ? k : 0);
					for(size_t j=0; j<mt->samples_by_leaf.size(); j++){
						auto &cell = mt->samples_by_leaf.at(j);
						if(cell.size()!=0){
							auto predictions = decoder->leaf_kfold(k, &coords, &values, &folds, &cell, &(mt->leaf_params.at(j)), CellContext{i, static_cast<int>(j), this->seed});
							for(size_t l=0; l<cell.size(); l++){
								results.at(cell.at(l)).at(i) = predictions.at(l);
							}
						}
						if(!marking) continue;
						// a fold that takes every datum of the cell leaves it empty: one mark per cell
						// and fold, from the data outside the fold
						for(int f=0; f<k; f++){
							bool any_in = false, any_out = false;
							for(int d: cell){ if(folds.at(d)==f) any_in = true; else any_out = true; }
							if(!any_in || any_out) continue;
							std::vector<int> rows;
							for(int x: cell) if(folds.at(x)==f) rows.push_back(x);
							targets_by_fold.at(f).push_back({static_cast<int>(j), mt->leaf_point(static_cast<int>(j)),
							                                 sptlz::mark_key_kfold(static_cast<int>(j), f), rows});
						}
					}
					// each fold draws its marks from the data outside it
					for(int f=0; f<static_cast<int>(targets_by_fold.size()); f++){
						fill_marks(i, mt, targets_by_fold.at(f), [&folds, f](int x){ return(folds.at(x) != f); }, results);
					}
				});

				progress->stop();

				delete logger;
				delete progress;
				return(results);
			}
	};
}

#endif

#ifndef _SPTLZ_DECODER_
#define _SPTLZ_DECODER_

#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include "spatialize/partition.hpp"
#include "spatialize/interrupt.hpp"

namespace sptlz{
	// Where a decoder is called: the tree and the cell of the ensemble, and the run's seed. Decoders
	// that draw use them to derive their random numbers, so that a draw depends neither on the
	// number of threads nor on the other queries.
	struct CellContext {
		int tree;
		int cell;
		unsigned int seed;
		sptlz::Interrupt *interrupt = nullptr;  // the Ctrl-C state of the loop, for decoders with long cells

		// true once the loop in progress must stop; a decoder that solves many systems in one cell asks
		// between them, so that Ctrl-C need not wait for the whole cell
		bool interrupted() const { return(interrupt != nullptr && interrupt->requested()); }

		// The seed of a refit of the cell's parameters without some of its data, for a held-out
		// prediction (leave-one-out, k-fold). It depends on the run, the tree, the cell and `item` only,
		// so a refit is the same for any number of threads.
		unsigned int refit_seed(int item) const {
			uint64_t h = static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ULL;
			h ^= (static_cast<uint64_t>(static_cast<uint32_t>(tree)) << 32) ^ static_cast<uint32_t>(cell);
			h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ULL;
			h ^= static_cast<uint64_t>(static_cast<uint32_t>(item)) * 0x94D049BB133111EBULL;
			h = (h ^ (h >> 27)) * 0x94D049BB133111EBULL;
			return(static_cast<unsigned int>(h ^ (h >> 31)));
		}
	};

	// Two-fold cross-fitting of a cell's parameters for its held-out predictions (leave-one-out,
	// k-fold). A decoder that fits its parameters on the cell's own data, by the cell's own errors,
	// would let a held-out datum help choose the parameters that predict it. Instead the cell's data
	// are split in two halves, by `group` (0 or 1), and each half is predicted with the parameters
	// fitted on the other: two fits on half the data, so honest scores cost about as much as one fit.
	// `fit(ids, seed)` fits the parameters of the data `ids` (global indices).
	template <typename Fit>
	std::vector<std::vector<float>> cross_fitted(const std::vector<int> &ids, const std::vector<int> &group,
	                                             const CellContext &cell, Fit fit){
		std::vector<int> half[2];
		for(size_t j=0; j<ids.size(); j++) half[group[j]].push_back(ids[j]);
		std::vector<std::vector<float>> params(2);
		for(int h=0; h<2; h++){
			// the parameters predicting half h, fitted on the other half (none when it is empty: the
			// caller then refits on the training data of each held-out set)
			if(!half[h].empty() && !half[1-h].empty()) params[h] = fit(half[1-h], cell.refit_seed(h));
		}
		return(params);
	}

	// Local interpolator applied inside one cell (the "decoder"). It never sees the partition
	// process: the ensemble hands it the samples of a cell and the locations to predict there.
	class Decoder {
		public:
			virtual ~Decoder(){}

			// Whether leaf_estimation / leaf_loo / leaf_kfold may run on several cells at once, from
			// threads other than the caller's (false for decoders that call into Python).
			virtual bool thread_safe(){ return true; }

			// Per-cell fitting run once after the forest is built (e.g. adaptive parameters).
			// `rng` is the ensemble's generator, positioned right after the forest was drawn.
			virtual void fit(std::vector<Partition*> *forest,
			                 std::vector<std::vector<float>> *coords,
			                 std::vector<float> *values,
			                 std::mt19937 &rng,
			                 std::function<int(std::string)> visitor,
			                 std::string class_name){}

			// The parameters of one region fitted on the given data (a coarser cell standing for an
			// empty one); none for the decoders without per-cell parameters. `seed` replaces the
			// generator of fit(), so the result does not depend on the order of the regions.
			virtual std::vector<float> fit_cell(std::vector<std::vector<float>> *coords, std::vector<float> *values,
			                                    const std::vector<int> &samples, unsigned int seed){
				return(std::vector<float>());
			}

			// Whether a coarser region may take the parameters already fitted in one of its cells
			// (Partition::Coarse::donor) in place of fit_cell. Fitting a region near the root of a
			// tree costs a search over hundreds of data (the adaptive decoders), while the cell with
			// the most data inside it holds a fitted answer. For the decoders without per-cell
			// parameters both are empty. The custom decoder fits its region, as its user defined.
			virtual bool reuses_descendant_params(){ return true; }

			// Whether leaf_loo and leaf_kfold read the cell's parameters (`params`). Decoders that refit
			// them without the held-out data (sptlz::cross_fitted) do not, so in cross-validation the
			// ensemble fits a cell's parameters only where a coarser cell or a mark needs them.
			virtual bool held_out_uses_cell_params(){ return true; }

			virtual std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params, const CellContext &cell){
				throw std::runtime_error("must override");
			}

			virtual std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params, const CellContext &cell){
				throw std::runtime_error("must override");
			}

			virtual std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *fold, std::vector<int> *samples_id, std::vector<float> *params, const CellContext &cell){
				throw std::runtime_error("must override");
			}
	};
}

#endif

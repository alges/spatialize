#ifndef _SPTLZ_DECODER_
#define _SPTLZ_DECODER_

#include <random>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include "spatialize/partition.hpp"

namespace sptlz{
	// Where a decoder is called: the tree and the cell of the ensemble, and the run's seed. Decoders
	// that draw use them to derive their random numbers, so that a draw depends neither on the
	// number of threads nor on the other queries.
	struct CellContext {
		int tree;
		int cell;
		unsigned int seed;
	};

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

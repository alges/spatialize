#ifndef _SPTLZ_DECODER_
#define _SPTLZ_DECODER_

#include <random>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include "spatialize/partition.hpp"

namespace sptlz{
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

			virtual std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<std::vector<float>> *locations, std::vector<int> *locations_id, std::vector<float> *params){
				throw std::runtime_error("must override");
			}

			virtual std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *samples_id, std::vector<float> *params){
				throw std::runtime_error("must override");
			}

			virtual std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values, std::vector<int> *fold, std::vector<int> *samples_id, std::vector<float> *params){
				throw std::runtime_error("must override");
			}
	};
}

#endif

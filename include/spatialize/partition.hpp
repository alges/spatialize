#ifndef _SPTLZ_PARTITION_
#define _SPTLZ_PARTITION_

#include <vector>
#include <functional>

namespace sptlz{
	// One random partition of the domain (the "encoder" draw of one ensemble member).
	// Concrete partitions (MondrianTree, VoronoiTree) build their cells and assign the samples;
	// decoders only see cells through this interface.
	class Partition {
		public:
			std::vector<int> leaf_for_sample;              // cell of each sample
			std::vector<std::vector<int>> samples_by_leaf; // samples of each cell
			std::vector<std::vector<float>> leaf_params;   // per-cell decoder parameters (set by Decoder::fit)

			virtual ~Partition(){}

			// number of cells
			virtual int n_leaves() = 0;

			// cell containing a point
			virtual int search_leaf(std::vector<float> point) = 0;

			// a point representing a cell (the centre of a box, the nucleus of a Voronoi cell), from
			// which the nearness of cells is measured
			virtual std::vector<float> leaf_point(int leaf) = 0;

			// The coarser region that holds data (`usable` tells which data count), standing for an
			// empty cell: its `id` (-1 if there is none), whether it is a cell of the partition with
			// parameters already fitted (`fitted`), its usable `samples`, and, for a region made of
			// several cells (an ancestor), the cell inside it whose parameters may stand for its own
			// (`donor`: the cell with the most data, all of them usable; -1 if none).
			struct Coarse {
				int id = -1;
				bool fitted = false;
				std::vector<int> samples;
				int donor = -1;
			};
			virtual Coarse coarser(int leaf, const std::vector<float> &point, const std::function<bool(int)> &usable) = 0;
	};
}

#endif

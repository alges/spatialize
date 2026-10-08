#ifndef _SPTLZ_PARTITION_
#define _SPTLZ_PARTITION_

#include <vector>

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
	};
}

#endif

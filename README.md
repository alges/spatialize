# Spatialize: A Python/C++ library for Ensemble Spatial Analysis (ESA)
An open source library for spatial analysis that combines the simplicity of basic methods with the power of geostatistical tools.

## Overview
Spatialize implements **Ensemble Spatial Analysis (ESA)**, which encompasses two complementary approaches: **Ensemble Spatial Interpolation (ESI)** and **Ensemble Spatial Simulation (ESS)**. These novel methods address the limitations of traditional geostatistical approaches by leveraging ensemble learning techniques.

ESI works by generating multiple estimates for each target location by creating different spatial partitions of the sample data and applying an interpolation algorithm within each local subset. These local estimates are then aggregated to produce robust predictions. ESS extends this framework to provide stochastic simulation capabilities.

Designed to bridge the gap between expert and non-expert users of geostatistics, Spatialize provides automated tools that eliminate the need for manual spatial analysis and extensive domain expertise.

## Main features:
- **Automated Spatial Estimation**: Minimal user intervention required
- **Stochastic Modelling & Ensemble Learning**: Robust, scalable and suitable for large datasets
- **Uncertainty Quantification**: Provides both point estimates and empirical posterior distributions
- **Flexible Data Support**: Works with both gridded and non-gridded data
- **Hyperparameter Optimization**: Built-in grid search with cross-validation
- **High Performance**: C++ core with Python interface

## Installation
The source code is currently hosted on GitHub at:
https://github.com/alges/spatialize

Direct installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/spatialize).

### PyPI
```bash
pip install spatialize
```

### System Requirements
- Python 3.8+
- Compatible with Linux, macOS, and Windows

### Dependencies
- [NumPy: Powerful n-dimensional arrays and numerical computing tools](https://www.numpy.org)
- [pandas: Fast, powerful, flexible and easy to use open source data analysis and manipulation tool](https://pandas.pydata.org)
- [Matplotlib: Visualization with Python](https://matplotlib.org/)
- [scikit-learn: Machine Learning in Python](https://scikit-learn.org/)
- [SciPy: Fundamental algorithms for scientific computing in Python](https://scipy.org/)

## Core Concepts
| Function | Description |
|----------|-------------|
| `esi_griddata()` | Spatial interpolation for points on a regular grid |
| `esi_nongriddata()` | Spatial interpolation for scattered points |
| `esi_hparams_search()` | Automated hyperparameter optimization with cross-validation |

### Local Interpolators
- **IDW (Inverse Distance Weighting)**: Simple yet powerful with configurable distance exponent
- **Kriging**: Geostatistical method with multiple variogram models (spherical, exponential, cubic and gaussian)

### Partition Methods
- **Mondrian Forests**: Uses recursive, axis-aligned partitions (supports up to 5D)
- **Voronoi Forests**: Uses Voronoi diagram-based partitions (supports up to 2D)

## Quick Start
Here are a few examples to get you started.

### Basic Gridded Data Estimation
```python
import numpy as np
from spatialize.gs.esi import esi_griddata

# Generate sample data
def func(x, y):		# a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2

points = np.random.random((100, 2))
values = func(points[:, 0], points[:, 1])

# Define the estimation grid
grid_x, grid_y = np.mgrid[0:1:50j, 0:1:50j]

# Perform ESI estimation
result = esi_griddata(points, values, (grid_x, grid_y),
		      local_interpolator="idw",
		      p_process="mondrian",
		      n_partitions=300,
		      alpha=0.8,
		      exponent=1.0
		      )

# Get results
estimation = result.estimation()
precision = result.precision()

# Quick visualization
result.quick_plot()
```

### Non-gridded Data Estimation
```python
from spatialize.gs.esi import esi_nongriddata

# Define target locations
target_points = np.random.random((50, 2))

# Perform estimation, using Kriging as local interpolator
result = esi_nongriddata(points, values, target_points,
		         local_interpolator="kriging",
		         model="spherical",
		         nugget=0.1,
		         range=10.0,
		         sill=1.0
		         )
```

### Automated Hyperparameter Search
```python
from spatialize.gs.esi import esi_hparams_search

# Search for optimal parameters
search_result = esi_hparams_search(points, values, (grid_x, grid_y),
			           local_interpolator="idw",
			           griddata=True,
			           k=10,
			           exponent=[1.0, 2.0, 3.0, 4.0],
			           alpha=[0.7, 0.8, 0.9],
			           n_partitions=[100, 300, 500]
			           )

# Perform estimation using best parameters found
best_result = esi_griddata(points, values, (grid_x, grid_y),
			   local_interpolator="idw",
			   best_params_found=search_result.best_result()
			   )

# Visualize search results
search_result.plot_cv_error()
```

## License
[Apache-2.0](LICENSE)

## Citing Spatialize
Please refer to the following articles when publishing work relating to this library or the ESI model:

	@article{
		title = {Spatial distributional estimation via ensemble spatial analysis},
		journal = {AIMS Mathematics},
		volume = {10},
		number = {11},
		pages = {26351-26388},
		year = {2025},
		issn = {2473-6988},
		doi = {10.3934/math.20251159},
		url = {https://www.aimspress.com/article/doi/10.3934/math.20251159},
		author = {Alvaro F. Ega{\~n}a and Gonzalo D{\'i}az and Felipe Navarro and Mohammad Maleki and Juan F. S{\'a}nchez-P{\'e}rez},
		keywords = {geostatistics, computational geostatistics, generative geostatistics, non-linear geostatistics, distributional geostatistics, geostatistical simulation, empirical copula, data-driven methods},
		}

	@article{spatialize2025,
		author  = {Navarro, Felipe and Ega{\~n}a, {\'A}lvaro F. and Ehrenfeld, Alejandro and Garrido, Felipe and Valenzuela, Mar{\'i}a Jes{\'u}s and S{\'a}nchez-P{\'e}rez, Juan F. },
		title   = {Spatialize v1.0: A Python/C++ Library for Ensemble Spatial Interpolation},
		journal = {},
		year    = {2025},
		volume  = {},
		number  = {},
		pages   = {},
		doi     = {https://doi.org/10.48550/arXiv.2507.17867},
		url     = {https://arxiv.org/abs/2507.17867},
		issn    = {}
		}

	@article{AdaptiveESI2025,
		author  = {Ega{\~n}a, {\'A}lvaro F. and Valenzuela, María Jesús and Maleki, Mohammad and S{\'a}nchez-P{\'e}rez, Juan F. and Díaz, Gonzalo},
		title   = {Adaptive ensemble spatial analysis},
		journal = {Scientific Reports},
		year    = {2025},
		volume  = {15},
		number  = {1},
		pages   = {26599},
		doi     = {10.1038/s41598-025-08844-z},
		url     = {https://doi.org/10.1038/s41598-025-08844-z},
		issn    = {2045-2322}
		}

	@article{ESI2021,
		author  = {Ega{\~n}a, {\'A}lvaro F. and Navarro, Felipe and Maleki, Mohammad and Grand{\'o}n, Francisca and Carter, Francisco and Soto, Fabi{\'a}n},
		title   = {Ensemble Spatial Interpolation: A New Approach to Natural or Anthropogenic Variable Assessment},
		journal = {Natural Resources Research},
		volume  = {30},
		number  = {5},
		pages   = {3777--3793},
		year    = {2021},
		doi     = {https://doi.org/10.1007/s11053-021-09860-2},
		url     = {https://link.springer.com/article/10.1007/s11053-021-09860-2}
		}

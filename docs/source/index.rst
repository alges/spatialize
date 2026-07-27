Spatialize
==========

Spatialize is an open source Python/C++ library for **Ensemble Spatial Analysis
(ESA)**, a family of methods that combine the simplicity of basic interpolation
techniques with the power of classical geostatistical tools such as Kriging. It
bridges the gap between expert and non-expert users of geostatistics by providing
automated tools that rival traditional methods, with a Python 3.x API and a C++
core for performance.

ESA encompasses two complementary approaches. **Ensemble Spatial Interpolation
(ESI)** generates multiple estimates for each target location by building many
random space partitions of the sample data and applying a local interpolator within
each subset; these local estimates are then aggregated into robust predictions.
**Ensemble Spatial Simulation (ESS)** extends this framework to stochastic
simulation.

Features
------------

- Automated spatial estimation requiring minimal user intervention.
- Stochastic modelling and ensemble learning — robust, scalable, suitable for large datasets.
- Interpolation for both continuous and categorical data
- Uncertainty quantification: both point estimates and empirical posterior distributions.
- Works with gridded and non-gridded data, with built-in hyperparameter optimization.
- Implemented in Python 3.x with a C++ core for performance.

.. _installation:

Installation
------------

.. code-block:: bash

   pip install spatialize

Python 3.8+ is required. Spatialize is tested on Linux, macOS, and Windows.

.. _getting-started:

Getting Started
----------------

The example below interpolates scattered 2D data onto a regular grid using
ensemble spatial interpolation with an IDW local interpolator.

.. code-block:: python

   import numpy as np
   from spatialize.gs.esi import esi_griddata

   # Scattered sample data
   def f(x, y):
       return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2

   points = np.random.random((100, 2))
   values = f(points[:, 0], points[:, 1])
   grid_x, grid_y = np.mgrid[0:1:50j, 0:1:50j]

   # Ensemble spatial interpolation
   result = esi_griddata(points, values, (grid_x, grid_y),
                         local_interpolator="idw",
                         n_partitions=300, alpha=0.8, exponent=1.0)

   estimation = result.estimation()   # point estimates
   precision  = result.precision()    # uncertainty / error metric
   result.quick_plot()                # visualize

See the :doc:`API Reference <reference/index>` for the full set of parameters and interpolators.

.. _citation:

Citation
--------

If you use Spatialize in your research, please cite:

- Egaña, Á.F., Díaz, G., Navarro, F., Maleki, M., Sánchez-Pérez, J.F. (2025). *Spatial distributional estimation via ensemble spatial analysis*. AIMS Mathematics 10(11), 26351-26388. https://doi.org/10.3934/math.20251159
- Navarro, F., Egaña, Á.F., Ehrenfeld, A., Garrido, F., Valenzuela, M.J., Sánchez-Pérez, J.F. (2026). *Spatialize v1.0: a Python/C++ library for ensemble spatial interpolation*. Geoscientific Model Development 19(10), 4633-4660. https://doi.org/10.5194/gmd-19-4633-2026
- Egaña, Á.F., Valenzuela, M.J., Maleki, M. et al. (2025). *Adaptive ensemble spatial analysis*. Scientific Reports 15, 26599. https://doi.org/10.1038/s41598-025-08844-z
- Egaña, Á.F., Navarro, F., Maleki, M., et al. (2021). *Ensemble Spatial Interpolation: A New Approach to Natural or Anthropogenic Variable Assessment*. Natural Resources Research 30(5), 3777-3793. https://doi.org/10.1007/s11053-021-09860-2

.. toctree::
   :maxdepth: 1
   :hidden:

   reference/index
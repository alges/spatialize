.. _idw:

*********
Plain IDW
*********

Plain inverse distance weighting, without partitions, kept as a baseline to compare the ensemble estimates with.
The data within ``radius`` of a location are weighted by :math:`1/d^p` relative to the nearest of them,
:math:`(d_{\min}/d)^p`, as the ensemble's IDW decoder weighs the data of a cell, so the estimate does not
depend on the units of the coordinates. A datum at distance 0 takes all the weight.

.. currentmodule:: spatialize.gs.idw

.. autofunction:: idw_griddata

.. autofunction:: idw_nongriddata

.. autofunction:: idw_hparams_search

.. autoclass:: IDWResult
   :members:
   :exclude-members: load, save
   :undoc-members:
   :inherited-members:

.. autoclass:: IDWGridSearchResult
   :members:
   :exclude-members: load, save
   :undoc-members:
   :inherited-members:

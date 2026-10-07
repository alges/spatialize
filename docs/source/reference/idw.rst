.. _idw:

*********
Plain IDW
*********

Plain inverse distance weighting, without partitions, kept as a baseline to compare the ensemble estimates with.

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

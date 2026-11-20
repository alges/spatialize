.. _kriging:

**************
Plain kriging
**************

Plain ordinary or simple kriging, without partitions, with an anisotropic search neighborhood and a variogram of
one or more nested structures. Besides the estimate it gives the kriging variance. Only the search ``radius`` is
tuned by the grid search; the variogram is set by the user.

.. currentmodule:: spatialize.gs.kriging

.. autofunction:: kriging_griddata

.. autofunction:: kriging_nongriddata

.. autofunction:: kriging_hparams_search

.. autoclass:: KrigingResult
   :members:
   :exclude-members: load, save
   :undoc-members:
   :inherited-members:

.. autoclass:: KrigingGridSearchResult
   :members:
   :exclude-members: load, save
   :undoc-members:
   :inherited-members:

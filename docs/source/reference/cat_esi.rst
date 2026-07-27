.. _cat_esi:

**************************
Categorical ESI (Cat-ESI)
**************************

.. currentmodule:: spatialize.gs.cat_esi

Core functions
==============

.. autofunction:: cat_esi_griddata

.. autofunction:: cat_esi_nongriddata

.. autofunction:: cat_esi_hparams_search

Result classes
==============

.. autoclass:: CatESIResult
   :members:
   :exclude-members: load, save
   :undoc-members:
   :inherited-members:

.. autoclass:: CatESIGridSearchResult
   :members:
   :exclude-members: load, save
   :undoc-members:
   :inherited-members:

Aggregation functions
======================

.. autofunction:: aggregate_with_mv

.. autofunction:: aggregate_with_ordinal_mv

.. autofunction:: categorical_feature_precision

.. autofunction:: categorical_precision_cube

Score functions
================

.. autofunction:: accuracy

.. autofunction:: f1_macro

.. autofunction:: f1_weighted

.. autofunction:: f1_micro

.. autofunction:: precision_macro

.. autofunction:: recall_macro

.. autofunction:: cohen_kappa

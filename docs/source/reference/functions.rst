.. _functions:

**********************************
Pluggable Functions
**********************************

These are the callables that can be passed as ``agg_function``, ``loss_function``,
or used to score cross-validation results in the ESI API.

Aggregation functions
======================

.. py:module:: spatialize.gs.esi.aggfunction

.. autofunction:: mean

.. autofunction:: median

.. autofunction:: MAP

.. autofunction:: identity

.. autofunction:: bilateral_filter

.. autoclass:: Percentile
   :members:
   :special-members: __call__
   :undoc-members:

.. autoclass:: WeightedAverage
   :members:
   :special-members: __call__
   :undoc-members:

Loss functions
===============

.. py:module:: spatialize.gs.esi.lossfunction

.. autofunction:: loss

.. autofunction:: mse_loss

.. autofunction:: mae_loss

.. autofunction:: mse_cube

.. autofunction:: mae_cube

.. autoclass:: OperationalErrorLoss
   :members:
   :special-members: __call__
   :undoc-members:

Score functions
=================

.. py:module:: spatialize.gs.esi.scorefunction

.. autofunction:: mae

.. autofunction:: mse

.. autofunction:: rmse

.. autofunction:: neg_log_likelihood

.. autofunction:: crps

.. _functions:

*******************
Pluggable functions
*******************

The callables passed as ``agg_function``, which chooses the reading of the members reported as the
map (:doc:`../theory/sde`), as ``loss_function``, or as the scores of the cross-validation searches
(:doc:`selection`).

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

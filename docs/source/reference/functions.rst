.. _functions:

*******************
Pluggable functions
*******************

The callables passed as ``agg_function``, which chooses the reading of the members reported as the
map (:doc:`../theory/sde`), as ``loss_function``, or as the scores of the cross-validation searches
(:doc:`selection`). Version 1.3 renamed their modules, ``aggfunction`` becoming ``functionals``,
``lossfunction`` becoming ``loss`` and ``scorefunction`` becoming ``scoring``. The old names still
import, with a warning, until version 1.4.

Functionals
===========

.. py:module:: spatialize.gs.esi.functionals

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

.. autoclass:: Expectile
   :members:
   :special-members: __call__

.. autoclass:: ExpectedShortfall
   :members:
   :special-members: __call__

.. autoclass:: AsymmetricLoss
   :members:
   :special-members: __call__

Loss functions
===============

.. py:module:: spatialize.gs.esi.loss

.. autofunction:: loss

.. autofunction:: mse_loss

.. autofunction:: mae_loss

.. autofunction:: mse_cube

.. autofunction:: mae_cube

.. autoclass:: OperationalErrorLoss
   :members:
   :special-members: __call__
   :undoc-members:

Scoring
=======

.. py:module:: spatialize.gs.esi.scoring

.. autofunction:: mae

.. autofunction:: mse

.. autofunction:: rmse

.. autofunction:: neg_log_likelihood

.. autofunction:: crps

.. autofunction:: asymmetric_loss

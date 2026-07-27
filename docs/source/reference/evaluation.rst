.. _evaluation:

********************
Evaluation Toolkit
********************

.. currentmodule:: spatialize.evaluation

Metric functions
==================

.. autofunction:: error

.. autofunction:: absolute_error

.. autofunction:: bias

.. autofunction:: MAE

.. autofunction:: MSE

.. autofunction:: RMSE

.. autofunction:: R2

.. autofunction:: accuracy

.. autofunction:: precision

.. autofunction:: recall

.. autofunction:: f1_score

.. autofunction:: operational

.. autofunction:: op_error

.. autofunction:: op_mae

.. autofunction:: op_rmse

.. autofunction:: op_mse

Validation helpers
====================

.. autofunction:: loo_validation

.. autofunction:: kfold_validation

Baselines
==========

.. autofunction:: auto_krige

.. autofunction:: auto_scipy_prediction

.. autofunction:: kriging_prediction

.. autofunction:: scipy_prediction

Scenario generators
======================

.. autoclass:: SyntheticScenario
   :members:
   :undoc-members:

.. autoclass:: PrecipitationCaseStudy
   :members:
   :undoc-members:

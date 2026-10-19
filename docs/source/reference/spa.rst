.. _spa:

******************
Posterior analysis
******************

The cross-validated predictive law at each datum and its analysis, which ranks the data by how far into the tails of their laws they fall, checks the calibration of the laws, reads the evidence behind each of them and turns the readings into a list of actions, the data to measure again and the places to sample more (:doc:`../theory/posterior`).

.. currentmodule:: spatialize.gs.spa

.. autofunction:: posterior_audit

.. autoclass:: PosteriorAudit
   :members:

Version 1.2 interface
=====================

The names of version 1.2 work on top of :func:`posterior_audit`.

.. autofunction:: cv_sample_pred_posterior

.. autoclass:: PosteriorSampleAnalyzer
   :members:
   :exclude-members: load, save
   :undoc-members:
   :inherited-members:

.. currentmodule:: spatialize.gs.spa._main

.. autofunction:: plot_histogram_grid_with_pdf_cdf

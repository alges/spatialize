.. _futures:

*******
Futures
*******

.. automodule:: spatialize.futures
   :no-members:

.. warning::

   The features on this page are under study. They work and are tested, while their theory or
   their design is still being settled, so their interface and their results may change between
   versions without notice. Each one warns once per session with an
   :class:`~spatialize.futures.ExperimentalWarning`.

.. autoclass:: spatialize.futures.ExperimentalWarning

Information measures
====================

The entropy of the predictive law at each location, and the mutual information between the laws of
two variables measured over the same region (:doc:`../theory/other`).

.. currentmodule:: spatialize.futures.esmi

.. autoclass:: SpatialEntropy
   :members:

.. autoclass:: SpatialMutualInformation
   :members:

Co-estimation
=============

.. automodule:: spatialize.futures.coesi
   :no-members:

.. currentmodule:: spatialize.futures.coesi

.. autofunction:: coesi_nongriddata

.. autofunction:: coesi_marginal_cv

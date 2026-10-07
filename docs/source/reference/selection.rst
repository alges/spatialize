.. _selection:

****************
Choosing a model
****************

The searches over the granularity, the number of partitions and the decoder's parameters, by
cross-validation alone or on the encoder–decoder Pareto frontier (:doc:`../theory/error`). The
scores they use are listed with the other pluggable functions (:doc:`functions`).

.. currentmodule:: spatialize.gs.esi

.. autofunction:: esi_hparams_search

.. autofunction:: esi_pareto_hparams_search

.. autoclass:: ESIGridSearchResult
   :members:
   :exclude-members: load, save
   :undoc-members:
   :inherited-members:

.. autoclass:: ESIParetoResult
   :members:
   :exclude-members: load, save
   :undoc-members:
   :inherited-members:

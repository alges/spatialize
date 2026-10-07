.. _esi:

******************************
Ensemble spatial interpolation
******************************

Estimation on a grid or at a list of locations, returning the members of the ensemble at each
location together with the reported map and its uncertainty (:doc:`../theory/esi`). The partition
is chosen with ``p_process`` (:doc:`../theory/encoders`) and the local model with
``local_interpolator`` (:doc:`../theory/decoders`).

.. currentmodule:: spatialize.gs.esi

.. autofunction:: esi_griddata

.. autofunction:: esi_nongriddata

.. autoclass:: ESIResult
   :members:
   :exclude-members: load, save
   :undoc-members:
   :inherited-members:

.. _theory-decoders:

########
Decoders
########

The decoder predicts at a location from the data of its cell alone. Since the cells are small and
change from one partition to the next, a decoder can be cheap, and the choice among them changes
what the ensemble is good at.

Averaging and drawing
=====================

Decoders come in two families.

- An *averaging* decoder returns a weighted average of the cell's data. With positive weights the
  prediction stays within the range of the data. The members at a location vary only because the
  cells vary, so the ensemble law captures the uncertainty that comes from the partition and misses
  the dispersion of the values inside a cell. Its maps are smooth, while its intervals tend to be too narrow.
- A *drawing* decoder returns one of the cell's data, drawn at random with the weights of an
  averaging decoder. The mean of its draws is that averaging decoder's prediction, while their
  spread is the dispersion of the cell around it. The ensemble then carries both sources of
  uncertainty without anything added afterwards. Every member is a value that was actually observed. Its intervals cover better, while each member is rougher than an averaged one.

The averaging decoders
======================

**Cell mean** (``cellmean``). Every location of a cell gets the mean of the cell's data, so all the
spatial detail comes from the partition. Prefer it when the object of study is the latent geometry
of the field, which blocks of values hang together, since the theory's relations between the field's
dependence and the partition hold exactly for it.

**Inverse distance weighting** (``idw``). The weights fall with distance as :math:`1/d^p`. A large
exponent :math:`p` makes the nearest datum dominate, a small one approaches the cell mean. Prefer it
as a fast, transparent baseline, on fields without a marked direction, and for quick
hyperparameter searches.

**Kriging** (``kriging``). The weights solve the kriging system of a variogram model inside each
cell. Prefer it when a variogram is known and trusted, on smooth fields close to Gaussian. Its
weights can be negative, so the estimate can leave the range of the data, which makes it a poor
choice for skewed or heavy-tailed variables and for variables with many zeros.

**Adaptive IDW** (``adaptiveidw``). Inside each cell it fits an exponent, a direction and an
elongation by leave-one-out, so the weights follow the anisotropy of the field without being told
about it. Prefer it when the field is continuous along a direction that is unknown or varies across
the domain. The per-cell fit makes it slower than IDW. It works in two and three dimensions.

**Sharpened adaptive IDW** (``sharpidw``). It starts from the adaptive fit and changes it in two
ways. A datum that the rest of its cell predicts badly gets more weight, since on a smooth field a
large residual marks a place where the field moves, which is where the datum informs. The
exponent rises in a cell with a strong gradient, so the prediction leans on the nearest data
instead of averaging across the change. The weights stay positive, so the prediction stays within
the range of the data. On the theory's anisotropic test field it gives a better map than adaptive
IDW on every field tried. Prefer it when the map itself is the goal. Its constants
``kappa_r``, ``kappa_g`` and ``rho_max`` default to the theory's values, and setting both kappas to
0 gives adaptive IDW exactly.

The drawing decoders
====================

**Uniform draw** (``draw``). The member at a location is a datum of its cell, every datum equally
likely. The law at a location is then the shares of the values its cells contain, whatever its
position inside them, so the maps lose contrast while the intervals reach their nominal coverage.
Prefer it for piecewise-constant fields and for a variable with an atom, such as daily rainfall with its many dry days. An average turns the dry days near wet ones into small positive amounts, while a draw keeps them at zero.

**Weighted draws** (``wdraw_idw``, ``wdraw_adaptiveidw``, ``wdraw_sharpidw``, ``wdraw_kriging``).
The member is a datum drawn with the weights of the decoder named, so its mean is that decoder's prediction. Its map is almost as accurate, while the ensemble covers its intervals far more often. They keep atoms and the support of the data. On heavy-tailed fields their median is the most reliable reading among the decoders. Choose the weights as for the averaging decoders, IDW for
simplicity, adaptive for anisotropy, sharpened for gradients, kriging when the variogram is trusted.
The kriging weights are made non-negative first, by setting the negative ones to 0 (the default) or
by taking their absolute value.

Choosing by purpose
===================

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Purpose
     - Decoder
   * - a map of best values
     - ``sharpidw``, or ``adaptiveidw`` when speed matters
   * - intervals and probabilities of exceedance
     - a weighted draw, ``wdraw_sharpidw`` or ``wdraw_adaptiveidw``
   * - a variable with an atom (zeros, detection limits)
     - ``draw`` or a weighted draw
   * - a heavy-tailed variable
     - a weighted draw, read through its median
   * - the latent geometry of the field
     - ``cellmean``
   * - a known variogram on a smooth field
     - ``kriging``

In Spatialize
=============

Every decoder is a value of ``local_interpolator`` in the public functions
(:func:`~spatialize.gs.esi.esi_griddata`), with its parameters as keyword arguments. The
conformance tests check that every member of a drawing decoder is an observed value and that the
mean and spread of its draws match the averaging decoder (:doc:`../scenarios/catalog_estimator_properties`).

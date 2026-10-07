.. _theory-esi:

###############################
Ensemble spatial interpolation
###############################

Ensemble spatial interpolation (ESI) is the estimator of :doc:`sde` read as an interpolator. Given
data :math:`O` and locations :math:`v_1, \dots, v_N` to estimate, on a grid or in a list, it returns
the members :math:`\hat z^{(t)}(v_k)`, :math:`t \le T`, from which every reading follows.

The map
=======

The map reports one functional of the members at each location,

.. math::

   \hat z(v) = A\big(\hat z^{(1)}(v), \dots, \hat z^{(T)}(v)\big),

with :math:`A` the aggregation, the mean by default. The median suits skewed variables and is the
safer report for heavy-tailed ones. A quantile suits a use where over- and under-prediction cost
differently (:doc:`sde`). No variogram is needed, since the partition supplies the spatial coupling
while the decoder adapts to the data inside each cell.

The uncertainty
===============

The same members give the uncertainty of the map, location by location.

- The *precision*, the average loss of the members around the reported value,

  .. math::

     \operatorname{prec}(v) = \frac1T \sum_{t=1}^T L\big(\hat z(v), \hat z^{(t)}(v)\big),

  which with the squared loss :math:`L(a, z) = (a - z)^2` is the variance of the members when
  :math:`\hat z` is their mean. It measures how much the estimate depends on the partition.
- The probability of exceeding a threshold :math:`t`, :math:`\hat p_t(v) = \frac1T \sum_t
  \mathbf 1\{\hat z^{(t)}(v) > t\}`.
- An interval :math:`[\hat q_{\alpha/2}(v), \hat q_{1-\alpha/2}(v)]` of nominal coverage
  :math:`1 - \alpha`.

With an averaging decoder the members vary only through the cells. By the law of total variance,
the variance of the field near :math:`v` splits as

.. math::

   \operatorname{Var} Z = \mathbb E\big[\operatorname{Var}(Z \mid C)\big]
   + \operatorname{Var}\big(\mathbb E[Z \mid C]\big),

over the cells :math:`C` containing :math:`v`. An averaging decoder only sees the second term.
Intervals read from its raw members are therefore too narrow. A drawing decoder (:doc:`decoders`)
puts the first term back into the members, while the widened laws of simulation (:doc:`ess`) restore it afterwards.

Coherence
=========

Every reading comes from one ensemble, so the readings agree with one another. At each location
:math:`\hat p_t(v)` never rises with :math:`t`, and :math:`\hat q_\alpha(v)` never falls with
:math:`\alpha`. Across locations, the members are fields, so a question about many locations at
once, an area or a block average, is answered from the members directly.

Variables
=========

The same construction handles continuous variables in any dimension the decoder supports, and
categorical variables, whose decoders are classifiers fitted inside each cell (:doc:`other`).

In Spatialize
=============

:func:`~spatialize.gs.esi.esi_griddata` estimates on a grid and
:func:`~spatialize.gs.esi.esi_nongriddata` at a list of locations. Their result holds the members
(``esi_samples``), the map (``estimation``) and the precision (``precision``, with its
``loss_function``), together with plotting helpers (:doc:`../reference/esi`).

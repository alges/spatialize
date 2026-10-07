.. _theory-ess:

###############################
Ensemble spatial simulation
###############################

Simulation draws whole fields that honour the estimated laws, for questions whose answer needs many
plausible versions of the field, a flow through a porous medium, a mine plan tested against grade
uncertainty, a risk computed over a region.

Fields from marginals
=====================

Ensemble spatial simulation (ESS) fits a density :math:`\hat p_{v_i}` to the members at each location
:math:`v_i`, then draws every location independently from its own,

.. math::

   \tilde Z_i^{(m)} \sim \hat p_{v_i}
   \quad\text{independently over } i = 1, \dots, N \text{ and } m = 1, \dots, M,

so the :math:`M` simulated fields sample the product law :math:`\prod_i \hat p_{v_i}`. Drawing
independently sounds like a recipe for noise, yet the fields carry the geometry of the variable.
Neighbouring locations are estimated from almost the same data, so their laws almost coincide. Two
independent draws from two nearly equal laws come back nearly equal, so the structure lies in the
slow drift of :math:`v \mapsto \hat p_v` across space, not in the draws.

The draws carry the family of local laws, not the full joint law. The members of the ensemble carry
more, since each is a field computed under one partition, with the dependence the co-occurrence
induces. For a question about many locations at once, such as the area above a limit
:math:`|\{v : Z(v) > t\}|`, the members give the right spread of the answer, while independently
simulated fields understate it.

Widening
========

The members at a location vary only as far as the cells around it vary, so their variance
:math:`s^2` understates how much the variable changes near the location, by the variance of the
local means (:doc:`esi`). Simulation therefore widens each local law to a target variance
:math:`\tau^2`, read from the data nearest to the location, typically the variance of its :math:`k`
nearest data.

With members :math:`z_1, \dots, z_T` of mean :math:`\bar z` and variance :math:`s^2`, write
:math:`\mathrm{need} = \tau^2 - s^2`. When :math:`\mathrm{need} \le 0` the members already spread enough, so their law is kept. Otherwise each member is replaced by a law :math:`G_{z_t,
\mathrm{need}}` of mean :math:`z_t` and variance :math:`\mathrm{need}`. The widened law is the mixture

.. math::

   \hat p_v = \frac1T \sum_{t=1}^T G_{z_t,\, \mathrm{need}},
   \qquad
   \mathbb E\, \hat p_v = \bar z,
   \qquad
   \operatorname{Var} \hat p_v = \mathrm{need} + s^2 = \tau^2,

the variance following from the law of total variance over the mixture index. The family
:math:`G` suits the variable, Gamma laws for positive variables, which keep the draws positive,
and skew-normal laws otherwise, with a skewness read from the nearest data.

A drawing decoder (:doc:`decoders`) puts the dispersion of the cells into the members themselves, so
:math:`s^2` is larger and less widening is needed.

Fitting the local laws
======================

The density :math:`\hat p_v` is fitted to the members by a kernel density estimate,

.. math::

   \hat p_v(z) = \frac{1}{T h} \sum_{t=1}^T K\Big(\frac{z - \hat z^{(t)}(v)}{h}\Big),

which follows the members closely, or by a mixture of Gaussians, which summarises them with a few
components and suits members that take a few distinct values.

In Spatialize
=============

:func:`~spatialize.gs.ess.ess_sample` simulates from the result of an ESI estimation. The densities
and the widening are set by :class:`~spatialize.empirical.FittedModelFactory`, whose
``point_model_name`` chooses the density (``"kde"``, ``"emm"``, ``"vim"``), ``widening`` the family
(``"gamma"``, ``"skew_normal"``, or ``"auto"``, which takes Gamma laws when the members are
non-negative) and ``widening_knn`` the number :math:`k` of nearest data the
target :math:`\tau^2` is read from (:doc:`../reference/ess`, :doc:`../reference/empirical`).

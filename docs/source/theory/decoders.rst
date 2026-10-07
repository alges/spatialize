.. _theory-decoders:

########
Decoders
########

The decoder :math:`g` predicts at :math:`v` from the data :math:`(v_j, z_j)_{j \le m}` of its cell
alone. Since the cells are small and change from one partition to the next, a decoder can be cheap,
and the choice among them changes what the ensemble is good at.

Averaging and drawing
=====================

Every decoder of Spatialize starts from non-negative weights :math:`w_j(v)` on the data of the cell
and does one of two things with them.

An *averaging* decoder returns their weighted mean,

.. math::

   g(v) = \sum_{j=1}^m \varpi_j(v)\, z_j,
   \qquad
   \varpi_j(v) = \frac{w_j(v)}{\sum_{l} w_l(v)} .

With non-negative weights the prediction lies in :math:`[\min_j z_j, \max_j z_j]`. The members at
:math:`v` vary only because the cells vary, so the ensemble captures the uncertainty that comes
from the partition and misses the dispersion of the values inside a cell. Its maps are smooth, while
its intervals tend to be too narrow.

A *drawing* decoder returns one of the cell's data, drawn with the same probabilities,

.. math::

   g^{\text{draw}}(v) = z_J, \qquad \Pr(J = j) = \varpi_j(v),

the index :math:`J` drawn afresh at each location and each partition. Its law within the cell has

.. math::

   \mathbb E\big[g^{\text{draw}}(v)\big] = \sum_j \varpi_j(v)\, z_j = g(v),
   \qquad
   \operatorname{Var}\big[g^{\text{draw}}(v)\big] = \sum_j \varpi_j(v)\,\big(z_j - g(v)\big)^2,

so the draw agrees with the averaging decoder in mean, while its variance is the weighted dispersion
of the cell around that mean. The ensemble then carries both sources of uncertainty without anything added afterwards. Every member is a value that was actually observed,
:math:`g^{\text{draw}}(v) \in \{z_1, \dots, z_m\}`. Its intervals cover better, while each member is
rougher than an averaged one.

The weights
===========

**Cell mean** (``cellmean``), :math:`w_j = 1`. Every location of a cell gets the mean of the cell's
data, so all the spatial detail comes from the partition. Prefer it when the object of study is the
latent geometry of the field, which blocks of values hang together, since the theory's relations
between the field's dependence and the co-occurrence hold exactly for it.

**Inverse distance weighting** (``idw``),

.. math::

   w_j(v) = \lVert v - v_j \rVert^{-p},

with a datum at distance 0 taking all the weight. A large exponent :math:`p` makes the nearest datum
dominate, while :math:`p \to 0` approaches the cell mean. Prefer it as a fast, transparent baseline,
on fields without a marked direction, and for quick hyperparameter searches.

**Kriging** (``kriging``). The weights :math:`\lambda_j(v)` solve the ordinary kriging system of a
variogram :math:`\gamma` inside the cell,

.. math::

   \begin{pmatrix} \Gamma & \mathbf 1 \\ \mathbf 1^\top & 0 \end{pmatrix}
   \begin{pmatrix} \boldsymbol\lambda(v) \\ \mu \end{pmatrix}
   =
   \begin{pmatrix} \boldsymbol\gamma(v) \\ 1 \end{pmatrix},
   \qquad
   \Gamma_{jl} = \gamma(v_j - v_l), \quad \boldsymbol\gamma(v)_j = \gamma(v - v_j).

The prediction is :math:`\sum_j \lambda_j(v) z_j`. Prefer it when a variogram is known and
trusted, on smooth fields close to Gaussian. The :math:`\lambda_j` can be negative, so the estimate
can leave the range of the data, which makes it a poor choice for skewed or heavy-tailed variables
and for variables with many zeros.

**Adaptive IDW** (``adaptiveidw``). Inside each cell the distance is anisotropic,

.. math::

   d_\theta(u, v) = \lVert A_\theta (u - v) \rVert,
   \qquad
   w_j(v) = d_\theta(v_j, v)^{-p},

where :math:`A_\theta` rotates by an azimuth :math:`\varphi` and stretches by an elongation
:math:`a_f`, three angles and two elongations in three dimensions. The exponent and
:math:`\theta = (\varphi, a_f)` are fitted once per cell by minimising the leave-one-out error of
the cell's data. Prefer it when the field is continuous along a direction that is unknown or varies
across the domain. The per-cell fit makes it slower than IDW. It works in two and three dimensions.

**Sharpened adaptive IDW** (``sharpidw``). It starts from the adaptive fit :math:`(p, \theta)` of
the cell and changes it twice. The leave-one-out residuals of that fit and their median,

.. math::

   r_j = z_j - \frac{\sum_{l \ne j} d_\theta(v_l, v_j)^{-p}\, z_l}{\sum_{l \ne j} d_\theta(v_l, v_j)^{-p}},
   \qquad s = \operatorname{med}_j |r_j|,

measure how badly the rest of the cell predicts each datum. On a smooth field a large residual
marks a place where the field moves, which is where the datum informs, so such data get more weight.
The slope :math:`\hat\beta` of the least-squares plane through the cell, in the transformed
coordinates :math:`\tilde v_j = A_\theta v_j`, measures how much the field changes across the cell,

.. math::

   \varrho_C = \frac{\lVert \hat\beta \rVert\, \bar\ell_C}{\operatorname{med}_j |z_j - \bar z|},
   \qquad
   \bar\ell_C = \frac1m \sum_j \lVert \tilde v_j - \bar{\tilde v} \rVert,

and where it is large the exponent rises, so the prediction leans on the nearest data instead of
averaging across the change. The weights are

.. math::

   w_j(v) = \Big(1 + \kappa_r \frac{|r_j|}{s}\Big)\, d_\theta(v_j, v)^{-p_C},
   \qquad
   p_C = p\,\big(1 + \kappa_g \min(\varrho_C, \varrho_{\max})\big),

with the theory's constants :math:`\kappa_r = 3/2`, :math:`\kappa_g = 1/5` and
:math:`\varrho_{\max} = 3` as defaults. They are positive, so the prediction stays within the range
of the data. Setting :math:`\kappa_r = \kappa_g = 0` returns adaptive IDW exactly. On the theory's
anisotropic test field it gives a better map than adaptive IDW on every field tried. Prefer it when
the map itself is the goal.

The drawing decoders
====================

**Uniform draw** (``draw``), the draw with :math:`w_j = 1`. If the cell meets blocks of the field
holding distinct values :math:`m_1, \dots, m_r`, with :math:`n_i` of its :math:`m` data in block
:math:`i`,

.. math::

   \Pr\big(g^{\text{draw}}(v) = m_i\big) = \frac{n_i}{m}, \qquad i = 1, \dots, r,

at every :math:`v` of the cell, whatever its position inside it. The maps lose contrast, while the
intervals reach their nominal coverage. Prefer it for piecewise-constant fields and for a variable
with an atom, such as daily rainfall with its many dry days. An average turns the dry days near wet
ones into small positive amounts, while a draw keeps them at zero.

**Weighted draws** (``wdraw_idw``, ``wdraw_adaptiveidw``, ``wdraw_sharpidw``, ``wdraw_kriging``),
the draw with the weights of the decoder named. By the identities above their mean is that
decoder's prediction, so their map is almost as accurate, while the ensemble covers its intervals
far more often. They keep atoms and the support of the data. On heavy-tailed fields their median is
the most reliable reading among the decoders. The kriging weights are made non-negative first,

.. math::

   w_j = \max(\lambda_j, 0) \quad (\texttt{"clip"}, \text{the default})
   \qquad\text{or}\qquad
   w_j = |\lambda_j| \quad (\texttt{"abs"}).

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
(:func:`~spatialize.gs.esi.esi_griddata`), with :math:`p` as ``exponent``, the variogram as
``model``, ``nugget``, ``range`` and ``sill``, and :math:`\kappa_r, \kappa_g, \varrho_{\max}` as
``kappa_r``, ``kappa_g`` and ``rho_max``. The conformance tests check that every member of a drawing
decoder is an observed value and that the mean and variance of its draws match the identities above
(:doc:`../scenarios/catalog_estimator_properties`).

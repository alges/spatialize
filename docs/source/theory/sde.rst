.. _theory-sde:

#################################
Spatial distributional estimation
#################################

The estimator
=============

The estimator cuts the domain into cells at random, fits a cheap model inside each cell, and
repeats. Two ingredients define it.

- The *encoder* is a random partition :math:`\Pi` of the domain, drawn from a partition process
  :math:`\Pr_\Pi`. Write :math:`C_\Pi(v)` for the cell containing :math:`v`, and
  :math:`O_\Pi(v) = \{(v_i, z_i) \in O : v_i \in C_\Pi(v)\}` for the data it holds
  (:doc:`encoders`).
- The *decoder* is a local model :math:`g` that predicts at :math:`v` from those data alone
  (:doc:`decoders`).

Drawing :math:`T` independent partitions :math:`\Pi^{(1)}, \dots, \Pi^{(T)}` gives at every
location :math:`T` predictions, the *members* of the ensemble,

.. math::

   \hat z^{(t)}(v) = g\big(v;\, O_{\Pi^{(t)}}(v)\big), \qquad t = 1, \dots, T.

The estimate at :math:`v` is their empirical law, the *predictive law*

.. math::

   \hat F_v^{(T)}(z) = \frac{1}{T} \sum_{t=1}^T \mathbf 1\big\{\hat z^{(t)}(v) \le z\big\}.

Each :math:`\hat F_v^{(T)}` is a distribution function by construction, which settles the first
requirement of :doc:`problem`.

Why the laws fit together
=========================

The members at different locations are computed under the same partitions, so the vector
:math:`\big(\hat z^{(t)}(v_1), \dots, \hat z^{(t)}(v_N)\big)` is one field drawn under
:math:`\Pi^{(t)}`, and the :math:`T` of them sample a joint law. Its dependence comes from the
sharing of cells. The *co-occurrence* of a set of locations :math:`S` is the probability that they
fall in one cell,

.. math::

   e(S) = \Pr_\Pi\big(\text{all of } S \text{ lie in one cell of } \Pi\big),

a property of the partition process alone. Two locations with a high co-occurrence are predicted
from the same data most of the time, so their members move together, while two locations that
rarely share a cell are nearly independent. For the Mondrian process of rate :math:`\lambda`, for
instance,

.. math::

   e(\{x, y\}) = \exp\big(-\lambda \lVert x - y \rVert_1\big).

The co-occurrence of pairs plays the role of the covariance, while the co-occurrence of triples and
larger sets carries the dependence of higher order that a covariance cannot hold. This settles the
second requirement.

Readings of the law
===================

Every quantity reported is a functional of the members.

.. math::

   \begin{gathered}
   \bar z(v) = \frac1T \sum_t \hat z^{(t)}(v), \qquad
   \hat q_\alpha(v) = \inf\{z : \hat F_v^{(T)}(z) \ge \alpha\},
   \\
   \hat p_t(v) = \frac1T \sum_t \mathbf 1\{\hat z^{(t)}(v) > t\},
   \end{gathered}

the mean, the quantile of level :math:`\alpha` and the probability of exceeding :math:`t`, with
:math:`[\hat q_{\alpha/2}(v), \hat q_{1-\alpha/2}(v)]` an interval of nominal coverage
:math:`1-\alpha`.

Which single number to report depends on the cost of an error, the best report being

.. math::

   \hat z^\star(v) = \arg\min_{a} \frac1T \sum_t L\big(a, \hat z^{(t)}(v)\big)

for a loss :math:`L`. A squared loss gives the mean, an absolute loss the median. When
over-predicting costs :math:`c` times as much as under-predicting,

.. math::

   L(a, z) = c\,(a - z)_+ + (z - a)_+
   \quad\Longrightarrow\quad
   \hat z^\star(v) = \hat q_{1/(1+c)}(v),

a quantile, which needs no second moment and so remains meaningful for heavy-tailed variables, where
the mean misleads. Choosing the aggregation of the members is choosing that loss.

One location against the whole map
==================================

A question about one location, such as :math:`\hat p_t(v)`, is read from the law at :math:`v`. A
question about many locations at once is read from the members, each a whole field. The average
over a block :math:`B` of :math:`m` locations, for instance, has the law of

.. math::

   \bar Z_B^{(t)} = \frac1m \sum_{v \in B} \hat z^{(t)}(v), \qquad t = 1, \dots, T,

whose quantiles are not the averages of the quantiles at the points of :math:`B`. Averaging the
quantiles assumes that all the locations of the block move together, and overstates the risk of the
block.

Cells without data
==================

A partition can leave a location in a cell that holds none of the data, more often far from the
data and with fine partitions. The decoder then has nothing to predict from. The theory's answer,
which Spatialize follows on request, comes from the block-mark model (:doc:`blockmark`). Such a
cell receives one value, drawn by default from the cells with data around it, shared by every
location in it. The share of partitions in which it happens, the *residual weight*, measures how far a location lies outside the sample.

The number of partitions
========================

Each reading is an average over :math:`T` independent draws of the partition, so its Monte Carlo
error shrinks as

.. math::

   \operatorname{sd}\big(\bar z(v)\big) = \frac{\sigma_v}{\sqrt T},

with :math:`\sigma_v` the spread of the members at :math:`v`. A few hundred partitions usually make
it negligible. More partitions do not change what the ensemble estimates, so they cannot remove a
bias that comes from the partition scale or from the decoder.

What the ensemble converges to
==============================

As :math:`T \to \infty` the predictive law converges to the law of :math:`g(v; O_\Pi(v))` under
:math:`\Pi \sim \Pr_\Pi`, with the data held fixed. As the data grow denser, filling the cells, that law converges in turn. Its limit describes how the predictions from this one realisation of the
field vary as the cells move, not how the field would vary from one realisation to another. In
practice the spread of the members reflects the partition, so it tends to understate the
variability of the field near a location. Simulation corrects for it by widening the local laws
(:doc:`ess`), and the choice of granularity balances the errors involved (:doc:`error`).

In Spatialize
=============

:func:`~spatialize.gs.esi.esi_griddata` and :func:`~spatialize.gs.esi.esi_nongriddata` compute the
ensemble with :math:`T` = ``n_partitions``. The members :math:`\hat z^{(t)}(v)` are
``esi_samples()`` of the result, and ``agg_function`` (:doc:`../reference/functions`) chooses the
reading reported as the estimate. The session setting ``empty_cells`` (:mod:`spatialize.session`) chooses the treatment of cells
without data, and ``empty_cell_fraction()`` of the result estimates the residual weight at each
location.

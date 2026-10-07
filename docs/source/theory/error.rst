.. _theory-error:

###############################
Error and the choice of a model
###############################

Two sources of error
====================

Write :math:`X` for a location drawn at random in the domain, :math:`Z = Z(X)` for the value there and
:math:`U = \eta(X)` for the cell the encoder assigns it, so that the estimator predicts :math:`Z` from
:math:`U` alone, with a law :math:`q^\theta_{Z \mid U}` given by the decoder of parameters
:math:`\theta`. Its quality is measured by the expected log-loss, whose excess over the irreducible
entropy :math:`H(Z \mid X)` is the *total approximation error*,

.. math::

   \mathrm{TAE} = \mathbb E\big[-\log q^\theta(Z \mid U)\big] - H(Z \mid X).

It splits exactly into two parts,

.. math::

   \mathrm{TAE} = \underbrace{\delta(\eta, \theta)}_{\text{decoder error}}
   + \underbrace{\varepsilon(\eta)}_{\text{encoder error}},
   \qquad
   \varepsilon(\eta) = I(X; Z \mid U),
   \qquad
   \delta(\eta, \theta) = \mathbb E_U\, D\big(\mu_{Z \mid U} \,\big\Vert\, q^\theta_{Z \mid U}\big).

- The *encoder error* :math:`\varepsilon` is the information about :math:`Z` that the exact
  location carries beyond its cell. It vanishes when the law of :math:`Z` is the same everywhere
  inside each cell, and grows when cells pool places with different laws. It depends on the partition
  alone, whatever the decoder.
- The *decoder error* :math:`\delta` is the Kullback–Leibler divergence between the law of the
  values inside a cell and the law the decoder assigns them.

The two move in opposite directions as the cells shrink. Finer cells are more homogeneous, which
lowers :math:`\varepsilon`, but they hold fewer data, which raises :math:`\delta`. The total is
minimised at an intermediate granularity, which model selection looks for.

What cross-validation sees
==========================

Cross-validation predicts each datum from the others. With a strictly proper score its risk
:math:`R_{\mathrm{CV}}(\lambda)`, for a configuration :math:`\lambda` of the hyperparameters,
estimates :math:`H(Z \mid X) + \delta + \varepsilon`, so

.. math::

   \liminf_{n \to \infty} R_{\mathrm{CV}}(\lambda) \;\ge\; H(Z \mid X) + \varepsilon(\lambda)
   \quad\text{in probability.}

No decoder lowers that floor, since :math:`\varepsilon` does not depend on the decoder. A search
that minimises the cross-validation risk alone picks the best decoder for each partition, while the
encoder error of the partition stays hidden in the floor.

The encoder error can be bounded from the data. Fit a law :math:`\hat p_i` to the cross-validated
predictions at each datum, and compare the laws at pairs of data that share a cell,

.. math::

   \hat\varepsilon = \max_{t \le T}\ \max_{u \in \Pi^{(t)}}\ \max_{i, j \,:\, v_i, v_j \in u}
   D\big(\hat p_i \,\Vert\, \hat p_j\big),

the largest divergence among data that one of the partitions puts in the same cell :math:`u`, that is,
among neighbours the partition treats as alike. Model selection then
minimises the risk under a constraint on it,

.. math::

   \min_{\lambda}\ R_{\mathrm{CV}}(\lambda)
   \qquad\text{subject to}\qquad
   \hat\varepsilon(\lambda) \le \tau,

so that robustness is a requirement and not one more term to trade away.

The Pareto frontier
===================

Each configuration gives a pair of errors :math:`(\varepsilon(\lambda), \delta(\lambda))`. A
configuration is *Pareto-optimal* when no other has both errors at most as large and one strictly
smaller. Solving the constrained problem for every :math:`\tau \ge 0` traces the set of
Pareto-optimal configurations, the *frontier*. The frontier need not be convex, so minimising a
weighted sum :math:`\delta + \gamma\, \varepsilon` misses parts of it, which is why the constrained
form is used. In practice it is read in one of three ways.

- The configuration of lowest decoder error, when the partition is trusted.
- The *knee*, where lowering one error starts to cost much more of the other.
- The best configuration under a level :math:`\tau` of encoder error.

Scoring a predictive law
========================

The cross-validation score :math:`S(\hat F_{v_i}, z_i)` compares the predictive law at a held-out
datum with its value. The absolute and squared errors of a point reading judge the map,

.. math::

   S = |\hat z(v_i) - z_i| \qquad\text{or}\qquad S = \big(\hat z(v_i) - z_i\big)^2,

while the negative log-likelihood and the continuous ranked probability score judge the whole law,

.. math::

   S = -\log \hat f_{v_i}(z_i),
   \qquad
   S = \int_{-\infty}^{\infty} \big(\hat F_{v_i}(z) - \mathbf 1\{z \ge z_i\}\big)^2\, dz,

with :math:`\hat f` a density fitted to the members. The scores of the law need enough partitions to
estimate it at each datum, a few tens at least.

In Spatialize
=============

:func:`~spatialize.gs.esi.esi_hparams_search` searches by cross-validation alone, over
``n_partitions``, ``alpha`` and the decoder's parameters, with k-fold or leave-one-out.
:func:`~spatialize.gs.esi.esi_pareto_hparams_search` estimates :math:`\hat\varepsilon` and the risk for
each configuration and returns the frontier. Its ``best_result`` reads the frontier by lowest
decoder error (``"min_decoder"``) or by knee (``"knee"``), while ``best_for_tau`` picks the best
configuration under a level :math:`\tau` (:doc:`../reference/selection`). The scores are in
:mod:`spatialize.gs.esi.scorefunction`.

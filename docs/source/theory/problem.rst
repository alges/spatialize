.. _theory-problem:

###########################
Estimating a law, not a map
###########################

The setting
===========

Let :math:`Z = \{Z(v) : v \in D\}` be a random field on a domain :math:`D \subset \mathbb R^d`, of
which we observe one realisation at :math:`n` locations,

.. math::

   O = \{(v_i, z_i)\}_{i=1}^n, \qquad z_i = Z(v_i).

A map reports one value :math:`\hat z(v)` at each location. The object Spatialize estimates is
larger, the law of the field given the data,

.. math::

   \begin{gathered}
   F_v(z) = \Pr\big(Z(v) \le z \mid O\big) \quad\text{at every } v,
   \\
   \Pr\big(Z(v_1) \le z_1, \dots, Z(v_N) \le z_N \mid O\big)
   \quad\text{for any } v_1, \dots, v_N .
   \end{gathered}

Many practical questions are functionals of these laws and not of a map.

- The probability of exceeding a limit :math:`t` at a location, :math:`1 - F_v(t)`, a question about
  the tail of one marginal law.
- The area above the limit, :math:`|\{v \in D : Z(v) > t\}|`, whose law depends on how the values at
  different locations move together, so it needs the joint law.
- The tonnage above a cut-off grade, again a functional of the joint law, which a map of averages
  gets wrong by smoothing the extremes away.

What kriging sees
=================

Ordinary kriging predicts with a linear combination of the data,

.. math::

   \hat z_{\text{OK}}(v) = \sum_{i=1}^n \lambda_i(v)\, z_i, \qquad \sum_i \lambda_i(v) = 1,

whose weights solve a system built from the covariance :math:`C(h) = \operatorname{Cov}(Z(v),
Z(v+h))`, or equivalently the variogram :math:`\gamma(h) = C(0) - C(h)`. The covariance only
involves pairs of locations. Two fields can share it while their third and higher joint moments
differ, in connectivity, in skewness or in how often extremes occur together, and kriging cannot
tell them apart. Its optimality, best linear unbiased prediction, is a statement about first and
second moments.

The usual extension to laws, indicator kriging, estimates :math:`F_v(t_k)` at thresholds
:math:`t_1 < \dots < t_K` with one kriging system per threshold,

.. math::

   \hat F_v(t_k) = \sum_i \lambda_i^{(k)}(v)\, \mathbf 1\{z_i \le t_k\}.

The systems know nothing of one another. Nothing forces :math:`\hat F_v(t_1) \le \hat F_v(t_2)`,
nor :math:`0 \le \hat F_v \le 1`, since the weights :math:`\lambda_i^{(k)}` can be negative. Nothing ties the laws at different locations into a
joint law either.

Two requirements
================

A coherent estimate of the field's law must meet two requirements.

1. At every location, :math:`\hat F_v` is a distribution function, non-decreasing from 0 to 1.
2. The laws at different locations come from one shared mechanism, so that they are the marginals
   of a single joint law.

The estimator of the next page meets both by construction.

In Spatialize
=============

The estimators return, at each location, the members of an ensemble, from which every functional
of the law is computed (:doc:`sde`). The conformance tests check the coherence claims on simulated
fields (:doc:`../scenarios/index`).

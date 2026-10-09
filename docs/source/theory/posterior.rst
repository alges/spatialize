.. _theory-posterior:

##############################
Posterior analysis of the data
##############################

Classical exploratory analysis studies the data before any model, with summaries and the expert
knowledge of the analyst, who decides which values look wrong. Posterior analysis reads each datum
against what the rest of the data say about it. The evidence comes from the data themselves,
through the partitions and the decoder, with no variogram or model chosen beforehand. It serves the
quality control of a set of data, pointing to the values their surroundings do not support.

The law of a datum
==================

Cross-validation predicts each datum :math:`z_i` from the other data, one member per partition, so
that every datum receives a predictive law :math:`\hat F_{-i}` built without it (:doc:`error`). The
datum itself takes no part in its law, so a value its neighbours do not support keeps all of its
surprise.

An isolated datum often sits alone in its cell. Under ``empty_cells="nan"`` those partitions give it
no member, its law resting on the others. The share of partitions that give a member, the
*support* of the datum, is the complement of the residual weight of the block-mark model
(:doc:`blockmark`). A low support marks a law resting on few partitions. The policies ``"mark"``
and ``"coarsen"`` give every datum a member in every partition.

Reading one datum
=================

The position of the datum in its law,

.. math::

   u_i = \hat F_{-i}(z_i),

tells how far into its tails the datum falls. Its two-sided tail probability,

.. math::

   p_i = \min\{1,\ 2\min(u_i,\ 1 - u_i)\},

is the p-value of the hypothesis that the datum was drawn from its law. The level of a datum is the
widest central interval :math:`[\hat q_{(1-\alpha)/2},\ \hat q_{(1+\alpha)/2}]` of its law that leaves
it out, for a few probabilities :math:`\alpha`. A datum drawn from its law falls outside the interval
of probability :math:`\alpha` with probability :math:`1 - \alpha`.

The log score of the datum, :math:`-\log \hat f_{-i}(z_i)`, measures its surprise in the units of
the scoring rule the hyperparameter searches use (:doc:`error`). Its expectation under the law is
the entropy of the law, so the excess of the score over the entropy is near 0 for a datum typical of
its law and large for one its law does not expect.

Many data at once
=================

With :math:`n` data, a share :math:`1 - \alpha` of them falls outside the interval of probability
:math:`\alpha` by chance alone, even when every law is right. Listing the data of the outer level
would therefore list about :math:`(1 - \alpha)\,n` clean data. Spatialize flags the data with the
Benjamini–Hochberg procedure, which sorts the p-values :math:`p_{(1)} \le \dots \le p_{(n)}` and
flags the :math:`k` smallest, :math:`k` being the largest index with

.. math::

   p_{(k)} \le \frac{k}{n}\, q.

When the p-values are right, the expected share of clean data among the flagged ones, the *false
discovery rate*, is at most :math:`q`.

Calibration comes first
=======================

The guarantee rests on laws whose probabilities are right, which the laws read by
cross-validation are not. Each member averages the data of a cell, so the members spread less than the data do
around them. Read as they are, the laws put many more data in their tails than their probabilities
say. The tails of the laws are also too light, since a few hundred members say little about values
beyond them.

Spatialize corrects the laws in three steps before reading them.

1. **Widening.** The members of each datum are widened to the spread of its nearest other data, the
   ensemble widening of simulation (:doc:`ess`). The spread is measured robustly, by the median
   absolute deviation, so an erroneous neighbour does not widen a law enough to hide another error.
2. **One factor.** Every law's spread around its median is multiplied by one factor, fitted so that
   90 % of the data fall inside the central 90 % interval of their law. A factor above 1 tells that
   the widened laws were still too narrow.
3. **Tails.** The law is carried past its sample by a model of its tails, so that a value far
   beyond every member gets a p-value as small as its distance warrants. Counting the members could
   not give less than one over their number, which no flag can pass among hundreds of data. The
   model is a kernel density with Student-t kernels of 3 degrees of freedom, by default, or
   generalized Pareto tails beyond the 10 % and 90 % quantiles, with one shape per side pooled over
   all the laws. Gaussian kernels give the lightest tails.

The laws can also be read on a transformed scale, which makes a skewed variable more symmetric. A
Yeo–Johnson transform fitted to the data keeps how far an extreme value lies. Normal scores make the
data normal, while bringing every extreme value to the largest score, so a gross error stands out
less.

Version 1.2 controlled the tails another way, adding the datum to its own sample. The law then
always reached the datum, which kept its tail probability away from 0, at the price of the surprise
the analysis looks for. The tail model replaces that device.

What the corrections achieve
----------------------------

The corrections were measured on synthetic fields, without a guarantee for other data. Each field
holds 300 data with values :math:`\sin(2\pi x) + 0.5\cos(2\pi y)` plus Gaussian noise of standard
deviation 0.1, or the exponential of that sum with noise 0.3 (a lognormal field). The data are
placed uniformly, or half of them in four clusters. Each field is read clean and with three planted
errors, a value multiplied by 10, a value shifted by 3 standard deviations and a value shifted down
by 2.5 (multiplied by 0.1 on the lognormal field). Eight fields of each kind were read with IDW on
Mondrian partitions (``alpha`` = 0.85, 300 partitions), flags at :math:`q = 0.05`.

.. list-table:: Clean fields with a false flag, and planted errors found
   :header-rows: 1
   :widths: 32 17 17 17 17

   * - setting
     - Gaussian, uniform
     - Gaussian, clustered
     - lognormal, uniform
     - lognormal, clustered
   * - Gaussian kernels, no correction
     - 8/8 · 23/24
     - 8/8 · 23/24
     - 8/8 · 23/24
     - 8/8 · 24/24
   * - Gaussian kernels
     - 8/8 · 23/24
     - 8/8 · 21/24
     - 7/8 · 15/24
     - 6/8 · 14/24
   * - Student-t kernels (default)
     - 5/8 · 22/24
     - 4/8 · 17/24
     - 1/8 · 12/24
     - 4/8 · 12/24
   * - generalized Pareto tails
     - 0/8 · 19/24
     - 3/8 · 13/24
     - 5/8 · 5/24
     - 7/8 · 6/24
   * - Student-t, Yeo–Johnson scale
     - 5/8 · 22/24
     - 5/8 · 20/24
     - 3/8 · 5/24
     - 4/8 · 6/24
   * - Student-t, normal scores
     - 4/8 · 20/24
     - 3/8 · 13/24
     - 3/8 · 4/24
     - 2/8 · 3/24

Each cell gives the clean fields with at least one flag, out of 8, and the planted errors flagged,
out of 24. Without correction the laws flag between 140 and 1 041 clean data over the eight fields
with planted errors, and the coverage of their central 99 % interval falls to between 0.63 and 0.91.
With the corrections that coverage lies between 0.97 and 0.99.

On a clean field every flag is a false discovery. The procedure promises that some flag appears on
at most 5 % of clean fields. No setting keeps that promise on every kind of field, the tails of the
laws remaining too light at the 99 % level. Heavier tails trade false flags for power. Clustered
data and skewed values are the hardest cases. The calibration of the laws is therefore part of the
result. Spatialize reports it, the coverage of the central intervals before and after the factor,
with a verdict, in whose light the flags are to be read.

An error or an unrepresented place
==================================

A value its neighbours do not support may be an error, a sample switched in the laboratory or a
decimal point misplaced, or it may record a part of the domain the other data do not represent, a
thin seam or a stream that only runs in flood. Nothing in the number tells the two apart, since the
difference lies in the provenance of the datum. Posterior analysis gives the evidence and an order,
from the most to the least surprising datum. The analyst, who knows the provenance, decides.

In Spatialize
=============

Posterior analysis is :func:`~spatialize.gs.spa.posterior_audit`, whose result
:class:`~spatialize.gs.spa.PosteriorAudit` holds the readings of every datum
(:doc:`../reference/spa`).

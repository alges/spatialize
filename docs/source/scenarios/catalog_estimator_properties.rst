.. _scenarios-catalog-t2:

###########################
T2 — Estimator properties
###########################

Scenarios of tier T2 test propositions the theory proves about the ensemble estimator, such as how
fast the ensemble converges, what the decoders return, how the estimated law is built and how cells
without data are treated. A proposition holds for every field, so most of these checks are exact
(almost sure) or identities with a known standard error.

P2 and P3 are implemented. P5, P8 and P9 need the empty-cell policy, a planned feature of
Spatialize, while P1, P4, P6 and P7 can be implemented now.

.. _scenario-P1:

P1 — Convergence of the ensemble
================================

:Status: ready (not yet implemented)
:Source in the theory: convergence of the ensemble estimate as the number of partitions grows
:Claim: the spread of the estimate across independent ensembles decreases as :math:`T^{-1/2}` with
  the number of members :math:`T`.

**Setup.** A Mondrian block-mark field with 35 cells, 250 uniformly placed data and the IDW decoder.
For each of several ensemble sizes :math:`T`, many independent ensembles (different seeds) are run on
the same data.

**Check.**

- *identity* on the slope of :math:`\log(\text{spread})` against :math:`\log T`, which must equal
  :math:`-1/2`, with the standard error of the fitted slope.

.. _scenario-P2:

P2 — Weighted-draw decoder
==========================

:Status: **implemented**
:Evaluator: ``draw_laws``
:Source in the theory: the distance-weighted draw and its support, mean and variance
:Claim: a weighted-draw member is always a value of the cell's data. The mean of many members equals
  the estimate of the decoder averaging with the same weights, while their variance equals the
  weighted dispersion of the cell's values.

**Setup.** 150 data placed uniformly on the unit square carry a smooth field plus noise,
:math:`\sin(2\pi x) + 0.5\cos(2\pi y) + \varepsilon` with :math:`\varepsilon \sim \mathcal N(0, 0.3^2)`,
so their values are distinct. 60 queries are placed uniformly. Every estimator uses the Mondrian
partition of rate 4, with :math:`T = 2\,000` members in ``ci`` and :math:`8\,000` in ``full``.

**Reading against a reference.** A draw decoder is compared with its *reference*, the decoder that
averages over the same weights, run with the same seed. Spatialize's runner gives estimators that
share a seed the same partitions, so at each query the member of the draw and that of the
reference come from the same cell. The draw minus the reference then has mean 0. Its square has
the mean of the weighted dispersion, which is the reference applied to the squared values minus
the square of the reference.

**Checks.**

- *almost-sure* (``support``). Every finite member of ``wdraw_idw``, ``draw``, ``wdraw_kriging``,
  ``wdraw_adaptiveidw`` and ``wdraw_sharpidw`` is one of the data values.
- *identity* (``mean``). At each query, the mean over the members of ``wdraw_idw`` minus ``idw``,
  divided by its standard error, is a :math:`z`; the 60 values are combined as
  :math:`\sum z^2 \sim \chi^2_{60}`. The same for ``draw`` against ``cellmean``,
  ``wdraw_adaptiveidw`` against ``adaptiveidw`` and ``wdraw_sharpidw`` against ``sharpidw``.
- *identity* (``variance``). The same for the squared difference minus the weighted dispersion, for
  ``wdraw_idw`` and ``draw``. The dispersion is read as the reference applied to the squared values
  minus the square of the reference, which needs weights that do not depend on the values. The
  adaptive and sharpened weights are fitted from the values, so their variance identity is left
  out. It was tried and failed on every seed, because running the reference on the squared values
  refits its weights.
- *negative control*. The mean of ``wdraw_idw`` against ``cellmean``, a reference with other
  weights, must reject.

**Results.** On seeds 1 to 3 every identity passes, with p-values between 0.04 and 0.98, while the
control rejects with :math:`|z|` up to 65.

.. _scenario-P3:

P3 — Uniform draw
=================

:Status: **implemented**
:Evaluator: ``draw_laws``
:Source in the theory: the uniform draw is the block-mark decoder
:Claim: each value of a cell is drawn with a frequency equal to its share among the cell's data.

**Setup.** 30 data and 6 queries on the unit square, with the field of P2, and the Mondrian
partition of rate 2, with :math:`T = 3\,000` members in ``ci`` and :math:`12\,000` in ``full``.

**Expected counts.** Under one partition, the uniform draw at a query returns datum :math:`j` with
probability one over the occupancy of the query's cell when :math:`j` lies in it, and 0 otherwise.
``cellmean`` run on the indicator of datum :math:`j` returns exactly that probability at the
query. Summing it over the same partitions gives the expected number of times datum
:math:`j` is drawn at the query.

**Check.**

- *gof-closed*. The Pearson statistic of the observed against the expected counts, over the data
  with a positive expectation at each query, referred to :math:`\chi^2` with :math:`k - q` degrees
  of freedom (:math:`k` counts, :math:`q` queries). Since the probabilities change from one
  partition to the next, the counts vary less than multinomial ones, which makes the test
  conservative.
- *negative control*. The counts of ``wdraw_idw``, which draws by distance, must not match the cell
  shares.

**Results.** On seeds 1 to 3 the check passes (p between 0.34 and 0.92), while the control rejects with
:math:`\chi^2 \approx 20\,000` on 174 degrees of freedom.

.. _scenario-P4:

P4 — The estimated law is a law
===============================

:Status: ready (not yet implemented)
:Source in the theory: validity of the distribution estimated from the ensemble members
:Claim: at every location and threshold the estimated cumulative distribution function is
  non-decreasing in the threshold and lies within :math:`[0, 1]`.

**Check.**

- *almost-sure*, over all locations and a grid of thresholds, for every reading of the law
  Spatialize offers (empirical, fitted models, widened laws).

.. _scenario-P5:

P5 — Residual weight of the decoders
====================================

:Status: blocked — needs the empty-cell policy (planned)
:Source in the theory: the limit law of the estimator as a mixture of the data values with the mark
  law
:Claim: at a location, the weights on the observed values sum to one together with the *residual
  weight*, the proportion of partitions in which the location's cell has no data. The residual weight
  grows with the distance to the data. With ``empty_cells="mark"`` on a block-mark truth, the law of
  the members equals the mixture of the data values with the estimated mark law, weighted that way.

**Checks.**

- *almost-sure*. The weights sum to one.
- *paired-relation*. The residual weight is larger far from the data.
- *gof* (:math:`\chi^2` or Kolmogorov–Smirnov) of the member law against the mixture.

.. _scenario-P6:

P6 — Locality
=============

:Status: ready (not yet implemented), **expected to reject on Spatialize**
:Source in the theory: the estimate at a location depends only on the data and the partition law
:Claim: the law of the estimate at a location does not change when the *other* query locations
  change.

**Setup.** The same data and the same location are estimated together with two different sets of
other queries (e.g. a small grid, then a large one extending beyond the data).

**Check.**

- *two-sample* (Kolmogorov–Smirnov) between the two laws at the location.

By default Spatialize draws its partitions on the box of data *and queries*, so its law depends on
the other queries, which makes the test expected to reject. That run serves as a negative control
for the test itself (:doc:`encoders`). With a session domain (:doc:`../reference/session`) the box is
fixed, so the same test is expected to pass.

.. _scenario-P7:

P7 — Covariance of an uncorrelated field
========================================

:Status: ready (not yet implemented), target to be derived before implementation
:Source in the theory: the covariance the ensemble induces between estimates of an uncorrelated field
:Claim: for an uncorrelated field and partitions much coarser than the data spacing, the covariance
  of the estimates at two locations follows a closed form in the partition rate and their distance.

**Check.**

- *gof-closed* per distance class. The closed form is derived independently, then computed by the
  evaluator, before the scenario is added.

.. _scenario-P8:

P8 — Data-free cells share one mark
===================================

:Status: blocked — needs the empty-cell policy (planned)
:Source in the theory: a cell without data contributes a single mark, shared by all its locations
:Claim: with ``empty_cells="mark"``, two queries far from the data have members whose covariance
  equals the variance of the estimated mark law times their co-occurrence probability.

**Checks.**

- *identity* on the covariance.
- *negative control*: drawing an independent value per location gives covariance 0, which the test
  must reject.

.. _scenario-P9:

P9 — Mark law under preferential sampling
=========================================

:Status: blocked — needs the empty-cell policy (planned)
:Source in the theory: the block-mark model has one mark per cell
:Claim: under preferential sampling, estimating the mark law with one draw per data-bearing cell
  (``mark_source="cells"``) gives an unbiased estimate. Drawing among all data
  (``mark_source="data"``) gives one biased towards the densely sampled zone.

**Setup.** A preferential design, with 150 uniform data plus 150 in the central quarter of the
domain.

**Checks.**

- *gof* of the cell-weighted estimate against the true mark law.
- *paired-relation*. The data-weighted estimate lies farther from it.

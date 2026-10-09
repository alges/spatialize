.. _scenarios-catalog-t2:

###########################
T2 — Estimator properties
###########################

Scenarios of tier T2 test propositions the theory proves about the ensemble estimator, such as how
fast the ensemble converges, what the decoders return, how the estimated law is built and how cells
without data are treated. A proposition holds for every field, so most of these checks are exact
(almost sure) or identities with a known standard error.

P2, P3, P5, P6, P8, P9, P10, P11 and P12 are implemented. P1, P4 and P7 can be implemented now. P5, P8
and P9 read the law of the marks of empty cells, with targets derived for the mark strategies
Spatialize implements (:doc:`../theory/blockmark`).

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

:Status: **implemented**
:Evaluator: ``mark_law``
:Source in the theory: the limit law of the estimator as a mixture of the data values with the mark
  law
:Claim: at a location, the weights on the observed values sum to one together with the *residual
  weight*, the proportion of partitions in which the location's cell has no data. The residual weight
  grows with the distance to the data. With ``empty_cells="mark"`` the law of the members is the
  mixture of the data values with the mark law, weighted that way.

**Setup.** 30 data uniform in :math:`[0, 0.6]^2` and 30 in :math:`[0, 0.2]^2`, a cluster, with the
field of P10. Six locations run along the diagonal from :math:`(0.3, 0.3)` to :math:`(1, 1)`. The
estimators use the Mondrian partition of rate 4 and the Voronoi partition with uniform nuclei of
intensity 12, with :math:`T = 2\,000` members in ``ci`` and :math:`8\,000` in ``full``.

**Target.** With the decoder ``draw`` and ``mark_value="datum"`` every member is one datum. Under a
partition :math:`t`, write :math:`C` for the location's cell, :math:`\mathcal C_t` for the cells
holding data and :math:`n_{C'}` for the number of data in :math:`C'`. The member is the datum
:math:`z_i` with probability

.. math::

   p_t(i) = \begin{cases}
   1/n_C & \text{if } C \text{ holds data and } i \in C, \\
   1/(|\mathcal C_t|\, n_{C(i)}) & \text{if } C \text{ is empty, under } \texttt{mark\_source="cells"}, \\
   1/n & \text{if } C \text{ is empty, under } \texttt{mark\_source="data"},
   \end{cases}

and 0 otherwise. The evaluator reads the cells with the runner's method ``cells``, so the expected
count of each datum at a location is :math:`\sum_t p_t(i)`.

**Checks.**

- *gof-closed* (``law-cells``, ``law-data``). At each location, the count of each datum against its
  expectation. The Pearson statistic is referred to :math:`\chi^2` with :math:`k - 6` degrees of
  freedom, k being the counts with positive expectation.
- *almost-sure* (``weights``). Run on the indicator of each datum, the cell mean under ``"nan"`` gives
  the weights :math:`w_i(x)` as the mean over the partitions. They sum to one with the residual
  weight, the share of NaN members, up to :math:`10^{-4}`.
- *paired-relation* (``residual``). The location :math:`(1, 1)` has an empty cell more often than
  :math:`(0.3, 0.3)`, paired by partition.
- *negative control* (``control-law``). The marks drawn among the data, tested against the
  cell-weighted law, must be rejected. The cluster makes the two laws differ.

**Results.** Not run yet.

.. _scenario-P6:

P6 — Locality
=============

:Status: **implemented**
:Evaluator: ``locality``
:Source in the theory: the estimate at a location depends only on the data and the partition law
:Claim: the law of the estimate at a location does not change when the *other* query locations
  change.

**Setup.** 150 data on the unit square carry the field of P2. The location :math:`v = (0.5, 0.5)` is
estimated twice with one seed, together with 20 other queries, then together with 400. The estimators
use IDW on the default Mondrian under its two profile names (the theory's process since version
1.3), on the partition of version 1.2 (``mondrian-legacy``) and on the Voronoi partition with uniform
nuclei, with :math:`T = 3\,000` members in ``ci`` and :math:`12\,000` in ``full``.

**Checks.**

- *two-sample* (``inside``). With the 400 other queries inside the domain, the Kolmogorov–Smirnov test
  between the two laws at :math:`v` must not reject.
- *two-sample* (``beyond``). With the 400 other queries spread over :math:`[-0.5, 1.5]^2`, past the
  domain, the test must not reject on the theory's Mondrian process. It is expected to reject on the
  partition of version 1.2 and on the Voronoi partition, as negative controls.

Spatialize draws its partitions on the box of the data and the queries. The runner pins that box to
the declared domain by adding its corners as queries, so other queries inside the domain leave it
unchanged, while queries beyond it enlarge it. The rate of the default Mondrian comes from the box of
the data, so it stays the same. The theory's Mondrian process is consistent under restriction, so on
a larger box its cells around :math:`v` keep their law (:doc:`../theory/encoders`). The partition of
version 1.2 measures its rate on the enlarged box, which coarsens its cells. The Voronoi nuclei spread
over the larger box. A session domain removes the dependence for every partition
(:doc:`../reference/session`).

**Results.** Before version 3 of the scenario, the rate came from the box of data and queries. On
seeds 1 to 3 the law at :math:`v` was then the same with any other queries inside the domain (p = 1
for every estimator), while queries beyond it changed it (p at most :math:`10^{-6}`). Version 3 has
not been run yet.

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

:Status: **implemented**
:Evaluator: ``mark_law``
:Source in the theory: a cell without data contributes a single mark, shared by all its locations,
  the marks of distinct cells being independent
:Claim: with ``empty_cells="mark"``, two queries far from the data have members whose covariance
  equals the variance of the mark law times the probability that they share an empty cell.

**Setup.** 40 data uniform in :math:`[0, 0.6]^2` with the field of P10, and the locations
:math:`a = (0.9, 0.9)` and :math:`b = (0.95, 0.95)`. The estimators use the decoder ``draw`` with
marks drawn as data, on the Mondrian partition of rate 4 (marks from the cells and from the data)
and on the Voronoi partition of intensity 12 (marks from the cells), with :math:`T = 4\,000` members
in ``ci`` and :math:`16\,000` in ``full``.

**Target.** Given the cells of partition :math:`t`, with :math:`p_t` the law of P5 at each location,

.. math::

   E[Y_a Y_b \mid t] = \begin{cases}
   \sum_i p_t(i)\, z_i^2 & \text{if } a, b \text{ share an empty cell}, \\
   \big(\sum_i p_{t,a}(i)\, z_i\big)\big(\sum_i p_{t,b}(i)\, z_i\big) & \text{otherwise},
   \end{cases}

since one shared mark gives the second moment of the mark law, while distinct cells give independent
members. Averaging over the partitions gives the covariance of the claim.

**Checks.**

- *identity* (``product``). The mean over the partitions of :math:`Y_a Y_b - E[Y_a Y_b \mid t]`,
  divided by its standard error, is referred to :math:`N(0, 1)`.
- *negative control* (``control-independent``). Taking :math:`Y_b` from the next partition, as if
  each location drew its own value, must be rejected.

The marks of distinct empty cells are independent draws, as in the block-mark model. Distinct
sources within a partition would make them negatively correlated, which this check would detect.

**Results.** Not run yet.

.. _scenario-P9:

P9 — Mark law under preferential sampling
=========================================

:Status: **implemented**
:Evaluator: ``mark_law``
:Source in the theory: the block-mark model has one mark per cell
:Claim: under preferential sampling, the marks drawn with one vote per cell with data
  (``mark_source="cells"``) lie closer to the spatial law of the field than the marks drawn among
  all the data (``mark_source="data"``), which lean towards the densely sampled zone.

**Setup.** On each of 20 fields in ``ci`` and 40 in ``full``, 150 data uniform in the unit square
and 150 in the central quarter :math:`[0.25, 0.75]^2`. The field is 2 in the central quarter and 0
elsewhere, plus noise of standard deviation 0.3, so its spatial mean is 0.5. The estimators use the
decoder ``draw`` with marks drawn as data on the Mondrian partition of rate 6, with :math:`T = 400`
members in ``ci`` and :math:`1\,000` in ``full``. The marks are read at :math:`(1.5, 1.5)`, beyond the
domain, in the partitions where its cell is empty.

**Check.**

- *paired-relation* (``closer``). Over the fields, the error of the mean of the data-drawn marks
  about 0.5 exceeds that of the cell-drawn marks (paired t-test).

One vote per cell is unbiased only approximately. Cells straddling the central quarter draw its
data more often, and cells without data take no vote. The check is therefore a paired relation, not
a goodness of fit to the true mark law.

**Results.** Not run yet.

.. _scenario-P10:

P10 — Empty-cell policies
=========================

:Status: **implemented**
:Evaluator: ``empty_cells``
:Source in the theory: the residual weight of the limit law goes to a mark, one per cell without
  data (the block-mark model)
:Claim: each empty-cell policy does what it declares, whatever the field. Under ``"nan"`` a member
  is NaN exactly when no datum lies in the query's cell. ``"mark"`` and ``"coarsen"`` leave the
  cells with data unchanged, with no member undefined. Under ``"mark"`` the locations of an
  empty cell share one value.

**Setup.** 40 data lie uniformly in :math:`[0, 0.6]^2`, 36 % of the unit square, so that many cells
hold no datum. They carry the field of P2. The queries are the centres of a :math:`12 \times 12`
grid over the whole square. Two partitions are used, the Mondrian partition of rate 4 (25 cells on
average) and the Voronoi partition with nuclei uniform in the square, of intensity 12. On each, the
IDW decoder runs under ``"nan"``, under ``"coarsen"`` and under ``"mark"`` with three strategies:

- ``local``, the default, the prediction of a nearby cell's decoder at the empty cell (8 cells);
- ``cells`` with ``datum``, one datum of a cell drawn among the cells with data (the block-mark
  model);
- ``data``, one datum drawn among all the data.

The cell mean runs under ``"nan"``, ``"mark"`` (``local``) and ``"coarsen"``. Every estimator uses
:math:`T = 200` members in ``ci`` and :math:`1\,000` in ``full``, with one seed, so all of them see
the same partitions.

**Reading the cells.** A query lies in an empty cell when the member of its estimator under
``"nan"``, the *reference*, is NaN. The checks that compare the locations of one cell read the cells
of the partitions through the runner (:doc:`extending`).

**Checks.** All are almost sure except ``data-law``.

- ``nan``. Under ``"nan"`` a member is NaN exactly when no datum shares the query's cell, for IDW and
  the cell mean on both partitions.
- ``unchanged``. Where the reference is finite, every policy gives the same member, bit for bit.
- ``filled``. No member of ``"mark"`` or ``"coarsen"`` is NaN.
- ``one-mark``. Under ``"mark"``, in each partition, the queries of one empty cell share one value,
  for every strategy.
- ``observed``. A mark drawn as a datum (``cells`` with ``datum``, and ``data``) is a data value.
- ``cell-mean``. With the cell mean, a mark is the mean of the data of a cell of the same partition,
  and so is the value ``"coarsen"`` gives on a Voronoi partition, where the coarser cell is the one
  with data whose nucleus lies nearest. On a Mondrian partition the coarser cell is an ancestor in
  the tree of cuts, a union of cells, so this check leaves it out.
- ``data-law``, *gof-closed*. At the query nearest to the corner :math:`(1, 1)`, opposite the data,
  the marks drawn among the data (``data``) are uniform over the 40 data. The members of the
  partitions where the query's cell is empty, one per partition and so independent, are counted per
  datum, and the Pearson statistic is referred to :math:`\chi^2_{39}`. Each empty cell draws its
  datum uniformly, independently of the other empty cells.
- *negative controls*. ``"coarsen"`` with IDW predicts at each location, so the queries of one empty
  cell must not share one value. ``"mark"`` with the ``local`` strategy predicts with IDW at the
  empty cell's point, so its marks must not be data values.

The rule that the empty cells of one partition draw distinct source cells, while candidates remain,
cannot be read from the queries alone, since the number of empty cells of a partition is unknown.
The library's own tests check it for the strategy ``data``
(``tests/unit/test_empty_cells.py``).

**Results.** On seeds 1 to 3, in both modes, every almost-sure check passes with no violation,
among about 29 000 members per estimator in ``ci``, of which 40 % to 46 % lie in empty cells. The
law of the marks drawn among the data passes with p between 0.15 and 0.93. The controls
reject, with 80 % to 91 % of the empty cells holding several values under ``"coarsen"`` and 69 % to
87 % of the IDW marks differing from every datum.

The goodness-of-fit check has no negative control. The natural one, the block-mark strategy
(``cells`` with ``datum``), weighs each datum by one over the occupancy of its cell, a law close to
the uniform one on this design. With 1 000 members its counts are rejected on every seed tried
(p at most :math:`10^{-8}`). With 200 they are not rejected reliably, so a control would fail in
``ci``.

.. _scenario-P11:

P11 — Selection bias of the cross-validation
============================================

:Status: **implemented**
:Evaluator: ``cv_selection``
:Source in the theory: the law conditioned on the cell having data is not the theory's law, whose
  residual weight goes to a mark; scoring the predictive law
:Claim: a cross-validation score that drops undefined members scores only the data whose cells keep
  other data, so its value is optimistic and favours fine partitions. ``"mark"`` and ``"coarsen"``
  define every leave-one-out member, so a score covers every datum.

**Setup.** 180 data on the unit square, 40 in each of three Gaussian clusters of standard deviation
0.04 and 60 scattered uniformly, carry the field of P2 with noise of standard deviation 0.02. IDW
predicts each datum from the others (leave-one-out) on the theory's Mondrian process at rate 32,
fine enough to leave the scattered data alone in their cells, under ``"nan"``, ``"coarsen"`` and
``"mark"`` (strategies ``local``, and ``cells`` with ``datum``). The almost-sure checks also run on
Spatialize's default Mondrian at rate 12. Every estimator uses :math:`T = 200` members in ``ci``
and :math:`1\,000` in ``full``, with one seed. The leave-one-out ensembles come from the runner's
optional method ``loo`` (:doc:`extending`).

**Checks.**

- ``loo-filled``, *almost-sure*. Under ``"mark"`` and ``"coarsen"`` no leave-one-out member is NaN.
- ``loo-unchanged``, *almost-sure*. Where the member under ``"nan"`` is defined, the other policies
  give the same member, bit for bit.
- ``control-selection``, *negative control* (two-sample). A score that needs 15 % of finite members
  at a datum, the NLL's 30 out of 200, leaves out the data below that share under ``"nan"``. Each
  datum's error is read under ``"coarsen"``, defined at every datum, as the absolute difference
  between the mean of its members and its value. The Kolmogorov–Smirnov test between the errors of
  the data kept and of those left out must reject, since the data left out are the isolated ones,
  the hardest to predict.

**Results.** On seeds 1 to 3, in both modes, the almost-sure checks pass with no violation. The
control rejects with p at most :math:`2.4 \cdot 10^{-11}`. About 37 of the 180 data are left out.
Their mean error is 0.16 against 0.03 for the data kept, so the score of the kept data, about 0.040,
understates the error over all the data, about 0.060, by a third.

The searches of Spatialize record, for each configuration, the share of data left out of the score
and warn above the session setting ``max_left_out`` (:doc:`../reference/session`).

The design was chosen on probes before the first run. With noise 0.05 and rate 24 the control's
p-values, between :math:`10^{-4}` and :math:`10^{-7}`, came too close to its Holm level.

.. _scenario-P12:

P12 — Posterior analysis of the data
====================================

:Status: **implemented**
:Evaluator: ``posterior_audit``
:Source in the theory: a value its neighbours do not support may be an error or an unrepresented
  part of the domain, which only its provenance decides (:doc:`../theory/posterior`)
:Claim: read against the laws the other data give, planted errors are flagged, the neighbours of an
  isolated error and of a raised patch are surprised in opposite ways, and the partitions decluster
  a preferential design. The flags of clean fields stay within the false discovery rate, a claim
  recorded as a known failure.

**Setup.** Each replicate field holds 300 data uniform on the unit square, with values
:math:`\sin(2\pi x) + 0.5\cos(2\pi y)` plus Gaussian noise of standard deviation 0.1, read in four
versions:

- clean;
- with three planted errors, a value multiplied by 10, one raised by 3 standard deviations and one
  lowered by 2.5;
- with a square of side 0.2 raised by 1.5 (about 12 data) and one isolated error, raised by 2.5;
- preferential, half of the data drawn where the field exceeds 0.8.

The leave-one-out ensembles come from the runner's optional method ``loo``, with IDW on the Mondrian
partition of rate :math:`10/3` (Spatialize's ``alpha`` = 0.85) and :math:`T = 300` members, and the
cells of the partitions from ``cells``. The readings are those of
:class:`~spatialize.gs.spa.PosteriorAudit` with its defaults, flags at :math:`q = 0.05`. The scenario
uses 20 fields in ``ci`` and 40 in ``full``.

**Checks.**

- ``errors-found``, *one-sided*. The share of the planted errors flagged exceeds one half.
- ``clean-flags``, *one-sided*, known failure. The share of clean fields with a flag stays below
  0.05. The laws' tails remain too light at the 99 % level, so the false discovery rate is not
  controlled.
- ``shift-patch``, *paired-relation*. The mean shift of the square's data exceeds that of the other
  data.
- ``shift-error``, *one-sided*. The shift at the isolated error is negative.
- ``declustering``, *paired-relation*. The declustered mean lies closer to the field's mean over the
  domain than the plain mean, under the preferential design.
- ``weights``, *almost-sure*. The declustering weights are positive and sum to 1.

**Results.** On seeds 1 to 3 in ``ci`` and seed 1 in ``full`` every check passes, with 93 % to 95 %
of the planted errors flagged, a shift of the square above the others by 0.22 (p at most
:math:`2.1 \cdot 10^{-7}`), a shift of -0.36 at the isolated error and a declustered mean closer
to the field's mean by 0.37. Between 30 % and 45 % of the clean fields carry a flag, the known
failure.

**History.** The first version checked the coherence of the square, which passed at its limit, the
coherence being a weak signal (0.06 to 0.10). Inside the square a datum's law is built from
neighbours of the square, so the datum itself is hardly surprised while its neighbourhood is. The
shift reading replaced it before the first commit, with the sizes computed by the power rule
(``provenance`` of the descriptor).

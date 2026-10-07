.. _scenarios-catalog-t3:

####################################
T3 — Geostatistical scenarios
####################################

Scenarios of tier T3 set up complete geostatistical situations with a known truth, such as
anisotropy, an atom at zero, heavy tails, non-stationarity, preferential sampling or degenerate
designs, in which the ensemble estimators must behave as a practitioner needs them to. They follow
the worked examples of the theory together with situations that break estimators in practice. Their
criteria compare estimators with each other, with reference estimators and with the truth, including
the visual criteria (:doc:`visual`, :doc:`catalog_visual_criteria`).

The truths come from three generators. A *Voronoi block-mark field* carries one mark per cell of a
Voronoi tessellation with uniform nuclei, the marks independent of the cells. A *Mondrian block-mark
field* does the same on a Mondrian partition. A *stationary Gaussian field* is drawn jointly at data
and queries, so the grid holds the field itself.

.. _scenario-S01:

S01 — Simulation keeps the geometry
===================================

:Status: ready (not yet implemented)
:Source in the theory: ensemble spatial simulation
:Claim: fields simulated from the ensemble keep the spatial structure of the truth, which independent draws from the same per-location laws lose.

**Setup.** The truth is a Voronoi block-mark field, with one mark per cell of a Voronoi tessellation with uniform nuclei, here with 25 cells and centred Gaussian marks. 400 uniformly placed data are drawn on each replicate field, estimated by Mondrian IDW (exponent 2) with :math:`T = 200`, then simulated by ensemble spatial simulation.

**Checks.**

- *paired-relation*. The lag-1 correlation of a simulated field exceeds that of a field of independent draws from the same per-location laws.

.. _scenario-S02:

S02 — Where more partitions stop helping
========================================

:Status: ready (not yet implemented)
:Source in the theory: convergence of the ensemble and its error floor
:Claim: the error against the truth stops decreasing beyond some ensemble size, while the estimate itself keeps converging (P1).

**Setup.** The truth is a Mondrian block-mark field with 35 cells, sampled at 250 uniformly placed data, estimated by Mondrian IDW with :math:`T` up to 3000.

**Checks.**

- *identity*. P1's :math:`T^{-1/2}` law holds on this field.
- *identity*. Beyond the estimated threshold :math:`T_0`, the slope of the error against the truth versus :math:`T` equals 0.

**Depends on.** A Mondrian block-mark truth generator.

.. _scenario-S03:

S03 — Anisotropic field
=======================

:Status: **implemented** (visual criteria V1, V4 and V5, with further checks planned below)
:Evaluator: ``map_visual``

**Source.** The theory's worked example of an anisotropic stationary Gaussian field.

**Truth.** A stationary Gaussian field on the unit square with exponential covariance
:math:`C(h) = \exp(-\sqrt{h^\top A h})`, :math:`A = R_\vartheta^\top \operatorname{diag}(a_1^{-2},
a_2^{-2}) R_\vartheta`, ranges :math:`a_1 = 0.45`, :math:`a_2 = 0.09` and orientation
:math:`\vartheta = 30°`, a 5:1 anisotropy of the kind a variogram is designed to capture.

**Data.** 400 uniformly placed samples and a 40×40 grid of queries (cell centres, row-major), drawn
jointly by Cholesky factorisation so that samples and truth belong to one realisation. Eighty
replicate fields are pinned (generator seed 20261005), stored as ``.npy`` with SHA-256 checksums.

**Estimators.** Seven ensembles take part, none of them told about the anisotropy, since the cuts are
axis-aligned or Voronoi cells while the decoders are isotropic. They use :math:`T = 100` members in
``ci`` and 300 in ``full``, the point map being the median of the members at each location.

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - id
     - encoder
     - decoder and parameters
   * - ``idw``
     - Mondrian, :math:`\lambda = 5`
     - IDW, exponent 2
   * - ``aidw``
     - Mondrian, :math:`\lambda = 5`
     - adaptive IDW, metric MAE (run in parallel; results identical to serial)
   * - ``krig``
     - Mondrian, :math:`\lambda = 5`
     - kriging, isotropic exponential, nugget 0, sill 1, practical range 0.604, three times the
       geometric mean of the true ranges, :math:`\sqrt{0.45 \cdot 0.09} = 0.201`, which keeps the
       area of the anisotropy ellipse (Spatialize's exponential model is :math:`e^{-3d/r}`)
   * - ``vor-uniform-idw``
     - Voronoi, uniform nuclei, :math:`\lambda_V = 36`
     - IDW, exponent 2
   * - ``vor-data-idw``
     - Voronoi, nuclei at the data, :math:`\lambda_V = 36`
     - IDW, exponent 2
   * - ``vor-uniform-krig``
     - Voronoi, uniform nuclei, :math:`\lambda_V = 36`
     - kriging, as ``krig`` (through the generic engine entry point)
   * - ``vor-uniform-aidw``
     - Voronoi, uniform nuclei, :math:`\lambda_V = 36`
     - adaptive IDW, as ``aidw`` (through the generic engine entry point)

The Voronoi intensity gives the same expected number of cells as the Mondrian rate, since
:math:`\lambda_V |H| = 36 = (1 + \lambda)^2`, the expected number of cells of the theory's Mondrian
process of rate 5 on the unit square.

**Claim (V1).** The elongation at 30° must nevertheless be clearly visible in each estimator's point
map. Two pre-registered map functionals (:doc:`visual`), the same for every estimator,
are computed on each of the :math:`K = 80` fields (the reason for 80 is given under V4).

- **v1a — direction** (equivalence, TOST). The orientation error :math:`\theta(\text{map}) - 30°`
  lies within :math:`\pm 10°`.
- **v1b — strength** (one-sided :math:`t`). The coherence of the map is more than half that of the
  truth, :math:`c(\text{map}) / c(\text{truth}) > 0.5`.

The results below come from ``ci`` mode with seed 12345, as means :math:`\pm` standard errors over
the 80 fields.

.. list-table::
   :header-rows: 1
   :widths: 22 26 13 26 13

   * - estimator
     - orientation error (v1a)
     - p
     - coherence ratio (v1b)
     - p
   * - ``idw``
     - :math:`-0.3° \pm 0.4°`
     - :math:`1.1 \cdot 10^{-42}`
     - :math:`0.61 \pm 0.009`
     - :math:`1.1 \cdot 10^{-20}`
   * - ``aidw``
     - :math:`-1.7° \pm 0.2°`
     - :math:`6.4 \cdot 10^{-50}`
     - :math:`0.97 \pm 0.006`
     - :math:`8.7 \cdot 10^{-78}`
   * - ``krig``
     - :math:`-0.5° \pm 0.3°`
     - :math:`1.5 \cdot 10^{-45}`
     - :math:`0.73 \pm 0.010`
     - :math:`1.1 \cdot 10^{-37}`
   * - ``vor-uniform-idw``
     - :math:`-0.5° \pm 0.3°`
     - :math:`2.2 \cdot 10^{-42}`
     - :math:`0.64 \pm 0.009`
     - :math:`5.9 \cdot 10^{-25}`
   * - ``vor-data-idw``
     - :math:`-0.4° \pm 0.3°`
     - :math:`2.3 \cdot 10^{-42}`
     - :math:`0.64 \pm 0.009`
     - :math:`1.4 \cdot 10^{-24}`
   * - ``vor-uniform-krig``
     - :math:`-0.4° \pm 0.3°`
     - :math:`9.3 \cdot 10^{-46}`
     - :math:`0.73 \pm 0.010`
     - :math:`4.8 \cdot 10^{-37}`
   * - ``vor-uniform-aidw``
     - :math:`-1.0° \pm 0.3°`
     - :math:`2.3 \cdot 10^{-50}`
     - :math:`0.91 \pm 0.007`
     - :math:`2.4 \cdot 10^{-65}`

All fourteen checks pass, stably across seeds. With kriging, the partition barely matters for the
median map, since with about eleven data per cell and a fixed variogram kriging inside a cell comes
close to global kriging. The Mondrian and Voronoi median maps correlate at 0.999 on a field, although
their members differ.

.. figure:: /_static/scenarios/S03_compare_13.png
   :width: 100%
   :alt: Truth and the seven median maps of S03 on field 13

   S03, field 13 (``ci``, seed 12345), with the truth and its samples next to the median map of each
   estimator on one colour scale, with the declared and the measured orientations drawn through the
   centre.

**Contrast (V5).** A map can show the right direction while being washed out. V5 compares the spread
of each estimator's median map with that of the best linear predictor, simple kriging with the true
covariance computed from the same data (:doc:`visual`). Check **v5** passes when
:math:`\mathrm{std}(\text{map}) / \mathrm{std}(\text{reference}) > 0.9`, by a one-sided :math:`t`
test over the 80 fields, with the threshold fixed before the criterion was first run.

.. list-table::
   :header-rows: 1
   :widths: 30 30 20 20

   * - estimator
     - contrast ratio (v5)
     - p
     - result
   * - ``aidw``
     - :math:`0.96 \pm 0.002`
     - :math:`2.4 \cdot 10^{-48}`
     - pass
   * - ``vor-uniform-aidw``
     - :math:`0.98 \pm 0.002`
     - :math:`3.1 \cdot 10^{-59}`
     - pass
   * - ``krig``
     - :math:`0.96 \pm 0.002`
     - :math:`1.0 \cdot 10^{-43}`
     - pass
   * - ``vor-uniform-krig``
     - :math:`0.96 \pm 0.002`
     - :math:`1.4 \cdot 10^{-44}`
     - pass
   * - ``idw``
     - :math:`0.85 \pm 0.004`
     - 1
     - known failure
   * - ``vor-uniform-idw``
     - :math:`0.895 \pm 0.004`
     - 0.90
     - known failure
   * - ``vor-data-idw``
     - :math:`0.895 \pm 0.004`
     - 0.92
     - known failure

IDW with exponent 2 is 10–15 % more washed out than the best linear predictor on this field, on
Mondrian and on Voronoi partitions alike. The shortfall is recorded as a known failure
(:doc:`statistics`), leaving the threshold unchanged.

**Mondrian blocks averaged away (V4).** Each ensemble member of a Mondrian estimator consists of
axis-aligned blocks, which the median of many members should not keep. V4 measures *axis-locking*,
the spectral energy on the coordinate axes against the diagonals of the median map minus the same
quantity for the same estimator run on the data rotated 45° about the domain centre and evaluated at
the same physical points (:func:`~spatialize.scenarios.stats.maps.axis_lock`). Both maps share data,
decoder and smoothing, so only artefacts tied to the axes remain, which leaves 0 for a rotation-invariant
partition (Voronoi). Three checks are pre-registered with :math:`\delta = 0.5` (log ratio of energies).

- **v4a** (Mondrian estimators). The axis-locking of the median map stays below :math:`\delta`.
- **v4b** (power check). Single members are axis-locked, the mean over 5 members per field exceeding 0.
- **v4c** (calibration, Voronoi estimators). The axis-locking of the median map lies within :math:`\pm\delta`
  of 0 (TOST) — the functional itself is checked on every run.

.. list-table::
   :header-rows: 1
   :widths: 24 22 13 26 15

   * - estimator
     - median map (v4a / v4c)
     - p
     - single members (v4b)
     - p
   * - ``idw``
     - :math:`0.34 \pm 0.015`
     - :math:`1.0 \cdot 10^{-17}`
     - :math:`2.55 \pm 0.061`
     - :math:`5.0 \cdot 10^{-56}`
   * - ``krig``
     - :math:`0.27 \pm 0.014`
     - :math:`4.5 \cdot 10^{-27}`
     - :math:`2.83 \pm 0.054`
     - :math:`1.9 \cdot 10^{-63}`
   * - ``aidw``
     - :math:`0.39 \pm 0.024`
     - :math:`9.6 \cdot 10^{-6}`
     - :math:`1.98 \pm 0.059`
     - :math:`4.3 \cdot 10^{-49}`
   * - ``vor-uniform-idw`` (v4c)
     - :math:`0.01 \pm 0.011`
     - :math:`1.2 \cdot 10^{-56}`
     - —
     - —
   * - ``vor-uniform-krig`` (v4c)
     - :math:`-0.02 \pm 0.009`
     - :math:`8.8 \cdot 10^{-63}`
     - —
     - —
   * - ``vor-uniform-aidw`` (v4c)
     - :math:`-0.05 \pm 0.020`
     - :math:`1.3 \cdot 10^{-35}`
     - —
     - —

Single members are strongly axis-locked (+2.0 to +2.9) while the Voronoi controls sit at 0, so the
test sees what it is meant to see. At :math:`T = 100` the Mondrian medians keep a residual of about
0.3, roughly a third more energy on the axes than in the rotated run. The residual stays well below a
small ensemble's and within the pre-registered :math:`\delta`, without vanishing.

**Why 80 fields.** With 40 fields, check v4a on adaptive IDW failed in 4 of 14 runs with different
seeds, although its mean axis-locking never came near the bound. Over those runs the mean lay
between 0.34 and 0.41, averaging 0.37, against :math:`\delta = 0.5`. The failures came from the test
lacking power, not from the estimator. The argument runs as follows.

- The axis-locking of a single field has a standard deviation of about 0.19, so the mean over
  :math:`K` fields has a standard error of :math:`0.19/\sqrt{K}`.
- Under Holm, v4a-aidw is tested at a level of about :math:`2.5 \cdot 10^{-4}`, which takes a
  :math:`t` statistic near 3.7. The check therefore passes only when the mean lies below
  :math:`0.5 - 3.7 \times 0.19/\sqrt{K}`.
- The mean itself varies from one run to another, with a standard deviation of 0.025 at
  :math:`K = 40`, shrinking as :math:`1/\sqrt{K}`.

.. list-table::
   :header-rows: 1
   :widths: 10 22 34 34

   * - :math:`K`
     - standard error
     - the check passes when the mean is below
     - expected share of failing runs
   * - 40
     - 0.030
     - 0.386
     - about 1 in 4 (observed 4 in 14)
   * - 60
     - 0.025
     - 0.409
     - about 1 in 50
   * - 80
     - 0.021
     - 0.423
     - about 1 in 1 000

At 40 fields the bound a run must clear sits within the spread of the mean itself, so about one run
in four fails while the decoder behaves as intended. At 80 fields the bound moves above that spread,
so a failure becomes a rare event (about 1 in 1 000) that deserves a look. The thresholds and
:math:`\delta` stay as pre-registered. Only the number of fields grows, the first 40 being the same
as before. The run time of S03 grows in proportion.

.. note::

   **History.** The first version of S03 had only ``idw`` and :math:`K = 6` fields, with v1b passing
   at :math:`p \approx 4 \cdot 10^{-4}`, just below a Holm level of :math:`10^{-3}`. Version 2 added
   four estimators with the same thresholds, fixed before their first run, raising :math:`K` first to
   12, then to 20 when the Voronoi orientation checks (standard error about 2°) passed close to their
   levels. More fields buy power, while the thresholds never changed. Version 3 added kriging and
   adaptive IDW on the uniform Voronoi profile, which the encoder/decoder refactor made available,
   again with the same thresholds. Version 4 added V5. Version 5 added V4. The axis-artefact index of
   the specification (gradient energy near the axes) had failed its power check, since it measured
   smoothing (the Voronoi controls gave the same values), so axis-locking against a rotated run
   replaced it. Its threshold was chosen knowing diagnostic values on 8 fields, and :math:`K` rose to
   40 when two of its checks lacked power at 20. Version 6 records the correction of the IDW kernel
   (note below), with the thresholds unchanged. Version 8 raised :math:`K` to 80 for the power of
   v4a on adaptive IDW, as argued under V4.

.. note::

   **What coherence does not see.** Until the IDW kernel was corrected, the public Voronoi IDW
   weighted the data by :math:`1/(1+d^p)`, close to a cell mean on the unit square. Its maps looked
   much smoother than the others (contrast 0.67) while their coherence ratio stayed high (0.82),
   because a smooth map dominated by one large-scale trend has very aligned gradients. With the
   corrected weights :math:`1/d^p` its contrast is 0.89 with a coherence ratio of 0.64, in line with
   Mondrian IDW. Smoothness and contrast are separate properties, which V5 tests separately.

**Planned checks.** The following relations for this field will be pre-registered as the estimators
they need become available.

- *paired-relation*. Ordinary kriging with the true covariance, the optimal linear predictor here, has
  a lower RMSE than every ensemble estimator (needs an ordinary-kriging baseline).
- *paired-relation*. The coverage of widened prediction intervals is at least that of raw ensemble
  intervals.
- *paired-relation*. Weighted-draw estimators cover better than averaging ones (needs the draw
  decoders).
- visual **V2**: the coherence of the ensemble map exceeds that of ordinary kriging with a fitted
  isotropic variogram, which blurs the short axis (needs the kriging baseline).
- visual **V3**: the anisotropy is present in single members and simulated fields, not only in the
  median map (coherence equivalent to the truth's).

.. _scenario-S04:

S04 — Zero-inflated field (rain)
================================

:Status: partly ready (not yet implemented)
:Source in the theory: worked example of a variable with an atom at zero
:Claim: an averaging decoder dilutes the atom, giving dry areas small positive values. A draw decoder recovers the dry fraction, while a distance-weighted draw also keeps its placement.

**Setup.** The truth is a Voronoi block-mark field, with one mark per cell of a Voronoi tessellation with uniform nuclei, here with 35 cells whose marks are 0 with probability 0.45 and lognormal otherwise (log-mean 1, log-sd 0.7). Each replicate field has 400 uniformly placed data with queries on a 36×36 grid. Mondrian partitions carry three decoders, IDW, draw and weighted draw (IDW weights), with :math:`T = 300`. The readings are the probability of dry, :math:`p_{dry}` (the fraction of members below a small threshold), and the mean of the wet members.

**Checks.**

- *paired-relation*. The error in the dry fraction is lower for draw than for IDW.
- *paired-relation*. The contrast of :math:`p_{dry}` between dry and wet areas is higher for weighted draw than for draw.
- *almost-sure*. Every draw member is a data value.
- Visual **V6**. The region :math:`p_{dry} > 1/2` overlaps the true dry region better (IoU) and has sharper edges for weighted draw than for draw.

**Depends on.** The draw decoders (planned) for its main checks.

.. _scenario-S05:

S05 — Heavy-tailed field
========================

:Status: partly ready (not yet implemented)
:Source in the theory: worked example of a heavy-tailed variable
:Claim: with heavy tails the median of the members makes a better point estimate than their mean. The ensemble never estimates below the data minimum, while extremes are underestimated and kriging produces halos around them.

**Setup.** The truth is a Voronoi block-mark field, with one mark per cell of a Voronoi tessellation with uniform nuclei, here with 45 cells and Pareto marks (type I, minimum 1, tail index 1.05). Each replicate field has 350 uniformly placed data with queries on a 36×36 grid. The estimators combine Mondrian partitions with IDW, adaptive IDW, kriging and weighted draw, compared with an ordinary-kriging baseline and with widened laws.

**Checks.**

- *paired-relation*. The MAE of the median map lies below the MAE of the mean map, for every decoder.
- *almost-sure*. No ensemble value falls below the data minimum (IDW, adaptive IDW, weighted draw).
- *one-sided*. The signed error in the top decile is negative.
- Visual **V7**. Ordinary kriging shows negative halos around the top-decile cells, while the ensemble maps never go below the data minimum.

**Depends on.** An ordinary-kriging baseline (external dependency) for the comparisons with kriging, and the draw decoders for weighted draw.

.. _scenario-S06:

S06 — Order relations in a non-stationary field
===============================================

:Status: partly ready (not yet implemented)
:Source in the theory: worked example of a field whose law changes across the domain
:Claim: the law estimated by the ensemble stays a proper distribution everywhere, with no order-relation violations, which indicator kriging cannot guarantee.

**Setup.** The truth is a Voronoi block-mark field with 60 cells, its marks Gaussian (sd 0.5) on the left half and lognormal (log-sd 1.1) on the right half. Each replicate field has 400 uniformly placed data, with queries on a 40×40 grid and five thresholds. Mondrian ensemble estimators are compared with an indicator-kriging baseline.

**Checks.**

- *almost-sure*. The ensemble's estimated law shows no order-relation violation and no negative probability.
- *negative control*. The indicator-kriging baseline does show violations.

**Depends on.** An indicator-kriging baseline (external dependency).

.. _scenario-S07:

S07 — Exceedance areas
======================

:Status: ready (not yet implemented), with the area-variance identity to be derived before implementation
:Source in the theory: worked example of the law of the area above a threshold
:Claim: the law of the exceeded area read from the members covers the true area at the nominal rate. The variance of the area over simulated fields follows a closed form.

**Setup.** The truth is a Voronoi block-mark field, with one mark per cell of a Voronoi tessellation with uniform nuclei, here with 30 cells and lognormal marks. Each replicate field has 350 uniformly placed data with queries on a 36×36 grid, estimated by Mondrian IDW with :math:`T = 300`.

**Checks.**

- *binomial coverage*. Over replicate fields, the true area falls in the members' interval at the nominal rate.
- *identity*. The variance of the area over simulated fields matches its closed form, derived independently before implementation.
- Visual **V8**. The region :math:`P(Z > \ell) > 1/2` overlaps the true exceeded region (IoU above a pre-set level), with member regions bracketing the true area.

.. _scenario-S08:

S08 — Support effect on tonnage curves
======================================

:Status: ready (not yet implemented)
:Source in the theory: worked example of the support effect
:Claim: a map of means has a smoother distribution than the truth, reporting more tonnage above a low cut-off and less above a high one.

**Setup.** The truth is a Voronoi block-mark field, with one mark per cell of a Voronoi tessellation with uniform nuclei, here with 30 cells and lognormal marks (log-sd 0.8). Each replicate field has 350 uniformly placed data, estimated by Mondrian IDW.

**Checks.**

- *paired-relation*. The tonnage of the map of means exceeds the truth's at a low cut-off and falls below it at a high cut-off.

.. _scenario-S09:

S09 — Resource categories under preferential sampling
=====================================================

:Status: ready (not yet implemented)
:Source in the theory: worked example of classifying estimates by confidence
:Claim: categories assigned from the estimated law (measured, indicated, inferred) are ordered in error, measured blocks being estimated best.

**Setup.** The truth is a Voronoi block-mark field, with one mark per cell of a Voronoi tessellation with uniform nuclei, here with 30 cells and lognormal marks (log-sd 0.8). A preferential design places 150 uniform data plus 150 in the central quarter, with the estimates on a 36×36 grid aggregated into 3×3 blocks. The estimator is Mondrian IDW.

**Checks.**

- *paired-relation*. Over fields, the error of measured blocks is below that of indicated blocks, itself below that of inferred blocks.

**Depends on.** A preferential-design generator.

.. _scenario-S10:

S10 — Choosing a quantile
=========================

:Status: ready (not yet implemented)
:Source in the theory: worked example of loss-optimal quantiles
:Claim: the empirically optimal quantile level decreases as the cost of over-estimation grows. A block's quantile differs from the mean of its points' quantiles in the predicted direction.

**Setup.** The truth is a Voronoi block-mark field, with one mark per cell of a Voronoi tessellation with uniform nuclei, here with 40 cells and centred lognormal marks (log-sd 0.9). Each replicate field has 300 uniformly placed data with queries on a 40×40 grid, estimated by Mondrian IDW.

**Checks.**

- *paired-relation*. The optimal level decreases with the cost ratio.
- *paired-relation*. The block quantile minus the mean of the point quantiles has the predicted sign below and above the median.

.. _scenario-S11:

S11 — The estimated covariance at two granularities
===================================================

:Status: ready (not yet implemented)
:Source in the theory: the effect of partition size relative to the field's range
:Claim: with coarse partitions (small rate × range) the members are smooth, and adding members does not improve the estimate's error, a "false cure". Fine partitions keep the contrast, while the shape of the estimated covariance changes with granularity.

**Setup.** The truth is a stationary Gaussian field with exponential covariance of range 0.3, sampled at 600 uniformly placed data (1500 in ``full``). A Mondrian cell-mean estimator runs at two granularities, with rate × range equal to 0.5 and to 10.

**Checks.**

- *two-sample*. The member spread differs between the two granularities, while the spread of the fitted law does not.
- *paired-relation*. The RMSE is lower with fine partitions.
- Visual **V5** (false cure). With coarse partitions, axis artefacts shrink from 8 to 200 members without the RMSE improving. With fine partitions, the contrast ratio reaches at least 0.8.
- Visual **V4** on this field, the axis-locking of the median map.
- Visual **V11**. The iso-lines of the estimated covariance are diamond-shaped (:math:`\ell_1`) with coarse partitions and closer to circular with fine ones.

**Depends on.** A cell-mean decoder in the scenario runner.

.. _scenario-S12:

S12 — Edge cases
================

:Status: **implemented**
:Evaluator: ``edge_cases``

**Purpose.** Every estimator must give well-defined output in configurations that real data produce
and naive code mishandles. The theory supplies no target here, so all checks are almost sure, any
single violation counting as a defect.

**Data.** On a Voronoi block-mark field (12 cells, lognormal marks), each of 5 pinned fields
(20 in ``full``) has 30 data confined to :math:`[0.1, 0.9]^2`, plus 3 duplicated locations with the
same value and 3 with a different value (36 samples). The queries form a 10×10 grid over the unit
square, part of which lies outside the data box, plus 5 queries placed exactly on data that are not
duplicated.

**Estimators.** Nine estimators combine IDW, kriging and adaptive IDW with Mondrian partitions (rate
2, about 9 cells) and with both Voronoi profiles (intensity 8), using :math:`T = 50` members (200 in
``full``).

**Checks.** Each check gives one outcome per estimator.

.. list-table::
   :header-rows: 1
   :widths: 8 50 42

   * - id
     - claim
     - estimators
   * - e1
     - the estimator runs without error, with one member per partition at every query
     - all
   * - e2
     - no infinite value (finite, or NaN for an empty cell)
     - all
   * - e3
     - finite members within the data range (up to :math:`10^{-5}` × range, float rounding)
     - the positive-weight decoders, IDW and adaptive IDW (kriging is excluded, since its weights can
       be negative)
   * - e4
     - at a query on a datum, every member equals the datum (up to :math:`10^{-3}` × range)
     - all nine (every decoder interpolates exactly)

**Result.** All 33 outcomes pass, across seeds and in ``full`` mode. The first run found a defect.
Adaptive IDW did not return the datum at its own location in about 11 % of the members (median error
1.4 %, maximum 20 % of the data range), because the guard against division by zero capped the
datum's weight below that of close neighbours under large fitted exponents. The fix gives a query on
a datum that datum's value, as IDW does. Version 2 added the Voronoi IDW estimators to e4. Version 1
had excluded them, since the public Voronoi IDW then weighted the data by :math:`1/(1+d^p)`, which
does not interpolate exactly, before the weights were corrected to :math:`1/d^p`.

.. _scenario-S13:

S13 — Connectivity of high-value bodies
=======================================

:Status: ready (not yet implemented)
:Source in the theory: connectivity as a property of fields, not of point estimates
:Claim: single members and simulated fields keep the connectivity of elongated high-value bodies, which the map of means merges or loses.

**Setup.** The truth is a Voronoi block-mark field with elongated connected high-value bodies, sampled at uniformly placed data on replicate fields. Mondrian IDW provides the members, with ensemble spatial simulation providing simulated fields.

**Checks.**

- Visual **V10**. The Euler-characteristic curve over thresholds of members and simulated fields is equivalent to the truth's (*equivalence*), while the map of means departs from it (*paired-relation*).

**Depends on.** A connectivity functional (Euler characteristic over thresholds).


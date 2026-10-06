.. _scenarios-catalog-t3:

####################################
T3 — Geostatistical scenarios
####################################

Scenarios of tier T3 are **complete geostatistical situations with a known truth** — anisotropy, an
atom at zero, heavy tails, non-stationarity, preferential sampling, degenerate designs — in which the
ensemble estimators must behave as a practitioner needs them to. They are modelled on the worked
examples of the theory and on the situations that break estimators in practice. Their criteria
compare estimators with each other, with reference estimators and with the truth, and include the
**visual criteria** (:doc:`visual`, :doc:`catalog_visual_criteria`).

Truth generators used below: *Voronoi block-mark field* — one mark per cell of a Voronoi
tessellation with uniform nuclei, marks independent of the cells; *Mondrian block-mark field* — the
same on a Mondrian partition; *stationary Gaussian field* — drawn jointly at data and queries, so the
grid holds the field itself.

.. _scenario-S01:

S01 — Simulation keeps the geometry
===================================

:Status: ready (not yet implemented)
:Source in the theory: ensemble spatial simulation
:Claim: fields simulated from the ensemble keep the spatial structure of the truth, unlike independent draws from the same marginal laws.

**Truth.** Voronoi block-mark field (one mark per cell of a Voronoi tessellation with uniform nuclei), 25 cells, centred Gaussian marks.

**Data.** 400 uniformly placed data; replicate fields.

**Estimators.** Mondrian IDW (exponent 2), :math:`T = 200`, ensemble spatial simulation.

**Checks.**

- *paired-relation*: the lag-1 correlation of a simulated field is higher than that of a field of independent draws from the same per-location laws (shuffled).

.. _scenario-S02:

S02 — Where more partitions stop helping
========================================

:Status: ready (not yet implemented)
:Source in the theory: convergence of the ensemble and its error floor
:Claim: the error against the truth stops decreasing beyond some ensemble size, while the estimate itself keeps converging (P1).

**Truth.** Mondrian block-mark field, 35 cells.

**Data.** 250 uniformly placed data.

**Estimators.** Mondrian IDW with :math:`T` up to 3000.

**Checks.**

- P1's :math:`T^{-1/2}` law on this field;
- *identity*: the slope of the error against the truth versus :math:`T` is 0 beyond the estimated threshold :math:`T_0`.

**Depends on.** a Mondrian block-mark truth generator.

.. _scenario-S03:

S03 — Anisotropic field
=======================

:Status: **implemented** (visual criteria V1, V4, V5; other planned checks below)
:Evaluator: ``map_visual``

**Source.** The theory's worked example of an anisotropic stationary Gaussian field.

**Truth.** A stationary Gaussian field on the unit square with exponential covariance
:math:`C(h) = \exp(-\sqrt{h^\top A h})`, :math:`A = R_\vartheta^\top \operatorname{diag}(a_1^{-2},
a_2^{-2}) R_\vartheta`, ranges :math:`a_1 = 0.45`, :math:`a_2 = 0.09` and orientation
:math:`\vartheta = 30°` — a 5:1 anisotropy, the variogram's own ground.

**Data.** 400 uniformly placed samples and a 40×40 grid of queries (cell centres, row-major),
drawn **jointly** by Cholesky factorisation so samples and truth are one realisation. Forty pinned
replicate fields (generator seed 20261005), stored as ``.npy`` with SHA-256 checksums.

**Estimators.** Seven ensembles, none of them told about the anisotropy — cuts are axis-aligned or
Voronoi cells, decoders are isotropic. :math:`T = 100` members in ``ci`` and 300 in ``full``; the
point map is the median of the members at each location.

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
     - kriging, isotropic exponential, nugget 0, sill 1, practical range 0.604 — three times the
       geometric mean of the true ranges, :math:`\sqrt{0.45 \cdot 0.09} = 0.201`, which keeps the
       area of the anisotropy ellipse (spatialize's exponential model is :math:`e^{-3d/r}`)
   * - ``vor-uniform-idw``
     - ``voronoi/spatialize-v1-uniform``, :math:`\lambda_V = 36`
     - IDW, exponent 2
   * - ``vor-data-idw``
     - ``voronoi/spatialize-v1-data``, :math:`\lambda_V = 36`
     - IDW, exponent 2
   * - ``vor-uniform-krig``
     - ``voronoi/spatialize-v1-uniform``, :math:`\lambda_V = 36`
     - kriging, as ``krig`` (through the generic engine entry point)
   * - ``vor-uniform-aidw``
     - ``voronoi/spatialize-v1-uniform``, :math:`\lambda_V = 36`
     - adaptive IDW, as ``aidw`` (through the generic engine entry point)

The Voronoi intensity gives the same expected number of cells as the Mondrian rate:
:math:`\lambda_V |H| = 36 = (1 + \lambda)^2`, the expected number of cells of the book's Mondrian
process of rate 5 on the unit square.

**Claim (V1).** The elongation at 30° must nevertheless be *clearly visible* in each estimator's
point map. Two pre-registered map functionals (:doc:`visual`), the same for every estimator,
computed on each of the :math:`K = 40` fields:

- **v1a — direction** (equivalence, TOST): the orientation error :math:`\theta(\text{map}) - 30°`
  lies within :math:`\pm 10°`.
- **v1b — strength** (one-sided :math:`t`): the coherence of the map is more than half that of the
  truth, :math:`c(\text{map}) / c(\text{truth}) > 0.5`.

Results (``ci``, seed 12345; mean :math:`\pm` standard error over the 40 fields):

.. list-table::
   :header-rows: 1
   :widths: 22 26 13 26 13

   * - estimator
     - orientation error (v1a)
     - p
     - coherence ratio (v1b)
     - p
   * - ``idw``
     - :math:`-0.3° \pm 0.5°`
     - :math:`7.9 \cdot 10^{-23}`
     - :math:`0.61 \pm 0.012`
     - :math:`6.4 \cdot 10^{-12}`
   * - ``aidw``
     - :math:`-1.7° \pm 0.3°`
     - :math:`5.4 \cdot 10^{-27}`
     - :math:`0.98 \pm 0.007`
     - :math:`4.0 \cdot 10^{-43}`
   * - ``krig``
     - :math:`-0.5° \pm 0.4°`
     - :math:`8.5 \cdot 10^{-25}`
     - :math:`0.73 \pm 0.013`
     - :math:`4.8 \cdot 10^{-21}`
   * - ``vor-uniform-idw``
     - :math:`-0.5° \pm 0.5°`
     - :math:`4.8 \cdot 10^{-22}`
     - :math:`0.64 \pm 0.013`
     - :math:`3.4 \cdot 10^{-14}`
   * - ``vor-data-idw``
     - :math:`-0.4° \pm 0.5°`
     - :math:`1.0 \cdot 10^{-21}`
     - :math:`0.64 \pm 0.013`
     - :math:`8.7 \cdot 10^{-14}`
   * - ``vor-uniform-krig``
     - :math:`-0.4° \pm 0.4°`
     - :math:`2.4 \cdot 10^{-24}`
     - :math:`0.73 \pm 0.013`
     - :math:`1.4 \cdot 10^{-20}`
   * - ``vor-uniform-aidw``
     - :math:`-1.1° \pm 0.4°`
     - :math:`1.2 \cdot 10^{-25}`
     - :math:`0.91 \pm 0.010`
     - :math:`2.7 \cdot 10^{-34}`

All fourteen checks pass, stably across seeds. With kriging, the partition barely matters for the
median map (Mondrian and Voronoi median maps correlate at 0.999 on a field, although their members
differ): with about eleven data per cell and a fixed variogram, kriging inside a cell is already
close to global kriging.

.. figure:: /_static/scenarios/S03_compare_13.png
   :width: 100%
   :alt: Truth and the seven median maps of S03 on field 13

   S03, field 13 (``ci``, seed 12345): the truth with its samples and the median map of each
   estimator, on one colour scale. White: the declared 30°; red: the measured orientation.

**Contrast (V5).** A map can show the right direction and still be washed out. V5 compares the
spread of each estimator's median map with that of the best linear predictor — simple kriging with
the true covariance, computed from the same data (:doc:`visual`): **v5** passes when
:math:`\mathrm{std}(\text{map}) / \mathrm{std}(\text{reference}) > 0.9` (one-sided :math:`t` over
the 40 fields). The threshold was fixed before the criterion was first run.

.. list-table::
   :header-rows: 1
   :widths: 30 30 20 20

   * - estimator
     - contrast ratio (v5)
     - p
     - result
   * - ``aidw``
     - :math:`0.96 \pm 0.003`
     - :math:`6.7 \cdot 10^{-26}`
     - pass
   * - ``vor-uniform-aidw``
     - :math:`0.98 \pm 0.002`
     - :math:`1.7 \cdot 10^{-31}`
     - pass
   * - ``krig``
     - :math:`0.96 \pm 0.003`
     - :math:`3.3 \cdot 10^{-23}`
     - pass
   * - ``vor-uniform-krig``
     - :math:`0.96 \pm 0.003`
     - :math:`1.6 \cdot 10^{-23}`
     - pass
   * - ``idw``
     - :math:`0.85 \pm 0.006`
     - 1
     - known failure
   * - ``vor-uniform-idw``
     - :math:`0.89 \pm 0.005`
     - 1
     - known failure
   * - ``vor-data-idw``
     - :math:`0.89 \pm 0.005`
     - 1
     - known failure

IDW with exponent 2 is 11–15 % more washed out than the best linear predictor on this field, on
Mondrian and on Voronoi partitions alike; it is recorded as a known failure (:doc:`statistics`), and
the threshold is not changed.

**Mondrian blocks averaged away (V4).** Each ensemble member of a Mondrian estimator is made of
axis-aligned blocks; the median of many members should not keep them. V4 measures *axis-locking*:
the spectral energy on the coordinate axes against the diagonals of the median map, minus the same
quantity for the same estimator run on the data rotated 45° about the domain centre and evaluated
at the same physical points (:func:`~spatialize.scenarios.stats.maps.axis_lock`). Both maps share
data, decoder and smoothing, so only artefacts tied to the axes remain: a rotation-invariant
partition (Voronoi) gives 0. Pre-registered with :math:`\delta = 0.5` (log ratio of energies):

- **v4a** (Mondrian estimators): axis-locking of the median map below :math:`\delta`;
- **v4b** (power check): single members are axis-locked, mean over 5 members per field above 0;
- **v4c** (calibration, Voronoi estimators): axis-locking of the median map within :math:`\pm\delta`
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
     - :math:`0.35 \pm 0.021`
     - :math:`4.5 \cdot 10^{-9}`
     - :math:`2.59 \pm 0.091`
     - :math:`6.1 \cdot 10^{-28}`
   * - ``krig``
     - :math:`0.28 \pm 0.020`
     - :math:`6.2 \cdot 10^{-14}`
     - :math:`2.88 \pm 0.082`
     - :math:`2.0 \cdot 10^{-31}`
   * - ``aidw``
     - :math:`0.37 \pm 0.028`
     - :math:`1.8 \cdot 10^{-5}`
     - :math:`2.04 \pm 0.081`
     - :math:`5.4 \cdot 10^{-26}`
   * - ``vor-uniform-idw`` (v4c)
     - :math:`-0.01 \pm 0.016`
     - :math:`3.0 \cdot 10^{-29}`
     - —
     - —
   * - ``vor-uniform-krig`` (v4c)
     - :math:`0.01 \pm 0.017`
     - :math:`8.9 \cdot 10^{-29}`
     - —
     - —
   * - ``vor-uniform-aidw`` (v4c)
     - :math:`-0.07 \pm 0.021`
     - :math:`7.7 \cdot 10^{-23}`
     - —
     - —

Single members are strongly axis-locked (+2.0 to +2.9) and the Voronoi controls are at 0, so the
test sees what it is meant to see. At :math:`T = 100` the Mondrian medians keep a residual of about
0.3 (roughly a third more energy on the axes than in the rotated run): clearly below a small
ensemble's, within the pre-registered :math:`\delta`, but not zero.

.. note::

   **History.** The first version of S03 had only ``idw`` and :math:`K = 6` fields; v1b then passed
   with :math:`p \approx 4 \cdot 10^{-4}`, just below a Holm level of :math:`10^{-3}`. Version 2
   added the other four estimators with the same thresholds, fixed before their first run, and
   raised :math:`K` — first to 12, then to 20 when the Voronoi orientation checks (standard error
   about 2°) passed close to their levels. More fields buy power; thresholds were never relaxed.
   Version 3 added kriging and adaptive IDW on the uniform Voronoi profile, which the encoder/decoder refactor made available, again with the
   same thresholds fixed before their first run. Version 4 added V5. Version 5 added V4: the axis-artefact
   index of the specification (gradient energy near the axes) failed its power check and turned out
   to measure smoothing — Voronoi controls gave the same values — so it was replaced by axis-locking
   against a rotated run; its threshold was set by the user knowing diagnostic values on 8 fields,
   and :math:`K` was raised to 40 when two of its checks lacked power at 20. Version 6 records the
   correction of the IDW kernel (see the note below); thresholds unchanged.

.. note::

   **What coherence does not see.** Until the IDW kernel was corrected, the public Voronoi IDW
   weighted the data by :math:`1/(1+d^p)`, which on the unit square is close to a cell mean: its maps
   were visibly much smoother than the others (contrast 0.67), yet their coherence ratio was high
   (0.82), because a smooth map dominated by one large-scale trend has very aligned gradients. With
   the corrected weights :math:`1/d^p` its contrast is 0.89 and its coherence ratio 0.64, in line with
   Mondrian IDW. Smoothness and contrast are separate properties, for separate criteria — which is
   why V5 exists.

**Planned checks.** Further pre-registered relations for this field, to be added as the estimators
they need become available:

- *paired-relation*: ordinary kriging with the true covariance has a lower RMSE than every ensemble
  estimator (it is the optimal linear predictor here) — needs an ordinary-kriging baseline;
- *paired-relation*: the coverage of widened prediction intervals is at least that of raw ensemble
  intervals;
- *paired-relation*: weighted-draw estimators cover better than averaging ones — needs the draw
  decoders;
- visual **V2**: coherence of the ensemble map above that of ordinary kriging with a *fitted
  isotropic* variogram (the fitted model blurs the short axis) — needs the kriging baseline;
- visual **V3**: the anisotropy is present in single members and simulated fields, not only in the
  median map (coherence equivalent to the truth's).

.. _scenario-S04:

S04 — Zero-inflated field (rain)
================================

:Status: partly ready (not yet implemented)
:Source in the theory: worked example of a variable with an atom at zero
:Claim: an averaging decoder dilutes the atom (dry areas get small positive values); a draw decoder recovers the dry fraction; a distance-weighted draw also keeps its placement.

**Truth.** Voronoi block-mark field (one mark per cell of a Voronoi tessellation with uniform nuclei), 35 cells; marks 0 with probability 0.45, otherwise lognormal (log-mean 1, log-sd 0.7).

**Data.** 400 uniformly placed data; queries on a 36×36 grid; replicate fields.

**Estimators.** Mondrian with IDW, draw and weighted-draw (IDW) decoders, :math:`T = 300`.

**Checks.**

- readings: probability of dry (:math:`p_{dry}`, fraction of members below a small threshold) and mean of the wet members;
- *paired-relation*: the error in the dry fraction is lower for draw than for IDW;
- *paired-relation*: the contrast of :math:`p_{dry}` between dry and wet areas is higher for weighted draw than for draw;
- *almost-sure*: every draw member is a data value;
- visual **V6**: the region :math:`p_{dry} > 1/2` overlaps the true dry region better (IoU) and has sharper edges for weighted draw than for draw.

**Depends on.** the draw decoders (planned) for its main checks.

.. _scenario-S05:

S05 — Heavy-tailed field
========================

:Status: partly ready (not yet implemented)
:Source in the theory: worked example of a heavy-tailed variable
:Claim: with heavy tails the median of the members is a better point estimate than their mean; the ensemble never estimates below the data minimum; extremes are underestimated; kriging produces halos around them.

**Truth.** Voronoi block-mark field (one mark per cell of a Voronoi tessellation with uniform nuclei), 45 cells; Pareto (type I, minimum 1, tail index 1.05) marks.

**Data.** 350 uniformly placed data; queries on a 36×36 grid; replicate fields.

**Estimators.** Mondrian with IDW, adaptive IDW, kriging and weighted draw; ordinary kriging as baseline; widened laws.

**Checks.**

- *paired-relation*: MAE of the median map below MAE of the mean map, per decoder;
- *almost-sure*: no ensemble value below the data minimum (IDW, adaptive IDW, weighted draw);
- *one-sided*: the signed error in the top decile is negative;
- visual **V7**: ordinary kriging shows negative halos around the top-decile cells; the ensemble maps never go below the data minimum.

**Depends on.** an ordinary-kriging baseline (external dependency) for the comparisons with kriging; the draw decoders for weighted draw.

.. _scenario-S06:

S06 — Non-stationary field and order relations
==============================================

:Status: partly ready (not yet implemented)
:Source in the theory: worked example of a field whose law changes across the domain
:Claim: the law estimated by the ensemble is a proper distribution everywhere (no order-relation violations), unlike indicator kriging.

**Truth.** Voronoi block-mark field, 60 cells, Gaussian marks (sd 0.5) on the left half and lognormal (log-sd 1.1) on the right half.

**Data.** 400 uniformly placed data; queries on a 40×40 grid; five thresholds.

**Estimators.** Mondrian ensemble estimators; indicator kriging as baseline.

**Checks.**

- *almost-sure*: no order-relation violation and no negative probability in the ensemble's estimated law;
- negative control: the indicator-kriging baseline does show violations.

**Depends on.** an indicator-kriging baseline (external dependency).

.. _scenario-S07:

S07 — Exceedance areas
======================

:Status: ready (not yet implemented); area-variance identity to be derived before implementation
:Source in the theory: worked example of the law of the area above a threshold
:Claim: the law of the exceeded area read from the members covers the true area at the nominal rate; the variance of the area over simulated fields follows a closed form.

**Truth.** Voronoi block-mark field (one mark per cell of a Voronoi tessellation with uniform nuclei), 30 cells, lognormal marks.

**Data.** 350 uniformly placed data; queries on a 36×36 grid; :math:`T = 300`.

**Estimators.** Mondrian IDW.

**Checks.**

- *binomial coverage*: over replicate fields, the true area falls in the members' interval at the nominal rate;
- *identity*: variance of the area over simulated fields against its closed form (derived independently before implementation);
- visual **V8**: the region :math:`P(Z > \ell) > 1/2` overlaps the true exceeded region (IoU above a pre-set level) and member regions bracket the true area.

.. _scenario-S08:

S08 — Support effect on tonnage curves
======================================

:Status: ready (not yet implemented)
:Source in the theory: worked example of the support effect
:Claim: a map of means has a smoother distribution than the truth: above a low cut-off it reports more tonnage, above a high cut-off less.

**Truth.** Voronoi block-mark field (one mark per cell of a Voronoi tessellation with uniform nuclei), 30 cells, lognormal marks (log-sd 0.8).

**Data.** 350 uniformly placed data; replicate fields.

**Estimators.** Mondrian IDW.

**Checks.**

- *paired-relation*: tonnage of the map of means above the truth's at a low cut-off and below it at a high cut-off.

.. _scenario-S09:

S09 — Resource categories under preferential sampling
=====================================================

:Status: ready (not yet implemented)
:Source in the theory: worked example of classifying estimates by confidence
:Claim: categories assigned from the estimated law (measured, indicated, inferred) are ordered in error: measured blocks are estimated best.

**Truth.** Voronoi block-mark field (one mark per cell of a Voronoi tessellation with uniform nuclei), 30 cells, lognormal marks (log-sd 0.8).

**Data.** Preferential design: 150 uniform data plus 150 in the central quarter; 36×36 grid aggregated into 3×3 blocks.

**Estimators.** Mondrian IDW.

**Checks.**

- *paired-relation* over fields: error of measured < indicated < inferred blocks.

**Depends on.** a preferential-design generator.

.. _scenario-S10:

S10 — Choosing a quantile
=========================

:Status: ready (not yet implemented)
:Source in the theory: worked example of loss-optimal quantiles
:Claim: the empirically optimal quantile level decreases as the cost of over-estimation grows; a block's quantile differs from the mean of its points' quantiles in the predicted direction.

**Truth.** Voronoi block-mark field (one mark per cell of a Voronoi tessellation with uniform nuclei), 40 cells, centred lognormal marks (log-sd 0.9).

**Data.** 300 uniformly placed data; queries on a 40×40 grid.

**Estimators.** Mondrian IDW.

**Checks.**

- *paired-relation*: optimal level decreasing in the cost ratio;
- *paired-relation*: sign of block quantile minus mean of point quantiles below and above the median.

.. _scenario-S11:

S11 — Granularity and the estimated covariance
==============================================

:Status: ready (not yet implemented)
:Source in the theory: the effect of partition size relative to the field's range
:Claim: with coarse partitions (rate × range small) the members are smooth and the estimate's error does not improve by adding members (a "false cure"); with fine partitions contrast is kept; the shape of the estimated covariance changes with granularity.

**Truth.** Stationary Gaussian field with exponential covariance, range 0.3.

**Data.** 600 uniformly placed data (1500 in ``full``).

**Estimators.** Mondrian cell-mean estimator at two granularities (rate × range 0.5 and 10).

**Checks.**

- *two-sample*: member spread differs between the two granularities while the spread of the fitted law does not;
- *paired-relation*: RMSE lower with fine partitions;
- visual **V5** (false cure): with coarse partitions, axis artefacts shrink from 8 to 200 members but the RMSE does not improve; with fine partitions the contrast ratio is at least 0.8;
- visual **V4** on this field: axis-locking of the median map;
- visual **V11**: iso-lines of the estimated covariance diamond-shaped (:math:`\ell_1`) with coarse partitions, closer to circular with fine ones.

**Depends on.** a cell-mean decoder in the scenario runner.

.. _scenario-S12:

S12 — Edge cases
================

:Status: **implemented**
:Evaluator: ``edge_cases``

**Purpose.** Every estimator must give well-defined output in the configurations real data produce
and that break naive code. There is no target from the theory: all checks are almost sure, so a
single violation is a defect.

**Data.** On a Voronoi block-mark field (12 cells, lognormal marks), each of 5 pinned fields
(20 in ``full``) has 30 data confined to :math:`[0.1, 0.9]^2`, plus 3 duplicated locations with the
same value and 3 with a different value (36 samples). The queries are a 10×10 grid over the unit
square — so part of them lie outside the data box — and 5 queries placed exactly on data that are
not duplicated.

**Estimators.** Nine: IDW, kriging and adaptive IDW on Mondrian partitions (rate 2, about 9 cells)
and on both Voronoi profiles (intensity 8); :math:`T = 50` members (200 in ``full``).

**Checks** (one outcome per estimator):

.. list-table::
   :header-rows: 1
   :widths: 8 50 42

   * - id
     - claim
     - estimators
   * - e1
     - runs: no error, one member per partition at every query
     - all
   * - e2
     - no infinite value (finite, or NaN for an empty cell)
     - all
   * - e3
     - finite members within the data range (up to :math:`10^{-5}` × range, float rounding)
     - positive-weight decoders: IDW and adaptive IDW; not kriging, whose weights can be negative
   * - e4
     - at a query on a datum, every member equals the datum (up to :math:`10^{-3}` × range)
     - all nine (every decoder interpolates exactly)

**Result.** All 33 outcomes pass, across seeds and in ``full`` mode. The first run found a defect:
adaptive IDW did not return the datum at its own location in about 11 % of the members (median
error 1.4 %, maximum 20 % of the data range), because the guard against division by zero capped the
datum's weight below that of close neighbours under large fitted exponents. It was fixed by giving
a query on a datum that datum's value, as IDW does. Version 2 added the Voronoi IDW estimators to e4, from which version 1 excluded them because the
public Voronoi IDW then weighted the data by :math:`1/(1+d^p)` and did not interpolate exactly; the
weights were corrected to :math:`1/d^p`.

.. _scenario-S13:

S13 — Connectivity of high-value bodies
=======================================

:Status: ready (not yet implemented)
:Source in the theory: connectivity as a property of fields, not of point estimates
:Claim: single members and simulated fields keep the connectivity of elongated high-value bodies; the map of means merges or loses them.

**Truth.** Voronoi block-mark field with elongated connected high-value bodies.

**Data.** Uniformly placed data; replicate fields.

**Estimators.** Mondrian IDW; ensemble spatial simulation.

**Checks.**

- visual **V10**: the Euler-characteristic curve over thresholds of members and simulated fields equivalent to the truth's (*equivalence*); the map of means departs from it (*paired-relation*).

**Depends on.** a connectivity functional (Euler characteristic over thresholds).


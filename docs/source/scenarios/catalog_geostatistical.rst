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

S03 and S12 are implemented. The other scenarios are ordered below by priority, which weighs what a
scenario adds to the claims already tested against what it still needs. Each scenario's section
repeats its rank and lists what its implementation takes.

#. **S02**, the error floor of the ensemble. It complements P1 and needs only a Mondrian block-mark
   generator.
#. **S04**, a zero-inflated field. It is the main practical case of the draw decoders, now in the
   catalogue.
#. **S08**, the support effect on tonnage curves. One functional on existing generators.
#. **S01**, simulation keeps the geometry. It is the first test of simulation, needing a runner
   method for simulated fields.
#. **S11**, granularity and covariance. Its visual functionals exist, except the shape of the
   estimated covariance.
#. **S10**, loss-optimal quantiles. New readings on existing generators.
#. **S09**, resource categories. A preferential-design generator and a classification rule fixed in
   advance.
#. **S07**, exceedance areas. Its coverage check is ready, its identity awaits a derivation.
#. **S05**, a heavy-tailed field. Its main checks are ready, its comparison with kriging awaits a
   baseline.
#. **S06**, order relations in a non-stationary field. P4 already covers the ensemble's side, the
   negative control awaits an indicator-kriging baseline.
#. **S13**, connectivity. A new generator, a new functional and simulated fields.

.. _scenario-S01:

S01 — Simulation keeps the geometry
===================================

:Status: ready (not yet implemented)
:Priority: 4 of 11 among the pending scenarios
:Source in the theory: ensemble spatial simulation
:Claim: fields simulated from the ensemble keep the spatial structure of the truth, which independent draws from the same per-location laws lose.

**Setup.** The truth is a Voronoi block-mark field, with one mark per cell of a Voronoi tessellation with uniform nuclei, here with 25 cells and centred Gaussian marks. 400 uniformly placed data are drawn on each replicate field, estimated by Mondrian IDW (exponent 2) with :math:`T = 200`, then simulated by ensemble spatial simulation.

**Checks.**

- *paired-relation*. The lag-1 correlation of a simulated field exceeds that of a field of independent draws from the same per-location laws.

**To implement.**

- A runner method for simulated fields, ``simulate(estimator, samples, values, queries, *,
  n_fields, seed)``, optional like ``cells``, which Spatialize's runner gives through
  :func:`~spatialize.gs.ess.ess_sample`. The suite has no reading of simulation yet.
- Pinned replicate fields from the existing Voronoi block-mark generator (25 cells, centred Gaussian
  marks), written with checksums as for S12.
- An evaluator reading the lag-1 correlation along the rows and columns of the grid, for each
  simulated field and for a field of independent draws from the members at each location, paired by
  field.
- The number of fields, fixed before the first run from
  :func:`~spatialize.scenarios.stats.budget.min_sign_test_fields` with a margin for power.

.. _scenario-S02:

S02 — Where more partitions stop helping
========================================

:Status: ready (not yet implemented)
:Priority: 1 of 11 among the pending scenarios
:Source in the theory: convergence of the ensemble and its error floor
:Claim: the error against the truth stops decreasing beyond some ensemble size, while the estimate itself keeps converging (P1).

**Setup.** The truth is a Mondrian block-mark field with 35 cells, sampled at 250 uniformly placed data, estimated by Mondrian IDW with :math:`T` up to 3000.

**Checks.**

- *identity*. P1's :math:`T^{-1/2}` law holds on this field.
- *identity*. Beyond the estimated threshold :math:`T_0`, the slope of the error against the truth versus :math:`T` equals 0.

**To implement.**

- A Mondrian block-mark generator next to the Voronoi one in ``generators/fields.py``. It draws the
  cells of the theory's Mondrian process on the unit square, at the rate giving 35 cells on average
  (:math:`(1 + \lambda)^2 = 35`, :math:`\lambda \approx 4.9`), with one independent mark per
  cell. Its cells can be checked against E2's co-occurrence before use.
- Pinned replicate fields, 250 data each, with a grid of queries.
- An evaluator computing the RMSE of the ensemble mean against the truth for ensemble sizes on a
  logarithmic grid up to 3 000, averaged over independent ensembles. It reuses P1's slope check
  (evaluator ``convergence``) on this field.
- A rule for :math:`T_0`, fixed before the first run, for instance the smallest size whose error
  lies within one standard error of the error at 3 000 members, then the slope test beyond it.

.. _scenario-S03:

S03 — Anisotropic field
=======================

:Status: **implemented** (visual criteria V1, V4 and V5 and the ordering of the decoders, with further
   checks planned below)
:Evaluator: ``map_visual``

**Source.** The theory's worked example of an anisotropic stationary Gaussian field.

**Truth.** A stationary Gaussian field on the unit square with exponential covariance
:math:`C(h) = \exp(-\sqrt{h^\top A h})`, :math:`A = R_\vartheta^\top \operatorname{diag}(a_1^{-2},
a_2^{-2}) R_\vartheta`, ranges :math:`a_1 = 0.45`, :math:`a_2 = 0.09` and orientation
:math:`\vartheta = 30°`, a 5:1 anisotropy of the kind a variogram is designed to capture.

**Data.** 400 uniformly placed samples and a 40×40 grid of queries (cell centres, row-major), drawn
jointly by Cholesky factorisation so that samples and truth belong to one realisation. Eighty
replicate fields are pinned (generator seed 20261005), stored as ``.npy`` with SHA-256 checksums.

**Estimators.** Thirteen ensembles take part, none of them told about the anisotropy, since the cuts
are axis-aligned or Voronoi cells while the decoders are isotropic. They use :math:`T = 100` members in
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
   * - ``sharp``
     - Mondrian, :math:`\lambda = 5`
     - sharpened adaptive IDW, metric MAE, :math:`\kappa_r = 1.5`, :math:`\kappa_g = 0.2`,
       :math:`\varrho_{\max} = 3`
   * - ``wdraw-idw``
     - Mondrian, :math:`\lambda = 5`
     - weighted draw with the IDW weights, exponent 2
   * - ``wdraw-krig``
     - Mondrian, :math:`\lambda = 5`
     - weighted draw with the kriging weights of ``krig``, negative weights clipped
   * - ``wdraw-aidw``
     - Mondrian, :math:`\lambda = 5`
     - weighted draw with the adaptive IDW weights, metric MAE
   * - ``wdraw-sharp``
     - Mondrian, :math:`\lambda = 5`
     - weighted draw with the sharpened weights, as ``sharp``
   * - ``draw``
     - Mondrian, :math:`\lambda = 5`
     - uniform draw

The Voronoi intensity gives the same expected number of cells as the Mondrian rate, since
:math:`\lambda_V |H| = 36 = (1 + \lambda)^2`, the expected number of cells of the theory's Mondrian
process of rate 5 on the unit square. The last six estimators hold the decoders of
:doc:`../theory/decoders` that average with sharper weights or draw one datum of the cell. They
share the partitions of the first three for the same seed, so their comparison with those isolates
the decoder.

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
     - :math:`3.4 \cdot 10^{-50}`
     - :math:`0.97 \pm 0.006`
     - :math:`5.1 \cdot 10^{-78}`
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
     - :math:`-1.1° \pm 0.2°`
     - :math:`6.8 \cdot 10^{-51}`
     - :math:`0.90 \pm 0.007`
     - :math:`2.5 \cdot 10^{-65}`
   * - ``sharp``
     - :math:`-1.7° \pm 0.2°`
     - :math:`4.1 \cdot 10^{-51}`
     - :math:`0.96 \pm 0.006`
     - :math:`3.7 \cdot 10^{-78}`
   * - ``wdraw-idw``
     - :math:`-1.0° \pm 0.4°`
     - :math:`4.3 \cdot 10^{-39}`
     - :math:`0.57 \pm 0.009`
     - :math:`1.7 \cdot 10^{-11}`
   * - ``wdraw-krig``
     - :math:`-1.2° \pm 0.3°`
     - :math:`4.4 \cdot 10^{-41}`
     - :math:`0.61 \pm 0.009`
     - :math:`4.7 \cdot 10^{-20}`
   * - ``wdraw-aidw``
     - :math:`-1.8° \pm 0.2°`
     - :math:`5.4 \cdot 10^{-49}`
     - :math:`0.87 \pm 0.007`
     - :math:`1.4 \cdot 10^{-64}`
   * - ``wdraw-sharp``
     - :math:`-1.7° \pm 0.2°`
     - :math:`2.4 \cdot 10^{-50}`
     - :math:`0.88 \pm 0.006`
     - :math:`2.5 \cdot 10^{-69}`
   * - ``draw``
     - :math:`-5.0° \pm 1.2°`
     - :math:`3.7 \cdot 10^{-5}`
     - :math:`0.41 \pm 0.016`
     - 1 (known failure)

Every estimator finds the direction. All the strength checks pass except the uniform draw's, stably
across seeds for the first seven. The adaptive and sharpened decoders give the most coherent maps of
the scenario. A weighted draw loses some coherence against the averaging decoder whose weights it borrows, 0.87
against 0.97 with the adaptive weights, since each member holds observed values only. The uniform
draw keeps the direction, although less precisely, while its strength falls below the bound. Its
median at a location is close to the median of the cell's data, so the map follows the cells more
than the field. The decoder is built for intervals, not for a map (:doc:`../theory/decoders`), so
the failure is recorded as a known one. The threshold stays unchanged.

With kriging, the partition barely matters for the
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
     - :math:`0.97 \pm 0.002`
     - :math:`9.3 \cdot 10^{-50}`
     - pass
   * - ``vor-uniform-aidw``
     - :math:`0.99 \pm 0.002`
     - :math:`1.4 \cdot 10^{-61}`
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
   * - ``sharp``
     - :math:`1.01 \pm 0.001`
     - :math:`7.3 \cdot 10^{-76}`
     - pass
   * - ``wdraw-idw``
     - :math:`0.95 \pm 0.003`
     - :math:`3.0 \cdot 10^{-34}`
     - pass
   * - ``wdraw-krig``
     - :math:`1.00 \pm 0.002`
     - :math:`2.9 \cdot 10^{-58}`
     - pass
   * - ``wdraw-aidw``
     - :math:`1.02 \pm 0.002`
     - :math:`3.4 \cdot 10^{-69}`
     - pass
   * - ``wdraw-sharp``
     - :math:`1.05 \pm 0.002`
     - :math:`4.2 \cdot 10^{-72}`
     - pass
   * - ``draw``
     - :math:`0.55 \pm 0.010`
     - 1
     - known failure

IDW with exponent 2 is 10–15 % more washed out than the best linear predictor on this field, on
Mondrian and on Voronoi partitions alike. The shortfall is recorded as a known failure
(:doc:`statistics`), leaving the threshold unchanged.

A weighted draw restores the contrast its averaging decoder loses. With the IDW weights the ratio
rises from 0.85 to 0.95, enough to pass, and with the adaptive or sharpened weights the median map is
as contrasted as the reference. A draw returns observed values, so the median of the members keeps
the spread of the data where an average shrinks it. The uniform draw ignores the position of the
location inside its cell, which washes its map out to half the reference's contrast, a second face of
the failure seen under V1.

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
     - :math:`0.37 \pm 0.022`
     - :math:`4.1 \cdot 10^{-8}`
     - :math:`1.91 \pm 0.055`
     - :math:`7.5 \cdot 10^{-50}`
   * - ``sharp``
     - :math:`0.29 \pm 0.022`
     - :math:`6.3 \cdot 10^{-15}`
     - :math:`1.67 \pm 0.050`
     - :math:`6.5 \cdot 10^{-49}`
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
     - :math:`-0.05 \pm 0.017`
     - :math:`4.2 \cdot 10^{-41}`
     - —
     - —

Single members are strongly axis-locked (+2.0 to +2.9) while the Voronoi controls sit at 0, so the
test sees what it is meant to see. At :math:`T = 100` the Mondrian medians keep a residual of about
0.3, roughly a third more energy on the axes than in the rotated run. The residual stays well below a
small ensemble's and within the pre-registered :math:`\delta`, without vanishing.

The sharpened decoder keeps fewer of the blocks than adaptive IDW, 0.29 against 0.37. In version 10
it seemed to keep more (0.46), which came from the adaptive weights then used, :math:`1/(10^{-10} +
d^p)`. They saturated at short distances with large exponents, giving many nearby data the same
weight, so the members followed their cells' data in blocks. The weights now used are exact.

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

**The order of the decoders.** The decoders differ less in the map than in what the ensemble says
about its own error. At each grid point the members give an interval from their 5 % and 95 %
quantiles, of nominal coverage 90 %. Its actual coverage is the share of the grid where the truth lies
inside, computed on each field. The error of the median map is its RMSE against the truth. The theory
orders four decoders that share one partition, from the averaged adaptive decoder through its
sharpened form and the weighted draw to the uniform draw, each covering more than the one before. It
also expects the sharpened map to be the more accurate of the two averaging ones. Four paired checks,
field by field over the 80 fields, test these relations
(:func:`~spatialize.scenarios.stats.families.paired_relation`).

.. list-table::
   :header-rows: 1
   :widths: 30 26 20 24

   * - check
     - mean paired difference
     - p
     - result
   * - coverage, ``sharp`` above ``aidw``
     - :math:`+0.026 \pm 0.001`
     - :math:`1.6 \cdot 10^{-43}`
     - pass
   * - coverage, ``wdraw-aidw`` above ``sharp``
     - :math:`+0.128 \pm 0.002`
     - :math:`6.2 \cdot 10^{-73}`
     - pass
   * - coverage, ``draw`` above ``wdraw-aidw``
     - :math:`+0.157 \pm 0.002`
     - :math:`4.9 \cdot 10^{-75}`
     - pass
   * - RMSE, ``sharp`` below ``aidw``
     - :math:`+0.0011 \pm 0.0005`
     - 0.99
     - known failure

The coverage and the error of every Mondrian estimator, read with the same seeds, place the checks
in context.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - estimator
     - coverage of the 90 % interval
     - RMSE of the median map
   * - ``idw``
     - :math:`0.46 \pm 0.003`
     - :math:`0.500 \pm 0.003`
   * - ``krig``
     - :math:`0.47 \pm 0.003`
     - :math:`0.462 \pm 0.002`
   * - ``aidw``
     - :math:`0.60 \pm 0.003`
     - :math:`0.437 \pm 0.002`
   * - ``sharp``
     - :math:`0.63 \pm 0.003`
     - :math:`0.438 \pm 0.002`
   * - ``wdraw-krig``
     - :math:`0.74 \pm 0.003`
     - :math:`0.515 \pm 0.003`
   * - ``wdraw-sharp``
     - :math:`0.75 \pm 0.002`
     - :math:`0.474 \pm 0.002`
   * - ``wdraw-aidw``
     - :math:`0.76 \pm 0.002`
     - :math:`0.470 \pm 0.002`
   * - ``wdraw-idw``
     - :math:`0.80 \pm 0.002`
     - :math:`0.520 \pm 0.003`
   * - ``draw``
     - :math:`0.91 \pm 0.002`
     - :math:`0.697 \pm 0.007`

An averaging decoder sees only the variation between cells (:doc:`../theory/esi`), so its intervals
miss the truth more than half the time with IDW or kriging. Adapting the weights to the cell widens
them a little, sharpening a little more. A weighted draw restores most of the variation within the
cell, at the cost of a slightly worse median map, the uniform draw reaching the nominal 90 % with a
map far less accurate. The coverage ordering holds by wide margins, so the theory's comparison of
coverage carries over to this field. Among the weighted draws, ``wdraw-idw`` covers slightly more
than ``wdraw-aidw``, which suggests that the adaptive fit concentrates the weights more than
exponent 2 does. No decoder gives both the best map and nominal intervals, which is why the choice
follows the purpose (:doc:`../theory/decoders`).

The sharpened map is not more accurate than the adaptive one here, its RMSE being 0.001 higher, so
the fourth check is recorded as a known failure. Its two factors pull in opposite directions. On ten
fields the boost of the data with large leave-one-out residuals alone lowers the RMSE by 0.004,
while the exponent raised with the cell's gradient alone raises it by 0.003, concentrating the
weight on the nearest data more than this field rewards. Version 10 had found the sharpened map
better by 0.004, an advantage that came from the adaptive weights then used, which saturated at
short distances (see V4 above).

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
   v4a on adaptive IDW, as argued under V4. Version 10 added the sharpened and drawing decoders with
   V1 and V5 at the same thresholds, V4 for the sharpened map and the four ordering checks, all fixed
   before their first run. That run failed three of them, recorded as known failures. Version 11
   follows the correction of the adaptive weights, which had saturated at short distances. V4 on the
   sharpened map passes since then, while the sharpened map lost its advantage in error (the fourth
   ordering check), both explained above.

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
- visual **V2**: the coherence of the ensemble map exceeds that of ordinary kriging with a fitted
  isotropic variogram, which blurs the short axis (needs the kriging baseline).
- visual **V3**: the anisotropy is present in single members and simulated fields, not only in the
  median map (coherence equivalent to the truth's).

.. _scenario-S04:

S04 — Zero-inflated field (rain)
================================

:Status: ready (not yet implemented)
:Priority: 2 of 11 among the pending scenarios
:Source in the theory: worked example of a variable with an atom at zero
:Claim: an averaging decoder dilutes the atom, giving dry areas small positive values. A draw decoder recovers the dry fraction, while a distance-weighted draw also keeps its placement.

**Setup.** The truth is a Voronoi block-mark field, with one mark per cell of a Voronoi tessellation with uniform nuclei, here with 35 cells whose marks are 0 with probability 0.45 and lognormal otherwise (log-mean 1, log-sd 0.7). Each replicate field has 400 uniformly placed data with queries on a 36×36 grid. Mondrian partitions carry three decoders, IDW, draw and weighted draw (IDW weights), with :math:`T = 300`. The readings are the probability of dry, :math:`p_{dry}` (the fraction of members below a small threshold), and the mean of the wet members.

**Checks.**

- *paired-relation*. The error in the dry fraction is lower for draw than for IDW.
- *paired-relation*. The contrast of :math:`p_{dry}` between dry and wet areas is higher for weighted draw than for draw.
- *almost-sure*. Every draw member is a data value.
- Visual **V6**. The region :math:`p_{dry} > 1/2` overlaps the true dry region better (IoU) and has sharper edges for weighted draw than for draw.

**To implement.**

- A zero-inflated mark sampler for the existing Voronoi block-mark generator (0 with probability
  0.45, lognormal otherwise), with pinned replicate fields.
- An evaluator with the readings :math:`p_{dry}`, the share of members below a small threshold fixed
  before the first run, and the mean of the wet members. The dry-fraction error and the contrast are
  then paired relations over the fields. The decoders ``idw``, ``draw`` and ``wdraw_idw`` are in the
  catalogue.
- The support check, which the evaluator ``draw_laws`` already decides (kind ``support``), applied
  to these fields.
- For V6, the overlap through the existing ``level_set_iou``. The sharpness of the edges needs a
  functional to define, such as the mean gradient of :math:`p_{dry}` across the true dry boundary.

.. _scenario-S05:

S05 — Heavy-tailed field
========================

:Status: partly ready (not yet implemented)
:Priority: 9 of 11 among the pending scenarios
:Source in the theory: worked example of a heavy-tailed variable
:Claim: with heavy tails the median of the members makes a better point estimate than their mean. The ensemble never estimates below the data minimum, while extremes are underestimated and kriging produces halos around them.

**Setup.** The truth is a Voronoi block-mark field, with one mark per cell of a Voronoi tessellation with uniform nuclei, here with 45 cells and Pareto marks (type I, minimum 1, tail index 1.05). Each replicate field has 350 uniformly placed data with queries on a 36×36 grid. The estimators combine Mondrian partitions with IDW, adaptive IDW, kriging and weighted draw, compared with an ordinary-kriging baseline and with widened laws.

**Checks.**

- *paired-relation*. The MAE of the median map lies below the MAE of the mean map, for every decoder.
- *almost-sure*. No ensemble value falls below the data minimum (IDW, adaptive IDW, weighted draw).
- *one-sided*. The signed error in the top decile is negative.
- Visual **V7**. Ordinary kriging shows negative halos around the top-decile cells, while the ensemble maps never go below the data minimum.

**To implement.**

- A Pareto mark sampler (type I, minimum 1, tail index 1.05) for the Voronoi block-mark generator,
  with pinned replicate fields.
- An evaluator for the MAE of the median map against that of the mean map (paired, per decoder),
  the data minimum (almost-sure) and the signed error in the top decile (one-sided). None of these
  needs a baseline. The decoders, weighted draw included, are in the catalogue.
- The widened laws, read through the runner's ``law_cdf`` (P4).
- For V7 and the comparisons with kriging, an ordinary-kriging baseline. The suite already computes
  simple kriging in NumPy (``simple_kriging_exponential``), so ordinary kriging can be added the
  same way, with its variogram fixed before the first run, which keeps the suite free of external
  packages.

.. _scenario-S06:

S06 — Order relations in a non-stationary field
===============================================

:Status: partly ready (not yet implemented)
:Priority: 10 of 11 among the pending scenarios
:Source in the theory: worked example of a field whose law changes across the domain
:Claim: the law estimated by the ensemble stays a proper distribution everywhere, with no order-relation violations, which indicator kriging cannot guarantee.

**Setup.** The truth is a Voronoi block-mark field with 60 cells, its marks Gaussian (sd 0.5) on the left half and lognormal (log-sd 1.1) on the right half. Each replicate field has 400 uniformly placed data, with queries on a 40×40 grid and five thresholds. Mondrian ensemble estimators are compared with an indicator-kriging baseline.

**Checks.**

- *almost-sure*. The ensemble's estimated law shows no order-relation violation and no negative probability.
- *negative control*. The indicator-kriging baseline does show violations.

**To implement.**

- A mark sampler that depends on the nucleus's position, Gaussian on the left half and lognormal on
  the right, with pinned replicate fields.
- The ensemble's check, reading the law at five thresholds through the runner's ``law_cdf``. P4
  already decides this property on a stationary field, so S06 adds the non-stationary case.
- For the negative control, an indicator-kriging baseline, the ordinary kriging of each threshold's
  indicator. It can be written in NumPy as for S05, with the violations of the order relations
  counted at each location.

.. _scenario-S07:

S07 — Exceedance areas
======================

:Status: ready (not yet implemented), with the area-variance identity to be derived before implementation
:Priority: 8 of 11 among the pending scenarios
:Source in the theory: worked example of the law of the area above a threshold
:Claim: the law of the exceeded area read from the members covers the true area at the nominal rate. The variance of the area over simulated fields follows a closed form.

**Setup.** The truth is a Voronoi block-mark field, with one mark per cell of a Voronoi tessellation with uniform nuclei, here with 30 cells and lognormal marks. Each replicate field has 350 uniformly placed data with queries on a 36×36 grid, estimated by Mondrian IDW with :math:`T = 300`.

**Checks.**

- *binomial coverage*. Over replicate fields, the true area falls in the members' interval at the nominal rate.
- *identity*. The variance of the area over simulated fields matches its closed form, derived independently before implementation.
- Visual **V8**. The region :math:`P(Z > \ell) > 1/2` overlaps the true exceeded region (IoU above a pre-set level), with member regions bracketing the true area.

**To implement.**

- Pinned replicate fields from the Voronoi block-mark generator with lognormal marks.
- The coverage check (binomial, over the fields), which needs no new derivation. The members'
  exceeded areas give an interval, against which the true area is read.
- The closed form of the variance of the exceeded area over simulated fields, derived independently
  before the identity is added. That check also needs the runner method for simulated fields of
  S01.
- V8 through the existing ``level_set_iou``, with its pre-set level fixed before the first run.

.. _scenario-S08:

S08 — Support effect on tonnage curves
======================================

:Status: ready (not yet implemented)
:Priority: 3 of 11 among the pending scenarios
:Source in the theory: worked example of the support effect
:Claim: a map of means has a smoother distribution than the truth, reporting more tonnage above a low cut-off and less above a high one.

**Setup.** The truth is a Voronoi block-mark field, with one mark per cell of a Voronoi tessellation with uniform nuclei, here with 30 cells and lognormal marks (log-sd 0.8). Each replicate field has 350 uniformly placed data, estimated by Mondrian IDW.

**Checks.**

- *paired-relation*. The tonnage of the map of means exceeds the truth's at a low cut-off and falls below it at a high cut-off.

**To implement.**

- Pinned replicate fields from the Voronoi block-mark generator with lognormal marks (log-sd 0.8).
- A tonnage functional, the share of the grid above a cut-off, for the map of means and for the
  truth.
- The two cut-offs, fixed before the first run (for instance the truth's 20 % and 80 % quantiles),
  with one paired relation over the fields at each.

.. _scenario-S09:

S09 — Resource categories under preferential sampling
=====================================================

:Status: ready (not yet implemented)
:Priority: 7 of 11 among the pending scenarios
:Source in the theory: worked example of classifying estimates by confidence
:Claim: categories assigned from the estimated law (measured, indicated, inferred) are ordered in error, measured blocks being estimated best.

**Setup.** The truth is a Voronoi block-mark field, with one mark per cell of a Voronoi tessellation with uniform nuclei, here with 30 cells and lognormal marks (log-sd 0.8). A preferential design places 150 uniform data plus 150 in the central quarter, with the estimates on a 36×36 grid aggregated into 3×3 blocks. The estimator is Mondrian IDW.

**Checks.**

- *paired-relation*. Over fields, the error of measured blocks is below that of indicated blocks, itself below that of inferred blocks.

**To implement.**

- A preferential-design generator in ``generators/fields.py``, generalising the design P9 draws
  (150 uniform data plus 150 in the central quarter).
- A classification rule from the estimated law, fixed before the first run. One choice is the
  relative width of each 3×3 block's 90 % interval, below a first threshold for measured blocks and
  below a second for indicated ones, the others being inferred.
- The block aggregation of the members, each member's mean over the block's points, then the paired
  relations of the block errors across the categories.

.. _scenario-S10:

S10 — Choosing a quantile
=========================

:Status: ready (not yet implemented)
:Priority: 6 of 11 among the pending scenarios
:Source in the theory: worked example of loss-optimal quantiles
:Claim: the empirically optimal quantile level decreases as the cost of over-estimation grows. A block's quantile differs from the mean of its points' quantiles in the predicted direction.

**Setup.** The truth is a Voronoi block-mark field, with one mark per cell of a Voronoi tessellation with uniform nuclei, here with 40 cells and centred lognormal marks (log-sd 0.9). Each replicate field has 300 uniformly placed data with queries on a 40×40 grid, estimated by Mondrian IDW.

**Checks.**

- *paired-relation*. The optimal level decreases with the cost ratio.
- *paired-relation*. The block quantile minus the mean of the point quantiles has the predicted sign below and above the median.

**To implement.**

- Pinned replicate fields from the Voronoi block-mark generator with centred lognormal marks
  (log-sd 0.9).
- An asymmetric linear loss over a grid of cost ratios. The empirically optimal level is the
  quantile level of the members that minimises the realised loss against the truth. A paired
  relation then reads its decrease with the cost ratio.
- The block quantile, the quantile of the members' block means, against the mean of the point
  quantiles, with the sign predicted below and above the median.

.. _scenario-S11:

S11 — The estimated covariance at two granularities
===================================================

:Status: ready (not yet implemented)
:Priority: 5 of 11 among the pending scenarios
:Source in the theory: the effect of partition size relative to the field's range
:Claim: with coarse partitions (small rate × range) the members are smooth, and adding members does not improve the estimate's error, a "false cure". Fine partitions keep the contrast, while the shape of the estimated covariance changes with granularity.

**Setup.** The truth is a stationary Gaussian field with exponential covariance of range 0.3, sampled at 600 uniformly placed data (1500 in ``full``). A Mondrian cell-mean estimator runs at two granularities, with rate × range equal to 0.5 and to 10.

**Checks.**

- *two-sample*. The member spread differs between the two granularities, while the spread of the fitted law does not.
- *paired-relation*. The RMSE is lower with fine partitions.
- Visual **V5** (false cure). With coarse partitions, axis artefacts shrink from 8 to 200 members without the RMSE improving. With fine partitions, the contrast ratio reaches at least 0.8.
- Visual **V4** on this field, the axis-locking of the median map.
- Visual **V11**. The iso-lines of the estimated covariance are diamond-shaped (:math:`\ell_1`) with coarse partitions and closer to circular with fine ones.

**To implement.**

- Pinned fields from the existing stationary Gaussian generator (exponential covariance, range
  0.3).
- Two estimators with the cell mean, which is in the catalogue, at rate × range 0.5 and 10.
- The member spread (two-sample) and the RMSE (paired relation), with functionals the suite has.
- V5 and V4 through the existing ``axis_artifact_index``, ``contrast_ratio`` and ``axis_lock``.
  V11 reads the shape of the estimated covariance's iso-lines, for which ``axis_diagonal_log_ratio``
  can serve, applied to the covariance of the members.

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
2, about 9 cells) and with both Voronoi profiles (intensity 8). Fourteen more, added in version 5,
combine the cell mean, the uniform draw, the weighted draws and the sharpened adaptive IDW with the
Mondrian and the uniform Voronoi partitions. All use :math:`T = 50` members (200 in ``full``).

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
     - the positive-weight decoders and the draws, everything but kriging, whose weights can be
       negative
   * - e4
     - at a query on a datum, every member equals the datum (up to :math:`10^{-3}` × range)
     - the exact interpolators, all but the cell mean and the uniform draw, which do not
       interpolate, and the kriging draw, whose weights at a datum are a unit vector only up to
       float32 rounding

**Result.** All 83 outcomes pass, across seeds and in ``full`` mode, the 50 of version 5 included. The first run found a defect.
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
:Priority: 11 of 11 among the pending scenarios
:Source in the theory: connectivity as a property of fields, not of point estimates
:Claim: single members and simulated fields keep the connectivity of elongated high-value bodies, which the map of means merges or loses.

**Setup.** The truth is a Voronoi block-mark field with elongated connected high-value bodies, sampled at uniformly placed data on replicate fields. Mondrian IDW provides the members, with ensemble spatial simulation providing simulated fields.

**Checks.**

- Visual **V10**. The Euler-characteristic curve over thresholds of members and simulated fields is equivalent to the truth's (*equivalence*), while the map of means departs from it (*paired-relation*).

**To implement.**

- A generator of elongated connected high-value bodies, for instance a Voronoi block-mark field on
  stretched coordinates or a thresholded anisotropic Gaussian field.
- An Euler-characteristic functional over thresholds, computed on the grid in NumPy by counting the
  vertices, edges and faces of the excursion set.
- The runner method for simulated fields of S01.
- The equivalence of the curves of the members and of the simulated fields to the truth's (TOST),
  with its margin fixed before the first run, and the paired relation for the map of means.


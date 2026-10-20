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

S01, S02, S03, S04, S07, S08, S09, S10, S11 and S12 are implemented. The other scenarios are ordered below by priority, which
weighs what a scenario adds to the claims already tested against what it still needs. Each scenario's
section repeats its rank and lists what its implementation takes.

#. **S05**, a heavy-tailed field. Its main checks are ready, its comparison with kriging awaits a
   baseline.
#. **S06**, order relations in a non-stationary field. P4 already covers the ensemble's side, the
   negative control awaits an indicator-kriging baseline.
#. **S13**, connectivity. A new generator, a new functional and simulated fields.

.. _scenario-S01:

S01 — Simulated fields keep the local texture
=============================================

:Status: **implemented**
:Evaluator: ``simulation_texture``
:Source in the theory: ensemble spatial simulation, and the reading of local statistics in the paper on
  spatial distributional estimation (Egaña et al. 2025, AIMS Mathematics, sec. 4.3)
:Claim: fields simulated from the ensemble reproduce the histograms of the truth within small windows,
  which the map of means smooths away.

**Setup.** The truth is a Voronoi block-mark field with 25 cells and standard normal marks. Each pinned
replicate field has 400 uniformly placed data with the field on a 48×48 grid. Mondrian IDW (exponent 2,
rate 5, :math:`T = 200`, ``empty_cells="coarsen"``) gives the members, from which ensemble spatial
simulation draws 5 fields, through the runner's optional method ``simulate``, with the default local
model, a kernel density with Gaussian kernels. The reading is the mean Jensen-Shannon distance, in bits,
between the histograms of the truth and of a map in 60 square windows placed at random, the same
windows for both, with ten bins between the truth's quantiles. The scenario uses 12 fields in ``ci``
and 24 in ``full``.

**Checks.**

- ``local-histograms-7``, *paired-relation*. In 7×7 windows the distance of the simulated fields,
  averaged over them, is lower than that of the map of means.
- ``local-histograms-13``, *paired-relation*. The same in 13×13 windows.

**Results.** On seeds 1 and 2 in ``ci`` both checks pass, the simulated fields lying closer to the
truth than the map of means by 0.03 in 7×7 windows and by 0.04 in 13×13 windows. A run takes about 10 s.
The maps of the truth, of the map of means and of three simulated fields are saved with
``--save-maps``. The simulated fields keep the units of the truth, their placement and their level,
with a texture within each that the map of means blurs.

**History.** The specification claimed that a simulated field keeps a lag-1 correlation which
independent draws from the same local laws lose. Ensemble spatial simulation draws each location
independently from its law, the dependence lying between the laws, which share the data of their
neighbourhoods. A simulated field and a field of independent draws from the members are then the
same kind of object. On 8 fields their lag-1 correlations were 0.845 and 0.848, against 0.843 for the
truth and 0.930 for the map of means. The check now reads the structure that the paper on spatial
distributional estimation reads, the local histograms in windows.

.. _scenario-S02:

S02 — Where more partitions stop helping
========================================

:Status: **implemented**
:Evaluator: ``error_floor``
:Source in the theory: convergence of the ensemble and its error floor
:Claim: the ensemble mean keeps converging as the number of partitions grows (P1), while its error
  against the truth stops decreasing beyond some ensemble size.

**Setup.** The truth is a Mondrian block-mark field, one standard normal mark per cell of the theory's
Mondrian process of budget 4.9 on the unit square, about 35 cells. Each pinned replicate field holds
250 uniformly placed data, with the field on a 30×30 grid. Mondrian IDW (exponent 2, rate 5,
``empty_cells="coarsen"``) computes, for each of nine ensemble sizes from 5 to 3000, 30 independent
ensembles (60 in ``full``) on each of 8 fields (16 in ``full``).

**Checks.**

- ``spread``, *identity*. The slope against :math:`\log T` of half the logarithm of the variance of
  the ensemble mean across the independent ensembles, averaged over the grid, is :math:`-1/2`. One
  standardised difference per field, its standard error from 300 bootstrap resamples of the ensembles.
- ``floor``, *equivalence*. Beyond the size :math:`T_0`, the smallest whose mean error against the
  truth lies within one combined standard error of the error at 3000 members, the slope of the
  logarithm of the error against :math:`\log T` lies within :math:`\pm 0.01`, less than 1 % of error
  for each e-fold of members.

**Results.** On seeds 1 and 2 in ``ci`` both checks pass, the largest standardised difference of the
spread being 1.5 and 2.0. The mean slope beyond :math:`T_0` was -0.00005 and -0.0002, against a
margin of 0.01. A run takes about 1 min 35 s.

**History.** The spread is read through the variance averaged over the grid. The mean over the grid of
the logarithm of the standard deviation, P1's reading, gave a slope of -0.481 on the same ensembles
before the first run, against -0.502 for the variance. On a block-mark field the members are far from
normal, a cell reaching a datum or not, and the bias of the logarithm of a standard deviation changes
with :math:`T`. The floor is an equivalence rather than an identity, since the error approaches its
floor without reaching it exactly. Its slope beyond :math:`T_0` was -0.0003 per field, which an
identity with more fields would reject although it has no practical consequence.

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
- visual **V3** on simulated fields, which waits for the runner method for simulated fields of S01.

**Visual criteria V3 and V9 (version 13).** V3 reads the anisotropy in single members, the first
five members of each field without empty cells. Their coherence over the truth's must lie within
:math:`\pm 0.25` of 1, and the log of their directional range ratio over the truth's within
:math:`\pm \log 1.5`, by equivalence over the 80 fields. On seed 1 the adaptive and sharpened decoders
pass both (coherence ratios 0.83 and 0.84, log range ratios :math:`-0.12` and :math:`-0.09`).
Kriging and Voronoi IDW pass the ranges but keep only 0.62 and 0.52 of the coherence, and Mondrian
IDW keeps 0.50 of it, its range ratio (log :math:`-0.38`) lying too close to the margin. These four
outcomes are recorded as known failures.

V9 compares the roughness of the median map with that of simple kriging with the true covariance.
The averaging decoders give smoother maps, IDW 0.75, kriging 0.64 and Voronoi IDW 0.73 times the
reference's roughness. The adaptive ones fit a high exponent, raised further by the sharpened one, so
their maps are rougher by design, 1.14 and 1.29 times. The control, simple kriging from the 8 nearest
data, is 1.012 times as rough as the global one, which shows the functional sees the jumps of a
cropped neighbourhood. These results come from a run of the 16 new checks alone (``--check``), whose
Holm levels are less strict than those of a run of the catalogue.

A pilot on fields 0–4 had suggested that every member lost the anisotropy. It averaged members with
empty cells, whose NaN gave a coherence of 0, and was set aside once the run measured the members
without them.

.. _scenario-S04:

S04 — Zero-inflated field (rain)
================================

:Status: **implemented**
:Evaluator: ``atom_at_zero``
:Source in the theory: worked example of a variable with an atom at zero
:Claim: an averaging decoder dilutes the atom, giving dry areas small positive values. A draw decoder
  recovers the dry fraction, while a distance-weighted draw also keeps its placement.

**Setup.** The truth is a Voronoi block-mark field with 35 cells whose marks are 0 with probability
0.45 and lognormal otherwise (log-mean 1, log-sd 0.7). Each pinned replicate field has 400 uniformly
placed data with the field on a 36×36 grid. Mondrian partitions of rate 5 carry three decoders, IDW,
draw and weighted draw (IDW weights), with :math:`T = 300` and ``empty_cells="coarsen"``. The reading
is the probability of a dry location, :math:`p_{dry}`, the share of members below 0.1, a level the wet
marks reach with probability near :math:`10^{-6}`. The scenario uses 10 fields in ``ci`` and 20 in
``full``.

**Checks.**

- ``dry-fraction``, *paired-relation*. The error of the dry share of the grid, the mean of
  :math:`p_{dry}` against the true share, is lower for the draw than for IDW.
- ``contrast``, *paired-relation*. The mean of :math:`p_{dry}` over the true dry region minus that over
  the wet region is higher for the weighted draw than for the draw.
- ``support``, *almost-sure*. Every member of the draw and of the weighted draw is a data value.
- ``dry-overlap``, *paired-relation*, visual **V6**. The overlap (IoU) of the region
  :math:`p_{dry} > 1/2` with the true dry region is higher for the weighted draw than for the draw.
- ``dry-edges``, *paired-relation*, visual **V6**. The mean magnitude of the gradient of
  :math:`p_{dry}` over the pixels of the true dry boundary is higher for the weighted draw than for the
  draw.

**Results.** On seeds 1 and 2 in ``ci`` every check passes. IDW misses the dry share by 0.28 more than
the draw. The weighted draw raises the contrast by 0.37, the overlap by 0.15 and the sharpness of the
edges by 0.20 per pixel, while none of 7.8 million drawn members is anything but a data value. A run
takes about a second. The maps of :math:`p_{dry}` are saved with ``--save-maps``, IDW marking as dry
only the places close to a dry datum and the draw blurring the edges that the weighted draw keeps.

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

S07 — The area above a limit
============================

:Status: **implemented**
:Evaluator: ``exceedance_area``
:Source in the theory: the law of the area above a limit, read from the members, against the product law
  of the simulated fields (Proposition 12.3.9 in the draft)
:Claim: the members' band of the exceeded area covers the true area at least at its nominal rate when the
  draws are coupled within each cell, while averaged members lose that spread. Over simulated fields, which
  draw every location independently, the variance of the area follows a closed form, which the members
  exceed through the partition they share.

**Setup.** The truth is a Voronoi block-mark field with 30 cells and lognormal marks of log-scale 1.0.
Each pinned replicate field holds 350 uniformly placed data, with the field on a 36×36 grid. The limit
:math:`\ell` is the truth's 80 % quantile, the area :math:`A` the share of the grid above it. Two Mondrian
estimators of rate 5 (:math:`T = 300`, ``empty_cells="coarsen"``) give the members, IDW (exponent 2) and
the weighted draw ``wdraw_idw`` (exponent 2) with ``coupling="cell"``. Each member gives one area. The
share of members above :math:`\ell` at a location gives its probability of exceeding the limit,
:math:`E_v`. From the IDW members, ensemble spatial simulation draws 400 fields with Gamma widening. The
scenario uses 48 fields in ``ci`` and 96 in ``full``.

Under ensemble spatial simulation the locations are drawn independently, each from its own law, so the
indicators :math:`1\{\tilde Z(v) > \ell\}` are independent Bernoulli variables of parameters :math:`E_v`.
The exceeded area :math:`A = |V|^{-1} \sum_v 1\{\tilde Z(v) > \ell\}` then has

.. math::

   \mathbb E\, A = \frac{1}{|V|} \sum_{v \in V} E_v,
   \qquad
   \operatorname{Var} A = \frac{1}{|V|^2} \sum_{v \in V} E_v (1 - E_v),

the covariances between distinct indicators vanishing. The members of an ensemble share a partition,
so their indicators are positively dependent, which makes their area vary more.

**Checks.**

- ``coverage``, *coverage*. The central 90 % interval of the areas of the weighted draw's members
  holds the true area in at least 90 % of the fields, an exact binomial test which a conservative band
  passes.
- ``coverage-averaged``, *coverage*, negative control. The averaged members of IDW lose the spread of
  the cells, so their band covers far less often, which the test must reject.
- ``area-variance``, *identity*. Over the simulated fields the variance of the area equals the closed
  form at their own exceedance probabilities, corrected by :math:`M/(M-1)` for :math:`M` fields, with
  the standard error of a sample variance of a normal area under the null.
- ``shared-partition``, *one-sided*. The logarithm of the IDW members' area variance over the closed
  form at their probabilities exceeds 0.
- ``v8-region``, *one-sided*, visual V8. The overlap of the region :math:`E_v > 1/2` of IDW with the
  true region above :math:`\ell` exceeds 0.5.

**Results.** On seeds 1 to 3 in ``ci`` and seed 1 in ``full`` every check passes. The weighted draw's
band covers the true area in every field, a conservative band, while that of IDW covers 44 % to 48 % of
them. The simulated fields follow the closed form, with standardised differences of mean near 0. The
IDW members' area varies about 11 times more than the closed form gives, a logarithm of 2.4. The overlap
of V8 is 0.64. A run takes 24 s in ``ci``, 47 s in ``full``.

**History.** The specification asked for the closed form to be derived before the identity was added.
The theory states and proves it. A study on 12 and 24 fields found the IDW members' band covering the
true area in 1 of 12 fields, the members understating the area, 0.150 against 0.179. Widened member by
member they covered 11 of 24, coupled by location in the weighted draw 13 of 24, coupled by cell 23 of
24, so the coverage check reads the weighted draw coupled by cell, with IDW as a negative control. The
specification's bracketing of the true area by the member regions is that coverage. The first runs
found the band covering every field, which a two-sided test at the nominal rate would reject with more
fields, so the check asks for at least the nominal rate. The identity first took its standard error
from the fourth moment of the simulated areas. That error shrinks with a low sample variance, so it
inflated such fields, one of them reaching :math:`z = -4.2` with the indicators of adjacent locations
uncorrelated. It now takes the standard error under the null.

.. _scenario-S08:

S08 — Support effect on selection curves
========================================

:Status: **implemented**
:Evaluator: ``selection_curve``
:Source in the theory: worked example of the support effect
:Claim: a map of means is smoother than the field it estimates, so it holds more of the domain above a
  low cut-off and less above a high one.

**Setup.** The truth is a Voronoi block-mark field with 30 cells and lognormal marks (log-sd 0.8). Each
pinned replicate field has 350 uniformly placed data with the field on a 36×36 grid, estimated by
Mondrian IDW (exponent 2, rate 5, :math:`T = 300`, ``empty_cells="coarsen"``). The selection curve is
the share of the grid above a cut-off, read on the map of means and on the truth. The scenario uses 12
fields in ``ci`` and 24 in ``full``.

**Checks.**

- ``above-low``, *paired-relation*. Above the truth's 10 % quantile the map of means holds more of the
  domain than the truth.
- ``above-high``, *paired-relation*. Above the truth's 90 % quantile the map of means holds less of the
  domain than the truth.

**Results.** On seeds 1 and 2 in ``ci`` both checks pass. Above the low cut-off the map of means holds
11 % more of the domain than the truth, above the high one 3 % less. A run takes under a second.

**History.** The specification named the 20 % and 80 % quantiles as cut-offs. A study before the first
run measured the difference of the shares at several levels, 4.5 standard deviations per field at 20 %
against -0.5 at 80 %, -1.8 at 90 % and -0.8 at 95 %. On a block-mark field the truth does not vary within
a cell, so the high cells keep much of their height in the map of means where data reach them, and only
their edges are lost. The 90 % quantile reads that loss best, with the 10 % quantile as its
counterpart.

.. _scenario-S09:

S09 — Confidence classes under preferential sampling
====================================================

:Status: **implemented**
:Evaluator: ``confidence_classes``
:Source in the theory: classifying blocks by the confidence in their average, read from its law, and
  the kriging variance, which depends on the data locations alone
:Claim: classes read from the law of each block average are ordered in the error they commit, the
  well estimated blocks erring least. Classes cut from the kriging variance with the same counts
  separate the error less, since that variance follows the density of the data whatever their values.

**Setup.** The truth is a Voronoi block-mark field with 30 cells and lognormal marks of log-scale 0.8.
Each pinned replicate field holds 300 data, 150 uniformly placed and 150 more in the central square
:math:`[0.25, 0.75]^2`, a quarter of the area holding half the data. The field lies on a 36×36 grid
cut into 144 blocks of 3×3 cells. Mondrian IDW (exponent 2, rate 5, :math:`T = 300`,
``empty_cells="coarsen"``) gives the members, widened member by member with Gamma laws
(:func:`~spatialize.gs.ess.widen_members`). Their block averages, column by column, give the law of
each block average. Its relative half-width :math:`h_B = (q_{0.95} - q_{0.05}) / (2 q_{0.5})` places a
block among the well estimated up to 0.15, the moderately estimated up to 0.30 and the poorly
estimated beyond. A block's error is :math:`|q_{0.5} - Z_B| / Z_B`, with :math:`Z_B` the true block
average, a class's error the median over its blocks. The proxy ranks the blocks by their
ordinary-kriging variance under an exponential variogram fixed in advance, with a practical range of
0.183, the side of a square of the mean cell area, and a nugget of 0.4 of the sill
(:func:`~spatialize.scenarios.generators.fields.block_kriging_variance_exponential`). Its classes
receive the law's counts. The scenario uses 12 fields in ``ci`` and 24 in ``full``.

**Checks.**

- ``well-below-moderate``, *paired-relation*. The well estimated blocks commit a lower error than
  the moderately estimated ones.
- ``moderate-below-poor``, *paired-relation*. The moderately estimated blocks commit a lower error
  than the poorly estimated ones.
- ``law-against-proxy``, *paired-relation*. The error of the poorly estimated class minus that of the
  well estimated class is greater for the law's classes than for the proxy's.

A field where a class holds no block drops out of the checks on that class.

**Results.** On seeds 1 and 2 in ``ci`` the three checks pass on all 12 fields. The error grows by
0.05 from the well to the moderately estimated blocks and by 0.16 to 0.17 from the moderately to the
poorly estimated ones. Across the law's classes it spreads more than across the proxy's, by 0.11 and
0.13. In the maps the proxy's classes follow the dense centre of the design, while the law's classes do not.
A run takes about 4 s.

**History.** The specification named the classes of resource reporting, measured, indicated and
inferred, here replaced by general terms. On the raw members a study on 12 fields found the classes
ordered, while the block law covered its 90 % interval in 61 % of the blocks. On the members widened
member by member it covered 87 %, so the scenario reads the widened members. The comparison with the
kriging variance was added after the same study. With a variogram fixed in advance the proxy orders
the error a little, 0.095, 0.148 and 0.183 across its classes against 0.037, 0.096 and 0.239 for the
law.

.. _scenario-S10:

S10 — Choosing a quantile
=========================

:Status: **implemented**
:Evaluator: ``quantile_choice``
:Source in the theory: the functional an asymmetric loss chooses, and the law of a block average
:Claim: when over-estimation costs :math:`c` times under-estimation, the level that minimises the
  realised loss falls as :math:`c` grows, the theory placing it at :math:`1/(1 + c)`. A block's quantile
  differs from the average of its locations' quantiles, lying above it below the median and below it
  above, at a lower realised loss.

**Setup.** The truth is a Voronoi block-mark field with 40 cells and lognormal marks of log-scale 0.9,
centred by their sample mean. Each pinned replicate field holds 300 uniformly placed data, with the
field on a 40×40 grid cut into 100 blocks of 4×4 cells. Mondrian IDW (exponent 2, rate 5,
:math:`T = 300`, ``empty_cells="coarsen"``) gives the members, widened member by member with
skew-normal laws (:func:`~spatialize.gs.ess.widen_members`, through the runner's optional method
``widened_members``). A block's law is that of the block averages of the widened members, column by
column. The loss of reporting :math:`\hat z` for the truth :math:`z` is :math:`c(\hat z - z)` above
the truth and :math:`z - \hat z` below. The optimal level is the level, among 0.01, 0.02, …, 0.99,
whose quantile map has the lowest mean loss against the truth. The scenario uses 12 fields in ``ci``
and 24 in ``full``.

**Checks.**

- ``level-falls-5``, *paired-relation*. The optimal level is lower at :math:`c = 5` than at
  :math:`c = 1`.
- ``level-falls-1``, *paired-relation*. The optimal level is lower at :math:`c = 1` than at
  :math:`c = 1/5`.
- ``level-near-5`` and ``level-near-0.2``, *equivalence*, known failures. The optimal level lies within
  0.05 of :math:`1/(1 + c)`, 1/6 at :math:`c = 5` and 5/6 at :math:`c = 1/5`.
- ``block-gap-low``, *paired-relation*. At level 1/6 the block's quantile exceeds the average of its
  locations' quantiles, averaged over the blocks.
- ``block-gap-high``, *paired-relation*. At level 0.9 it lies below.
- ``block-loss-low`` and ``block-loss-high``, *paired-relation*. At level 1/6, the cost
  :math:`c = 5`, and at level 0.9, the cost :math:`c = 1/9`, the block's quantile incurs a lower mean
  loss against the true block averages than the averaged quantiles.

**Results.** On seeds 1 and 2 in ``ci`` the six checks pass and the two known failures fail. The
optimal level is lower by 0.21 at :math:`c = 5` than at equal costs, by 0.24 at equal costs than at
:math:`c = 1/5`. At :math:`c = 5` it lies 0.13 above 1/6, at :math:`c = 1/5` 0.08 below 5/6, so the
widened laws are too wide in their tails on this field. The block's quantile exceeds the averaged
quantiles by 0.51 at level 1/6 and falls short of them by 0.71 at level 0.9, in nearly every block.
The averaged quantiles incur a mean loss higher by 0.32 at :math:`c = 5` and by 0.04 at level 0.9. A
run takes about 9 s.

**History.** The specification read the levels on the raw members. A study on 12 fields found its
claims there, but with gaps of 0.06 and 0.09 and the block loss reversed, the raw members being
under-dispersed. The theory reads the widened members draw by draw, while the library's widening,
drawing each location's law anew, lost the member index. The library gained
:func:`~spatialize.gs.ess.widen_members`, which keeps it, so the scenario reads the widened members. The two checks
against :math:`1/(1 + c)` are registered as known failures, since the study placed the optimal level
0.145 above 1/6 and 0.078 below 5/6 with standard deviations near 0.02.

.. _scenario-S11:

S11 — The estimated covariance at two granularities
===================================================

:Status: **implemented**
:Evaluator: ``granularity``
:Source in the theory: the size of the partition's cells against the field's range, the co-occurrence of
  the Mondrian process and the widening of the ensemble's law
:Claim: coarse partitions give a smooth map whose error more members cannot bring near that of fine
  partitions, while its axis artefacts fade, a "false cure". The map keeps the partition's
  :math:`\ell_1` covariance. Fine partitions keep the contrast. Once widened, the laws of both
  granularities spread alike.

**Setup.** The truth is a stationary Gaussian field with an isotropic exponential covariance of range
0.3. Each pinned replicate field holds 600 uniformly placed data, with the field on a 40×40 grid. A
Mondrian cell-mean estimator (:math:`T = 200`, ``empty_cells="coarsen"``) runs at two granularities,
with rate × range equal to 0.5, about 7 cells, and to 10, about 1180 cells. Every reading uses the
median map, of the 200 members or of the first 8. The axis-locking compares it with the median map of
the same estimator on the data rotated by 30°. The scenario uses 40 fields in ``ci`` and 80 in
``full``.

**Checks.**

- ``law-spread``, *equivalence*. The mean standard deviations of 50 fields drawn from the widened laws,
  through the runner's method ``simulate`` with ``widening="auto"``, differ by less than a factor 1.15
  between the granularities.
- ``error``, *paired-relation*. The error of the median map against the truth, its root mean square
  over the grid, is lower with fine partitions.
- ``v5-artefacts``, *paired-relation*, visual V5. With coarse partitions the axis-locking of the median
  map falls from 8 to 200 members.
- ``v5-error``, *paired-relation*, visual V5. The error of coarse partitions with 200 members stays
  above that of fine partitions with 8.
- ``v5-contrast``, *one-sided*, visual V5. With fine partitions the contrast ratio of the median map
  exceeds 0.8.
- ``v4-axis-lock``, *paired-relation*, visual V4. The axis-locking of the median map is stronger with
  coarse partitions.
- ``v11-diamond``, *paired-relation*, visual V11. The covariance of the median map is more
  diamond-shaped with coarse partitions. The reading divides the e-folding length of the map's
  autocovariance along the axes by that along the diagonals
  (:func:`~spatialize.scenarios.stats.maps.covariance_diamond_index`), about 1 for circular iso-lines
  and :math:`\sqrt 2` for diamonds.

**Results.** On seeds 1 and 2 in ``ci`` every check passes. The error of the median map is lower by 0.40
with fine partitions. With coarse partitions the axis-locking falls by 2.4 to 2.5 from 8 to 200
members, while their error stays 0.38 above that of fine partitions with 8 members. The contrast ratio
of fine partitions is 0.89. With coarse partitions the axis-locking is higher by 1.1 and 1.4, the
diamond index by 0.13 and 0.14. The widened spreads differ by 4 %. A run takes about 1 min 45 s.

**History.** The specification changed in five points before the first run, after a study on 8 fields.

#. The member spread did not differ consistently between the granularities, 0.28 against 0.32, the
   coarse one ranging from 0.15 to 0.49. The spread check keeps the widened law alone, whose spread the
   two granularities share, 0.46 against 0.44.
#. The coarse error falls a little with more members, from 0.757 to 0.728, so the false cure asks for
   it to stay above the fine error with 8 members.
#. V4's bound of 0.5 from S03, calibrated on IDW, fails for the cell mean with fine partitions (0.87),
   so V4 compares the two granularities.
#. Read on the covariance of the members, V11 is degenerate, since their co-occurrence under the
   Mondrian process gives :math:`\ell_1` iso-lines at any granularity. V11 reads the covariance of the
   median map instead.
#. The specification asked for 1500 data in ``full``. Both modes use 600, differing in their number of
   fields.

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


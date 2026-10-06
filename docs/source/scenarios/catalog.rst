.. _scenarios-catalog:

#########
Catalogue
#########

Each scenario lives in its own directory under ``spatialize/scenarios/catalog/``, with a
``scenario.yaml`` descriptor (the setup, the estimators and the pre-registered checks) and, for
pinned data, ``data/*.npy`` files with a ``CHECKSUMS.sha256`` manifest. The descriptor is the
normative definition of the scenario; this page explains the two scenarios that are implemented
and lists those that are planned.

Example run (``ci`` mode, seed 12345, about six minutes on a multi-core machine; excerpt):

.. code-block:: text

   runner=spatialize mode=ci seed=12345 α_suite=0.001 (Holm)
   scenario/check                                    family          p-value     level  expect result
   E2-mondrian-pair-cooccurrence/c1                  gof-closed     2.43e-87   3.2e-05  reject PASS  — k=8 N=3100 ...
   S03-anisotropic-field/v1a                         equivalence    7.86e-23   6.7e-05  pass   PASS  — mean=-0.3027 se=0.472 ...
   S03-anisotropic-field/v1b                         one-sided      6.41e-12   0.00011  pass   PASS  — mean=0.6097 se=0.0116 ...
   ...
   S03-anisotropic-field/v1a-vor-uniform-aidw [run]  equivalence    1.22e-25   4.8e-05  pass   PASS  — mean=-1.079 se=0.363 ...
   ...
   S03-anisotropic-field/v4a-idw                     one-sided      4.53e-09   0.00013  pass   PASS  — mean=0.3495 se=0.0207 ...
   S03-anisotropic-field/v4b-idw                     one-sided      6.08e-28     4e-05  pass   PASS  — mean=2.594 se=0.0915 ...
   ...
   S03-anisotropic-field/v4c-vor-uniform-krig [run]  equivalence    8.94e-29   3.8e-05  pass   PASS  — mean=0.005446 se=0.0166 ...
   ...
   S03-anisotropic-field/v5-idw                      one-sided             1   0.00033  pass   KNOWN  — mean=0.8461 se=0.00607 ...
   ...
   S03-anisotropic-field/v5-krig                     one-sided      3.26e-23   5.9e-05  pass   PASS  — mean=0.956 se=0.00266 ...
   S03-anisotropic-field/v5-vor-uniform              one-sided             1    0.0005  pass   KNOWN  — mean=0.6674 se=0.0133 ...
   ...
   S12-edge-cases/e1-m-idw                           almost-sure           1     exact  pass   PASS  — 0 violations ...
   ...
   S12-edge-cases/e4-m-aidw                          almost-sure           1     exact  pass   PASS  — 0 violations ...
   ...

   59 passed, 0 failed, 3 known failures, 0 skipped (reproduce with --mode ci --seed 12345)

The ``level`` column is the Holm level each p-value was compared with (:doc:`statistics`). ``[run]``
marks checks whose estimator is not in the public API yet and was reached through the compiled
engine's generic entry point (:doc:`extending`). ``exact`` marks almost-sure checks, which spend
no error budget, and ``KNOWN`` a recorded known failure (:doc:`statistics`).

.. _scenario-E2:

E2 — Mondrian pair co-occurrence (T1)
=====================================

**Source.** The theory's definition of the Mondrian process and the closed form of its pair
co-occurrence (a classical property of the Mondrian process; Roy & Teh, 2009).

**Claim.** For a Mondrian process of rate :math:`\lambda`, the probability that two locations
:math:`x, y` fall in the same cell is

.. math::

   e(\{x, y\}) = \exp\!\left(-\lambda\, \lVert x - y \rVert_1\right).

**How it is read through the estimator.** The suite has no access to an implementation's
partitions — it only sees ensembles. So it uses a single datum (value 1) at the centre of the unit
square and the ``"nan"`` empty-cell policy: a member is finite exactly when the query falls in the
datum's cell. The fraction of finite members at a query at displacement :math:`h` therefore
estimates :math:`e(\{x_0, x_0 + h\})`. Displacements 0.05, 0.1, 0.2 and 0.3 are taken along an
axis and along the diagonal (with :math:`\ell_1`-normalised steps), giving :math:`k = 8`
proportions; rate :math:`\lambda = 3`.

**Test.** gof-closed: :math:`\sum_{i=1}^{8} z_i^2 \sim \chi^2_8` under the closed form.
:math:`N = 3\,100` members in ``ci`` (:math:`\delta = 0.05`) and :math:`19\,300` in ``full``
(:math:`\delta = 0.02`), which give power 0.9 at :math:`\delta` for up to 50 tests under Holm.

**Expectation per profile.** ``mondrian/book``: pass. ``mondrian/spatialize-v1``: **reject** —
this is a negative control. Spatialize's Mondrian deviates from the book's process by design
(:doc:`encoders`), by 4–21 % on the unit square; the test must see it, and does, with
:math:`p \approx 10^{-87}` in ``ci`` (largest :math:`|z| = 9.6`, at displacement 0.2 along the
diagonal: :math:`\hat p = 0.463` against :math:`0.549`) and :math:`|z| \approx 27` in ``full``. A
test that did not reject here would not have the power its "passes" elsewhere claim.

.. _scenario-S03:

S03 — Anisotropic field, visual criterion V1 (T3)
=================================================

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
     - :math:`4.9 \cdot 10^{-21}`
   * - ``vor-uniform-idw``
     - :math:`-0.7° \pm 1.3°`
     - :math:`1.1 \cdot 10^{-8}`
     - :math:`0.82 \pm 0.027`
     - :math:`6.9 \cdot 10^{-15}`
   * - ``vor-data-idw``
     - :math:`-0.4° \pm 1.4°`
     - :math:`2.8 \cdot 10^{-8}`
     - :math:`0.79 \pm 0.026`
     - :math:`3.6 \cdot 10^{-14}`
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
     - :math:`0.67 \pm 0.013`
     - 1
     - known failure
   * - ``vor-data-idw``
     - :math:`0.67 \pm 0.013`
     - 1
     - known failure

IDW with exponent 2 is 15 % more washed out than the best linear predictor on this field; the
public Voronoi IDW, whose weights :math:`1/(1+d^p)` are close to uniform on the unit square, 33 %.
Both are recorded as known failures (:doc:`statistics`); the threshold is not changed.

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
     - :math:`6.3 \cdot 10^{-14}`
     - :math:`2.88 \pm 0.082`
     - :math:`2.0 \cdot 10^{-31}`
   * - ``aidw``
     - :math:`0.37 \pm 0.028`
     - :math:`1.8 \cdot 10^{-5}`
     - :math:`2.04 \pm 0.081`
     - :math:`5.4 \cdot 10^{-26}`
   * - ``vor-uniform-idw`` (v4c)
     - :math:`-0.07 \pm 0.087`
     - :math:`6.7 \cdot 10^{-6}`
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
   and :math:`K` was raised to 40 when two of its checks lacked power at 20.

.. note::

   **What coherence does not see.** The two Voronoi IDW maps are visibly much smoother than the
   others, yet their coherence ratio is high: coherence measures how aligned the gradients are, and
   a smooth map dominated by one large-scale trend has very aligned gradients. Its orientation is
   correspondingly noisier (field 13 above: 14–15° against 25° in the truth). The smoothing comes
   from the IDW kernel of the public Voronoi estimator, :math:`1/(1+d^p)`
   (:doc:`../development/architecture`), not from the Voronoi partition: kriging and adaptive IDW
   on the same partitions are not smoothed, and the standard kernel :math:`1/d^p` on the same
   partitions gives a map standard deviation of 0.86 instead of 0.70 and a coherence ratio of 0.68
   instead of 0.99 on field 0. Smoothness and contrast are separate properties, for separate
   criteria (contrast ratio, roughness; :doc:`visual`).

.. _scenario-S12:

S12 — Edge cases (T3)
=====================

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
     - exact interpolators: Mondrian IDW, kriging, adaptive IDW; not the public Voronoi IDW, whose
       weights :math:`1/(1+d^p)` do not interpolate exactly

**Result.** All 31 outcomes pass, across seeds and in ``full`` mode. The first run found a defect:
adaptive IDW did not return the datum at its own location in about 11 % of the members (median
error 1.4 %, maximum 20 % of the data range), because the guard against division by zero capped the
datum's weight below that of close neighbours under large fitted exponents. It was fixed by giving
a query on a datum that datum's value, as IDW does.

Catalogue status
================

Every scenario of the specification, with what it needs before it can be implemented. Target
values taken from examples of the theory are derived independently (and, where possible, computed
by the evaluator) before a scenario is implemented, because the theory is still a draft.

- **implemented** — in the catalogue and run on every change;
- **ready** — needs only new evaluators or generators;
- **negative control** — its target is the theory's Mondrian process, which Spatialize does not
  implement (:doc:`encoders`): on Spatialize it can only be a test expected to reject;
- **blocked / partly ready** — needs a feature not yet in Spatialize or an external baseline.

.. list-table::
   :header-rows: 1
   :widths: 7 45 18 30

   * - id
     - claim
     - status
     - depends on
   * - 
     - **T1 — encoder law**
     - 
     - 
   * - E1
     - exact law of the partition of four points on a line
     - negative control
     - targets on the theory's Mondrian; derive the exact law independently
   * - E2
     - pair co-occurrence of the Mondrian process
     - **implemented**
     - —
   * - E3
     - co-occurrence of three points
     - negative control
     - as E2; adds little until a profile closer to the theory exists
   * - E4
     - Poisson–Voronoi co-occurrence (isotropy in 2D, decay in 1D)
     - needs a reading method
     - a single datum gives a single Voronoi nucleus, so E2's reading does not apply
   * - E5
     - fourth joint cumulant of a block-mark field on a line
     - negative control
     - targets on the theory's Mondrian; derive the value independently
   * - E6
     - conditional covariance under one uniform cut
     - negative control
     - targets on the theory's Mondrian; derive the value independently
   * - 
     - **T2 — estimator properties**
     - 
     - 
   * - P1
     - spread across ensembles decreases as :math:`T^{-1/2}`
     - ready
     - —
   * - P2
     - weighted-draw decoder: draws are data values; their mean is the IDW estimate
     - blocked
     - draw decoders (phase 2)
   * - P3
     - draw decoder: frequencies match the cell's values
     - blocked
     - draw decoders (phase 2)
   * - P4
     - estimated CDFs are monotone and within [0, 1]
     - ready
     - —
   * - P5
     - weights on the data plus residual weight sum to 1
     - blocked
     - empty-cell policy and its diagnostic (phase 3)
   * - P6
     - the law at a location does not depend on the other queries
     - ready (expected to reject)
     - —
   * - P7
     - covariance of an uncorrelated field against its closed form
     - ready
     - derive the closed form independently
   * - P8
     - empty cells share one mark
     - blocked
     - empty-cell policy (phase 3)
   * - P9
     - cell-weighted marks are unbiased under preferential sampling
     - blocked
     - empty-cell policy (phase 3)
   * - 
     - **T3 — geostatistical scenarios**
     - 
     - 
   * - S01
     - simulation keeps the geometry of the field
     - ready
     - —
   * - S02
     - error stops decreasing beyond some ensemble size
     - ready
     - —
   * - S03
     - anisotropic field (visual criterion V1)
     - **implemented**
     - new decoders are added as they land
   * - S04
     - zero-inflated field
     - partly ready
     - draw decoders (phase 2) for its main claims
   * - S05
     - heavy-tailed field
     - partly ready
     - ordinary kriging baseline (external dependency)
   * - S06
     - non-stationary field, order relations of the estimated law
     - partly ready
     - indicator kriging baseline (external dependency)
   * - S07
     - exceedance areas
     - ready
     - derive the area-variance identity independently
   * - S08
     - support effect on tonnage curves
     - ready
     - —
   * - S09
     - resource categories under preferential sampling
     - ready
     - preferential design generator
   * - S10
     - optimal quantile levels
     - ready
     - —
   * - S11
     - granularity and covariance
     - ready
     - —
   * - S12
     - edge cases: few data, duplicates, queries on data and outside the data box
     - **implemented**
     - —
   * - S13
     - connectivity of high-value bodies
     - ready
     - connectivity functional
   * - 
     - **V — visual criteria**
     - 
     - 
   * - V1
     - anisotropy visible (S03)
     - **implemented**
     - —
   * - V2
     - anisotropy against a fitted isotropic kriging (S03)
     - blocked
     - ordinary kriging baseline (external dependency)
   * - V3
     - anisotropy in members and simulations (S03)
     - ready
     - —
   * - V4
     - Mondrian blocks averaged away (S03); S11
     - **implemented** on S03
     - S11 for the coarse-partition case
   * - V5
     - contrast: maps not more washed out than the best linear predictor (S03); the "false cure"
       of coarse partitions (S11)
     - **implemented** on S03
     - S11 for the "false cure"
   * - V6
     - sharp dry-region boundaries (S04)
     - blocked
     - draw decoders (phase 2)
   * - V7
     - halos around extreme values (S05)
     - partly ready
     - ordinary kriging baseline for the halo comparison
   * - V8
     - exceedance regions (S07)
     - ready
     - —
   * - V9
     - roughness against a global simple-kriging reference
     - ready
     - reference computed in numpy
   * - V10
     - connectivity of high-value bodies (S13)
     - ready
     - connectivity functional
   * - V11
     - shape of estimated covariance level curves
     - ready
     - —

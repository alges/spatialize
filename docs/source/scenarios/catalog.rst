.. _scenarios-catalog:

#########
Catalogue
#########

Each scenario lives in its own directory under ``spatialize/scenarios/catalog/``, with a
``scenario.yaml`` descriptor (the setup, the estimators and the pre-registered checks) and, for
pinned data, ``data/*.npy`` files with a ``CHECKSUMS.sha256`` manifest. The descriptor is the
normative definition of the scenario; this page explains the two scenarios that are implemented
and lists those that are planned.

Example run (``ci`` mode, seed 12345, about a minute and a half on a multi-core machine):

.. code-block:: text

   runner=spatialize mode=ci seed=12345 α_suite=0.001 (Holm)
   scenario/check                                    family          p-value     level  expect result
   E2-mondrian-pair-cooccurrence/c1                  gof-closed     2.43e-87   6.7e-05  reject PASS  — k=8 N=3100 ...
   S03-anisotropic-field/v1a                         equivalence    1.51e-12   0.00017  pass   PASS  — mean=-0.2454 se=0.629 ...
   S03-anisotropic-field/v1b                         one-sided      3.35e-08   0.00025  pass   PASS  — mean=0.6183 se=0.0139 ...
   S03-anisotropic-field/v1a-aidw                    equivalence     1.6e-14   8.3e-05  pass   PASS  — mean=-1.243 se=0.438 ...
   S03-anisotropic-field/v1b-aidw                    one-sided      2.31e-25   7.1e-05  pass   PASS  — mean=0.9813 se=0.00634 ...
   S03-anisotropic-field/v1a-krig                    equivalence    1.95e-14   9.1e-05  pass   PASS  — mean=-0.4098 se=0.485 ...
   S03-anisotropic-field/v1b-krig                    one-sided      2.36e-14   0.00011  pass   PASS  — mean=0.7514 se=0.0128 ...
   S03-anisotropic-field/v1a-vor-uniform             equivalence    5.32e-05    0.0005  pass   PASS  — mean=-0.3353 se=1.98 ...
   S03-anisotropic-field/v1b-vor-uniform             one-sided      1.07e-08    0.0002  pass   PASS  — mean=0.8409 se=0.0372 ...
   S03-anisotropic-field/v1a-vor-data                equivalence    8.49e-05     0.001  pass   PASS  — mean=-0.1086 se=2.12 ...
   S03-anisotropic-field/v1b-vor-data                one-sided      4.93e-08   0.00033  pass   PASS  — mean=0.8004 se=0.0362 ...
   S03-anisotropic-field/v1a-vor-uniform-krig [run]  equivalence    3.28e-14   0.00013  pass   PASS  — mean=-0.3517 se=0.502 ...
   S03-anisotropic-field/v1b-vor-uniform-krig [run]  one-sided       3.4e-14   0.00014  pass   PASS  — mean=0.749 se=0.013 ...
   S03-anisotropic-field/v1a-vor-uniform-aidw [run]  equivalence    2.01e-14    0.0001  pass   PASS  — mean=-0.8784 se=0.462 ...
   S03-anisotropic-field/v1b-vor-uniform-aidw [run]  one-sided      2.39e-21   7.7e-05  pass   PASS  — mean=0.9301 se=0.00924 ...

   15 passed, 0 failed, 0 skipped (reproduce with --mode ci --seed 12345)

The ``level`` column is the Holm level each p-value was compared with (:doc:`statistics`). ``[run]``
marks checks whose estimator is not in the public API yet and was reached through the compiled
engine's generic entry point (:doc:`extending`).

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
drawn **jointly** by Cholesky factorisation so samples and truth are one realisation. Twenty pinned
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
computed on each of the :math:`K = 20` fields:

- **v1a — direction** (equivalence, TOST): the orientation error :math:`\theta(\text{map}) - 30°`
  lies within :math:`\pm 10°`.
- **v1b — strength** (one-sided :math:`t`): the coherence of the map is more than half that of the
  truth, :math:`c(\text{map}) / c(\text{truth}) > 0.5`.

Results (``ci``, seed 12345; mean :math:`\pm` standard error over the 20 fields):

.. list-table::
   :header-rows: 1
   :widths: 22 26 13 26 13

   * - estimator
     - orientation error (v1a)
     - p
     - coherence ratio (v1b)
     - p
   * - ``idw``
     - :math:`-0.2° \pm 0.6°`
     - :math:`1.5 \cdot 10^{-12}`
     - :math:`0.62 \pm 0.014`
     - :math:`3.4 \cdot 10^{-8}`
   * - ``aidw``
     - :math:`-1.2° \pm 0.4°`
     - :math:`1.6 \cdot 10^{-14}`
     - :math:`0.98 \pm 0.006`
     - :math:`2.3 \cdot 10^{-25}`
   * - ``krig``
     - :math:`-0.4° \pm 0.5°`
     - :math:`2.0 \cdot 10^{-14}`
     - :math:`0.75 \pm 0.013`
     - :math:`2.4 \cdot 10^{-14}`
   * - ``vor-uniform-idw``
     - :math:`-0.3° \pm 2.0°`
     - :math:`5.3 \cdot 10^{-5}`
     - :math:`0.84 \pm 0.037`
     - :math:`1.1 \cdot 10^{-8}`
   * - ``vor-data-idw``
     - :math:`-0.1° \pm 2.1°`
     - :math:`8.5 \cdot 10^{-5}`
     - :math:`0.80 \pm 0.036`
     - :math:`4.9 \cdot 10^{-8}`
   * - ``vor-uniform-krig``
     - :math:`-0.4° \pm 0.5°`
     - :math:`3.3 \cdot 10^{-14}`
     - :math:`0.75 \pm 0.013`
     - :math:`3.4 \cdot 10^{-14}`
   * - ``vor-uniform-aidw``
     - :math:`-0.9° \pm 0.5°`
     - :math:`2.0 \cdot 10^{-14}`
     - :math:`0.93 \pm 0.009`
     - :math:`2.4 \cdot 10^{-21}`

All fourteen checks pass, stably across seeds. With kriging, the partition barely matters for the
median map (Mondrian and Voronoi median maps correlate at 0.999 on a field, although their members
differ): with about eleven data per cell and a fixed variogram, kriging inside a cell is already
close to global kriging.

.. figure:: /_static/scenarios/S03_compare_13.png
   :width: 100%
   :alt: Truth and the seven median maps of S03 on field 13

   S03, field 13 (``ci``, seed 12345): the truth with its samples and the median map of each
   estimator, on one colour scale. White: the declared 30°; red: the measured orientation.

.. note::

   **History.** The first version of S03 had only ``idw`` and :math:`K = 6` fields; v1b then passed
   with :math:`p \approx 4 \cdot 10^{-4}`, just below a Holm level of :math:`10^{-3}`. Version 2
   added the other four estimators with the same thresholds, fixed before their first run, and
   raised :math:`K` — first to 12, then to 20 when the Voronoi orientation checks (standard error
   about 2°) passed close to their levels. More fields buy power; thresholds were never relaxed. Version 3 added kriging and adaptive IDW on
   the uniform Voronoi profile, which the encoder/decoder refactor made available, again with the
   same thresholds fixed before their first run.

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

Catalogue status
================

Every scenario of the specification, with what it needs before it can be implemented. Target
values taken from examples of the theory are derived independently (and, where possible, computed
by the evaluator) before a scenario is implemented, because the theory is still a draft.

- **implemented** — in the catalogue and run on every change;
- **ready** — needs only new evaluators or generators; *(next)* marks the following ones;
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
     - ready (next)
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
     - axis artefacts averaged away (S03, S11)
     - ready (next)
     - —
   * - V5
     - contrast and the "false cure" of coarse partitions (S03, S11)
     - ready (next)
     - —
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

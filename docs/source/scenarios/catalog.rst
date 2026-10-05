.. _scenarios-catalog:

#########
Catalogue
#########

Each scenario lives in its own directory under ``spatialize/scenarios/catalog/``, with a
``scenario.yaml`` descriptor (the setup, the estimators and the pre-registered checks) and, for
pinned data, ``data/*.npy`` files with a ``CHECKSUMS.sha256`` manifest. The descriptor is the
normative definition of the scenario; this page explains the two scenarios that are implemented
and lists those that are planned.

Example run (``ci`` mode, seed 12345):

.. code-block:: text

   runner=spatialize mode=ci seed=12345 α_suite=0.001 (Holm)
   scenario/check                     family         p-value   level  expect result
   E2-mondrian-pair-cooccurrence/c1   gof-closed    2.43e-87 0.00033  reject PASS  — k=8 N=3100 max|z|=9.61 at #6 (p̂=0.4629 vs 0.5488)
   S03-anisotropic-field/v1a          equivalence   4.36e-05  0.0005  pass   PASS  — mean=-0.1152 se=0.86 margin=(-10, 10) K=6
   S03-anisotropic-field/v1b          one-sided     0.000365   0.001  pass   PASS  — mean=0.6022 se=0.0139 bound=0.5 K=6

The ``level`` column is the Holm level each p-value was compared with (:doc:`statistics`).

.. _scenario-E2:

E2 — Mondrian pair co-occurrence (T1)
=====================================

**Source.** Def 2.3.1, eq. (5.2.2) and Fig 2.4g of the theory.

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

**Source.** §12.3.1, Fig 12.1, eqs. (12.3.8)–(12.3.9) of the theory.

**Truth.** A stationary Gaussian field on the unit square with exponential covariance
:math:`C(h) = \exp(-\sqrt{h^\top A h})`, :math:`A = R_\vartheta^\top \operatorname{diag}(a_1^{-2},
a_2^{-2}) R_\vartheta`, ranges :math:`a_1 = 0.45`, :math:`a_2 = 0.09` and orientation
:math:`\vartheta = 30°` — a 5:1 anisotropy, the variogram's own ground.

**Data.** 400 uniformly placed samples and a 40×40 grid of queries (cell centres, row-major),
drawn **jointly** by Cholesky factorisation so samples and truth are one realisation. Six pinned
replicate fields (generator seed 20261005), stored as ``.npy`` with SHA-256 checksums.

**Estimator.** Mondrian ESI with an IDW decoder (exponent 2), rate :math:`\lambda = 5`,
:math:`T = 100` members in ``ci`` and 300 in ``full``. The estimator is told nothing about the
anisotropy: cuts are axis-aligned and the decoder is isotropic. The point map is the median of the
members at each location.

**Claim (V1).** The elongation at 30° must nevertheless be *clearly visible* in the point map. Two
pre-registered map functionals (:doc:`visual`), computed on each of the :math:`K = 6` fields:

- **v1a — direction** (equivalence, TOST): the orientation error :math:`\theta(\text{map}) - 30°`
  lies within :math:`\pm 10°`. Observed: mean :math:`-0.1° \pm 0.9°` (s.e.), :math:`p = 4.4 \cdot
  10^{-5}`.
- **v1b — strength** (one-sided :math:`t`): the coherence of the map is more than half that of the
  truth, :math:`c(\text{map}) / c(\text{truth}) > 0.5`. Observed: mean ratio
  :math:`0.60 \pm 0.014`, :math:`p = 3.7 \cdot 10^{-4}`.

Both pass, stably across seeds and in ``full`` mode.

.. note::

   **v1b is close to its level.** With :math:`K = 6` fields its p-value (about
   :math:`4 \cdot 10^{-4}`) sits just below the level Holm currently allots it
   (:math:`10^{-3}`, because it is the largest of only three p-values). As the catalogue grows, the
   smallest Holm levels fall as :math:`\alpha/m`, and a p-value of this size may no longer clear its
   level. The remedy is to increase the number of replicate fields of S03 (the margin is
   consistent, so a few more fields lower the p-value quickly), not to relax the criterion.

Planned scenarios
=================

The following are specified and will be added to the catalogue. Section numbers refer to the
theory.

**T1 — encoder law** (closed forms; negative controls on ``spatialize-v1``)

.. list-table::
   :header-rows: 1
   :widths: 8 18 74

   * - id
     - source
     - target
   * - E1
     - Fig 2.1
     - exact law of the partition of four points on a line under Poisson cuts (8 interval
       groupings with known probabilities; 7 non-interval groupings impossible)
   * - E3
     - (3.6.17)
     - co-occurrence of three points :math:`e(S) = \exp(-\lambda \sum_c \mathrm{range}_c(S))`
   * - E4
     - Fig 2.4h, (8.4.4)
     - Poisson–Voronoi co-occurrence: isotropic in 2D; known decay in 1D
   * - E5
     - Fig 5.2a, Cor 5.4.4
     - fourth joint cumulant of four points on a line, maximum 27/128 at :math:`s = \log(4/3)`
   * - E6
     - Fig 5.2b, Prop 5.5.1
     - conditional covariance under one uniform cut, :math:`-0.21\,\beta^2`

**T2 — estimator properties**

.. list-table::
   :header-rows: 1
   :widths: 8 18 74

   * - id
     - source
     - check
   * - P1
     - Thm 4.2.6
     - spread across independent ensembles decreases as :math:`T^{-1/2}`
   * - P2
     - Prop 12.3.6
     - weighted-draw members are data values; their mean equals the IDW estimate
   * - P3
     - Prop 12.3.7
     - draw frequencies match the share of each value in the cell
   * - P4
     - Thm 1.5.1
     - estimated CDFs are monotone and within :math:`[0, 1]` (almost sure)
   * - P5
     - Thm 12.1.1
     - weights on the data plus residual weight sum to 1; residual weight grows with distance to data
   * - P6
     - locality
     - the law at a location does not change when the other queries change (expected to reject on
       ``spatialize-v1``, whose partition box depends on the queries)
   * - P7
     - §11.3
     - estimated covariance of an uncorrelated field matches its closed form
   * - P8
     - Thm 12.2.1
     - with ``empty_cells="mark"``, two queries far from the data share one mark per empty cell
   * - P9
     - Def 2.5.1
     - under preferential sampling, cell-weighted marks are unbiased; data-weighted marks are not

**T3 — geostatistical scenarios** (§12.3 and companions): S01 simulation keeps geometry; S02
ensemble size floor; S04 zero-inflated field; S05 heavy tail; S06 non-stationary field and order
relations; S07 exceedance areas; S08 support effect on tonnage; S09 resource categories; S10
optimal quantile levels; S11 granularity and covariance; S12 edge cases; S13 connectivity.

**Visual criteria V2–V11**: anisotropy against a fitted isotropic kriging (V2) and in members and
simulations (V3); axis artefacts averaged away (V4) and the "false cure" of coarse partitions (V5);
sharp dry-region boundaries (V6); halos around extreme values (V7); exceedance regions (V8);
roughness (V9); connectivity of high-value bodies (V10); shape of estimated covariance level
curves (V11).

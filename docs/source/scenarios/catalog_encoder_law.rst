.. _scenarios-catalog-t1:

##################
T1 — Encoder law
##################

Scenarios of tier T1 test the law of the partition process itself, through the probability that
locations share a cell, the law of the partition of a few points and the joint moments of block-mark
fields built on it. Their targets are closed forms of the theory's processes. Spatialize's Mondrian
process differs from the theory's (:doc:`encoders`), so on Spatialize the Mondrian scenarios can only
be *negative controls*, tests expected to reject, which show that the test can see a deviation of
that size. For any implementation of the theory's processes they become ordinary checks.

Every target value below follows from the definitions, independently of the examples in the theory,
which is still a draft.

.. _scenario-E1:

E1 — Partition law of four points on a line
===========================================

:Status: negative control (not yet implemented)
:Source in the theory: worked example of the partition induced on a few points by Poisson cuts
:Claim: the probabilities of all the ways four points on a line can be grouped into cells.

**Setup.** One dimension, with domain :math:`[0, 2]` and points :math:`x = (0.2, 0.5, 0.8, 1.3)`. The
cuts form a Poisson process of rate :math:`\lambda = 2` (the one-dimensional Mondrian process).

**Target.** Points can only be grouped into intervals of consecutive points, since each of the three
gaps (0.3, 0.3, 0.5) is cut independently with probability :math:`1 - e^{-\lambda g}`. Writing
:math:`a = b = e^{-0.6}` and :math:`c = e^{-1}` for the probabilities of no cut in each gap, the eight
interval groupings have the following probabilities.

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - grouping
     - probability
     - value
   * - {1,2,3}{4}
     - :math:`a\,b\,(1-c)`
     - 0.1904
   * - {1,2}{3}{4}
     - :math:`a\,(1-b)(1-c)`
     - 0.1565
   * - {1}{2,3}{4}
     - :math:`(1-a)\,b\,(1-c)`
     - 0.1565
   * - {1}{2}{3}{4}
     - :math:`(1-a)(1-b)(1-c)`
     - 0.1287
   * - {1,2,3,4}
     - :math:`a\,b\,c`
     - 0.1108
   * - {1,2}{3,4}
     - :math:`a\,(1-b)\,c`
     - 0.0911
   * - {1}{2,3,4}
     - :math:`(1-a)\,b\,c`
     - 0.0911
   * - {1}{2}{3,4}
     - :math:`(1-a)(1-b)\,c`
     - 0.0749

The seven other set partitions of four points (e.g. {1,3}{2,4}) are impossible.

**Checks.**

- *gof-closed* (G-test) of the observed frequencies of the eight interval groupings over :math:`N`
  partition draws, sized for power 0.9 at the declared minimum detectable effect.
- *almost-sure*. No non-interval grouping is ever observed.

**Reading.** The partition of the points is read through the estimator as in :ref:`E2
<scenario-E2>`, placing one datum at a time with empty cells as NaN and recording which queries share
its cell.

**Depends on.** An ordinary pass needs an implementation of the theory's Mondrian process. On
Spatialize, whose root cell is always split, the scenario is a negative control.

.. _scenario-E2:

E2 — Mondrian pair co-occurrence
================================

:Status: **implemented**
:Evaluator: ``pair_cooccurrence``

**Source.** The theory's definition of the Mondrian process, with the closed form of its pair
co-occurrence (a classical property of the Mondrian process, Roy & Teh, 2009).

**Claim.** For a Mondrian process of rate :math:`\lambda`, the probability that two locations
:math:`x, y` fall in the same cell is

.. math::

   e(\{x, y\}) = \exp\!\left(-\lambda\, \lVert x - y \rVert_1\right).

**Reading through the estimator.** The suite sees ensembles, never an implementation's partitions.
It therefore places a single datum (value 1) at the centre of the unit square under the ``"nan"``
empty-cell policy, so a member is finite exactly when the query falls in the datum's cell. The
fraction of finite members at a query at displacement :math:`h` then estimates
:math:`e(\{x_0, x_0 + h\})`. Displacements of 0.05, 0.1, 0.2 and 0.3 along an axis and along the
diagonal (with :math:`\ell_1`-normalised steps) give :math:`k = 8` proportions, at rate
:math:`\lambda = 3`.

**Test.** A *gof-closed* test, with :math:`\sum_{i=1}^{8} z_i^2 \sim \chi^2_8` under the closed
form. It uses :math:`N = 3\,100` members in ``ci`` (:math:`\delta = 0.05`) and :math:`19\,300` in
``full`` (:math:`\delta = 0.02`), which give power 0.9 at :math:`\delta` for up to 50 tests under Holm.

**Expectation per profile.** An implementation of the theory's Mondrian process should pass, while
Spatialize's Mondrian partition should reject, making the check a negative control. Spatialize's Mondrian
deviates from the theory's process by design (:doc:`encoders`), by 4–21 % on the unit square. The
test sees the deviation with :math:`p \approx 10^{-87}` in ``ci``, its largest :math:`|z| = 9.6`
falling at displacement 0.2 along the diagonal (:math:`\hat p = 0.463` against :math:`0.549`), and
with :math:`|z| \approx 27` in ``full``. A test unable to reject here would lack the power its passes
elsewhere claim.

.. _scenario-E3:

E3 — Co-occurrence of three points
==================================

:Status: negative control (not yet implemented)
:Source in the theory: co-occurrence of a set of locations under the Mondrian process
:Claim: the probability that a set :math:`S` of locations lies in one cell is
  :math:`e(S) = \exp(-\lambda \sum_c \mathrm{range}_c(S))`, the sum running over the coordinates.

**Setup.** Two dimensions, with Mondrian rate :math:`\lambda` on the unit square and triples of
locations around the centre in several shapes (collinear along an axis, along the diagonal,
triangles).

**Target.** The set :math:`S` stays in one cell when no cut falls inside the box it spans. Cuts on
coordinate :math:`c` fall in that box at rate :math:`\lambda\, \mathrm{range}_c(S)`, hence the closed
form, which reduces to E2 for two points.

**Check.**

- *gof-closed* per triple shape, read as in E2 with one datum and queries at the other two points, a
  member being finite at both exactly when the three share a cell.

On Spatialize it adds little to E2, its value lying in testing implementations of the theory's
process.

.. _scenario-E4:

E4 — Poisson–Voronoi co-occurrence
==================================

:Status: needs a reading method (not yet implemented)
:Source in the theory: co-occurrence function of the Poisson–Voronoi partition
:Claim: in 2D the co-occurrence of two locations depends only on their distance (isotropy). In 1D
  it decays with distance faster than for the Mondrian process of the same mean cell length.

**Setup.** A Poisson–Voronoi partition of intensity 3 in 2D, with pairs at equal distances along the
axes and the diagonals, and of intensity 1 in 1D.

**Checks.**

- *two-sample* (2D). The co-occurrence along the axes equals that along the diagonals at the same
  distance.
- *gof-closed* (1D) against a high-precision reference that the evaluator computes by simulating the
  theory's process.

**Reading.** E2's single-datum reading does not apply to Spatialize's Voronoi partitions, since one
datum gives at most one nucleus, whose cell then holds every query. A reading with many data, which
asks whether two given locations fall in the cell of the same nucleus, has to be designed first.

.. _scenario-E5:

E5 — Fourth joint cumulant of a block-mark field
================================================

:Status: negative control (not yet implemented)
:Source in the theory: higher-order cumulants of block-mark fields
:Claim: for four equally spaced points on a line, the fourth joint cumulant of a block-mark field
  with Gaussian marks has a closed form in the spacing.

**Setup.** Four points on a line with gaps :math:`s`, cut by a Mondrian process (Poisson cuts) of
rate 1, with one mark per cell drawn iid from :math:`\mathcal N(0, 1)`.

**Target.** Given the partition, the values are Gaussian with covariance 1 within a cell and 0
across cells. Writing :math:`q = e^{-s}` for the probability of no cut in a gap,

.. math::

   \kappa_4 = \underbrace{q^2 + q^3 + q^3}_{E[Z_1Z_2Z_3Z_4]}
   - \underbrace{(q\cdot q + q^2\cdot q^2 + q^3\cdot q)}_{\text{pairings of covariances}}
   = 2q^3(1-q) = 2e^{-3s}(1-e^{-s}),

which is largest, :math:`27/128`, at :math:`s = \log(4/3)`.

**Check.**

- *identity* at several spacings, with a bootstrap standard error of the empirical cumulant over the
  ensemble members.

.. _scenario-E6:

E6 — Conditional covariance under one random cut
================================================

:Status: negative control (not yet implemented)
:Source in the theory: conditioning on an observed value under a random partition
:Claim: observing a value at one location induces a negative covariance between two others that it
  can never share a cell with together.

**Setup.** The interval :math:`(0, 1)` is cut once at a uniform position, with marks iid
:math:`\mathcal N(0,1)` and locations :math:`x = 0`, :math:`b = 0.3`, :math:`y = 1`, so that :math:`x`
and :math:`y` always fall in different cells.

**Target.** :math:`b` shares the left cell with probability 0.7 whatever its value
:math:`\beta`, so :math:`E[Z_x \mid Z_b=\beta] = 0.7\beta`, :math:`E[Z_y \mid Z_b=\beta] =
0.3\beta` and :math:`E[Z_xZ_y \mid Z_b=\beta] = 0`, giving

.. math::

   \mathrm{Cov}(Z_x, Z_y \mid Z_b = \beta) = -0.21\,\beta^2 .

**Check.**

- *identity* per bin of :math:`\beta`.

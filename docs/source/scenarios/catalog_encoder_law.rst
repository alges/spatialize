.. _scenarios-catalog-t1:

##################
T1 — Encoder law
##################

Scenarios of tier T1 test the law of the partition process itself, through the probability that
locations share a cell, the law of the partition of a few points and the joint moments of block-mark
fields built on it. Their targets are closed forms of the theory's processes. Spatialize's default
Mondrian partition, ``"mondrian"``, implements the theory's process since version 1.3, so on it
they are ordinary checks. The Mondrian partition of version 1.2, ``"mondrian-legacy"``, differs from
the theory's process (:doc:`encoders`), so on it the Mondrian scenarios are *negative controls*,
tests expected to reject, which show that the test can see a deviation of that size.

Every target value below follows from the definitions, independently of the examples in the theory,
which is still a draft.

.. _scenario-E1:

E1 — Partition law of four points on a line
===========================================

:Status: **implemented**
:Evaluator: ``partition_law``
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

- *gof-closed*. The Pearson statistic of the observed frequencies of the eight interval groupings
  against :math:`N` times their probabilities, referred to :math:`\chi^2_7`, with :math:`N = 5\,000`
  partitions in ``ci`` and :math:`20\,000` in ``full``.
- *almost-sure*. No non-interval grouping is ever observed.

**Reading.** The suite sees ensembles, never partitions. With the four points as data, the
``cellmean`` decoder run on the indicator of point :math:`a` returns, at each of the four points,
:math:`1/n_C` when the point shares :math:`a`'s cell :math:`C` and 0 otherwise. Four runs with one seed
see the same partitions, so they give, partition by partition, which points share a cell.

**Results.** On seeds 1 to 3 the theory's process passes (p between 0.2 and 0.97) while the Mondrian
partition of version 1.2 rejects with :math:`\chi^2` near 500 on 7 degrees of freedom. Both only ever group the
points into intervals.

**Profiles.** The scenario is an ordinary check on ``"mondrian"``, the theory's process. On
the Mondrian partition of Spatialize 1.2 (``"mondrian-legacy"``), whose root cell is always split, it is a negative control.

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

**Estimators.** The check reads two estimators with the IDW decoder, ``idw-theory`` on the
theory's Mondrian process (Spatialize's default ``"mondrian"``) and ``idw`` on the Mondrian partition
of Spatialize 1.2 (``"mondrian-legacy"``).

**Expectation per profile.** The theory's process should pass, while the partition of Spatialize 1.2
should reject, making that outcome a negative control. That partition deviates
from the theory's process by design (:doc:`encoders`), by 4–21 % on the unit square. The
test sees the deviation with :math:`p \approx 10^{-87}` in ``ci``, its largest :math:`|z| = 9.6`
falling at displacement 0.2 along the diagonal (:math:`\hat p = 0.463` against :math:`0.549`), and
with :math:`|z| \approx 27` in ``full``. A test unable to reject here would lack the power its passes
elsewhere claim.

.. _scenario-E3:

E3 — Co-occurrence of three points
==================================

:Status: **implemented**
:Evaluator: ``partition_law``
:Source in the theory: co-occurrence of a set of locations under the Mondrian process
:Claim: the probability that a set :math:`S` of locations lies in one cell is
  :math:`e(S) = \exp(-\lambda \sum_c \mathrm{range}_c(S))`, the sum running over the coordinates.

**Setup.** Two dimensions, with Mondrian rate :math:`\lambda = 3` on the unit square and five
triples around the centre, collinear along an axis, collinear along the diagonal, two triangles and a
small right angle.

**Target.** The set :math:`S` stays in one cell when no cut falls inside the box it spans. Cuts on
coordinate :math:`c` fall in that box at rate :math:`\lambda\, \mathrm{range}_c(S)`, hence the closed
form, which reduces to E2 for two points.

**Check.**

- *gof-closed*. The share of partitions putting each triple in one cell against :math:`e(S)`,
  combined as :math:`\sum_i z_i^2 \sim \chi^2_5`, with :math:`N = 4\,000` partitions in ``ci`` and
  :math:`16\,000` in ``full``. The cells are read as in E1.

**Results.** On seeds 1 to 3 the theory's process passes (p between 0.24 and 0.58) while the Mondrian
partition of version 1.2 rejects with :math:`|z|` up to 11.6, on the collinear triples above all.

Beyond E2, it tests the process on sets of more than two locations, on ``"mondrian"`` as an
ordinary check and on the Mondrian partition of version 1.2 as a negative control.

.. _scenario-E4:

E4 — Poisson–Voronoi co-occurrence
==================================

:Status: **implemented** (E4 on a line, E4b in the plane)
:Evaluator: ``partition_law``
:Source in the theory: co-occurrence function of the Poisson–Voronoi partition
:Claim: in 2D the co-occurrence of two locations depends only on their distance (isotropy). In 1D
  it decays with distance faster than for the Mondrian process of the same mean cell length.

**Setup.** Spatialize's Voronoi partition with nuclei uniform in the box is a Poisson–Voronoi
partition, its number of nuclei being Poisson. On the line :math:`[0, 1]` the intensity is
:math:`\rho = 20`, the origin 0.4 and the distances 0.02, 0.05, 0.1 and 0.2. In the unit square the
intensity is 60, the centre :math:`(0.5, 0.5)` and the distances 0.05, 0.1 and 0.15. Each check
uses :math:`20\,000` partitions in ``ci`` and :math:`80\,000` in ``full``.

**Target.** Two locations share a cell when their nearest nucleus is the same. On a line, at
distance :math:`h`, either no nucleus lies between them and the nearest one lies on the same side
for both, with probability :math:`e^{-2\rho h}`, or exactly one lies between them, nearer to each
than the nuclei outside, with probability :math:`\rho h\, e^{-2\rho h}`. Hence

.. math::

   e_V(h) = (1 + \rho h)\, e^{-2\rho h} < e^{-\rho h},

the right-hand side being the Mondrian process with the same mean cell length :math:`1/\rho`. The
closed form was checked against a direct simulation of a Poisson–Voronoi line, 200 000 draws per
distance, within two standard errors. The origin lies at least 0.4 from either end, which keeps the
edge effects below :math:`e^{-8}`.

**Checks.**

- *gof-closed* (E4, ``line``). The share of partitions putting the origin and each other location
  in one cell against :math:`e_V(h)`, each distance with its own seed, combined as
  :math:`\sum z^2 \sim \chi^2_4`.
- *identity* (E4b, ``isotropy``). At each distance the share along the first axis equals the share
  along the diagonal, paired by partition.
- *negative controls*. The Mondrian process of rate :math:`\rho` must reject :math:`e_V`, and the
  Mondrian process in the plane, measuring distance in :math:`\ell_1`, must reject the isotropy.

**Reading.** E2's single-datum reading does not apply to Voronoi partitions, since one datum gives
at most one nucleus. The runner's method ``cells`` gives the cells instead, with filler data
uniform in the domain (200 on the line, 400 in the plane), so that the partition can reach its
intensity.

**Results.** On seed 1 in ``ci`` (2026-10-09) the line passes, :math:`p = 0.091`, its largest
deviation :math:`|z| = 2.8` at :math:`h = 0.05` (:math:`\hat p = 0.262` against 0.271). The isotropy
passes with :math:`p = 0.61`, largest :math:`|z| = 1.0`. Both Mondrian controls reject, the line
with :math:`|z| = 64` at :math:`h = 0.2` (:math:`\hat p = 0.020` against 0.0017) and the isotropy
with :math:`|z| = 38`.

.. _scenario-E5:

E5 — Fourth joint cumulant of a block-mark field
================================================

:Status: **implemented**
:Evaluator: ``partition_law``
:Source in the theory: higher-order cumulants of block-mark fields
:Claim: for four equally spaced points on a line, the fourth joint cumulant of a block-mark field
  with Gaussian marks has a closed form in the spacing.

**Setup.** Four points on a line with gaps :math:`s`, cut by a Mondrian process (Poisson cuts) of
rate 1, with one mark per cell drawn iid from :math:`\mathcal N(0, 1)`.

**Target.** Given the partition, the values are Gaussian with covariance 1 within a cell and 0
across cells. Writing :math:`q = e^{-s}` for the probability of no cut in a gap,

.. math::

   \begin{aligned}
   \kappa_4 &= \underbrace{q^2 + q^3 + q^3}_{E[Z_1Z_2Z_3Z_4]}
   - \underbrace{(q\cdot q + q^2\cdot q^2 + q^3\cdot q)}_{\text{pairings of covariances}}
   \\
   &= 2q^3(1-q) = 2e^{-3s}(1-e^{-s}),
   \end{aligned}

which is largest, :math:`27/128`, at :math:`s = \log(4/3)`.

**Check.**

- *identity*. At the spacings :math:`s = 0.1, \log(4/3), 0.5, 0.8`, the empirical fourth cumulant
  minus :math:`2q^3(1-q)`, divided by its bootstrap standard error, combined as
  :math:`\sum z^2 \sim \chi^2_4`, with :math:`N = 80\,000` partitions in ``ci`` and
  :math:`320\,000` in ``full``.

**Reading.** The partitions are read as in E1. Under each partition the evaluator paints every block
with an independent standard Gaussian mark, which makes the four values a block-mark field built on
the estimator's partitions.

**Results.** With :math:`N = 20\,000` the control on the Mondrian partition of version 1.2 rejected on two seeds out of
three, its cumulant differing from the closed form by 0.05 to 0.06 at the two larger spacings and by
about 0.016 at the smaller ones. :math:`N` was raised to :math:`80\,000` before the scenario was
released. On seeds 1 to 4 the theory's process then passes (p between 0.45 and 0.95) while the Mondrian
partition of version 1.2 rejects with p at most :math:`10^{-9}`.

.. _scenario-E6:

E6 — Conditional covariance under one random cut
================================================

:Status: **implemented**
:Evaluator: ``partition_law``
:Source in the theory: conditioning on an observed value under a random partition
:Claim: observing a value at one location induces a negative covariance between two others that it
  can never share a cell with together.

**Setup.** Locations :math:`x = 0`, :math:`b = 0.3` and :math:`y = 1` on the interval :math:`[0, 1]`,
with marks iid of mean 0 and variance 1. The theory's example cuts the interval once at a uniform
position. Spatialize offers no such partition, so the scenario reads the same quantity on the
Mondrian process of rate 1 and on the Mondrian partition of version 1.2, with :math:`20\,000`
partitions in ``ci`` and :math:`80\,000` in ``full``.

**Target.** Given the partition, :math:`Z_x` equals :math:`\beta` when :math:`x` shares
:math:`b`'s cell and is an independent mark otherwise, and likewise :math:`Z_y`. Hence

.. math::

   \mathrm{Cov}(Z_x, Z_y \mid Z_b = \beta) = \beta^2 \big[P(x \sim y) - P(x \sim b)\, P(b \sim y)\big]
   = \beta^2 D .

- One uniform cut, the theory's example: :math:`P(x \sim y) = 0`, :math:`P(x \sim b) = 0.7`,
  :math:`P(b \sim y) = 0.3`, so :math:`D = -0.21`.
- The Mondrian process: on a line its cuts are a Poisson process, independent on disjoint
  intervals, so :math:`D = 0`.
- The partition of version 1.2: it always cuts the whole domain once at a uniform position, then as
  the Poisson process, so :math:`D = -(b - x)(y - b)\, e^{-\lambda (y - x)}`, the example's value as
  :math:`\lambda \to 0`.

Both closed forms were checked against a direct simulation, 400 000 draws each (:math:`D` within
:math:`5 \times 10^{-4}`).

**Checks.**

- *identity* (``poisson``). On the Mondrian process, :math:`\hat D = 0`, with a bootstrap standard
  error over the partitions.
- *identity* (``forced-cut``). On the partition of version 1.2, :math:`\hat D` equals its closed
  form.
- *negative controls*. Each implementation must reject the other's closed form.

**Reading.** Which of the three locations share a cell is read through the estimator, as in E1.

**Results.** On seed 1 in ``ci`` (2026-10-09) the Mondrian process gives :math:`\hat D = -0.0013`,
:math:`p = 0.40` against 0, and the partition of version 1.2 gives :math:`\hat D = -0.0782`,
:math:`p = 0.40` against :math:`-0.0773`. Each rejects the other's closed form, with :math:`|z|` of
50 and 69.

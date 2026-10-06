.. _scenarios-statistics:

#####################################
The statistical acceptance framework
#####################################

.. currentmodule:: spatialize.scenarios.stats

This page explains how the suite decides that a check passes, and why each rule is what it is.
The implementation is in :mod:`spatialize.scenarios.stats.families` (test families) and
:mod:`spatialize.scenarios.stats.budget` (error budget and power).

Test families and their pass rules
==================================

Each check belongs to one family. The family fixes the statistic, the null hypothesis and —
crucially — whether passing means *not rejecting* or *rejecting* the null hypothesis.

.. list-table::
   :header-rows: 1
   :widths: 16 30 30 24

   * - Family
     - Use
     - Statistic
     - Pass rule
   * - **gof-closed**
     - an estimated probability or curve against a closed form or exact law
     - :math:`z_i = (\hat p_i - p_i)/\sqrt{p_i(1-p_i)/N}`, combined as :math:`\sum z_i^2 \sim \chi^2_k`
     - do not reject, **and** the declared power holds
   * - **almost-sure**
     - an invariant the theory guarantees (support of a draw decoder, monotone CDF, ...)
     - number of violations
     - zero violations
   * - **identity**
     - an expectation identity (e.g. mean of draws = IDW estimate)
     - :math:`z` or :math:`t` on the difference, with its Monte Carlo standard error
     - do not reject
   * - **one-sided**
     - "the mean of a functional exceeds a bound" across fields
     - one-sample :math:`t`
     - reject :math:`H_0:\ \mu \le b`
   * - **paired-relation**
     - "estimator A beats estimator B on metric M" across :math:`K` independent fields
     - paired :math:`t` on :math:`D = M_A - M_B`; sign or Wilcoxon test when :math:`D` is far from normal
     - reject :math:`H_0`: no advantage
   * - **equivalence**
     - a quantity lies within a margin of a target (TOST)
     - two one-sided :math:`t` tests, :math:`p = \max(p_{\text{low}}, p_{\text{high}})`
     - reject non-equivalence
   * - **two-sample**
     - a law must not change (e.g. the law at a location when other queries change)
     - Kolmogorov–Smirnov / Anderson–Darling, two-proportion :math:`z`
     - do not reject, **and** the declared power holds

**Why the asymmetry matters.** For a check of the form "the law equals X", *not rejecting* is the
pass condition. A test with too few draws never rejects — so, alone, it would pass anything. That
is why those families carry a power requirement (below). When the claim is that something is
*equivalent* to a target, the suite uses TOST (two one-sided tests): there, passing *requires*
evidence. Concretely, for a margin :math:`(L, U)` the suite tests :math:`H_0^{-}:\ \mu \le L` and
:math:`H_0^{+}:\ \mu \ge U` and passes only if both are rejected, i.e. if
:math:`p = \max(p^-, p^+)` is below the allotted level.

What the level controls in each case:

- in *not-reject* families, the level bounds **false failures** (rejecting a correct
  implementation); false passes are bounded by the power requirement;
- in *reject* families, the level bounds **false passes** (declaring an advantage or an
  equivalence that is not there); false failures are bounded by the choice of :math:`K` or :math:`N`.

The error budget: :math:`\alpha_{\text{suite}} = 10^{-3}` with Holm
===================================================================

**1. Many tests multiply false alarms.** A run executes tens to hundreds of tests. At the usual
level 0.05 per test, 100 independent tests would fail spuriously with probability
:math:`1 - 0.95^{100} = 0.994`: almost every run would be red for no reason, and red builds that mean
nothing end up being ignored. The level must therefore be controlled for the **whole run** — the
*family-wise error rate* — not test by test.

**2. Why** :math:`10^{-3}`. The suite runs on every change. With about a thousand runs a year,
:math:`10^{-3}` means roughly one spurious failure a year across all of them: rare enough that a
failure is investigated, not dismissed. Going lower would buy little and cost sample size (point 5).

**3. Why Holm, and not Bonferroni or another procedure.** Holm's step-down procedure sorts the
:math:`m` p-values of the run, :math:`p_{(1)} \le \dots \le p_{(m)}`, and compares

.. math::

   p_{(1)} \le \frac{\alpha}{m},\quad p_{(2)} \le \frac{\alpha}{m-1},\quad \dots,\quad
   p_{(j)} \le \frac{\alpha}{m-j+1},

rejecting until the first comparison fails; no larger p-value is rejected after that. It controls
the family-wise error rate at :math:`\alpha` **under any dependence between the tests** — ours are
dependent, since several checks read the same fields and ensembles — and it is uniformly more
powerful than Bonferroni (which compares every p-value with :math:`\alpha/m`). Procedures that
assume independence (Šidák) or exploit positive dependence (Hochberg) are not guaranteed here.
:func:`budget.holm_levels` implements it; the report shows, for every check, the level it was
compared with.

*Example.* With :math:`m = 3` checks and p-values :math:`2.4\cdot10^{-87}`, :math:`4.4\cdot10^{-5}`
and :math:`3.7\cdot10^{-4}`, the levels are :math:`\alpha/3 = 3.3\cdot10^{-4}`,
:math:`\alpha/2 = 5\cdot10^{-4}` and :math:`\alpha/1 = 10^{-3}`; all three are rejected.

**4. Why a declared power.** Every check states a **minimum detectable effect** :math:`\delta` —
the smallest deviation that matters scientifically (e.g. a co-occurrence probability off by 0.02)
— and its sample size must give **power at least 0.9** at :math:`\delta`. Then "passed" is a
statement with content: *if the implementation were off by* :math:`\delta` *or more, this test would
have caught it with probability at least 0.9*. Negative controls (below) verify that the power is
real.

**5. What it costs.** For a proportion :math:`p` tested at a per-test level :math:`\alpha_i` with
power :math:`1-\beta` at effect :math:`\delta`,

.. math::

   N \;\ge\; \frac{\left(z_{1-\alpha_i/2} + z_{1-\beta}\right)^2\, p(1-p)}{\delta^2}

(:func:`budget.required_n`). With :math:`\alpha_{\text{suite}} = 10^{-3}` split over :math:`m` tests
(worst case :math:`\alpha/m`), :math:`p = 0.5` and power 0.9 (:math:`z_{0.9} = 1.282`):

.. list-table::
   :header-rows: 1

   * - tests :math:`m`
     - :math:`\alpha_i`
     - :math:`z_{1-\alpha_i/2}`
     - :math:`N` at :math:`\delta = 0.05`
     - :math:`N` at :math:`\delta = 0.02`
   * - 20
     - :math:`5.0\cdot10^{-5}`
     - 4.06
     - 2 849
     - 17 804
   * - 50
     - :math:`2.0\cdot10^{-5}`
     - 4.26
     - 3 077
     - 19 227
   * - 100
     - :math:`1.0\cdot10^{-5}`
     - 4.42
     - 3 248
     - 20 298

The cost grows only logarithmically with the number of tests (through :math:`z`), but
quadratically as :math:`\delta` shrinks: precision, not the error budget, is what is expensive.

**6. Enough replicate fields for a test to be able to pass.** A "beats" claim across :math:`K`
fields with a sign test has smallest possible p-value :math:`2^{-K}`. With :math:`K = 6` that is
:math:`1/64 \approx 0.016` — larger than the whole budget, so such a check could **never** pass.
Hence the suite uses a paired :math:`t` test when the differences are roughly normal (it reaches
small p-values with few fields when the effect is consistent), or a sign/Wilcoxon test with
:math:`K \ge \lceil \log_2(m/\alpha) \rceil` fields (:func:`budget.min_sign_test_fields`): 10 for a
single test, 16 for :math:`m = 50`. "Better on all six fields" is good evidence, but not a formal
test at this level.

**7. Two modes.** Same rules, two declared effects:

- ``ci`` (every change): :math:`\delta \approx 0.05`, a few thousand draws per check, fast; it
  catches gross breakage. A pass says "no deviation larger than about 0.05".
- ``full`` (before a release, or to certify another implementation): :math:`\delta \approx 0.02`,
  book-sized ensembles and enough fields, slow. Only a ``full`` pass says "about 0.02".

Almost-sure checks and the budget
=================================

An almost-sure check counts violations of a property that a correct implementation satisfies
with certainty (an output outside the data range for a positive-weight decoder, an infinite
value, ...). It is decided exactly — it passes with zero violations — and a correct implementation
cannot fail it, so its probability of a false failure is zero. It therefore spends **none** of the
error budget: Holm's procedure is applied to the statistical tests only, and the report shows
``exact`` in place of a level. Without this, a scenario with many almost-sure checks (such as
:ref:`S12 <scenario-S12>`) would lower the levels of every statistical test of the run without
adding any risk of a false alarm.

Known failures
==============

A pre-registered check that fails on Spatialize documents a finding; the remedy is to fix the code
or to accept the finding, never to relax the threshold. While a finding stands, the check carries
``known_failure: <reason>`` in its scenario file. It is still run, inside the Holm budget like any
other test, and reported as ``KNOWN`` without failing the run. If it starts passing it is reported
as ``XPASS`` and the run fails, so that the record is updated in the same change that fixed the
behaviour.

Negative controls
=================

Every family must be shown able to fail. Each tier includes checks that are **expected to
reject**, marked ``expect: reject`` in the scenario file for a given encoder profile. A negative
control passes when its test rejects; if it does not, the test is under-powered and its "passes"
elsewhere are not to be trusted. The first control of the catalogue is
:ref:`E2 <scenario-E2>`: the closed-form co-occurrence of the book's Mondrian process tested on
Spatialize's current Mondrian, which deviates from it by design (:doc:`encoders`); the test rejects
with :math:`p \approx 10^{-87}` at :math:`N = 3\,100`.

A failed power check is a finding about the test, not about the estimator. The first functional
tried for criterion V4 failed its check that eight members show more axis artefacts than a hundred,
and turned out to measure smoothing; it was replaced by a functional whose power check and
calibration both pass (:ref:`S03 <scenario-S03>`).

Seeds, independence and data
============================

- Ensemble members are independent given the data (one partition draw each); replicate fields are
  independent draws of the truth; independent runs use independent seeds.
- Seeds are **fresh per run** and printed in the report; set ``SPATIALIZE_SCENARIO_SEED`` to
  reproduce a run. No criterion depends on a particular seed.
- Data are ``pinned`` (stored ``.npy`` fields with SHA-256 checksums, verified when the catalogue is
  loaded) so every implementation is tested on the same fields. A ``fresh`` mode, in which each
  implementation regenerates fields from the normative description with its own random numbers,
  is planned.

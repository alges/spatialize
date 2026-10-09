.. _scenarios-statistics:

#####################################
The statistical acceptance framework
#####################################

.. currentmodule:: spatialize.scenarios.stats

A statistical rule decides every check of the suite. The test families are implemented in
:mod:`spatialize.scenarios.stats.families`, the error budget and the power calculations in
:mod:`spatialize.scenarios.stats.budget`.

Pass rules of the test families
===============================

Each check belongs to one family. The family fixes the statistic, the null hypothesis and whether
passing means *not rejecting* or *rejecting* the null hypothesis.

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
     - do not reject, with the declared power
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
     - reject the hypothesis of no advantage
   * - **equivalence**
     - a quantity lies within a margin of a target (TOST)
     - two one-sided :math:`t` tests, :math:`p = \max(p_{\text{low}}, p_{\text{high}})`
     - reject non-equivalence
   * - **two-sample**
     - a law must not change (e.g. the law at a location when other queries change)
     - Kolmogorov–Smirnov or Anderson–Darling, two-proportion :math:`z`
     - do not reject, with the declared power

**Why the asymmetry matters.** For a check of the form "the law equals X", not rejecting counts as a
pass. A test with too few draws never rejects, so on its own it would pass anything. Those families
therefore carry a power requirement (below). A claim of *equivalence* to a target is decided by
TOST (two one-sided tests), and there passing requires evidence. For a margin :math:`(L, U)` the
suite tests :math:`H_0^{-}:\ \mu \le L` and :math:`H_0^{+}:\ \mu \ge U`, passing only when both are
rejected, i.e. when :math:`p = \max(p^-, p^+)` falls below the allotted level.

The level controls a different error in each kind of family:

- in *not-reject* families, the level bounds false failures (rejecting a correct implementation),
  while the power requirement bounds false passes;
- in *reject* families, the level bounds false passes (declaring an advantage or an equivalence that
  is not there), while the choice of :math:`K` or :math:`N` bounds false failures.

The error budget
================

The budget of a run is :math:`\alpha_{\text{suite}} = 10^{-3}`, shared among its tests by Holm's
procedure.

**1. Many tests multiply false alarms.** A run executes tens to hundreds of tests. At the usual level
0.05 per test, 100 independent tests would fail spuriously with probability
:math:`1 - 0.95^{100} = 0.994`, so almost every run would turn red for no reason, and red builds that
mean nothing end up ignored. The level is therefore controlled for the whole run (the *family-wise
error rate*), not test by test.

**2. Why** :math:`10^{-3}`. The suite runs on every change. With about a thousand runs a year,
:math:`10^{-3}` gives roughly one spurious failure a year across all of them, rare enough for a
failure to be investigated every time. A lower level would add sample size for little gain
(point 5).

**3. Why Holm.** Holm's step-down procedure sorts the :math:`m` p-values of the run,
:math:`p_{(1)} \le \dots \le p_{(m)}`, and compares

.. math::

   p_{(1)} \le \frac{\alpha}{m},\quad p_{(2)} \le \frac{\alpha}{m-1},\quad \dots,\quad
   p_{(j)} \le \frac{\alpha}{m-j+1},

rejecting until the first comparison fails, after which no larger p-value is rejected. It controls the
family-wise error rate at :math:`\alpha` under any dependence between the tests, which matters here
since several checks read the same fields and ensembles. It is also uniformly more powerful than
Bonferroni, which compares every p-value with :math:`\alpha/m`. Procedures that assume independence
(Šidák) or exploit positive dependence (Hochberg) carry no guarantee in this setting.
:func:`budget.holm_levels` implements the procedure. The report shows the level each check was
compared with.

*Example.* With :math:`m = 3` checks and p-values :math:`2.4\cdot10^{-87}`, :math:`4.4\cdot10^{-5}`
and :math:`3.7\cdot10^{-4}`, the levels are :math:`\alpha/3 = 3.3\cdot10^{-4}`,
:math:`\alpha/2 = 5\cdot10^{-4}` and :math:`\alpha/1 = 10^{-3}`, so all three are rejected.

**4. Why a declared power.** Every check states a *minimum detectable effect* :math:`\delta`, the
smallest deviation that matters scientifically (e.g. a co-occurrence probability off by 0.02). Its
sample size must give power at least 0.9 at :math:`\delta`. A pass then carries content. Had the
implementation been off by :math:`\delta` or more, the test would have caught it with probability at
least 0.9. Negative controls (below) verify that the power is real.

**5. The cost.** For a proportion :math:`p` tested at a per-test level :math:`\alpha_i` with power
:math:`1-\beta` at effect :math:`\delta`,

.. math::

   N \;\ge\; \frac{\left(z_{1-\alpha_i/2} + z_{1-\beta}\right)^2\, p(1-p)}{\delta^2}

(:func:`budget.required_n`). The table gives :math:`N` for :math:`\alpha_{\text{suite}} = 10^{-3}`
split over :math:`m` tests (worst case :math:`\alpha/m`), with :math:`p = 0.5` and power 0.9
(:math:`z_{0.9} = 1.282`).

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

The cost grows only logarithmically with the number of tests (through :math:`z`) but quadratically
as :math:`\delta` shrinks, so precision, more than the error budget, drives the cost.

**6. Enough replicate fields for a test to be able to pass.** A "beats" claim across :math:`K` fields
decided by a sign test has smallest possible p-value :math:`2^{-K}`. With :math:`K = 6` that value is
:math:`1/64 \approx 0.016`, larger than the whole budget, so such a check could never pass. The suite
therefore uses a paired :math:`t` test when the differences are roughly normal, since it reaches small
p-values with few fields when the effect is consistent, or a sign or Wilcoxon test with
:math:`K \ge \lceil \log_2(m/\alpha) \rceil` fields (:func:`budget.min_sign_test_fields`), which
means 10 fields for a single test and 16 for :math:`m = 50`. "Better on all six fields" counts as
good evidence, short of a formal test at this level.

**7. Two modes.** The rules stay the same while the declared effect changes.

- ``ci`` (every change) declares :math:`\delta \approx 0.05`, with a few thousand draws per check,
  and runs fast enough to catch gross breakage. A pass says "no deviation larger than about 0.05".
- ``full`` (before a release, or to certify another implementation) declares
  :math:`\delta \approx 0.02`, with larger ensembles and more fields, and runs slowly. Only a
  ``full`` pass says "about 0.02".

Almost-sure checks outside the budget
=====================================

An almost-sure check counts violations of a property that a correct implementation satisfies with
certainty (an output outside the data range for a positive-weight decoder, an infinite value, ...).
It passes with zero violations. Since a correct implementation cannot fail it, its probability of a
false failure is zero, so it spends none of the error budget. Holm's procedure applies to the
statistical tests only, the report showing ``exact`` in place of a level. Without this rule, a
scenario with many almost-sure checks (such as :ref:`S12 <scenario-S12>`) would lower the levels of
every statistical test of the run without adding any risk of a false alarm.

Known failures
==============

A pre-registered check that fails on Spatialize documents a finding. Either the code is fixed or the
finding is accepted, while the threshold itself never changes. While a finding stands, the check carries
``known_failure: <reason>`` in its scenario file. It still runs inside the Holm budget like any other
test, reported as ``KNOWN`` without failing the run. Once it passes, the report shows ``XPASS``,
failing the run so that the record gets updated in the same change that fixed the behaviour.

Negative controls
=================

Every family must be shown able to fail. Each tier includes checks *expected to reject*, marked
``expect: reject`` in the scenario file for a given encoder profile. A negative control passes when
its test rejects. If it does not, the test lacks power, so its passes elsewhere cannot be trusted.
The first control of the catalogue, :ref:`E2 <scenario-E2>`, tests the closed-form co-occurrence of
the theory's Mondrian process on the Mondrian partition of Spatialize 1.2 (``mondrian-legacy``),
which deviates from it by design (:doc:`encoders`), while the same check passes on the theory's
process, Spatialize's default since version 1.3. The test rejects with :math:`p \approx 10^{-87}` at :math:`N = 3\,100`.

A failed power check says something about the test, not about the estimator. The first functional
tried for criterion V4 failed its check that eight members show more axis artefacts than a hundred,
since it measured smoothing. A functional whose power check and calibration both pass replaced it
(:ref:`S03 <scenario-S03>`).

Reproducibility of a run
========================

- Ensemble members are independent given the data, with one partition draw each. Replicate fields
  are independent draws of the truth, and independent runs use independent seeds.
- Seeds are drawn afresh for every run and printed in the report. Setting
  ``SPATIALIZE_SCENARIO_SEED`` reproduces a run, and no criterion depends on a particular seed.
- Data are ``pinned``, stored as ``.npy`` fields with SHA-256 checksums verified when the catalogue
  is loaded, so every implementation is tested on the same fields. A ``fresh`` mode, in which each
  implementation regenerates fields from the normative description with its own random numbers, is
  planned.

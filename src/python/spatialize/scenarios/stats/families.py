"""Statistical test families of the scenario suite.

Every check of a scenario is decided by one of these functions. Each returns a
:class:`TestResult` holding the test statistic, the p-value and the family's **pass rule**:

- ``pass_if="not_reject"`` — goodness of fit to a closed form, identities, two-sample comparisons
  (:func:`gof_proportions`, :func:`two_sample_ks`): an implementation conforms unless the data
  reject the null hypothesis. The level bounds *false failures*; the declared power at the minimum
  detectable effect bounds false passes.
- ``pass_if="reject"`` — one-sided bounds, equivalence (TOST) and paired relations
  (:func:`mean_greater`, :func:`mean_less`, :func:`tost_mean`, :func:`paired_relation`): passing
  requires evidence against the null hypothesis. The level bounds *false passes*; the number of
  fields bounds false failures.
- :func:`almost_sure` — properties a correct implementation never violates; decided exactly and
  outside the error budget.

The p-values are combined over a run by Holm's procedure (:mod:`.budget`); see the documentation
page *The statistical acceptance framework* for the reasoning behind each rule.
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    """Outcome of one statistical test.

    Attributes
    ----------
    family : str
        Test family: ``"gof-closed"``, ``"almost-sure"``, ``"equivalence"``, ``"one-sided"``,
        ``"paired-relation"`` or ``"two-sample"``.
    statistic : float
        The test statistic (χ², number of violations, mean, ...), as named in ``detail``.
    p_value : float
        p-value of the test (0 or 1 for almost-sure checks).
    pass_if : {"not_reject", "reject"}
        Pass rule of the family (see the module documentation).
    detail : str
        Human-readable summary (sample sizes, estimates, margins) shown in the report.
    """
    family: str
    statistic: float
    p_value: float
    pass_if: str            # "not_reject" | "reject"
    detail: str = ""


def gof_proportions(p_hat, p0, n):
    r"""Goodness of fit of estimated proportions to closed-form targets.

    Parameters
    ----------
    p_hat : array_like of shape (k,)
        Estimated proportions, each from ``n`` independent Bernoulli draws.
    p0 : array_like of shape (k,)
        Target proportions under the null hypothesis, in (0, 1).
    n : int
        Number of draws behind each proportion.

    Returns
    -------
    TestResult
        Family ``"gof-closed"``, pass rule ``"not_reject"``; ``detail`` names the proportion with the
        largest standardised deviation.

    Notes
    -----
    With :math:`z_i = (\hat p_i - p_i)/\sqrt{p_i(1-p_i)/n}`, the statistic
    :math:`\sum_i z_i^2` is asymptotically :math:`\chi^2_k` under the null hypothesis when the
    proportions are independent.
    """
    p_hat, p0 = np.atleast_1d(p_hat).astype(float), np.atleast_1d(p0).astype(float)
    z = (p_hat - p0) / np.sqrt(p0 * (1 - p0) / n)
    chi2 = float(np.sum(z ** 2))
    p = float(stats.chi2.sf(chi2, df=len(z)))
    worst = int(np.argmax(np.abs(z)))
    return TestResult("gof-closed", chi2, p, "not_reject",
                      f"k={len(z)} N={n} max|z|={abs(z[worst]):.2f} at #{worst} "
                      f"(p̂={p_hat[worst]:.4f} vs {p0[worst]:.4f})")


def almost_sure(n_violations, n_checked):
    """Decide an almost-sure property: it holds exactly when there is no violation.

    Parameters
    ----------
    n_violations : int
        Number of violations observed.
    n_checked : int
        Number of cases examined (reported only).

    Returns
    -------
    TestResult
        Family ``"almost-sure"`` with p-value 0 (violated) or 1 (held). Almost-sure checks are
        decided exactly and do not take part in Holm's procedure.
    """
    p = 0.0 if n_violations > 0 else 1.0
    return TestResult("almost-sure", float(n_violations), p, "not_reject",
                      f"{n_violations} violations in {n_checked}")


def tost_mean(x, low, high):
    """Equivalence test (TOST): the mean of ``x`` lies inside ``(low, high)``.

    Parameters
    ----------
    x : array_like of shape (K,)
        One value per replicate field.
    low, high : float
        Equivalence margin.

    Returns
    -------
    TestResult
        Family ``"equivalence"``, pass rule ``"reject"``: passing requires rejecting both
        :math:`H_0^-: \\mu \\le` ``low`` and :math:`H_0^+: \\mu \\ge` ``high``; the p-value is the
        larger of the two one-sided t-test p-values.
    """
    x = np.asarray(x, float)
    n, m, se = len(x), x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    if se == 0:
        p = 0.0 if low < m < high else 1.0
    else:
        p_low = stats.t.sf((m - low) / se, df=n - 1)       # H0: mean <= low
        p_high = stats.t.cdf((m - high) / se, df=n - 1)    # H0: mean >= high
        p = float(max(p_low, p_high))
    return TestResult("equivalence", float(m), p, "reject",
                      f"mean={m:.4g} se={se:.3g} margin=({low:g}, {high:g}) K={n}")


def mean_greater(x, bound):
    """One-sided t-test: the mean of ``x`` exceeds ``bound``.

    Parameters
    ----------
    x : array_like of shape (K,)
        One value per replicate field.
    bound : float
        Lower bound to exceed.

    Returns
    -------
    TestResult
        Family ``"one-sided"``, pass rule ``"reject"`` (passes when :math:`H_0: \\mu \\le` ``bound``
        is rejected).
    """
    x = np.asarray(x, float)
    n, m, se = len(x), x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    p = (0.0 if m > bound else 1.0) if se == 0 else float(stats.t.sf((m - bound) / se, df=n - 1))
    return TestResult("one-sided", float(m), p, "reject", f"mean={m:.4g} se={se:.3g} bound={bound:g} K={n}")


def mean_less(x, bound):
    """One-sided t-test: the mean of ``x`` is below ``bound``.

    Parameters
    ----------
    x : array_like of shape (K,)
        One value per replicate field.
    bound : float
        Upper bound to stay under.

    Returns
    -------
    TestResult
        Family ``"one-sided"``, pass rule ``"reject"`` (passes when :math:`H_0: \\mu \\ge` ``bound``
        is rejected).
    """
    x = np.asarray(x, float)
    n, m, se = len(x), x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    p = (0.0 if m < bound else 1.0) if se == 0 else float(stats.t.cdf((m - bound) / se, df=n - 1))
    return TestResult("one-sided", float(m), p, "reject", f"mean={m:.4g} se={se:.3g} bound=<{bound:g} K={n}")


def paired_relation(a, b, better="greater", test="paired_t"):
    """Paired relation across replicate fields: estimator A beats estimator B.

    Parameters
    ----------
    a, b : array_like of shape (K,)
        The same metric for A and B on each of K fields (paired by field).
    better : {"greater", "less"}
        Direction in which A should beat B.
    test : {"paired_t", "sign"}
        Paired t-test on the differences, or sign test (needs at least
        :func:`~spatialize.scenarios.stats.budget.min_sign_test_fields` fields to be able to pass).

    Returns
    -------
    TestResult
        Family ``"paired-relation"``, pass rule ``"reject"`` (passes when "no advantage" is
        rejected).

    Raises
    ------
    ValueError
        If ``test`` is unknown.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    if better == "less":
        d = -d
    if test == "paired_t":
        r = mean_greater(d, 0.0)
        return TestResult("paired-relation", r.statistic, r.p_value, "reject", "paired t: " + r.detail)
    if test == "sign":
        k = int(np.sum(d > 0)); n = int(np.sum(d != 0))
        p = float(stats.binom.sf(k - 1, n, 0.5)) if n else 1.0
        return TestResult("paired-relation", float(k), p, "reject", f"sign test {k}/{n} positive")
    raise ValueError(f"unknown test {test}")


def two_sample_ks(x, y):
    """Two-sample Kolmogorov–Smirnov test: ``x`` and ``y`` come from the same law.

    Parameters
    ----------
    x, y : array_like
        Independent samples.

    Returns
    -------
    TestResult
        Family ``"two-sample"``, pass rule ``"not_reject"``.
    """
    r = stats.ks_2samp(x, y)
    return TestResult("two-sample", float(r.statistic), float(r.pvalue), "not_reject", f"KS n={len(x)},{len(y)}")

"""Statistical test families of the suite.

Every function returns a :class:`TestResult` with a p-value; the pass rule depends on the family:

- ``pass_if="not_reject"`` (GOF, identity, two-sample): the implementation conforms unless the data
  reject H0. The level bounds *false failures*; power (declared δ) bounds false passes.
- ``pass_if="reject"`` (paired-relation, equivalence, one-sided bounds): passing requires evidence
  against H0. The level bounds *false passes*; power bounds false failures.
- ``almost_sure``: a single violation is an event of probability 0 under H0 (p-value 0 or 1).
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    family: str
    statistic: float
    p_value: float
    pass_if: str            # "not_reject" | "reject"
    detail: str = ""


def gof_proportions(p_hat, p0, n):
    """Estimated proportions vs closed-form targets: Σ z² ~ χ²(k) under H0 (independent draws)."""
    p_hat, p0 = np.atleast_1d(p_hat).astype(float), np.atleast_1d(p0).astype(float)
    z = (p_hat - p0) / np.sqrt(p0 * (1 - p0) / n)
    chi2 = float(np.sum(z ** 2))
    p = float(stats.chi2.sf(chi2, df=len(z)))
    worst = int(np.argmax(np.abs(z)))
    return TestResult("gof-closed", chi2, p, "not_reject",
                      f"k={len(z)} N={n} max|z|={abs(z[worst]):.2f} at #{worst} "
                      f"(p̂={p_hat[worst]:.4f} vs {p0[worst]:.4f})")


def almost_sure(n_violations, n_checked):
    p = 0.0 if n_violations > 0 else 1.0
    return TestResult("almost-sure", float(n_violations), p, "not_reject",
                      f"{n_violations} violations in {n_checked}")


def tost_mean(x, low, high):
    """Equivalence: the mean of x lies in (low, high). p = max of the two one-sided t-tests."""
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
    """One-sided: the mean of x exceeds ``bound`` (t-test, pass if H0: mean <= bound is rejected)."""
    x = np.asarray(x, float)
    n, m, se = len(x), x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    p = (0.0 if m > bound else 1.0) if se == 0 else float(stats.t.sf((m - bound) / se, df=n - 1))
    return TestResult("one-sided", float(m), p, "reject", f"mean={m:.4g} se={se:.3g} bound={bound:g} K={n}")


def paired_relation(a, b, better="greater", test="paired_t"):
    """A beats B across K paired fields. ``better``: 'greater' or 'less' (direction of A - B)."""
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
    r = stats.ks_2samp(x, y)
    return TestResult("two-sample", float(r.statistic), float(r.pvalue), "not_reject", f"KS n={len(x)},{len(y)}")

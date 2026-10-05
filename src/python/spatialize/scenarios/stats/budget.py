"""Error budget and power (documentation: The statistical acceptance framework).

α_suite = 1e-3 per run, distributed over the run's tests with Holm's step-down procedure, which
is valid under any dependence between tests and uniformly more powerful than Bonferroni.
"""
import numpy as np
from scipy import stats

ALPHA_SUITE = 1e-3
POWER = 0.9


def holm_levels(p_values, alpha=ALPHA_SUITE):
    """Holm step-down: per-test levels and rejection decisions, in the input order.

    The i-th smallest p-value is compared with alpha/(m - i); once one test is not rejected, no
    larger p-value is rejected either.
    """
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    levels = np.empty(m)
    reject = np.zeros(m, bool)
    still = True
    for rank, idx in enumerate(order):
        levels[idx] = alpha / (m - rank)
        still = still and p[idx] <= levels[idx]
        reject[idx] = still
    return levels, reject


def required_n(p, delta, alpha_i, power=POWER):
    """Draws needed to detect a deviation δ in a proportion p (two-sided level α_i, given power)."""
    z = stats.norm.ppf(1 - alpha_i / 2) + stats.norm.ppf(power)
    return int(np.ceil(z ** 2 * p * (1 - p) / delta ** 2))


def min_sign_test_fields(m, alpha=ALPHA_SUITE):
    """Smallest K for which a sign test can reach the worst-case Holm level α/m (2^-K <= α/m)."""
    return int(np.ceil(np.log2(m / alpha)))

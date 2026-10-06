"""Error budget and power.

A run of the suite controls its family-wise error rate at :data:`ALPHA_SUITE` with Holm's
step-down procedure, which is valid under any dependence between the tests and uniformly more
powerful than Bonferroni. Sample sizes are chosen so that every check has power :data:`POWER` at its
declared minimum detectable effect. See the documentation page *The statistical acceptance
framework* for the reasoning.
"""
import numpy as np
from scipy import stats

#: Family-wise error rate of a run.
ALPHA_SUITE = 1e-3

#: Power every check must have at its declared minimum detectable effect.
POWER = 0.9


def holm_levels(p_values, alpha=ALPHA_SUITE):
    """Holm's step-down procedure.

    Parameters
    ----------
    p_values : array_like of shape (m,)
        p-values of the run's statistical tests.
    alpha : float, default ALPHA_SUITE
        Family-wise error rate.

    Returns
    -------
    levels : ndarray of shape (m,)
        Level each p-value is compared with, in the input order: the i-th smallest p-value
        (0-based rank i) gets :math:`\\alpha/(m - i)`.
    reject : ndarray of bool, shape (m,)
        Decisions in the input order. Once a p-value is not rejected, no larger p-value is.

    Examples
    --------
    >>> levels, reject = holm_levels([2.4e-87, 4.4e-5, 3.7e-4])
    >>> reject.tolist()
    [True, True, True]
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
    """Number of independent draws needed to detect a deviation in a proportion.

    Parameters
    ----------
    p : float
        Proportion under the null hypothesis.
    delta : float
        Minimum detectable effect (absolute deviation of the proportion).
    alpha_i : float
        Two-sided level of this test (worst case under Holm: ``ALPHA_SUITE / m``).
    power : float, default POWER
        Required power at ``delta``.

    Returns
    -------
    int
        :math:`\\lceil (z_{1-\\alpha_i/2} + z_{\\text{power}})^2\\, p(1-p) / \\delta^2 \\rceil`.

    Examples
    --------
    >>> required_n(0.5, 0.05, 1e-3 / 50)
    3077
    """
    z = stats.norm.ppf(1 - alpha_i / 2) + stats.norm.ppf(power)
    return int(np.ceil(z ** 2 * p * (1 - p) / delta ** 2))


def min_sign_test_fields(m, alpha=ALPHA_SUITE):
    """Smallest number of fields for which a sign test can pass under the worst-case Holm level.

    Parameters
    ----------
    m : int
        Number of statistical tests in the run.
    alpha : float, default ALPHA_SUITE
        Family-wise error rate.

    Returns
    -------
    int
        Smallest K with :math:`2^{-K} \\le \\alpha/m`: with fewer fields even "better on every field"
        cannot reach the level.
    """
    return int(np.ceil(np.log2(m / alpha)))

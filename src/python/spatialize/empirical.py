import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.spatial import cKDTree
from scipy.stats import skew as _scipy_skew, skewnorm

from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.exceptions import ConvergenceWarning

# just to turn warnings off
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# --- Ensemble widening (corrects (A)ESA/ESI ensembles that are under-dispersed because
# each member is a partition-*average* and so only captures between-partition variance) ---

#todo: review unused funtions _local_target_variance, _loo_target_variance
def _local_target_variance(obs_pts, obs_vals, query_pts, knn=12):
    """Target local variance at each query point = variance of its knn nearest OBSERVED
    neighbours (captures the local spread, including the nugget)."""
    obs_pts = np.asarray(obs_pts, float)
    obs_vals = np.asarray(obs_vals, float)
    query_pts = np.asarray(query_pts, float)
    k = min(knn, len(obs_pts))
    _, idx = cKDTree(obs_pts).query(query_pts, k=k)
    # scipy squeezes the trailing axis when k==1, returning shape (m,) instead of (m,1);
    # np.atleast_2d would prepend rather than append the missing axis in that case, so
    # reshape explicitly against the known number of query points instead.
    idx = np.reshape(idx, (len(query_pts), k))
    return np.var(obs_vals[idx], axis=1)


def _loo_target_variance(obs_pts, obs_vals, knn=12):
    """Leave-one-out variant of `_local_target_variance` for when the query points ARE the
    observed points (e.g. cross-validation) -- excludes each point's match to itself, which
    would otherwise be a zero-distance neighbour deflating its own target variance. Excludes
    by array index rather than by dropping the nearest column, since duplicate/co-located
    points can rank a distinct tied neighbour ahead of self."""
    obs_pts = np.asarray(obs_pts, float)
    obs_vals = np.asarray(obs_vals, float)
    n = len(obs_pts)
    k = min(knn + 1, n)
    _, idx = cKDTree(obs_pts).query(obs_pts, k=k)
    idx = np.reshape(idx, (n, k))
    target_var = np.empty(n)
    for i in range(n):
        neighbours = idx[i][idx[i] != i][:knn]
        if neighbours.size == 0:  # degenerate: no distinct neighbour available
            neighbours = idx[i][:knn]
        target_var[i] = np.var(obs_vals[neighbours])
    return target_var


def _local_target_skewness(obs_pts, obs_vals, query_pts, knn=12):
    """Target local skewness at each query point = (Fisher-Pearson) skewness of its knn
    nearest OBSERVED neighbours. Calibrates `widening='skew_normal'` the same way
    `_local_target_variance` calibrates 'gamma'."""
    obs_pts = np.asarray(obs_pts, float)
    obs_vals = np.asarray(obs_vals, float)
    query_pts = np.asarray(query_pts, float)
    k = min(knn, len(obs_pts))
    _, idx = cKDTree(obs_pts).query(query_pts, k=k)
    idx = np.reshape(idx, (len(query_pts), k))
    return _scipy_skew(obs_vals[idx], axis=1)


def _loo_target_skewness(obs_pts, obs_vals, knn=12):
    """Leave-one-out variant of `_local_target_skewness`, mirroring `_loo_target_variance`
    (see its docstring for why the self-match must be excluded by index)."""
    obs_pts = np.asarray(obs_pts, float)
    obs_vals = np.asarray(obs_vals, float)
    n = len(obs_pts)
    k = min(knn + 1, n)
    _, idx = cKDTree(obs_pts).query(obs_pts, k=k)
    idx = np.reshape(idx, (n, k))
    target_skew = np.empty(n)
    for i in range(n):
        neighbours = idx[i][idx[i] != i][:knn]
        if neighbours.size == 0:  # degenerate: no distinct neighbour available
            neighbours = idx[i][:knn]
        target_skew[i] = _scipy_skew(obs_vals[neighbours])
    return target_skew


def _resample_to_size(draws, n_out, rng):
    """Resample a pool of widened draws down/up to exactly `n_out`."""
    if draws.size >= n_out:
        return rng.choice(draws, size=n_out, replace=False)
    return rng.choice(draws, size=n_out, replace=True)


def _widen_precheck(z, target_var, n_out, rng, eps=1e-9):
    """Shared setup for the Gamma/SkewNormal widen components: drop non-finite members and
    compute `need` -- the extra variance (beyond the ensemble's own between-member variance)
    that must be injected so the mixture variance matches `target_var`. Returns `(early_exit,
    z, need)`: when `early_exit` is not None, the caller should return it immediately without
    calibrating -- either the sample was empty (NaN-filled draws) or the ensemble already
    disperses at least as much as the target (a true no-op: raw members, resampled
    untouched)."""
    z = z[np.isfinite(z)]
    if z.size == 0:
        return np.full(n_out, np.nan), z, None
    between = np.var(z) if z.size > 1 else 0.0
    need = target_var - between
    if need <= eps or not np.isfinite(need):
        return rng.choice(z, size=n_out, replace=True), z, None
    return None, z, need


def _gamma_widen_component(z, target_var, n_out, rng, eps=1e-9):
    """Widen one location's ensemble `z` into `n_out` draws from a Gamma-mixture predictive
    sample calibrated so its variance matches `target_var`. Each member `z_t` is treated as
    the mean of a `Gamma(shape=k, scale=z_t/k)` component (mean-preserving, positive
    support); `k` is solved from the law of total variance so the mixture variance equals
    `target_var`. Falls back to resampling the raw members when the ensemble already
    disperses at least as much as the target."""
    early_exit, z, need = _widen_precheck(z, target_var, n_out, rng, eps)
    if early_exit is not None:
        return early_exit
    if np.any(z < eps):
        # UserWarning, not RuntimeWarning: this module suppresses RuntimeWarning at import
        # time (line 12) for noisy sklearn/numpy internals, which would silently swallow
        # this warning too if it shared that category.
        warnings.warn(
            "widening='gamma' received non-positive ensemble members; they were clipped to "
            "a small positive epsilon before calibration. Use widening='skew_normal' or "
            "'auto' for variables that may be negative.",
            UserWarning,
        )
    z = np.clip(z, eps, None)  # Gamma needs positive means
    mean_sq = np.mean(z ** 2)
    k = float(np.clip(mean_sq / need, 0.05, 1e6))
    per = max(1, n_out // z.size)
    draws = rng.gamma(shape=k, scale=(z / k)[:, None], size=(z.size, per)).ravel()
    return _resample_to_size(draws, n_out, rng)


# limiting |skewness| of a skew-normal as its shape parameter -> +-infinity, i.e.
# (4-pi)/2 * (2/(pi-2))^1.5 (~0.9953); target skewness values are clipped to this range since
# no skew-normal can match anything more extreme.
_SKEW_NORMAL_MAX_ABS_SKEW = (4 - np.pi) / 2 * (2 / (np.pi - 2)) ** 1.5


def _delta_from_skewness(target_skew, eps=1e-9):
    """Method-of-moments estimate of a skew-normal's `delta = alpha/sqrt(1+alpha**2)` shape
    parameter from a target (Fisher-Pearson) skewness, via Azzalini's moment equations.
    Skewness depends only on `delta` (not on location/scale), and the mapping is monotonic
    and bounded by `+-_SKEW_NORMAL_MAX_ABS_SKEW`, so out-of-range or undefined (NaN, e.g. a
    zero-variance neighbourhood) targets fall back to `delta=0` -- the symmetric case."""
    if target_skew is None or not np.isfinite(target_skew):
        return 0.0
    target_skew = np.clip(target_skew, -_SKEW_NORMAL_MAX_ABS_SKEW + eps, _SKEW_NORMAL_MAX_ABS_SKEW - eps)
    c = ((4 - np.pi) / 2) ** (2 / 3)
    a = np.abs(target_skew) ** (2 / 3)
    delta2 = (np.pi / 2) * a / (a + c)
    return float(np.sign(target_skew) * np.sqrt(delta2))


def _skew_normal_widen_component(z, target_var, target_skew, n_out, rng, eps=1e-9):
    """Widen one location's ensemble `z` into `n_out` draws from a skew-normal-mixture
    predictive sample calibrated so its variance matches `target_var` and its shape matches
    `target_skew` (a k-NN estimate from observed data, see
    `_local_target_skewness`/`_loo_target_skewness`). Each member `z_t` is treated as the mean
    of a `SkewNormal(loc, omega, alpha)` component: `alpha` (via `delta`) is solved once from
    `target_skew` -- shared across members, since skewness doesn't depend on location/scale --
    `omega` from the law of total variance so the mixture variance equals `target_var`, and
    `loc` shifted per member so the component mean stays `z_t` (mean-preserving). Reduces to a
    symmetric Normal-mixture widening when `target_skew` is 0 (delta=0 => SkewNormal =
    Normal)."""
    early_exit, z, need = _widen_precheck(z, target_var, n_out, rng, eps)
    if early_exit is not None:
        return early_exit
    delta = _delta_from_skewness(target_skew, eps)
    omega = np.sqrt(need / max(1.0 - 2.0 * delta ** 2 / np.pi, eps))
    loc = z - omega * delta * np.sqrt(2.0 / np.pi)  # shift so component mean stays z_t
    alpha = delta / np.sqrt(max(1.0 - delta ** 2, eps))
    per = max(1, n_out // z.size)
    draws = skewnorm.rvs(alpha, loc=loc[:, None], scale=omega, size=(z.size, per),
                          random_state=rng)
    return _resample_to_size(draws.ravel(), n_out, rng)


def _widen_sample(sample, widening, target_var, target_skew, rng, eps=1e-9):
    """Dispatch to the Gamma or SkewNormal widening component per `widening`
    ('gamma'/'skew_normal'/'auto'). No-op when `target_var` is unavailable -- callers that
    can't supply a spatial target (e.g. no points/values in scope) simply get the unwidened
    sample back. `target_skew` is only used by 'skew_normal' (falls back to the symmetric
    case, `delta=0`, when None -- see `_delta_from_skewness`)."""
    if target_var is None:
        return sample
    mode = widening
    if mode == "auto":
        mode = "gamma" if np.all(sample >= 0) else "skew_normal"
    if mode == "gamma":
        return _gamma_widen_component(sample, target_var, sample.size, rng, eps)
    return _skew_normal_widen_component(sample, target_var, target_skew, sample.size, rng, eps)


class FittedModelFactory:
    def __init__(self, nan_model_name="replace", nan_replace_func_name="median",
                 point_model_name="kde", kernel="gaussian",
                 bgm_sample_size=1000, bgm_max_iter=100, n_components=3,
                 widening=False, widening_knn=12, seed=None):
        """
        Factory for creating and fitting probabilistic models on sample data.

        This class supports handling missing values (NaNs) and fitting various types of
        density estimation models, such as Kernel Density Estimation (KDE), Gaussian
        Mixture Models (GMM), and Variational Inference-based models.

        :param nan_model_name: Strategy for handling NaN values. Use "replace" to replace NaNs
            using a statistical function, or "ignore" to discard them. Default is "replace".
        :param nan_replace_func_name: Function used to replace NaNs when nan_model_name="replace".
            Valid values are "mean" and "median". Default is "median".
        :param point_model_name: Type of model used to estimate the probability density.
            Options include:
                - "kde": Kernel Density Estimation (fastest)
                - "emm": Expectation-Maximization for GMM
                - "vim": Variational Inference for GMM (slowest)
            Default is "kde".
        :param kernel: Kernel type for KDE. Ignored if model is not "kde". Default is "gaussian".
        :param bgm_sample_size: Sample size used for fitting the Bayesian Gaussian Mixture model.
            Default is 1000.
        :param bgm_max_iter: Maximum number of iterations for GMM fitting. Default is 100.
        :param n_components: Number of mixture components for GMM models. Default is 3.
        :param widening: Corrects ensembles that under-represent local (nugget) spread
            because each member is a partition-average. Options:
                - False: no widening (default).
                - "gamma": widen with mean-preserving Gamma components (requires
                  non-negative values).
                - "skew_normal": widen with mean-preserving SkewNormal components, shaped to
                  match a target local skewness (safe for negative values; falls back to a
                  symmetric Normal-mixture shape when no target skewness is available).
                - "auto": "gamma" if the sample is non-negative, otherwise "skew_normal".
            Widening is calibrated per-call against a target local variance (and, for
            "skew_normal", a target local skewness) estimated from observed data; it is a
            no-op when that target isn't available (see `create`).
        :param widening_knn: Number of nearest observed neighbours used to estimate the
            target local variance/skewness that widening is calibrated against (see
            `_local_target_variance`/`_loo_target_variance`/`_local_target_skewness`/
            `_loo_target_skewness`). Ignored when `widening` is False. Default is 12.
        :param seed: Random seed for reproducibility. Seeds the "emm"/"vim" model fits
            (passed as `random_state` to `GaussianMixture`/`BayesianGaussianMixture`,
            since their EM initialization is stochastic; "kde" fitting has no randomness
            and ignores this). Also used as the default seed for widening (see `create`),
            since resampling/Gamma/SkewNormal draws there are stochastic too. If None
            (default), all of the above remain unseeded/non-deterministic, matching prior
            behaviour.
        :raises ValueError: If invalid parameters are provided.
        """

        self.nan_model_name = nan_model_name
        self.point_model_name = point_model_name
        self.kernel = kernel
        self.bgm_sample_size = bgm_sample_size
        self.bgm_max_iter = bgm_max_iter
        self.n_components = n_components
        self.seed = seed

        if widening not in (False, "gamma", "skew_normal", "auto"):
            raise ValueError("widening must be one of False, 'gamma', 'skew_normal', 'auto'")
        self.widening = widening
        self.widening_knn = widening_knn

        if nan_replace_func_name == "median":
            nan_replace_func_def = np.nanmedian
        elif nan_replace_func_name == "mean":
            nan_replace_func_def = np.nanmean
        else:
            raise ValueError("nan_replace_func_name must be either 'mean' or 'median'")

        # just to memoize some common operations for the for-loop
        def replace_nan_with(arr, with_function=nan_replace_func_def):
            nan_value = with_function(arr)
            return np.nan_to_num(arr, nan=nan_value)

        def ignore_nan(arr):
            return arr[~np.isnan(arr)]

        if nan_model_name == "replace":
            self.nan_model = replace_nan_with
        elif nan_model_name == "ignore":
            self.nan_model = ignore_nan
        else:
            raise ValueError("nan_model_name must be either 'replace' or 'ignore'")

        # the bandwidth is calculated automatically using the silverman method
        if point_model_name == "kde":
            # KDE fitting is deterministic (no random_state in sklearn's KernelDensity
            # constructor); `seed` is not used here.
            self.model = KernelDensity(kernel=self.kernel, bandwidth="silverman", atol=0.5, rtol=0.5)
        elif point_model_name == "vim":  # Variational inference dirichlet process gaussian mixture model
            self.model = BayesianGaussianMixture(n_components=self.n_components, covariance_type='full',
                                                  random_state=self.seed)
        elif point_model_name == "emm":  # Expectation-Maximisation gaussian mixture model
            self.model = GaussianMixture(n_components=self.n_components, covariance_type='full',
                                          random_state=self.seed)
        else:
            raise ValueError("point_model_name must be either 'kde', 'vim', or 'emm'")

    def create(self, sample, target_var=None, target_skew=None, seed=None):
        """
        Fit the configured model to the given sample data.

        This method processes the input sample according to the specified
        NaN-handling strategy, optionally widens it (see `widening`), and then fits the
        selected model to the result.

        :param sample: NumPy array of data points to fit the model to.
        :param target_var: Target local variance to calibrate widening against (typically a
            k-NN estimate from observed data at this sample's location). Only used when
            `widening` is not False; if None, widening is skipped even if configured, since
            there is nothing to calibrate against.
        :param target_skew: Target local skewness to calibrate `widening="skew_normal"`
            against (typically a k-NN estimate from observed data, e.g.
            `_local_target_skewness`/`_loo_target_skewness`). Ignored by "gamma"; if None
            while `widening="skew_normal"`, falls back to the symmetric shape (delta=0).
        :param seed: Random seed for this call's widening draws, overriding the factory's
            `seed` for this sample only. Callers that invoke `create()` once per spatial
            location (e.g. `ess_sample`, `PosteriorSampleAnalyzer`) should derive a distinct
            per-location seed from a shared base seed (e.g. `base_seed + location_index`) so
            that widening doesn't draw identical noise at every location. Falls back to
            `self.seed` when None.
        :return: A tuple containing the fitted model and the processed (possibly widened)
            sample data.
        :raises ValueError: If the sample is not a NumPy array or becomes empty after
            NaN processing.
        """
        if not isinstance(sample, np.ndarray):
            raise ValueError("Sample must be a numpy array")

        # deal with NaNs in the sample data
        data = self.nan_model(sample)

        if data.size == 0:
            raise ValueError("Sample data is empty after removing NaNs.")

        if self.widening is not False:
            if target_var is None:
                # UserWarning, not RuntimeWarning: this module suppresses RuntimeWarning at
                # import time (line 12) for noisy sklearn/numpy internals, which would
                # silently swallow this warning too if it shared that category.
                warnings.warn(
                    f"widening={self.widening!r} is configured but no target_var was "
                    "supplied to create() -- widening is skipped for this sample. This "
                    "happens when the caller (e.g. a density model built directly from "
                    "EmpiricalModel/FittedModelFactory rather than through "
                    "ess_sample/cv_sample_pred_posterior) has no spatial target to "
                    "calibrate against.",
                    UserWarning,
                )
            elif self.widening == "skew_normal" and target_skew is None:
                warnings.warn(
                    "widening='skew_normal' is configured but no target_skew was supplied "
                    "to create() -- falling back to the symmetric shape for this sample.",
                    UserWarning,
                )
            effective_seed = seed if seed is not None else self.seed
            data = _widen_sample(data, self.widening, target_var, target_skew,
                                  rng=np.random.default_rng(effective_seed))

        return self.model.fit(data.reshape(-1, 1)), data

    def __repr__(self):
        return (f"model_name={self.point_model_name}, "
                f"nan_model_name={self.nan_model_name}, "
                f"kernel={self.kernel}, "
                f"n_components={self.n_components}, "
                f"bgm_sample_size={self.bgm_sample_size}, "
                f"bgm_max_iter={self.bgm_max_iter}, "
                f"widening={self.widening}, "
                f"widening_knn={self.widening_knn}, "
                f"seed={self.seed}")

# --- Base Class for Probabilistic Models ---

class BaseEmpiricalModel: # Renamed from BaseProbabilisticModel
    """
    Abstract Base Class for 1D probabilistic models.
    """

    def cdf(self, x):
        """
        Compute the Cumulative Distribution Function (CDF) at given x values.

        :param x: Points at which to evaluate the CDF.
        :type x: float or np.ndarray
        :returns: CDF values.
        :rtype: float or np.ndarray
        """
        raise NotImplementedError

    def inv_cdf(self, p):
        """
        Compute the inverse CDF (quantile function) at given probability values.

        :param p: Probability values (between 0 and 1).
        :type p: float or np.ndarray
        :returns: Quantile values.
        :rtype: float or np.ndarray
        """
        raise NotImplementedError

    def entropy(self, a=None, b=None, base=np.e, epsilon=1e-10):
        """
        Estimate the differential entropy over [a, b].

        .. math::

            H = - \\int_a^b p(x) \\log_b p(x) \\, dx

        :param a: Lower limit of integration, defaults to model's min.
        :type a: float, optional
        :param b: Upper limit of integration, defaults to model's max.
        :type b: float, optional
        :param base: Log base (e.g., `np.e` or `2`), defaults to np.e.
        :type base: float
        :param epsilon: Small value to avoid log(0), defaults to 1e-10.
        :type epsilon: float
        :returns: Estimated entropy.
        :rtype: float
        """
        raise NotImplementedError

    def central_entropy_interval(self, alpha=0.9, base=np.e, epsilon=1e-10):
        """
        Finds the narrowest interval [a, b] that contains `alpha` percentage
        of the total entropy.

        :param alpha: Desired percentage of total entropy (0 to 1), defaults to 0.9.
        :type alpha: float
        :param base: Log base for entropy calculation, defaults to np.e.
        :type base: float
        :param epsilon: Small value to avoid log(0), defaults to 1e-10.
        :type epsilon: float
        :returns: Dictionary containing the interval, its entropy, tail entropies, and total entropy.
        :rtype: dict
        """
        raise NotImplementedError

    def entropy_informative_interval(self, x0, alpha=0.9, base=np.e, epsilon=1e-10):
        """
        Finds an interval [a, b] around a given point x0 that contains `alpha`
        percentage of the total entropy. The interval is expanded outwards
        from x0 by adding points with the highest entropy contribution.

        :param x0: The central point around which to find the entropy interval.
        :type x0: float
        :param alpha: Desired percentage of total entropy (0 to 1), defaults to 0.9.
        :type alpha: float
        :param base: Log base for entropy calculation, defaults to np.e.
        :type base: float
        :param epsilon: Small value to avoid log(0), defaults to 1e-10.
        :type epsilon: float
        :returns: Dictionary containing the interval, its entropy, tail entropies, and total entropy.
        :rtype: dict
        """
        raise NotImplementedError

    def credible_interval(self, estimator, alpha=0.95):
        """
        Find the narrowest credible interval [a, b] around a given estimator
        such that the total probability mass in [a, b] is at least alpha.
        Also returns the left and right tails and their probability mass.

        :param estimator: Central estimate (e.g., MAP, mean, median).
        :type estimator: float
        :param alpha: Desired probability mass to include in the central interval, defaults to 0.95.
        :type alpha: float
        :returns: Dictionary with interval information.
        :rtype: dict
        """
        raise NotImplementedError

    def mean(self):
        """
        Compute the mean of the distribution.

        :returns: Mean value.
        :rtype: float
        """
        raise NotImplementedError

    def mode(self):
        """
        Compute the mode (peak) of the distribution.

        :returns: Mode value.
        :rtype: float
        """
        raise NotImplementedError

    def median(self):
        """
        Compute the median (inverse CDF at 0.5) of the distribution.

        :returns: Median value.
        :rtype: float
        """
        raise NotImplementedError

    def percentile(self, p):
        """
        Compute p-th percentile of the distribution.

        :param p: Value in [0, 1].
        :type p: float
        :returns: Percentile value.
        :rtype: float
        """
        raise NotImplementedError

    def variance(self):
        """
        Compute variance of the distribution.

        :returns: Variance value.
        :rtype: float
        """
        raise NotImplementedError

    def std(self):
        """
        Compute standard deviation of the distribution.

        :returns: Standard deviation value.
        :rtype: float
        """
        raise NotImplementedError

    def skewness(self):
        """
        Compute skewness of the distribution.

        :returns: Skewness value.
        :rtype: float
        """
        raise NotImplementedError

    def kurtosis(self):
        """
        Compute excess kurtosis of the distribution.

        :returns: Excess kurtosis value.
        :rtype: float
        """
        raise NotImplementedError

    def moment(self, k):
        """
        Compute raw moment of order k.

        :param k: Order of the moment.
        :type k: int
        :returns: Raw moment.
        :rtype: float
        """
        raise NotImplementedError

    def central_moment(self, k):
        """
        Compute central moment of order k.

        :param k: Order of the central moment.
        :type k: int
        :returns: Central moment.
        :rtype: float
        """
        raise NotImplementedError

    def map_estimator(self):
        """
        Return MAP estimate and its density.

        :returns: Dictionary with 'value' and 'density'.
        :rtype: dict
        """
        raise NotImplementedError

    def sample(self, size=1):
        """
        Draw random samples from the distribution.

        :param size: Number of samples to draw, defaults to 1.
        :type size: int
        :returns: Array of samples.
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def compute_confidence_measures(self, x0, alpha_credible_target = 0.95, alpha_entropy_target = 0.05):
        """
        Computes the three proposed confidence measures around a specific point x0.

        This method is implemented in the base class as it relies on abstract
        methods (credible_interval, entropy_interval_around_point) and properties
        (d_min, d_max) that concrete subclasses must implement.

        :param x0: The central point around which to compute confidence.
        :type x0: float
        :param alpha_credible_target: The desired probability mass for the credible interval.
        :type alpha_credible_target: float
        :param alpha_entropy_target: The desired percentage of total entropy for the entropy interval.
        :type alpha_entropy_target: float
        :returns: A dictionary containing the computed confidence scores and the parameters used.
        :rtype: dict
        """
        _ensure_numba()
        # get Parameters from Credible Interval
        credible_info = self.credible_interval(x0, alpha=alpha_credible_target)
        alpha_prob = credible_info['mass']
        L_alpha_prob = credible_info['interval'][1] - credible_info['interval'][0]

        # get Parameters from Entropy Interval
        entropy_info = self.entropy_informative_interval(x0, alpha=alpha_entropy_target)
        alpha_entropy_percent = alpha_entropy_target # The 'alpha' input to entropy_interval_around_point is the percentage inside
        L_alpha_entropy = entropy_info['interval'][1] - entropy_info['interval'][0]

        # normalization constant Z for Simple Relative Confidence
        # Use the square of the total model range as a reasonable scaling factor
        Z_normalization = (self.d_max - self.d_min) ** 2
        if Z_normalization == 0: Z_normalization = 1.0 # Prevent division by zero if range is 0

        # calculate Confidence Measures
        simple_rel_conf = _compute_simple_relative_confidence_numba(
            alpha_prob, L_alpha_prob, alpha_entropy_percent, L_alpha_entropy, Z_normalization
        )
        harm_rel_conf = _compute_relative_harmonic_confidence_numba(
            alpha_prob, L_alpha_prob, alpha_entropy_percent, L_alpha_entropy
        )
        log_ratio_conf = _compute_log_ratio_confidence_numba(
            alpha_prob, L_alpha_prob, alpha_entropy_percent, L_alpha_entropy
        )

        return {
            'x0': x0,
            'alpha_credible_target': alpha_credible_target,
            'alpha_entropy_target': alpha_entropy_target,
            'parameters': {
                'alpha_prob': alpha_prob,
                'L_alpha_prob': L_alpha_prob,
                'alpha_entropy_percent': alpha_entropy_percent,
                'L_alpha_entropy': L_alpha_entropy,
                'Z_normalization': Z_normalization
            },
            'confidence_scores': {
                'simple_relative_confidence': simple_rel_conf,
                'relative_harmonic_confidence': harm_rel_conf,
                'log_ratio_confidence': log_ratio_conf
            },
            'intervals_info': {
                'credible_interval': credible_info,
                'entropy_interval': entropy_info
            }
        }

    def mc_estimator(self, confidence_measure_type, alpha_credible_target = 0.95, alpha_entropy_target = 0.05):
        """
        Finds an estimator (x_e) that maximizes a specified confidence measure.

        This method iterates through the model's x-grid, calculates the specified
        confidence measure for each point, and returns the point that yields
        the maximum confidence.

        :param confidence_measure_type: The type of confidence measure to maximize.
                                        Must be one of 'simple_relative',
                                        'relative_harmonic', or 'log_ratio'.
        :type confidence_measure_type: str
        :param alpha_credible_target: The desired probability mass for the credible interval, defaults to 0.95.
        :type alpha_credible_target: float
        :param alpha_entropy_target: The desired percentage of total entropy for the entropy interval, defaults to 0.05.
        :type alpha_entropy_target: float
        :returns: A dictionary containing the optimal estimator (x_e) and its maximum confidence score.
        :rtype: dict
        :raises ValueError: If an unknown confidence_measure_type is provided.
        """
        valid_confidence_measures = [
            'simple_relative',
            'relative_harmonic',
            'log_ratio'
        ]

        if confidence_measure_type not in valid_confidence_measures:
            raise ValueError(f"Unknown confidence measure type: {confidence_measure_type}. "
                             f"Must be one of {valid_confidence_measures}")

        optimal_xe = None
        max_confidence_score = -np.inf

        # Iterate through each point on the x-grid
        for x_val in self.x_:
            confidence_results = self.compute_confidence_measures(
                x_val,
                alpha_credible_target=alpha_credible_target,
                alpha_entropy_target=alpha_entropy_target
            )
            # Access the confidence score using the adjusted key
            current_confidence_score = confidence_results['confidence_scores'][confidence_measure_type + '_confidence']

            if current_confidence_score > max_confidence_score:
                max_confidence_score = current_confidence_score
                optimal_xe = x_val
        
        return {
            'optimal_estimator': optimal_xe,
            'max_confidence_score': max_confidence_score,
            'confidence_measure_type': confidence_measure_type,
            'alpha_credible_target': alpha_credible_target,
            'alpha_entropy_target': alpha_entropy_target
        }        

def _find_central_entropy_interval_numba(pdf, x, dx, alpha, log_base, epsilon):
    """
    Numba-optimized function to find the narrowest interval [i, j]
    that contains `alpha` percentage of the total entropy.
    It prioritizes non-zero width intervals if they exist.

    :param pdf: Probability density function values on a grid.
    :type pdf: np.ndarray
    :param x: The x-coordinates corresponding to the PDF values.
    :type x: np.ndarray
    :param dx: The step size between x-coordinates.
    :type dx: float
    :param alpha: Desired percentage of total entropy (0 to 1).
    :type alpha: float
    :param log_base: The natural logarithm of the base for entropy calculation.
    :type log_base: float
    :param epsilon: Small value to avoid log(0).
    :type epsilon: float
    :returns: A tuple containing the start index, end index, and total entropy.
    :rtype: tuple
    """
    n = len(pdf)

    total_entropy = _compute_entropy_numba(pdf, 0, n, dx, log_base, epsilon)
    target_entropy = alpha * total_entropy

    best_i_nonzero_width = -1 # Stores the best non-zero width interval
    best_j_nonzero_width = -1
    min_width_nonzero_width = np.inf

    best_i_any_width = -1 # Stores the absolute narrowest interval (could be zero-width)
    best_j_any_width = -1
    min_width_any_width = np.inf

    found_any_valid_interval = False

    # Iterate through all possible intervals
    for i in range(n):
        for j in range(i, n):
            current_ent = _compute_entropy_numba(pdf, i, j + 1, dx, log_base, epsilon)
            
            if current_ent >= target_entropy:
                found_any_valid_interval = True
                current_width = x[j] - x[i]

                # Update best_i_any_width (absolute narrowest)
                if current_width < min_width_any_width:
                    min_width_any_width = current_width
                    best_i_any_width = i
                    best_j_any_width = j
                
                # Update best_i_nonzero_width if current_width is > 0
                if current_width > 0 and current_width < min_width_nonzero_width:
                    min_width_nonzero_width = current_width
                    best_i_nonzero_width = i
                    best_j_nonzero_width = j
                
                # For a fixed 'i', if we found an interval (i,j) that satisfies,
                # any (i, j') where j' > j will be wider and also satisfy (or have same entropy).
                # So we can break the inner loop for this 'i'.
                break 
    
    if best_i_nonzero_width != -1: # If a non-zero width interval was found
        return best_i_nonzero_width, best_j_nonzero_width, total_entropy
    elif found_any_valid_interval: # If no non-zero width, but some valid interval (must be zero-width)
        return best_i_any_width, best_j_any_width, total_entropy
    else: # No interval (not even full range) satisfied the target entropy
        return 0, n - 1, total_entropy # Fallback to full range


def _find_entropy_informative_interval_numba(pdf, x, dx, center_idx, target_entropy, log_base, epsilon):
    """
    Numba-optimized function to find an interval [a, b] around `center_idx`
    that contains `target_entropy`. The interval is expanded outwards
    by adding points with the highest entropy contribution.

    :param pdf: Probability density function values on a grid.
    :type pdf: np.ndarray
    :param x: The x-coordinates corresponding to the PDF values.
    :type x: np.ndarray
    :param dx: The step size between x-coordinates.
    :type dx: float
    :param center_idx: Index of the central point in the x grid.
    :type center_idx: int
    :param target_entropy: The target entropy value to achieve.
    :type target_entropy: float
    :param log_base: The natural logarithm of the base for entropy calculation.
    :type log_base: float
    :param epsilon: Small value to avoid log(0).
    :type epsilon: float
    :returns: A tuple containing the left index, right index, and the current interval entropy.
    :rtype: tuple
    """
    n = len(pdf)
    left, right = center_idx, center_idx
    current_interval_entropy = _compute_entropy_numba(pdf, left, right + 1, dx, log_base, epsilon)
    if current_interval_entropy >= target_entropy: return left, right, current_interval_entropy
    while current_interval_entropy < target_entropy and (left > 0 or right < n - 1):
        expand_left, expand_right = (left > 0), (right < n - 1)
        left_contrib_val, right_contrib_val = -np.inf, -np.inf
        if expand_left:
            p_left_next = pdf[left - 1]
            if p_left_next < epsilon: p_left_next = epsilon
            left_contrib_val = -p_left_next * np.log(p_left_next)
        if expand_right:
            p_right_next = pdf[right + 1]
            if p_right_next < epsilon: p_right_next = epsilon
            right_contrib_val = -p_right_next * np.log(p_right_next)
        
        if expand_left and (not expand_right or left_contrib_val >= right_contrib_val): left -= 1
        elif expand_right: right += 1
        else: break
        current_interval_entropy = _compute_entropy_numba(pdf, left, right + 1, dx, log_base, epsilon)
    return left, right, current_interval_entropy

def _compute_entropy_numba(pdf, start_idx, end_idx, dx, log_base, epsilon):
    """
    Numba-optimized function to compute differential entropy.
    Assumes pdf is already normalized such that sum(pdf * dx) = 1.
    `end_idx` is exclusive.

    :param pdf: Probability density function values on a grid.
    :type pdf: np.ndarray
    :param start_idx: The starting index for integration (inclusive).
    :type start_idx: int
    :param end_idx: The ending index for integration (exclusive).
    :type end_idx: int
    :param dx: The step size between x-coordinates.
    :type dx: float
    :param log_base: The natural logarithm of the base for entropy calculation.
    :type log_base: float
    :param epsilon: Small value to avoid log(0).
    :type epsilon: float
    :returns: The estimated differential entropy.
    :rtype: float
    """
    entropy = 0.0
    start_idx, end_idx = max(0, start_idx), min(len(pdf), end_idx)
    for i in range(start_idx, end_idx):
        p = pdf[i]
        if p < epsilon: p = epsilon
        entropy -= p * np.log(p)
    return entropy * dx / log_base

def _find_credible_interval_numba(pdf, x, dx, center_idx, alpha):
    """
    Numba-optimized function to find the smallest interval [i, j] around `center_idx` such that
    the total probability mass is >= alpha.

    :param pdf: Probability density function values on a grid.
    :type pdf: np.ndarray
    :param x: The x-coordinates corresponding to the PDF values.
    :type x: np.ndarray
    :param dx: The step size between x-coordinates.
    :type dx: float
    :param center_idx: Index of the central point in the x grid.
    :type center_idx: int
    :param alpha: Desired total probability mass.
    :type alpha: float
    :returns: A tuple containing the left index, right index, and the cumulative mass in the interval.
    :rtype: tuple
    """
    n = len(pdf)
    left, right = center_idx, center_idx
    cumulative_mass = pdf[center_idx] * dx
    while cumulative_mass < alpha and (left > 0 or right < n - 1):
        expand_left, expand_right = (left > 0), (right < n - 1)
        if expand_left and expand_right:
            left_val, right_val = pdf[left - 1], pdf[right + 1]
            if left_val > right_val:
                left -= 1
                cumulative_mass += left_val * dx
            else:
                right += 1
                cumulative_mass += right_val * dx
        elif expand_left:
            left -= 1
            cumulative_mass += pdf[left] * dx
        elif expand_right:
            right += 1
            cumulative_mass += pdf[right] * dx
        else: break
    return left, right, cumulative_mass


class EmpiricalModel(BaseEmpiricalModel):
    """
    Empirical 1D probability distribution using a probability density estimation model.

    Fits a model from raw samples or uses a given sklearn model to create
    a numerical PDF, CDF, and inverse CDF, and provides tools for entropy,
    credible intervals, and sampling.

    :param skl_model: A pre-fitted scikit-learn density model. If provided, the model
        is used directly and a large sample is drawn from it to determine support range.
        If not provided, ``sample`` must be given.
    :type skl_model: sklearn.base.BaseEstimator, optional
    :param sample: A 1D NumPy array of data points used to fit a model if ``skl_model``
        is not provided.
    :type sample: numpy.ndarray, optional
    :param support_sample_size: Number of points in the uniform support grid used for
        evaluating the PDF and CDF. Defaults to 1000.
    :type support_sample_size: int, optional
    :param fitted_model_factory: An instance of :class:`FittedModelFactory` used to create a
        model when fitting from sample data. Defaults to a new factory instance.
    :type fitted_model_factory: FittedModelFactory, optional
    """

    class F:
        """
        A wrapper for an Akima interpolation function.

        :param x: Grid for interpolation.
        :type x: array-like
        :param y: Values to interpolate.
        :type y: array-like

        .. py:method:: __call__(x)
            Evaluate the interpolant at x.

            :param x: The point(s) at which to evaluate the interpolant.
            :type x: float or array-like
            :returns: Interpolated value(s).
            :rtype: float or numpy.ndarray

        .. py:method:: derivative()
            Return derivative interpolant.

            :returns: The derivative interpolant.
            :rtype: Akima1DInterpolator
        """
        def __init__(self, x, y):
            # Ensure x is sorted for interpolation
            sort_indices = np.argsort(x)
            x_sorted = x[sort_indices]
            y_sorted = y[sort_indices]
            
            # Ensure x is strictly increasing for interpolation
            # Filter out non-increasing points if any
            unique_x, unique_indices = np.unique(x_sorted, return_index=True)
            unique_y = y_sorted[unique_indices]
            
            self.__f = Akima1DInterpolator(unique_x, unique_y)

        def __call__(self, x):
            return self.__f(x)

        def derivative(self):
            return self.__f.derivative()

    def __init__(self, skl_model=None, sample=None,
                       support_sample_size=1000,
                       fitted_model_factory=FittedModelFactory(),
                       target_var=None, target_skew=None, seed=None):
        """
        Initialize the model wrapper and prepare PDF, CDF, and inverse CDF functions.

        :param skl_model: A pre-fitted scikit-learn density model.
        :type skl_model: sklearn.base.BaseEstimator, optional
        :param sample: A 1D NumPy array of data points.
        :type sample: numpy.ndarray, optional
        :param support_sample_size: Number of points for the uniform support grid.
        :type support_sample_size: int, optional
        :param fitted_model_factory: Factory to create a model when fitting from sample data.
        :type fitted_model_factory: FittedModelFactory, optional
        :param target_var: Target local variance passed through to
            ``fitted_model_factory.create`` for ensemble widening; ignored unless the
            factory is configured with ``widening``.
        :type target_var: float, optional
        :param target_skew: Target local skewness passed through to
            ``fitted_model_factory.create`` for ``widening="skew_normal"``; ignored by other
            widening modes.
        :type target_skew: float, optional
        :param seed: Random seed passed through to ``fitted_model_factory.create`` for this
            sample's widening draws, overriding the factory's own ``seed`` (see
            ``FittedModelFactory.create``). Ignored when ``skl_model`` is given.
        :type seed: int, optional
        :raises ValueError: If neither ``skl_model`` nor ``sample`` is provided, or if ``sample``
            is not a NumPy array.
        :raises Warning: If the estimated PDF integrates to zero (e.g., due to model or data issues).
        """
        if skl_model is not None:
            self.model = skl_model
            self.data_ = None
            # Sample from the model to determine min/max range for grid
            s = self.model.sample(support_sample_size * 10)[0] # Use a larger sample for range
            self.d_min, self.d_max = s.min(), s.max()
        else:
            if sample is None:
                raise ValueError("Either skl_model or sample must be provided")
            if not isinstance(sample, np.ndarray):
                raise ValueError("Sample must be a numpy array")

            self.model, data = fitted_model_factory.create(sample, target_var=target_var,
                                                            target_skew=target_skew, seed=seed)
            # the (possibly widened) data actually used to fit `self.model` -- distinct from
            # `sample` whenever widening resampled it; plotting code should histogram this,
            # not the pre-widening input, or the overlaid PDF/CDF won't match the histogram
            self.data_ = data
            self.d_min, self.d_max = data.min(), data.max()

        # Extend the range slightly to avoid interpolation issues at boundaries
        range_extension = (self.d_max - self.d_min) * 0.1
        self.d_min -= range_extension
        self.d_max += range_extension

        self.log_likelihood = lambda s: self.model.score_samples(s.reshape(-1, 1))
        self.x_, step = np.linspace(self.d_min, self.d_max, support_sample_size, retstep=True)

        # Calculate raw PDF values
        raw_pdf_values = np.exp(self.log_likelihood(self.x_.reshape(-1, 1)))

        # Normalize PDF to ensure it integrates to 1
        # Use trapezoidal rule for better accuracy if needed, but sum * dx is fine for uniform grid
        integral_sum = np.sum(raw_pdf_values) * step
        if integral_sum == 0:
            self.pdf_ = np.zeros_like(raw_pdf_values)
            raise Warning("Estimated PDF integrates to zero. Check sample data or model parameters.")
        else:
            self.pdf_ = raw_pdf_values / integral_sum

        # Calculate CDF
        cdf_ = np.cumsum(self.pdf_) * step # Accumulate with step size
        self.cdf_ = cdf_ / np.max(cdf_) # Normalize to ensure max is 1.0

        # Create interpolators
        self.pdf = self.F(self.x_, self.pdf_)
        self.cdf = self.F(self.x_, self.cdf_)
        self.inv_cdf = self.F(self.cdf_, self.x_)

    def entropy(self, a=None, b=None, base=np.e, epsilon=1e-10):
        """
        Estimate the differential entropy over [a, b].

        .. math::

            H = - \\int_a^b p(x) \\log_b p(x) \\, dx

        :param a: Lower limit of integration.
        :type a: float, optional
        :param b: Upper limit of integration.
        :type b: float, optional
        :param base: Log base (e.g., ``np.e`` or ``2``).
        :type base: float, optional
        :param epsilon: Small value to avoid log(0).
        :type epsilon: float, optional
        :returns: Estimated entropy.
        :rtype: float
        """
        _ensure_numba()
        if a is None:
            a = self.x_[0]
        if b is None:
            b = self.x_[-1]

        # Ensure a and b are within the model's defined range
        a = max(a, self.x_[0])
        b = min(b, self.x_[-1])

        if a >= b:
            # If interval is a point or invalid, entropy is 0
            return 0.0

        # Find indices for integration limits
        # Using np.searchsorted with 'left' and 'right' to handle boundaries
        start_idx = np.searchsorted(self.x_, a, side='left')
        end_idx = np.searchsorted(self.x_, b, side='right')

        # Adjust end_idx if it goes out of bounds
        end_idx = min(end_idx, len(self.x_))

        dx = self.x_[1] - self.x_[0]

        # Call the Numba-optimized function
        # Note: pdf_ is already normalized such that sum(pdf_ * dx) = 1 in __init__
        log_base = np.log(base)

        # Ensure pdf is normalized:
        # this is different from the normalization in __init__ where we normalize the entire pdf_ to integrate to 1.
        # It's to avoid numerical issues in the entropy calculation - for example, if the pdf is not normalized,
        # the differential entropy could be negative or diverge.
        pdf = self.pdf_ / np.sum(self.pdf_)
        return _compute_entropy_numba(pdf, start_idx, end_idx, dx, log_base, epsilon)


    def central_entropy_interval(self, alpha=0.9, base=np.e, epsilon=1e-10):
        """
        Finds the narrowest interval [a, b] that contains ``alpha`` percentage
        of the total entropy.

        :param alpha: Desired percentage of total entropy (0 to 1).
        :type alpha: float, optional
        :param base: Log base for entropy calculation.
        :type base: float, optional
        :param epsilon: Small value to avoid log(0).
        :type epsilon: float, optional
        :returns: Contains the interval, its entropy, tail entropies, and total entropy.
        :rtype: dict
        """
        _ensure_numba()
        dx = self.x_[1] - self.x_[0]

        # Ensure pdf is normalized:
        # this is different from the normalization in __init__ where we normalize the entire pdf_ to integrate to 1.
        # It's to avoid numerical issues in the entropy calculation - for example, if the pdf is not normalized,
        # the differential entropy could be negative or diverge.
        pdf_for_entropy = self.pdf_ / np.sum(self.pdf_)
        log_base = np.log(base)

        # Call the Numba-optimized function to find the interval
        i, j, total_entropy = _find_central_entropy_interval_numba(pdf_for_entropy, self.x_, dx, alpha, log_base, epsilon)

        a, b = self.x_[i], self.x_[j]

        return {
            'interval': (a, b),
            'central_entropy': self.entropy(a, b, base=base, epsilon=epsilon),
            'left_tail': (self.x_[0], a),
            'left_tail_entropy': self.entropy(self.x_[0], a, base=base, epsilon=epsilon),
            'right_tail': (b, self.x_[-1]),
            'right_tail_entropy': self.entropy(b, self.x_[-1], base=base, epsilon=epsilon),
            'total_entropy': total_entropy
        }

    def entropy_informative_interval(self, x0, alpha=0.9, base=np.e, epsilon=1e-10):
        """
        Finds an interval [a, b] around a given point x0 that contains ``alpha``
        percentage of the total entropy. The interval is expanded outwards
        from x0 by adding points with the highest entropy contribution.

        :param x0: The central point around which to find the entropy interval.
        :type x0: float
        :param alpha: Desired percentage of total entropy (0 to 1).
        :type alpha: float, optional
        :param base: Log base for entropy calculation.
        :type base: float, optional
        :param epsilon: Small value to avoid log(0).
        :type epsilon: float, optional
        :returns: Contains the interval, its entropy, tail entropies, and total entropy.
        :rtype: dict
        """
        _ensure_numba()
        dx = self.x_[1] - self.x_[0]

        # Ensure pdf is normalized:
        # this is different from the normalization in __init__ where we normalize the entire pdf_ to integrate to 1.
        # It's to avoid numerical issues in the entropy calculation - for example, if the pdf is not normalized,
        # the differential entropy could be negative or diverge.
        pdf_for_entropy = self.pdf_ / np.sum(self.pdf_)

        log_base = np.log(base)

        total_entropy = self.entropy(base=base, epsilon=epsilon)
        target_entropy = alpha * total_entropy

        # Find the index closest to x0
        center_idx = np.argmin(np.abs(self.x_ - x0))

        # Call the Numba-optimized function to find the interval around x0
        left_idx, right_idx, current_interval_entropy = \
            _find_entropy_informative_interval_numba(
                pdf_for_entropy, self.x_, dx, center_idx, target_entropy, log_base, epsilon
            )

        a, b = self.x_[left_idx], self.x_[right_idx]

        return {
            'center_point': x0,
            'interval': (a, b),
            'central_entropy': self.entropy(a, b, base=base, epsilon=epsilon), # Recalculate for exactness
            'left_tail': (self.x_[0], a),
            'left_tail_entropy': self.entropy(self.x_[0], a, base=base, epsilon=epsilon),
            'right_tail': (b, self.x_[-1]),
            'right_tail_entropy': self.entropy(b, self.x_[-1], base=base, epsilon=epsilon),
            'total_entropy': total_entropy
        }


    def credible_interval(self, estimator, alpha=0.95):
        """
        Find the narrowest credible interval [a, b] around a given estimator
        such that the total probability mass in [a, b] is at least alpha.
        Also returns the left and right tails and their probability mass.

        :param estimator: Central estimate (e.g., MAP, mean, median).
        :type estimator: float
        :param alpha: Desired probability mass to include in the central interval.
        :type alpha: float, optional
        :returns: Contains the interval, its mass, tail masses, and total mass.
        :rtype: dict
        :returns: A dictionary with the following keys:
            - ``center`` (float): The central estimator.
            - ``interval`` (tuple): A tuple (a, b) representing the credible interval.
            - ``mass`` (float): The probability mass within the credible interval.
            - ``left_tail`` (tuple): A tuple (x_min, a) representing the left tail.
            - ``left_tail_mass`` (float): The probability mass of the left tail.
            - ``right_tail`` (tuple): A tuple (b, x_max) representing the right tail.
            - ``right_tail_mass`` (float): The probability mass of the right tail.
            - ``total_mass`` (float): The total probability mass (approximately 1.0).
        """
        _ensure_numba()
        dx = self.x_[1] - self.x_[0]
        # Use the already normalized pdf_ from __init__
        pdf_for_mass = self.pdf_

        center_idx = np.argmin(np.abs(self.x_ - estimator))
        left_idx, right_idx, central_mass = _find_credible_interval_numba(pdf_for_mass, self.x_, dx, center_idx, alpha)

        a = self.x_[left_idx]
        b = self.x_[right_idx]

        # Calculate tail masses using the normalized pdf_
        left_mass = np.sum(pdf_for_mass[:left_idx]) * dx
        right_mass = np.sum(pdf_for_mass[right_idx + 1:]) * dx

        return {
            'center': estimator,
            'interval': (a, b),
            'mass': central_mass,
            'left_tail': (self.x_[0], a),
            'left_tail_mass': left_mass,
            'right_tail': (b, self.x_[-1]),
            'right_tail_mass': right_mass,
            'total_mass': central_mass + left_mass + right_mass
        }

    def mean(self):
        """
        Compute the mean.

        :returns: The mean of the distribution.
        :rtype: float
        """
        dx = self.x_[1] - self.x_[0]
        return np.sum(self.x_ * self.pdf_) * dx

    def mode(self):
        """
        Compute the mode.

        :returns: The mode of the distribution.
        :rtype: float
        """
        return self.x_[np.argmax(self.pdf_)]

    def median(self):
        """
        Compute the median (inverse CDF at 0.5).

        :returns: The median of the distribution.
        :rtype: float
        """
        return self.inv_cdf(0.5)

    def percentile(self, p):
        """
        Compute p-th percentile.

        :param p: Value in [0, 1].
        :type p: float
        :raises ValueError: If percentile ``p`` is not in the range [0, 1].
        :returns: The p-th percentile.
        :rtype: float
        """
        if not (0 <= p <= 1):
            raise ValueError("Percentile must be in [0, 1]")
        return self.inv_cdf(p)

    def variance(self):
        """
        Compute variance.

        :returns: The variance of the distribution.
        :rtype: float
        """
        dx = self.x_[1] - self.x_[0]
        mu = self.mean()
        return np.sum((self.x_ - mu) ** 2 * self.pdf_) * dx

    def std(self):
        """
        Compute standard deviation.

        :returns: The standard deviation of the distribution.
        :rtype: float
        """
        return np.sqrt(self.variance())

    def skewness(self):
        """
        Compute skewness.

        :returns: The skewness of the distribution.
        :rtype: float
        """
        return self.central_moment(3) / (self.std() ** 3)

    def kurtosis(self):
        """
        Compute excess kurtosis.

        :returns: The excess kurtosis of the distribution.
        :rtype: float
        """
        return self.central_moment(4) / (self.std() ** 4) - 3

    def log_pdf(self, x):
        """
        Log of PDF at x.

        :param x: Point(s) at which to evaluate the log PDF.
        :type x: float or array-like
        :returns: Log PDF value(s).
        :rtype: float or numpy.ndarray
        """
        return np.log(self.pdf(x))

    def moment(self, k):
        """
        Compute raw moment of order k.

        :param k: The order of the moment.
        :type k: int
        :returns: The raw moment of order k.
        :rtype: float
        """
        dx = self.x_[1] - self.x_[0]
        return np.sum((self.x_ ** k) * self.pdf_) * dx

    def central_moment(self, k):
        """
        Compute central moment of order k.

        :param k: The order of the central moment.
        :type k: int
        :returns: The central moment of order k.
        :rtype: float
        """
        dx = self.x_[1] - self.x_[0]
        mu = self.mean()
        return np.sum(((self.x_ - mu) ** k) * self.pdf_) * dx

    def map_estimator(self):
        """
        Return MAP estimate and its density.

        :returns: A dictionary containing the MAP value and its density.
        :rtype: dict
        :returns: A dictionary with the following keys:
            - ``value`` (float): The Maximum A Posteriori (MAP) estimate.
            - ``density`` (float): The probability density at the MAP estimate.
        """
        idx = np.argmax(self.pdf_)
        return {'value': self.x_[idx], 'density': self.pdf_[idx]}

    def sample(self, size=1):
        """
        Draw random samples.

        :param size: The number of samples to draw.
        :type size: int, optional
        :returns: An array of random samples.
        :rtype: numpy.ndarray
        """
        u = np.random.uniform(0, 1, size)
        return self.inv_cdf(u)

# --- Functions for Confidence Measures (Optimized with @njit and no type hints) ---

def _compute_simple_relative_confidence_numba(alpha_prob, L_alpha_prob,
                                             alpha_entropy_percent, L_alpha_entropy, Z):
    """
    Calculates the Simple Relative Confidence (C_simple^rel).

    .. math::

        C_{\\text{simple}}^{\\text{rel}} = \\frac{\\alpha_{\\text{prob}} (1 - \\alpha_{\\text{entropy_percent}})}{\\alpha_{\\text{prob}} (1 - \\alpha_{\\text{entropy_percent}}) + \\frac{L_{\\alpha_{\\text{prob}}} L_{\\alpha_{\\text{entropy}}}}{Z}}

    :param alpha_prob: Probability mass in the credible interval [0, 1].
    :type alpha_prob: float
    :param L_alpha_prob: Width of the credible interval.
    :type L_alpha_prob: float
    :param alpha_entropy_percent: Percentage of total entropy in the entropy interval [0, 1].
    :type alpha_entropy_percent: float
    :param L_alpha_entropy: Width of the entropy interval.
    :type L_alpha_entropy: float
    :param Z: Normalization constant, typically (model_range_width)^2.
    :type Z: float
    :returns: The simple relative confidence score [0, 1].
    :rtype: float
    """
    if alpha_prob <= 0 or alpha_entropy_percent <= 0:
        return 0.0
    
    numerator = alpha_prob * alpha_entropy_percent
    width_product = L_alpha_prob * L_alpha_entropy

    # Ensure Z is not zero to prevent ZeroDivisionError
    if Z == 0: 
        Z = 1.0 # Fallback
    
    denominator = numerator + width_product / Z
    
    # If denominator becomes zero (e.g., if numerator is 0), return 0.0
    if denominator == 0:
        return 0.0
        
    return numerator / denominator

def _compute_relative_harmonic_confidence_numba(alpha_prob, L_alpha_prob,
                                               alpha_entropy_percent, L_alpha_entropy):
    """
    Calculates the Relative Harmonic Confidence (C_harm^rel).

    .. math::

        C_{\\text{harm}}^{\\text{rel}} = \\frac{1}{1 + \\left( \\frac{L_{\\alpha_{\\text{prob}}}}{\\alpha_{\\text{prob}}} + \\frac{L_{\\alpha_{\\text{entropy}}}}{\\alpha_{\\text{entropy_percent}}} \\right)}

    :param alpha_prob: Probability mass in the credible interval [0, 1].
    :type alpha_prob: float
    :param L_alpha_prob: Width of the credible interval.
    :type L_alpha_prob: float
    :param alpha_entropy_percent: Percentage of total entropy in the entropy interval [0, 1].
    :type alpha_entropy_percent: float
    :param L_alpha_entropy: Width of the entropy interval.
    :type L_alpha_entropy: float
    :returns: The relative harmonic confidence score [0, 1].
    :rtype: float
    """
    if alpha_prob <= 0 or alpha_entropy_percent <= 0:
        return 0.0 # No mass or no entropy captured means no confidence

    # Handle cases where alpha_prob or alpha_entropy_percent could be 0, leading to ideal terms (0)
    # The initial check `if alpha_prob <= 0 or alpha_entropy_percent <= 0` handles the division by zero.
    term1 = L_alpha_prob / alpha_prob
    term2 = L_alpha_entropy / alpha_entropy_percent
    
    return 1 / (1 + term1 + term2)

def _compute_log_ratio_confidence_numba(alpha_prob, L_alpha_prob,
                                       alpha_entropy_percent, L_alpha_entropy):
    """
    Calculates the Log-Ratio Confidence (C_log-ratio).

    .. math::

        C_{\\text{log-ratio}} = \\frac{1}{1 + \\frac{L_{\\alpha_{\\text{prob}}} L_{\\alpha_{\\text{entropy}}}}{\\alpha_{\\text{prob}}(1-\\alpha_{\\text{entropy_percent}})}}

    :param alpha_prob: Probability mass in the credible interval [0, 1].
    :type alpha_prob: float
    :param L_alpha_prob: Width of the credible interval.
    :type L_alpha_prob: float
    :param alpha_entropy_percent: Percentage of total entropy in the entropy interval [0, 1].
    :type alpha_entropy_percent: float
    :param L_alpha_entropy: Width of the entropy interval.
    :type L_alpha_entropy: float
    :returns: The log-ratio confidence score (0, 1).
    :rtype: float
    """
    if alpha_prob <= 0 or alpha_entropy_percent <= 0:
        return 0.0 # No mass or no entropy captured means no confidence

    width_product = L_alpha_prob * L_alpha_entropy
    concentration_product = alpha_prob * alpha_entropy_percent

    R = width_product / concentration_product
    return 1 / (1 + R)
# --- End Functions for Confidence Measures ---


# --- Lazy Numba Initialization ---

_numba_initialized = False


def _ensure_numba():
    """Lazily apply @njit to all numba-accelerated functions on first use.

    This avoids importing numba at module load time, which is important for
    callers (e.g. ESI scoring) that only use EmpiricalModel's CDF/PDF
    attributes and never invoke the entropy or credible-interval methods.
    """
    global _numba_initialized
    global _compute_entropy_numba, _find_central_entropy_interval_numba
    global _find_entropy_informative_interval_numba, _find_credible_interval_numba
    global _compute_simple_relative_confidence_numba
    global _compute_relative_harmonic_confidence_numba, _compute_log_ratio_confidence_numba

    if _numba_initialized:
        return
    _numba_initialized = True

    from numba import njit

    # compile leaf function first so dependent functions see a numba Dispatcher
    # when they are compiled on their first call
    _compute_entropy_numba = njit(_compute_entropy_numba)
    _find_central_entropy_interval_numba = njit(_find_central_entropy_interval_numba)
    _find_entropy_informative_interval_numba = njit(_find_entropy_informative_interval_numba)
    _find_credible_interval_numba = njit(_find_credible_interval_numba)
    _compute_simple_relative_confidence_numba = njit(_compute_simple_relative_confidence_numba)
    _compute_relative_harmonic_confidence_numba = njit(_compute_relative_harmonic_confidence_numba)
    _compute_log_ratio_confidence_numba = njit(_compute_log_ratio_confidence_numba)

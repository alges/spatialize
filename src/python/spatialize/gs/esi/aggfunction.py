import warnings

import numpy as np
import scipy as sci

from spatialize._math_util import BilateralFilteringFusion


def _over_members(reduce, samples, *args):
    """``reduce(samples, *args, axis=1)`` over the members, a location without any defined member
    giving NaN without numpy's warning about an empty or all-NaN slice (it happens under the session
    setting ``empty_cells="nan"`` far from the data, and is reported by ``empty_cell_fraction``)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="(Mean of empty slice|All-NaN slice encountered)")
        return reduce(samples, *args, axis=1)


def mean(samples):
    """Aggregate ESI samples by taking their mean over the ensemble axis.

    This is the default `agg_function` used by :func:`~spatialize.gs.esi.esi_griddata`
    and :func:`~spatialize.gs.esi.esi_nongriddata` to collapse the ESI
    samples into a single point estimate per location.

    Parameters
    ----------
    samples : array_like, shape (n_points, n_partitions)
        The raw ESI samples, one column per random partition. May contain
        ``NaN`` values, which are ignored.

    Returns
    -------
    ndarray, shape (n_points,)
        The per-point ensemble mean.

    See Also
    --------
    spatialize.gs.esi.lossfunction : Loss functions comparing this estimate
        against the ESI samples.
    spatialize.gs.esi.scorefunction : Cross-validation scores computed
        directly from the raw ESI samples.
    """
    return _over_members(np.nanmean, samples)


def median(samples):
    """Aggregate ESI samples by taking their median over the ensemble axis.

    Parameters
    ----------
    samples : array_like, shape (n_points, n_partitions)
        The raw ESI samples, one column per random partition. May contain
        ``NaN`` values, which are ignored.

    Returns
    -------
    ndarray, shape (n_points,)
        The per-point ensemble median.

    See Also
    --------
    spatialize.gs.esi.lossfunction : Loss functions comparing this estimate
        against the ESI samples.
    spatialize.gs.esi.scorefunction : Cross-validation scores computed
        directly from the raw ESI samples.
    """
    return _over_members(np.nanmedian, samples)


def MAP(samples):
    """Aggregate ESI samples by taking their mode over the ensemble axis.

    Computes the Maximum A Posteriori (MAP) estimate at each point as the
    most frequent value across the ensemble of ESI samples.

    Parameters
    ----------
    samples : array_like, shape (n_points, n_partitions)
        The raw ESI samples, one column per random partition. May contain
        ``NaN`` values, which are ignored.

    Returns
    -------
    ndarray, shape (n_points, 1)
        The per-point ensemble mode.

    See Also
    --------
    spatialize.gs.esi.lossfunction : Loss functions comparing this estimate
        against the ESI samples.
    spatialize.gs.esi.scorefunction : Cross-validation scores computed
        directly from the raw ESI samples.
    """
    return sci.stats.mode(samples, axis=1, keepdims=True, nan_policy="omit").mode


class Percentile:
    """Aggregation callable that reduces ESI samples to a given percentile.

    Instances are used as `agg_function` in :func:`~spatialize.gs.esi.esi_griddata`
    and :func:`~spatialize.gs.esi.esi_nongriddata` in place of :func:`mean`
    or :func:`median`, to obtain e.g. a conservative (high-percentile) or
    optimistic (low-percentile) estimate from the ensemble.

    Parameters
    ----------
    q : float, optional
        Percentile to compute, between 0 and 100. Default: ``75``.

    See Also
    --------
    spatialize.gs.esi.lossfunction : Loss functions comparing this estimate
        against the ESI samples.
    spatialize.gs.esi.scorefunction : Cross-validation scores computed
        directly from the raw ESI samples.
    """
    def __init__(self, q=75):
        """Store the target percentile.

        Parameters
        ----------
        q : float, optional
            Percentile to compute, between 0 and 100. Default: ``75``.
        """
        self.q = q

    def __call__(self, samples):
        """Compute the configured percentile of the samples over the ensemble axis.

        Parameters
        ----------
        samples : array_like, shape (n_points, n_partitions)
            The raw ESI samples, one column per random partition. May
            contain ``NaN`` values, which are ignored.

        Returns
        -------
        ndarray, shape (n_points,)
            The per-point `q`-th percentile.
        """
        return _over_members(np.nanpercentile, samples, self.q)

    def __repr__(self):
        return f"percentile({self.q})"


def _expectile(samples, alpha, steps=60):
    """The expectile of each row at level ``alpha`` by bisection over the range of its members, NaN
    for a row without a defined member."""
    x = np.asarray(samples, dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        lo, hi = np.nanmin(x, axis=1), np.nanmax(x, axis=1)
        for _ in range(steps):
            e = 0.5 * (lo + hi)
            above = np.nanmean(np.clip(x - e[:, None], 0, None), axis=1)
            below = np.nanmean(np.clip(e[:, None] - x, 0, None), axis=1)
            positive = alpha * above - (1 - alpha) * below > 0       # the root lies above e
            lo, hi = np.where(positive, e, lo), np.where(positive, hi, e)
    return 0.5 * (lo + hi)


class Expectile:
    """Aggregation callable that reduces ESI samples to their expectile at a given level.

    The expectile :math:`e_\\alpha` of a law balances the mean departures on either side,
    :math:`\\alpha\\, \\mathrm{E}[(Z - e)^+] = (1 - \\alpha)\\, \\mathrm{E}[(e - Z)^+]`. It is the reading that
    minimises a squared loss weighing an over-prediction by :math:`1 - \\alpha` and an
    under-prediction by :math:`\\alpha` (:class:`AsymmetricLoss` with ``kind="squared"``). At
    :math:`\\alpha = 1/2` it is the mean. Unlike a quantile, it moves with how far the members of the
    tail lie, so it needs a finite second moment to be stable.

    Parameters
    ----------
    alpha : float
        Level in (0, 1). Default: ``0.5``, the mean.

    See Also
    --------
    Percentile : the quantile, its counterpart under a piecewise-linear loss.
    AsymmetricLoss : the reading a pair of costs selects.
    """
    def __init__(self, alpha=0.5):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = alpha

    def __call__(self, samples):
        """The expectile of the members at each location (NaN where none is defined)."""
        return _expectile(samples, self.alpha)

    def __repr__(self):
        return f"expectile({self.alpha})"


class ExpectedShortfall:
    """Aggregation callable that reduces ESI samples to their expected shortfall beyond a quantile.

    The upper expected shortfall at level :math:`\\alpha` is the mean of the members at or above their
    quantile at :math:`\\alpha`, the value at risk, :math:`\\mathrm{E}[Z \\mid Z \\ge q_\\alpha]`. It answers
    how large the value is when it exceeds that quantile. The lower one, ``tail="lower"``, is the mean of
    the members at or below the quantile. The value at risk itself is ``Percentile(100 * alpha)``.

    Parameters
    ----------
    alpha : float
        Level of the quantile, in (0, 1). Default: ``0.9``.
    tail : {"upper", "lower"}
        The tail averaged. Default: ``"upper"``.

    See Also
    --------
    Percentile : the value at risk.
    """
    def __init__(self, alpha=0.9, tail="upper"):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if tail not in ("upper", "lower"):
            raise ValueError("tail must be 'upper' or 'lower'")
        self.alpha, self.tail = alpha, tail

    def __call__(self, samples):
        """The expected shortfall of the members at each location (NaN where none is defined)."""
        x = np.asarray(samples, dtype=float)
        q = _over_members(np.nanquantile, x, self.alpha)
        beyond = x >= q[:, None] if self.tail == "upper" else x <= q[:, None]
        return _over_members(np.nanmean, np.where(beyond, x, np.nan))

    def __repr__(self):
        return f"expected_shortfall({self.alpha}, {self.tail})"


class AsymmetricLoss:
    """Aggregation callable that reads the members by the functional an asymmetric loss selects.

    A use that pays ``over`` for each unit of over-prediction and ``under`` for each unit of
    under-prediction is best served, at each location, by the reading of the law that minimises the
    expected loss. With :math:`\\alpha = \\text{under} / (\\text{over} + \\text{under})`,

    - ``kind="linear"``, the loss :math:`\\text{over}\\,(\\hat z - z)^+ + \\text{under}\\,(z - \\hat z)^+`, is
      minimised by the quantile at :math:`\\alpha`,
    - ``kind="squared"``, the loss :math:`\\text{over}\\,((\\hat z - z)^+)^2 + \\text{under}\\,((z - \\hat z)^+)^2`,
      by the expectile at :math:`\\alpha`,
    - ``kind="zero-one"``, every departure costing alike, by the mode, whatever the costs.

    Equal costs give the median and the mean. The more an over-prediction costs, the lower the reading.
    The book behind Spatialize derives these readings, in its chapter on higher-order spatial
    prediction, from the predictive law the members give.

    Parameters
    ----------
    over : float
        Cost of one unit of over-prediction, positive. Default: ``1``.
    under : float
        Cost of one unit of under-prediction, positive. Default: ``1``.
    kind : {"linear", "squared", "zero-one"}
        Shape of the loss. Default: ``"linear"``.

    Attributes
    ----------
    alpha : float
        The level the costs select, ``under / (over + under)``.

    See Also
    --------
    Percentile, Expectile, MAP : the readings selected.
    spatialize.gs.esi.scorefunction.asymmetric_loss : the same loss as a cross-validation score.
    """
    def __init__(self, over=1.0, under=1.0, kind="linear"):
        if over <= 0 or under <= 0:
            raise ValueError("over and under must be positive")
        if kind not in ("linear", "squared", "zero-one"):
            raise ValueError("kind must be 'linear', 'squared' or 'zero-one'")
        self.over, self.under, self.kind = float(over), float(under), kind
        self.alpha = self.under / (self.over + self.under)

    def __call__(self, samples):
        """The reading of the members that minimises the expected loss at each location."""
        if self.kind == "linear":
            return _over_members(np.nanquantile, np.asarray(samples, dtype=float), self.alpha)
        if self.kind == "squared":
            return _expectile(samples, self.alpha)
        return np.ravel(MAP(samples))

    def __repr__(self):
        return f"asymmetric_loss(over={self.over:g}, under={self.under:g}, {self.kind})"


class WeightedAverage:
    """Aggregation callable that reduces ESI samples to a weighted average.

    Instances are used as `agg_function` in :func:`~spatialize.gs.esi.esi_griddata`
    and :func:`~spatialize.gs.esi.esi_nongriddata` in place of :func:`mean`,
    to give different random partitions unequal contribution to the final
    estimate.

    Parameters
    ----------
    normalize : bool, optional
        If ``True``, the weighted average is z-score normalized and then
        rescaled to the mean and standard deviation of the unweighted
        samples, so the weighting only reshapes the estimate's relative
        variation rather than its overall scale. Default: ``False``.
    weights : array_like, shape (n_partitions,) or None, optional
        Weight assigned to each partition. If ``None``, weights are drawn
        from a symmetric Dirichlet distribution the first time the instance
        is called. Default: ``None``.
    force_resample : bool, optional
        If ``True``, new weights are drawn from a Dirichlet distribution on
        every call, ignoring any previously set/computed `weights`.
        Default: ``True``.

    See Also
    --------
    spatialize.gs.esi.lossfunction : Loss functions comparing this estimate
        against the ESI samples.
    spatialize.gs.esi.scorefunction : Cross-validation scores computed
        directly from the raw ESI samples.
    """
    def __init__(self, normalize=False, weights=None, force_resample=True):
        """Store the weighting configuration.

        Parameters
        ----------
        normalize : bool, optional
            If ``True``, z-score normalize the weighted average and rescale
            it to the unweighted samples' mean/standard deviation. Default:
            ``False``.
        weights : array_like, shape (n_partitions,) or None, optional
            Weight assigned to each partition. If ``None``, weights are
            drawn from a Dirichlet distribution on the first call. Default:
            ``None``.
        force_resample : bool, optional
            If ``True``, resample the weights from a Dirichlet distribution
            on every call. Default: ``True``.
        """
        self.normalize = normalize
        self.weights = weights
        self.force_resample = force_resample

    def __call__(self, samples):
        """Compute the weighted average of the samples over the ensemble axis.

        Parameters
        ----------
        samples : array_like, shape (n_points, n_partitions)
            The raw ESI samples, one column per random partition. May
            contain ``NaN`` values, which are excluded from the average.

        Returns
        -------
        ndarray, shape (n_points,)
            The per-point weighted average, optionally normalized (see
            `normalize`).
        """
        s = samples.shape[1]
        if self.weights is None or self.force_resample:
            rng = np.random.default_rng()
            self.weights = rng.dirichlet([1] * s)
        m_samples = np.ma.array(samples, mask=np.isnan(samples))
        estimation = np.ma.getdata(np.ma.average(m_samples, axis=1, weights=self.weights))
        if self.normalize:
            zscore_estimation = (estimation - np.mean(estimation)) / np.std(estimation)
            return zscore_estimation * np.nanstd(samples) + np.nanmean(samples)
        else:
            return estimation


def identity(samples):
    """Return the samples unchanged.

    Used as the `agg_function` when a "no aggregation" pass-through is
    needed, e.g. by :func:`~spatialize.gs.esi.lossfunction.mse_cube` and
    :func:`~spatialize.gs.esi.lossfunction.mae_cube`, which report loss per
    partition instead of aggregating it over the ensemble.

    Parameters
    ----------
    samples : array_like
        The raw ESI samples (or, when used inside a loss function, the
        per-partition loss cube).

    Returns
    -------
    array_like
        `samples`, unchanged.

    See Also
    --------
    spatialize.gs.esi.lossfunction : Loss functions comparing this estimate
        against the ESI samples.
    spatialize.gs.esi.scorefunction : Cross-validation scores computed
        directly from the raw ESI samples.
    """
    return samples


# Bilateral filter
def bilateral_filter(samples):
    """Aggregate gridded ESI samples with an edge-preserving bilateral filter.

    Treats `samples` as an image cube (two spatial axes plus one axis per
    partition), denoises it with a bilateral filter, and fuses the result
    into a single 2D estimate. Unlike :func:`mean` or :func:`median`, this
    preserves sharp spatial discontinuities in the estimate instead of
    smoothing across them, at the cost of requiring gridded (2D) `xi`.

    Parameters
    ----------
    samples : array_like, shape (d1, d2, n_partitions)
        The raw ESI samples reshaped as a gridded image cube, where `d1`
        and `d2` are the grid dimensions and `n_partitions` indexes the
        random partitions.

    Returns
    -------
    ndarray, shape (d1, d2)
        The fused, bilaterally-filtered estimate.

    See Also
    --------
    spatialize.gs.esi.lossfunction : Loss functions comparing this estimate
        against the ESI samples.
    spatialize.gs.esi.scorefunction : Cross-validation scores computed
        directly from the raw ESI samples.
    """
    bff = BilateralFilteringFusion(cube=samples)
    fusion = bff.eval()
    two_dims_fusion = np.flip(fusion.reshape(fusion.shape[0], fusion.shape[1]), 1)

    return two_dims_fusion

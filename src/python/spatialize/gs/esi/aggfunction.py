import numpy as np
import scipy as sci

from spatialize._math_util import BilateralFilteringFusion


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
    return np.nanmean(samples, axis=1)


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
    return np.nanmedian(samples, axis=1)


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
        return np.nanpercentile(samples, self.q, axis=1)

    def __repr__(self):
        return f"percentile({self.q})"


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

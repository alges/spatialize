import numpy as np


def mae(true_values, samples):
    """
    Computes the Mean Absolute Error (MAE) between the true values and the
    point estimate obtained by averaging the ESI samples.

    Parameters
    ----------
    true_values : array-like, shape (n_points,)
        The actual observed values at each point.
    samples : array-like, shape (n_points, n_partitions)
        ESI samples from cross-validation.

    Returns
    -------
    float
        MAE of the ensemble mean against the true values.
    """
    point_estimate = np.nanmean(samples, axis=1)
    return np.nanmean(np.abs(true_values - point_estimate))


def mse(true_values, samples):
    """
    Computes the Mean Squared Error (MSE) between the true values and the
    point estimate obtained by averaging the ESI samples.

    Parameters
    ----------
    true_values : array-like, shape (n_points,)
        The actual observed values at each point.
    samples : array-like, shape (n_points, n_partitions)
        ESI samples from cross-validation.

    Returns
    -------
    float
        MSE of the ensemble mean against the true values.
    """
    point_estimate = np.nanmean(samples, axis=1)
    return np.nanmean((true_values - point_estimate) ** 2)


def rmse(true_values, samples):
    """
    Computes the Root Mean Squared Error (RMSE) between the true values and the
    point estimate obtained by averaging the ESI samples.

    Parameters
    ----------
    true_values : array-like, shape (n_points,)
        The actual observed values at each point.
    samples : array-like, shape (n_points, n_partitions)
        ESI samples from cross-validation.

    Returns
    -------
    float
        RMSE of the ensemble mean against the true values.
    """
    return np.sqrt(mse(true_values, samples))


def neg_log_likelihood(true_values, samples, min_points=30, degenerate="floor", resolution=0.001):
    r"""
    Computes the average negative log-likelihood score for hyperparameter selection.

    Computes the negative log-likelihood of the true observed values under the predictive
    distributions represented by the ESI samples, each read through a Gaussian kernel density
    with the bandwidth of Silverman's rule computed from its own spread
    (:func:`~spatialize.empirical.silverman_bandwidth`), evaluated exactly. Changing the units
    of the variable by a factor c shifts the score by log c, so the ranking of configurations
    does not depend on the units. This corresponds to maximum likelihood
    estimation - minimizing this score finds hyperparameters that maximize the probability
    of the observed data.

    Parameters
    ----------
    true_values : array-like, shape (n_points,)
        The actual observed values at each point.
    samples : array-like, shape (n_points, n_partitions)
        ESI samples from cross-validation. samples[i, :] contains the predictions
        for point i from each partition when point i was held out.
    min_points : int, default=30
        Minimum number of valid (non-NaN) samples required to fit a distribution.
        Must not exceed n_partitions — use this function only when n_partitions >= 30.
    degenerate : {"floor", "atom", "none"}, default="floor"
        How a law narrower than the resolution :math:`\delta` is read (see Notes).
        ``"floor"``: its bandwidth is raised to :math:`\delta`. ``"atom"``: it is read as a point
        mass at resolution :math:`\delta`. ``"none"``: Silverman's bandwidth whatever its size,
        the reading of versions 1.3 and earlier.
    resolution : float, default=0.001
        :math:`\delta` as a share of the standard deviation of `true_values`, the finest scale the
        score tells apart.

    Returns
    -------
    float
        Average negative log-likelihood. Lower values indicate better fit.
        When true_values is provided, this is the MLE score.

    Notes
    -----
    It is the logarithmic score, the one the theory's error decomposition is
    written with: its cross-validation risk estimates the irreducible entropy
    plus the decoder and encoder errors, which gives the Pareto search its
    guarantees. It judges the calibration of the members' law. With an
    averaging decoder the members are too close together, missing the
    dispersion within the cells, so the score favours configurations that add
    spread; the drawing decoder of the same weights keeps that dispersion. See
    the theory page on error and the choice of a model.

    A datum whose members are all equal (alone in its cell in every partition, say) has a law that is
    a point mass, which no density describes. With ``degenerate="floor"`` no bandwidth falls below
    :math:`\delta`, which bounds the density by :math:`1/(\delta\sqrt{2\pi})`. With
    ``degenerate="atom"`` a law whose bandwidth falls below :math:`\delta` scores
    :math:`-\log(\max(k, 1/2) / (T \delta))`, with :math:`k` the members within :math:`\delta/2`
    of the datum among the :math:`T` valid ones: a high density when the point mass meets the datum,
    a bounded penalty when it misses. With ``degenerate="none"`` such a law keeps Silverman's
    bandwidth, a needle; :func:`crps`, exact on the members, needs no such choice. Since
    :math:`\delta` scales with the data, every option keeps the score free of the units. Members
    tied up to rounding no longer shrink Silverman's bandwidth to the scale of the rounding
    (:func:`~spatialize.empirical.silverman_bandwidth`), which had made the average score swing by
    up to 0.87 under a move of one ulp in every member.
    """
    from sklearn.neighbors import KernelDensity      # local import to avoid circularities
    from spatialize.empirical import silverman_bandwidth

    if degenerate not in ("floor", "atom", "none"):
        raise ValueError("degenerate must be 'floor', 'atom' or 'none'")
    truth = np.asarray(true_values, dtype=float)
    finite = truth[np.isfinite(truth)]
    scale = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    if scale <= 0:
        scale = max(1.0, float(np.abs(finite).max())) if finite.size else 1.0
    delta = float(resolution) * scale

    n_points = len(samples)
    valid_points = n_points
    total_nll = 0.0

    for i in range(n_points):
        # obtain samples for point i
        point_samples = samples[i, :]
        clean_samples = point_samples[np.isfinite(point_samples)]
        try:
            assert len(clean_samples) >= min_points     # ensure a minimum of non-null points

            h = silverman_bandwidth(clean_samples)
            if degenerate == "atom" and h < delta:
                # a near point mass, read at resolution delta
                k = np.sum(np.abs(clean_samples - truth[i]) <= delta / 2)
                point_nll = -np.log(max(k, 0.5) / (len(clean_samples) * delta))
            else:
                if degenerate == "floor":
                    h = max(h, delta)
                kde = KernelDensity(kernel="gaussian", bandwidth=h)
                kde.fit(clean_samples.reshape(-1, 1))
                point_nll = -kde.score_samples([[truth[i]]])[0]

            # add to total neg-log-likelihood
            total_nll += point_nll
        except:
            valid_points -= 1

    if valid_points == 0:
        return np.inf

    return total_nll / valid_points


def _nll_variant(degenerate):
    """The logarithmic score with another reading of near point masses, as a scorer of its own
    (``"nll-atom"``, ``"nll-none"`` in the searches)."""
    def score(true_values, samples):
        return neg_log_likelihood(true_values, samples, degenerate=degenerate)
    score.__name__ = f"neg_log_likelihood_{degenerate}"
    score.base = neg_log_likelihood
    return score


neg_log_likelihood_atom = _nll_variant("atom")
neg_log_likelihood_none = _nll_variant("none")


def is_distribution_scorer(scoring):
    """Whether `scoring` reads the members' law (the logarithmic score in any of its readings, or the
    CRPS), which needs enough members per datum."""
    base = getattr(scoring, "base", getattr(scoring, "func", scoring))
    return base in (neg_log_likelihood, crps)


def crps(true_values, samples):
    """
    Computes the average Continuous Ranked Probability Score (CRPS) of the members.

    The CRPS of a law :math:`F` at an observation :math:`y` is
    :math:`\\int (F(x) - \\mathbf{1}\\{x \\ge y\\})^2\\,dx`. For the empirical law of the members it
    has the exact form :math:`\\mathrm{E}|X - y| - \\tfrac12\\,\\mathrm{E}|X - X'|`, with :math:`X, X'`
    two members drawn independently, so no density is fitted and an atom among the members (several
    members with the same value, as the drawing decoders give) counts as it is.

    Parameters
    ----------
    true_values : array-like, shape (n_points,)
        The actual observed values at each point.
    samples : array-like, shape (n_points, n_partitions)
        ESI samples from cross-validation. samples[i, :] contains the predictions
        for point i from each partition when point i was held out.

    Returns
    -------
    float
        Average CRPS over the points with at least two finite members, in the units of the
        values. Lower is better. ``inf`` when no point has two finite members.
    """
    samples = np.asarray(samples, dtype=float)
    true_values = np.asarray(true_values, dtype=float)
    scores = []
    for y, members in zip(true_values, samples):
        x = np.sort(members[np.isfinite(members)])
        n = len(x)
        if n < 2:
            continue
        spread = 2.0 * np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2     # E|X - X'|
        scores.append(np.mean(np.abs(x - y)) - 0.5 * spread)
    return float(np.mean(scores)) if scores else np.inf


def asymmetric_loss(over=1.0, under=1.0, kind="linear"):
    """
    A cross-validation score that charges the reading an asymmetric loss selects.

    Each datum is predicted by the reading of its members that minimises the expected loss,
    :class:`~spatialize.gs.esi.functionals.AsymmetricLoss` with the same costs, and charged that loss
    against its value, ``over`` per unit of over-prediction and ``under`` per unit of under-prediction,
    linearly or squared. A search scored by it chooses the ensemble the use itself prefers.

    Parameters
    ----------
    over : float
        Cost of one unit of over-prediction, positive. Default: ``1``.
    under : float
        Cost of one unit of under-prediction, positive. Default: ``1``.
    kind : {"linear", "squared"}
        Shape of the loss. Default: ``"linear"``.

    Returns
    -------
    callable
        ``score(true_values, samples) -> float``, the mean loss over the data, lower being better.
    """
    from spatialize.gs.esi.functionals import AsymmetricLoss
    if kind not in ("linear", "squared"):
        raise ValueError("kind must be 'linear' or 'squared'")
    reading = AsymmetricLoss(over, under, kind)

    def score(true_values, samples):
        z = np.asarray(true_values, dtype=float)
        r = reading(np.asarray(samples, dtype=float))
        excess, shortfall = np.clip(r - z, 0, None), np.clip(z - r, 0, None)
        if kind == "squared":
            excess, shortfall = excess ** 2, shortfall ** 2
        return float(np.nanmean(reading.over * excess + reading.under * shortfall))

    score.__name__ = f"asymmetric_loss_{kind}"
    score.min_valid = 2
    return score


# The fewest valid (non-NaN) members each scorer needs to score a datum. A datum with fewer is left
# out of its score: under the session setting empty_cells="nan", the members of a datum whose cell is
# left empty by cross-validation are NaN.
mae.min_valid = 1
mse.min_valid = 1
rmse.min_valid = 1
neg_log_likelihood.min_valid = 30
neg_log_likelihood_atom.min_valid = 30
neg_log_likelihood_none.min_valid = 30
crps.min_valid = 2


def left_out(samples, scoring):
    """How much of the cross-validation the score could not use.

    Parameters
    ----------
    samples : array-like, shape (n_points, n_partitions)
        The cross-validation members of each datum.
    scoring : callable
        The scoring function. Its attribute ``min_valid`` gives the fewest valid members it needs to
        score a datum (1 when it has none).

    Returns
    -------
    data : float
        The share of the data with fewer valid members than ``scoring`` needs, left out of the score.
    members : float
        The share of NaN members.
    """
    samples = np.asarray(samples, dtype=float)
    valid = np.isfinite(samples).sum(axis=1)
    need = getattr(scoring, "min_valid", 1)
    return float(np.mean(valid < need)), float(np.mean(~np.isfinite(samples)))


def warn_left_out(rows, labels, best, max_left_out, best_name="the best"):
    """Warn about the configurations of a search whose score left out more than ``max_left_out`` of
    the data, listing all of them and marking ``best`` (a set of row positions).

    ``rows`` holds ``(left_out_share, nan_member_share)`` per configuration and ``labels`` a short
    description of each."""
    import warnings
    over = [i for i, (d, _) in enumerate(rows) if d > max_left_out]
    if not over:
        return
    lines = [f"  {labels[i]}: {100 * rows[i][0]:.1f} % of the data left out, "
             f"{100 * rows[i][1]:.1f} % of the members NaN{f'  <- {best_name}' if i in best else ''}"
             for i in over]
    warnings.warn(
        f"{len(over)} of {len(rows)} configurations left more than {100 * max_left_out:g} % of the data "
        "out of the cross-validation score, since their cells were left empty (session setting "
        "empty_cells=\"nan\"). Their scores rest on the data that keep neighbours, which favours "
        "partitions too fine for the data:\n" + "\n".join(lines) +
        "\nSetting empty_cells=\"mark\" or \"coarsen\" (spatialize.session), or a smaller alpha, avoids "
        "it; the threshold is the session setting max_left_out.", UserWarning, stacklevel=3)

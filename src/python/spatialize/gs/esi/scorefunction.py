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


def neg_log_likelihood(true_values, samples, min_points=30):
    """
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
    """
    from sklearn.neighbors import KernelDensity      # local import to avoid circularities
    from spatialize.empirical import silverman_bandwidth

    n_points = len(samples)
    valid_points = n_points
    total_nll = 0.0

    for i in range(n_points):
        # obtain samples for point i
        point_samples = samples[i, :]
        clean_samples = point_samples[np.isfinite(point_samples)]
        try:
            assert len(clean_samples) >= min_points     # ensure a minimum of non-null points

            # Fit KDE
            kde = KernelDensity(kernel="gaussian", bandwidth=silverman_bandwidth(clean_samples))
            kde.fit(clean_samples.reshape(-1, 1))

            log_prob = kde.score_samples([[true_values[i]]])
            point_nll = -log_prob[0]

            # add to total neg-log-likelihood
            total_nll += point_nll
        except:
            valid_points -= 1

    if valid_points == 0:
        return np.inf

    return total_nll / valid_points


def crps(true_values, samples):
    """
    Computes the average Continuous Ranked Probability Score (CRPS).

    Computes CRPS as the integral of the squared difference between the predicted CDF
    and the Heaviside step function at the true observation:

        CRPS(F, y) = integral[(F(x) - H(x - y))^2 dx]

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
        Average CRPS. Lower values indicate better probabilistic predictions.
    """
    from spatialize.empirical import EmpiricalModel      # local import to avoid circularities

    n_points = len(samples)
    total_crps = 0.0
    valid_points = n_points

    for i in range(n_points):
        point_samples = samples[i, :]
        clean_samples = point_samples[np.isfinite(point_samples)]

        if len(clean_samples) < 2:
            valid_points -= 1
            continue

        try:
            em = EmpiricalModel(sample=clean_samples)
            x_grid = em.x_
            cdf = em.cdf(x_grid)

            y_true = true_values[i]
            # Heaviside step function: H(x - y) = 1 if x >= y, else 0
            indicator = (x_grid >= y_true).astype(float)
            diff_squared = (cdf - indicator) ** 2
            point_crps = np.trapezoid(diff_squared, x_grid)

            total_crps += point_crps
        except:
            valid_points -= 1

    if valid_points == 0:
        return np.inf

    return total_crps / valid_points
  

# The fewest valid (non-NaN) members each scorer needs to score a datum. A datum with fewer is left
# out of its score: under the session setting empty_cells="nan", the members of a datum whose cell is
# left empty by cross-validation are NaN.
mae.min_valid = 1
mse.min_valid = 1
rmse.min_valid = 1
neg_log_likelihood.min_valid = 30
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

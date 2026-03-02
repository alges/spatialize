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
    distributions represented by the ESI samples. This corresponds to maximum likelihood
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
    """
    from sklearn.neighbors import KernelDensity      # local import to avoid circularities

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
            kde = KernelDensity(kernel="gaussian", bandwidth="silverman", atol=0.5, rtol=0.5)
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
    from spatialize.gs.spa.empirical import EmpiricalModel      # local import to avoid circularities

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
  
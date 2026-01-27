import numpy as np

def neg_log_likelihood(samples, true_values=None, min_points=30):
    """
    Computes the average negative log-likelihood score for hyperparameter selection.

    When `true_values` is provided (recommended for MLE-based hyperparameter selection),
    this function computes the negative log-likelihood of the true observed values under
    the predictive distributions represented by the ESI samples. This corresponds to
    maximum likelihood estimation - minimizing this score finds hyperparameters that
    maximize the probability of the observed data.

    When `true_values` is None (legacy behavior), computes the average NLL of the samples
    themselves, which measures distribution sharpness rather than predictive accuracy.

    Parameters
    ----------
    samples : array-like, shape (n_points, n_partitions)
        ESI samples from cross-validation. samples[i, :] contains the predictions
        for point i from each partition when point i was held out.
    true_values : array-like, shape (n_points,), optional
        The actual observed values at each point. When provided, the function
        computes the likelihood of these values under the predictive distributions.
    min_points : int, default=30
        Minimum number of valid (non-NaN) samples required to fit a distribution.

    Returns
    -------
    float
        Average negative log-likelihood. Lower values indicate better fit.
        When true_values is provided, this is the MLE score.
    """
    from spatialize.gs.spa.empirical import EmpiricalModel      # local import to avoid circularities

    n_points = len(samples)
    valid_points = n_points
    total_nll = 0.0

    for i in range(n_points):
        # obtain samples for point i
        point_samples = samples[i, :]
        clean_samples = point_samples[np.isfinite(point_samples)]
        try:
            assert len(clean_samples) >= min_points     # ensure a minimum of non-null points

            em = EmpiricalModel(sample=clean_samples)

            if true_values is not None:
                # MLE approach: compute likelihood of the true observed value
                # under the predictive distribution fitted to the ESI samples
                true_val = np.array([true_values[i]])
                log_prob = em.log_likelihood(true_val)
                point_nll = -log_prob[0]
            else:
                # Legacy behavior: compute likelihood of samples themselves
                log_probs = em.log_likelihood(clean_samples)
                point_nll = -np.mean(log_probs)

            # add to total neg-log-likelihood
            total_nll += point_nll
        except:
            valid_points -= 1

    if valid_points == 0:
        return np.inf

    return total_nll / valid_points

def crps(samples, true_values=None):
    """
    Computes the average Continuous Ranked Probability Score (CRPS).

    When `true_values` is provided (recommended for proper scoring), this function
    computes CRPS as the integral of the squared difference between the predicted CDF
    and the Heaviside step function at the true observation:

        CRPS(F, y) = integral[(F(x) - H(x - y))^2 dx]

    This is the standard CRPS used in probabilistic forecasting evaluation.

    When `true_values` is None (legacy behavior), computes average CRPS over
    the samples themselves.

    Parameters
    ----------
    samples : array-like, shape (n_points, n_partitions)
        ESI samples from cross-validation. samples[i, :] contains the predictions
        for point i from each partition when point i was held out.
    true_values : array-like, shape (n_points,), optional
        The actual observed values at each point.

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

            if true_values is not None:
                # Standard CRPS: compare predicted CDF to step function at true value
                y_true = true_values[i]
                # Heaviside step function: H(x - y) = 1 if x >= y, else 0
                indicator = (x_grid >= y_true).astype(float)
                diff_squared = (cdf - indicator) ** 2
                point_crps = np.trapezoid(diff_squared, x_grid)
            else:
                # Legacy behavior: average CRPS over all samples
                indicators = (x_grid[None, :] >= clean_samples[:, None]).astype(float)
                diff_squared = (cdf[None, :] - indicators) ** 2
                crps_per_obs = np.trapezoid(diff_squared, x_grid, axis=1)
                point_crps = np.mean(crps_per_obs)

            total_crps += point_crps
        except:
            valid_points -= 1

    if valid_points == 0:
        return np.inf

    return total_crps / valid_points
  
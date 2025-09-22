import numpy as np

def neg_log_likelihood(samples, min_points = 30):
    """
    Computes the average negative logarithmic likelihood of all esi samples.
    """
    from spatialize.gs.spa.empirical import EmpiricalModel      # local import to avoid circularities
    
    n_points = len(samples)
    valid_points = n_points
    total_nll = 0.0

    for i in range(n_points):
        # obtain samples for point i
        point_samples = samples[i,:]
        clean_samples = point_samples[np.isfinite(point_samples)]
        try:
            assert len(clean_samples) >= min_points     # ensure a minimum of non-null points

            em = EmpiricalModel(sample=clean_samples)

            # compute neg-log-likelihood for point i
            log_probs = em.log_likelihood(clean_samples)
            point_nll = -np.sum(log_probs)

            # add to total neg-log-likelihood
            total_nll += point_nll
        except:
            valid_points -= 1

    return total_nll/valid_points

def crps(samples):
    """
    Computes the average continuous ranked probability score (CRPS) of all esi samples.
    """
    from spatialize.gs.spa.empirical import EmpiricalModel      # local import to avoid circularities

    n_points = len(samples)
    total_crps = 0.0

    for i in range(n_points):
        point_samples = samples[i,:]
        em = EmpiricalModel(sample=point_samples)
        x_grid = em.x_
        cdf = em.cdf(x_grid)

        # Calculate all CRPS values
        indicators = (x_grid[None, :] >= point_samples[:, None]).astype(float)
        diff_squared = (cdf[None, :] - indicators) ** 2
        crps_per_obs = np.trapezoid(diff_squared, x_grid, axis=1)       # Integrate over axis=1 (grid)
        point_crps = np.sum(crps_per_obs)
        
        total_crps += point_crps

    return total_crps/n_points
  
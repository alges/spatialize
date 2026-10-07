import time, numpy as np, libspatialize as lsp

from matplotlib import pyplot as plt
from numba import njit

from spatialize import logging
from spatialize.data import load_drill_holes_andes_2D
from spatialize.gs.esi import ESIResult
from spatialize.logging import default_singleton_callback

logging.log.setLevel("DEBUG")

@njit
def idw_local_interpolator(cell_points: np.ndarray,
                           cell_values: np.ndarray,
                           cell_xi: np.ndarray,
                           cell_params: np.ndarray) -> np.ndarray:
    power = cell_params[0] if cell_params.shape[0] > 0 else 2.0  # fallback if empty

    result = []
    for q in cell_xi:
        dists = np.array([np.sqrt(np.sum(np.power(s - q, power * np.ones((cell_points.shape[1],))))) for s in cell_points])
        weights = 1.0 / (1 + np.power(dists, power * np.ones((len(dists),))))
        norm = np.sum(weights)
        result.append(np.sum(cell_values * weights) / norm)
    return (np.array(result))

@njit
def idw_local_interpolator_anisotropic(
        cell_points: np.ndarray,
        cell_values: np.ndarray,
        cell_xi: np.ndarray,
        cell_params: np.ndarray) -> np.ndarray:
    n_qry = cell_xi.shape[0]
    n_smp = cell_points.shape[0]
    n_dim = cell_points.shape[1]

    power = cell_params[0] if cell_params.shape[0] > 0 else 2.0  # fallback if empty

    # Use scaling if provided, else default to ones
    if cell_params.size > 1:
        scaling = cell_params[1:]
    else:
        scaling = np.ones(n_dim, dtype=cell_params.dtype)

    result = np.empty(n_qry, dtype=np.float64)
    eps = 1e-12  # small epsilon for numerical stability

    for i in range(n_qry):
        q = cell_xi[i]
        weights_sum = 0.0
        weighted_val_sum = 0.0

        for j in range(n_smp):
            s = cell_points[j]
            dist_sq = 0.0
            for k in range(n_dim):
                diff = (s[k] - q[k]) * scaling[k]
                dist_sq += diff * diff
            dist = np.sqrt(dist_sq)
            w = 1.0 / (eps + dist ** power)
            weights_sum += w
            weighted_val_sum += cell_values[j] * w

        result[i] = weighted_val_sum / weights_sum if weights_sum > 0 else 0.0

    return result

# loading data
# the samples included in the spatialize package
samples, locations, krig, _ = load_drill_holes_andes_2D()

# estimation data and result shape
w, h = 300, 200

# input variables for non gridded estimation spatialize functions
points = samples[['x', 'y']].values
values = samples[['cu']].values[:, 0]
xi = locations[['x', 'y']].values

# general parameters
n_partitions = 100
alpha = 0.9
exponent = 2.0

# pure C++ implementation
print("running pure c++ esi idw:")
est1 = lsp.run(
    points, values, xi,
    "mondrian", alpha, n_partitions, 206936,
    "idw", {"exponent": exponent},
    visitor=default_singleton_callback
)

esi_result1 = ESIResult(np.nanmean(est1[1], axis=1), est1[1], False, None, xi)
esi_result1.quick_plot()

# custom implementation using numba and C++
print("running custom (numba and c++) esi idw:")
params_iso = np.array([exponent])  # isotropic (no scaling specified)
params_aniso = np.array([exponent, 1.0, 1.0])  # anisotropic with all ones scaling

@njit
def set_cell_params(cell_points, cell_values):
    return params_aniso

est2 = lsp.run(
    points, values, xi,
    "mondrian", alpha, n_partitions, 206936,
    "custom", {"post_creation": set_cell_params, "estimation": idw_local_interpolator_anisotropic},
    visitor=default_singleton_callback
)

esi_result2 = ESIResult(np.nanmean(est2[1], axis=1), est2[1], False, None, xi)
esi_result2.quick_plot()

d = esi_result1.esi_samples(raw=True) - esi_result2.esi_samples(raw=True)
d = d[~np.isnan(d)]

r = sorted(list(d.flatten()))
print(f'max error:{r[-1]}')

plt.show()
"""A decoder written in Python, plain and compiled with numba, against the built-in IDW.

The three estimations below give the same members up to float rounding; the script prints how long
each takes. See the developer guide, "Writing a decoder in Python".
"""
import time

import numpy as np
from numba import njit

from spatialize.data import load_drill_holes_andes_2D
from spatialize.gs.esi import esi_nongriddata
from spatialize.logging import singleton_null_callback


def idw_python(points, values, queries, params):
    """Inverse distance weighting with exponent params[0], in plain Python (NumPy)."""
    p = params[0] if params.size else 2.0
    d = np.sqrt(((queries[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1))
    out = np.empty(len(queries))
    for i in range(len(queries)):
        at = d[i] == 0
        if at.any():                       # a query on a datum takes its value
            out[i] = values[at].mean()
            continue
        w = d[i] ** -p
        out[i] = (w * values).sum() / w.sum()
    return out


@njit(cache=True)
def idw_numba(points, values, queries, params):
    """The same decoder compiled with numba: explicit loops, no Python objects."""
    p = params[0] if params.size > 0 else 2.0
    out = np.empty(queries.shape[0])
    for i in range(queries.shape[0]):
        sw, swv, exact, n_exact = 0.0, 0.0, 0.0, 0
        for j in range(points.shape[0]):
            d2 = 0.0
            for c in range(points.shape[1]):
                diff = points[j, c] - queries[i, c]
                d2 += diff * diff
            if d2 == 0.0:
                exact += values[j]
                n_exact += 1
                continue
            w = d2 ** (-p / 2)
            sw += w
            swv += w * values[j]
        out[i] = exact / n_exact if n_exact > 0 else swv / sw
    return out


@njit(cache=True)
def exponent_two(points, values):
    """The parameters of a cell, computed once per cell: here a fixed exponent."""
    return np.array([2.0])


if __name__ == "__main__":
    samples, locations, _, _ = load_drill_holes_andes_2D()
    points = samples[["x", "y"]].values
    values = samples[["cu"]].values[:, 0]
    xi = locations[["x", "y"]].values
    common = dict(n_partitions=100, alpha=0.9, seed=206936, callback=singleton_null_callback)

    runs = {
        "built-in idw (C++)": dict(local_interpolator="idw", exponent=2.0),
        "custom, plain Python": dict(local_interpolator="custom", estimation=idw_python,
                                     post_creation=exponent_two),
        "custom, numba": dict(local_interpolator="custom", estimation=idw_numba,
                              post_creation=exponent_two),
    }
    idw_numba(np.zeros((2, 2)), np.zeros(2), np.ones((1, 2)), np.array([2.0]))   # compile once
    members = {}
    for name, kw in runs.items():
        t = time.perf_counter()
        members[name] = esi_nongriddata(points, values, xi, **common, **kw).esi_samples(raw=True)
        print(f"{name:22s} {time.perf_counter() - t:6.2f} s")
    reference = members["built-in idw (C++)"]
    for name in list(runs)[1:]:
        print(f"{name}: largest difference from the built-in decoder "
              f"{np.nanmax(np.abs(members[name] - reference)):.1e}")

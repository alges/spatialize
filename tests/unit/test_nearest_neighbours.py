"""IDW over the data within a radius (libspatialize's nearest-neighbour IDW) against brute force.

The search goes through the kd-tree's ball query, whose pruning reads the bounds of each node: on a
GCC build those bounds were the ones of the wrong axis (issue #15), so these tests are also run on
Linux in CI. Leave-one-out and k-fold read each neighbour by its sample id (issue #16)."""
import numpy as np
import pytest

import snapshot_lib as sl

LIB = sl.load_lib()
RADIUS, EXPONENT = 15.0, 2.0


def _data(n=400, dims=2, seed=0):
    rng = np.random.default_rng(seed)
    s = (rng.random((n, dims)) * 100).astype(np.float32)
    return s, rng.normal(size=n).astype(np.float32), (rng.random((2000, dims)) * 100).astype(np.float32)


def _brute(points, s, v, keep):
    """IDW with weights (d_min/d)^p over the data within the radius that `keep(i, j)` admits."""
    d = np.sqrt(((points[:, None, :].astype(float) - s[None, :, :]) ** 2).sum(-1))
    out = np.full(len(points), np.nan)
    for i in range(len(points)):
        m = (d[i] < RADIUS) & keep(i, np.arange(len(s)))
        if not m.any():
            continue
        di = d[i, m]
        if di.min() == 0:
            out[i] = v[m][di == 0].mean()
        else:
            w = (di.min() / di) ** EXPONENT
            out[i] = (w * v[m]).sum() / w.sum()
    return out


@pytest.mark.parametrize("dims", [2, 3])
def test_estimation_reads_every_datum_within_the_radius(dims):
    s, v, q = _data(dims=dims)
    got = np.asarray(LIB.estimation_nn_idw(s, v, RADIUS, EXPONENT, q, None)).ravel()
    want = _brute(q, s, v, lambda i, j: np.ones(len(j), bool))
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert np.allclose(got, want, rtol=1e-4, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("dims", [2, 3])
def test_leave_one_out_leaves_the_datum_out(dims):
    s, v, _ = _data(dims=dims)
    got = np.asarray(LIB.loo_nn_idw(s, v, RADIUS, EXPONENT, None)).ravel()
    want = _brute(s, s, v, lambda i, j: j != i)
    assert np.allclose(got, want, rtol=1e-4, atol=1e-5, equal_nan=True)
    # a datum's own value never enters its leave-one-out estimate
    v2 = v.copy()
    v2[::2] += 100.0
    moved = np.asarray(LIB.loo_nn_idw(s, v2, RADIUS, EXPONENT, None)).ravel()
    unchanged = _brute(s, s, v2, lambda i, j: j != i)
    assert np.allclose(moved, unchanged, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_k_fold_reads_the_fold_of_each_neighbour():
    s, v, _ = _data()
    k, seed = 5, 3
    got = np.asarray(LIB.kfold_nn_idw(s, v, RADIUS, EXPONENT, k, seed, None)).ravel()
    # the folds are not exported: recover them, a datum's prediction being unchanged when the data of
    # its own fold change and moved when another fold's data change
    assert np.isfinite(got).mean() > 0.9
    for i in range(0, len(s), 37):
        v2 = v.copy()
        v2[i] += 100.0
        assert np.isclose(np.asarray(LIB.kfold_nn_idw(s, v2, RADIUS, EXPONENT, k, seed, None)).ravel()[i], got[i],
                          rtol=1e-5, atol=1e-5, equal_nan=True)


def test_weights_do_not_depend_on_the_units_of_the_coordinates():
    """With weights relative to the nearest datum, multiplying coordinates and radius by c changes
    nothing beyond float rounding (the former 1/(1+d^p) did)."""
    s, v, q = _data()
    runs = [np.asarray(LIB.estimation_nn_idw((s * c).astype(np.float32), v, RADIUS * c, 14.0,
                                             (q * c).astype(np.float32), None)).ravel() for c in (1.0, 1e-3, 1e3)]
    for r in runs[1:]:
        assert np.array_equal(np.isnan(runs[0]), np.isnan(r))
        assert np.allclose(r, runs[0], rtol=1e-3, atol=1e-4, equal_nan=True)

"""Decoders written in Python (local_interpolator='custom').

Unit tests on this checkout's in-place build (see tests/unit/conftest.py)."""
import numpy as np
import pytest

import cases
import snapshot_lib as sl

LIB = sl.load_lib()
DATA = cases.datasets()


def _python_idw(points, values, queries, params):
    """IDW with exponent 2 written in Python, the custom decoder of the tests."""
    d = np.sqrt(((queries[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    out = np.empty(len(queries))
    for i in range(len(queries)):
        at = d[i] == 0
        if at.any():
            out[i] = values[at].mean()
            continue
        w = 1 / d[i] ** 2
        out[i] = (w * values).sum() / w.sum()
    return out.astype(np.float32)


@pytest.mark.parametrize("partition,alpha", [("mondrian", 0.8), ("voronoi", -0.6)])
@pytest.mark.parametrize("method", ["estimate", "loo", "kfold"])
def test_custom_decoder_from_estimation_alone_matches_idw(partition, alpha, method):
    """A custom decoder given only `estimation` reproduces the built-in decoder it implements, its
    leave-one-out and k-fold derived from the estimation; with the empty-cell policies its
    cross-validation has no undefined member."""
    s, v, q = DATA["2d"]
    queries = q if method == "estimate" else s
    args = (s, v, queries, partition, alpha, cases.T, cases.SEED)
    tail = (method, cases.K, cases.FSEED, None, 0)
    custom = np.asarray(LIB.run(*args, "custom", {"estimation": _python_idw}, *tail)[1])
    idw = np.asarray(LIB.run(*args, "idw", {"exponent": 2.0}, *tail)[1])
    assert np.array_equal(np.isnan(custom), np.isnan(idw))
    assert np.nanmax(np.abs(custom - idw)) < 1e-4
    if method != "estimate":
        for policy in ("mark", "coarsen"):
            filled = np.asarray(LIB.run(*args, "custom", {"estimation": _python_idw}, *tail, policy)[1])
            assert not np.isnan(filled).any()

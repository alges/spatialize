"""Identities between decoders that must hold bit for bit, on this platform's build.

The sharpened adaptive decoder reduces to the adaptive one when its two factors are switched off
(kappa_r = kappa_g = 0); both share the adaptive fit, so the outputs must be the same bits.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import cases  # noqa: E402
import snapshot_lib as sl  # noqa: E402

LIB = sl.load_lib()
DATA = cases.datasets()


@pytest.mark.parametrize("ds", ["1d", "2d", "3d"])
@pytest.mark.parametrize("method", ["estimate", "loo", "kfold"])
def test_sharpidw_without_its_factors_is_adaptiveidw(ds, method):
    s, v, q = DATA[ds]
    args = (s, v, q, "mondrian", cases.ALPHA, cases.T_ADAPTIVE, cases.SEED)
    tail = (method, cases.K, cases.FSEED)
    adaptive = LIB.run(*args, "adaptiveidw", {}, *tail)[1]
    sharp = LIB.run(*args, "sharpidw", {"kappa_r": 0.0, "kappa_g": 0.0}, *tail)[1]
    assert sl.bitwise_equal(np.asarray(sharp), np.asarray(adaptive))


@pytest.mark.parametrize("partition,alpha", [("mondrian", 0.6), ("mondrian-raw", 0.6), ("voronoi", 0.5), ("voronoi", -0.5)])
def test_cells_agree_with_the_estimators_partitions(partition, alpha):
    """libspatialize.cells labels the partitions run draws: two locations share a cell exactly when
    the cell mean of the indicator of one is positive at the other, under the same seed."""
    s, _, _ = DATA["2d"]
    pts = s[:15]
    labels = LIB.cells(pts, pts, partition, alpha, cases.T, cases.SEED)
    same = labels[:, None, :] == labels[None, :, :]
    for k in range(len(pts)):
        indicator = (np.arange(len(pts)) == k).astype(np.float32)
        members = LIB.run(pts, indicator, pts, partition, alpha, cases.T, cases.SEED, "cellmean", {})[1]
        assert np.array_equal(same[k], members > 0)

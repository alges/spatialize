"""Partitions: the cells read by ``cells`` are those of the estimators and of the Mondrian leaf function.

Unit tests on this checkout's in-place build (see tests/unit/conftest.py)."""
import numpy as np
import pytest

import cases
import snapshot_lib as sl

LIB = sl.load_lib()
DATA = cases.datasets()


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


def _canonical(labels):
    """Labels renumbered by first appearance in each column, so equal groupings give equal arrays."""
    out = np.empty_like(labels)
    for t in range(labels.shape[1]):
        seen = {}
        out[:, t] = [seen.setdefault(x, len(seen)) for x in labels[:, t]]
    return out


@pytest.mark.parametrize("ds", ["2d", "3d"])
def test_cells_group_the_data_as_the_mondrian_leaf_function(ds):
    s, _, _ = DATA[ds]
    leaf = np.asarray(LIB.get_leaf_for_samples_using_esi(s, cases.T, cases.ALPHA, None, cases.SEED))
    labels = np.asarray(LIB.cells(s, s, "mondrian", cases.ALPHA, cases.T, cases.SEED))
    assert np.array_equal(_canonical(leaf), _canonical(labels))

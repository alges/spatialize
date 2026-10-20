"""Partitions: the cells read by ``cells`` are those of the estimators and of the Mondrian leaf function.

Unit tests on this checkout's in-place build (see tests/unit/conftest.py)."""
import numpy as np
import pytest

import cases
import snapshot_lib as sl

LIB = sl.load_lib()
DATA = cases.datasets()


@pytest.mark.parametrize("partition,alpha", [("mondrian", 0.6), ("mondrian-legacy", 0.6), ("voronoi", 0.5), ("voronoi", -0.5)])
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


def test_mondrian_law_at_a_location_does_not_depend_on_the_other_queries():
    """The lifetime comes from the data's box, and the Mondrian process is consistent under
    restriction: queries far beyond the data enlarge the box the partition is drawn on, but the
    probability that two nearby locations share a cell stays exp(-lambda |h|_1)."""
    s, _, _ = DATA["2d"]
    s = s[:40]
    alpha, n = 0.6, 20000
    lam = 1 / (float((s.max(0) - s.min(0)).sum()) * (1 - alpha))
    c, ext = s.mean(0), s.max(0) - s.min(0)
    near = np.array([c, c + [0.1 * ext[0], 0.0], c + [0.0, 0.12 * ext[1]]], np.float32)
    h = np.abs(near[1:] - near[0]).sum(1)
    far = np.array([s.max(0) + 3 * ext, s.min(0) - 2 * ext], np.float32)
    for q in (near, np.vstack([near, far])):
        labels = LIB.cells(s, q, "mondrian", alpha, n, cases.SEED)
        p = (labels[1:3] == labels[0]).mean(1)
        e = np.exp(-lam * h)
        assert np.all(np.abs(p - e) < 4 * np.sqrt(e * (1 - e) / n)), (p, e)


def test_mondrian_block_mark_generator_follows_the_theory():
    """The Mondrian cells of the scenarios' generator: two points share a cell with probability
    exp(-rate * |x - y|_1), and the unit square holds (1 + rate)^2 cells on average."""
    from spatialize.scenarios.generators import fields
    rng = np.random.default_rng(0)
    rate = 4.9
    x, y = np.array([[0.3, 0.4]]), np.array([[0.42, 0.47]])
    same, n_cells = [], []
    for _ in range(4000):
        z = fields.mbm(rng, rate, lambda r, n: np.arange(n, dtype=float))
        same.append(z(x)[0] == z(y)[0])
        n_cells.append(z.n_cells)
    expected = np.exp(-rate * 0.19)
    assert abs(np.mean(same) - expected) < 4 * np.sqrt(expected * (1 - expected) / 4000)
    assert abs(np.mean(n_cells) - (1 + rate) ** 2) < 4 * np.std(n_cells) / np.sqrt(4000)


def test_block_kriging_variance_follows_the_data_locations():
    """The ordinary-kriging variance of scenario S09's proxy: zero for a point block on a datum without
    nugget, at least the nugget there with one (the nugget acting as measurement error), growing away
    from the data, beyond the sill far from them (the error of the estimated mean adds to it), and
    smaller for a block than for one of its points."""
    from spatialize.scenarios.generators.fields import block_kriging_variance_exponential as bkv
    rng = np.random.default_rng(0)
    s = rng.random((50, 2))
    on_datum, near, far = s[:1], s[:1] + 0.01, np.array([[2.0, 2.0]])
    v = bkv(s, [on_datum, near, far], practical_range=0.3)
    assert abs(v[0]) < 1e-10 and 0 < v[1] < v[2] and v[2] > 1.0
    assert bkv(s, [on_datum], practical_range=0.3, nugget=0.4)[0] >= 0.4
    block = np.array([[0.5, 0.5], [0.52, 0.5], [0.5, 0.52], [0.52, 0.52]])
    assert bkv(s, [block], 0.3)[0] < bkv(s, [block[:1]], 0.3)[0]


def test_coverage_family_rejects_under_coverage_only():
    """The coverage family of scenario S07: a conservative interval passes, an interval covering well
    below its level rejects, with the exact binomial lower tail as p-value."""
    from scipy import stats
    from spatialize.scenarios.stats.families import coverage_at_least
    assert coverage_at_least(48, 48, 0.9).p_value == 1.0
    low = coverage_at_least(21, 48, 0.9)
    assert abs(low.p_value - stats.binom.cdf(21, 48, 0.9)) < 1e-15 and low.p_value < 1e-10
    assert low.pass_if == "not_reject"


def test_ordinary_kriging_baseline_interpolates_and_keeps_constants():
    """Scenario S05's ordinary kriging: without nugget it returns the data at their own locations, and its
    weights summing to 1 reproduce a constant field anywhere."""
    from spatialize.scenarios.generators.fields import ordinary_kriging_exponential as ok
    rng = np.random.default_rng(1)
    s, v = rng.random((60, 2)), rng.lognormal(size=60)
    assert np.allclose(ok(s, v, s[:10], practical_range=0.3), v[:10], atol=1e-8)
    assert np.allclose(ok(s, np.full(60, 2.5), rng.random((30, 2)), practical_range=0.3, nugget=0.3), 2.5)


def test_halo_share_reads_the_ring_around_high_cells():
    from spatialize.scenarios.stats.maps import halo_share
    z = np.ones((20, 20))
    top = np.zeros((20, 20), bool)
    top[9:11, 9:11] = True
    z[8, 8:12] = -1.0                                     # one row of the ring dips below the floor
    ring, rest = halo_share(z, top, floor=0.0, width=2)
    assert abs(ring - 4 / 20) < 1e-12 and rest == 0.0       # 20 pixels within two 4-neighbour steps

"""Identities between decoders that must hold bit for bit, on this platform's build.

The sharpened adaptive decoder reduces to the adaptive one when its two factors are switched off
(kappa_r = kappa_g = 0); both share the adaptive fit, so the outputs must be the same bits.
The cells of the data read by ``cells`` are those of the Mondrian trees that
``get_leaf_for_samples_using_esi`` draws, on which ESMI and the Pareto encoder error were built.
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


def test_mutual_information_marginals_integrate_the_joint_density():
    """The marginal entropy read from the joint partition is that of the joint histogram density
    integrated over the other variable (computed here on a fine grid); the projections of the cells
    overlap, so treating them as disjoint bins would overstate it."""
    from spatialize.gs.esmi._main import SpatialMutualInformation
    rng = np.random.default_rng(cases.GENERATOR_SEED)
    u = rng.normal(size=300)
    v = 0.5 * u + rng.normal(size=300)
    smi = SpatialMutualInformation(T=300, M=5, alpha_m=0.8, callback=lambda *a, **k: None)
    _, parts, leaves = smi._calculate_joint_entropy(u, v, 1)
    for marginal, (lo, hi) in (("u", (1, 2)), ("v", (3, 4))):
        h = []
        for part, leaf in zip(parts, leaves.T):
            ids, counts = np.unique(leaf, return_counts=True)
            cells = np.asarray(part)[ids]
            x = np.linspace(cells[:, lo].min(), cells[:, hi].max(), 20001)
            mid, dx = (x[:-1] + x[1:]) / 2, x[1] - x[0]
            f = np.zeros_like(mid)
            for c, n in zip(cells, counts):
                if c[hi] > c[lo]:
                    f += ((mid >= c[lo]) & (mid < c[hi])) * (n / smi.T) / (c[hi] - c[lo])
            h.append(-np.sum(f[f > 0] * np.log2(f[f > 0])) * dx)
        assert abs(smi._calculate_marginal_entropy_from_joint(parts, leaves, marginal) - np.mean(h)) < 1e-3


@pytest.mark.parametrize("scale", [0.01, 100.0])
def test_density_readings_do_not_depend_on_units(scale):
    """The KDE of the local laws (ESS, NLL, Pareto) takes its bandwidth from the sample's own spread,
    so changing the units of the variable changes no conclusion: the KDE spread scales with the data,
    the NLL shifts by log(scale) and the encoder error does not change."""
    from spatialize.empirical import FittedModelFactory
    from spatialize.gs.esi import scorefunction as sf
    from spatialize.gs.esi.pareto import EmpiricalRobustnessBound
    from spatialize import session
    rng = np.random.default_rng(cases.GENERATOR_SEED)
    members, truth = rng.normal(size=(40, 60)), rng.normal(size=40)

    model, _ = FittedModelFactory().create(members[0] * scale)
    reference, _ = FittedModelFactory().create(members[0])
    assert np.isclose(model.bandwidth_, reference.bandwidth_ * scale, rtol=1e-12)

    nll = sf.neg_log_likelihood(truth, members)
    assert np.isclose(sf.neg_log_likelihood(truth * scale, members * scale), nll + np.log(scale), rtol=1e-9)

    s, v, _ = DATA["2d"]
    with session.override(parallel=False):
        eps = EmpiricalRobustnessBound(30, cases.ALPHA, "idw", "mondrian", cases.SEED).estimate(s, v)
        eps_scaled = EmpiricalRobustnessBound(30, cases.ALPHA, "idw", "mondrian", cases.SEED).estimate(
            s, (v * scale).astype(np.float32))
    assert np.isclose(eps, eps_scaled, rtol=1e-4)

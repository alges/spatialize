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
    from spatialize.futures.esmi._main import SpatialMutualInformation
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


@pytest.mark.parametrize("p_process,alpha,data_cond", [("mondrian", 0.9, True), ("mondrian-raw", 0.9, True),
                                                       ("voronoi", 0.8, True), ("voronoi", 0.8, False)])
@pytest.mark.parametrize("griddata", [False, True])
def test_empty_cell_fraction_is_the_share_of_nan_members(p_process, alpha, data_cond, griddata):
    """Under empty_cells="nan" a member is NaN exactly when its cell held no datum, so the share of
    NaN members at each location equals empty_cell_fraction(), read from the partitions alone."""
    from spatialize.gs.esi import esi_griddata, esi_nongriddata
    s, v, q = DATA["2d"]
    kw = dict(local_interpolator="idw", exponent=2.0, p_process=p_process, data_cond=data_cond, alpha=alpha,
              n_partitions=cases.T, seed=cases.SEED, callback=lambda *a, **k: None)
    if griddata:
        grid = np.mgrid[0:1:15j, 0:1:15j]
        r = esi_griddata(s, v, (grid[0], grid[1]), **kw)
    else:
        r = esi_nongriddata(s, v, q, **kw)
    members = r.esi_samples(raw=True)
    share = np.isnan(members).mean(axis=1)
    fraction = np.asarray(r.empty_cell_fraction()).ravel()
    # Voronoi with nuclei at the data has a datum in every cell; the other cases have empty cells
    assert (share.max() == 0) if (p_process == "voronoi" and data_cond) else (share.max() > 0)
    assert np.array_equal(share, fraction)


@pytest.mark.parametrize("decoder", ["adaptiveidw", "sharpidw", "wdraw_adaptiveidw", "wdraw_sharpidw"])
def test_adaptive_decoders_do_not_depend_on_coordinate_units(decoder):
    """The adaptive weights are relative to the nearest datum, so with coordinates 10^4 times larger
    no member of a cell with data is NaN and the members are the same up to float rounding (a
    rounding may flip a discrete choice of the fit or of a draw in a few members)."""
    s, v, q = DATA["2d"]
    runs = [np.asarray(LIB.run((s * c).astype(np.float32), v, (q * c).astype(np.float32), "mondrian", cases.ALPHA,
                               cases.T_ADAPTIVE, cases.SEED, decoder, {}, "estimate")[1]) for c in (1.0, 1e4)]
    assert np.array_equal(np.isnan(runs[0]), np.isnan(runs[1]))
    differ = np.abs(runs[0] - runs[1]) > 1e-4 * (1 + np.abs(runs[0]))
    assert np.nanmean(differ) < 0.01


@pytest.mark.parametrize("partition,alpha", [("mondrian", 0.9), ("mondrian-raw", 0.9), ("voronoi", -0.8)])
@pytest.mark.parametrize("mark_source", ["local", "cells", "data"])
@pytest.mark.parametrize("decoder,params", [("idw", {"exponent": 2.0}), ("draw", {})])
@pytest.mark.parametrize("mark_value", ["decoder", "datum"])
def test_mark_fills_each_empty_cell_with_one_value(partition, alpha, mark_source, decoder, params, mark_value):
    """Under empty_cells="mark" every location of an empty cell takes one value, shared by the cell;
    the cells with data keep the members of "nan"; with a drawing decoder (or mark_source="data") the
    marks are observed values; leave-one-out and k-fold have no NaN; the marks do not depend on the
    number of threads; the empty cells of one partition draw distinct data under "data"."""
    s, v, _ = DATA["2d"]
    rng = np.random.default_rng(cases.GENERATOR_SEED)
    v = rng.normal(size=len(v)).astype(np.float32)     # continuous values: distinct data, distinct marks
    q = rng.uniform(-0.3, 1.3, (400, 2)).astype(np.float32)
    run = lambda policy, method="estimate", threads=0, queries=q: np.asarray(LIB.run(
        s, v, queries, partition, alpha, cases.T, cases.SEED, decoder, params, method, cases.K,
        cases.FSEED, None, threads, policy, mark_source, 8, mark_value)[1])
    mark, nan = run("mark"), run("nan")
    labels = np.asarray(LIB.cells(s, np.vstack([q, s]), partition, alpha, cases.T, cases.SEED))
    at_q, at_s = labels[:len(q)], labels[len(q):]
    empty = np.array([~np.isin(at_q[:, t], at_s[:, t]) for t in range(cases.T)]).T
    assert empty.any() and not np.isnan(mark).any()
    assert np.array_equal(mark[~empty], nan[~empty])
    if decoder == "draw" or mark_source == "data" or mark_value == "datum":
        assert np.isin(mark[empty], v).all()
    for t in range(cases.T):
        cells = np.unique(at_q[empty[:, t], t])
        values = []
        for c in cells:
            shared = set(mark[at_q[:, t] == c, t])
            assert len(shared) == 1
            values += list(shared)
        if mark_source == "data" and len(cells) <= len(v):
            assert len(set(values)) == len(values)      # no datum serves two empty cells
    assert np.array_equal(mark, run("mark", threads=1))
    for method in ("loo", "kfold"):
        assert not np.isnan(run("mark", method, queries=s)).any()


@pytest.mark.parametrize("partition,alpha", [("mondrian", 0.9), ("mondrian-raw", 0.9), ("voronoi", -0.8)])
@pytest.mark.parametrize("decoder,params", [("cellmean", {}), ("idw", {"exponent": 2.0}), ("adaptiveidw", {}),
                                            ("sharpidw", {}), ("wdraw_adaptiveidw", {}), ("draw", {})])
def test_coarsen_predicts_empty_cells_from_a_coarser_cell(partition, alpha, decoder, params):
    """Under empty_cells="coarsen" the cells with data keep the members of "nan" and every member is
    defined; with the cell mean, all the locations of an empty Mondrian cell take one value (one
    ancestor serves the cell) and a location of an empty Voronoi cell takes the mean of another cell
    of the partition (the nearest nucleus with data); leave-one-out and k-fold have no NaN; the
    result does not depend on the number of threads."""
    s, v, _ = DATA["2d"]
    rng = np.random.default_rng(cases.GENERATOR_SEED)
    v = rng.normal(size=len(v)).astype(np.float32)
    q = rng.uniform(-0.3, 1.3, (300, 2)).astype(np.float32)
    forest = cases.T_ADAPTIVE if "adaptive" in decoder or "sharp" in decoder else cases.T
    run = lambda policy, method="estimate", threads=0, queries=q: np.asarray(LIB.run(
        s, v, queries, partition, alpha, forest, cases.SEED, decoder, params, method, cases.K,
        cases.FSEED, None, threads, policy)[1])
    coarse, nan = run("coarsen"), run("nan")
    labels = np.asarray(LIB.cells(s, np.vstack([q, s]), partition, alpha, forest, cases.SEED))
    at_q, at_s = labels[:len(q)], labels[len(q):]
    empty = np.array([~np.isin(at_q[:, t], at_s[:, t]) for t in range(forest)]).T
    assert empty.any() and not np.isnan(coarse).any()
    assert np.array_equal(coarse[~empty], nan[~empty])
    if decoder == "cellmean":
        for t in range(forest):
            means = {c: v[at_s[:, t] == c].mean() for c in np.unique(at_s[:, t])}
            for c in np.unique(at_q[empty[:, t], t]):
                values = coarse[at_q[:, t] == c, t]
                if partition.startswith("mondrian"):
                    assert len(set(values)) == 1
                else:
                    assert all(np.isclose(x, list(means.values()), atol=1e-6).any() for x in values)
    assert np.array_equal(coarse, run("coarsen", threads=1))
    for method in ("loo", "kfold"):
        assert not np.isnan(run("coarsen", method, queries=s)).any()


def test_left_out_counts_the_data_a_scorer_cannot_use():
    """A datum is left out of a score when it has fewer valid members than the scorer needs (1 for
    the absolute and squared errors, 30 for the NLL, 2 for the CRPS); the warning of the searches
    lists every configuration over the threshold and marks the best."""
    import warnings
    from spatialize.gs.esi import scorefunction as sf
    samples = np.ones((4, 40))
    samples[0, :] = np.nan            # no valid member
    samples[1, 1:] = np.nan           # one
    samples[2, 20:] = np.nan          # twenty
    assert sf.left_out(samples, sf.mae) == (0.25, (40 + 39 + 20) / 160)
    assert sf.left_out(samples, sf.crps)[0] == 0.5
    assert sf.left_out(samples, sf.neg_log_likelihood)[0] == 0.75
    assert sf.left_out(samples, lambda t, s: 0.0)[0] == 0.25      # a scorer without min_valid needs 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sf.warn_left_out([(0.0, 0.0), (0.2, 0.3), (0.5, 0.6)], ["a", "b", "c"], {1}, 0.05)
        sf.warn_left_out([(0.01, 0.02)], ["d"], {0}, 0.05)
    assert len(caught) == 1
    text = str(caught[0].message)
    assert "2 of 3" in text and "b: 20.0 %" in text and "<- the best" in text and "\n  a:" not in text


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


def test_posterior_audit_builds_each_law_from_the_other_data():
    """posterior_audit computes the same members as the 1.2 function given the data as queries; the
    law of a datum is fitted to its defined members alone, never to the datum; a cumulative
    distribution function is 0 below its grid and 1 above, so a datum far outside its law gets
    0 or 1 and not NaN."""
    from spatialize.gs.spa import posterior_audit, cv_sample_pred_posterior
    from spatialize.empirical import FittedModelFactory
    s, v, _ = DATA["2d"]
    v = v.copy()
    v[3] = v.max() + 10 * (v.max() - v.min())          # far above every other value
    kw = dict(local_interpolator="idw", exponent=2.0, n_partitions=cases.T, alpha=cases.ALPHA, seed=cases.SEED,
              callback=lambda *a, **k: None)
    audit = posterior_audit(s, v, fitted_model_factory=FittedModelFactory(), **kw)
    old = cv_sample_pred_posterior(s, v, s, fitted_model_factory=FittedModelFactory(), **kw)
    assert np.array_equal(audit.members, old.members, equal_nan=True)
    assert np.array_equal(audit.support, np.isfinite(audit.members).mean(axis=1))
    model = audit.model(3)
    assert np.array_equal(np.sort(model.data_), np.sort(audit.members[3][np.isfinite(audit.members[3])]))
    assert old.sample_quantiles[3] == 1.0
    assert model.cdf(model.x_[0] - 1.0) == 0.0 and model.pdf(model.x_[-1] + 1.0) == 0.0


def test_posterior_audit_readings_are_consistent():
    """The readings of the posterior audit agree with each other: the flags are those of the
    Benjamini–Hochberg procedure on the p-values; the levels follow the positions; the p-values
    never reach 0 and are 1 at most; the 1.2 ranking is the probability levels; the spread factor
    brings the 90 % coverage of the widened laws to nominal; every scale maps back to the values;
    every tail model gives a reading at every datum."""
    from spatialize.gs.spa import posterior_audit, cv_sample_pred_posterior
    from spatialize.gs.spa._main import _scale_maps, _fit_spread
    from spatialize.empirical import FittedModelFactory
    s, v, _ = DATA["2d"]
    v = np.exp(v - v.mean()).astype(np.float32)
    kw = dict(n_partitions=cases.T, alpha=cases.ALPHA, seed=cases.SEED, callback=lambda *a, **k: None)
    for options in ({}, {"tails": "gpd"}, {"tails": "normal", "calibrate": False}, {"scale": "yeojohnson"},
                    {"scale": "normal_scores"}):
        audit = posterior_audit(s, v, **kw, **options)
        t = audit.table()
        p = t.tail_p.to_numpy()
        assert np.all((p > 0) & (p <= 1))
        order = np.argsort(p)
        k = np.flatnonzero(p[order] <= 0.05 * np.arange(1, len(p) + 1) / len(p))
        expected = np.zeros(len(p), bool)
        if k.size:
            expected[order[:k[-1] + 1]] = True
        assert np.array_equal(t.flag.to_numpy(), expected)
        u = t.pit.to_numpy()
        outside99 = (u < 0.005) | (u > 0.995)
        assert np.array_equal(t.level.to_numpy() == "level_0", outside99)
        assert np.isfinite(t[["pit", "tail_p", "surprisal", "entropy"]].to_numpy()).all()
        fwd, inv = _scale_maps(audit.scale, audit.values)
        assert np.allclose(inv(fwd(audit.values)), audit.values, rtol=1e-6, atol=1e-9)
        if audit.calibrate:
            laws, z = audit._widened(), audit._tvalues
            c = audit.spread_factor
            pit = np.array([np.mean(np.median(x) + c * (x - np.median(x)) < zi) for x, zi in zip(laws, z)])
            assert abs(np.mean((pit >= 0.05) & (pit <= 0.95)) - 0.9) <= 2.0 / len(z)
    old = cv_sample_pred_posterior(s, v, s, fitted_model_factory=FittedModelFactory(), **kw)
    assert list(old.rank_samples().category) == list(old.levels())

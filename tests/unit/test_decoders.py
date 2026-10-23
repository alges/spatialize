"""Decoders: the identities between them and their independence from the units of the coordinates.

Unit tests on this checkout's in-place build (see tests/unit/conftest.py)."""
import numpy as np
import pytest

import cases
import snapshot_lib as sl

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
    assert np.nanmean(differ) < 0.02   # 1.2 % with the theory's Mondrian (2026-10-09), under 1 % before


@pytest.mark.parametrize("decoder", ["idw", "wdraw_idw"])
@pytest.mark.parametrize("method", ["estimate", "loo"])
def test_idw_decoders_do_not_depend_on_coordinate_units(decoder, method):
    """The IDW weights are 1/d^p relative to the nearest datum, so with a large exponent and
    coordinates 10^3 times smaller or larger the members stay finite and the same up to float rounding
    (1/d^p was infinite for small coordinates, every member NaN and every draw the cell's last datum,
    and 0 for large ones; issue #12)."""
    s, v, q = DATA["2d"]
    pts = s if method == "loo" else q
    runs = [np.asarray(LIB.run((s * c).astype(np.float32), v, (pts * c).astype(np.float32), "mondrian", cases.ALPHA,
                               cases.T, cases.SEED, decoder, {"exponent": 14.0}, method)[1]) for c in (1.0, 1e-3, 1e3)]
    for r in runs[1:]:
        assert np.array_equal(np.isnan(runs[0]), np.isnan(r))
        differ = np.abs(runs[0] - r) > 1e-4 * (1 + np.abs(runs[0]))
        assert np.nanmean(differ) < 0.01


def _ordinary_kriging(s, v, q, model, nugget, rng, sill):
    """Ordinary kriging with all the data, written out in numpy from the covariance of the model."""
    g = {1: lambda h: np.where(h >= 1, 1.0, 1.5 * h - 0.5 * h ** 3),
         2: lambda h: 1 - np.exp(-3 * h),
         3: lambda h: np.where(h >= 1, 1.0, 7 * h ** 2 - 35 * h ** 3 / 4 + 3.5 * h ** 5 - 0.75 * h ** 7),
         4: lambda h: 1 - np.exp(-3 * h ** 2)}[model]

    def cov(a, b):
        d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
        return np.where(d <= 0, sill, (sill - nugget) * (1 - g(d / rng)))

    n = len(s)
    lhs = np.ones((n + 1, n + 1))
    lhs[:n, :n], lhs[n, n] = cov(s, s), 0.0
    rhs = np.vstack([cov(s, q), np.ones((1, len(q)))])
    return np.linalg.solve(lhs, rhs)[:n].T @ v


@pytest.mark.parametrize("model", [1, 2, 3, 4])
@pytest.mark.parametrize("nugget", [0.0, 0.4])
def test_kriging_in_an_uncut_cell_is_ordinary_kriging(model, nugget):
    """A partition that never cuts leaves every datum in one cell, so its member is ordinary kriging
    with all the data. The nugget is the jump of the covariance at the origin, and spherical and
    cubic stay at the sill beyond the range (the range, 0.3, is short against the data's spread)."""
    gen = np.random.default_rng(0)
    s = gen.uniform(0, 1, (30, 2)).astype(np.float32)
    v = (np.sin(4 * s[:, 0]) + gen.normal(0, 0.3, 30)).astype(np.float32)
    q = gen.uniform(0, 1, (25, 2)).astype(np.float32)
    params = {"model": model, "nugget": nugget, "range": 0.3, "sill": 1.0}
    members = np.asarray(LIB.run(s, v, q, "mondrian", 0.0, 60, cases.SEED, "kriging", params, "estimate")[1])
    expected = _ordinary_kriging(s.astype(float), v.astype(float), q.astype(float), model, nugget, 0.3, 1.0)
    uncut = np.all(np.abs(members - expected[:, None]) < 1e-3, axis=0)
    assert uncut.sum() >= 5


def test_kriging_nugget_changes_the_prediction():
    s, v, q = DATA["2d"]
    run = [np.asarray(LIB.run(s, v, q, "mondrian", cases.ALPHA, 20, cases.SEED, "kriging",
                              {"model": 2, "nugget": n, "range": 0.3, "sill": 1.0}, "estimate")[1])
           for n in (0.0, 0.5)]
    assert np.nanmax(np.abs(run[0] - run[1])) > 1e-2


@pytest.mark.parametrize("decoder", ["adaptiveidw", "sharpidw", "wdraw_adaptiveidw", "wdraw_sharpidw"])
@pytest.mark.parametrize("method", ["loo", "kfold"])
def test_a_held_out_datum_takes_no_part_in_its_prediction(decoder, method):
    """The adaptive fit chooses its parameters by the cell's own errors, so a held-out prediction refits
    them without the held-out data: changing a held-out value leaves its own prediction unchanged."""
    s, v, q = DATA["2d"]
    changed = v.copy()
    changed[0] += 100.0                         # datum 0, held out in leave-one-out and in its fold
    run = [np.asarray(LIB.run(s, values, s, "mondrian", cases.ALPHA, 10, cases.SEED, decoder, {}, method,
                              cases.K, cases.FSEED)[1]) for values in (v, changed)]
    assert np.array_equal(run[0][0], run[1][0], equal_nan=True)


def _members(decoder, params, s, v, q, alpha=cases.ALPHA, trees=cases.T):
    return np.asarray(LIB.run(s, v, q, "mondrian", alpha, trees, cases.SEED, decoder, params, "estimate")[1])


def test_uniform_draws_coupled_by_cell_give_one_datum_per_cell():
    """Coupled by cell, the uniform draw gives every location of a cell the same datum:
    in each member, queries with the same cell mean (the same cell) take the same value. Drawn
    location by location, they do not."""
    s, v, q = DATA["2d"]
    cell = _members("cellmean", {}, s, v, q)
    coupled, located = _members("draw", {"coupling": "cell"}, s, v, q), _members("draw", {}, s, v, q)
    split = 0
    for t in range(cell.shape[1]):
        for m in np.unique(cell[~np.isnan(cell[:, t]), t]):
            same = cell[:, t] == m
            assert np.unique(coupled[same, t]).size == 1
            split += np.unique(located[same, t]).size > 1
    assert split > 0


@pytest.mark.parametrize("decoder, params", [("wdraw_idw", {"exponent": cases.EXP}),
                                             ("wdraw_kriging", {"model": 2, "nugget": 0.0, "range": 0.3, "sill": 1.0})])
def test_weighted_draws_coupled_by_cell_are_comonotone(decoder, params):
    """In an uncut cell, the draws coupled by cell read one level per tree on each location's weighted
    law, so the members are ordered alike at every location. (The adaptive weights are fitted again
    in each tree, so their laws differ between trees and the property holds within a tree only.)"""
    gen = np.random.default_rng(3)
    s = gen.uniform(0, 1, (30, 2)).astype(np.float32)
    v = (np.sin(4 * s[:, 0]) + gen.normal(0, 0.3, 30)).astype(np.float32)
    q = gen.uniform(0, 1, (25, 2)).astype(np.float32)
    cell = _members("cellmean", {}, s, v, q, alpha=0.0, trees=200)
    members = _members(decoder, dict(params, coupling="cell"), s, v, q, alpha=0.0, trees=200)
    uncut = members[:, np.all(np.abs(cell - v.mean()) < 1e-5, axis=0)]
    assert uncut.shape[1] >= 10
    d = uncut[:, :, None] - uncut[:, None, :]                  # [location, member, member]
    assert np.all(d[:, None] * d[None] >= -1e-6)               # same order at every pair of locations


@pytest.mark.parametrize("decoder, params", [("draw", {}), ("wdraw_idw", {"exponent": cases.EXP})])
def test_coupling_keeps_the_law_at_each_location(decoder, params):
    """The coupling changes the joint law within a cell, never the law at a location: over many trees
    the mean of the draws at each location is the weighted mean of its cells in both modes."""
    s, v, q = DATA["2d"]
    mean = {"draw": "cellmean", "wdraw_idw": "idw"}[decoder]
    expected = np.nanmean(_members(mean, params, s, v, q, trees=400), axis=1)
    spread = np.nanstd(v)
    for coupling in ("cell", "location"):
        draws = np.nanmean(_members(decoder, dict(params, coupling=coupling), s, v, q, trees=400), axis=1)
        assert np.nanmean(np.abs(draws - expected)) < 0.15 * spread

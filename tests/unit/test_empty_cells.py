"""The empty-cell policies: nan, mark and coarsen.

Unit tests on this checkout's in-place build (see tests/unit/conftest.py)."""
import numpy as np
import pytest

import cases
import snapshot_lib as sl

LIB = sl.load_lib()
DATA = cases.datasets()


@pytest.mark.parametrize("p_process,alpha,data_cond", [("mondrian", 0.9, True), ("mondrian-raw", 0.9, True),
                                                       ("voronoi", 0.8, True), ("voronoi", 0.8, False)])
@pytest.mark.parametrize("griddata", [False, True])
def test_empty_cell_fraction_is_the_share_of_nan_members(p_process, alpha, data_cond, griddata):
    """Under empty_cells="nan" a member is NaN exactly when its cell held no datum, so the share of
    NaN members at each location equals empty_cell_fraction(), read from the partitions alone."""
    from spatialize.gs.esi import esi_griddata, esi_nongriddata
    s, v, q = DATA["2d"]
    kw = dict(local_interpolator="idw", exponent=2.0, p_process=p_process, data_cond=data_cond, alpha=alpha,
              n_partitions=cases.T, seed=cases.SEED)
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

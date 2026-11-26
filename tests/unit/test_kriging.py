"""Plain kriging: the C++ engine behind spatialize.gs.kriging and libspatialize.cokriging."""
import numpy as np
import pytest

import libspatialize as lsp
from spatialize import SpatializeError
from spatialize.gs.kriging import kriging_griddata, kriging_nongriddata, kriging_hparams_search


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100, (200, 2))
    z = np.sin(pts[:, 0] / 15) + np.cos(pts[:, 1] / 20)
    return pts, z


def test_exact_interpolation_without_nugget(data):
    pts, z = data
    r = kriging_nongriddata(pts, z, pts[:20], range=40, nugget=0.0, radius=60)
    np.testing.assert_allclose(r.estimation(), z[:20], atol=1e-4)
    np.testing.assert_allclose(r.variance(), 0.0, atol=1e-4)


def test_grid_shapes_and_accuracy(data):
    pts, z = data
    xi = np.mgrid[10:90:20j, 10:90:20j]
    r = kriging_griddata(pts, z, xi, range=40, nugget=0.01, radius=60)
    truth = np.sin(xi[0] / 15) + np.cos(xi[1] / 20)
    assert r.estimation().shape == r.variance().shape == r.neighbors().shape == (20, 20)
    assert np.nanmean(np.abs(r.estimation() - truth)) < 0.05


def test_simple_kriging_uses_the_mean(data):
    pts, z = data
    far = np.array([[1e4, 1e4]])
    # the only datum in reach is none: the location stays missing
    assert np.isnan(kriging_nongriddata(pts, z, far, range=40, radius=60, kriging_type="simple").estimation()).all()
    # a constant field is reproduced whatever the mean
    c = np.full_like(z, 3.0)
    r = kriging_nongriddata(pts, c, pts[:5] + 0.5, range=40, radius=60, kriging_type="simple", mean=0.0, nugget=0.1)
    assert np.all(np.abs(r.estimation() - 3.0) < 1.0)
    r = kriging_nongriddata(pts, c, pts[:5] + 0.5, range=40, radius=60, kriging_type="ordinary", nugget=0.1)
    np.testing.assert_allclose(r.estimation(), 3.0, atol=1e-4)


def test_hparams_search_loo_and_kfold(data):
    pts, z = data
    for k in (5, "loo"):
        res = kriging_hparams_search(pts, z, None, k=k, radius=(20, 60), range=40, nugget=0.01, folding_seed=1)
        best = res.best_result()
        assert set(best) >= {"radius", "cv_error"} and best["cv_error"] < 0.2


def test_missing_range_is_an_error(data):
    pts, z = data
    with pytest.raises(SpatializeError):
        kriging_nongriddata(pts, z, pts[:3])


def _vg(n=2):
    sill = np.eye(n) * 0.5 + 0.1
    return {"nugget": np.eye(n) * 0.01, "model": [{"type": "spherical", "sill": sill, "ranges": [30, 30, 30], "angles": [0, 0, 0]}]}


def test_cokriging_multivariate_and_nan(data):
    pts, z = data
    p3 = np.c_[pts, np.zeros(len(pts))]
    v = np.c_[z, 2 * z]
    v[::7, 1] = np.nan
    params = dict(kriging_type="ordinary", nb_radius=[50, 50, 50], nb_angles=[0, 0, 0], nb_ndata=[3, 10], variogram=_vg())
    pred, var, nb = lsp.cokriging(p3, v, params, "scattered", p3[:10] + 0.3, None, 0, 0, None)
    assert pred.shape == var.shape == nb.shape == (10, 2)
    assert np.isfinite(pred).all() and (var >= -1e-6).all()


def test_cokriging_singular_system_flagged():
    p3 = np.zeros((4, 3))  # coincident data and no nugget: singular system
    params = dict(kriging_type="ordinary", nb_radius=[5, 5, 5], nb_angles=[0, 0, 0], nb_ndata=[1, 4], variogram={
        "nugget": [[0.0]], "model": [{"type": "spherical", "sill": [[1.0]], "ranges": [3, 3, 3], "angles": [0, 0, 0]}]})
    pred, var, nb = lsp.cokriging(p3, np.arange(4.0).reshape(-1, 1), params, "scattered", np.ones((1, 3)), None, 0, 0, None)
    assert nb[0, 0] == -1 and np.isnan(pred[0, 0])


def test_kdtree_ball_query_is_complete():
    """Every datum inside the search ball is a neighbour (the kd-tree must not lose any)."""
    rng = np.random.default_rng(3)
    p3 = np.c_[rng.uniform(0, 100, (500, 2)), np.zeros(500)]
    q = np.c_[rng.uniform(0, 100, (60, 2)), np.zeros(60)]
    params = dict(kriging_type="ordinary", nb_radius=[20, 20, 20], nb_angles=[0, 0, 0], nb_ndata=[1, 1000],
                  variogram={"nugget": [[0.01]], "model": [{"type": "spherical", "sill": [[1.0]], "ranges": [30, 30, 30], "angles": [0, 0, 0]}]})
    _, _, nb = lsp.cokriging(p3, np.ones((500, 1)), params, "scattered", q, None, 0, 0, None)
    expected = [(np.linalg.norm(p3 - x, axis=1) < 20).sum() for x in q]
    np.testing.assert_array_equal(nb[:, 0], expected)

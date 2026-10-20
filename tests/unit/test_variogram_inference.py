"""Variogram inference from the ensemble (spatialize.futures.variogram): the exact pieces of the reading.

Unit tests on this checkout's in-place build (see tests/unit/conftest.py)."""
import warnings

import numpy as np
import pytest


def _stripes(shape, spacing, direction, wavelength=0.25):
    """A map constant along ``direction`` (unit vector in physical coordinates), varying across it."""
    axes = [s * np.arange(n) for s, n in zip(spacing, shape)]
    coords = np.meshgrid(*axes, indexing="ij")
    d = np.asarray(direction, float)
    normal = np.zeros_like(d)
    normal[np.argmin(np.abs(d))] = 1.0
    normal -= (normal @ d) * d
    normal /= np.linalg.norm(normal)
    phase = sum(c * n for c, n in zip(coords, normal))
    return np.sin(2 * np.pi * phase / wavelength), tuple(coords)


def test_grid_geometry_follows_either_index_order_and_spacing():
    from spatialize.futures.variogram import _grid_geometry
    x, y = np.linspace(0, 1, 11), np.linspace(0, 3, 7)
    coord_of, step = _grid_geometry(np.meshgrid(x, y))                  # 'xy': rows follow y
    assert list(coord_of) == [1, 0] and np.allclose(step, [0.5, 0.1])
    coord_of, step = _grid_geometry(np.meshgrid(x, y, indexing="ij"))
    assert list(coord_of) == [0, 1] and np.allclose(step, [0.1, 0.5])


@pytest.mark.parametrize("angle", [0.0, 30.0, 75.0, 120.0])
def test_structure_tensor_reads_the_direction_of_stripes_in_2d(angle):
    """On stripes the principal direction of continuity is the stripes' own, whatever the grid's spacing."""
    from spatialize.futures.variogram import _grid_geometry, _principal_axes, _structure_tensor
    u = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    z, xi = _stripes((80, 60), (0.0125, 0.02), u)
    coord_of, step = _grid_geometry(xi)
    axes, coherence = _principal_axes(_structure_tensor(z, coord_of, step, 2.0))
    assert np.degrees(np.arccos(min(1.0, abs(axes[0] @ u)))) < 2.0
    assert coherence > 0.9


def test_structure_tensor_reads_the_direction_of_stripes_in_3d():
    from spatialize.futures.variogram import _grid_geometry, _principal_axes, _structure_tensor
    u = np.array([1.0, 1.0, 0.5])
    u /= np.linalg.norm(u)
    z, xi = _stripes((30, 30, 30), (1 / 30, 1 / 30, 1 / 30), u)
    coord_of, step = _grid_geometry(xi)
    axes, _ = _principal_axes(_structure_tensor(z, coord_of, step, 1.0))
    normal = axes[2]                                                   # the direction of variation
    assert abs(normal @ u) < 0.05                                      # orthogonal to the stripes' direction


def test_efold_interpolates_the_crossing_and_flags_a_lower_bound():
    from spatialize.futures.variogram import _efold
    lags = np.linspace(0.01, 0.5, 50)
    a = 0.137
    length, censored = _efold(lags, 1.0 - np.exp(-lags / a), 1.0)
    assert abs(length - a) < 2e-3 and not censored
    length, censored = _efold(lags, 1.0 - np.exp(-lags / 5.0), 1.0)
    assert censored and length == lags[-1]


def test_directional_semivariogram_is_zero_along_stripes():
    from spatialize.futures.variogram import _directional_semivariogram, _grid_geometry, _lags
    u = np.array([np.cos(np.radians(30)), np.sin(np.radians(30))])
    z, xi = _stripes((64, 64), (1 / 64, 1 / 64), u)
    coord_of, step = _grid_geometry(xi)
    lags = _lags(z.shape, step)[:10]
    along = _directional_semivariogram(z, u, coord_of, step, lags)
    across = _directional_semivariogram(z, np.array([-u[1], u[0]]), coord_of, step, lags)
    assert np.nanmax(along) < 0.02 * np.nanmax(across)


def test_read_variogram_needs_a_grid():
    from spatialize.futures.variogram import read_variogram
    from spatialize.gs.esi import ESIResult
    m = np.zeros((5, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            read_variogram(ESIResult(m.mean(axis=1), m, xi=np.zeros((5, 2))))


def test_lengths_are_read_at_a_level():
    """Each member's length is where its correlation first falls below the level, linearly interpolated: on
    a correlation that starts at 1 - s, reading at e^-1 gives a shorter length than reading at (1 - s) e^-1."""
    from spatialize.futures.variogram import _lengths_at
    lags = np.linspace(0.01, 0.6, 60)
    a, s = 0.12, 0.3
    gamma = 1.0 - (1.0 - s) * np.exp(-lags / a)          # a member's semivariogram with a nugget, unit variance
    at_e, _ = _lengths_at([gamma], [1.0], lags, np.exp(-1.0))
    below, _ = _lengths_at([gamma], [1.0], lags, np.exp(-1.0) * (1 - s))
    assert at_e[0] < a and abs(below[0] - a) < 2e-3


def test_read_variogram_reports_the_residual_weight_on_the_grid():
    """On a small estimate the reading returns directions, lengths and the residual weight at every node,
    the share of partitions whose cell holds no datum, within [0, 1]."""
    import spatialize
    from spatialize.futures.variogram import read_variogram
    from spatialize.gs.esi import esi_griddata
    from spatialize.logging import singleton_null_callback as quiet
    rng = np.random.default_rng(5)
    p = rng.random((150, 2))
    v = np.sin(6 * p[:, 0] + 2 * p[:, 1])
    g = np.linspace(0.02, 0.98, 24)
    xi = tuple(np.meshgrid(g, g))
    with spatialize.session.override(empty_cells="coarsen"):
        r = esi_griddata(p, v, xi, local_interpolator="wdraw_idw", exponent=2.0, alpha=0.8, n_partitions=30,
                         seed=1, callback=quiet)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = read_variogram(r)
    assert out.residual_weight.shape == (24, 24)
    assert np.all((out.residual_weight >= 0) & (out.residual_weight <= 1))
    assert out.lengths.shape == (2,) and np.all(out.lengths > 0)


@pytest.mark.parametrize("tau", [0.0, 0.3])
def test_error_bound_is_exact_on_a_constant_field_with_error(tau):
    """On a constant field the data differ only by their errors, so the within-cell variance is the error's
    variance at every granularity: the bound recovers it; on pure noise it is the whole variance."""
    from spatialize.futures.variogram import _error_bound
    rng = np.random.default_rng(8)
    p = rng.random((400, 2))
    v = 5.0 + tau * rng.standard_normal(400)
    bound, se, alpha, dof = _error_bound(p, v, p)
    realised = np.var(v, ddof=1)                         # the variance of this sample's errors
    assert abs(bound - realised) <= 0.15 * realised + 1e-12   # the minimum over granularities dips a little
    noise = rng.standard_normal(400)
    assert abs(_error_bound(p, noise, p)[0] - np.var(noise, ddof=1)) < 0.1

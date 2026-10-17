"""Plots: values at scattered locations become images with rows along y and columns along x."""
import numpy as np
import pytest


def test_scattered_values_become_an_image_whatever_their_order():
    """Values on a regular grid of locations give the same image in any order of the locations; a
    w/h that disagrees with the grid is ignored; locations that do not fill a grid give no image,
    unless w and h reshape the values; w by h must hold the values."""
    from spatialize import SpatializeError
    from spatialize.viz import _as_image
    x, y = np.meshgrid(np.arange(5.0), np.arange(3.0))          # 5 columns (x) by 3 rows (y)
    xi = np.column_stack([x.ravel(), y.ravel()])
    z = 10 * xi[:, 1] + xi[:, 0]
    expected = z.reshape(3, 5)
    order = np.random.default_rng(0).permutation(len(z))
    assert np.array_equal(_as_image(z[order], xi_locations=xi[order]), expected)
    assert np.array_equal(_as_image(z, 5, 3, xi), expected)
    assert np.array_equal(_as_image(z, 3, 5, xi), expected)           # swapped: the locations decide
    assert _as_image(z[:-1], xi_locations=xi[:-1]) is None            # not a full grid
    assert _as_image(z[:-1], 7, 2, xi[:-1]).shape == (2, 7)           # w and h reshape it as given
    with pytest.raises(SpatializeError):
        _as_image(z, 4, 4)


def test_plot_nongriddata_lays_values_on_their_lattice():
    """Locations on part of a lattice land on their nodes, the others left empty, without xarray."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from spatialize.viz import plot_nongriddata
    locations = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
    fig, ax = plt.subplots()
    plot_nongriddata(np.array([1.0, 2.0, 3.0, 4.0]), locations, ax=ax, show_colorbar=False)
    image = np.ma.filled(ax.images[0].get_array().astype(float), np.nan)     # [y, x]
    expected = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, 4.0]])
    assert np.array_equal(image, expected, equal_nan=True)
    plt.close(fig)


def test_quick_plot_of_simulations_on_a_grid():
    """The simulated fields of a gridded result are drawn on the grid (they were drawn as points with
    coordinates taken from the grid itself, and failed, before version 1.3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from spatialize.empirical import FittedModelFactory
    from spatialize.gs.esi import esi_griddata
    from spatialize.gs.ess import ess_sample
    from spatialize.logging import singleton_null_callback as quiet
    rng = np.random.default_rng(0)
    xi = np.mgrid[0:10, 0:12].astype(float)
    result = esi_griddata(rng.random((60, 2)) * 9, rng.normal(size=60), xi, local_interpolator="idw", exponent=2.0,
                          n_partitions=20, alpha=0.8, seed=0)
    ess_sample(result, n_sims=4, fitted_model_factory=FittedModelFactory(seed=0), callback=quiet).quick_plot(n_imgs=4, n_cols=2)
    assert plt.gcf().axes[0].images[0].get_array().shape == (12, 10)
    plt.close("all")

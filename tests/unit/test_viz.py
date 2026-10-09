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

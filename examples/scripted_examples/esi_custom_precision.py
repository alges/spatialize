import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from spatialize import logging
import spatialize.gs.esi.functionals as fn
from spatialize.gs.esi import esi_griddata
from spatialize.gs.esi.loss import loss


logging.log.setLevel("DEBUG")


def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2

# grid definition
grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

# data points and values creation for estimation functions
rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])

# custom precision function declaration
def op_error_precision(estimation, esi_samples):
    dyn_range = np.abs(np.nanmin(esi_samples) - np.nanmax(esi_samples))

    @loss(fn.mean)
    def _op_error(x, y):
        return np.abs(x - y) / dyn_range

    return _op_error(estimation, esi_samples)

# plot function that includes original function, estimation,
# mse precision and custom loss function - op_error_precision
def plot_result(result, title):
    grid_cmap, prec_cmap = 'coolwarm', 'bwr'
    fig = plt.figure(figsize=(10,3), dpi=150)
    gs = fig.add_gridspec(1, 4, wspace=1.0)
    (ax1, ax2, ax3, ax4) = gs.subplots()

    # plot original
    img1 = ax1.imshow(func(grid_x, grid_y).T, origin='lower', cmap=grid_cmap)
    ax1.set_title("original")
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.1)
    colorbar(img1, orientation='vertical', cax=cax1)

    # plot estimation
    result.plot_estimation(ax=ax2, cmap=grid_cmap)
    ax2.set_title(title)

    # plot the default mse precision
    result.plot_precision(ax=ax3, cmap=prec_cmap)
    ax3.set_title('mse')
    ax3.plot(points[:, 0], points[:, 1], 'y.', ms=0.5)

    # plot a custom precision
    # here the custom precision (loss function) is passed
    result.precision(op_error_precision)
    result.plot_precision(ax=ax4, cmap=prec_cmap)
    ax4.set_title('op error')
    ax4.plot(points[:, 0], points[:, 1], 'y.', ms=0.5)

    plt.show()


# estimation functions encapsulation
# to be able to use only one of them at each runtime
# if needed
def esi_idw():
    result = esi_griddata(points, values, (grid_x, grid_y),
                          local_interpolator="idw",
                          exponent=1.0,
                          n_partitions=500, alpha=0.98,
                          agg_function=fn.median)

    plot_result(result, 'esi idw')


def esi_kriging():
    result = esi_griddata(points, values, (grid_x, grid_y),
                          local_interpolator="kriging",
                          model="spherical", nugget=0.0, range=10.0,
                          n_partitions=100, alpha=0.9,
                          agg_function=fn.median)

    plot_result(result, 'esi kriging')


if __name__ == '__main__':
    esi_idw()
    esi_kriging()

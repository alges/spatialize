import numpy as np
from rich import print
from matplotlib import pyplot as plt
from matplotlib.pyplot import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from spatialize import logging
import spatialize.gs.esi.aggfunction as af
from spatialize.gs.esi import esi_hparams_search, esi_griddata
from spatialize.gs.idw import idw_hparams_search, idw_griddata

logging.log.setLevel("DEBUG")

grid_cmap, prec_cmap = 'coolwarm', 'bwr'


def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2


# grid points creation
grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

# data samples to input the estimations
rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])

# running the idw (not esi) grid search
search_result = idw_hparams_search(points, values, (grid_x, grid_y),
                                   griddata=True, k=-1,
                                   radius=[0.07, 0.08],
                                   exponent=(0.001, 0.01, 0.1, 1, 2))

# printing the params and the results (two options)
print(search_result.search_result_data)
print(search_result.best_params)
print(search_result.best_result(optimize_data_usage=False))

# plot the cross validation error
search_result.plot_cv_error()

# using grid search best result to estimate with idw (not esi)
result_idw = idw_griddata(points, values, (grid_x, grid_y),
                          best_params_found=search_result.best_result(optimize_data_usage=False))

# *** idw as local interpolator ***

# run the grid search
search_result = esi_hparams_search(points, values, (grid_x, grid_y),
                                   local_interpolator="idw", griddata=True, k=10,
                                   p_process="voronoi",
                                   n_partitions=[100],
                                   exponent=[0.001, 0.01, 0.1, 1, 2],
                                   alpha=(0.95, 0.97, 0.98, 0.985),
                                   seed=1500)

search_result.plot_cv_error()
plt.show()

result_esi_idw = esi_griddata(points, values, (grid_x, grid_y),
                              best_params_found=search_result.best_result()
                              )

# *** kriging as local interpolator ***
search_result = esi_hparams_search(points, values, (grid_x, grid_y),
                                   local_interpolator="kriging", griddata=True, k=10,
                                   model=["spherical", "exponential", "cubic", "gaussian"],
                                   n_partitions=[100],
                                   nugget=[0.0, 0.5, 1.0],
                                   range=[10.0, 50.0, 100.0, 200.0],
                                   alpha=[0.97, 0.96, 0.95],
                                   seed=1500)

search_result.plot_cv_error()
plt.show()

result_esi_kriging = esi_griddata(points, values, (grid_x, grid_y),
                                  best_params_found=search_result.best_result()
                                  )

# plot original, results and precisions
fig = plt.figure(dpi=150, figsize=(15, 10))
gs = fig.add_gridspec(2, 3, wspace=0.3)
(ax1, ax2) = gs.subplots()
ax1, ax2, ax3, ax4, ax5, ax6 = ax1[0], ax1[1], ax1[2], ax2[0], ax2[1], ax2[2]

ax1.set_aspect('equal')
ax1.set_title('idw')
img1 = result_idw.plot_estimation(ax1)

ax2.set_aspect('equal')
ax2.set_title('esi idw')
img2 = result_esi_idw.plot_estimation(ax2)

ax3.set_aspect('equal')
ax3.set_title('esi kriging')
img3 = result_esi_kriging.plot_estimation(ax3)

ax4.set_aspect('equal')
ax4.set_title('original data')
img4 = ax4.imshow(func(grid_x, grid_y).T, origin='lower', cmap=grid_cmap)
divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="5%", pad=0.1)
colorbar(img4, orientation='vertical', cax=cax)

ax5.set_title('mse esi idw')
ax5.plot(points[:, 0], points[:, 1], 'y.', ms=0.5)
img5 = result_esi_idw.plot_precision(ax=ax5, cmap=prec_cmap)

ax6.set_title('mse esi kriging')
ax6.plot(points[:, 0], points[:, 1], 'y.', ms=0.5)
img6 = result_esi_kriging.plot_precision(ax=ax6, cmap=prec_cmap)

plt.show()

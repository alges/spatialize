import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from spatialize import logging
from spatialize.data import load_drill_holes_andes_2D
from spatialize.gs.esi import esi_hparams_search, esi_nongriddata
import spatialize.gs.esi.lossfunction as lf

# for a more explanatory output of the spatialize functions
logging.log.setLevel("DEBUG")

# the samples included in the spatialize package
samples, locations, krig, _ = load_drill_holes_andes_2D()

# estimation data and result shape
w, h = 200, 300   # columns (distinct x) and rows (distinct y) of the location grid

# input variables for non gridded estimation spatialize functions
points = samples[['x', 'y']].values
values = samples[['cu']].values[:, 0]
xi = locations[['x', 'y']].values

# kriging estimation example result
krig_im = krig[['est_cu_case_esipaper']].values[:, 0].reshape(h, w)

# plotting original data along with the kriging estimation example
fig = plt.figure(dpi=150, figsize=(10, 5))
gs = fig.add_gridspec(1, 2, wspace=0.4)
(ax1, ax2) = gs.subplots()

ax1.set_aspect('equal')
ax1.set_title('original data')
img1 = ax1.scatter(x=points[:, 0], y=points[:, 1], c=values, cmap='coolwarm')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.1)
colorbar(img1, orientation='vertical', cax=cax)

ax2.set_aspect('equal')
ax2.set_title('ordinary kriging')
img2 = ax2.imshow(krig_im, origin='lower', cmap='coolwarm')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)
colorbar(img2, orientation='vertical', cax=cax)

plt.show()

# operational error function (loss function to be used) for the observed dynamic range
op_error = lf.OperationalErrorLoss(np.abs(np.nanmin(values) - np.nanmax(values)))

# esi_idw function definition that allows to choose the partition process
# this function executes a gridded search, then it estimates with the best parameters
def esi_idw(p_process):
    search_result = esi_hparams_search(points, values, xi,
                                       local_interpolator="idw", griddata=False, k=10,
                                       p_process=p_process,
                                       exponent=list(np.arange(1.0, 15.0, 1.0)),
                                       alpha=(0.5, 0.6, 0.8, 0.9, 0.95, 0.98),
                                       seed=1500)

    search_result.plot_cv_error()
    plt.show()

    result = esi_nongriddata(points, values, xi,
                             local_interpolator="idw",
                             p_process=p_process,
                             n_partitions=500,
                             best_params_found=search_result.best_result())
    print(result)
    result.precision(op_error)
    result.quick_plot(w=w, h=h, figsize=(10, 5))
    plt.show()

# esi_kriging function definition that executes a gridded search
# then it estimates with the best parameters
def esi_kriging():
    search_result = esi_hparams_search(points, values, xi,
                                       local_interpolator="kriging", griddata=False, k=10,
                                       model=["spherical", "exponential", "cubic", "gaussian"],
                                       nugget=[0.5, 1.0],
                                       range=[100.0, 500.0, 1000.0],
                                       alpha=list(np.flip(np.arange(0.90, 0.95, 0.01))),
                                       sill=[0.9, 1.0, 1.1],
                                       seed=1500)

    search_result.plot_cv_error()
    plt.show()

    result = esi_nongriddata(points, values, xi,
                             local_interpolator="kriging",
                             n_partitions=500,
                             best_params_found=search_result.best_result())
    print(result)
    result.precision(op_error)
    result.quick_plot(w=w, h=h, figsize=(10, 5))
    plt.show()

# this code allows to run both defined functions and to execute both partition processes
# for the esi_idw case ('voronoi' and 'mondrian')
if __name__ == '__main__':
    #esi_kriging()
    esi_idw("mondrian")
    #esi_idw("voronoi")

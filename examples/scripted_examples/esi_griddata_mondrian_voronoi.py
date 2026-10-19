import numpy as np
from matplotlib import pyplot as plt

import spatialize.gs.esi.functionals as fn
from spatialize import logging
from spatialize.gs.esi import esi_griddata

logging.log.setLevel("DEBUG")


def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2

# grid points definition
grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

# random sample of 1000 data points from "cubic" function
rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])


grid_cmap, prec_cmap = 'coolwarm', 'bwr'

# estimation with esi-idw and mondrian partition method
result = esi_griddata(points, values, (grid_x, grid_y),
                      local_interpolator="idw",
                      p_process="mondrian",
                      data_cond=False,
                      exponent=1.0,
                      n_partitions=500, alpha=0.985,
                      agg_function=fn.mean
                      )

esi_idw_mond = result.estimation()

# estimation with esi-idw and voronoi partition method
# with data conditioning
result_2 = esi_griddata(points, values, (grid_x, grid_y),
                      local_interpolator="idw",
                      p_process="voronoi",
                      data_cond=True,
                      exponent=1.0,
                      n_partitions=500, alpha=0.985,
                      agg_function=fn.mean
                      )

esi_idw_vor_t = result_2.estimation()

# estimation with esi-idw and voronoi partition method
# withouth data conditioning
result_3 = esi_griddata(points, values, (grid_x, grid_y),
                      local_interpolator="idw",
                      p_process="voronoi",
                      data_cond=False,
                      exponent=1.0,
                      n_partitions=500, alpha=0.985,
                      agg_function=fn.mean
                      )

esi_idw_vor_f = result_3.estimation()

# plotting original "cubic" function along with the three estimations with esi
fig = plt.figure(figsize=(10,5), dpi=150)
gs = fig.add_gridspec(1, 4, wspace=0.5, hspace=0.47)
(ax1, ax2, ax3, ax4) = gs.subplots()

# plot original
ax1.imshow(func(grid_x, grid_y).T, origin='lower', cmap=grid_cmap)
ax1.set_title("original")

ax2.imshow(esi_idw_mond.T, origin='lower', cmap=grid_cmap)
ax2.set_title("esi idw mondrian")

ax3.imshow(esi_idw_vor_f.T, origin='lower', cmap=grid_cmap)
ax3.set_title("esi idw voronoi dc=f")

ax4.imshow(esi_idw_vor_t.T, origin='lower', cmap=grid_cmap)
ax4.set_title("esi idw voronoi dc=t")

plt.show()
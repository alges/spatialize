import numpy as np
from matplotlib import pyplot as plt

import spatialize.gs.esi.functionals as fn
from spatialize import logging
from spatialize.gs.esi import esi_griddata
from scipy.interpolate import griddata

logging.log.setLevel("DEBUG")


def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2

# grid generation
grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

# random points to sample original function and to generate points and values
rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])

grid_cmap, prec_cmap = 'coolwarm', 'bwr'

# plot with original function and generated points
fig0 = plt.figure(dpi=300, figsize=(5, 5))
plt.scatter(points[:, 0]*100, points[:, 1]*200, s=0.5, color='magenta')
plt.imshow(func(grid_x, grid_y).T, origin='lower', cmap=grid_cmap)
plt.title('original and data points')
plt.show()

# estimation with SciPy library function to be compared with esi
grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')


# esi estimations with reasonable parameters
result = esi_griddata(points, values, (grid_x, grid_y),
                      local_interpolator="idw",
                      p_process="mondrian",
                      data_cond=False,
                      exponent=1.0,
                      n_partitions=500, alpha=0.985,
                      agg_function=fn.mean
                      )

print(result)

esi_idw_est = result.estimation()

result_2 = esi_griddata(points, values, (grid_x, grid_y),
                        local_interpolator="kriging",
                        model="spherical", nugget=0.0, range=10.0,
                        n_partitions=500, alpha=0.9,
                        agg_function=fn.mean)

print(result_2)

esi_krig_est = result_2.estimation()

# plotting 6 examples all together
fig = plt.figure(dpi=150, figsize=(10, 10))
gs = fig.add_gridspec(2, 3, wspace=0.1, hspace=0.47)
(ax1, ax2) = gs.subplots()
ax1, ax2, ax3, ax4, ax5, ax6 = ax1[0], ax1[1], ax1[2], ax2[0], ax2[1], ax2[2]

ax1.imshow(func(grid_x, grid_y).T, origin='lower', cmap=grid_cmap)
ax1.set_title("original")

ax2.imshow(esi_idw_est.T, origin='lower', cmap=grid_cmap)
ax2.set_title("esi idw")

ax3.imshow(esi_krig_est.T, origin='lower', cmap=grid_cmap)
ax3.set_title("esi kriging")

ax4.imshow(grid_z0.T, origin='lower', cmap=grid_cmap)
ax4.set_title("nearest")

ax5.imshow(grid_z1.T, origin='lower', cmap=grid_cmap)
ax5.set_title("linear")

ax6.imshow(grid_z2.T, origin='lower', cmap=grid_cmap)
ax6.set_title("cubic")

plt.show()

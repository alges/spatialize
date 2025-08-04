from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from spatialize import logging
import spatialize.gs.esi.aggfunction as af
from spatialize.gs.esi import esi_nongriddata
from spatialize.data import load_drill_holes_andes_3D

SHOW_PLOT = False

# get the data
samples, locations = load_drill_holes_andes_3D()
locations = locations.sort_values(["z", "y", "x"])

logging.log.setLevel("DEBUG")

#input variables for estimation
points = samples[['x', 'y', 'z']].values
x, y, z = samples[['x']].values, samples[['y']].values, samples[['z']].values
values = samples[['cu']].values[:, 0]

#3d locations for estimation values
xi = locations[['x', 'y', 'z']].values
grid_x, grid_y, grid_z = locations[['x']].values, locations[['y']].values, locations[['z']].values

#estimation function
result = esi_nongriddata(points, values, xi,
                      local_interpolator="idw",
                      p_process="mondrian",
                      exponent=1.0,
                      n_partitions=100,
                      alpha=0.7,
                      agg_function=af.mean
                      )

#plotting original data and resulting estimation points
fig = plt.figure(dpi=150, figsize=(10,5))
G = gridspec.GridSpec(1, 2)

ax1 = fig.add_subplot(G[0,0], projection='3d')
ax1.set_title('3d samples')
ax1.scatter3D(
        x, y, z,
        c=values,
        cmap='coolwarm'
)

ax2 = fig.add_subplot(G[0,1], projection='3d')
ax2.set_title('esi idw 3d estimation')
ax2.scatter3D(
        grid_x, grid_y, grid_z,
        c=result.estimation(),
        cmap='coolwarm'
)

if SHOW_PLOT:
        plt.show()

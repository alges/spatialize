from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import colorbar

from spatialize.data import load_drill_holes_andes_2D
from pykrige.ok import OrdinaryKriging
from pykrige.rk import Krige
from sklearn.model_selection import GridSearchCV


# get the data
samples, locations, krig, _ = load_drill_holes_andes_2D()
locations = locations.sort_values(["z", "y", "x"])

# estimation data and result shape
w, h = 200, 300   # columns (distinct x) and rows (distinct y) of the location grid

# input variables
points = samples[['x', 'y']].values.astype('float32')
x, y = samples[['x']].values, samples[['y']].values

values = samples[['cu']].values[:, 0].astype('float32')

xi = locations[['x', 'y']].values
xi_x, xi_y = locations[['x']].values, locations[['y']].values

# load previously calculated manual expert kriging results
krig_im = krig[['est_cu_case_esipaper']].values[:, 0].reshape(h, w)

# search parameters for the best variogram model
param_dict = {
    "variogram_model": ["linear", "gaussian", "spherical", "exponential"]
}

estimator = GridSearchCV(Krige(), param_dict,
                         verbose=True, return_train_score=True)

# run the gridsearch
estimator.fit(X=points, y=values)

if hasattr(estimator, "best_score_"):
    print("best_score R² = {:.3f}".format(estimator.best_score_))
    print("best_params = ", estimator.best_params_)

# run the ordinary kriging
krig = OrdinaryKriging(x, y, values, **estimator.best_params_)
estimate, _ = krig.execute("points", xi_x, xi_y)

# plot results
fig = plt.figure(dpi=150, figsize=(10,5))
gs = fig.add_gridspec(1, 2, wspace=0.4)
(ax1, ax2) = gs.subplots()

ax1.set_aspect('equal')
ax1.set_title('ordinary kriging')
img1 = ax1.imshow(krig_im, origin='lower', cmap='coolwarm')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.1)
colorbar(img1, orientation='vertical', cax=cax)

ax2.set_aspect('equal')
ax2.set_title('pykgrige automated o. kriging')
im = estimate.reshape(h, w)
img = ax2.imshow(im, origin='lower', cmap='coolwarm')
divider = make_axes_locatable(ax2)

cax = divider.append_axes("right", size="5%", pad=0.1)
colorbar(img, orientation='vertical', cax=cax)

plt.show()
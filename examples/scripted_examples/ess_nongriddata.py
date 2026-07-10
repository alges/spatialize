import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from spatialize import logging
from spatialize.data import load_simulated_anisotropic_data, load_result, save_result
from spatialize.gs.ess import ess_sample
from spatialize.gs.esi import esi_nongriddata
from spatialize.gs.ess._main import ess_sample
from spatialize.empirical import FittedModelFactory
from spatialize.viz import plot_colormap_data

# for a more explanatory output of the spatialize functions
logging.log.setLevel("INFO")

# the samples included in the spatialize package
_, _, data_reduced, ground_truth = load_simulated_anisotropic_data()

# the dataset we want to use
# this is the 10th simulation of the anisotropic data (out of 1000)
using_sim = 'sim10'

# directory where the model is saved and loaded from
# this is the model trained on 5% of the data
# and 200 partitions
model_dir_path = "./adaptive_idw_reduced_200_partitions"

# input variables for non gridded estimation spatialize functions
points = data_reduced[0][['x', 'y']].values
values = data_reduced[0][[using_sim]].values[:, 0]
xi = ground_truth[['x', 'y']].values

adaptive_esi_idw_n_partitions = 200
n_sims = 1000


def adaptive_esi_idw():
    result = esi_nongriddata(points, values, xi,
                             local_interpolator="adaptiveidw",
                             n_partitions=adaptive_esi_idw_n_partitions,
                             alpha=0.3)

    result.quick_plot(figsize=(10, 5))
    return result


if __name__ == '__main__':
    # the ESI stage
    result = adaptive_esi_idw()
    result.quick_plot(figsize=(10, 5))

    # the ESS stage
    # case 1
    sim_result = ess_sample(esi_result=result,
                            n_sims=n_sims,
                            fitted_model_factory=FittedModelFactory(
                                 point_model_name="kde",
                                 kernel="tophat",
                            ))
    seed = sim_result.quick_plot(n_imgs=9, n_cols=3, norm_lims=False,
                                 title="Point model: kde (tophat)",
                                 title_prefix="Scenario")

    # case 2
    sim_result = ess_sample(esi_result=result,
                            n_sims=n_sims,
                            fitted_model_factory=FittedModelFactory(
                                 point_model_name="kde",
                                 kernel="gaussian",
                            ))
    seed = sim_result.quick_plot(n_imgs=9, n_cols=3, norm_lims=False,
                                 title="Point model: kde (gaussian)",
                                 title_prefix="Scenario")

    # case 3
    sim_result = ess_sample(esi_result=result,
                            n_sims=n_sims,
                            fitted_model_factory=FittedModelFactory(
                                 point_model_name="emm",
                                 n_components=3,
                            ))
    seed = sim_result.quick_plot(n_imgs=9, n_cols=3, norm_lims=False,
                                 title="Point model: GMM (EM - 3 components)",
                                 title_prefix="Scenario")

    # case 4
    sim_result = ess_sample(esi_result=result,
                            n_sims=n_sims,
                            fitted_model_factory=FittedModelFactory(
                                 point_model_name="vim",
                                 n_components=3,
                            ))
    seed = sim_result.quick_plot(n_imgs=9, n_cols=3, norm_lims=False,
                                 title="Point model: GMM (VI - 3 components)",
                                 title_prefix="Scenario")
    plt.show()

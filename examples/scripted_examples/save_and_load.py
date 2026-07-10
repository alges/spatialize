import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from spatialize import logging
from spatialize.data import load_simulated_anisotropic_data, load_result, save_result
from spatialize.gs.ess import ess_sample
from spatialize.gs.esi import esi_nongriddata
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


def adaptive_esi_idw():
    result = esi_nongriddata(points, values, xi,
                             local_interpolator="adaptiveidw",
                             n_partitions=adaptive_esi_idw_n_partitions,
                             alpha=0.3)

    result.quick_plot(figsize=(10, 5))
    return result

if __name__ == '__main__':
    # loading an ESIResult
    try:
        result = load_result(model_dir_path, just_esi_result=True)
    except FileNotFoundError:
        result = adaptive_esi_idw()
        save_result(model_dir_path, result)

    result.quick_plot(figsize=(10, 5))

    # loading sumulations
    try:
        sim_results = load_result(model_dir_path)

        for sim_result in sim_results:
            seed = sim_result.quick_plot(n_imgs=9, n_cols=3, norm_lims=False, title_prefix="Scenario")
    except Exception as e:
        n_sims = 10
        sim_result = ess_sample(esi_result=result, n_sims=n_sims, fitted_model_factory=FittedModelFactory(
                                     point_model_name="kde",
                                     kernel="gaussian",
                                 ))
        save_result(model_dir_path, sim_result)

        sim_result = ess_sample(esi_result=result, n_sims=n_sims, fitted_model_factory=FittedModelFactory(
                                     point_model_name="kde",
                                     kernel="tophat",
                                 ))
        save_result(model_dir_path, sim_result)

        sim_result = ess_sample(esi_result=result, n_sims=n_sims, fitted_model_factory=FittedModelFactory(
                                     point_model_name="emm",
                                     n_components=1,
                                 ))
        save_result(model_dir_path, sim_result)

        sim_result = ess_sample(esi_result=result, n_sims=n_sims, fitted_model_factory=FittedModelFactory(
                                     point_model_name="emm",
                                     n_components=3,
                                 ))
        save_result(model_dir_path, sim_result)

        sim_result = ess_sample(esi_result=result, n_sims=n_sims, fitted_model_factory=FittedModelFactory(
                                     point_model_name="vim",
                                     n_components=1,
                                 ))
        save_result(model_dir_path, sim_result)

        sim_result = ess_sample(esi_result=result, n_sims=n_sims, fitted_model_factory=FittedModelFactory(
                                     point_model_name="vim",
                                     n_components=3,
                                 ))
        save_result(model_dir_path, sim_result)


        sim_result = load_result(model_dir_path, simulation_desc="10sims_emm_1_components")
        seed = sim_result.quick_plot(n_imgs=9, n_cols=3, norm_lims=False, title_prefix="Scenario")

    plt.show()

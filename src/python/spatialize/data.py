import importlib.resources as rs
import os, json
from pathlib import Path

import numpy as np
import pandas as pd

from spatialize import EstimationResult, logging, SpatializeError
from spatialize.gs.esi import ESIResult
from spatialize.gs.ess import ESSResult
from spatialize.logging import log_message
from spatialize.resources import data


def load_result(result_dir_path, just_esi_result=False, simulation_desc=None):
    """
    Loads results from a result directory, including metadata, estimations, ESI samples,
    and optionally simulation results (ESSResults).

    :param result_dir_path: Path to the directory containing result files and metadata.
    :param just_esi_result: If True, returns only an ESIResult instance even if simulations are available.
    :param simulation_desc: Base name of the specific simulation file to load (without `.csv` extension).
                            If provided, only that simulation will be loaded.
    :raises SpatializeError: If the directory does not correspond to a simulation but a simulation description is provided.
    :raises SpatializeError: If the provided simulation description is not found in the metadata.

    :return: Depending on the available data, returns an instance of EstimationResult,
             ESIResult, ESSResult, or a list of ESSResult instances.
    :rtype: EstimationResult | ESIResult | ESSResult | list[ESSResult]
    """
    # load the metadata file
    meta_data_fn = os.path.join(result_dir_path, "metadata.json")
    with open(meta_data_fn, "r") as outfile:
        meta_data = json.load(outfile)

    if meta_data["main_result"] != "simulation" and not simulation_desc is None:
        raise SpatializeError("Not a simulation result directory")

    # load the estimation
    fn = os.path.join(result_dir_path, meta_data['estimation'])
    estimation = pd.read_csv(fn).values.reshape(-1)

    # load the xi locations
    fn = os.path.join(result_dir_path, meta_data['xi'])
    xi = pd.read_csv(fn).values

    # load the observed points/values used to build the result, if this directory was
    # saved by a version of spatialize that persisted them (needed to calibrate ensemble
    # widening after reload; older saved directories simply won't have these keys)
    points = values = None
    if 'points' in meta_data and 'values' in meta_data:
        fn = os.path.join(result_dir_path, meta_data['points'])
        points = pd.read_csv(fn).values
        fn = os.path.join(result_dir_path, meta_data['values'])
        values = pd.read_csv(fn).values.reshape(-1)

    try:
        # load the esi_samples
        fn = os.path.join(result_dir_path, meta_data['esi_samples'])
        esi_samples = pd.read_csv(fn).values
    except:
        esi_samples = None

    if esi_samples is None:
        log_message(logging.logger.info(f"an instances of EstimationResult was loaded"))
        return EstimationResult(estimation,
                                griddata=meta_data['griddata'],
                                original_shape=meta_data['original_shape'],
                                xi=xi, points=points, values=values)

    esi_result = ESIResult(estimation, esi_samples,
                           griddata=meta_data['griddata'],
                           original_shape=meta_data['original_shape'],
                           xi=xi, points=points, values=values)

    if meta_data["main_result"] == "estimation" or just_esi_result:
        log_message(logging.logger.info(f"an instance of ESIResult was loaded"))
        return esi_result

    if not simulation_desc is None:
        if not (simulation_desc + ".csv") in set(meta_data["simulations"]):
            raise SpatializeError(f"Simulation description not found in metadata {simulation_desc}")

        fn = os.path.join(result_dir_path, simulation_desc + ".csv")
        ess_scenarios = pd.read_csv(fn).values

        log_message(logging.logger.info(f"an instance of ESSResult was loaded ({simulation_desc})"))
        return ESSResult(ess_scenarios, esi_result, Path(simulation_desc).stem)

    sim_results = []
    for sim in meta_data["simulations"]:
        fn = os.path.join(result_dir_path, sim)
        ess_scenarios = pd.read_csv(fn).values
        sim_results.append(ESSResult(ess_scenarios, esi_result, Path(sim).stem))

    plural = "" if len(sim_results) == 1 else "s"
    tense = "was" if len(sim_results) == 1 else "were"
    log_message(logging.logger.info(f"{len(sim_results)}"
                                    f" instance{plural}"
                                    f" of ESSResult {tense} "
                                    f"loaded : {sim_results}"))

    return sim_results


def save_result(result_dir_path, result):
    """
    Saves an estimation or simulation result to a specified directory, including metadata,
    estimations, locations, ESI samples, and simulation data if applicable.

    :param result_dir_path: Path to the directory where the result should be saved.
                            If it does not exist, it will be created.
    :param result: The result object to save. Can be an instance of EstimationResult,
                   ESIResult, or ESSResult. If the directory already contains an
                   estimation result (ESIResult), and `result` is an ESSResult
                   (simulation), the metadata will be updated to reflect a simulation-type
                   result, and the simulation will be added to the existing simulations list.
    """

    def ensure_directory(path):
        if not os.path.exists(path):
            os.makedirs(path)
            return False
        else:
            return True

    already_exists = ensure_directory(result_dir_path)

    if not already_exists:
        meta_data = {}
    else:
        try:
            meta_data = json.load(open(os.path.join(result_dir_path, 'metadata.json')))
        except FileNotFoundError:
            meta_data = {}

    if isinstance(result, ESSResult):
        meta_data["main_result"] = "simulation"
        est_result = result.esi_result
    else:
        try:
            is_simulation = (meta_data["main_result"] == "simulation")
        except KeyError:
            is_simulation = False
        if not (already_exists and is_simulation):
            meta_data["main_result"] = "estimation"
        est_result = result

    meta_data.update({
        "estimation": "estimation.csv",
        "griddata": est_result.griddata,
        "original_shape": est_result.original_shape,
        "xi": "locations.csv"
    })

    # save the estimation
    fn = os.path.join(result_dir_path, meta_data['estimation'])
    columns = ["estimation"]
    pd.DataFrame(est_result.estimation()).to_csv(fn, index=False, header=columns)

    # save the xi locations
    fn = os.path.join(result_dir_path, meta_data['xi'])
    columns = ["x", "y"]
    pd.DataFrame(est_result._xi).to_csv(fn, index=False, header=columns)

    # save the observed points/values used to build the result, if available (needed to
    # calibrate ensemble widening after reload)
    if est_result.points is not None and est_result.values is not None:
        meta_data['points'] = "points.csv"
        meta_data['values'] = "values.csv"
        fn = os.path.join(result_dir_path, meta_data['points'])
        columns = [f"x{i}" for i in range(np.asarray(est_result.points).shape[1])]
        pd.DataFrame(est_result.points).to_csv(fn, index=False, header=columns)
        fn = os.path.join(result_dir_path, meta_data['values'])
        pd.DataFrame(est_result.values).to_csv(fn, index=False, header=["value"])

    if isinstance(est_result, ESIResult):
        # save the esi_samples
        meta_data['esi_samples'] = "esi_samples.csv"
        fn = os.path.join(result_dir_path, meta_data['esi_samples'])
        columns = [f"es{i}" for i in range(est_result.esi_samples(raw=True).shape[1])]
        pd.DataFrame(est_result.esi_samples(raw=True)).to_csv(fn, index=False, header=columns)
        meta_data['n_esi_samples'] = len(columns)

    if isinstance(result, ESSResult):
        # save the simulations
        fn = str(result) + ".csv"
        pn = os.path.join(result_dir_path, fn)
        columns = [f"sim{i}" for i in range(result.scenarios.shape[1])]
        pd.DataFrame(result.scenarios).to_csv(pn, index=False, header=columns)
        if not "simulations" in meta_data:
            meta_data["simulations"] = [fn]
        else:
            if fn not in meta_data["simulations"]:
                meta_data["simulations"].append(fn)

    # save the metadata file
    meta_data_fn = os.path.join(result_dir_path, "metadata.json")
    with open(meta_data_fn, "w") as outfile:
        json.dump(meta_data, outfile, indent=4)


# Toy data sets included in the library
def load_drill_holes_andes_2D():
    path = os.path.join(str(rs.files(data)), "dc1_input_data.csv")
    input_samples = pd.read_csv(path)

    path = os.path.join(str(rs.files(data)), "dc1_output_grid.dat")
    with open(path, 'r') as grid_data:
        lines = grid_data.readlines()
        lines = [l.strip().split() for l in lines[5:]]
        aux = np.float32(lines)
    output_locations = pd.DataFrame(aux, columns=['x', 'y', 'z'])

    path = os.path.join(str(rs.files(data)), "dc1_ok_kriging_example.csv")
    ok_kriging_example = pd.read_csv(path)

    path = os.path.join(str(rs.files(data)), "dc1_vario_cu_omni.csv")
    omi_exp_variogram_example = pd.read_csv(path)

    return input_samples, output_locations, ok_kriging_example, omi_exp_variogram_example


def load_drill_holes_andes_3D():
    path = os.path.join(str(rs.files(data)), "dc2_output_box.csv")
    output_locations = pd.read_csv(path)

    path = os.path.join(str(rs.files(data)), "dc2_input_muestras.dat")
    with open(path, 'r') as in_data:
        lines = in_data.readlines()
        lines = [l.strip().split() for l in lines[8:]]
        aux = np.float32(lines)
    input_samples = pd.DataFrame(aux, columns=['x', 'y', 'z', 'cu', 'au', 'rocktype'])

    return input_samples, output_locations


def load_simulated_anisotropic_data():
    prefix = "sim_data_geom_anis_nugg0"
    path = os.path.join(str(rs.files(data)), prefix + ".csv")
    ground_truth = pd.read_csv(path)

    path = os.path.join(str(rs.files(data)), prefix + ".1_1perc.csv")
    input_samples_1perc = pd.read_csv(path)

    path = os.path.join(str(rs.files(data)), prefix + ".1_5perc.csv")
    input_samples_5perc = pd.read_csv(path)

    path = os.path.join(str(rs.files(data)), prefix + ".1_reduced.csv")
    input_samples_reduced = pd.read_csv(path)

    # ordinary kriging example
    prefix = "sim_kriging_geom_anis_nugg0"
    path = os.path.join(str(rs.files(data)), prefix + ".1_1perc.csv")
    kriging_1perc = pd.read_csv(path)

    path = os.path.join(str(rs.files(data)), prefix + ".1_5perc.csv")
    kriging_5perc = pd.read_csv(path)

    path = os.path.join(str(rs.files(data)), prefix + ".1_reduced.csv")
    kriging_reduced = pd.read_csv(path)

    return ((input_samples_1perc, kriging_1perc),
            (input_samples_5perc, kriging_5perc),
            (input_samples_reduced, kriging_reduced),
            ground_truth)

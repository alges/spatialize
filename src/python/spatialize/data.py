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
    Load a result previously saved with `save_result`.

    Reads the ``metadata.json`` file in *result_dir_path* along with the
    estimation, target locations, ESI samples, and (if present) simulation
    scenarios it references, and reconstructs the corresponding result
    object.

    Parameters
    ----------
    result_dir_path : str
        Path to the directory containing the result files and
        ``metadata.json``.
    just_esi_result : bool, optional
        If True, return only the `spatialize.gs.esi.ESIResult` instance
        even if simulation results (ESSResults) are also available in the
        directory. Default is False.
    simulation_desc : str, optional
        Base name of a specific simulation file to load (without the
        ``.csv`` extension). If provided, only that simulation is loaded
        and returned as a single `spatialize.gs.ess.ESSResult`; otherwise
        all simulations found in the metadata are loaded.

    Returns
    -------
    EstimationResult or ESIResult or ESSResult or list of ESSResult
        Depending on what is available in *result_dir_path*: an
        `EstimationResult` if no ESI samples were saved, an `ESIResult` if
        `just_esi_result` is True or the directory holds an estimation-type
        result, a single `ESSResult` if `simulation_desc` is given, or a
        list of `ESSResult` instances (one per saved simulation) otherwise.

    Raises
    ------
    SpatializeError
        If `simulation_desc` is given but the directory does not
        correspond to a simulation result, or if `simulation_desc` is not
        found among the simulations recorded in the metadata.
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
    Save an estimation or simulation result to a directory.

    Writes ``metadata.json`` together with the estimation, target
    locations, observed points/values (if available), ESI samples, and
    simulation scenarios (if applicable) as CSV files, so the result can
    later be reconstructed with `load_result`.

    Parameters
    ----------
    result_dir_path : str
        Path to the directory where the result should be saved. Created if
        it does not already exist. If the directory already contains a
        saved result, its ``metadata.json`` is loaded and updated in
        place rather than overwritten from scratch.
    result : EstimationResult or ESIResult or ESSResult
        The result object to save. If the directory already contains an
        estimation result and `result` is an `spatialize.gs.ess.ESSResult`
        (simulation), the metadata is updated to reflect a simulation-type
        result and the new simulation is appended to the existing
        simulations list instead of replacing it.
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
    """
    Load a 2-D copper (Cu) drill-hole dataset from an Andean deposit.

    Bundled sample resource with input assay data, a 2-D estimation grid,
    a reference ordinary-kriging estimation, and an omnidirectional
    experimental variogram, used throughout the ESI tutorials to
    demonstrate 2-D interpolation.

    Returns
    -------
    input_samples : pandas.DataFrame of shape (400, 6)
        Drill-hole assay samples with columns ``'x'``, ``'y'`` (location,
        arbitrary distance units), ``'z'`` (elevation), ``'cu'`` (copper
        grade), ``'au'`` (gold grade), and ``'rocktype'`` (categorical rock
        type code).
    output_locations : pandas.DataFrame of shape (60000, 3)
        Regular 2-D grid of target locations to interpolate onto, with
        columns ``'x'``, ``'y'``, ``'z'``.
    ok_kriging_example : pandas.DataFrame of shape (60000, 13)
        Reference ordinary-kriging estimates and variances at
        `output_locations` for several variogram configurations, with
        columns ``'x'``, ``'y'``, ``'z'`` followed by
        ``'est_cu_case1'``/``'var_cu_case1'`` through
        ``'est_cu_case4'``/``'var_cu_case4'`` and
        ``'est_cu_case_esipaper'``/``'var_cu_case_esipaper'``.
    omi_exp_variogram_example : pandas.DataFrame of shape (22, 3)
        Omnidirectional experimental variogram computed from
        `input_samples`, with columns ``'gamma_1'`` (semivariance),
        ``'npairs_1'`` (pair count per lag), and ``'lags_1'`` (lag
        distance).

    Examples
    --------
    >>> from spatialize.data import load_drill_holes_andes_2D
    >>> samples, target_locations, _, _ = load_drill_holes_andes_2D()
    >>> points = samples[['x', 'y']].values
    >>> values = samples[['cu']].values[:, 0]
    >>> xi = target_locations[['x', 'y']].values
    """
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
    """
    Load a 3-D copper/gold drill-hole dataset from an Andean deposit.

    Bundled sample resource with input assay data and a 3-D estimation
    domain, used throughout the ESI tutorials to demonstrate 3-D
    interpolation.

    Returns
    -------
    input_samples : pandas.DataFrame of shape (2376, 6)
        Drill-hole assay samples with columns ``'x'``, ``'y'``, ``'z'``
        (location, arbitrary distance units), ``'cu'`` (copper grade),
        ``'au'`` (gold grade), and ``'rocktype'`` (categorical rock type
        code).
    output_locations : pandas.DataFrame of shape (72529, 3)
        3-D block-model locations to interpolate onto, with columns
        ``'x'``, ``'y'``, ``'z'``.

    Examples
    --------
    >>> from spatialize.data import load_drill_holes_andes_3D
    >>> samples, locations = load_drill_holes_andes_3D()
    >>> points = samples[['x', 'y', 'z']].values
    >>> values = samples[['cu']].values[:, 0]
    >>> xi = locations[['x', 'y', 'z']].values
    """
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
    """
    Load a synthetic geometrically anisotropic dataset with multiple sampling densities.

    Bundled resource built from 100 unconditional Gaussian simulations on a
    regular grid, exhibiting geometric (directional) anisotropy with a
    nugget effect of 0. Three subsets of grid nodes, at increasing sampling
    density, are provided as "sample" locations together with a matching
    ordinary-kriging reference estimate, plus the full grid of simulated
    values as ground truth. Used in the ESS tutorials to illustrate
    interpolation and simulation under anisotropy at different data
    densities.

    Returns
    -------
    (input_samples_1perc, kriging_1perc) : tuple of pandas.DataFrame
        Sparsest subset: 400 grid nodes (~1% of the full grid).
        `input_samples_1perc` has shape (400, 103) with columns ``'x'``,
        ``'y'``, ``'z'`` followed by ``'sim1'`` through ``'sim100'`` (one
        column per simulated realization). `kriging_1perc` has shape
        (40000, 4) with columns ``'x'``, ``'y'``, ``'z'``, ``'cond_ok'``
        (ordinary-kriging estimate over the full grid, conditioned on this
        subset).
    (input_samples_5perc, kriging_5perc) : tuple of pandas.DataFrame
        Medium-density subset: 2000 grid nodes (~5% of the full grid).
        Same column layout as the 1% pair; `input_samples_5perc` has shape
        (2000, 103) and `kriging_5perc` has shape (40000, 4).
    (input_samples_reduced, kriging_reduced) : tuple of pandas.DataFrame
        Smallest subset: 50 grid nodes. Same column layout as above;
        `input_samples_reduced` has shape (50, 103) and `kriging_reduced`
        has shape (40000, 4).
    ground_truth : pandas.DataFrame of shape (40000, 103)
        Full simulation grid, with columns ``'x'``, ``'y'``, ``'z'``
        followed by ``'sim1'`` through ``'sim100'``, one column per
        realization.

    Examples
    --------
    >>> from spatialize.data import load_simulated_anisotropic_data
    >>> data_1perc, data_5perc, data_reduced, ground_truth = load_simulated_anisotropic_data()
    >>> using_sim = 'sim10'
    >>> points = data_reduced[0][['x', 'y']].values
    >>> values = data_reduced[0][[using_sim]].values[:, 0]
    >>> xi = ground_truth[['x', 'y']].values
    """
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

"""Saving and loading results."""
import os

import numpy as np

import cases
import snapshot_lib as sl

LIB = sl.load_lib()
DATA = cases.datasets()


def test_a_simulation_is_saved_under_its_description_and_loaded_back(tmp_path):
    """An ESS result is saved in a file named after its description, never after its summary, and
    loads back with the same scenarios."""
    from spatialize.data import save_result, load_result
    from spatialize.gs.esi import esi_nongriddata
    from spatialize.gs.ess import ess_sample
    s, v, q = DATA["2d"]
    r = esi_nongriddata(s, v, q, local_interpolator="idw", exponent=2.0, n_partitions=cases.T, seed=cases.SEED)
    sim = ess_sample(r, n_sims=3)
    path = str(tmp_path / "result")
    save_result(path, r)
    save_result(path, sim)
    assert os.path.exists(os.path.join(path, f"{sim.desc}.csv"))
    loaded = load_result(path, simulation_desc=sim.desc)
    assert loaded.desc == sim.desc
    assert np.allclose(loaded.scenarios, sim.scenarios, equal_nan=True)

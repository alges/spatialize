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


def test_lithology_golden_is_a_truth_with_samples_drawn_from_it():
    """The Golden lithology: a 120 × 120 grid of six lithologies and four ordered eras, samples drawn
    reproducibly among the cells with a lithology, the provenance and licence shipped with it."""
    import importlib.resources as rs
    from spatialize.data import load_lithology_golden, LITHOLOGIES_GOLDEN, ERAS_GOLDEN
    from spatialize.resources import data
    s, g, u = load_lithology_golden(n_samples=200, seed=3)
    assert g.shape == (14400, 6) and len(u) == 53
    assert set(g["lithology"]) - {""} == set(LITHOLOGIES_GOLDEN)
    assert set(g["era"]) - {""} == set(ERAS_GOLDEN)
    assert len(s) == 200 and (s["lithology"] != "").all()
    assert s.equals(load_lithology_golden(n_samples=200, seed=3)[0])
    merged = s.merge(g, on=["x", "y"], suffixes=("", "_grid"))
    assert len(merged) == 200 and (merged["lithology"] == merged["lithology_grid"]).all()
    assert os.path.exists(os.path.join(str(rs.files(data)), "lithology_golden", "README.md"))


def test_facies_pyrcz_ships_with_its_licence():
    """The GeoDataSets facies data load with common column names, and their MIT licence is bundled."""
    import importlib.resources as rs
    from spatialize.data import load_facies_pyrcz
    from spatialize.resources import data
    for biased, n in ((False, 450), (True, 368)):
        d = load_facies_pyrcz(biased)
        assert d.shape == (n, 6) and set(d["facies"]) == {0, 1}
    licence = open(os.path.join(str(rs.files(data)), "facies_pyrcz", "LICENSE")).read()
    assert "Michael Pyrcz" in licence and "Permission is hereby granted" in licence


def test_exhaustive_fields_are_truths_with_samples_drawn_from_them():
    """The Walker Lake elevation and the Sundarbans reflectance: full grids as the truth, samples drawn
    reproducibly from their cells, the provenance shipped with them."""
    import importlib.resources as rs
    from spatialize.data import load_elevation_walker_lake, load_reflectance_sundarbans
    from spatialize.resources import data
    for load, folder, col, shape, n in ((load_elevation_walker_lake, "elevation_walker_lake", "elevation",
                                         (78000, 3), 780),
                                        (load_reflectance_sundarbans, "reflectance_sundarbans", "reflectance",
                                         (232000, 3), 2320)):
        s, g = load()
        assert g.shape == shape and len(s) == n and g[col].notna().all()
        s2 = load(n_samples=100, seed=5)[0]
        assert s2.equals(load(n_samples=100, seed=5)[0])
        merged = s2.merge(g, on=["x", "y"], suffixes=("", "_grid"))
        assert len(merged) == 100 and (merged[col] == merged[f"{col}_grid"]).all()
        assert os.path.exists(os.path.join(str(rs.files(data)), folder, "README.md"))


def test_precipitation_maipo_loads_the_gauges_and_the_basin():
    from spatialize.data import load_precipitation_maipo
    samples, locations = load_precipitation_maipo()
    days = ["2018-07-04", "2018-07-05", "2018-07-06"]
    assert list(samples.columns) == ["x", "y", "z"] + days
    assert list(locations.columns) == ["x", "y", "z"]
    assert len(samples) == 69 and len(locations) == 3824
    for day in days:
        rain = samples[day].dropna()
        assert (rain >= 0).all() and 0 < (rain == 0).mean() < 0.5

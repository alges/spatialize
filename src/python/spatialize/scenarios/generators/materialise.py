"""Materialise the pinned data of a scenario from its ``scenario.yaml`` and write CHECKSUMS.sha256.

    python -m spatialize.scenarios.generators.materialise S03-anisotropic-field

Pinned data are immutable once a suite version is released: re-materialising a released scenario
requires bumping its ``version``.
"""
import hashlib
import os
import sys

import numpy as np

from . import fields
from .. import engine


def materialise(sc):
    """Write the pinned data of a scenario and its checksum manifest.

    Dispatches on ``truth.generator``: ``sgf_exponential`` (stationary Gaussian fields, S03) or
    ``vbm_edge_cases`` (degenerate designs, S12). The generator stream is sequential, so increasing
    the number of fields keeps the existing ones byte for byte.

    Parameters
    ----------
    sc : Scenario

    Returns
    -------
    list of str
        Paths written, relative to the scenario's directory.

    Raises
    ------
    NotImplementedError
        For an unknown generator.
    """
    generator = sc.spec["truth"]["generator"]
    if generator == "sgf_exponential":
        written = _sgf_fields(sc)
    elif generator == "vbm_edge_cases":
        written = _edge_case_fields(sc)
    else:
        raise NotImplementedError(generator)
    with open(sc.file("CHECKSUMS.sha256"), "w") as fh:
        for rel in written:
            fh.write(f"{hashlib.sha256(open(sc.file(rel), 'rb').read()).hexdigest()} {rel}\n")
    return written


def _n_fields(d):
    return max(v for v in d["fields"].values()) if isinstance(d["fields"], dict) else d["fields"]


def _save(sc, k, arrays, written):
    for name, a in arrays.items():
        rel = f"data/{name}_{k}.npy"
        np.save(sc.file(rel), a)
        written.append(rel)


def _sgf_fields(sc):
    """Stationary Gaussian fields: samples and grid drawn jointly (S03)."""
    s = sc.spec
    t, d = s["truth"], s["data"]
    rng = np.random.default_rng(d["generator_seed"])
    grid = fields.grid(d["grid"])
    K = max(v for v in d["fields"].values()) if isinstance(d["fields"], dict) else d["fields"]
    os.makedirs(sc.file("data"), exist_ok=True)
    written = []
    for k in range(K):
        samples = fields.uniform_design(rng, d["design"]["n"])
        z = fields.sgf_exponential(rng, np.vstack([samples, grid]), t["a1"], t["a2"], t["theta_deg"])
        arrays = {"samples": samples.astype(np.float32), "values": z[: len(samples)].astype(np.float32),
                  "truth": z[len(samples):].astype(np.float32)}
        for name, a in arrays.items():
            rel = f"data/{name}_{k}.npy"
            np.save(sc.file(rel), a)
            written.append(rel)
    return written


def _edge_case_fields(sc):
    """Degenerate designs on a Voronoi block-mark field (S12): few data in an inner box, duplicated
    locations (same and different values), queries on a grid that leaves the data box and exactly on
    data. ``exact`` holds (query index, datum value) for the queries placed on non-duplicated data."""
    s = sc.spec
    t, d = s["truth"], s["data"]
    rng = np.random.default_rng(d["generator_seed"])
    lo, hi = np.asarray(d["design"]["box"], float).T
    mu, sigma = t["marks"]["lognormal"]
    os.makedirs(sc.file("data"), exist_ok=True)
    written = []
    for k in range(_n_fields(d)):
        z = fields.vbm(rng, t["n_cells"], lambda r, n: r.lognormal(mu, sigma, n))
        base = lo + (hi - lo) * rng.random((d["design"]["n"], 2))
        n_same, n_diff = d["duplicates"]["same_value"], d["duplicates"]["different_value"]
        dup = rng.choice(len(base), n_same + n_diff, replace=False)
        base_values = z(base)
        dup_values = np.concatenate([base_values[dup[:n_same]],
                                     base_values[dup[n_same:]] * rng.lognormal(0.0, 0.5, n_diff)])
        samples = np.vstack([base, base[dup]])
        values = np.concatenate([base_values, dup_values])
        free = np.setdiff1d(np.arange(len(base)), dup)
        on = rng.choice(free, d["queries"]["on_data"], replace=False)
        grid = fields.grid(d["queries"]["grid"])
        queries = np.vstack([grid, base[on]])
        exact = np.column_stack([len(grid) + np.arange(len(on)), base_values[on]])
        _save(sc, k, {"samples": samples.astype(np.float32), "values": values.astype(np.float32),
                      "queries": queries.astype(np.float32), "exact": exact.astype(np.float32)}, written)
    return written


if __name__ == "__main__":
    for sid in sys.argv[1:]:
        sc = engine.Scenario(sid, os.path.join(engine.CATALOG, sid),
                             engine._yaml().safe_load(open(os.path.join(engine.CATALOG, sid, "scenario.yaml"))))
        print(sid, len(materialise(sc)), "files")

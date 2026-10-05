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
    s = sc.spec
    t, d = s["truth"], s["data"]
    if t["generator"] != "sgf_exponential":
        raise NotImplementedError(t["generator"])
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
    with open(sc.file("CHECKSUMS.sha256"), "w") as fh:
        for rel in written:
            fh.write(f"{hashlib.sha256(open(sc.file(rel), 'rb').read()).hexdigest()} {rel}\n")
    return written


if __name__ == "__main__":
    for sid in sys.argv[1:]:
        sc = engine.Scenario(sid, os.path.join(engine.CATALOG, sid),
                             engine._yaml().safe_load(open(os.path.join(engine.CATALOG, sid, "scenario.yaml"))))
        print(sid, len(materialise(sc)), "files")

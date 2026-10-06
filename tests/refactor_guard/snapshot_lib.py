"""Shared helpers: locate the in-place extension, platform tag, snapshot I/O."""
import json
import os
import platform
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SNAP_ROOT = os.path.join(os.path.dirname(__file__), "snapshots")


def _check_fresh_build():
    """Refuse a stale binary: `setup.py build_ext` does not track header dependencies, so editing a
    .hpp and running `make` silently keeps the old extension (verified 2026-10-05)."""
    import glob
    so = glob.glob(os.path.join(REPO, "libspatialize*.so")) + glob.glob(os.path.join(REPO, "libspatialize*.pyd"))
    if not so:
        raise RuntimeError("no in-place build of libspatialize; run: python setup.py build_ext --inplace --force")
    sources = glob.glob(os.path.join(REPO, "src", "c++", "*.cpp")) + glob.glob(os.path.join(REPO, "include", "spatialize", "**", "*.hpp"), recursive=True)
    newest = max(sources, key=os.path.getmtime)
    if os.path.getmtime(newest) > max(os.path.getmtime(f) for f in so):
        raise RuntimeError(f"libspatialize is older than {os.path.relpath(newest, REPO)}; rebuild with: "
                           "python setup.py build_ext --inplace --force")


def load_lib():
    """Import the extension built in place (repo root), never an installed wheel."""
    _check_fresh_build()
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import libspatialize
    if not os.path.abspath(libspatialize.__file__).startswith(REPO):
        raise RuntimeError(f"libspatialize imported from {libspatialize.__file__}, expected the in-place "
                           f"build in {REPO} (run `make` first)")
    return libspatialize


def platform_tag():
    return f"{sys.platform}-{platform.machine()}"


def snap_dir():
    return os.path.join(SNAP_ROOT, platform_tag())


def save(name, inputs, outputs, meta):
    os.makedirs(snap_dir(), exist_ok=True)
    arrays = {f"in.{k}": v for k, v in inputs.items()}
    arrays.update({f"out.{k}": v for k, v in outputs.items()})
    np.savez_compressed(os.path.join(snap_dir(), f"{name}.npz"), **arrays,
                        meta=np.array(json.dumps(meta)))


def load(name):
    z = np.load(os.path.join(snap_dir(), f"{name}.npz"))
    inputs = {k[3:]: z[k] for k in z.files if k.startswith("in.")}
    outputs = {k[4:]: z[k] for k in z.files if k.startswith("out.")}
    return inputs, outputs, json.loads(str(z["meta"]))


def bitwise_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

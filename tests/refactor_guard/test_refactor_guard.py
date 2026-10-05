"""Internal refactor guard: every libspatialize function must reproduce its snapshot bit for bit.

Snapshots are machine-specific (compiler, libm, OpenMP); the test is skipped on platforms without
snapshots. See README.md.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import cases  # noqa: E402
import snapshot_lib as sl  # noqa: E402

if not os.path.isdir(sl.snap_dir()):
    pytest.skip(f"no refactor-guard snapshots for platform {sl.platform_tag()}", allow_module_level=True)

LIB = sl.load_lib()
CASES = {name: fn for name, _, fn in cases.cases(LIB)}


@pytest.mark.parametrize("name", sorted(CASES))
def test_bitwise(name):
    path = os.path.join(sl.snap_dir(), f"{name}.npz")
    if not os.path.exists(path):
        pytest.fail(f"missing snapshot {path}")
    inputs, expected, meta = sl.load(name)
    got = cases.normalise(CASES[name](inputs["samples"], inputs["values"], inputs["queries"]))
    assert sorted(got) == sorted(expected), f"output structure changed: {sorted(got)} vs {sorted(expected)}"
    for key in expected:
        assert sl.bitwise_equal(got[key], expected[key]), (
            f"{name}[{key}] differs from snapshot (commit {meta['commit'][:8]}): "
            f"dtype {got[key].dtype}/{expected[key].dtype}, shape {got[key].shape}/{expected[key].shape}")

"""Guard checks: every libspatialize function must reproduce its snapshot bit for bit.

Snapshots are machine-specific (compiler, libm, OpenMP); the test is skipped on platforms without
snapshots. See README.md.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import cases  # noqa: E402
import public_cases  # noqa: E402
import snapshot_lib as sl  # noqa: E402

if not os.path.isdir(sl.snap_dir()):
    pytest.skip(f"no guard-check snapshots for platform {sl.platform_tag()}", allow_module_level=True)

LIB = sl.load_lib()
CASES = {name: fn for name, _, fn in cases.cases(LIB)}
CASES.update({name: fn for name, _, fn in public_cases.public_cases()})


@pytest.mark.parametrize("name", sorted(CASES))
def test_bitwise(name):
    path = os.path.join(sl.snap_dir(), f"{name}.npz")
    if not os.path.exists(path):
        pytest.fail(f"missing snapshot {path}")
    inputs, expected, meta = sl.load(name)
    got = CASES[name](inputs["samples"], inputs["values"], inputs["queries"])
    if not name.startswith("api."):
        got = cases.normalise(got)
    assert sorted(got) == sorted(expected), f"output structure changed: {sorted(got)} vs {sorted(expected)}"
    for key in expected:
        assert sl.bitwise_equal(got[key], expected[key]), (
            f"{name}[{key}] differs from snapshot (commit {meta['commit'][:8]}): "
            f"dtype {got[key].dtype}/{expected[key].dtype}, shape {got[key].shape}/{expected[key].shape}")

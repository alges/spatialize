"""spatialize conformance: run the whole scenario catalogue with spatialize's runner.

One Holm budget covers the run (α_suite = 1e-3). Fresh seed per run unless SPATIALIZE_SCENARIO_SEED
is set; the seed is printed so a failure can be reproduced. Mode: SPATIALIZE_SCENARIO_MODE (ci|full).

    python setup.py build_ext --inplace --force
    DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python -m pytest -q tests/scenarios   # macOS + conda
"""
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "src", "python")]   # in-place build and sources first

from spatialize import scenarios  # noqa: E402
from spatialize.scenarios.runners.spatialize import SpatializeRunner  # noqa: E402

MODE = os.environ.get("SPATIALIZE_SCENARIO_MODE", "ci")
SEED = os.environ.get("SPATIALIZE_SCENARIO_SEED")


@pytest.fixture(scope="module")
def report():
    rep = scenarios.run(scenarios.catalog(), SpatializeRunner(), mode=MODE,
                        seed=None if SEED is None else int(SEED))
    print("\n" + rep.table())
    return rep


def _ids():
    out = []
    for sc in scenarios.catalog().values():
        out += [f"{sc.id}/{c['id']}" for c in sc.spec["checks"]]
    return out


@pytest.mark.parametrize("check", _ids())
def test_check(report, check):
    o = next(o for o in report.outcomes if f"{o.scenario}/{o.check}" == check)
    if o.skipped:
        pytest.skip(o.skipped)
    assert o.passed, (f"{check} failed (seed {report.seed}, mode {report.mode}): {o.result.family} "
                      f"p={o.result.p_value:.3g} level={o.level:.2g} expect={o.expect} — {o.result.detail}")

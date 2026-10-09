"""Unit tests run on this checkout: the in-place build of libspatialize and the sources of
src/python, never an installed wheel. The data and the constants they share with the guard checks
live in tests/guard (cases.py, snapshot_lib.py)."""
import os
import sys

_TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_TESTS)
for path in (os.path.join(_TESTS, "guard"), os.path.join(_REPO, "src", "python"), _REPO):
    if path not in sys.path:
        sys.path.insert(0, path)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet_progress():
    """No progress bars in the unit tests (session setting ``progress``); warnings still show."""
    from spatialize import session
    with session.override(progress=False):
        yield

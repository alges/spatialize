"""Decoders: the identities between them and their independence from the units of the coordinates.

Unit tests on this checkout's in-place build (see tests/unit/conftest.py)."""
import numpy as np
import pytest

import cases
import snapshot_lib as sl

LIB = sl.load_lib()
DATA = cases.datasets()


@pytest.mark.parametrize("ds", ["1d", "2d", "3d"])
@pytest.mark.parametrize("method", ["estimate", "loo", "kfold"])
def test_sharpidw_without_its_factors_is_adaptiveidw(ds, method):
    s, v, q = DATA[ds]
    args = (s, v, q, "mondrian", cases.ALPHA, cases.T_ADAPTIVE, cases.SEED)
    tail = (method, cases.K, cases.FSEED)
    adaptive = LIB.run(*args, "adaptiveidw", {}, *tail)[1]
    sharp = LIB.run(*args, "sharpidw", {"kappa_r": 0.0, "kappa_g": 0.0}, *tail)[1]
    assert sl.bitwise_equal(np.asarray(sharp), np.asarray(adaptive))


@pytest.mark.parametrize("decoder", ["adaptiveidw", "sharpidw", "wdraw_adaptiveidw", "wdraw_sharpidw"])
def test_adaptive_decoders_do_not_depend_on_coordinate_units(decoder):
    """The adaptive weights are relative to the nearest datum, so with coordinates 10^4 times larger
    no member of a cell with data is NaN and the members are the same up to float rounding (a
    rounding may flip a discrete choice of the fit or of a draw in a few members)."""
    s, v, q = DATA["2d"]
    runs = [np.asarray(LIB.run((s * c).astype(np.float32), v, (q * c).astype(np.float32), "mondrian", cases.ALPHA,
                               cases.T_ADAPTIVE, cases.SEED, decoder, {}, "estimate")[1]) for c in (1.0, 1e4)]
    assert np.array_equal(np.isnan(runs[0]), np.isnan(runs[1]))
    differ = np.abs(runs[0] - runs[1]) > 1e-4 * (1 + np.abs(runs[0]))
    assert np.nanmean(differ) < 0.02   # 1.2 % with the theory's Mondrian (2026-10-09), under 1 % before

"""Scores and searches: density readings free of units, the data a scorer leaves out.

Unit tests on this checkout's in-place build (see tests/unit/conftest.py)."""
import numpy as np
import pytest

import cases
import snapshot_lib as sl

LIB = sl.load_lib()
DATA = cases.datasets()


@pytest.mark.parametrize("scale", [0.01, 100.0])
def test_density_readings_do_not_depend_on_units(scale):
    """The KDE of the local laws (ESS, NLL, Pareto) takes its bandwidth from the sample's own spread,
    so changing the units of the variable changes no conclusion: the KDE spread scales with the data,
    the NLL shifts by log(scale) and the encoder error does not change."""
    from spatialize.empirical import FittedModelFactory
    from spatialize.gs.esi import scorefunction as sf
    from spatialize.gs.esi.pareto import EmpiricalRobustnessBound
    from spatialize import session
    rng = np.random.default_rng(cases.GENERATOR_SEED)
    members, truth = rng.normal(size=(40, 60)), rng.normal(size=40)

    model, _ = FittedModelFactory().create(members[0] * scale)
    reference, _ = FittedModelFactory().create(members[0])
    assert np.isclose(model.bandwidth_, reference.bandwidth_ * scale, rtol=1e-12)

    nll = sf.neg_log_likelihood(truth, members)
    assert np.isclose(sf.neg_log_likelihood(truth * scale, members * scale), nll + np.log(scale), rtol=1e-9)

    s, v, _ = DATA["2d"]
    with session.override(parallel=False):
        eps = EmpiricalRobustnessBound(30, cases.ALPHA, "idw", "mondrian", cases.SEED).estimate(s, v)
        eps_scaled = EmpiricalRobustnessBound(30, cases.ALPHA, "idw", "mondrian", cases.SEED).estimate(
            s, (v * scale).astype(np.float32))
    assert np.isclose(eps, eps_scaled, rtol=1e-4)


def test_left_out_counts_the_data_a_scorer_cannot_use():
    """A datum is left out of a score when it has fewer valid members than the scorer needs (1 for
    the absolute and squared errors, 30 for the NLL, 2 for the CRPS); the warning of the searches
    lists every configuration over the threshold and marks the best."""
    import warnings
    from spatialize.gs.esi import scorefunction as sf
    samples = np.ones((4, 40))
    samples[0, :] = np.nan            # no valid member
    samples[1, 1:] = np.nan           # one
    samples[2, 20:] = np.nan          # twenty
    assert sf.left_out(samples, sf.mae) == (0.25, (40 + 39 + 20) / 160)
    assert sf.left_out(samples, sf.crps)[0] == 0.5
    assert sf.left_out(samples, sf.neg_log_likelihood)[0] == 0.75
    assert sf.left_out(samples, lambda t, s: 0.0)[0] == 0.25      # a scorer without min_valid needs 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sf.warn_left_out([(0.0, 0.0), (0.2, 0.3), (0.5, 0.6)], ["a", "b", "c"], {1}, 0.05)
        sf.warn_left_out([(0.01, 0.02)], ["d"], {0}, 0.05)
    assert len(caught) == 1
    text = str(caught[0].message)
    assert "2 of 3" in text and "b: 20.0 %" in text and "<- the best" in text and "\n  a:" not in text

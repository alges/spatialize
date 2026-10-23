"""The scenario suite's bookkeeping, checked without computing any estimator."""
import pytest

from spatialize import scenarios
from spatialize.scenarios import engine


class _NoRunner:
    """A runner that supports no estimator, so every check is skipped before anything is computed."""
    name = "none"

    def supports(self, est):
        return False

    def profile(self, encoder):
        return encoder


def test_proportions_read_from_the_same_draws_are_tested_at_their_level_with_their_covariance():
    """Proportions read from the same draws are correlated: given their covariance, the test rejects a
    true null hypothesis at its level, and indicators that always agree count once (issue #10)."""
    import numpy as np
    from spatialize.scenarios.stats import families
    rng = np.random.default_rng(10)
    p0 = np.array([0.2, 0.35, 0.5, 0.5, 0.7])          # the two 0.5 indicators coincide at every draw
    cov = np.minimum.outer(p0, p0) - np.outer(p0, p0)  # indicators 1{U < p_i} of one uniform U per draw
    n, reps = 400, 3000
    naive, with_cov = [], []
    for _ in range(reps):
        p_hat = (rng.random(n)[:, None] < p0).mean(axis=0)
        naive.append(families.gof_proportions(p_hat, p0, n).p_value)
        r = families.gof_proportions(p_hat, p0, n, cov=cov)
        with_cov.append(r.p_value)
    assert "rank=4" in r.detail
    assert abs(np.mean(np.array(with_cov) < 0.05) - 0.05) < 0.015
    assert np.mean(np.array(naive) < 0.05) > 0.08


@pytest.mark.parametrize("sc", list(scenarios.catalog().values()), ids=lambda sc: sc.id)
def test_every_scenario_records_its_outcomes_under_the_ids_check_ids_names(sc):
    """External runners and the conformance tests look outcomes up by check_ids, so each evaluator must
    record exactly those ids, one per estimator for a check listing several (issue #9)."""
    rep = engine.run(sc, _NoRunner(), mode="ci", seed=1)
    assert sorted(o.check for o in rep.outcomes) == sorted(engine.check_ids(sc))

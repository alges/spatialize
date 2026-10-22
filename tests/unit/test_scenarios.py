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


@pytest.mark.parametrize("sc", list(scenarios.catalog().values()), ids=lambda sc: sc.id)
def test_every_scenario_records_its_outcomes_under_the_ids_check_ids_names(sc):
    """External runners and the conformance tests look outcomes up by check_ids, so each evaluator must
    record exactly those ids, one per estimator for a check listing several (issue #9)."""
    rep = engine.run(sc, _NoRunner(), mode="ci", seed=1)
    assert sorted(o.check for o in rep.outcomes) == sorted(engine.check_ids(sc))

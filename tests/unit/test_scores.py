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
    from spatialize.gs.esi import scoring as sc
    from spatialize.gs.esi.pareto import EmpiricalRobustnessBound, reference_laws
    from spatialize import session
    rng = np.random.default_rng(cases.GENERATOR_SEED)
    members, truth = rng.normal(size=(40, 60)), rng.normal(size=40)

    model, _ = FittedModelFactory().create(members[0] * scale)
    reference, _ = FittedModelFactory().create(members[0])
    assert np.isclose(model.bandwidth_, reference.bandwidth_ * scale, rtol=1e-12)

    nll = sc.neg_log_likelihood(truth, members)
    assert np.isclose(sc.neg_log_likelihood(truth * scale, members * scale), nll + np.log(scale), rtol=1e-9)

    s, v, _ = DATA["2d"]
    with session.override(parallel=False):
        eps = EmpiricalRobustnessBound(30, cases.ALPHA, "mondrian", cases.SEED).estimate(s, reference_laws(s, v))
        eps_scaled = EmpiricalRobustnessBound(30, cases.ALPHA, "mondrian", cases.SEED).estimate(
            s, reference_laws(s, (v * scale).astype(np.float32)))
    assert np.isclose(eps, eps_scaled, rtol=1e-4)


def test_left_out_counts_the_data_a_scorer_cannot_use():
    """A datum is left out of a score when it has fewer valid members than the scorer needs (1 for
    the absolute and squared errors, 30 for the NLL, 2 for the CRPS); the warning of the searches
    lists every configuration over the threshold and marks the best."""
    import warnings
    from spatialize.gs.esi import scoring as sc
    samples = np.ones((4, 40))
    samples[0, :] = np.nan            # no valid member
    samples[1, 1:] = np.nan           # one
    samples[2, 20:] = np.nan          # twenty
    assert sc.left_out(samples, sc.mae) == (0.25, (40 + 39 + 20) / 160)
    assert sc.left_out(samples, sc.crps)[0] == 0.5
    assert sc.left_out(samples, sc.neg_log_likelihood)[0] == 0.75
    assert sc.left_out(samples, lambda t, s: 0.0)[0] == 0.25      # a scorer without min_valid needs 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sc.warn_left_out([(0.0, 0.0), (0.2, 0.3), (0.5, 0.6)], ["a", "b", "c"], {1}, 0.05)
        sc.warn_left_out([(0.01, 0.02)], ["d"], {0}, 0.05)
    assert len(caught) == 1
    text = str(caught[0].message)
    assert "2 of 3" in text and "b: 20.0 %" in text and "<- the best" in text and "\n  a:" not in text


def test_encoder_error_reads_the_cells_alone():
    """The encoder error compares laws fitted once for the search, the largest divergence of a
    partition averaged over the partitions (the book's empirical robustness level, per partition).
    So it does not change with the decoder, it falls as the cells shrink, and it does not grow with
    the number of partitions; laws given as a matrix are those of the default when it holds the same
    draws."""
    from scipy.spatial import cKDTree
    from spatialize import session
    from spatialize.gs.esi import esi_pareto_hparams_search
    from spatialize.gs.esi.pareto import EmpiricalRobustnessBound, reference_laws
    from spatialize.logging import singleton_null_callback
    s, v, _ = DATA["2d"]
    with session.override(parallel=False):
        r = esi_pareto_hparams_search(s, v, local_interpolator="idw", n_partitions=[30, 120], alpha=[0.3, 0.95],
                                      exponent=[1.0, 3.0], scoring=lambda t, m: 0.0, k=5, seed=cases.SEED,
                                      folding_seed=cases.FSEED, callback=singleton_null_callback)
        eps = {(x["params"]["n_partitions"], x["params"]["alpha"], x["params"]["exponent"]): x["epsilon"]
               for x in r.all_results}
        for T in (30, 120):
            for a in (0.3, 0.95):
                assert eps[(T, a, 1.0)] == eps[(T, a, 3.0)]
            assert eps[(T, 0.95, 1.0)] < 0.5 * eps[(T, 0.3, 1.0)]
        assert 0.7 < eps[(120, 0.3, 1.0)] / eps[(30, 0.3, 1.0)] < 1.4

        _, nb = cKDTree(s).query(s, k=17)
        given = EmpiricalRobustnessBound(30, 0.8, "mondrian", cases.SEED).estimate(
            s, reference_laws(s, v, laws=np.asarray(v, float)[nb[:, 1:]]))
        default = EmpiricalRobustnessBound(30, 0.8, "mondrian", cases.SEED).estimate(s, reference_laws(s, v))
    assert given == default


def test_divergence_of_far_laws_has_no_ceiling():
    """Two laws that barely overlap diverge by a large, finite amount, close to the Gaussians' closed
    form (separation 8 standard deviations: 32 nats; the kernels fall a little faster beyond the
    farthest draws, so within a fifth), and not by a ceiling set by the grid."""
    from spatialize.gs.esi.pareto import reference_laws, _kl_from_models
    rng = np.random.default_rng(cases.GENERATOR_SEED)
    draws = np.vstack([rng.normal(0.0, 1.0, 4000), rng.normal(8.0, 1.0, 4000)])
    laws = reference_laws(np.zeros((2, 2)), np.zeros(2), laws=draws)
    assert 32 * 0.9 < _kl_from_models(laws[0], laws[1]) < 32 * 1.2


def test_leave_one_out_is_named_the_same_in_every_search():
    """Every cross-validated search takes ``k="loo"`` for leave-one-out, as ``-1`` and the number of
    data, with the same result."""
    from spatialize.gs.esi import esi_hparams_search
    from spatialize.gs.idw import idw_hparams_search
    from spatialize.logging import singleton_null_callback
    s, v, q = DATA["2d"]
    common = dict(griddata=False, n_partitions=[cases.T], alpha=[cases.ALPHA], exponent=[cases.EXP],
                  seed=cases.SEED, callback=singleton_null_callback)
    errors = [esi_hparams_search(s, v, q, k=k, **common).search_result_data["cv_error"].values
              for k in ("loo", -1, len(v))]
    assert np.array_equal(errors[0], errors[1]) and np.array_equal(errors[0], errors[2])
    idw = [idw_hparams_search(s, v, q, k=k, griddata=False, exponent=[cases.EXP], callback=singleton_null_callback)
           .search_result_data["cv_error"].values for k in ("loo", -1)]
    assert np.array_equal(idw[0], idw[1])


def test_alpha_near_one_warns_only_on_the_partition_of_version_1_2():
    """Near alpha = 1 the partition of version 1.2 multiplies its cells, so it warns; the theory's
    Mondrian process and Voronoi do not."""
    import warnings
    from spatialize.gs.esi import esi_nongriddata
    from spatialize.logging import singleton_null_callback
    s, v, q = DATA["2d"]
    for p_process, warns in (("mondrian", False), ("voronoi", False), ("mondrian-legacy", True)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            esi_nongriddata(s, v, q, local_interpolator="idw", exponent=cases.EXP, n_partitions=5, alpha=0.97,
                            p_process=p_process, seed=cases.SEED, callback=singleton_null_callback)
        assert any("very close to 1" in str(w.message) for w in caught) == warns, p_process


def test_a_point_mass_law_gets_the_spread_floor():
    """Where a datum's neighbours are all alike (a flat surface), its reference law gets the floor
    of spread, a fraction of the data's standard deviation, so its divergence from another law is
    that of two Gaussians at that scale and not an artefact of a vanishing bandwidth."""
    from spatialize.gs.esi.pareto import reference_laws, _kl_from_models
    rng = np.random.default_rng(cases.GENERATOR_SEED)
    values = rng.normal(0.0, 10.0, 200)
    draws = np.vstack([np.full(400, 0.0), rng.normal(2.0, 0.5, 400)])
    laws = reference_laws(np.zeros((200, 2)), values, laws=np.vstack([draws, np.zeros((198, 400))]),
                          min_spread=0.05)
    floor = 0.05 * np.std(values)
    assert np.isclose(laws[0][4], floor)
    gauss = np.log(laws[1][4] / floor) + (floor ** 2 + 4.0) / (2 * laws[1][4] ** 2) - 0.5
    assert 0.8 * gauss < _kl_from_models(laws[0], laws[1]) < 1.2 * gauss


def test_crps_is_the_exact_score_of_the_members():
    import spatialize.gs.esi.scoring as sc
    """E|X - y| - E|X - X'| / 2 over the members: brute force, an atom at the observation, a normal law."""
    rng = np.random.default_rng(1)
    members, observed = rng.random((5, 40)), rng.random(5)
    brute = np.mean([np.mean(np.abs(m - y)) - 0.5 * np.mean(np.abs(m[:, None] - m[None]))
                     for m, y in zip(members, observed)])
    assert abs(sc.crps(observed, members) - brute) < 1e-12
    assert sc.crps(np.array([0.0]), np.zeros((1, 10))) == 0.0
    normal = rng.standard_normal((1, 200_000))
    assert abs(sc.crps(np.array([0.0]), normal) - (2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))) < 0.005


@pytest.mark.parametrize("name", ["crps", "mae", "rmse", "nll"])
def test_pareto_search_takes_the_scoring_by_name(name):
    import spatialize
    from spatialize.gs.esi import esi_pareto_hparams_search
    rng = np.random.default_rng(0)
    points, values = rng.random((60, 2)), rng.random(60)
    with spatialize.session.override(progress=False):
        search = esi_pareto_hparams_search(points, values, local_interpolator="idw", n_partitions=[50],
                                           alpha=[0.7], exponent=[2.0], k="loo", scoring=name, seed=0)
    assert np.isfinite(search.all_results[0]["decoder_error"])


@pytest.mark.parametrize("over,under", [(1.0, 1.0), (5.0, 1.0), (1.0, 3.0)])
@pytest.mark.parametrize("kind", ["linear", "squared"])
def test_asymmetric_loss_reading_minimises_the_expected_loss(over, under, kind):
    """For a fixed law, the reading AsymmetricLoss returns has the smallest expected loss among nearby
    readings, a quantile for the linear loss and an expectile for the squared one, both at level
    under / (over + under)."""
    from spatialize.gs.esi import functionals as fn
    rng = np.random.default_rng(0)
    members = rng.lognormal(0, 1, (1, 20_000))
    reading = fn.AsymmetricLoss(over, under, kind)(members)[0]

    def expected(r):
        excess, shortfall = np.clip(r - members, 0, None), np.clip(members - r, 0, None)
        if kind == "squared":
            excess, shortfall = excess ** 2, shortfall ** 2
        return np.mean(over * excess + under * shortfall)
    assert all(expected(reading) <= expected(reading + d) + 1e-12 for d in (-0.05, -0.01, 0.01, 0.05))
    alpha = under / (over + under)
    target = np.quantile(members, alpha) if kind == "linear" else fn.Expectile(alpha)(members)[0]
    assert abs(reading - target) < 1e-9


def test_expectile_and_expected_shortfall():
    from spatialize.gs.esi import functionals as fn
    rng = np.random.default_rng(1)
    members = rng.normal(0, 1, (3, 5000))
    members[0, :10] = np.nan
    assert np.allclose(fn.Expectile(0.5)(members), np.nanmean(members, axis=1), atol=1e-9)
    e = fn.Expectile(0.3)(members)
    for row, value in zip(members, e):
        row = row[np.isfinite(row)]
        assert abs(0.3 * np.mean(np.clip(row - value, 0, None)) - 0.7 * np.mean(np.clip(value - row, 0, None))) < 1e-9
    es = fn.ExpectedShortfall(0.9)(members)
    for row, value in zip(members, es):
        row = row[np.isfinite(row)]
        assert abs(value - row[row >= np.quantile(row, 0.9)].mean()) < 1e-12
    assert np.all(fn.ExpectedShortfall(0.1, tail="lower")(members) < fn.Expectile(0.5)(members))


def test_asymmetric_loss_score_in_a_search():
    import spatialize
    from spatialize.gs.esi import esi_pareto_hparams_search, scoring as sc
    rng = np.random.default_rng(0)
    points, values = rng.random((60, 2)), rng.lognormal(0, 1, 60)
    with spatialize.session.override(progress=False):
        search = esi_pareto_hparams_search(points, values, local_interpolator="wdraw_idw", n_partitions=[50],
                                           alpha=[0.7], exponent=[2.0], k="loo",
                                           scoring=sc.asymmetric_loss(over=5, under=1), seed=0)
    assert np.isfinite(search.all_results[0]["decoder_error"])


def test_renamed_modules_keep_their_old_names_until_1_4():
    """The modules renamed in 1.3 (aggfunction, lossfunction, scorefunction, and the categorical
    agg_functions and score_functions) are still importable under their old names, which give the new
    modules themselves with a FutureWarning."""
    import importlib
    import sys
    import warnings
    pairs = [("spatialize.gs.esi.aggfunction", "spatialize.gs.esi.functionals"),
             ("spatialize.gs.esi.lossfunction", "spatialize.gs.esi.loss"),
             ("spatialize.gs.esi.scorefunction", "spatialize.gs.esi.scoring"),
             ("spatialize.gs.cat_esi.agg_functions", "spatialize.gs.cat_esi.functionals"),
             ("spatialize.gs.cat_esi.score_functions", "spatialize.gs.cat_esi.scoring")]
    for old, new in pairs:
        sys.modules.pop(old, None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            module = importlib.import_module(old)
        assert module is importlib.import_module(new)
        assert any(issubclass(w.category, FutureWarning) and new in str(w.message) for w in caught)

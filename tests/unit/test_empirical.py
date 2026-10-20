"""Distributional readings of the ensemble (spatialize.empirical).

Unit tests on this checkout's in-place build (see tests/unit/conftest.py)."""
import warnings

import numpy as np
import pytest

import cases


@pytest.mark.parametrize("point_model", ["kde", "emm", "vim"])
def test_a_prefitted_model_gives_the_law_of_its_sample(point_model):
    """EmpiricalModel built from a fitted model (skl_model) spans the model's range and reads the
    same law as one built from the sample; KernelDensity.sample returns X alone, the mixtures
    (X, labels), and taking the first row of X had collapsed the grid of a KDE to one point."""
    from spatialize.empirical import FittedModelFactory, EmpiricalModel
    rng = np.random.default_rng(cases.GENERATOR_SEED)
    s = rng.normal(size=400)
    factory = FittedModelFactory(point_model_name=point_model, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = factory.create(s)
        em = EmpiricalModel(skl_model=model)
    x = np.linspace(-2, 2, 41)
    assert np.all(np.abs(np.asarray(em.cdf(x)) - np.mean(s[:, None] <= x, axis=0)) < 0.1)


def test_a_near_degenerate_mixture_component_still_gives_a_law():
    """A Gaussian mixture fitted to a few members can put a component of variance 1e-6 on an
    isolated value: the cumulative probability then has plateaus whose steps of 1e-17 gave its
    inverse infinite slopes, and the model could not be built. It now reads a valid law."""
    from spatialize.empirical import FittedModelFactory, EmpiricalModel
    s = np.array([0.76667386, 1.59646392, 1.64215267, 1.64677191, 1.64998078, 1.6649102,
                  1.66730094, 1.67172146, 1.67935383, 1.72108686])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        em = EmpiricalModel(sample=s, fitted_model_factory=FittedModelFactory(point_model_name="emm", seed=1))
    F = np.asarray(em.cdf(np.linspace(0, 3, 3001)))
    assert np.all((F >= 0) & (F <= 1)) and np.all(np.diff(F) >= 0)
    q = np.asarray(em.inv_cdf(np.array([0.05, 0.5, 0.95])))
    assert np.all(np.isfinite(q)) and np.all(np.diff(q) > 0)


@pytest.mark.parametrize("point_model", ["emm", "vim"])
def test_mixture_draws_differ_between_locations(point_model):
    """ess_sample draws each location from its own law with its own seed, so two locations with the
    same members get different draws (with the factory's seed at every location they were identical)."""
    from spatialize.empirical import FittedModelFactory
    from spatialize.gs.ess import ess_sample
    from spatialize.gs.esi import esi_nongriddata
    from spatialize.logging import singleton_null_callback as quiet
    rng = np.random.default_rng(0)
    p = rng.random((60, 2)); v = rng.normal(size=60)
    q = np.array([[0.5, 0.5], [0.5, 0.5]])                   # one place, listed twice
    r = esi_nongriddata(p, v, q, local_interpolator="idw", exponent=2.0, n_partitions=50, alpha=0.7, seed=0)
    sims = np.asarray(ess_sample(r, n_sims=40, callback=quiet,
                                 fitted_model_factory=FittedModelFactory(point_model_name=point_model, seed=3)).scenarios)
    sims = sims.reshape(2, -1)
    assert not np.allclose(sims[0], sims[1])


@pytest.mark.parametrize("sigma", [1.0, 0.1])
def test_entropy_is_the_differential_entropy(sigma):
    """The entropy of a normal law is log(sigma * sqrt(2 pi e)), negative below sigma ~ 0.242; the
    model fitted to a large sample comes close."""
    from spatialize.empirical import EmpiricalModel
    s = np.random.default_rng(0).normal(0.0, sigma, 20000)
    expected = np.log(sigma * np.sqrt(2 * np.pi * np.e))
    assert abs(EmpiricalModel(sample=s).entropy() - expected) < 0.05


def test_auto_widening_follows_the_sign_of_the_data():
    """With widening="auto", ess_sample chooses the family once for the variable: data that take negative
    values get a skew-normal widening everywhere, even where a location's members are all positive (a
    Gamma law there came out sharply peaked, its entropy several bits too low)."""
    from spatialize.empirical import FittedModelFactory
    from spatialize.gs.esi import esi_nongriddata
    from spatialize.gs.ess import ess_sample
    from spatialize.logging import singleton_null_callback as quiet
    rng = np.random.default_rng(0)
    p = rng.random((400, 2))
    spread = lambda x: 0.1 + 1.5 * np.linalg.norm(x - 0.5, axis=1)
    q = np.array([[0.5, 0.5], [0.52, 0.5], [0.48, 0.52]])            # near the centre, where the spread is small
    r = esi_nongriddata(p, rng.normal(0, spread(p)), q, local_interpolator="idw", exponent=2.0, alpha=0.8,
                        n_partitions=300, seed=0)
    auto = np.asarray(ess_sample(r, n_sims=300, callback=quiet, fitted_model_factory=FittedModelFactory(widening="auto", seed=0)).scenarios)
    skew = np.asarray(ess_sample(r, n_sims=300, callback=quiet, fitted_model_factory=FittedModelFactory(widening="skew_normal", seed=0)).scenarios)
    assert np.allclose(auto, skew)


def _widening_case(T, values_shift=0.0, members=None, seed=0):
    """An ESI result built by hand: 200 data, 6 locations, members sharing one draw per member across
    the locations (perfectly dependent), so that what widening keeps of that dependence shows."""
    from spatialize.gs.esi import ESIResult
    rng = np.random.default_rng(seed)
    p = rng.random((200, 2))
    v = rng.normal(0, 1, 200) + values_shift
    q = np.array([[0.3, 0.3], [0.32, 0.3], [0.3, 0.32], [0.7, 0.7], [0.72, 0.7], [0.7, 0.72]])
    if members is None:
        members = values_shift + 0.3 * np.tile(rng.normal(size=T), (len(q), 1))
    return ESIResult(members.mean(axis=1), members, xi=q, points=p, values=v), q


@pytest.mark.parametrize("family,shift", [("skew_normal", 0.0), ("gamma", 5.0)])
def test_widened_members_have_the_moments_of_the_widened_law(family, shift):
    """Each location's widened members keep the members' mean and reach the target variance of the
    nearest data, as the mixture of Definition 12.3.2 (draft) does; Gamma members stay positive."""
    from spatialize.empirical import _local_target_variance
    from spatialize.gs.ess import widen_members
    r, q = _widening_case(20000, values_shift=shift)
    w = widen_members(r, widening=family, seed=1).esi_samples(raw=True)
    raw = r.esi_samples(raw=True)
    target = _local_target_variance(r.points, r.values, q, knn=12)
    assert np.allclose(w.mean(axis=1), raw.mean(axis=1), atol=0.03)
    assert np.allclose(w.var(axis=1), target, rtol=0.05)
    if family == "gamma":
        assert np.all(w > 0)


def test_widened_members_keep_their_index():
    """Column t stays the widened version of member t: with members shared by every location, the average
    over the locations keeps the members' variance and adds the widening noise divided by their number,
    while drawing each location independently (as ess_sample does) would lose the members' part."""
    from spatialize.empirical import _local_target_variance
    from spatialize.gs.ess import widen_members
    r, q = _widening_case(20000)
    raw = r.esi_samples(raw=True)
    w = widen_members(r, widening="skew_normal", seed=1).esi_samples(raw=True)
    target = _local_target_variance(r.points, r.values, q, knn=12)
    kept = raw.mean(axis=0).var() + (target - raw.var(axis=1)).sum() / len(q) ** 2
    independent = target.sum() / len(q) ** 2
    assert kept > 1.2 * independent
    assert abs(w.mean(axis=0).var() / kept - 1) < 0.05
    shuffled = np.array([np.random.default_rng(i).permutation(row) for i, row in enumerate(w)])
    assert abs(shuffled.mean(axis=0).var() / independent - 1) < 0.05


def test_widening_leaves_wide_and_undefined_members_alone():
    """Members spreading more than the target are kept as they are, NaN members stay NaN, the draws
    repeat with the seed and the original result is not modified."""
    from spatialize.gs.ess import widen_members
    rng = np.random.default_rng(3)
    members = 0.3 * rng.normal(size=(6, 500))
    members[3:] *= 100                                    # far wider than the data near them
    members[0, :10] = np.nan
    r, _ = _widening_case(500, members=members.copy())
    a = widen_members(r, seed=7).esi_samples(raw=True)
    b = widen_members(r, seed=7).esi_samples(raw=True)
    assert np.array_equal(a, b, equal_nan=True)
    assert np.array_equal(a[3:], members[3:])
    assert np.isnan(a[0, :10]).all() and np.isfinite(a[0, 10:]).all()
    assert not np.allclose(a[1:3], members[1:3])
    assert np.array_equal(r.esi_samples(raw=True), members, equal_nan=True)


def test_widen_members_needs_the_data():
    from spatialize.gs.esi import ESIResult
    from spatialize.gs.ess import widen_members
    m = np.zeros((2, 5))
    with pytest.raises(ValueError):
        widen_members(ESIResult(m.mean(axis=1), m, xi=np.zeros((2, 2))))
    r, _ = _widening_case(10)
    with pytest.raises(ValueError):
        widen_members(r, widening="normal")

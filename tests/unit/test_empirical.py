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

"""Posterior analysis of the data (spatialize.gs.spa).

Unit tests on this checkout's in-place build (see tests/unit/conftest.py)."""
import numpy as np
import pytest

import cases
import snapshot_lib as sl

LIB = sl.load_lib()
DATA = cases.datasets()


def test_posterior_audit_builds_each_law_from_the_other_data():
    """posterior_audit computes the same members as the 1.2 function given the data as queries; the
    law of a datum is fitted to its defined members alone, never to the datum; a cumulative
    distribution function is 0 below its grid and 1 above, so a datum far outside its law gets
    0 or 1 and not NaN."""
    from spatialize.gs.spa import posterior_audit, cv_sample_pred_posterior
    from spatialize.empirical import FittedModelFactory
    s, v, _ = DATA["2d"]
    v = v.copy()
    v[3] = v.max() + 10 * (v.max() - v.min())          # far above every other value
    kw = dict(local_interpolator="idw", exponent=2.0, n_partitions=cases.T, alpha=cases.ALPHA, seed=cases.SEED)
    audit = posterior_audit(s, v, fitted_model_factory=FittedModelFactory(), **kw)
    old = cv_sample_pred_posterior(s, v, s, fitted_model_factory=FittedModelFactory(), **kw)
    assert np.array_equal(audit.members, old.members, equal_nan=True)
    assert np.array_equal(audit.support, np.isfinite(audit.members).mean(axis=1))
    model = audit.model(3)
    assert np.array_equal(np.sort(model.data_), np.sort(audit.members[3][np.isfinite(audit.members[3])]))
    assert old.sample_quantiles[3] == 1.0
    assert model.cdf(model.x_[0] - 1.0) == 0.0 and model.pdf(model.x_[-1] + 1.0) == 0.0


def test_posterior_audit_readings_are_consistent():
    """The readings of the posterior audit agree with each other: the flags are those of the
    Benjamini–Hochberg procedure on the p-values; the levels follow the positions; the p-values
    never reach 0 and are 1 at most; the 1.2 ranking is the probability levels; the spread factor
    brings the 90 % coverage of the widened laws to nominal; every scale maps back to the values;
    every tail model gives a reading at every datum."""
    from spatialize.gs.spa import posterior_audit, cv_sample_pred_posterior
    from spatialize.gs.spa._main import _scale_maps
    from spatialize.empirical import FittedModelFactory
    s, v, _ = DATA["2d"]
    v = np.exp(v - v.mean()).astype(np.float32)
    kw = dict(n_partitions=cases.T, alpha=cases.ALPHA, seed=cases.SEED)
    for options in ({}, {"tails": "gpd"}, {"tails": "normal", "calibrate": False}, {"scale": "yeojohnson"},
                    {"scale": "normal_scores"}):
        audit = posterior_audit(s, v, **kw, **options)
        t = audit.table()
        p = t.tail_p.to_numpy()
        assert np.all((p > 0) & (p <= 1))
        order = np.argsort(p)
        k = np.flatnonzero(p[order] <= 0.05 * np.arange(1, len(p) + 1) / len(p))
        expected = np.zeros(len(p), bool)
        if k.size:
            expected[order[:k[-1] + 1]] = True
        assert np.array_equal(t.flag.to_numpy(), expected)
        u = t.pit.to_numpy()
        outside99 = (u < 0.005) | (u > 0.995)
        assert np.array_equal(t.level.to_numpy() == "level_0", outside99)
        assert np.isfinite(t[["pit", "tail_p", "surprisal", "entropy"]].to_numpy()).all()
        fwd, inv = _scale_maps(audit.scale, audit.values)
        assert np.allclose(inv(fwd(audit.values)), audit.values, rtol=1e-6, atol=1e-9)
        if audit.calibrate:
            laws, z = audit._widened(), audit._tvalues
            c = audit.spread_factor
            pit = np.array([np.mean(np.median(x) + c * (x - np.median(x)) < zi) for x, zi in zip(laws, z)])
            assert abs(np.mean((pit >= 0.05) & (pit <= 0.95)) - 0.9) <= 2.0 / len(z)
    old = cv_sample_pred_posterior(s, v, s, fitted_model_factory=FittedModelFactory(), **kw)
    assert list(old.rank_samples().category) == list(old.levels())


def test_posterior_audit_declustering_and_neighbours():
    """The declustering weights sum to 1, come from the partitions of the cross-validation (the cells
    of the data read by cells() are those of the members' partitions: a datum alone in its cell has
    no member there) and weigh a duplicated datum half of its single copy; the plain summaries are
    those of numpy; the coherence lies in [0, 1]; the duplicates are the co-located pairs; an audit
    built without partitions refuses the weights."""
    from spatialize import SpatializeError
    from spatialize.gs.spa import posterior_audit, PosteriorAudit
    s, v, _ = DATA["2d"]
    kw = dict(n_partitions=cases.T, alpha=cases.ALPHA, seed=cases.SEED)
    audit = posterior_audit(s, v, **kw)
    w = audit.weights(n_probes=5000)
    assert np.isclose(w.sum(), 1.0) and np.all(w > 0)
    data_cells, _ = audit._cells(10, audit.seed)
    alone = np.array([[np.sum(data_cells[:, t] == data_cells[i, t]) == 1 for t in range(data_cells.shape[1])]
                      for i in range(len(v))])
    assert np.array_equal(alone, np.isnan(audit.members))
    d = audit.declustered()
    assert np.isclose(d.loc["mean", "naive"], np.mean(audit.values))
    c = audit.coherence()
    assert np.all((c[np.isfinite(c)] >= 0) & (c[np.isfinite(c)] <= 1))
    s2, v2 = np.vstack([s, s[:1]]), np.append(v, v[0])
    dup = posterior_audit(s2, v2, **kw)
    w2 = dup.weights(n_probes=5000)
    assert np.isclose(w2[0], w2[-1])
    pairs = dup.duplicates()
    assert len(pairs) >= 1 and (pairs.i.iloc[0], pairs.j.iloc[0]) == (0, len(v)) and pairs.difference.iloc[0] == 0
    bare = PosteriorAudit(audit.members, s, v)
    with pytest.raises(SpatializeError):
        bare.weights()


def test_posterior_audit_shift_is_the_neighbours_mean_signed_position():
    """The shift of a datum is the mean of 2u - 1 over its k nearest other data, within [-1, 1]."""
    from scipy.spatial import cKDTree
    from spatialize.gs.spa import posterior_audit
    s, v, _ = DATA["2d"]
    audit = posterior_audit(s, v, n_partitions=cases.T, alpha=cases.ALPHA, seed=cases.SEED)
    u = audit.pit()
    _, idx = cKDTree(np.asarray(s, float)).query(np.asarray(s, float), k=4)
    expected = np.array([np.mean(2 * u[[j for j in row if j != i][:3]] - 1) for i, row in enumerate(idx)])
    sh = audit.shift(k=3)
    assert np.allclose(sh, expected) and np.all(np.abs(sh) <= 1)

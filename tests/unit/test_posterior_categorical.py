"""Posterior analysis of categorical data (spatialize.gs.spa.cat_posterior_audit).

Unit tests on this checkout's in-place build (see tests/unit/conftest.py)."""
import numpy as np
import pytest


def _golden(n=80, seed=3):
    from spatialize.data import load_lithology_golden
    samples, _, _ = load_lithology_golden(n_samples=n, seed=seed)
    return samples


def _audit(points, values, **kw):
    from spatialize.gs.spa import cat_posterior_audit
    from spatialize.logging import singleton_null_callback as quiet
    return cat_posterior_audit(points, values, n_partitions=30, alpha=0.8, seed=1, callback=quiet, **kw)


def test_categorical_laws_and_readings_are_consistent():
    """Every law sums to 1 over the classes and the mixture weights to 1; the positions lie in [0, 1] and
    are the p-values of nominal classes, while ordinal classes read them on both sides; the q-values and
    the flags are those of Benjamini–Hochberg; the review list holds the most surprising data in order,
    with the number expected by chance; the information check is the mean excess of the table, its
    variance exact; a datum's law never holds its own class among the members."""
    from spatialize.gs.spa._main import _bh, _qvalues
    s = _golden()
    pts, lith, era = s[["x", "y"]].values, s["lithology"].values, s["era"].values
    a = _audit(pts, lith)
    P = a.laws().to_numpy()
    assert np.allclose(P.sum(axis=1), 1) and np.isclose(a.mixture.sum(), 1) and (a.mixture >= 0).all()
    u, p = a.pit(), a.tail_p()
    assert np.all((u >= 0) & (u <= 1)) and np.allclose(np.clip(u, np.finfo(float).tiny, 1), p)
    t = a.table()
    assert np.array_equal(t.flag, _bh(p, 0.05)) and np.allclose(t.q_value, _qvalues(p))
    assert np.allclose(t.probability, P[np.arange(len(lith)), a.codes])
    rev = a.review(share=0.1)
    assert len(rev) == 8 and rev.tail_p.is_monotonic_increasing
    assert np.allclose(rev.expected_by_chance, len(lith) * rev.tail_p)
    info = a.calibration(information=True)["information"]
    assert np.isclose(info["mean_excess"], t.excess.mean())
    from spatialize.data import ERAS_GOLDEN
    o = _audit(pts, era, ordinal_order=ERAS_GOLDEN)
    uo = o.pit()
    assert o.ordinal and np.allclose(o.tail_p(), np.clip(np.minimum(1, 2 * np.minimum(uo, 1 - uo)), np.finfo(float).tiny, 1))
    assert list(o.classes) == [str(e) for e in ERAS_GOLDEN]


def test_categorical_audit_lists_a_mislabel_and_a_conflicting_pair():
    """A mislabelled datum is among the data to check first or to measure again; a co-located pair of
    different classes is listed to classify again; declustered class shares sum to 1; cross reads a numeric
    and a categorical covariate, its flags and order entering the list; null_flags counts the flags of
    replicates; readings of other data or at another rate are refused; the map draws the actions."""
    import matplotlib
    matplotlib.use("Agg")
    from spatialize import SpatializeError
    s = _golden()
    pts, lith = s[["x", "y"]].values, s["lithology"].values.astype(object)
    classes = sorted(set(lith))
    i = int(np.argmax(s["lithology"].values == "gneiss"))
    lith[i] = "basalt" if "basalt" in classes else classes[0]
    pts2, lith2 = np.vstack([pts, pts[5]]), np.append(lith, [c for c in classes if c != lith[5]][0])
    a = _audit(pts2, lith2)
    assert np.isclose(a.declustered().declustered.sum(), 1)
    cross_num = a.cross(np.append(s["age_ma"].values, s["age_ma"].values[5]).astype(float))
    cross_cat = a.cross(np.append(s["era"].values, s["era"].values[5]))
    assert cross_num.q_value.notna().all() and cross_cat.q_value.notna().all()
    null = a.null_flags(n_replicates=2)
    rec = a.recommend(share=0.1, cross=cross_num, null=null)
    action = dict(zip(rec.datum, rec.action))
    assert action.get(i) in ("measure again", "check first", "check provenance")
    assert {5, len(lith2) - 1} <= set(action)
    pair = rec.set_index("datum").loc[[5, len(lith2) - 1]]
    assert all("co-located with datum" in r for r in pair.reason)
    assert rec.attrs["context"]["observed"] == null.attrs["observed"]
    with pytest.raises(SpatializeError):
        a.recommend(cross=a.cross(np.zeros(len(lith2)), q=0.1))
    with pytest.raises(SpatializeError):
        _audit(pts, lith).recommend(null=null)
    legend = [x.get_text() for x in a.plot_recommendations(rec).axes[0].get_legend().get_texts()]
    assert any(x.startswith("check first (") or x.startswith("measure again (") for x in legend)

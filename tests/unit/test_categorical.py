"""Categorical ESI: majority vote (the default) and the Dawid-Skene alternative.

Unit tests on this checkout's in-place build (see tests/unit/conftest.py)."""
import warnings

import numpy as np
import pytest

import cases


def _annotators(n_items, accuracies, n_classes=3, biased=0, seed=cases.GENERATOR_SEED):
    """A true category per item and one label per annotator, right with the annotator's accuracy and
    uniform among the classes otherwise; ``biased`` more annotators always answer "0"."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, n_classes, n_items).astype(str)
    labels = np.empty((n_items, len(accuracies) + biased), dtype=object)
    for j, acc in enumerate(accuracies):
        right = rng.random(n_items) < acc
        labels[:, j] = np.where(right, truth, rng.integers(0, n_classes, n_items).astype(str))
    labels[:, len(accuracies):] = "0"
    return truth, labels


def test_dawid_skene_is_built_and_reproducible():
    """The compiled module ships with Spatialize; a seed fixes EM's initialisation, and the posterior
    probabilities form a (locations, categories) array whose rows sum to one."""
    from spatialize.gs.cat_esi.functionals import aggregate_with_btd
    _, labels = _annotators(150, [0.8] * 12)
    a = aggregate_with_btd(labels, categories_list=["0", "1", "2"], seed=3)
    b = aggregate_with_btd(labels, categories_list=["0", "1", "2"], seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    assert a[1].shape == (150, 3) and np.allclose(a[1].sum(axis=1), 1.0)
    assert np.array_equal(a[0], np.array(["0", "1", "2"], dtype=object)[a[1].argmax(axis=1)])


def test_dawid_skene_weighs_reliable_partitions_where_majority_vote_cannot():
    """Ten reliable annotators and fifteen that always answer "0": the majority vote follows the
    biased ones, Dawid-Skene learns each annotator's confusion matrix and follows the reliable ones."""
    from spatialize.gs.cat_esi.functionals import aggregate_with_btd, aggregate_with_mv
    truth, labels = _annotators(400, [0.9] * 10, biased=15)
    mv = np.mean(aggregate_with_mv(labels) == truth)
    btd = np.mean(aggregate_with_btd(labels, categories_list=["0", "1", "2"], seed=1)[0] == truth)
    assert btd > 0.9 and btd > mv + 0.3


@pytest.mark.parametrize("categories,kind,order", [(["0", "1"], "binary", None), (["0", "1", "2"], "nominal", None),
                                                   (["0", "1", "2"], "ordinal", {"0": 0, "1": 1, "2": 2})])
def test_dawid_skene_kinds_of_variables(categories, kind, order):
    """Binary (a two-class nominal variable), nominal and ordinal variables all aggregate to one of
    their categories."""
    from spatialize.gs.cat_esi.functionals import aggregate_with_btd
    _, labels = _annotators(100, [0.85] * 10, n_classes=len(categories))
    est, prob, _ = aggregate_with_btd(labels, category_type=kind, categories_list=categories,
                                      ordinal_order_map=order, seed=2)
    assert set(est) <= set(categories) and prob.shape == (100, len(categories))


def test_dawid_skene_reports_through_the_visitor_protocol():
    """Messages, warnings and progress of the compiled code reach the callback as Spatialize's JSON
    protocol (an info message, a warning, one progress token per EM iteration, then done), never the
    console."""
    import json
    from spatialize.gs.cat_esi.functionals import aggregate_with_btd
    _, labels = _annotators(50, [0.8] * 6)
    msgs = []
    aggregate_with_btd(labels, category_type="ordinal", categories_list=["0", "1", "2"], seed=0,
                       callback=msgs.append)
    parsed = [json.loads(m) for m in msgs]
    levels = [m["message"]["level"] for m in parsed if "message" in m]
    assert "INFO" in levels and "WARNING" in levels
    assert any("progress" in m and isinstance(m["progress"], dict) and "init" in m["progress"] for m in parsed)
    assert parsed[-1] == {"progress": "done"}


def test_dawid_skene_does_not_depend_on_the_number_of_threads():
    """The E-step, the M-step and the log-likelihood keep the serial order of their sums under
    OpenMP: the same bits with one thread and with all."""
    from spatialize import session
    from spatialize.gs.cat_esi.functionals import aggregate_with_btd
    _, labels = _annotators(3000, [0.85] * 12, biased=8)
    out = []
    for threads in (1, None):
        with session.override(num_threads=threads):
            out.append(aggregate_with_btd(labels, categories_list=["0", "1", "2"], seed=5))
    assert np.array_equal(out[0][1], out[1][1]) and out[0][2] == out[1][2]


def test_majority_vote_stays_the_default_and_dawid_skene_is_an_alternative():
    """cat_esi's estimation is the majority vote of its members; re_estimate('btd') gives the
    Dawid-Skene alternative on the same members, and re_estimate('mv') returns to the default."""
    from spatialize.gs.cat_esi import cat_esi_nongriddata
    from spatialize.gs.cat_esi.functionals import aggregate_with_mv
    s, v, q = cases.datasets()["2d"]
    cats = np.where(v > np.median(v), "high", "low")
    r = cat_esi_nongriddata(s, cats, q, n_partitions=cases.T, alpha=cases.ALPHA, seed=cases.SEED)
    samples = r.esi_samples(raw=True)
    default = np.asarray(r.estimation()).astype(str)
    assert np.array_equal(default, np.asarray(aggregate_with_mv(samples)).astype(str))
    btd = np.asarray(r.re_estimate("btd", seed=1)).astype(str)
    assert set(btd) <= {"high", "low"}
    assert np.array_equal(np.asarray(r.re_estimate("mv")).astype(str), default)


def test_class_probabilities_are_the_members_shares():
    """The probability of each category is its share among the defined members; the majority vote
    is the category of largest probability."""
    from spatialize.gs.cat_esi import cat_esi_nongriddata
    s, v, q = cases.datasets()["2d"]
    cats = np.where(v > np.median(v), "high", "low")
    r = cat_esi_nongriddata(s, cats, q, n_partitions=cases.T, alpha=cases.ALPHA, seed=cases.SEED)
    names, prob = r.class_probabilities()
    assert names == ["high", "low"] and prob.shape == (len(q), 2)
    ok = np.isfinite(prob).all(axis=1)
    assert np.allclose(prob[ok].sum(axis=1), 1.0)
    est = np.asarray(r.estimation()).astype(str)
    clear = ok & (np.abs(prob[:, 0] - prob[:, 1]) > 1e-9)
    assert np.array_equal(est[clear], np.array(names)[prob[clear].argmax(axis=1)])


def test_spatial_dawid_skene_reads_the_neighbours():
    """Two locations tied between A and B, each voted by its own partitions: the one next to C, which A
    may not touch, leaves A once the penalty is strong enough; the one next to B keeps A. A penalty
    blind to the neighbours (the earlier implementation) moved both."""
    from spatialize.gs.cat_esi.functionals import aggregate_with_btd
    from spatialize.logging import singleton_null_callback
    row = ["C", "X1", "C", "A", "A", "B", "X2", "B", "C", "A", "B", "C"]
    labels = np.empty((len(row), 40), dtype=object)
    for i, c in enumerate(row):
        labels[i] = c
        if c == "X1":
            labels[i] = None
            labels[i, :10], labels[i, 10:20] = "A", "B"
        elif c == "X2":
            labels[i] = None
            labels[i, 20:30], labels[i, 30:40] = "A", "B"
    probs = {}
    for factor in (0.0, 5.0):
        _, p, _ = aggregate_with_btd(labels, categories_list=["A", "B", "C"], map_dimensions=(1, len(row)),
                                     spatial_constraints=[("A", "C")], spatial_penalty_factor=factor, seed=0,
                                     callback=singleton_null_callback)
        probs[factor] = p
    assert probs[0.0][1, 0] > 0.5 and probs[0.0][6, 0] > 0.5           # without penalty both take A
    assert probs[5.0][1, 0] < 0.5 and probs[5.0][6, 0] > 0.5           # next to C leaves A, next to B keeps it


def test_spatial_dawid_skene_does_not_depend_on_the_number_of_threads():
    """The mean-field E-step reads the previous responsibilities only: the same bits with one thread
    and with all, on a 30 × 30 grid."""
    from spatialize import session
    from spatialize.gs.cat_esi.functionals import aggregate_with_btd
    from spatialize.logging import singleton_null_callback
    _, labels = _annotators(900, [0.7] * 15, biased=5)
    out = []
    for threads in (1, None):
        with session.override(num_threads=threads):
            out.append(aggregate_with_btd(labels, categories_list=["0", "1", "2"], map_dimensions=(30, 30),
                                          spatial_constraints=[("0", "2")], spatial_penalty_factor=1.5,
                                          seed=4, callback=singleton_null_callback))
    assert np.array_equal(out[0][1], out[1][1]) and out[0][2] == out[1][2]


def test_codes_path_matches_the_claims_path():
    """aggregate_with_btd passes the members as integer codes (no string claims); the compiled
    module's claims path gives the same bits, spatial and with members of empty cells."""
    from spatialize.gs.cat_esi import _dawid_skene as ds
    from spatialize.gs.cat_esi.functionals import aggregate_with_btd
    from spatialize.logging import singleton_null_callback
    _, labels = _annotators(600, [0.8] * 10, biased=4)
    labels[::5, 2] = None
    dims = (20, 30)
    coords = {f"Item_{i + 1}": (i // dims[1], i % dims[1]) for i in range(600)}
    kw = dict(categories_list=["0", "1", "2"], spatial_constraints=[("0", "2")], spatial_penalty_factor=1.5)
    old = ds.aggregate_with_btd_em(esi_samples=labels.tolist(), map_dimensions=dims, item_coordinates=coords,
                                   seed=7, **kw)
    est, prob, ll = aggregate_with_btd(labels, map_dimensions=dims, seed=7, callback=singleton_null_callback, **kw)
    old_prob = np.array([[old[1][f"Item_{i + 1}"].get(c, 0.0) for c in ["0", "1", "2"]] for i in range(600)])
    assert np.array_equal(old_prob, prob) and old[2] == ll and list(old[0]) == list(est)


def test_weighted_majority_vote():
    """Equal weights give the majority vote; weights on the reliable partitions recover the truth where
    the majority vote follows a biased group; members of empty cells take no part."""
    from spatialize.gs.cat_esi.functionals import aggregate_with_mv, aggregate_with_weighted_mv
    truth, labels = _annotators(400, [0.9] * 10, biased=15)
    labels[::9, 0] = None
    assert np.array_equal(aggregate_with_weighted_mv(labels, np.ones(25)), aggregate_with_mv(labels))
    w = np.r_[np.ones(10), np.full(15, 0.1)]
    assert np.mean(aggregate_with_weighted_mv(labels, w) == truth) > 0.95 > np.mean(aggregate_with_mv(labels) == truth)
    with pytest.raises(ValueError):
        aggregate_with_weighted_mv(labels, np.ones(24))


def test_knn_pca_uses_the_anisotropy_of_the_categories():
    """Bands of categories across a direction at 30°: knn_pca finds that direction, stretches the
    distances across the bands (an anisotropic metric is chosen) and classifies better than the
    isotropic metric. Before 2026-10-10 the stretch was inverted, so cross-validation always chose the
    isotropic metric."""
    from spatialize.gs.cat_esi import classifiers as C
    rng = np.random.default_rng(cases.GENERATOR_SEED)
    normal = np.array([np.cos(np.radians(30)), np.sin(np.radians(30))])
    bands = lambda x: (np.floor((x @ normal) * 6) % 3).astype(np.int64)
    pts = rng.random((300, 2))
    qry = rng.random((3000, 2))
    params = C._set_cell_params_knn_pca(pts, bands(pts), seed=1)
    angle = np.degrees(np.arctan2(params[1], params[0])) % 180
    assert abs(angle - 30) < 8
    assert params[4] > 1 > params[5]                                   # stretched across the bands
    iso = params.copy()
    iso[4:6] = 1.0
    acc = lambda p: np.mean(C._knn_pca_predict(pts, bands(pts), qry, p) == bands(qry))
    assert acc(params) > acc(iso) + 0.02

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
    from spatialize.gs.cat_esi.agg_functions import aggregate_with_btd
    _, labels = _annotators(150, [0.8] * 12)
    a = aggregate_with_btd(labels, categories_list=["0", "1", "2"], seed=3)
    b = aggregate_with_btd(labels, categories_list=["0", "1", "2"], seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    assert a[1].shape == (150, 3) and np.allclose(a[1].sum(axis=1), 1.0)
    assert np.array_equal(a[0], np.array(["0", "1", "2"], dtype=object)[a[1].argmax(axis=1)])


def test_dawid_skene_weighs_reliable_partitions_where_majority_vote_cannot():
    """Ten reliable annotators and fifteen that always answer "0": the majority vote follows the
    biased ones, Dawid-Skene learns each annotator's confusion matrix and follows the reliable ones."""
    from spatialize.gs.cat_esi.agg_functions import aggregate_with_btd, aggregate_with_mv
    truth, labels = _annotators(400, [0.9] * 10, biased=15)
    mv = np.mean(aggregate_with_mv(labels) == truth)
    btd = np.mean(aggregate_with_btd(labels, categories_list=["0", "1", "2"], seed=1)[0] == truth)
    assert btd > 0.9 and btd > mv + 0.3


@pytest.mark.parametrize("categories,kind,order", [(["0", "1"], "binary", None), (["0", "1", "2"], "nominal", None),
                                                   (["0", "1", "2"], "ordinal", {"0": 0, "1": 1, "2": 2})])
def test_dawid_skene_kinds_of_variables(categories, kind, order):
    """Binary (a two-class nominal variable), nominal and ordinal variables all aggregate to one of
    their categories."""
    from spatialize.gs.cat_esi.agg_functions import aggregate_with_btd
    _, labels = _annotators(100, [0.85] * 10, n_classes=len(categories))
    est, prob, _ = aggregate_with_btd(labels, category_type=kind, categories_list=categories,
                                      ordinal_order_map=order, seed=2)
    assert set(est) <= set(categories) and prob.shape == (100, len(categories))


def test_dawid_skene_reports_through_the_visitor_protocol():
    """Messages, warnings and progress of the compiled code reach the callback as Spatialize's JSON
    protocol (an info message, a warning, one progress token per EM iteration, then done), never the
    console."""
    import json
    from spatialize.gs.cat_esi.agg_functions import aggregate_with_btd
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
    from spatialize.gs.cat_esi.agg_functions import aggregate_with_btd
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
    from spatialize.gs.cat_esi.agg_functions import aggregate_with_mv
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

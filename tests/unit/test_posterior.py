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


def test_auto_widening_is_decided_once_for_the_variable():
    """widening="auto" on data of both signs gives every datum the skew-normal widening, even a datum whose
    members are all positive (it got a sharply peaked Gamma law before)."""
    from spatialize.gs.spa._main import PosteriorAudit
    rng = np.random.default_rng(0)
    points = rng.random((80, 2))
    values = rng.normal(size=80)
    members = values[:, None] + rng.normal(0, 0.3, (80, 40))
    members[0] = np.abs(members[0]) + 0.1                       # one datum whose members are all positive
    laws = lambda widening: PosteriorAudit(members, points, values, widening=widening, seed=1, calibrate=False)._widened()
    auto, skew = laws("auto"), laws("skew_normal")
    assert all(np.allclose(a, b) for a, b in zip(auto, skew))


def _laws_and_draws(n=300, T=200, seed=0, tails="t"):
    """Members of n laws of different centres, spreads and skewness, and one value drawn from the
    kernel density each law is read as (no widening, no spread factor)."""
    from scipy.stats import norm, t as student
    from spatialize.empirical import silverman_bandwidth
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2))
    centres = np.sin(4 * points[:, 0])
    members = centres[:, None] + (0.1 + 0.5 * points[:, 1])[:, None] * rng.gamma(2, 1, (n, T))
    kernel, args = (student, (3.0,)) if tails == "t" else (norm, ())
    values = np.array([m[rng.integers(T)] + silverman_bandwidth(m) * kernel.rvs(*args, random_state=rng)
                       for m in members])
    return points, members, values, centres


def test_information_check_is_centred_when_the_data_follow_their_laws():
    """With every datum drawn from the density its law is read as, the excess surprise has mean 0:
    the z of the information check stays within 3 in repeated draws, for both kernels; laws too
    narrow give a large positive z, laws too wide a negative one."""
    from spatialize.gs.spa import PosteriorAudit
    from spatialize.logging import singleton_null_callback as quiet
    zs = []
    for seed in range(6):
        for tails in ("t", "normal"):
            points, members, values, _ = _laws_and_draws(seed=seed, tails=tails)
            audit = PosteriorAudit(members, points, values, widening=False, calibrate=False, tails=tails,
                                   callback=quiet, seed=seed)
            info = audit.calibration(information=True)["information"]
            assert info["n"] == len(values) and info["se"] > 0
            zs.append(info["z"])
    assert np.all(np.abs(zs) < 3) and abs(np.mean(zs)) < 1.5
    points, members, values, centres = _laws_and_draws(seed=10)
    read = lambda v: PosteriorAudit(members, points, v, widening=False, calibrate=False, callback=quiet
                                    ).calibration(information=True)["information"]
    narrow, wide = read(centres + 2 * (values - centres)), read(centres + 0.5 * (values - centres))
    assert narrow["z"] > 10 and narrow["verdict"] == "too narrow or off-centre"
    assert wide["z"] < -3 and wide["verdict"] == "too wide"
    assert "information" not in PosteriorAudit(members, points, values, callback=quiet).calibration()


def test_information_readings_follow_the_base_and_not_the_units():
    """base changes only the unit of the information readings; the excess and the information gain do
    not depend on the units of the values, while an entropy shifts by log(a) when they are multiplied
    by a; a bad base is refused; the gain is the entropy of all the data minus that of the law."""
    from spatialize import SpatializeError
    from spatialize.gs.spa import PosteriorAudit
    from spatialize.logging import singleton_null_callback as quiet
    points, members, values, _ = _laws_and_draws(n=120, T=100)
    audit = lambda m, v, base: PosteriorAudit(m, points, v, widening=False, calibrate=False, callback=quiet,
                                              base=base)
    nats, bits = audit(members, values, np.e), audit(members, values, 2)
    for column in ("surprisal", "entropy", "excess"):
        assert np.allclose(bits.table()[column], nats.table()[column] / np.log(2))
    scaled = audit(100 * members, 100 * values, 2)
    assert np.allclose(scaled.table().excess, bits.table().excess)
    assert np.allclose(scaled.evidence().gain, bits.evidence().gain)
    assert np.allclose(scaled.table().entropy - bits.table().entropy, np.log2(100))
    ev = bits.evidence()
    assert np.allclose(ev.gain, ev.attrs["prior_entropy"] - ev.entropy)
    assert np.array_equal(ev.support, bits.support)
    for base in (1, 0, -2, np.nan):
        with pytest.raises(SpatializeError):
            audit(members, values, base)


def test_entropy_of_a_kernel_density_is_read_from_the_density():
    """The entropy of a law is that of its kernel density: for a large normal sample read with Gaussian
    kernels, the density is normal of variance 1 + h^2, whose entropy is known."""
    from spatialize.gs.spa._main import _kde_entropy, _kernel
    from spatialize.empirical import silverman_bandwidth
    x = np.random.default_rng(3).normal(size=4000)
    h = silverman_bandwidth(x)
    expected = 0.5 * np.log(2 * np.pi * np.e * (np.var(x) + h ** 2))
    entropy, var = _kde_entropy(x, _kernel("normal", 3.0), np.random.default_rng(4))
    assert abs(entropy - expected) < 0.03 and abs(var - 0.5) < 0.05


def test_plot_evidence_draws_support_and_gain():
    """plot_evidence returns a figure with a panel for the support and one for the gain, each with its
    colour bar, without showing it."""
    import matplotlib
    matplotlib.use("Agg")
    from spatialize.gs.spa import PosteriorAudit
    from spatialize.logging import singleton_null_callback as quiet
    points, members, values, _ = _laws_and_draws(n=60, T=40)
    members[:5, :30] = np.nan                                    # five data with little support
    fig = PosteriorAudit(members, points, values, callback=quiet, base=2).plot_evidence()
    titles = [a.get_title() for a in fig.axes]
    assert "Support of each law" in titles and "Information from the other data" in titles
    assert len(fig.axes) == 4


def test_gpd_tails_read_one_spliced_law():
    """Under tails="gpd" every reading is of one law, spliced from a Gaussian kernel density between the
    10 % and 90 % quantiles and pooled generalized Pareto tails beyond them: its density integrates to 1,
    its distribution function is continuous at the thresholds, its closed forms agree with scipy; data
    drawn from it give an information check within 3 and uniform positions; a datum beyond the end of a
    bounded tail is counted apart and makes the laws too narrow."""
    from scipy.stats import genpareto, kstest
    from spatialize.gs.spa import PosteriorAudit
    from spatialize.gs.spa._main import _spliced_laws, _gpd_logpdf, _gpd_sf
    from spatialize.logging import singleton_null_callback as quiet
    e = np.linspace(0, 3, 50)
    for shape in (-0.3, 0.0, 0.4):
        assert np.allclose(_gpd_logpdf(e, shape, 0.7), genpareto.logpdf(e, shape, 0, 0.7))
        assert np.isclose(_gpd_sf(1.3, shape, 0.7), genpareto.sf(1.3, shape, 0, 0.7))
    rng = np.random.default_rng(5)
    zs = []
    for seed in range(5):
        points, members, _, _ = _laws_and_draws(seed=seed)
        laws = _spliced_laws(list(members))
        values = np.array([law.rvs(1, rng)[0] for law in laws])
        audit = PosteriorAudit(members, points, values, widening=False, calibrate=False, tails="gpd",
                               callback=quiet, seed=seed)
        info = audit.calibration(information=True)["information"]
        zs.append(info["z"])
        assert info["outside_support"] == 0 and kstest(audit.pit(), "uniform").pvalue > 1e-3
    assert np.all(np.abs(zs) < 3)
    law = laws[0]
    grid = np.linspace(law.m + law.sc * (law.ul - 40), law.m + law.sc * (law.uh + 40), 200001)
    assert abs(np.trapz(np.exp(law.logpdf(grid)), grid) - 1) < 1e-3
    for y in (law.ul, law.uh):
        z = law.m + law.sc * y
        assert np.allclose(law.below_above(z - 1e-9), law.below_above(z + 1e-9), atol=1e-6)
    shape, scale = law.lo
    assert shape < 0                                   # the lower tail of these skewed laws is bounded
    beyond = law.m + law.sc * (law.ul + scale / shape - 1.0)
    values[0] = beyond
    audit = PosteriorAudit(members, points, values, widening=False, calibrate=False, tails="gpd", callback=quiet)
    info = audit.calibration(information=True)["information"]
    assert info["outside_support"] == 1 and info["verdict"] == "too narrow or off-centre"
    assert np.isinf(audit.table().surprisal[0])


def test_null_flags_counts_the_flags_of_clean_replicates():
    """null_flags audits replicates drawn from the laws with the same partitions and settings: one row per
    replicate, a replicate flagged at q exactly when its smallest adjusted p-value is at most q, the
    summary in attrs computed from the rows, the same rows for the same seed; the smallest adjusted
    p-value is that of Benjamini–Hochberg; values drawn from a datum's law have uniform positions in it;
    an audit without refit refuses."""
    from scipy.stats import kstest
    from spatialize import SpatializeError
    from spatialize.gs.spa import posterior_audit, PosteriorAudit
    from spatialize.gs.spa._main import _min_qvalue, _bh
    from spatialize.logging import singleton_null_callback as quiet
    s, v, _ = DATA["2d"]
    v = v.copy()
    v[5] = v.max() + 5 * np.std(v)                               # one error, flagged
    audit = posterior_audit(s, v, n_partitions=cases.T, alpha=cases.ALPHA, seed=cases.SEED, callback=quiet)
    out = audit.null_flags(n_replicates=3, q=0.05)
    assert len(out) == 3 and audit.flags()[5]
    assert np.array_equal(out["flags"] > 0, out["min_qvalue"] <= 0.05)
    a = out.attrs
    assert a["observed"] == int(audit.flags().sum()) and np.isclose(a["expected"], out["flags"].mean())
    assert np.isclose(a["share_with_flags"], np.mean(out["flags"] > 0))
    assert np.isclose(a["p_value"], (1 + np.sum(out["flags"] >= a["observed"])) / 4)
    strongest = _min_qvalue(audit.tail_p())
    assert np.isclose(a["p_value_strongest"], (1 + np.sum(out["min_qvalue"] <= strongest)) / 4)
    assert a["p_value_strongest"] == 0.25                        # the error lies beyond every replicate
    assert out.equals(audit.null_flags(n_replicates=3, q=0.05))
    p = np.random.default_rng(0).random(50) ** 3
    for q in (0.01, 0.05, 0.2):
        assert _bh(p, q).any() == (_min_qvalue(p) <= q)
    for tails in ("t", "normal"):                     # the spliced laws of "gpd" draw in test_gpd_tails_...
        points, members, values, _ = _laws_and_draws(n=40, T=150)
        law = PosteriorAudit(members, points, values, widening=False, calibrate=False, tails=tails, callback=quiet)
        rng = np.random.default_rng(1)
        draws = np.array([law._draw_values(rng)[0] for _ in range(400)])
        positions = [PosteriorAudit(members[:1], points[:1], [d], widening=False, calibrate=False, tails=tails,
                                    callback=quiet).pit()[0] for d in draws]
        assert kstest(positions, "uniform").pvalue > 1e-3
    with pytest.raises(SpatializeError):
        PosteriorAudit(audit.members, s, v).null_flags(1)


def _two_variables(n=240, seed=1):
    """Two variables at the same locations sharing a component of the sample, which the neighbours do
    not see: u and v both follow it, w is independent of u."""
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2))
    shared = np.sin(2 * np.pi * points[:, 0]) + rng.normal(0, 0.5, n)
    return points, shared + rng.normal(0, 0.15, n), shared + rng.normal(0, 0.15, n), rng.normal(size=n)


def test_cross_reads_each_datum_given_a_second_variable():
    """The cross reading flags a value shifted against the covariate measured with it, which the audit of
    the values alone does not flag, and no datum of the clean data; its information gain is large for a
    linked covariate and near 0 for an independent one; a datum without covariate gets NaN readings and
    takes no part in the others' laws; the readings follow base; a covariate of another length, or an
    audit without partitions, is refused. Clean data get a false flag by chance on about 2 % of fields
    (1 of 64 in the study of the method), so the data are drawn from a seed without one."""
    from spatialize import SpatializeError
    from spatialize.gs.spa import posterior_audit, PosteriorAudit
    from spatialize.logging import singleton_null_callback as quiet
    points, u, v, w = _two_variables()
    kw = dict(n_partitions=150, alpha=0.85, seed=1, callback=quiet)
    clean = posterior_audit(points, u, **kw)
    linked = clean.cross(v)
    assert not linked.flag.any() and np.all((linked.pit > 0) & (linked.pit < 1))
    assert np.median(linked.gain) > 0.8 and abs(np.median(clean.cross(w).gain)) < 0.2
    shifted = u.copy()
    shifted[7] += 1.2
    audit = posterior_audit(points, shifted, **kw)
    assert audit.cross(v).flag[7] and not audit.flags()[7]
    partial = v.copy()
    partial[3] = np.nan
    t = clean.cross(partial)
    assert t.loc[3, ["pit", "tail_p", "gain"]].isna().all() and t.drop(index=3)[["pit", "tail_p"]].notna().all().all()
    bits = posterior_audit(points, u, base=2, **kw).cross(v)
    assert np.allclose(bits.entropy, linked.entropy / np.log(2)) and np.allclose(bits.gain, linked.gain / np.log(2))
    with pytest.raises(SpatializeError):
        clean.cross(v[:-1])
    with pytest.raises(SpatializeError):
        PosteriorAudit(clean.members, points, u).cross(v)


def test_cross_is_robust_to_a_gross_error_among_the_pairs():
    """A datum the audit of the values flags takes no part in the others' laws given the covariate, so a
    value shifted next to a decimal error is still flagged, the error too, and the clean data are not;
    the local line is exact on pairs on a line."""
    from spatialize.gs.spa import posterior_audit
    from spatialize.gs.spa._main import _local_line
    from spatialize.logging import singleton_null_callback as quiet
    points, u, v, _ = _two_variables()
    i = 7
    j = int(np.argsort(np.linalg.norm(points - points[i], axis=1))[1])      # its nearest neighbour
    bad = u.copy()
    bad[j] *= 10
    bad[i] += 1.2
    audit = posterior_audit(points, bad, n_partitions=150, alpha=0.85, seed=1, callback=quiet)
    assert audit.flags()[j]
    t = audit.cross(v)
    assert t.flag[i] and t.flag[j] and t.flag.sum() == 2
    x = np.linspace(-1, 1, 30)
    lever, centre, r, scale = _local_line(x - 0.3, 2 * x + 1, np.ones(30))
    assert np.isclose(centre, 1.6) and np.allclose(r, 0) and scale < 1e-12 and np.isclose(lever.sum(), 1)

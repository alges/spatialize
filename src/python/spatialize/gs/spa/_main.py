"""Posterior analysis of the data: each datum read against the predictive law the other data give.

Cross-validation predicts each datum from the others, with the partitions and the decoder alone, so
the law at a datum carries the evidence the rest of the data give about it, without a variogram or
a model chosen beforehand. :func:`posterior_audit` computes these laws and returns a
:class:`PosteriorAudit`. :func:`cv_sample_pred_posterior` and :class:`PosteriorSampleAnalyzer`, the
names of version 1.2, are kept on top of it.
"""
import random as rd
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

import spatialize.gs.esi.aggfunction as af
from spatialize import SpatializeError, logging
from spatialize._util import signature_overload, per_call, random_seed
from spatialize.empirical import (EmpiricalModel, FittedModelFactory, _loo_target_variance,
                                  _loo_target_skewness)
from spatialize.gs import (lib_spatialize_facade, partitioning_process, local_interpolator as li,
                           with_more_decoders, decoder_arguments)
from spatialize.logging import resolve_callback, log_message
from spatialize.result import Summarised
from spatialize.viz import PlotStyle


def _default_factory():
    return FittedModelFactory(nan_model_name="ignore", point_model_name="vim", n_components=3,
                              bgm_sample_size=1000, bgm_max_iter=100)


_COMMON = {"k": -1,
           "p_process": partitioning_process.MONDRIAN,
           "data_cond": True,
           "n_partitions": 200,
           "alpha": 0.8,
           "seed": random_seed,
           "folding_seed": random_seed,
           "fitted_model_factory": per_call(_default_factory),
           "callback": None,
           "best_params_found": None}

_SPECIFIC = with_more_decoders({
    li.IDW: {"exponent": 2.0},
    li.KRIGING: {"model": "spherical", "nugget": 0.5, "range": 50.0, "sill": 0.9},
    li.ADAPTIVE_IDW: {"metric": "mae"},
})


def _with_best_params(kwargs):
    """The arguments with those of ``best_params_found`` in force, except ``n_partitions``."""
    best = kwargs.get("best_params_found")
    if best is None:
        return kwargs
    log_message(logging.logger.debug(f"using best params found: {best}"))
    out = dict(kwargs)
    out.update({key: value for key, value in best.items() if key != "n_partitions"})
    return out


def _cells_reader(points, kwargs):
    """A function giving the cells of the data and of uniform probe locations in the partitions of
    the cross-validation, or in partitions of another granularity ``alpha`` drawn the same way:
    ``(n_probes, seed, alpha=None) -> (data cells (n, T), probe cells (n_probes, T))``."""
    from spatialize import session
    points = np.asarray(points, dtype=np.float32)
    negative = kwargs["p_process"] == partitioning_process.VORONOI and not kwargs["data_cond"]

    def read(n_probes, seed, alpha=None):
        alpha = kwargs["alpha"] if alpha is None else alpha
        alpha = -alpha if negative else alpha
        domain = session.get("domain")
        box = (np.asarray(domain, dtype=float) if domain is not None
               else np.stack([points.min(axis=0), points.max(axis=0)], axis=1).astype(float))
        rng = np.random.default_rng(seed)
        probes = (box[:, 0] + rng.random((n_probes, len(box))) * (box[:, 1] - box[:, 0])).astype(np.float32)
        labels = np.asarray(lib_spatialize_facade.cells(points, np.vstack([points, probes]), kwargs["p_process"],
                                                        alpha, kwargs["n_partitions"], kwargs["seed"]))
        return labels[: len(points)], labels[len(points):]
    return read


def _cv_members(points, values, queries, kwargs):
    """The cross-validation members at the data, (n, n_partitions), computed through ``run``.

    ``queries`` only take part in the box of the partitions, as in version 1.2."""
    points = np.asarray(points, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    n = points.shape[0]
    k = kwargs["k"]
    method = "loo" if k in ("loo", -1, n) else "kfold"
    alpha = kwargs["alpha"]
    if alpha >= 1.0:
        raise ValueError(f"alpha must be < 1 (got {alpha})")
    if kwargs["p_process"] == partitioning_process.VORONOI and not kwargs["data_cond"]:
        alpha = -alpha
    decoder = kwargs["local_interpolator"]
    params = {name: kwargs.get(name) for name in decoder_arguments(decoder)}
    try:
        _, members = lib_spatialize_facade.run(
            points, values, np.asarray(queries, dtype=np.float32), kwargs["p_process"], decoder, params, alpha,
            kwargs["n_partitions"], kwargs["seed"], method=method, k=0 if method == "loo" else k,
            folding_seed=kwargs["folding_seed"], callback=kwargs["callback"])
    except Exception as e:
        raise SpatializeError(e) from e
    return np.asarray(members)


def _bh(p, q):
    """Benjamini–Hochberg at false discovery rate q over the finite p-values."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), dtype=bool)
    m = int(ok.sum())
    if m == 0:
        return out
    idx = np.flatnonzero(ok)[np.argsort(p[ok], kind="stable")]
    passed = np.flatnonzero(p[idx] <= q * np.arange(1, m + 1) / m)
    if passed.size:
        out[idx[: passed[-1] + 1]] = True
    return out


def _weighted_silverman(x, w):
    """Silverman's bandwidth of ``x`` under the weights ``w``, with the effective number of values."""
    w = w / w.sum()
    m = float(np.sum(w * x))
    sd = float(np.sqrt(np.sum(w * (x - m) ** 2)))
    n_eff = 1.0 / float(np.sum(w ** 2))
    return 0.9 * max(sd, 1e-12) * n_eff ** -0.2, n_eff


def _local_line(x, y, w):
    """A line of ``y`` on ``x`` by weighted least squares: the leverage of each pair in its value at
    x = 0, that value, the residuals and their standard deviation under the weights ``w``."""
    w = w / w.sum()
    X = np.column_stack([np.ones(x.size), x])
    A = np.linalg.pinv(X.T @ (w[:, None] * X))
    lever = (A @ (X * w[:, None]).T)[0]
    r = y - X @ (A @ (X.T @ (w * y)))
    scale = float(np.sqrt(np.sum(w * r ** 2) / max(1.0 - np.sum(w ** 2), 1e-12)))
    return lever, float(lever @ y), r, scale


def _kernel_logpdf(z, nu):
    """The log density of the standard Student-t law of ``nu`` degrees of freedom, Gaussian for an
    infinite ``nu``, in closed form."""
    from scipy.special import gammaln
    if not np.isfinite(nu):
        return -0.5 * z * z - 0.5 * np.log(2 * np.pi)
    const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
    return const - 0.5 * (nu + 1) * np.log1p(z * z / nu)


def _mixture_entropy(centres, weights, h, nu, size, rng):
    """The entropy, in nats, of the mixture of Student-t kernels (``nu`` degrees of freedom, Gaussian
    for an infinite ``nu``) of scale ``h`` at ``centres``, by ``size`` draws."""
    from scipy.special import logsumexp
    pick = rng.choice(centres.size, size=size, p=weights)
    eps = rng.standard_normal(size) if not np.isfinite(nu) else rng.standard_t(nu, size)
    draws = centres[pick] + h * eps
    logs = logsumexp(_kernel_logpdf((draws[:, None] - centres[None, :]) / h, nu) + np.log(weights)[None, :],
                     axis=1) - np.log(h)
    return -float(np.mean(logs))


def _qvalues(p):
    """The Benjamini–Hochberg adjusted p-values (q-values): datum i is flagged at rate q exactly when its
    q-value is at most q. NaN where p is NaN."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.size, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size:
        order = ok[np.argsort(p[ok], kind="stable")]
        q = p[order] * ok.size / np.arange(1, ok.size + 1)
        out[order] = np.minimum(np.minimum.accumulate(q[::-1])[::-1], 1.0)
    return out


def _min_qvalue(p):
    """The smallest Benjamini–Hochberg adjusted p-value over the finite p-values: some datum is flagged
    at rate q exactly when it is at most q."""
    p = np.sort(np.asarray(p, dtype=float)[np.isfinite(p)])
    if not p.size:
        return np.nan
    return float(np.min(p * p.size / np.arange(1, p.size + 1)))


def _scale_maps(scale, values):
    """The map to the scale of the readings and its inverse, fitted to the data values."""
    identity = lambda a: np.asarray(a, dtype=np.float64)
    if scale == "raw":
        return identity, identity
    if scale == "yeojohnson":
        from scipy.stats import yeojohnson
        from scipy.special import inv_boxcox
        _, lam = yeojohnson(np.asarray(values, dtype=np.float64))

        def fwd(a):
            a = np.asarray(a, dtype=np.float64)
            out = np.empty_like(a)
            pos = a >= 0
            out[pos] = np.log1p(a[pos]) if abs(lam) < 1e-12 else ((a[pos] + 1) ** lam - 1) / lam
            neg = ~pos
            out[neg] = (-np.log1p(-a[neg]) if abs(lam - 2) < 1e-12
                        else -((1 - a[neg]) ** (2 - lam) - 1) / (2 - lam))
            return out

        def inv(y):
            y = np.asarray(y, dtype=np.float64)
            out = np.full_like(y, np.nan)
            pos = y >= 0
            if abs(lam) < 1e-12:
                out[pos] = np.expm1(y[pos])
            else:
                # with lambda < 0 the scale is bounded above by -1/lambda, which maps to +infinity
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[pos] = np.where(lam * y[pos] + 1 > 0, inv_boxcox(y[pos], lam) - 1, np.inf)
            neg = y < 0
            if abs(lam - 2) < 1e-12:
                out[neg] = -np.expm1(-y[neg])
            else:
                # with lambda > 2 the scale is bounded below by 1/(2 - lambda), which maps to -infinity
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[neg] = np.where((2 - lam) * (-y[neg]) + 1 > 0, 1 - inv_boxcox(-y[neg], 2 - lam), -np.inf)
            return out
        return fwd, inv
    # normal scores, linear between the data and continued linearly past the extremes
    from scipy.stats import norm
    srt = np.sort(np.asarray(values, dtype=np.float64))
    sc = norm.ppf((np.arange(1, len(srt) + 1) - 0.5) / len(srt))

    def extend(x, xs, ys):
        y = np.interp(x, xs, ys)
        lo_slope = (ys[1] - ys[0]) / max(xs[1] - xs[0], 1e-12)
        hi_slope = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1e-12)
        y = np.where(x < xs[0], ys[0] + lo_slope * (x - xs[0]), y)
        return np.where(x > xs[-1], ys[-1] + hi_slope * (x - xs[-1]), y)
    return (lambda a: extend(np.asarray(a, dtype=np.float64), srt, sc),
            lambda y: extend(np.asarray(y, dtype=np.float64), sc, srt))


def _fit_spread(laws, values, target=0.9):
    """The factor c such that, with every law spread by c around its median, a share ``target`` of
    the data lies inside the central ``target`` interval of its law."""
    centred = [(np.median(x), x - np.median(x)) for x in laws]
    keep = [i for i, x in enumerate(laws) if x.size]
    if not keep:
        return 1.0
    a, b = (1 - target) / 2, (1 + target) / 2

    def coverage(c):
        pit = np.array([np.mean(centred[i][0] + c * centred[i][1] < values[i]) for i in keep])
        return np.mean((pit >= a) & (pit <= b))
    lo, hi = 0.05, 50.0
    for _ in range(40):
        c = np.sqrt(lo * hi)
        lo, hi = (c, hi) if coverage(c) < target else (lo, c)
    return float(np.sqrt(lo * hi))


def _kernel(tails, nu):
    """The kernel of the kernel densities: Student-t of ``nu`` degrees of freedom, or Gaussian (``"normal"``,
    and ``"gpd"``, whose spliced laws have a Gaussian kernel density between their thresholds)."""
    from scipy.stats import norm, t as student
    return (norm, ()) if tails in ("normal", "gpd") else (student, (nu,))


def _kde_entropy(x, kernel, rng):
    """The entropy, in nats, of the kernel density of the sample ``x`` (Silverman bandwidth), and the
    variance of its log density under itself, both by ``x.size`` draws from that density."""
    from scipy.special import logsumexp
    from spatialize.empirical import silverman_bandwidth
    h = silverman_bandwidth(x)
    draws = x + h * kernel[0].rvs(*kernel[1], size=x.size, random_state=rng)
    logs = np.concatenate([logsumexp(kernel[0].logpdf((d[:, None] - x[None, :]) / h, *kernel[1]), axis=1)
                           for d in np.array_split(draws, max(1, x.size // 2000))]) - np.log(x.size * h)
    return -float(np.mean(logs)), float(np.var(logs))


def _tail_readings(laws, values, tails, nu, q=0.9, seed=0):
    """Per datum, under the tail model: log density at the datum, the entropy of the law's density
    and the variance of its log density under itself (both by draws from that density, seeded by
    ``seed`` plus the datum's index), and the probabilities of the law below and above the datum."""
    from scipy.special import logsumexp
    from scipy.stats import genpareto
    from spatialize.empirical import silverman_bandwidth
    if tails == "gpd":
        return _gpd_readings(laws, values, q, seed)
    n = len(values)
    logf, ent, var, below, above = (np.full(n, np.nan) for _ in range(5))
    kernel = _kernel(tails, nu)
    for i, (x, z) in enumerate(zip(laws, values)):
        if x.size < 2:
            continue
        h = silverman_bandwidth(x)
        # the draws that read the entropy are drawn from the density itself, not its sample: then the
        # expected log score of a datum drawn from its law is the entropy, exactly
        draws = x + h * kernel[0].rvs(*kernel[1], size=x.size, random_state=np.random.default_rng(seed + i))
        u = (np.append(draws, z)[:, None] - x[None, :]) / h
        logs = logsumexp(kernel[0].logpdf(u, *kernel[1]), axis=1) - np.log(x.size * h)
        logf[i], ent[i], var[i] = logs[-1], -float(np.mean(logs[:-1])), float(np.var(logs[:-1]))
        uz = (z - x) / h
        below[i] = float(np.exp(logsumexp(kernel[0].logcdf(uz, *kernel[1])) - np.log(x.size)))
        above[i] = float(np.exp(logsumexp(kernel[0].logsf(uz, *kernel[1])) - np.log(x.size)))
    return logf, ent, var, below, above


def _gpd_logpdf(e, shape, scale):
    """The log density of a generalized Pareto law at the excesses ``e`` ≥ 0 (-inf beyond a bounded
    support)."""
    if abs(shape) < 1e-12:
        return -np.log(scale) - e / scale
    t = 1.0 + shape * e / scale
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t > 0, -np.log(scale) - (1.0 + 1.0 / shape) * np.log(np.where(t > 0, t, 1.0)), -np.inf)


def _gpd_sf(e, shape, scale):
    """The survival function of a generalized Pareto law at the excess ``e`` ≥ 0."""
    if abs(shape) < 1e-12:
        return float(np.exp(-e / scale))
    t = 1.0 + shape * e / scale
    return float(t ** (-1.0 / shape)) if t > 0 else 0.0


def _gpd_rvs(size, shape, scale, rng):
    """``size`` draws of a generalized Pareto law, by inversion."""
    u = rng.random(size)
    if abs(shape) < 1e-12:
        return -scale * np.log1p(-u)
    return scale / shape * ((1.0 - u) ** (-shape) - 1.0)


class _SplicedLaw:
    """A law spliced from the Gaussian kernel density of a sample between its quantiles 1 - q and q,
    holding the probability 2q - 1, and a generalized Pareto law beyond each of them, holding 1 - q,
    all on the sample standardised by its median ``m`` and scale ``sc`` (``tails="gpd"``). ``hi`` and
    ``lo`` are the (shape, location, scale) of the Pareto law of each side, location 0."""

    _LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)

    def __init__(self, x, m, sc, q, hi, lo):
        from scipy.special import ndtr
        from spatialize.empirical import silverman_bandwidth
        self.m, self.sc = m, sc
        self.hi, self.lo = (hi[0], hi[2]), (lo[0], lo[2])
        self.w = (x - m) / sc
        self.uh, self.ul = np.quantile(self.w, q), np.quantile(self.w, 1 - q)
        self.tail, self.body = 1.0 - q, 2.0 * q - 1.0
        self.h = silverman_bandwidth(self.w)
        self.k_lo, self.k_hi = (float(np.mean(ndtr((t - self.w) / self.h))) for t in (self.ul, self.uh))
        self.mass = max(self.k_hi - self.k_lo, np.finfo(float).tiny)   # the kernel density's mass in the body

    def logpdf(self, z):
        from scipy.special import logsumexp
        y = (np.atleast_1d(np.asarray(z, dtype=float)) - self.m) / self.sc
        out = np.empty(y.size)
        up, down = y > self.uh, y < self.ul
        mid = ~(up | down)
        out[up] = np.log(self.tail) + _gpd_logpdf(y[up] - self.uh, *self.hi)
        out[down] = np.log(self.tail) + _gpd_logpdf(self.ul - y[down], *self.lo)
        if mid.any():
            u = (y[mid, None] - self.w[None, :]) / self.h
            out[mid] = (np.log(self.body / self.mass) - np.log(self.w.size * self.h) - self._LOG_SQRT_2PI
                        + logsumexp(-0.5 * u * u, axis=1))
        return out - np.log(self.sc)

    def below_above(self, z):
        """The probabilities of the law below and above ``z``, each computed on its own side so that a
        small one keeps its precision."""
        from scipy.special import ndtr
        y = (float(z) - self.m) / self.sc
        if y > self.uh:
            above = self.tail * _gpd_sf(y - self.uh, *self.hi)
            return 1.0 - above, above
        if y < self.ul:
            below = self.tail * _gpd_sf(self.ul - y, *self.lo)
            return below, 1.0 - below
        below = self.tail + self.body * (float(np.mean(ndtr((y - self.w) / self.h))) - self.k_lo) / self.mass
        return below, 1.0 - below

    def rvs(self, size, rng):
        """``size`` draws: each side's share from its Pareto law, the body from the kernel density kept
        between the thresholds."""
        n_up, n_down, n_mid = rng.multinomial(size, [self.tail, self.tail, self.body])
        mid = np.empty(0)
        while mid.size < n_mid:
            k = 2 * (n_mid - mid.size) + 8
            c = self.w[rng.integers(self.w.size, size=k)] + self.h * rng.standard_normal(k)
            mid = np.concatenate([mid, c[(c >= self.ul) & (c <= self.uh)]])
        y = np.concatenate([self.uh + _gpd_rvs(n_up, *self.hi, rng), self.ul - _gpd_rvs(n_down, *self.lo, rng),
                            mid[:n_mid]])
        return self.m + self.sc * rng.permutation(y)


def _spliced_laws(laws, q=0.9):
    """The spliced law of every sample of ``laws`` (None for fewer than two values), with the shape and
    scale of each Pareto side pooled over all the samples standardised by their median and interquartile
    range, or None when too few values lie beyond the thresholds to fit them."""
    from scipy.stats import genpareto
    std, ex_hi, ex_lo = [], [], []
    for x in laws:
        if x.size < 2:
            std.append(None)
            continue
        m = np.median(x)
        sc = float(np.subtract(*np.percentile(x, [75, 25]))) or float(np.std(x)) or 1.0
        w = (x - m) / sc
        uh, ul = np.quantile(w, q), np.quantile(w, 1 - q)
        std.append((m, sc))
        ex_hi += list(w[w > uh] - uh)
        ex_lo += list(ul - w[w < ul])
    if len(ex_hi) <= 10 or len(ex_lo) <= 10:
        return None
    hi, lo = genpareto.fit(ex_hi, floc=0), genpareto.fit(ex_lo, floc=0)
    return [None if st is None else _SplicedLaw(x, st[0], st[1], q, hi, lo) for x, st in zip(laws, std)]


def _gpd_readings(laws, values, q=0.9, seed=0):
    """The readings of ``_tail_readings`` under ``tails="gpd"``, on the spliced law of each datum
    (:class:`_SplicedLaw`): every reading, the probabilities, the log score and the entropy, of one law."""
    spliced = _spliced_laws(laws, q)
    if spliced is None:
        log_message(logging.logger.warning(
            "too few values beyond the thresholds to fit the generalized Pareto tails; the laws are read "
            "with Gaussian kernels instead"))
        return _tail_readings(laws, values, "normal", None, q, seed)
    n = len(values)
    logf, ent, var, below, above = (np.full(n, np.nan) for _ in range(5))
    for i, (law, z) in enumerate(zip(spliced, values)):
        if law is None:
            continue
        logs = law.logpdf(law.rvs(law.w.size, np.random.default_rng(seed + i)))
        logf[i], ent[i], var[i] = float(law.logpdf(z)[0]), -float(np.mean(logs)), float(np.var(logs))
        below[i], above[i] = law.below_above(z)
    return logf, ent, var, below, above


class PosteriorAudit(Summarised):
    """The predictive law of each datum built from the other data, and its readings.

    Parameters
    ----------
    members : ndarray of shape (n, T)
        The cross-validation members at each datum, one per partition: each datum predicted from the
        other data. NaN where the datum's cell held no other datum (``empty_cells="nan"``).
    points : ndarray of shape (n, d)
        The data locations.
    values : ndarray of shape (n,)
        The data values.
    fitted_model_factory : FittedModelFactory, optional
        The density model of :meth:`model`, for plots (:mod:`spatialize.empirical`). Default: a
        variational Gaussian mixture of three components.
    callback : callable, optional
        Where the progress and the messages go. Default: ``None``, shown as the session settings
        ``display``, ``progress`` and ``verbosity`` say (:mod:`spatialize.session`). A callable
        receiving the messages of :mod:`spatialize.logging` sends them elsewhere, such as an
        application's own interface, while :func:`~spatialize.logging.singleton_null_callback`
        drops them.
    widening : {"auto", "gamma", "skew_normal", False}, optional
        How each datum's members are widened before they are read (:meth:`law`). Each member is the
        prediction of a partition-average, so the members spread less than the data do around it,
        and laws read without widening put too many data in their tails (:meth:`calibration`). The
        members are widened to the variance of the datum's ``widening_knn`` nearest other data, the
        ensemble widening of :class:`~spatialize.empirical.FittedModelFactory`. ``"auto"``
        chooses ``"gamma"`` for non-negative values and ``"skew_normal"`` otherwise. Default:
        ``"auto"``.
    widening_knn : int, optional
        The number of nearest other data the widening reads. Default: 12.
    seed : int, optional
        Seed of the widening draws. Default: 0.
    scale : {"raw", "yeojohnson", "normal_scores"}, optional
        The scale on which the laws are read. ``"raw"`` reads the values as they are;
        ``"yeojohnson"`` through a Yeo–Johnson transform fitted to the data, monotone and smooth, which
        makes a skewed variable more symmetric and keeps how far an extreme value lies;
        ``"normal_scores"`` through the normal scores of the data, which make the data normal but
        bring every extreme value to the largest score, so a gross error stands out less. The
        readings in probability (positions, p-values, levels, flags) do not depend on a monotone
        scale by themselves, only through the kernels and the widening, which act on that scale.
        Default: ``"raw"``.
    calibrate : bool, optional
        Whether every law's spread around its median is multiplied by one factor, fitted so that 90 %
        of the data lie inside the central 90 % interval of their law (:attr:`spread_factor`).
        Default: True.
    tails : {"t", "gpd", "normal"}, optional
        The model of the law beyond its sample, for every reading (positions, p-values, log scores,
        entropies). ``"t"``, a kernel density with Student-t kernels of ``nu`` degrees of freedom;
        ``"gpd"``, a law spliced from a Gaussian kernel density between the 10 % and 90 % quantiles
        of the sample, holding 80 % of the probability, and generalized Pareto tails beyond them,
        holding 10 % each, with one shape and scale per side pooled over all the laws standardised
        by their median and interquartile range; ``"normal"``, a kernel density with Gaussian
        kernels. Default: ``"t"``.
    nu : float, optional
        The degrees of freedom of the Student-t kernels. Default: 3.
    cells : callable, optional
        ``cells(n_probes, seed, alpha=None) -> (data_cells, probe_cells)``, the labels of the cells
        of the data, shape (n, T), and of ``n_probes`` locations drawn uniformly in the box of the
        partitions, shape (n_probes, T), in the partitions of the members, or in partitions of
        granularity ``alpha`` drawn the same way; needed by :meth:`weights` and :meth:`cross`.
        :func:`posterior_audit` provides it. Default: None.
    base : float, optional
        The base of the logarithms of the information readings (``surprisal``, ``entropy`` and
        ``excess`` of :meth:`table`, :meth:`evidence`, the information check of :meth:`calibration`):
        e for nats, 2 for bits, 10 for hartleys. Default: e.
    refit : callable, optional
        ``refit(values) -> members``, the cross-validation members of other values at the same
        locations, with the same partitions and decoder; needed by :meth:`null_flags`.
        :func:`posterior_audit` provides it. Default: None.

    Attributes
    ----------
    members, points, values
        As given.
    support : ndarray of shape (n,)
        The share of defined members of each datum: the share of partitions in which its law rests
        on other data. A low support marks an isolated datum, whose law rests on few partitions.
    spread_factor : float
        The factor of ``calibrate`` (1 without it): above 1, the widened laws were still too narrow.

    Notes
    -----
    The law of a datum never contains the datum itself, so a value its neighbours do not support
    keeps all of its surprise. The target of the widening is a robust variance (the squared scaled
    median absolute deviation of the neighbours), so an erroneous neighbour does not widen a law
    enough to hide another error.

    The laws of an ensemble read by cross-validation are narrow, since each member averages a cell,
    and light in their tails. Read as they are, they put many more data in their tails than their
    probabilities say, so a test at a given level flags clean data. The widening, the factor and the
    tail model correct this in turn, measured on synthetic fields (:doc:`/theory/posterior`), without
    a guarantee: :meth:`calibration` tells how well the laws are calibrated for the data at hand.
    """

    def __init__(self, members, points, values, fitted_model_factory=None, callback=None,
                 widening="auto", widening_knn=12, seed=None, scale="raw", calibrate=True, tails="t", nu=3.0,
                 cells=None, base=np.e, refit=None):
        self.members = np.asarray(members, dtype=np.float64)
        self.points = np.asarray(points)
        self.values = np.asarray(values, dtype=np.float64).ravel()
        if self.members.shape[0] != len(self.values):
            raise SpatializeError(f"members has {self.members.shape[0]} rows for {len(self.values)} data")
        self.fitted_model_factory = fitted_model_factory if fitted_model_factory is not None else _default_factory()
        self.callback = resolve_callback(callback)
        self.support = np.isfinite(self.members).mean(axis=1)
        self._models = {}
        self._targets = None
        self.widening = widening
        self.widening_knn = int(widening_knn)
        self.seed = 0 if seed is None else int(seed)
        if scale not in ("raw", "yeojohnson", "normal_scores"):
            raise SpatializeError(f"scale must be 'raw', 'yeojohnson' or 'normal_scores'; got {scale!r}")
        if tails not in ("t", "gpd", "normal"):
            raise SpatializeError(f"tails must be 't', 'gpd' or 'normal'; got {tails!r}")
        self.scale, self.calibrate, self.tails, self.nu = scale, bool(calibrate), tails, float(nu)
        if not (np.isfinite(base) and base > 0 and base != 1):
            raise SpatializeError(f"base must be a positive number other than 1; got {base!r}")
        self.base = float(base)
        self._fwd, self._inv = _scale_maps(scale, self.values)
        self._tvalues = self._fwd(self.values)
        self._laws = None
        self._factor = None
        self._readings = None
        self._cells = cells         # the cells of the data and of probe locations in the partitions
        self._refit = refit         # the members of other values, with the same partitions and decoder
        self._weights = None

    def _widening_targets(self):
        f = self._resolved_factory()
        if self._targets is None:
            # leave-one-out targets: each datum's k nearest OTHER data, so its own value does not
            # deflate its target
            var = _loo_target_variance(self.points, self.values, knn=f.widening_knn) if f.widening else None
            skew = (_loo_target_skewness(self.points, self.values, knn=f.widening_knn)
                    if f.widening == "skew_normal" else None)
            self._targets = (var, skew)
        return self._targets

    def _resolved_factory(self):
        """The fitted-model factory with ``widening="auto"`` decided once for the variable, by the sign
        of its data, as :func:`~spatialize.gs.ess.ess_sample` does."""
        f = self.fitted_model_factory
        if f is not None and f.widening == "auto":
            import copy
            f = copy.copy(f)
            f.widening = "gamma" if np.nanmin(self.values) >= 0 else "skew_normal"
        return f

    def model(self, i):
        """The density model of datum ``i``'s law, fitted to its defined members, or None.

        Parameters
        ----------
        i : int
            Index of the datum.

        Returns
        -------
        EmpiricalModel or None
            None when fewer than two members are defined or the fit fails.
        """
        if i not in self._models:
            sample = self.members[i][np.isfinite(self.members[i])]
            f = self._resolved_factory()
            var, skew = self._widening_targets()
            model = None
            if sample.size >= 2:
                try:
                    model = EmpiricalModel(sample=sample, fitted_model_factory=f,
                                           target_var=None if var is None else var[i],
                                           target_skew=None if skew is None else skew[i],
                                           seed=None if f.seed is None else f.seed + i)
                except Exception as e:
                    log_message(logging.logger.debug(f"no law for datum {i} (value {self.values[i]}): {e}"))
            self._models[i] = model
        return self._models[i]

    def models(self):
        """The density models of every datum (see :meth:`model`), warning once about the data
        without one.

        Returns
        -------
        list of EmpiricalModel or None
        """
        out = [self.model(i) for i in range(len(self.values))]
        missing = sum(m is None for m in out)
        if missing:
            log_message(logging.logger.warning(
                f"{missing} of {len(out)} data have no fitted law (fewer than two defined members or a "
                f"failed fit); their readings are NaN"))
        return out

    # ------------------------------------------------------------------ readings of each datum
    def _summary(self, q=0.05):
        from spatialize import _display
        n, T = self.members.shape
        cal = self.calibration()
        cov = cal["coverage"].set_index("alpha")
        p = self.tail_p()
        flags = _bh(p, q)
        audit = [("data", n), ("members per datum", T), ("median support", float(np.median(self.support))),
                 ("data with support < 0.5", int(np.sum(self.support < 0.5))),
                 ("tails", self.tails + (f" (ν = {self.nu:g})" if self.tails == "t" else "")),
                 ("scale", self.scale)]
        calib = [("spread factor", f"{cal['spread_factor']:.3f}"),
                 ("inside 90 % interval", f"{100 * cov.loc[0.9, 'observed']:.1f} %" if 0.9 in cov.index else None),
                 ("inside 99 % interval", f"{100 * cov.loc[0.99, 'observed']:.1f} %" if 0.99 in cov.index else None),
                 ("tails", cal["verdict"]), ("centre", cal["centre"]),
                 (f"flagged (FDR {q:g})", int(flags.sum()))]
        order = np.argsort(np.where(np.isfinite(p), p, np.inf))[:8]
        pit, shift = self.pit(), self.shift()
        rows = [[f"#{int(i)}", float(self.values[i]), float(pit[i]), float(p[i]), "yes" if flags[i] else "",
                 float(shift[i])] for i in order]
        return _display.Summary(
            "Posterior analysis of the data",
            blocks=[("kv", "Laws", audit), ("kv", "Calibration and flags", calib),
                    ("table", "Most surprising data", ["datum", "value", "position", "p-value", "flag", "shift"], rows)],
            footer="table() · calibration() · flags() · weights() · declustered() · plot_map()")

    def _robust_targets(self):
        """Per datum, the robust variance and the skewness of its nearest other data, on the scale
        of the readings."""
        from scipy.spatial import cKDTree
        pts, z, k = np.asarray(self.points, float), self._tvalues, self.widening_knn
        n = len(z)
        _, idx = cKDTree(pts).query(pts, k=min(k + 1, n))
        idx = np.reshape(idx, (n, -1))
        var = np.empty(n)
        for i in range(n):
            nb = z[idx[i][idx[i] != i][:k]]
            var[i] = (1.4826 * np.median(np.abs(nb - np.median(nb)))) ** 2 if nb.size else 0.0
        skew = _loo_target_skewness(pts, z, knn=k)
        return var, skew

    def _widened(self):
        """Each datum's defined members on the scale of the readings, widened."""
        if self._laws is None:
            from spatialize.empirical import _widen_sample
            var, skew = self._robust_targets() if self.widening else (None, None)
            # "auto" is decided once for the variable, by the sign of its data on the scale of the
            # readings, not datum by datum (a datum whose members are all positive, in a variable that
            # also takes negative values, would get a sharply peaked Gamma law)
            family = self.widening
            if family == "auto":
                family = "gamma" if np.nanmin(self._tvalues) >= 0 else "skew_normal"
            laws = []
            for j in range(len(self.values)):
                m = self.members[j]
                x = self._fwd(m[np.isfinite(m)])
                if self.widening and x.size >= 2 and var[j] > 0:
                    mode = family
                    if mode == "gamma" and np.any(x < 0):
                        mode = "skew_normal"
                    x = _widen_sample(x, mode, var[j], skew[j], np.random.default_rng(self.seed + j))
                laws.append(np.asarray(x, dtype=np.float64))
            self._laws = laws
        return self._laws

    @property
    def spread_factor(self):
        """The factor multiplying every law's spread around its median (see ``calibrate``)."""
        if self._factor is None:
            self._factor = _fit_spread(self._widened(), self._tvalues) if self.calibrate else 1.0
        return self._factor

    def law(self, i):
        """The sample datum ``i``'s readings come from, on the scale of the readings: its defined
        members, transformed by ``scale``, widened by ``widening`` and spread by
        :attr:`spread_factor`.

        Parameters
        ----------
        i : int
            Index of the datum.

        Returns
        -------
        ndarray
            Empty for a datum without defined members.
        """
        x = self._widened()[i]
        c = self.spread_factor
        if c == 1.0 or x.size == 0:
            return x
        m = np.median(x)
        return m + c * (x - m)

    def pit(self):
        r"""The position of each datum in its law, :math:`\hat F_{-i}(z_i)`, under the tail model.

        Returns
        -------
        ndarray of shape (n,)
            The probability of the law (:meth:`law`, with the kernels or tails of ``tails``) below
            the datum, the same law the p-values read. NaN for a datum without defined members.
        """
        _, _, _, below, above = self._kde()
        with np.errstate(invalid="ignore", divide="ignore"):
            return below / (below + above)

    def _kde(self):
        """Per datum, in nats: the log density of its law at the datum, the entropy of the law and the
        variance of its log density under itself; and the probabilities of the law below and above
        the datum, under ``tails``."""
        if self._readings is None:
            laws = [self.law(i) for i in range(len(self.values))]
            self._readings = _tail_readings(laws, self._tvalues, self.tails, self.nu, seed=self.seed)
        return self._readings

    def tail_p(self):
        r"""The two-sided surprise of each datum: how rarely its law gives a value as extreme.

        Returns
        -------
        ndarray of shape (n,)
            :math:`\min\{1, 2\min(P_-, P_+)\}`, with :math:`P_-` and :math:`P_+` the probabilities
            of the law below and above the datum under the tail model ``tails``. The model carries
            the law past its sample, so a datum far outside its law gets a p-value as small as its
            distance warrants, where counting the members could not go below one over their
            number. NaN for a datum without defined members.
        """
        _, _, _, below, above = self._kde()
        # a probability below the smallest float (a datum hundreds of spreads away) is kept positive
        return np.clip(2.0 * np.minimum(below, above), np.finfo(float).tiny, 1.0)

    def flags(self, q=0.05):
        """The data whose surprise survives the Benjamini–Hochberg procedure at false discovery
        rate ``q``.

        With n data, a share of them falls in the tails of their laws by chance alone. Among the
        flagged data, the expected share of such chance flags is at most ``q`` (for independent or
        positively dependent p-values).

        Parameters
        ----------
        q : float, optional
            The false discovery rate. Default: 0.05.

        Returns
        -------
        ndarray of bool, shape (n,)
        """
        return _bh(self.tail_p(), q)

    def levels(self, alphas=(0.5, 0.7, 0.9, 0.99)):
        r"""The level of each datum: the widest central probability interval of its law that leaves
        it out.

        Parameters
        ----------
        alphas : sequence of float, optional
            The probabilities :math:`\alpha` of the central intervals
            :math:`[\hat q_{(1-\alpha)/2}, \hat q_{(1+\alpha)/2}]`. Default: ``(0.5, 0.7, 0.9, 0.99)``.

        Returns
        -------
        ndarray of object, shape (n,)
            ``"level_j"``: ``level_0`` outside the widest interval, ``level_k`` (k the number of
            alphas) inside every one. None for a datum without defined members. A datum lies outside
            the interval of probability :math:`\alpha` with probability :math:`1 - \alpha` when its
            law is right, so the share of data at each level is read against these probabilities.
        """
        pit = self.pit()
        alphas = sorted(alphas)
        out = np.empty(len(pit), dtype=object)
        for i, u in enumerate(pit):
            if not np.isfinite(u):
                out[i] = None
                continue
            cat = len(alphas)
            for j, a in enumerate(reversed(alphas)):
                if u < (1 - a) / 2 or u > (1 + a) / 2:
                    cat = j
                    break
            out[i] = f"level_{cat}"
        return out

    def table(self, q=0.05, alphas=(0.5, 0.7, 0.9, 0.99)):
        r"""The readings of every datum, one row each.

        Parameters
        ----------
        q : float, optional
            The false discovery rate of the flags. Default: 0.05.
        alphas : sequence of float, optional
            The probabilities of the levels. Default: ``(0.5, 0.7, 0.9, 0.99)``.

        Returns
        -------
        pandas.DataFrame
            ``value``; ``pit`` (:meth:`pit`); ``tail_p`` (:meth:`tail_p`); ``flag`` (:meth:`flags`);
            ``level`` (:meth:`levels`); ``surprisal``, the log score :math:`-\log \hat f_{-i}(z_i)`
            of the datum under its law (the density of :meth:`law` under the tail model ``tails``);
            ``entropy``, the entropy of that density, the expected log score of a value drawn from
            it, so that ``excess`` = ``surprisal`` − ``entropy`` has mean 0 for a datum drawn from
            its law (:meth:`calibration`, ``information``); ``width90``, the width of the central 90 % interval of
            the law, in the units of the values; ``support`` (:attr:`support`); ``coherence``
            (:meth:`coherence`); ``shift`` (:meth:`shift`); ``weight`` (:meth:`weights`, NaN when the audit does not hold its
            partitions). The log scores and the entropy are on the scale of the readings
            (``scale``), in the units of ``base``. The entropy depends on the units of the values,
            shifting by :math:`\log a` when they are multiplied by :math:`a`; the excess does not.
        """
        logf, ent, _, below, above = self._kde()
        nats = np.log(self.base)
        tail = np.clip(2.0 * np.minimum(below, above), np.finfo(float).tiny, 1.0)
        q5 = np.array([np.percentile(self.law(i), 5) if self.law(i).size else np.nan for i in range(len(self.values))])
        q95 = np.array([np.percentile(self.law(i), 95) if self.law(i).size else np.nan for i in range(len(self.values))])
        lo, hi = self._inv(q5), self._inv(q95)
        return pd.DataFrame({"value": self.values, "pit": self.pit(), "tail_p": tail,
                             "flag": _bh(tail, q), "level": self.levels(alphas), "surprisal": -logf / nats,
                             "entropy": ent / nats, "excess": (-logf - ent) / nats, "width90": hi - lo,
                             "support": self.support, "coherence": self.coherence(), "shift": self.shift(),
                             "weight": self.weights() if self._cells is not None else np.nan})

    def coherence(self, k=8):
        """For each datum, the share of its ``k`` nearest other data that lie on the same side of
        their laws (above or below the median).

        A datum alone in its surprise, among neighbours on either side, has a coherence near 0.5,
        which suits an isolated error. A datum whose neighbours are surprised the same way has a
        coherence near 1, which suits a part of the domain the laws do not represent, a
        sub-population or a change of domain. The provenance decides (:doc:`/theory/posterior`).

        Parameters
        ----------
        k : int, optional
            The number of nearest other data. Default: 8.

        Returns
        -------
        ndarray of shape (n,)
            NaN for a datum without a law.
        """
        from scipy.spatial import cKDTree
        pts = np.asarray(self.points, float)
        n = len(pts)
        k = min(int(k), n - 1)
        if k < 1:
            return np.full(n, np.nan)
        side = np.sign(self.pit() - 0.5)
        _, idx = cKDTree(pts).query(pts, k=min(k + 1, n))
        idx = np.reshape(idx, (n, -1))
        out = np.full(n, np.nan)
        for i in range(n):
            nb = idx[i][idx[i] != i][:k]
            nb = nb[np.isfinite(side[nb])]
            if np.isfinite(side[i]) and nb.size:
                out[i] = float(np.mean(side[nb] == side[i]))
        return out

    def shift(self, k=8):
        r"""For each datum, the mean signed position of its ``k`` nearest other data in their laws,
        :math:`\frac1k \sum_{j} (2u_j - 1)`, between -1 and 1.

        Read together with the datum's own position, it tells an isolated error from a part of the
        domain the laws do not represent. An erroneous value pulls the laws of its neighbours towards
        it, so they fall on the other side: a datum far above its law with a negative shift. Data of
        an unrepresented part of the domain are surprised together: a datum above its law with a
        positive shift. Clean data have a shift near 0.

        Parameters
        ----------
        k : int, optional
            The number of nearest other data. Default: 8.

        Returns
        -------
        ndarray of shape (n,)
        """
        from scipy.spatial import cKDTree
        pts = np.asarray(self.points, float)
        n = len(pts)
        k = min(int(k), n - 1)
        if k < 1:
            return np.full(n, np.nan)
        pos = 2.0 * self.pit() - 1.0
        _, idx = cKDTree(pts).query(pts, k=min(k + 1, n))
        idx = np.reshape(idx, (n, -1))
        out = np.full(n, np.nan)
        for i in range(n):
            nb = idx[i][idx[i] != i][:k]
            nb = nb[np.isfinite(pos[nb])]
            if nb.size:
                out[i] = float(np.mean(pos[nb]))
        return out

    def _fingerprint(self):
        """A short digest of the locations and values, which ties a reading to the data it read."""
        import hashlib
        h = hashlib.sha1(np.ascontiguousarray(np.asarray(self.points, dtype=np.float64)).tobytes())
        h.update(np.ascontiguousarray(self.values).tobytes())
        return h.hexdigest()[:16]

    def _settings(self):
        """The arguments that read another audit the way this one reads its data."""
        return dict(fitted_model_factory=self.fitted_model_factory, widening=self.widening,
                    widening_knn=self.widening_knn, seed=self.seed, scale=self.scale, calibrate=self.calibrate,
                    tails=self.tails, nu=self.nu, base=self.base)

    def _draw_values(self, rng):
        """One value per datum, drawn from the law its readings use (:meth:`law` under ``tails``), in
        the units of the values; a datum without a law keeps its value."""
        laws = [self.law(i) for i in range(len(self.values))]
        out = self._tvalues.copy()
        if self.tails == "gpd":
            spliced = _spliced_laws(laws)
            if spliced is not None:
                for i, law in enumerate(spliced):
                    if law is not None:
                        out[i] = law.rvs(1, rng)[0]
                return self._inv(out)
        from spatialize.empirical import silverman_bandwidth
        kernel = _kernel("normal" if self.tails == "gpd" else self.tails, self.nu)
        for i, x in enumerate(laws):
            if x.size >= 2:
                out[i] = x[rng.integers(x.size)] + silverman_bandwidth(x) * float(
                    kernel[0].rvs(*kernel[1], random_state=rng))
        return self._inv(out)

    def null_flags(self, n_replicates=20, q=0.05, seed=None, clean=True):
        r"""How many flags clean data would get, at these locations and with these settings.

        No theorem bounds the false discovery rate of the flags (:meth:`flags`), since the laws are
        only approximately calibrated and neighbouring laws share data. This reading measures it on
        the data at hand. It draws replicates of the data free of errors, each datum from its own law,
        audits every replicate with the same partitions, decoder and settings, and counts its flags,
        every one of them a false discovery.

        Parameters
        ----------
        n_replicates : int, optional
            The number of replicates, each one a full audit. Default: 20.
        q : float, optional
            The false discovery rate of the flags counted. Default: 0.05.
        seed : int, optional
            Seed of the draws. Default: the audit's ``seed``.
        clean : bool, optional
            Whether the replicates are drawn from the laws of the data with the flagged data replaced
            by the median of their law, the laws being computed again. An erroneous datum pulls the
            laws of its neighbours towards it, so replicates drawn from them vary more than clean
            data would, which inflates their flags. Default: True.

        Returns
        -------
        pandas.DataFrame
            One row per replicate: ``flags``, the number of flags at ``q``, and ``min_qvalue``, the
            smallest Benjamini–Hochberg adjusted p-value, so that a replicate has some flag at a rate
            ``q'`` exactly when ``min_qvalue`` ≤ ``q'``. ``attrs`` holds ``q``; ``observed``, the
            flags of the data at ``q``; ``expected``, the mean number of flags of the replicates, the
            false flags the data would get without errors; ``share_with_flags``, the share of
            replicates with some flag, the chance that clean data get at least one; and ``p_value``,
            the Monte Carlo p-value :math:`(1 + \#\{r : F_r \ge F\})/(R + 1)` of the hypothesis
            that the data are as clean as the replicates, :math:`F` being the observed flags and
            :math:`F_r` those of replicate :math:`r`; and ``p_value_strongest``, the same hypothesis
            read through the most surprising datum, :math:`(1 + \#\{r : Q_r \le Q\})/(R + 1)` with
            :math:`Q` the smallest adjusted p-value of the data and :math:`Q_r` that of replicate
            :math:`r`. The count weighs every flag alike, so one datum a hundred spreads away counts
            as one; the strongest datum weighs how far it lies. ``attrs`` also holds ``data``, a digest
            of the locations and values read, by which :meth:`recommend` checks that the replicates
            belong to its audit.

        Raises
        ------
        SpatializeError
            When the audit was not built by :func:`posterior_audit`, which can compute the members of
            other values.

        Notes
        -----
        Each replicate draws every datum on its own from its law, the law the readings use, so the
        replicates keep the centres of the laws and the spread around them, the spatial pattern
        coming from the centres. On synthetic fields of 300 data, the replicates predicted the share
        of clean fields with a false flag (0.20 against an observed 0.19 with the default Student-t
        tails, 0.92 against 0.94 with Gaussian kernels), and with three planted errors in the data
        they still gave the rate of the clean data (:doc:`/theory/posterior`). The partitions are
        those of the audit, so the reading is conditional on them. With few replicates,
        ``share_with_flags`` is coarse, its standard error being about
        :math:`\sqrt{p(1 - p)/R}`, and ``p_value`` cannot fall below :math:`1/(R + 1)`, 0.048 with
        20 replicates and 0.01 with 99. Each replicate costs one audit, about a second for 300 data
        and 300 partitions with IDW.
        """
        if self._refit is None:
            raise SpatializeError("null_flags needs to compute the members of other values: build the audit "
                                  "with posterior_audit, or give PosteriorAudit its refit")
        from spatialize.logging import singleton_null_callback
        rng = np.random.default_rng(self.seed if seed is None else seed)
        settings = self._settings()
        observed = self.flags(q)
        source = self
        if clean and observed.any():
            z = self.values.copy()
            for i in np.flatnonzero(observed):
                z[i] = float(self._inv(np.median(self.law(i))))
            source = PosteriorAudit(self._refit(z), self.points, z, callback=singleton_null_callback, **settings)
        rows = []
        self.callback(logging.progress.init(int(n_replicates), 1, desc="replicates"))
        for _ in range(int(n_replicates)):
            z = source._draw_values(rng)
            replicate = PosteriorAudit(self._refit(z), self.points, z, callback=singleton_null_callback, **settings)
            p = replicate.tail_p()
            rows.append({"flags": int(_bh(p, q).sum()), "min_qvalue": _min_qvalue(p)})
            self.callback(logging.progress.inform())
        self.callback(logging.progress.stop())
        out = pd.DataFrame(rows)
        n_obs = int(observed.sum())
        strongest = _min_qvalue(self.tail_p())
        out.attrs.update({"q": q, "data": self._fingerprint(), "observed": n_obs, "expected": float(out["flags"].mean()),
                          "share_with_flags": float((out["flags"] > 0).mean()),
                          "p_value": float((1 + np.sum(out["flags"] >= n_obs)) / (len(out) + 1)),
                          "p_value_strongest": float((1 + np.sum(out["min_qvalue"] <= strongest)) / (len(out) + 1))})
        return out

    def weights(self, n_probes=20000):
        """The declustering weight of each datum, read from the partitions.

        Each partition shares the domain among the data, every cell giving its area equally to the
        data it holds and the area of the cells without data being shared out in proportion. The
        weight of a datum is its share averaged over the partitions, so the weights sum to 1. A
        datum in a dense cluster shares small cells with many others and weighs little, an isolated
        datum in a large cell weighs much. The areas are estimated with ``n_probes`` locations drawn
        uniformly in the box of the partitions.

        Parameters
        ----------
        n_probes : int, optional
            The number of locations that estimate the areas of the cells. Default: 20000.

        Returns
        -------
        ndarray of shape (n,)

        Raises
        ------
        SpatializeError
            When the audit was not built by :func:`posterior_audit`, which holds its partitions.
        """
        if self._cells is None:
            raise SpatializeError("the declustering weights need the partitions: build the audit with "
                                  "posterior_audit, or give PosteriorAudit its cells")
        if self._weights is None:
            data_cells, probe_cells = self._cells(int(n_probes), self.seed)
            n, T = data_cells.shape
            w = np.zeros(n)
            for t in range(T):
                labels, occupancy = np.unique(data_cells[:, t], return_counts=True)
                probe_labels, probe_counts = np.unique(probe_cells[:, t], return_counts=True)
                hits = dict(zip(probe_labels, probe_counts))
                area = np.array([hits.get(c, 0) for c in labels], dtype=float)
                if area.sum() <= 0:
                    continue
                share = dict(zip(labels, area / area.sum() / occupancy))
                w += np.array([share[c] for c in data_cells[:, t]])
            self._weights = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
        return self._weights

    def declustered(self, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)):
        """The summaries of the values with and without the declustering weights (:meth:`weights`).

        Preferential sampling, where more data were taken where the values are high or of interest,
        biases the plain summaries towards those values. The weighted ones correct for it, the
        partitions telling how much of the domain each datum represents.

        Parameters
        ----------
        quantiles : sequence of float, optional
            The quantiles reported. Default: ``(0.05, 0.25, 0.5, 0.75, 0.95)``.

        Returns
        -------
        pandas.DataFrame
            One row per statistic (``mean``, ``std`` and the quantiles), columns ``naive`` and
            ``declustered``.
        """
        z, w = self.values, self.weights()
        order = np.argsort(z)
        cw = np.cumsum(w[order]) - 0.5 * w[order]
        mean = float(np.sum(w * z))
        rows = {"mean": (float(np.mean(z)), mean),
                "std": (float(np.std(z)), float(np.sqrt(np.sum(w * (z - mean) ** 2))))}
        for q in quantiles:
            rows[f"q{q:g}"] = (float(np.quantile(z, q)), float(np.interp(q, cw, z[order])))
        return pd.DataFrame(rows, index=["naive", "declustered"]).T

    def proportional_effect(self):
        """How the spread of the laws follows their centre.

        Returns
        -------
        dict
            ``centre`` and ``spread``, the median and the width of the central 90 % interval of each
            datum's law in the units of the values, and ``spearman``, their rank correlation. A
            strong positive correlation is the proportional effect of skewed variables, where the
            laws widen with the values; reading the laws on a transformed scale (``scale``) may then
            suit them better.
        """
        from scipy.stats import spearmanr
        n = len(self.values)
        centre = np.array([float(self._inv(np.median(self.law(i)))) if self.law(i).size else np.nan for i in range(n)])
        q5 = np.array([np.percentile(self.law(i), 5) if self.law(i).size else np.nan for i in range(n)])
        q95 = np.array([np.percentile(self.law(i), 95) if self.law(i).size else np.nan for i in range(n)])
        spread = self._inv(q95) - self._inv(q5)
        ok = np.isfinite(centre) & np.isfinite(spread)
        rho = float(spearmanr(centre[ok], spread[ok]).statistic) if ok.sum() > 2 else np.nan
        return {"centre": centre, "spread": spread, "spearman": rho}

    def evidence(self):
        r"""How much the other data tell about each datum: the support of its law and the information
        the law carries beyond the distribution of all the data.

        The law of a datum may rest on little evidence in two ways. An isolated datum shares its cell
        with no other datum in many partitions, so its law rests on few of them (a low ``support``).
        A datum whose neighbours disagree gets a law spread nearly as widely as the data themselves
        (a low ``gain``), even when every partition defines it.

        Returns
        -------
        pandas.DataFrame
            One row per datum: ``support`` (:attr:`support`); ``entropy``, the entropy of its law
            (as in :meth:`table`); ``gain``, the entropy of the distribution of all the data minus
            that of the law, the information the other data give about the datum, in the units of
            ``base``. The entropy of the distribution of all the data, read as a kernel density of
            the values with the kernels of ``tails`` (Gaussian under ``"gpd"``), is in ``attrs["prior_entropy"]``.

        Notes
        -----
        An entropy depends on the units of the values, shifting by :math:`\log a` when they are
        multiplied by :math:`a`. The gain, a difference of two entropies, does not, so it can be
        compared across variables. A gain of one bit (``base=2``) halves the spread a law of normal
        shape would have; a gain near 0, or below it, says that the law of the datum is as wide as
        the distribution of all the data. The laws are read as :meth:`law` gives them, widened and
        spread by the factor of ``calibrate``, which lowers every gain.
        """
        _, ent, _, _, _ = self._kde()
        nats = np.log(self.base)
        z = self._tvalues[np.isfinite(self._tvalues)]
        prior = (_kde_entropy(z, _kernel(self.tails, self.nu), np.random.default_rng(self.seed + len(self.values)))[0]
                 if z.size >= 2 else np.nan)
        out = pd.DataFrame({"support": self.support, "entropy": ent / nats, "gain": (prior - ent) / nats})
        out.attrs["prior_entropy"] = prior / nats
        return out

    def cross(self, covariate, alpha=0.9, nu=10.0, knn=20, q=0.05, n_draws=400, passes=3):
        r"""Each datum read against its law given a second variable measured at the same locations.

        Two variables measured on one sample share what the sample holds, which its neighbours do not
        see. A value plausible among its neighbours may then be implausible given the other variable
        at the same place: a shifted value, two values swapped between samples, a unit misread in one
        variable. The law of datum :math:`i` given the covariate :math:`v_i` is built from the pairs
        :math:`(u_j, v_j)` of the other data, each weighted by the share of partitions of granularity
        ``alpha`` in which its location shares the cell of :math:`x_i` and by a Gaussian kernel in
        :math:`v_j - v_i`, whose width holds at least ``knn`` of them. A weighted line of :math:`u`
        on :math:`v` gives the centre of the law at :math:`v_i`, and its residuals, spread by
        Student-t kernels of ``nu`` degrees of freedom with the variance of the fitted centre added,
        give its shape (:doc:`/theory/posterior`).

        Parameters
        ----------
        covariate : array_like of shape (n,)
            The second variable at the locations of the data, NaN where it was not measured.
        alpha : float, optional
            The granularity of the partitions that weigh the pairs. Default: 0.9.
        nu : float, optional
            The degrees of freedom of the Student-t kernels; ``numpy.inf`` for Gaussian kernels.
            Default: 10.
        knn : int, optional
            The least number of pairs within the width of the kernel in the covariate. Default: 20.
        q : float, optional
            The false discovery rate of the flags. Default: 0.05.
        n_draws : int, optional
            The draws that read each entropy. Default: 400.
        passes : int, optional
            The most readings of the data in turn, each leaving the data flagged by the previous one
            out of the others' laws (see Notes). Default: 3.

        Returns
        -------
        pandas.DataFrame
            One row per datum: ``covariate``; ``pit``, ``tail_p`` and ``flag``, the position, the
            two-sided tail probability and the Benjamini–Hochberg flag at ``q`` under the law given
            the covariate; ``surprisal`` and ``entropy`` of that law, in the units of ``base``;
            ``gain``, the entropy of the law of :math:`u` among the weighted pairs minus that of the
            law given the covariate, the information the covariate gives about the datum, near 0 where
            it gives none, so that the cross reading has no power there; ``pairs``, the effective
            number of pairs the law rests on. NaN for a datum without a covariate. ``attrs`` holds
            ``q`` and ``data``, a digest of the locations and values read, by which :meth:`recommend`
            checks that the reading belongs to its audit.

        Raises
        ------
        SpatializeError
            When the audit was not built by :func:`posterior_audit`, which holds its partitions, or
            the covariate has another length than the data.

        Notes
        -----
        An error among the pairs would pull the line of its neighbours, widening their residuals,
        which hides their own errors. The data flagged by the audit of the values (:meth:`flags` at ``q``)
        therefore take no part in the others' laws, nor do, in a further reading, those flagged given
        the covariate, the readings repeated until the flags no longer change, ``passes`` at most.
        A datum's own law never holds the datum. A robust line fails here, since
        it treats the smaller of two branches of a relation, as with a link through a square, as errors.

        The values are read on the scale of the audit (``scale``), the covariate as given. The
        covariate must be measured at the locations of the data: what the two variables share at
        the scale of the sample, which the reading uses, is lost when one of them is predicted from
        its own neighbours. The gain compares the law given the covariate with the law of the
        values among the weighted pairs, which also holds the variation of the field within the
        neighbourhood, so it exceeds the mutual information of the two variables at a location.

        On synthetic pairs of variables sharing a component of the sample, 300 data per field,
        the reading flagged 1 of 64 clean fields. With a linear link, and four planted errors, it
        found a shifted value in 13 of 16 fields and each of two swapped values in 8 and 10 of 16,
        which the audit of the values alone never flagged, while 5 of its 50 flags fell on clean
        data, a share the false discovery rate allows once true errors are flagged. A link through the square of the shared
        component, which gives its size and not its sign, gave little power. A link whose sign
        changes across the domain was followed only in part, the partitions of 300 data mixing the
        two relations (:doc:`/theory/posterior`).
        """
        if self._cells is None:
            raise SpatializeError("the cross reading needs the partitions: build the audit with posterior_audit, "
                                  "or give PosteriorAudit its cells")
        from scipy.special import logsumexp
        from scipy.stats import norm, t as student
        v = np.asarray(covariate, dtype=np.float64).ravel()
        u = self._tvalues
        n = len(u)
        if v.size != n:
            raise SpatializeError(f"covariate has {v.size} values for {n} data")
        kernel = (norm, ()) if not np.isfinite(nu) else (student, (float(nu),))
        labels = self._cells(0, self.seed, alpha)[0]
        usable = np.isfinite(u) & np.isfinite(v)
        v_raw, v = v, np.where(usable, v, 0.0)          # missing values take no part, with weight 0
        u = np.where(usable, u, 0.0)

        def read(i, partner, full, cols, rng=None):
            # the law of datum i given its covariate, from the pairs of the data in ``partner``
            w = np.mean(labels == labels[i], axis=1)
            w[i] = 0.0
            w[~partner] = 0.0
            if np.count_nonzero(w) < 3:
                return
            hv, _ = _weighted_silverman(v, w)
            dv = np.sort(np.abs(v[w > 0] - v[i]))
            hv = max(hv, dv[min(int(knn), dv.size) - 1], 1e-12)     # at least knn pairs within the width
            c = w * np.exp(-0.5 * ((v - v[i]) / hv) ** 2)
            keep = c > 1e-8 * c.max()                     # pairs of negligible weight left out
            cc, uu, vv = c[keep] / c[keep].sum(), u[keep], v[keep] - v[i]
            # a weighted line of u on v: its value at v_i is the centre, its residuals the shape, and the
            # variance of the fitted centre widens the law where the pairs do not reach v_i
            lever, centre, r, scale = _local_line(vv, uu, cc)
            hu, n_eff = _weighted_silverman(r, cc)
            hu = float(np.sqrt(hu ** 2 + scale ** 2 * np.sum(lever ** 2)))
            z = (u[i] - centre - r) / hu
            below = float(np.sum(cc * kernel[0].cdf(z, *kernel[1])))
            above = float(np.sum(cc * kernel[0].sf(z, *kernel[1])))
            cols["pit"][i] = below / (below + above)
            cols["tail_p"][i] = min(1.0, 2.0 * min(below, above))
            cols["pairs"][i] = n_eff
            if full:
                cols["surprisal"][i] = -(float(logsumexp(_kernel_logpdf(z, nu) + np.log(cc))) - np.log(hu))
                cols["entropy"][i] = _mixture_entropy(centre + r, cc, hu, nu, int(n_draws), rng)
                # the law of u among the pairs the partitions weigh, whatever their covariate
                near = w > 1e-8 * w.max()
                um, wm = u[near], w[near] / w[near].sum()
                hm, _ = _weighted_silverman(um, wm)
                cols["gain"][i] = _mixture_entropy(um, wm, hm, nu, int(n_draws), rng) - cols["entropy"][i]

        def blank():
            return {name: np.full(n, np.nan) for name in ("pit", "tail_p", "surprisal", "entropy", "gain", "pairs")}

        # the flagged data take no part in the others' laws, since an error would pull the line of its
        # neighbours and widen their residuals, hiding their own errors: first the data the audit of the
        # values flags, then those flagged given the covariate, until the flags no longer change
        univariate = self.flags(q)
        partner = usable & ~univariate
        self.callback(logging.progress.init(int(passes) * n, 1, desc="cross readings"))
        for _ in range(int(passes) - 1):
            cols = blank()
            for i in np.flatnonzero(usable):
                read(i, partner, False, cols)
                self.callback(logging.progress.inform())
            tail = np.where(np.isfinite(cols["tail_p"]), np.clip(cols["tail_p"], np.finfo(float).tiny, 1.0), np.nan)
            updated = usable & ~univariate & ~_bh(tail, q)
            if np.array_equal(updated, partner):
                break
            partner = updated
        cols = blank()
        for i in np.flatnonzero(usable):
            read(i, partner, True, cols, np.random.default_rng(self.seed + i))
            self.callback(logging.progress.inform())
        self.callback(logging.progress.stop())
        nats = np.log(self.base)
        tail = np.where(np.isfinite(cols["tail_p"]), np.clip(cols["tail_p"], np.finfo(float).tiny, 1.0), np.nan)
        out = pd.DataFrame({"covariate": v_raw, "pit": cols["pit"], "tail_p": tail, "flag": _bh(tail, q),
                            "surprisal": cols["surprisal"] / nats, "entropy": cols["entropy"] / nats,
                            "gain": cols["gain"] / nats, "pairs": cols["pairs"]})
        out.attrs.update({"q": q, "data": self._fingerprint()})
        return out

    def recommend(self, q=0.05, cross=None, null=None, effect=True, passes=5, k=8, min_support=0.5,
                  min_gain=0.0, tol=0.0):
        r"""The data that call for an action, each with the action, the reason and its strength.

        The readings of the audit become one list. It recommends measuring a suspect datum again, never
        removing it: a measurement taken again confirms a true value, a part of the domain the other data
        do not represent included, while it corrects an error. The analyst, who knows the provenance of
        the data, decides.

        Parameters
        ----------
        q : float, optional
            The false discovery rate of the flags. Default: 0.05.
        cross : pandas.DataFrame, optional
            The result of :meth:`cross` on these data at the same ``q``, whose flags enter the list. It is
            not computed here. Default: None.
        null : pandas.DataFrame, optional
            The result of :meth:`null_flags` on these data at the same ``q``, which tells whether the flags
            exceed what clean data get. It is not computed here. Default: None.
        effect : bool, optional
            Whether to measure how much the cross-validated predictions of the data change when the data
            to measure again are replaced by the medians of their laws, at the cost of one more audit.
            Default: True.
        passes : int, optional
            The most cleaning passes (see Notes). Default: 5.
        k : int, optional
            The nearest data that link suspects into a group. Default: 8.
        min_support, min_gain : float, optional
            A law resting on fewer partitions than ``min_support``, or with an information gain below
            ``min_gain`` (:meth:`evidence`, in the units of ``base``), marks a place to sample more.
            Defaults: 0.5 and 0.
        tol : float, optional
            The distance below which two data count as co-located (:meth:`duplicates`). Default: 0.

        Returns
        -------
        pandas.DataFrame
            One row per datum that calls for an action, sorted by action, then by the q-value, or by the
            gain for the places to sample more, least first: ``datum``,
            ``value``, ``action``, ``reason``, ``q_value`` (the Benjamini–Hochberg adjusted p-value
            behind the action, for a place to sample more that of the datum once the suspects are
            cleaned, usually near 1, since such a datum is not a suspect), ``gain`` (the information the
            other data give about the datum, :meth:`evidence`, low where sampling more helps) and
            ``group`` (the suspects linked as neighbours surprised the same way share a number). Each
            datum takes the first action that applies, in this order, with the readings that point to
            it gathered in ``reason``:

            - ``"measure again"``. *Rule*: a suspect (flagged at ``q`` once the cleaning settles, see
              Notes) with no other suspect among its ``k`` nearest data on the same side of its law;
              or a datum flagged at ``q`` by ``cross``. *Evidence*: its value lies far in the tail of
              the law the other data give it, or far from what the second variable at the same place
              allows. *Not*: a proof of error, since a true value of a part of the domain the other
              data do not represent can stand alone in its tail. *Action*: measure the sample again
              or check its records, replacing the value only if the new measurement disagrees.
            - ``"check provenance"``. *Rule*: a suspect linked to other suspects, each among the
              ``k`` nearest of another and on the same side of its law. *Evidence*: neighbouring data
              surprising in the same direction. *Not*: necessarily errors, since a real part of the
              domain shows the same pattern. *Action*: check the campaign, the laboratory batch or the
              instrument, and sample around the group before deciding.
            - ``"measure one of the pair again"``. *Rule*: a datum within ``tol`` of another whose
              pair passes Benjamini–Hochberg at ``q`` on :math:`\min\{1, 2\min(p_i, p_j)\}` over the
              co-located pairs. *Evidence*: two values at one location differ beyond what their laws
              allow. *Not*: a verdict on which value is wrong, since each law sees the other datum as
              a neighbour. *Action*: measure the location again, or find the original record.
            - ``"sample more nearby"``. *Rule*: no other action applies, and the law rests on a share
              of partitions below ``min_support`` or has an information gain below ``min_gain``.
              *Evidence*: the other data say little about the datum. *Not*: a suspect value, its
              q-value lying near 1. *Action*: take more samples there, the lowest gain first.
            - ``"none"``. *Rule*: flagged by the audit at ``q``, not a suspect once the cleaning
              settles, and no other action applies. *Evidence*: a suspect beside it pulled its law.
              *Not*: cleared for good, only on the condition that the suspect beside it is an error.
              *Action*: none, until that suspect is measured again.

            ``attrs`` holds ``q``, ``passes`` (the cleaning passes run), ``settled`` (whether the
            suspects stopped changing), ``counts`` (data per action), ``context`` (with ``null``: the
            observed and expected flags, the two p-values of :meth:`null_flags` and ``verdict``,
            ``"beyond clean data"`` when either p-value is at most 0.05, ``"an order of review"``
            otherwise; None without ``null``) and, with ``effect``, ``effect`` (the mean and largest
            absolute change of the cross-validated mean prediction of the data, the latter's datum, and
            the mean change over the standard deviation of the values) and ``prediction_change`` (the
            change at every datum).

        Raises
        ------
        SpatializeError
            When the audit was not built by :func:`posterior_audit`, which can compute the members of
            other values, or ``cross`` or ``null`` were read on other data or at another ``q``.

        See Also
        --------
        plot_recommendations : the list drawn on the map of the data.

        Notes
        -----
        A gross error pulls the laws of its neighbours towards it, so neighbours far on the other side of
        their laws get flagged with it, while a second error beside it may hide. The suspects are therefore
        found by cleaning. The flagged data are replaced by the medians of their laws, the laws are computed
        again, and the original values are read against them. A datum's law never holds the datum, so this
        reads every datum against neighbours free of the suspects. The flags of that reading become the
        suspects, and the cleaning repeats until they no longer change, ``passes`` times at most. A datum
        flagged at first but not after cleaning was surprising only beside a suspect. The laws keep the
        spread factor of the audit (``calibrate``) through the cleaning, so they change through the data
        alone.

        A part of the domain the other data do not represent cannot be told from errors by the flags. Its
        data support each other, so the ensemble represents it and flags at most its edge, one or two data,
        as it flags an error. Measuring again is the action that suits both. Neighbouring suspects surprised
        the same way are grouped, since a group points to a batch of samples or to such a part.

        On 40 synthetic fields with two planted errors and a raised patch, the cleaning kept 58 of the 60
        errors and found 3 more that their neighbours hid, while it cleared half of the clean data flagged
        beside an error (:doc:`/theory/posterior`).
        """
        if self._refit is None:
            raise SpatializeError("recommend needs to compute the members of other values: build the audit with "
                                  "posterior_audit, or give PosteriorAudit its refit")
        from scipy.spatial import cKDTree
        from spatialize.logging import singleton_null_callback
        n = len(self.values)
        data = self._fingerprint()
        for name, reading in (("cross", cross), ("null", null)):
            if reading is None:
                continue
            if reading.attrs.get("data") != data:
                raise SpatializeError(f"{name} was read on other data than this audit's")
            if not np.isclose(float(reading.attrs.get("q", np.nan)), q):
                raise SpatializeError(f"{name} was read at q = {reading.attrs.get('q')}, not at q = {q}")
        settings = self._settings()
        quiet = singleton_null_callback

        # the suspects, found by cleaning until they no longer change
        initial = self.flags(q)
        suspects, base, reread = initial.copy(), self, self
        settled, used = not initial.any(), 0
        while not settled and used < int(passes):
            used += 1
            z = self.values.copy()
            z[suspects] = [float(base._inv(np.median(base.law(i)))) for i in np.flatnonzero(suspects)]
            cleaned = PosteriorAudit(self._refit(z), self.points, z, callback=quiet, **settings)
            reread = PosteriorAudit(cleaned.members, self.points, self.values, callback=quiet, **settings)
            # the cleaning changes the laws through the data only: the spread factor stays the audit's,
            # which a factor fitted again would narrow once a suspect no longer lies outside its interval
            cleaned._factor = reread._factor = self.spread_factor
            found = reread.flags(q)
            settled = bool(np.array_equal(found, suspects))
            suspects, base = found, cleaned
        if not settled:
            log_message(logging.logger.warning(
                f"the suspects still changed after {used} cleaning passes; the last reading is reported"))
        q_final, q_initial = _qvalues(reread.tail_p()), _qvalues(self.tail_p())

        # groups of neighbouring suspects surprised the same way
        pts = np.asarray(self.points, float)
        side = np.sign(reread.pit() - 0.5)
        _, idx = cKDTree(pts).query(pts, k=min(int(k) + 1, n))
        idx = np.reshape(idx, (n, -1))
        group = np.full(n, -1)
        label = 0
        for i in np.flatnonzero(suspects):
            if group[i] >= 0:
                continue
            stack, group[i] = [i], label
            while stack:
                a = stack.pop()
                for b in idx[a][1:]:
                    if suspects[b] and group[b] < 0 and side[b] == side[a]:
                        group[b] = label
                        stack.append(b)
            label += 1
        sizes = np.bincount(group[group >= 0]) if label else np.zeros(0, int)

        rows = {}

        def add(i, action, reason, strength, grp=None):
            if i in rows:
                rows[i]["reason"] += "; " + reason
                rows[i]["q_value"] = np.nanmin([rows[i]["q_value"], strength])
                return
            rows[i] = {"datum": int(i), "value": float(self.values[i]), "action": action, "reason": reason,
                       "q_value": float(strength), "group": grp}

        for i in np.flatnonzero(suspects):
            g = int(group[i])
            if sizes[g] == 1:
                add(i, "measure again", "surprising once its neighbours are cleaned", q_final[i])
            else:
                add(i, "check provenance", f"one of {sizes[g]} neighbours surprised the same way", q_final[i], g)
        if cross is not None:
            q_cross = _qvalues(cross["tail_p"].to_numpy())
            for i in np.flatnonzero(cross["flag"].to_numpy()):
                add(i, "measure again", "inconsistent with the second variable measured with it", q_cross[i])
        pairs = self.duplicates(tol)
        if len(pairs):
            p_self = self.tail_p()
            p_pair = np.minimum(1.0, 2.0 * np.minimum(p_self[pairs["i"]], p_self[pairs["j"]]))
            q_pair = _qvalues(p_pair)
            for (i, j, d), qp in zip(pairs[["i", "j", "difference"]].itertuples(index=False), q_pair):
                if qp <= q:
                    for a, b in ((int(i), int(j)), (int(j), int(i))):
                        add(a, "measure one of the pair again", f"co-located with datum {b}, the values {d:.4g} apart",
                            qp)
        for i in np.flatnonzero(initial & ~suspects):
            add(i, "none", "flagged only beside a suspect, unsurprising once it is cleaned", q_initial[i])
        ev = self.evidence()
        weak = (ev["support"].to_numpy() < min_support) | (ev["gain"].to_numpy() < min_gain)
        for i in np.flatnonzero(weak):
            if i not in rows:
                add(i, "sample more nearby",
                    f"law resting on little evidence (support {ev['support'][i]:.2f}, gain {ev['gain'][i]:.2f})",
                    q_final[i])

        order = {"measure again": 0, "check provenance": 1, "measure one of the pair again": 2,
                 "sample more nearby": 3, "none": 4}
        out = pd.DataFrame(list(rows.values()), columns=["datum", "value", "action", "reason", "q_value", "group"])
        out.insert(5, "gain", ev["gain"].to_numpy()[out["datum"].to_numpy(dtype=int)] if len(out) else [])
        # by action, then by strength: the q-value, or for the places to sample more the gain, least first
        out["_rank"] = np.where(out["action"] == "sample more nearby", out["gain"], out["q_value"])
        out = out.sort_values(["action", "_rank"], key=lambda c: c.map(order) if c.name == "action" else c,
                              na_position="last", ignore_index=True).drop(columns="_rank")
        out.attrs.update({"q": q, "passes": used, "settled": settled,
                          "counts": out["action"].value_counts().to_dict(), "context": None})
        if null is not None:
            a = null.attrs
            out.attrs["context"] = {
                "observed": a["observed"], "expected": a["expected"], "p_value": a["p_value"],
                "p_value_strongest": a["p_value_strongest"],
                "verdict": "beyond clean data" if min(a["p_value"], a["p_value_strongest"]) <= 0.05
                else "an order of review"}
        measure = out.loc[out["action"].isin(["measure again", "check provenance", "measure one of the pair again"]),
                          "datum"].to_numpy(dtype=int)
        if effect and measure.size:
            z = self.values.copy()
            z[measure] = [float(reread._inv(np.median(reread.law(i)))) for i in measure]
            with np.errstate(all="ignore"):
                change = np.nanmean(self._refit(z), axis=1) - np.nanmean(self.members, axis=1)
            others = np.setdiff1d(np.arange(n), measure)
            mag = np.abs(change[others])
            out.attrs["effect"] = {"mean_abs_change": float(np.nanmean(mag)), "max_abs_change": float(np.nanmax(mag)),
                                   "datum_most_changed": int(others[np.nanargmax(mag)]),
                                   "relative": float(np.nanmean(mag) / np.std(self.values))}
            out.attrs["prediction_change"] = change
        return out

    def duplicates(self, tol=0.0):
        """The pairs of data closer than ``tol`` (co-located when ``tol`` is 0), with their values.

        Co-located data with different values, a duplicated hole, a re-assay or a coordinate error,
        cannot both be right at one location; their laws say little about it, since each sees the
        other as a neighbour.

        Parameters
        ----------
        tol : float, optional
            The distance below which two data count as co-located. Default: 0.

        Returns
        -------
        pandas.DataFrame
            ``i``, ``j``, ``distance``, ``value_i``, ``value_j`` and ``difference`` (absolute), sorted
            by decreasing difference.
        """
        from scipy.spatial import cKDTree
        pts = np.asarray(self.points, float)
        pairs = np.array(sorted(cKDTree(pts).query_pairs(r=max(float(tol), 0.0) + 1e-12)), dtype=int).reshape(-1, 2)
        i, j = pairs[:, 0], pairs[:, 1]
        d = np.sqrt(((pts[i] - pts[j]) ** 2).sum(axis=1))
        out = pd.DataFrame({"i": i, "j": j, "distance": d, "value_i": self.values[i], "value_j": self.values[j],
                            "difference": np.abs(self.values[i] - self.values[j])})
        return out.sort_values("difference", ascending=False, ignore_index=True)

    # ------------------------------------------------------------------ global readings
    def calibration(self, alphas=(0.5, 0.8, 0.9, 0.95, 0.99), information=False):
        r"""Whether the laws are calibrated: read it before the flags.

        A datum lies inside the central interval of probability :math:`\alpha` of its law with
        probability :math:`\alpha` when the law is right, and its position (:meth:`pit`) is
        uniform. Laws too narrow put too many data in the tails, which inflates every flag; laws
        too wide hide surprises.

        Parameters
        ----------
        alphas : sequence of float, optional
            The probabilities of the intervals compared. Default: ``(0.5, 0.8, 0.9, 0.95, 0.99)``.
        information : bool, optional
            Whether to add the information check, which reads the whole density of every law
            through the log score instead of a few of its intervals (see Notes). Default: False.

        Returns
        -------
        dict
            ``coverage``, a DataFrame with, per ``alpha``, the share of data inside the interval
            (``observed``), its binomial standard error under calibration (``se``) and ``z`` =
            (observed − alpha)/se, and with ``calibrate`` the share before the factor
            (``before_factor``); ``spread_factor``; ``ks_p``, the p-value of the Kolmogorov–Smirnov test of the
            positions against the uniform law; ``n``, the data with a law; ``verdict``, the
            calibration of the tails, the intervals of probability 0.9 or more, which the flags read:
            ``"calibrated"``, ``"too narrow"`` or ``"too wide"`` (an interval more than 3 standard
            errors off), logged as a warning when not calibrated; ``centre``, the same for the
            intervals of probability below 0.9. With ``information``, also ``information``, a dict
            with ``mean_excess``, the mean over the data of the excess surprise (the ``excess``
            column of :meth:`table`), its standard error under calibration ``se``, ``z`` =
            mean_excess/se, ``n``, the data it averages, ``outside_support``, the data where their
            law has no density (possible only beyond the end of a bounded Pareto tail,
            ``tails="gpd"``), left out of the mean, ``base`` and ``verdict``: ``"calibrated"``,
            ``"too narrow or off-centre"`` (z above 3, or some datum outside the support of its law)
            or ``"too wide"`` (z below −3), logged as a warning when not calibrated.

        Notes
        -----
        No theorem makes the cross-validated laws of an ensemble calibrated, so this is a reading,
        not a property. Too narrow laws can be widened (:class:`~spatialize.empirical.FittedModelFactory`,
        ``widening``) or obtained from coarser partitions (a smaller ``alpha``).

        **The information check.** The log score of datum :math:`i` under its law,
        :math:`\ell_i = -\log \tilde f_i(z_i)`, has expectation :math:`H(\tilde f_i)`, the entropy of
        the law, when :math:`z_i` is drawn from it. The excess surprise
        :math:`e_i = \ell_i - \hat H_i` then has mean 0, with :math:`\hat H_i` the entropy estimated
        from :math:`N_i` draws of the law's density (as many as its members), and its variance is
        :math:`v_i (1 + 1/N_i)`, with :math:`v_i` the variance of :math:`\log \tilde f_i` under
        :math:`\tilde f_i`, read on the same draws. The check compares the mean excess
        :math:`\bar e = \frac1n \sum_i e_i` with its standard error
        :math:`\frac1n \big(\sum_i v_i (1 + 1/N_i)\big)^{1/2}`.

        The coverage reads a few intervals of every law, the information check its whole density,
        each part weighted by the log score, the strictly proper score the hyperparameter searches
        minimise. A positive mean excess says that the data are less probable under their laws than
        the laws expect: laws too narrow, centred away from the data, or with tails too light. A
        negative one says that the laws are too wide. Two cautions apply. The standard error
        assumes independent data, as the binomial errors of the coverage do, while neighbouring
        laws share data, so it is a lower bound and :math:`|z|` an upper one. A few gross errors
        can carry the mean on their own, their excess growing with the distance to the law (slowly
        with the Student-t kernels, as the square of the distance with the Gaussian ones): the
        ``excess`` column of :meth:`table` tells whether the mean comes from the bulk of the data or
        from a few of them. Every ``tails`` reads the check on the law its p-values use; under
        ``"gpd"`` that law is spliced from a Gaussian kernel density between the 10 % and 90 %
        quantiles of the sample and the pooled generalized Pareto tails beyond them.
        """
        from scipy import stats
        pit = self.pit()
        pit = pit[np.isfinite(pit)]
        n = len(pit)
        rows = []
        for a in sorted(alphas):
            inside = float(np.mean((pit >= (1 - a) / 2) & (pit <= (1 + a) / 2))) if n else np.nan
            se = np.sqrt(a * (1 - a) / n) if n else np.nan
            rows.append({"alpha": a, "observed": inside, "se": se, "z": (inside - a) / se if n else np.nan})
        cov = pd.DataFrame(rows)
        if self.calibrate:
            raw = self._widened()
            before = np.array([np.mean(x < z) if x.size else np.nan for x, z in zip(raw, self._tvalues)])
            before = before[np.isfinite(before)]
            cov["before_factor"] = [float(np.mean((before >= (1 - a) / 2) & (before <= (1 + a) / 2)))
                                    for a in cov["alpha"]]
        ks_p = float(stats.kstest(pit, "uniform").pvalue) if n else np.nan
        def judge(rows):
            if not n or rows.empty:
                return "calibrated"
            if (rows["z"] < -3).any():
                return "too narrow"
            return "too wide" if (rows["z"] > 3).any() else "calibrated"
        verdict = judge(cov[cov["alpha"] >= 0.9])       # the tails, which the flags read
        centre = judge(cov[cov["alpha"] < 0.9])
        if verdict != "calibrated":
            log_message(logging.logger.warning(
                f"the tails of the laws look {verdict}: " + ", ".join(
                    f"{r.alpha:g} → {r.observed:.3f}" for r in cov.itertuples()) +
                ("; flags are inflated, consider widening or coarser partitions" if verdict == "too narrow"
                 else "; surprises may be hidden")))
        out = {"coverage": cov, "ks_p": ks_p, "n": n, "verdict": verdict, "centre": centre,
               "spread_factor": self.spread_factor}
        if information:
            out["information"] = self._information_check()
        return out

    def _information_check(self):
        """The mean excess surprise against its standard error under calibration (:meth:`calibration`)."""
        logf, ent, var, _, _ = self._kde()
        draws = np.array([self.law(i).size for i in range(len(self.values))], dtype=float)
        ok = np.isfinite(logf) & np.isfinite(ent) & np.isfinite(var)
        # a datum where its law has no density (beyond the end of a bounded Pareto tail) is impossible
        # under it: counted apart, since its excess is infinite
        outside = int(np.sum(np.isneginf(logf) & np.isfinite(ent)))
        n, nats = int(ok.sum()), np.log(self.base)
        if not n:
            return {"mean_excess": np.nan, "se": np.nan, "z": np.nan, "n": 0, "outside_support": outside,
                    "base": self.base, "verdict": "too narrow or off-centre" if outside else "calibrated"}
        mean = float(np.mean(-logf[ok] - ent[ok])) / nats
        se = float(np.sqrt(np.sum(var[ok] * (1 + 1 / draws[ok])))) / n / nats
        z = mean / se if se > 0 else np.nan
        verdict = "too narrow or off-centre" if z > 3 or outside else "too wide" if z < -3 else "calibrated"
        if verdict != "calibrated":
            log_message(logging.logger.warning(
                f"the laws look {verdict} by the information check: mean excess surprise {mean:.3f} "
                f"(standard error {se:.3f}, z = {z:.1f})"
                + (f"; {outside} data lie where their laws have no density" if outside else "")))
        return {"mean_excess": mean, "se": se, "z": z, "n": n, "outside_support": outside, "base": self.base,
                "verdict": verdict}

    # ------------------------------------------------------------------ plots
    def plot_calibration(self, alphas=None, theme='alges', color=None, **figargs):
        """The histogram of the positions of the data in their laws, against the uniform law, and the
        observed coverage of the central intervals against their probability.

        Parameters
        ----------
        alphas : sequence of float, optional
            The probabilities of the coverage curve. Default: 0.05 to 0.99.
        theme : str, optional
            Plot theme (:class:`~spatialize.viz.PlotStyle`). Default: ``'alges'``.
        color : str, optional
            Main colour. Default: the theme's.
        **figargs
            Passed to :func:`matplotlib.pyplot.subplots`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        alphas = np.linspace(0.05, 0.99, 20) if alphas is None else np.asarray(alphas)
        pit = self.pit()
        pit = pit[np.isfinite(pit)]
        n = max(len(pit), 1)
        figargs.setdefault("figsize", (10, 4))
        with PlotStyle(theme=theme, color=color) as style:
            fig, ax = plt.subplots(1, 2, **figargs)
            bins = 10
            ax[0].hist(pit, bins=bins, range=(0, 1), density=True, histtype='stepfilled', alpha=0.8,
                       color=style.color, zorder=3)
            band = 2 * np.sqrt(bins / n * (1 - 1 / bins))
            ax[0].axhspan(1 - band, 1 + band, color='grey', alpha=0.2, zorder=1)
            ax[0].axhline(1, color='grey', lw=1, zorder=2)
            ax[0].set_xlabel("position of the datum in its law (PIT)")
            ax[0].set_ylabel("density")
            ax[0].set_title("Positions")
            obs = [np.mean((pit >= (1 - a) / 2) & (pit <= (1 + a) / 2)) for a in alphas]
            ax[1].plot([0, 1], [0, 1], color='grey', lw=1)
            ax[1].plot(alphas, obs, 'o-', color=style.color, ms=3)
            ax[1].set_xlabel("probability of the central interval")
            ax[1].set_ylabel("share of data inside")
            ax[1].set_title("Coverage")
            ax[1].set_aspect('equal', adjustable='box')
            fig.tight_layout()
        return fig

    def plot_map(self, q=0.05, theme='alges', cmap=None, **figargs):
        """The data on their first two coordinates, coloured by their surprise, the flagged data
        circled.

        Parameters
        ----------
        q : float, optional
            The false discovery rate of the flags. Default: 0.05.
        theme : str, optional
            Plot theme. Default: ``'alges'``.
        cmap : str or Colormap, optional
            Colour map of the surprise. Default: the theme's.
        **figargs
            Passed to :func:`matplotlib.pyplot.subplots`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        pts = np.asarray(self.points, dtype=float)
        y = pts[:, 1] if pts.shape[1] > 1 else np.zeros(len(pts))
        with np.errstate(divide="ignore"):
            s = -np.log10(self.tail_p())
        flag = self.flags(q)
        figargs.setdefault("figsize", (7, 6))
        with PlotStyle(theme=theme, cmap=cmap) as style:
            fig, ax = plt.subplots(1, 1, **figargs)
            base = matplotlib.colormaps[style.cmap] if isinstance(style.cmap, str) else style.cmap
            order = np.argsort(np.nan_to_num(s, nan=-1.0))      # the most surprising data drawn on top
            sc = ax.scatter(pts[order, 0], y[order], c=s[order], cmap=base.reversed(), s=25, zorder=3,
                            vmin=0, vmax=max(3.0, float(np.nanmax(s[np.isfinite(s)])) if np.isfinite(s).any() else 3.0))
            ax.scatter(pts[flag, 0], y[flag], s=120, facecolors='none', edgecolors=plt.rcParams['text.color'],
                       linewidths=1.2, zorder=4, label=f"flagged (FDR {q:g})")
            plt.colorbar(sc, ax=ax, label="surprise, −log10 p")
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("Surprise of each datum")
            if flag.any():
                ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
        return fig

    def plot_evidence(self, theme='alges', cmap=None, **figargs):
        """The data on their first two coordinates, coloured by the support of their laws and by the
        information gain (:meth:`evidence`), the data with a support below 0.5 circled in both.

        Parameters
        ----------
        theme : str, optional
            Plot theme (:class:`~spatialize.viz.PlotStyle`). Default: ``'alges'``.
        cmap : str or Colormap, optional
            Colour map of the support. Default: the theme's. The gain uses the diverging palette
            ``vik``, centred at 0.
        **figargs
            Passed to :func:`matplotlib.pyplot.subplots`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from matplotlib.colors import TwoSlopeNorm
        from spatialize.viz import get_palette_colormap
        ev = self.evidence()
        pts = np.asarray(self.points, dtype=float)
        y = pts[:, 1] if pts.shape[1] > 1 else np.zeros(len(pts))
        weak = ev["support"].to_numpy() < 0.5
        gain = ev["gain"].to_numpy()
        finite = gain[np.isfinite(gain)]
        span = max(float(np.max(np.abs(finite))) if finite.size else 1.0, 1e-9)
        unit = {2.0: "bits", 10.0: "hartleys"}.get(self.base, "nats" if np.isclose(self.base, np.e) else f"log base {self.base:g}")
        figargs.setdefault("figsize", (13, 5.5))
        with PlotStyle(theme=theme, cmap=cmap) as style:
            fig, ax = plt.subplots(1, 2, **figargs)
            panels = [(ax[0], ev["support"].to_numpy(), style.cmap, dict(vmin=0, vmax=1), "support (share of partitions)",
                       "Support of each law"),
                      (ax[1], gain, get_palette_colormap("vik"), dict(norm=TwoSlopeNorm(0, -span, span)),
                       f"information gain, {unit}", "Information from the other data")]
            for a, c, cm, scale, label, title in panels:
                # a thin outline keeps the points whose colour is the background's visible (a support of 1
                # everywhere under empty_cells="coarsen", a gain near 0)
                sc = a.scatter(pts[:, 0], y, c=c, cmap=cm, s=25, zorder=3, edgecolors="0.45", linewidths=0.3,
                               **scale)
                a.scatter(pts[weak, 0], y[weak], s=110, facecolors='none', edgecolors=plt.rcParams['text.color'],
                          linewidths=1.0, zorder=4, label="support < 0.5")
                plt.colorbar(sc, ax=a, label=label)
                a.set_aspect('equal', adjustable='box')
                a.set_xlabel("X")
                a.set_ylabel("Y")
                a.set_title(title)
                if weak.any():
                    a.legend(loc="best", fontsize=8)
            fig.tight_layout()
        return fig

    def plot_recommendations(self, recommendations, labels=True, theme='alges', **figargs):
        """The data on their first two coordinates, each datum of a list of :meth:`recommend` marked by
        its action.

        Parameters
        ----------
        recommendations : pandas.DataFrame
            The result of :meth:`recommend` on this audit (or on a
            :class:`~spatialize.gs.spa.CategoricalPosteriorAudit`).
        labels : bool, optional
            Whether to write the number of each datum to measure again, check or measure as one of a
            pair beside it, as in the ``datum`` column. Default: True.
        theme : str, optional
            Plot theme (:class:`~spatialize.viz.PlotStyle`). Default: ``'alges'``.
        **figargs
            Passed to :func:`matplotlib.pyplot.subplots`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        pts = np.asarray(self.points, dtype=float)
        y = pts[:, 1] if pts.shape[1] > 1 else np.zeros(len(pts))
        styles = {"measure again": dict(marker="o", s=110, facecolors="none", edgecolors="#c0392b", linewidths=1.8),
                  "check provenance": dict(marker="o", s=110, facecolors="none", edgecolors="#e67e22", linewidths=1.8),
                  "measure one of the pair again": dict(marker="D", s=90, facecolors="none", edgecolors="#7d3c98",
                                                        linewidths=1.6),
                  # the actions of the audit of categories
                  "check first": dict(marker="o", s=100, facecolors="none", edgecolors="#d81b60", linewidths=1.6),
                  "classify one of the pair again": dict(marker="D", s=90, facecolors="none", edgecolors="#7d3c98",
                                                         linewidths=1.6),
                  "sample more nearby": dict(marker="s", s=60, facecolors="none", edgecolors="#2e86c1", linewidths=1.2),
                  "none": dict(marker="o", s=70, facecolors="none", edgecolors="0.55", linewidths=1.0)}
        figargs.setdefault("figsize", (8, 7))
        with PlotStyle(theme=theme):
            fig, ax = plt.subplots(1, 1, **figargs)
            ax.scatter(pts[:, 0], y, s=8, c="0.75", zorder=1, label="data")
            for action, style in styles.items():
                rows = recommendations[recommendations["action"] == action]
                if not len(rows):
                    continue
                idx = rows["datum"].to_numpy(dtype=int)
                ax.scatter(pts[idx, 0], y[idx], zorder=3, label=f"{action} ({len(idx)})", **style)
                if labels and action in ("measure again", "check provenance", "measure one of the pair again",
                                         "check first", "classify one of the pair again"):
                    # below the point for a pair, whose other datum shares the location and is labelled above
                    offset = (7, -12) if action.endswith("of the pair again") else (7, 5)
                    for i in idx:
                        ax.annotate(str(i), (pts[i, 0], y[i]), xytext=offset, textcoords="offset points",
                                    fontsize=8, color=style["edgecolors"])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("Recommended actions")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8, frameon=False)
            fig.tight_layout()
        return fig

    def plot_value_pit(self, theme='alges', color=None, **figargs):
        """Each datum's value against its position in its law: high values systematically in the
        upper tail, or low ones in the lower, show laws that do not follow the values' range.

        Parameters
        ----------
        theme : str, optional
            Plot theme. Default: ``'alges'``.
        color : str, optional
            Point colour. Default: the theme's.
        **figargs
            Passed to :func:`matplotlib.pyplot.subplots`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        figargs.setdefault("figsize", (6, 4))
        with PlotStyle(theme=theme, color=color) as style:
            fig, ax = plt.subplots(1, 1, **figargs)
            ax.scatter(self.values, self.pit(), s=15, color=style.color, zorder=3)
            ax.axhline(0.5, color='grey', lw=1)
            ax.set_xlabel("value")
            ax.set_ylabel("position in its law (PIT)")
            ax.set_ylim(-0.02, 1.02)
            fig.tight_layout()
        return fig

    def plot_declustered(self, bins=30, theme='alges', color=None, **figargs):
        """The histogram of the values with and without the declustering weights (:meth:`weights`).

        Parameters
        ----------
        bins : int, optional
            Number of bins. Default: 30.
        theme : str, optional
            Plot theme. Default: ``'alges'``.
        color : str, optional
            Colour of the declustered histogram. Default: the theme's.
        **figargs
            Passed to :func:`matplotlib.pyplot.subplots`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        w = self.weights()
        figargs.setdefault("figsize", (6, 4))
        with PlotStyle(theme=theme, color=color) as style:
            fig, ax = plt.subplots(1, 1, **figargs)
            edges = np.histogram_bin_edges(self.values, bins=bins)
            ax.hist(self.values, bins=edges, density=True, histtype='step', lw=1.5, color='grey', label="naive",
                    zorder=3)
            ax.hist(self.values, bins=edges, weights=w, density=True, histtype='stepfilled', alpha=0.7,
                    color=style.color, label="declustered", zorder=2)
            ax.set_xlabel("value")
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
            fig.tight_layout()
        return fig

    def plot_proportional_effect(self, theme='alges', color=None, **figargs):
        """The width of each datum's law against its centre (:meth:`proportional_effect`).

        Parameters
        ----------
        theme : str, optional
            Plot theme. Default: ``'alges'``.
        color : str, optional
            Point colour. Default: the theme's.
        **figargs
            Passed to :func:`matplotlib.pyplot.subplots`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        pe = self.proportional_effect()
        figargs.setdefault("figsize", (6, 4))
        with PlotStyle(theme=theme, color=color) as style:
            fig, ax = plt.subplots(1, 1, **figargs)
            ax.scatter(pe["centre"], pe["spread"], s=15, color=style.color, zorder=3)
            ax.set_xlabel("centre of the law (median)")
            ax.set_ylabel("width of the central 90 % interval")
            ax.set_title(f"Spearman {pe['spearman']:.2f}")
            fig.tight_layout()
        return fig

    def plot_datum(self, i, bins=25, theme='alges', color=None, **figargs):
        """The law of datum ``i``: the histogram of its sample (:meth:`law`) and the datum.

        Parameters
        ----------
        i : int
            Index of the datum.
        bins : int, optional
            Number of bins. Default: 25.
        theme : str, optional
            Plot theme. Default: ``'alges'``.
        color : str, optional
            Histogram colour. Default: the theme's.
        **figargs
            Passed to :func:`matplotlib.pyplot.subplots`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        x = self._inv(self.law(i))
        x = x[np.isfinite(x)]
        figargs.setdefault("figsize", (6, 4))
        with PlotStyle(theme=theme, color=color) as style:
            fig, ax = plt.subplots(1, 1, **figargs)
            if x.size:
                ax.hist(x, bins=bins, density=True, histtype='stepfilled', alpha=0.8, color=style.color, zorder=2)
            ax.axvline(self.values[i], color='red', lw=1.5, zorder=4, label=f"datum {i}: {self.values[i]:.4g}")
            ax.set_xlabel("value")
            ax.set_ylabel("density")
            ax.set_title(f"Datum {i}: p = {self.tail_p()[i]:.3g}, support {self.support[i]:.2f}")
            ax.legend(fontsize=8)
            fig.tight_layout()
        return fig


class PosteriorSampleAnalyzer(PosteriorAudit):
    """The posterior analysis of version 1.2, on top of :class:`PosteriorAudit`.

    Parameters
    ----------
    cv_post_result : ndarray of shape (n, T)
        The cross-validation members at each datum.
    points : ndarray of shape (n, d)
        The data locations.
    sample_values : ndarray of shape (n,)
        The data values.
    fitted_model_factory : FittedModelFactory
        The density model fitted to each datum's members.
    callback : callable, optional
        Where the progress and the messages go. Default: ``None``, shown as the session settings
        ``display``, ``progress`` and ``verbosity`` say (:mod:`spatialize.session`). A callable
        receiving the messages of :mod:`spatialize.logging` sends them elsewhere, such as an
        application's own interface, while :func:`~spatialize.logging.singleton_null_callback`
        drops them.

    Attributes
    ----------
    post_result : ndarray
        The members, as ``members``.
    sample_values : ndarray
        The data values, as ``values``.
    emodels : dict of int to EmpiricalModel
        The fitted law of each datum that has one.
    sample_quantiles : dict of int to float
        The value of each datum's cumulative distribution function at the datum, NaN without a law.
    sample_entropy : dict of int to float
        The entropy of each datum's law, NaN without a law.

    Notes
    -----
    Since version 1.3 the law of a datum is fitted to its members alone. Version 1.2 added the datum
    to its own members, which put a kernel on it and capped its surprise.
    """

    def __init__(self, cv_post_result, points, sample_values, fitted_model_factory,
                 callback=None):
        super().__init__(cv_post_result, points, sample_values, fitted_model_factory, callback,
                         widening=False)
        self.post_result = cv_post_result
        self.sample_values = sample_values
        models = self.models()
        self.emodels = {i: m for i, m in enumerate(models) if m is not None}
        self.sample_quantiles, self.sample_entropy = {}, {}
        for i, m in enumerate(models):
            q = h = np.nan
            if m is not None:
                try:
                    h, q = float(m.entropy()), float(m.cdf(self.values[i]))
                except Exception as e:
                    log_message(logging.logger.debug(f"no readings for datum {i}: {e}"))
            self.sample_quantiles[i], self.sample_entropy[i] = q, h

    def rank_samples(self, entropy_mass_alphas=[0.5, 0.7, 0.9, 0.99]):
        """The level of each datum, as :meth:`PosteriorAudit.levels`.

        A sample is assigned to ``level_j`` if it falls outside the central probability interval of
        the ``(j+1)``-th largest alpha but inside every wider one; ``level_0`` holds the samples
        outside the widest interval (the tails), ``level_k`` (k = number of alphas) those inside
        every interval.

        Parameters
        ----------
        entropy_mass_alphas : list of float, optional
            The probabilities of the central intervals. Default: ``[0.5, 0.7, 0.9, 0.99]``.

        Returns
        -------
        pandas.DataFrame
            ``value`` and ``category`` (``"level_j"``, or None for a datum without defined members).

        Notes
        -----
        Since version 1.3 the intervals are central intervals of probability, read from the members,
        as the theory describes. Version 1.2 used the narrowest intervals holding a share of the
        entropy of the fitted density, whose probability differed from the share (about 0.55 for 0.5
        and 0.93 for 0.9), so the levels could not be read against their probabilities. The
        parameter keeps its name for compatibility.
        """
        categories = self.levels(entropy_mass_alphas)
        log_message(logging.logger.info(
            f"categorized {len(categories)} samples into {len(set(c for c in categories if c is not None))} categories."))
        return pd.DataFrame({'value': self.sample_values, 'category': list(categories)})

    def plot_summary(self, theme='alges', color=None, **figargs):
        """Histograms of the values, of the cumulative probabilities of the data in their laws and of
        the entropies of the laws.

        Parameters
        ----------
        theme : str, optional
            Plot theme (:class:`~spatialize.viz.PlotStyle`). Default: ``'alges'``.
        color : str, optional
            Histogram colour. Default: the theme's.
        **figargs
            Passed to :func:`matplotlib.pyplot.subplots`, e.g. ``figsize=(12, 4)``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        with PlotStyle(theme=theme, color=color) as style:
            fig, ax = plt.subplots(1, 3, **figargs)
            fig.suptitle("Posterior Sample Analysis")
            fig.subplots_adjust(wspace=0.3)
            for a, data, title in ((ax[0], self.values, "Value"),
                                   (ax[1], list(self.sample_quantiles.values()), "Percentiles"),
                                   (ax[2], list(self.sample_entropy.values()), "Entropy")):
                data = np.asarray(data, dtype=float)
                a.hist(data[np.isfinite(data)], 25, density=True, histtype='stepfilled', alpha=0.8,
                       color=style.color, zorder=3)
                a.set_title(title)
        return fig

    def quick_plot_models(self, n_imgs=6, n_cols=3, seed=42, theme='alges', cmap=None, **figargs):
        """A grid of the laws of randomly chosen data: histogram of the members, density and scaled
        cumulative distribution function.

        Parameters
        ----------
        n_imgs : int, optional
            Number of data shown. Default: 6.
        n_cols : int, optional
            Number of columns of the grid. Default: 3.
        seed : int, optional
            Seed of the choice of data. Default: 42.
        theme : str, optional
            Plot theme. Default: ``'alges'``.
        cmap : str or Colormap, optional
            Colours of the panels. Default: the theme's.
        **figargs
            Passed to :func:`matplotlib.pyplot.subplots`.

        Returns
        -------
        matplotlib.figure.Figure or None
            None when no datum has a law.
        """
        available = sorted(self.emodels)
        n_imgs = min(n_imgs, len(available))
        if not n_imgs:
            log_message(logging.logger.warning("No empirical models available to plot in quick_plot_models."))
            return None
        n_rows = -(-n_imgs // n_cols)
        state = rd.getstate()
        if seed is not None:
            rd.seed(seed)
        idx = rd.sample(available, k=n_imgs)
        rd.setstate(state)
        return plot_histogram_grid_with_pdf_cdf(self.post_result, idx, self.emodels, n_rows, n_cols, bins=25,
                                                theme=theme, cmap=cmap, **figargs)

    def plot_ranking(self, samples_ranking, theme='alges', color=None, cmap=None, figsize=(11, 6)):
        """The counts of each category and the data coloured by category, on their first two
        coordinates.

        Parameters
        ----------
        samples_ranking : pandas.DataFrame
            The output of :meth:`rank_samples`.
        theme : str, optional
            Plot theme. Default: ``'alges'``.
        color : str, optional
            Bar colour. Default: the theme's.
        cmap : str or Colormap, optional
            Colours of the categories. Default: the theme's.
        figsize : tuple, optional
            Figure size. Default: ``(11, 6)``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        with PlotStyle(theme=theme, color=color, cmap=cmap) as style:
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            categories = samples_ranking["category"]
            counts = categories.value_counts().sort_index()
            ax[0].bar(counts.index.astype(str), counts.values, color=style.color, zorder=3)
            ax[0].set_title("Categories")
            ax[0].tick_params(axis='x', rotation=45)

            names = sorted(set(str(c) for c in categories if c is not None))
            ax[1].set_title("Categorized Samples")
            if not names:
                log_message(logging.logger.warning("No valid categories to plot in plot_ranking."))
                fig.tight_layout()
                return fig
            number = {c: i for i, c in enumerate(names)}
            nums = np.array([number.get(str(c), -1) if c is not None else -1 for c in categories])
            ok = nums >= 0
            base = matplotlib.colormaps[style.cmap] if isinstance(style.cmap, str) else style.cmap
            cat_cmap = ListedColormap(base.resampled(max(len(names), 2))(np.linspace(0, 1, len(names))))
            pts = np.asarray(self.points, dtype=float)
            y = pts[ok, 1] if pts.shape[1] > 1 else np.zeros(ok.sum())
            sc = ax[1].scatter(pts[ok, 0], y, c=nums[ok], cmap=cat_cmap, s=30, edgecolor='white',
                               linewidth=0.5, zorder=3, vmin=-0.5, vmax=len(names) - 0.5)
            cbar = plt.colorbar(sc, ax=ax[1], ticks=list(range(len(names))), orientation='vertical')
            cbar.set_ticklabels(names)
            ax[1].set_aspect('equal', adjustable='box')
            ax[1].set_xlabel("X")
            ax[1].set_ylabel("Y")
            fig.tight_layout()
        return fig


@signature_overload(pivot_arg=("local_interpolator", li.IDW, "local interpolator"),
                    common_args=dict(_COMMON, widening="auto", widening_knn=12, scale="raw", calibrate=True,
                                     tails="t", nu=3.0, base=np.e),
                    specific_args=_SPECIFIC)
def posterior_audit(points, values, **kwargs):
    """The predictive law of each datum built from the other data.

    Each datum is predicted from the other data by cross-validation (leave-one-out, or k-fold with
    ``k``) with the partitions and the decoder, so its law carries the evidence the rest of the data
    give about it, with no variogram or model chosen beforehand.

    Parameters
    ----------
    points : array_like of shape (n, d)
        The data locations.
    values : array_like of shape (n,)
        The data values.
    local_interpolator : str, optional
        The decoder, any of the catalogue (:func:`~spatialize.gs.esi.esi_nongriddata`), with its
        parameters as keyword arguments. Default: ``"idw"``.
    k : int or str, optional
        ``"loo"``, ``-1`` or n for leave-one-out, otherwise the number of folds. Default: -1.
    p_process : {"mondrian", "mondrian-legacy", "voronoi"}, optional
        The partition process. Default: ``"mondrian"``.
    data_cond : bool, optional
        Voronoi only: nuclei among the data (True) or uniform in the box. Default: True.
    n_partitions : int, optional
        The number of partitions, the members of each law. Default: 200.
    alpha : float, optional
        The granularity of the partitions. Default: 0.8.
    seed, folding_seed : int, optional
        Seeds of the partitions and of the folds. Default: drawn at random.
    widening : {"auto", "gamma", "skew_normal", False}, optional
        How the members are widened before they are read (:class:`PosteriorAudit`). Default:
        ``"auto"``.
    widening_knn : int, optional
        The number of nearest other data the widening reads. Default: 12.
    scale : {"raw", "yeojohnson", "normal_scores"}, optional
        The scale the laws are read on (:class:`PosteriorAudit`). Default: ``"raw"``.
    calibrate : bool, optional
        Whether one factor on every law's spread brings the 90 % coverage to nominal. Default: True.
    tails : {"t", "gpd", "normal"}, optional
        The model of the laws' tails (:class:`PosteriorAudit`). Default: ``"t"``.
    nu : float, optional
        The degrees of freedom of the Student-t kernels. Default: 3.
    base : float, optional
        The base of the logarithms of the information readings: e for nats, 2 for bits
        (:class:`PosteriorAudit`). Default: e.
    fitted_model_factory : FittedModelFactory, optional
        The density model of :meth:`PosteriorAudit.model`, for plots. Default: a variational
        Gaussian mixture of three components.
    best_params_found : dict, optional
        The output of a search's ``best_result()``; its keys override the arguments, except
        ``n_partitions``. The dict is not modified. Default: None.
    callback : callable, optional
        Where the progress and the messages go. Default: ``None``, shown as the session settings
        ``display``, ``progress`` and ``verbosity`` say (:mod:`spatialize.session`). A callable
        receiving the messages of :mod:`spatialize.logging` sends them elsewhere, such as an
        application's own interface, while :func:`~spatialize.logging.singleton_null_callback`
        drops them.

    Returns
    -------
    PosteriorAudit

    Notes
    -----
    The partitions are drawn on the session domain (:mod:`spatialize.session`), or on the box of the
    data without one. Cells without other data follow the session setting ``empty_cells``: under
    ``"nan"``, the default, an isolated datum's law rests on the partitions in which its cell holds
    other data, the share :attr:`PosteriorAudit.support` reports.
    """
    kwargs = _with_best_params(kwargs)
    members = _cv_members(points, values, points, kwargs)
    audit = PosteriorAudit(members, points, values, kwargs["fitted_model_factory"], callback=kwargs["callback"],
                           widening=kwargs["widening"], widening_knn=kwargs["widening_knn"], seed=kwargs["seed"],
                           scale=kwargs["scale"], calibrate=kwargs["calibrate"], tails=kwargs["tails"],
                           nu=kwargs["nu"], cells=_cells_reader(points, kwargs), base=kwargs["base"],
                           refit=lambda z: _cv_members(points, z, points, kwargs))
    weak = int(np.sum(audit.support < 0.5))
    if weak:
        log_message(logging.logger.warning(
            f"{weak} of {len(audit.values)} data have a law resting on fewer than half of the partitions "
            f"(support < 0.5): isolated data, whose cells often hold no other datum. The session setting "
            f"empty_cells='mark' or 'coarsen' gives them a law in every partition"))
    return audit


@signature_overload(pivot_arg=("local_interpolator", li.IDW, "local interpolator"),
                    common_args=dict(_COMMON, griddata=False, agg_function=af.mean),
                    specific_args=_SPECIFIC)
def cv_sample_pred_posterior(points, values, xi, **kwargs):
    """The posterior analysis of version 1.2: the law of each datum built from the other data.

    Parameters
    ----------
    points : ndarray of shape (n, d)
        The data locations.
    values : ndarray of shape (n,)
        The data values.
    xi : ndarray or tuple
        Locations that only enlarge the box the partitions are drawn on, as in version 1.2.
        :func:`posterior_audit` has no such argument, its box being the session domain or that of
        the data.
    **kwargs
        As :func:`posterior_audit`; ``griddata`` and ``agg_function`` are accepted and ignored.

    Returns
    -------
    PosteriorSampleAnalyzer

    Raises
    ------
    SpatializeError
        If the cross-validation fails.
    """
    kwargs = _with_best_params(kwargs)
    queries = deepcopy(xi) if isinstance(xi, tuple) else np.asarray(xi).copy()
    members = _cv_members(points, values, queries, kwargs)
    log_message(logging.logger.info(f"using fitted model factory: {kwargs['fitted_model_factory']}"))
    return PosteriorSampleAnalyzer(members, points, values, kwargs['fitted_model_factory'],
                                   callback=kwargs['callback'])


def plot_histogram_grid_with_pdf_cdf(r, data_indices, emodels, n_rows, n_cols, bins=25, figsize=(15, 10),
                                     theme='alges', cmap=None):
    """A grid of the laws of the given data: histogram of the fitted data, density and scaled
    cumulative distribution function.

    Parameters
    ----------
    r : ndarray of shape (n, T)
        The members of each datum.
    data_indices : list of int
        The data shown.
    emodels : dict of int to EmpiricalModel, or list
        The fitted law of each datum.
    n_rows, n_cols : int
        The shape of the grid.
    bins : int, optional
        Number of bins of the histograms. Default: 25.
    figsize : tuple, optional
        Figure size. Default: ``(15, 10)``.
    theme : str, optional
        Plot theme. Default: ``'alges'``.
    cmap : str or Colormap, optional
        Colours of the panels. Default: the theme's.

    Returns
    -------
    matplotlib.figure.Figure
    """
    with PlotStyle(theme=theme, cmap=cmap) as style:
        base = matplotlib.colormaps[style.cmap] if isinstance(style.cmap, str) else style.cmap
        panel_colors = base(np.linspace(0.15, 0.85, max(n_rows * n_cols, 2)))
        pdf_color = plt.rcParams['text.color']
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            if i >= len(data_indices):
                ax.axis('off')
                continue
            idx = data_indices[i]
            try:
                emodel = emodels[idx]
            except (KeyError, IndexError, TypeError):
                log_message(logging.logger.warning(f"Empirical model for index {idx} not found."))
                ax.axis('off')
                continue
            data = getattr(emodel, "data_", None)
            if data is None:
                data = np.asarray(r[idx, :])
                data = data[np.isfinite(data)]
            ax.hist(data, bins=bins, density=True, histtype='stepfilled', alpha=0.8,
                    color=panel_colors[i % len(panel_colors)], edgecolor='black', zorder=2)
            ax.plot(emodel.x_, emodel.pdf_, '-', color=pdf_color, label="PDF", zorder=3)
            lo, hi = ax.get_ylim()
            ax.plot(emodel.x_, emodel.cdf_ * (hi - lo) + lo, '-b', label="CDF (scaled)", zorder=3)
            ax.set_title(f'Sample {idx}', fontsize=10)
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.legend(fontsize=6)
        fig.tight_layout()
    return fig

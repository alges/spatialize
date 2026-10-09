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
    the cross-validation: ``(n_probes, seed) -> (data cells (n, T), probe cells (n_probes, T))``."""
    from spatialize import session
    points = np.asarray(points, dtype=np.float32)
    alpha = kwargs["alpha"]
    if kwargs["p_process"] == partitioning_process.VORONOI and not kwargs["data_cond"]:
        alpha = -alpha

    def read(n_probes, seed):
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
    method = "loo" if k in (-1, n) else "kfold"
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


def _tail_readings(laws, values, tails, nu, q=0.9):
    """Per datum, under the tail model: log density at the datum, entropy read on the law's sample,
    and the probabilities of the law below and above the datum."""
    from scipy.special import logsumexp
    from scipy.stats import norm, t as student, genpareto
    from spatialize.empirical import silverman_bandwidth
    n = len(values)
    logf, ent, below, above = (np.full(n, np.nan) for _ in range(4))
    kernel = (norm, ()) if tails in ("normal", "gpd") else (student, (nu,))
    for i, (x, z) in enumerate(zip(laws, values)):
        if x.size < 2:
            continue
        h = silverman_bandwidth(x)
        u = (np.append(x, z)[:, None] - x[None, :]) / h
        logs = logsumexp(kernel[0].logpdf(u, *kernel[1]), axis=1) - np.log(x.size * h)
        logf[i], ent[i] = logs[-1], -float(np.mean(logs[:-1]))
        uz = (z - x) / h
        below[i] = float(np.exp(logsumexp(kernel[0].logcdf(uz, *kernel[1])) - np.log(x.size)))
        above[i] = float(np.exp(logsumexp(kernel[0].logsf(uz, *kernel[1])) - np.log(x.size)))
    if tails == "gpd":
        # pooled generalized Pareto tails over the laws standardised by median and IQR
        std, ex_hi, ex_lo = [], [], []
        for x in laws:
            if x.size < 2:
                std.append(None)
                continue
            m = np.median(x)
            sc = float(np.subtract(*np.percentile(x, [75, 25]))) or float(np.std(x)) or 1.0
            w = (x - m) / sc
            uh, ul = np.quantile(w, q), np.quantile(w, 1 - q)
            std.append((m, sc, w, uh, ul))
            ex_hi += list(w[w > uh] - uh)
            ex_lo += list(ul - w[w < ul])
        fit_hi = genpareto.fit(ex_hi, floc=0) if len(ex_hi) > 10 else None
        fit_lo = genpareto.fit(ex_lo, floc=0) if len(ex_lo) > 10 else None
        for i, (st, z) in enumerate(zip(std, values)):
            if st is None:
                continue
            m, sc, w, uh, ul = st
            xz = (z - m) / sc
            if xz > uh and fit_hi is not None:
                above[i] = (1 - q) * float(genpareto.sf(xz - uh, *fit_hi))
            else:
                above[i] = float(np.mean(w >= xz))
            if xz < ul and fit_lo is not None:
                below[i] = (1 - q) * float(genpareto.sf(ul - xz, *fit_lo))
            else:
                below[i] = float(np.mean(w <= xz))
    return logf, ent, below, above


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
        The model of the law beyond its sample, for the p-values and the log scores. ``"t"``, a
        kernel density with Student-t kernels of ``nu`` degrees of freedom; ``"gpd"``, generalized
        Pareto tails beyond the 10 % and 90 % quantiles of each law, with one shape and scale per
        side pooled over all the laws standardised by their median and interquartile range;
        ``"normal"``, a kernel density with Gaussian kernels. Default: ``"t"``.
    nu : float, optional
        The degrees of freedom of the Student-t kernels. Default: 3.
    cells : callable, optional
        ``cells(n_probes, seed) -> (data_cells, probe_cells)``, the labels of the cells of the data,
        shape (n, T), and of ``n_probes`` locations drawn uniformly in the box of the partitions,
        shape (n_probes, T), in the partitions of the members; needed by :meth:`weights`.
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
                 cells=None):
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
        self._fwd, self._inv = _scale_maps(scale, self.values)
        self._tvalues = self._fwd(self.values)
        self._laws = None
        self._factor = None
        self._readings = None
        self._cells = cells         # the cells of the data and of probe locations in the partitions
        self._weights = None

    def _widening_targets(self):
        f = self.fitted_model_factory
        if self._targets is None:
            # leave-one-out targets: each datum's k nearest OTHER data, so its own value does not
            # deflate its target
            var = _loo_target_variance(self.points, self.values, knn=f.widening_knn) if f.widening else None
            skew = (_loo_target_skewness(self.points, self.values, knn=f.widening_knn)
                    if f.widening == "skew_normal" else None)
            self._targets = (var, skew)
        return self._targets

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
            f = self.fitted_model_factory
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
            laws = []
            for j in range(len(self.values)):
                m = self.members[j]
                x = self._fwd(m[np.isfinite(m)])
                if self.widening and x.size >= 2 and var[j] > 0:
                    mode = self.widening
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
        _, _, below, above = self._kde()
        with np.errstate(invalid="ignore", divide="ignore"):
            return below / (below + above)

    def _kde(self):
        """Per datum: the log density of its law at the datum, the entropy of the law read on its
        sample, and the probabilities of the law below and above the datum, under ``tails``."""
        if self._readings is None:
            laws = [self.law(i) for i in range(len(self.values))]
            self._readings = _tail_readings(laws, self._tvalues, self.tails, self.nu)
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
        _, _, below, above = self._kde()
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
            of the datum under its law (with the kernels of ``tails``); ``entropy``, the entropy of
            that law read on its sample, so that ``excess`` = ``surprisal`` − ``entropy`` is near 0
            for a datum typical of its law; ``width90``, the width of the central 90 % interval of
            the law, in the units of the values; ``support`` (:attr:`support`); ``coherence``
            (:meth:`coherence`); ``shift`` (:meth:`shift`); ``weight`` (:meth:`weights`, NaN when the audit does not hold its
            partitions). The log scores and the entropy are on the scale of the readings
            (``scale``).
        """
        logf, ent, below, above = self._kde()
        tail = np.clip(2.0 * np.minimum(below, above), np.finfo(float).tiny, 1.0)
        q5 = np.array([np.percentile(self.law(i), 5) if self.law(i).size else np.nan for i in range(len(self.values))])
        q95 = np.array([np.percentile(self.law(i), 95) if self.law(i).size else np.nan for i in range(len(self.values))])
        lo, hi = self._inv(q5), self._inv(q95)
        return pd.DataFrame({"value": self.values, "pit": self.pit(), "tail_p": tail,
                             "flag": _bh(tail, q), "level": self.levels(alphas), "surprisal": -logf,
                             "entropy": ent, "excess": -logf - ent, "width90": hi - lo,
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
    def calibration(self, alphas=(0.5, 0.8, 0.9, 0.95, 0.99)):
        r"""Whether the laws are calibrated: read it before the flags.

        A datum lies inside the central interval of probability :math:`\alpha` of its law with
        probability :math:`\alpha` when the law is right, and its position (:meth:`pit`) is
        uniform. Laws too narrow put too many data in the tails, which inflates every flag; laws
        too wide hide surprises.

        Parameters
        ----------
        alphas : sequence of float, optional
            The probabilities of the intervals compared. Default: ``(0.5, 0.8, 0.9, 0.95, 0.99)``.

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
            intervals of probability below 0.9.

        Notes
        -----
        No theorem makes the cross-validated laws of an ensemble calibrated, so this is a reading,
        not a property. Too narrow laws can be widened (:class:`~spatialize.empirical.FittedModelFactory`,
        ``widening``) or obtained from coarser partitions (a smaller ``alpha``).
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
        return {"coverage": cov, "ks_p": ks_p, "n": n, "verdict": verdict, "centre": centre,
                "spread_factor": self.spread_factor}

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
                                     tails="t", nu=3.0),
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
    k : int, optional
        ``-1`` (or n) for leave-one-out, otherwise the number of folds. Default: -1.
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
                           nu=kwargs["nu"], cells=_cells_reader(points, kwargs))
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

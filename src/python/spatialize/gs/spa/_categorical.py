"""Posterior analysis of categorical data: each datum read against the law of the classes the other data
give it.

A single category carries little information, so the laws of the classes, once calibrated, give no class
much less than a few hundredths, while a flag among hundreds of data at a controlled false discovery rate
needs a probability of a few ten-thousandths. The flags of :class:`CategoricalPosteriorAudit` therefore
find gross errors only. Its main reading is an order of review, the data ranked by how surprising their
class is, which places most errors among the first few percent (:doc:`/theory/posterior`).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spatialize import SpatializeError, logging
from spatialize._util import signature_overload, random_seed
from spatialize.gs import lib_spatialize_facade, partitioning_process
from spatialize.logging import resolve_callback, log_message, singleton_null_callback
from spatialize.result import Summarised
from spatialize.viz import PlotStyle
from spatialize.gs.spa._main import (PosteriorAudit, _bh, _qvalues, _min_qvalue, _cells_reader,
                                     _weighted_silverman)


def _encode(values, ordinal_order=None):
    """The classes, in the order given or sorted by name, and the code of each value."""
    keys = np.array([str(v) for v in values])
    classes = [str(c) for c in ordinal_order] if ordinal_order is not None else sorted(set(keys))
    unknown = set(keys) - set(classes)
    if unknown:
        raise SpatializeError(f"values not in ordinal_order: {sorted(unknown)}")
    index = {c: k for k, c in enumerate(classes)}
    return classes, np.array([index[k] for k in keys], dtype=int)


def _cat_members(points, codes, kwargs):
    """The leave-one-out members, as class codes, NaN where the datum's cell held no other datum."""
    from spatialize.gs.cat_esi.classifiers import get_classifier_fns
    names = ("n_neighbors", "max_points", "n_cv_splits", "sklearn_classifier", "C", "kernel", "gamma",
             "n_estimators", "max_depth", "min_samples_split", "random_state")
    set_fn, clf_fn = get_classifier_fns(kwargs["classifier"], **{k: kwargs.get(k) for k in names})
    alpha = kwargs["alpha"]
    if kwargs["p_process"] == partitioning_process.VORONOI and not kwargs["data_cond"]:
        alpha = -alpha
    try:
        _, raw = lib_spatialize_facade.run(
            np.float64(points), np.asarray(codes, np.float64), np.float64(points), kwargs["p_process"], "custom",
            {"post_creation": set_fn, "estimation": clf_fn}, alpha, kwargs["n_partitions"], kwargs["seed"],
            method="loo", k=0, callback=kwargs["callback"])
    except Exception as e:
        raise SpatializeError(e) from e
    return np.asarray(raw, dtype=np.float64)


def _local_shares(points, codes, n_classes, k):
    """The class shares among the k nearest other data of each datum."""
    from scipy.spatial import cKDTree
    pts = np.asarray(points, float)
    n = len(codes)
    k = max(1, min(int(k), n - 1))
    _, idx = cKDTree(pts).query(pts, k=k + 1)
    idx = np.reshape(idx, (n, -1))
    out = np.zeros((n, n_classes))
    for i in range(n):
        nb = idx[i][idx[i] != i][:k]
        out[i] = np.bincount(codes[nb], minlength=n_classes) / len(nb)
    return out


def _fit_mixture(parts, codes):
    """The weights of the mixture of the class laws in ``parts`` (each of shape (n, K)) that maximise the
    mean log score of the observed classes, the first part taking what the others leave."""
    from scipy.optimize import minimize
    n = len(codes)
    rows = np.arange(n)

    def mix(theta):
        e = np.exp(theta)
        w = np.concatenate([[1.0], e]) / (1.0 + e.sum())
        return sum(wk * p for wk, p in zip(w, parts)), w

    def loss(theta):
        p = mix(theta)[0][rows, codes]
        return -np.mean(np.log(np.clip(p, 1e-300, None)))

    theta = minimize(loss, x0=np.full(len(parts) - 1, -2.0), method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000}).x
    return mix(theta)


def _positions(P, codes, rng, ordinal):
    """The randomised position of each observed class in its law: for nominal classes the mass of the
    classes less probable plus a uniform share of the ties, for ordinal classes the mass below plus a
    uniform share of the class. Uniform on [0, 1] when the observed class follows its law."""
    n = len(codes)
    rows = np.arange(n)
    U = rng.random(n)
    pz = P[rows, codes]
    if ordinal:
        below = np.array([P[i, :c].sum() for i, c in enumerate(codes)])
        return below + U * pz
    less = np.sum(np.where(P < pz[:, None] - 1e-15, P, 0.0), axis=1)
    ties = np.sum(np.where(np.abs(P - pz[:, None]) <= 1e-15, P, 0.0), axis=1)
    return less + U * ties


class CategoricalPosteriorAudit(Summarised):
    """The law of the classes at each datum built from the other data, and its readings.

    Parameters
    ----------
    members : ndarray of shape (n, T)
        The leave-one-out members at each datum, as class codes (positions in ``classes``), one per
        partition, NaN where the datum's cell held no other datum.
    points : ndarray of shape (n, d)
        The data locations.
    values : array_like of shape (n,)
        The data classes.
    classes : list of str
        The classes, in their order when ``ordinal`` is True.
    ordinal : bool, optional
        Whether the classes are ordered, as in ``classes``. Default: False.
    local_k : int, optional
        The nearest other data whose class shares enter each law. Default: 8.
    seed : int, optional
        Seed of the randomisation of the positions over ties. Default: 0.
    base : float, optional
        The base of the logarithms of the information readings: e for nats, 2 for bits. Default: e.
    callback : callable, optional
        Where the progress and the messages go (:mod:`spatialize.logging`). Default: ``None``.
    cells, refit : callable, optional
        As in :class:`~spatialize.gs.spa.PosteriorAudit`, the cells of the data in the partitions and the
        members of other classes at the same locations, ``refit(codes) -> members``; provided by
        :func:`cat_posterior_audit`. Default: None.

    Attributes
    ----------
    members, points, values, classes
        As given.
    codes : ndarray of int
        The position of each datum's class in ``classes``.
    support : ndarray
        The share of defined members of each datum.
    mixture : ndarray of shape (3,)
        The weights of the members' shares, the shares among the ``local_k`` nearest other data and the
        shares of all the data in every law, fitted by the log score.

    Notes
    -----
    The law of datum :math:`i` mixes three laws of the classes, none of which holds the datum: the shares of
    its members, the shares among its nearest other data and the shares of all the data. Members alone give
    a class that none of them predicts no probability at all, while a clean datum at the edge of a unit or
    in a thin unit often gets no member of its class, so the nearest data and the whole data set give every
    class a floor. The weights maximise the mean log score of the observed classes, a strictly proper
    score, so the laws are calibrated on the data, errors included.
    """

    def __init__(self, members, points, values, classes, ordinal=False, local_k=8, seed=0, base=np.e, callback=None,
                 cells=None, refit=None):
        self.members = np.asarray(members, dtype=np.float64)
        self.points = np.asarray(points)
        self.values = np.asarray(values, dtype=object).ravel()
        self.classes = [str(c) for c in classes]
        index = {c: k for k, c in enumerate(self.classes)}
        try:
            self.codes = np.array([index[str(v)] for v in self.values], dtype=int)
        except KeyError as e:
            raise SpatializeError(f"value {e} not among the classes") from e
        if self.members.shape[0] != len(self.codes):
            raise SpatializeError(f"members has {self.members.shape[0]} rows for {len(self.codes)} data")
        if not (np.isfinite(base) and base > 0 and base != 1):
            raise SpatializeError(f"base must be a positive number other than 1; got {base!r}")
        self.ordinal, self.local_k, self.seed, self.base = bool(ordinal), int(local_k), int(seed or 0), float(base)
        self.callback = resolve_callback(callback)
        self.support = np.isfinite(self.members).mean(axis=1)
        self._cells, self._refit = cells, refit
        self._laws = None
        self._weights = None

    # ------------------------------------------------------------------ laws
    def _fit(self):
        if self._laws is None:
            K, n = len(self.classes), len(self.codes)
            defined = np.isfinite(self.members)
            counts = np.stack([np.sum(defined & (self.members == c), axis=1) for c in range(K)], axis=1).astype(float)
            total = counts.sum(axis=1, keepdims=True)
            glob = np.bincount(self.codes, minlength=K) / n
            share = np.where(total > 0, counts / np.maximum(total, 1), glob[None, :])
            local = _local_shares(self.points, self.codes, K, self.local_k)
            self._laws, self.mixture = _fit_mixture([share, local, np.broadcast_to(glob, (n, K))], self.codes)
        return self._laws

    def law(self, i):
        """The probability of each class at datum ``i`` (in the order of :attr:`classes`), built from the
        other data.

        Parameters
        ----------
        i : int
            Index of the datum.

        Returns
        -------
        ndarray of shape (K,)
        """
        return self._fit()[int(i)].copy()

    def laws(self):
        """The laws of every datum, shape (n, K), rows in the order of the data, columns of :attr:`classes`.

        Returns
        -------
        pandas.DataFrame
        """
        return pd.DataFrame(self._fit(), columns=self.classes)

    # ------------------------------------------------------------------ readings of each datum
    def pit(self):
        """The randomised position of each datum's class in its law (see Notes of
        :class:`CategoricalPosteriorAudit`), uniform on [0, 1] when the classes follow their laws.

        For nominal classes it is the probability of the classes less probable than the observed one plus a
        uniform share of the classes as probable, so a low position marks a surprising class. For ordinal
        classes it is the probability of the classes below plus a uniform share of the observed one, read on
        both sides as for amounts.

        Returns
        -------
        ndarray of shape (n,)
        """
        return _positions(self._fit(), self.codes, np.random.default_rng(self.seed), self.ordinal)

    def tail_p(self):
        """The surprise of each datum's class as a p-value: the position itself for nominal classes, the
        two-sided tail probability :math:`2\\min(u, 1 - u)` for ordinal ones.

        Returns
        -------
        ndarray of shape (n,)
        """
        u = self.pit()
        p = np.minimum(1.0, 2.0 * np.minimum(u, 1.0 - u)) if self.ordinal else u
        return np.clip(p, np.finfo(float).tiny, 1.0)

    def flags(self, q=0.05):
        """The data whose surprise survives the Benjamini–Hochberg procedure at false discovery rate ``q``.

        A calibrated law of the classes gives no class much less than a few hundredths, while a flag among
        hundreds of data needs a few ten-thousandths, so the flags find gross errors only. Read the order
        of review (:meth:`review`) for the rest.

        Parameters
        ----------
        q : float, optional
            The false discovery rate. Default: 0.05.

        Returns
        -------
        ndarray of bool, shape (n,)
        """
        return _bh(self.tail_p(), q)

    def table(self, q=0.05):
        """The readings of every datum, one row each.

        Parameters
        ----------
        q : float, optional
            The false discovery rate of the flags. Default: 0.05.

        Returns
        -------
        pandas.DataFrame
            ``value``, the class; ``probability``, that of the class under the datum's law;
            ``most_probable`` and its probability ``top_probability``; ``pit`` (:meth:`pit`);
            ``tail_p`` (:meth:`tail_p`); ``q_value``; ``flag`` (:meth:`flags`); ``surprisal``,
            :math:`-\\log p_i(z_i)`; ``entropy`` of the law, its expectation; ``excess``, their
            difference, with mean 0 when the classes follow their laws; ``gain`` (:meth:`evidence`);
            ``support``. The information readings are in the units of ``base``.
        """
        P = self._fit()
        n = len(self.codes)
        rows = np.arange(n)
        nats = np.log(self.base)
        pz = P[rows, self.codes]
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.sum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
        surprisal = -np.log(np.clip(pz, 1e-300, None))
        p = self.tail_p()
        top = np.argmax(P, axis=1)
        return pd.DataFrame({"value": self.values, "probability": pz, "most_probable": [self.classes[c] for c in top],
                             "top_probability": P[rows, top], "pit": self.pit(), "tail_p": p, "q_value": _qvalues(p),
                             "flag": _bh(p, q), "surprisal": surprisal / nats, "entropy": ent / nats,
                             "excess": (surprisal - ent) / nats, "gain": self.evidence()["gain"].to_numpy(),
                             "support": self.support})

    def review(self, share=0.05):
        """The data ranked by the surprise of their class, the most surprising first, cut at a share of the
        data: the list to check first.

        Parameters
        ----------
        share : float, optional
            The share of the data in the list, at least one datum. Default: 0.05.

        Returns
        -------
        pandas.DataFrame
            The rows of :meth:`table` for the listed data, with their ``datum``, ``rank`` and
            ``expected_by_chance``, the number of data whose class would be at least as surprising by chance
            alone when every class follows its law, the number of data times the datum's p-value (the
            p-values being uniform then). A datum well below 1 stands out from chance, while further down the
            list the count reaches the rank, the data there being as surprising as chance makes some of
            them. ``attrs`` holds ``share``, ``threshold``, the largest p-value in the list, and
            ``expected_clean``, the number of data times the threshold, which for a list cut at a share
            of the data is about its length.
        """
        t = self.table()
        n = len(t)
        m = max(1, int(round(float(share) * n)))
        order = np.argsort(t["tail_p"].to_numpy(), kind="stable")[:m]
        out = t.iloc[order].reset_index(names="datum")
        out.insert(1, "rank", np.arange(1, m + 1))
        out.insert(2, "expected_by_chance", n * out["tail_p"].to_numpy())
        threshold = float(t["tail_p"].to_numpy()[order[-1]])
        out.attrs.update({"share": float(share), "threshold": threshold, "expected_clean": min(m, n * threshold)})
        return out

    def evidence(self):
        """How much the other data tell about each datum: the support of its law and its information gain,
        the entropy of the class shares of all the data minus that of the law, in the units of ``base``.

        Returns
        -------
        pandas.DataFrame
            ``support``, ``entropy`` and ``gain``; ``attrs["prior_entropy"]`` holds the entropy of the class
            shares of all the data.
        """
        P = self._fit()
        nats = np.log(self.base)
        glob = np.bincount(self.codes, minlength=len(self.classes)) / len(self.codes)
        prior = -float(np.sum(glob[glob > 0] * np.log(glob[glob > 0])))
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.sum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
        out = pd.DataFrame({"support": self.support, "entropy": ent / nats, "gain": (prior - ent) / nats})
        out.attrs["prior_entropy"] = prior / nats
        return out

    # ------------------------------------------------------------------ global readings
    def calibration(self, bins=10, information=False):
        """Whether the laws of the classes are calibrated.

        Parameters
        ----------
        bins : int, optional
            The bins of predicted probability of the reliability table. Default: 10.
        information : bool, optional
            Whether to add the information check (see :meth:`PosteriorAudit.calibration
            <spatialize.gs.spa.PosteriorAudit.calibration>`), here exact, the entropy and the variance of
            the log probability of a discrete law being sums over its classes. Default: False.

        Returns
        -------
        dict
            ``reliability``, a DataFrame with, per bin of predicted probability over every class of every
            datum, the mean predicted probability, the observed frequency of the class and the count;
            ``ks_p``, the p-value of the Kolmogorov–Smirnov test of the positions against the uniform law;
            ``mixture``, the weights of the laws; with ``information``, ``information`` (``mean_excess``,
            ``se``, ``z``, ``n``, ``base``, ``verdict``).
        """
        from scipy import stats
        P = self._fit()
        n, K = P.shape
        hit = np.zeros_like(P)
        hit[np.arange(n), self.codes] = 1.0
        edges = np.linspace(0, 1, int(bins) + 1)
        which = np.clip(np.digitize(P.ravel(), edges) - 1, 0, int(bins) - 1)
        rows = []
        for b in range(int(bins)):
            sel = which == b
            if sel.any():
                rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "predicted": float(P.ravel()[sel].mean()),
                             "observed": float(hit.ravel()[sel].mean()), "count": int(sel.sum())})
        out = {"reliability": pd.DataFrame(rows), "ks_p": float(stats.kstest(self.pit(), "uniform").pvalue),
               "mixture": self.mixture}
        if information:
            with np.errstate(divide="ignore", invalid="ignore"):
                logP = np.where(P > 0, np.log(P), 0.0)
            ent = -np.sum(P * logP, axis=1)
            var = np.sum(P * logP ** 2, axis=1) - ent ** 2
            excess = -np.log(np.clip(P[np.arange(n), self.codes], 1e-300, None)) - ent
            nats = np.log(self.base)
            mean, se = float(np.mean(excess)) / nats, float(np.sqrt(np.sum(var))) / n / nats
            z = mean / se if se > 0 else np.nan
            verdict = "too narrow or off-centre" if z > 3 else "too wide" if z < -3 else "calibrated"
            out["information"] = {"mean_excess": mean, "se": se, "z": z, "n": n, "base": self.base, "verdict": verdict}
        return out

    # the readings that depend on the locations alone are those of the audit of amounts
    weights = PosteriorAudit.weights
    plot_map = PosteriorAudit.plot_map
    plot_evidence = PosteriorAudit.plot_evidence
    plot_recommendations = PosteriorAudit.plot_recommendations
    _fingerprint = PosteriorAudit._fingerprint

    def declustered(self):
        """The class shares of the data, plain and weighted by the declustering weights (:meth:`weights`).

        Returns
        -------
        pandas.DataFrame
            One row per class, columns ``naive`` and ``declustered``.
        """
        w = self.weights()
        K = len(self.classes)
        return pd.DataFrame({"naive": np.bincount(self.codes, minlength=K) / len(self.codes),
                             "declustered": np.bincount(self.codes, weights=w, minlength=K)}, index=self.classes)

    def duplicates(self, tol=0.0):
        """The pairs of data closer than ``tol`` (co-located when 0) with different classes, which cannot
        both be right.

        Parameters
        ----------
        tol : float, optional
            The distance below which two data count as co-located. Default: 0.

        Returns
        -------
        pandas.DataFrame
            ``i``, ``j``, ``distance``, ``value_i``, ``value_j``.
        """
        from scipy.spatial import cKDTree
        pts = np.asarray(self.points, float)
        pairs = np.array(sorted(cKDTree(pts).query_pairs(r=max(float(tol), 0.0) + 1e-12)), dtype=int).reshape(-1, 2)
        pairs = pairs[self.codes[pairs[:, 0]] != self.codes[pairs[:, 1]]] if len(pairs) else pairs
        i, j = pairs[:, 0], pairs[:, 1]
        return pd.DataFrame({"i": i, "j": j, "distance": np.sqrt(((pts[i] - pts[j]) ** 2).sum(axis=1)),
                             "value_i": self.values[i], "value_j": self.values[j]})

    def _settings(self):
        return dict(classes=self.classes, ordinal=self.ordinal, local_k=self.local_k, seed=self.seed, base=self.base)

    def null_flags(self, n_replicates=20, q=0.05, seed=None):
        """How many flags clean data would get at these locations with these settings: replicates of the
        classes drawn from the laws, each datum on its own, audited with the same partitions and classifier.

        Parameters
        ----------
        n_replicates : int, optional
            The number of replicates, each one a full audit. Default: 20.
        q : float, optional
            The false discovery rate of the flags counted. Default: 0.05.
        seed : int, optional
            Seed of the draws. Default: the audit's ``seed``.

        Returns
        -------
        pandas.DataFrame
            As :meth:`PosteriorAudit.null_flags <spatialize.gs.spa.PosteriorAudit.null_flags>`: one row per
            replicate (``flags``, ``min_qvalue``), ``attrs`` with ``q``, ``data``, ``observed``,
            ``expected``, ``share_with_flags``, ``p_value`` and ``p_value_strongest``.

        Notes
        -----
        Each replicate is a full audit, whose classifier runs in Python in every cell, about half a minute
        for 300 data and 200 partitions with ``"knn"``.
        """
        if self._refit is None:
            raise SpatializeError("null_flags needs to compute the members of other classes: build the audit with "
                                  "cat_posterior_audit, or give CategoricalPosteriorAudit its refit")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        P = self._fit()
        rows = []
        self.callback(logging.progress.init(int(n_replicates), 1, desc="replicates"))
        for _ in range(int(n_replicates)):
            draw = np.array([rng.choice(len(self.classes), p=row / row.sum()) for row in P])
            values = np.array(self.classes, dtype=object)[draw]
            rep = CategoricalPosteriorAudit(self._refit(draw), self.points, values, callback=singleton_null_callback,
                                            **self._settings())
            p = rep.tail_p()
            rows.append({"flags": int(_bh(p, q).sum()), "min_qvalue": _min_qvalue(p)})
            self.callback(logging.progress.inform())
        self.callback(logging.progress.stop())
        out = pd.DataFrame(rows)
        n_obs = int(self.flags(q).sum())
        strongest = _min_qvalue(self.tail_p())
        out.attrs.update({"q": q, "data": self._fingerprint(), "observed": n_obs, "expected": float(out["flags"].mean()),
                          "share_with_flags": float((out["flags"] > 0).mean()),
                          "p_value": float((1 + np.sum(out["flags"] >= n_obs)) / (len(out) + 1)),
                          "p_value_strongest": float((1 + np.sum(out["min_qvalue"] <= strongest)) / (len(out) + 1))})
        return out

    def cross(self, covariate, alpha=0.9, knn=20, q=0.05):
        """Each datum's class read against its law given a second variable measured at the same location.

        The law of the classes at datum :math:`i` given the covariate :math:`v_i` comes from the other data,
        each weighted by the share of partitions of granularity ``alpha`` in which it shares the cell of
        :math:`x_i` and, for a numeric covariate, by a Gaussian kernel in :math:`v_j - v_i` whose width holds
        at least ``knn`` of them, or, for a categorical one, by whether :math:`v_j = v_i`. The class shares
        of these weighted data are mixed with those of all the data, the weight fitted by the log score.

        Parameters
        ----------
        covariate : array_like of shape (n,)
            The second variable at the locations of the data, numeric or categorical, NaN or None where it
            was not measured.
        alpha : float, optional
            The granularity of the partitions that weigh the data. Default: 0.9.
        knn : int, optional
            The least number of data within the width of the kernel of a numeric covariate. Default: 20.
        q : float, optional
            The false discovery rate of the flags. Default: 0.05.

        Returns
        -------
        pandas.DataFrame
            ``covariate``, ``probability`` (of the observed class given the covariate), ``pit``,
            ``tail_p``, ``q_value``, ``flag``, ``gain`` (the entropy of the class shares of all the data
            minus that of the law given the covariate, in the units of ``base``); ``attrs`` with ``q`` and
            ``data``.
        """
        if self._cells is None:
            raise SpatializeError("the cross reading needs the partitions: build the audit with cat_posterior_audit, "
                                  "or give CategoricalPosteriorAudit its cells")
        cov = np.asarray(covariate, dtype=object).ravel()
        n, K = len(self.codes), len(self.classes)
        if cov.size != n:
            raise SpatializeError(f"covariate has {cov.size} values for {n} data")
        numeric = all(isinstance(x, (int, float, np.integer, np.floating)) or x is None for x in cov)
        if numeric:
            v = np.array([np.nan if x is None else float(x) for x in cov])
            usable = np.isfinite(v)
            v0 = np.where(usable, v, 0.0)
        else:
            usable = np.array([x is not None and not (isinstance(x, float) and np.isnan(x)) for x in cov])
            v = np.array([str(x) for x in cov])
        labels = self._cells(0, self.seed, alpha)[0]
        glob = np.bincount(self.codes, minlength=K) / n
        raw = np.tile(glob, (n, 1))
        for i in np.flatnonzero(usable):
            w = np.mean(labels == labels[i], axis=1)
            w[i] = 0.0
            w[~usable] = 0.0
            if np.count_nonzero(w) < 3:
                continue
            if numeric:
                hv, _ = _weighted_silverman(v0, w)
                dv = np.sort(np.abs(v0[w > 0] - v0[i]))
                hv = max(hv, dv[min(int(knn), dv.size) - 1], 1e-12)
                c = w * np.exp(-0.5 * ((v0 - v0[i]) / hv) ** 2)
            else:
                c = w * (v == v[i])
            if c.sum() > 0:
                raw[i] = np.bincount(self.codes, weights=c, minlength=K) / c.sum()
        P, _ = _fit_mixture([raw, np.broadcast_to(glob, (n, K))], self.codes)
        u = _positions(P, self.codes, np.random.default_rng(self.seed), self.ordinal)
        p = np.minimum(1.0, 2.0 * np.minimum(u, 1.0 - u)) if self.ordinal else u
        p = np.where(usable, np.clip(p, np.finfo(float).tiny, 1.0), np.nan)
        nats = np.log(self.base)
        prior = -float(np.sum(glob[glob > 0] * np.log(glob[glob > 0])))
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.sum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
        out = pd.DataFrame({"covariate": cov, "probability": np.where(usable, P[np.arange(n), self.codes], np.nan),
                            "pit": np.where(usable, u, np.nan), "tail_p": p, "q_value": _qvalues(p), "flag": _bh(p, q),
                            "gain": np.where(usable, (prior - ent) / nats, np.nan)})
        out.attrs.update({"q": q, "data": self._fingerprint()})
        return out

    def recommend(self, q=0.05, share=0.05, cross=None, null=None, k=8, min_support=0.5, min_gain=0.0, tol=0.0):
        r"""The data that call for an action, each with the action, the reason and its strength.

        Parameters
        ----------
        q : float, optional
            The false discovery rate of the flags. Default: 0.05.
        share : float, optional
            The share of the data in the list to check first (:meth:`review`). Default: 0.05.
        cross : pandas.DataFrame, optional
            The result of :meth:`cross` on these data at the same ``q``, whose flags and order enter the
            list. It is not computed here. Default: None.
        null : pandas.DataFrame, optional
            The result of :meth:`null_flags` on these data at the same ``q``. Default: None.
        k : int, optional
            The nearest data that link data to check into a group. Default: 8.
        min_support, min_gain : float, optional
            A law resting on fewer partitions than ``min_support``, or with an information gain below
            ``min_gain``, marks a place to sample more. Defaults: 0.5 and 0.
        tol : float, optional
            The distance below which two data count as co-located. Default: 0.

        Returns
        -------
        pandas.DataFrame
            One row per datum that calls for an action: ``datum``, ``value``, ``action``, ``reason``,
            ``q_value``, ``gain`` and ``group``, sorted by action, then by the q-value, or by the gain for
            the places to sample more. Each datum takes the first action that applies, in this order, with
            the readings that point to it gathered in ``reason``:

            - ``"measure again"``. *Rule*: flagged at ``q`` by the audit or by ``cross``. *Evidence*: a class
              its law makes very improbable. *Not*: a proof of error, a thin unit the other data miss being
              possible. *Action*: check the record or classify the sample again.
            - ``"check first"``. *Rule*: among the ``share`` most surprising data of the audit, or of
              ``cross``, alone among its ``k`` nearest data in the list. *Evidence*: a class among the least
              probable under its law, without reaching a flag, which a category, carrying little
              information, seldom reaches. *Not*: a suspect value as such, since about
              ``attrs["expected_clean"]`` of the listed data would be there by chance. *Action*: check
              these first, in the order of their q-value.
            - ``"check provenance"``. *Rule*: in the list to check first, linked to other listed data among
              its ``k`` nearest with the same class. *Evidence*: neighbouring data with the same surprising
              class. *Not*: necessarily errors, a small unit the other data do not represent showing the
              same pattern. *Action*: check the campaign or the logging of the group, and sample around it.
            - ``"classify one of the pair again"``. *Rule*: a datum within ``tol`` of another of a different
              class. *Evidence*: two classes at one location, which cannot both be right. *Not*: a verdict
              on which one is wrong. *Action*: classify the location again or find the original record.
            - ``"sample more nearby"``. *Rule*: no other action applies, and the law rests on a share of
              partitions below ``min_support`` or has an information gain below ``min_gain``. *Evidence*:
              the other data say little about the datum. *Not*: a suspect class. *Action*: take more
              samples there, the lowest gain first.

            ``attrs`` holds ``q``, ``share``, ``expected_clean`` (:meth:`review`), ``counts`` and, with
            ``null``, ``context`` (as in :meth:`PosteriorAudit.recommend
            <spatialize.gs.spa.PosteriorAudit.recommend>`).

        Raises
        ------
        SpatializeError
            When ``cross`` or ``null`` were read on other data or at another ``q``.

        See Also
        --------
        plot_recommendations : the list drawn on the map of the data.
        """
        from scipy.spatial import cKDTree
        data = self._fingerprint()
        for name, reading in (("cross", cross), ("null", null)):
            if reading is None:
                continue
            if reading.attrs.get("data") != data:
                raise SpatializeError(f"{name} was read on other data than this audit's")
            if not np.isclose(float(reading.attrs.get("q", np.nan)), q):
                raise SpatializeError(f"{name} was read at q = {reading.attrs.get('q')}, not at q = {q}")
        n = len(self.codes)
        t = self.table(q)
        qv, gain = t["q_value"].to_numpy(), t["gain"].to_numpy()
        rev = self.review(share)
        listed = np.zeros(n, bool)
        listed[rev["datum"].to_numpy(dtype=int)] = True
        if cross is not None:
            m = max(1, int(round(float(share) * n)))
            cp = cross["tail_p"].to_numpy()
            listed[np.argsort(np.where(np.isfinite(cp), cp, np.inf), kind="stable")[:m]] = True
        rows = {}

        def add(i, action, reason, strength, grp=None):
            if i in rows:
                rows[i]["reason"] += "; " + reason
                rows[i]["q_value"] = np.nanmin([rows[i]["q_value"], strength])
                return
            rows[i] = {"datum": int(i), "value": self.values[i], "action": action, "reason": reason,
                       "q_value": float(strength), "group": grp}

        flagged = t["flag"].to_numpy()
        for i in np.flatnonzero(flagged):
            add(i, "measure again", "a class its law makes very improbable", qv[i])
        if cross is not None:
            cq = cross["q_value"].to_numpy()
            for i in np.flatnonzero(cross["flag"].to_numpy()):
                add(i, "measure again", "a class very improbable given the second variable measured with it", cq[i])
        # the data to check first, grouped when neighbours with the same class are listed together
        pts = np.asarray(self.points, float)
        _, idx = cKDTree(pts).query(pts, k=min(int(k) + 1, n))
        idx = np.reshape(idx, (n, -1))
        cand = listed & ~flagged
        group = np.full(n, -1)
        label = 0
        for i in np.flatnonzero(cand):
            if group[i] >= 0:
                continue
            stack, group[i] = [i], label
            while stack:
                a = stack.pop()
                for b in idx[a][1:]:
                    if cand[b] and group[b] < 0 and self.codes[b] == self.codes[a]:
                        group[b] = label
                        stack.append(b)
            label += 1
        sizes = np.bincount(group[group >= 0]) if label else np.zeros(0, int)
        cross_q = cross["q_value"].to_numpy() if cross is not None else np.full(n, np.nan)
        for i in np.flatnonzero(cand):
            strength = np.nanmin([qv[i], cross_q[i]])
            g = int(group[i])
            if sizes[g] == 1:
                add(i, "check first", "among the least probable classes under its law", strength)
            else:
                add(i, "check provenance", f"one of {sizes[g]} neighbours with the same surprising class", strength, g)
        pairs = self.duplicates(tol)
        for i, j in pairs[["i", "j"]].itertuples(index=False):
            for a, b in ((int(i), int(j)), (int(j), int(i))):
                add(a, "classify one of the pair again", f"co-located with datum {b} of class {self.values[b]}", qv[a])
        ev = self.evidence()
        weak = (ev["support"].to_numpy() < min_support) | (ev["gain"].to_numpy() < min_gain)
        for i in np.flatnonzero(weak):
            if i not in rows:
                add(i, "sample more nearby",
                    f"law resting on little evidence (support {ev['support'][i]:.2f}, gain {ev['gain'][i]:.2f})", qv[i])
        order = {"measure again": 0, "check first": 1, "check provenance": 2, "classify one of the pair again": 3,
                 "sample more nearby": 4}
        out = pd.DataFrame(list(rows.values()), columns=["datum", "value", "action", "reason", "q_value", "group"])
        out.insert(5, "gain", gain[out["datum"].to_numpy(dtype=int)] if len(out) else [])
        out["_rank"] = np.where(out["action"] == "sample more nearby", out["gain"], out["q_value"])
        out = out.sort_values(["action", "_rank"], key=lambda c: c.map(order) if c.name == "action" else c,
                              na_position="last", ignore_index=True).drop(columns="_rank")
        out.attrs.update({"q": q, "share": float(share), "expected_clean": rev.attrs["expected_clean"],
                          "counts": out["action"].value_counts().to_dict(), "context": None})
        if null is not None:
            a = null.attrs
            out.attrs["context"] = {"observed": a["observed"], "expected": a["expected"], "p_value": a["p_value"],
                                    "p_value_strongest": a["p_value_strongest"],
                                    "verdict": "beyond clean data" if min(a["p_value"], a["p_value_strongest"]) <= 0.05
                                    else "an order of review"}
        return out

    def _summary(self, q=0.05):
        from spatialize import _display
        n, T = self.members.shape
        cal = self.calibration(information=True)
        rev = self.review()
        laws = [("data", n), ("members per datum", T), ("classes", len(self.classes)),
                ("ordinal", "yes" if self.ordinal else "no"), ("median support", float(np.median(self.support))),
                ("weights (members, nearest, all)", ", ".join(f"{w:.3f}" for w in self.mixture))]
        calib = [("positions, KS p-value", f"{cal['ks_p']:.3f}"), ("information check", cal["information"]["verdict"]),
                 (f"flagged (FDR {q:g})", int(self.flags(q).sum())),
                 ("to check first (5 %)", len(rev)), ("expected by chance among them", f"{rev.attrs['expected_clean']:.1f}")]
        rows = [[f"#{int(r.datum)}", str(r.value), r.most_probable, float(r.probability), float(r.tail_p)]
                for r in rev.head(8).itertuples()]
        return _display.Summary("Posterior analysis of categorical data",
                                blocks=[("kv", "Laws", laws), ("kv", "Calibration and review", calib),
                                        ("table", "Most surprising data", ["datum", "class", "most probable", "probability",
                                                                           "p-value"], rows)],
                                footer="table() · review() · calibration() · recommend() · plot_recommendations()")


@signature_overload(pivot_arg=("classifier", "knn", "classifier"),
                    common_args={"p_process": partitioning_process.MONDRIAN, "data_cond": True, "n_partitions": 200,
                                 "alpha": 0.8, "seed": random_seed, "ordinal_order": None, "local_k": 8,
                                 "base": np.e, "callback": None, "best_params_found": None},
                    specific_args={"knn_pca": {"n_neighbors": None, "max_points": None, "n_cv_splits": None},
                                   "scikit-learn": {"sklearn_classifier": None},
                                   "knn": {"n_neighbors": None},
                                   "svm": {"C": None, "kernel": None, "gamma": None, "random_state": None},
                                   "rf": {"n_estimators": None, "max_depth": None, "random_state": None},
                                   "dt": {"max_depth": None, "min_samples_split": None, "random_state": None}})
def cat_posterior_audit(points, values, **kwargs):
    """The law of the classes at each datum built from the other data.

    Each datum is predicted from the other data by leave-one-out cross-validation, with the partitions and
    the classifier of categorical ensemble spatial interpolation, one member per partition. The members,
    the classes of the nearest other data and those of all the data give the law of each datum
    (:class:`CategoricalPosteriorAudit`).

    Parameters
    ----------
    points : array_like of shape (n, d)
        The data locations.
    values : array_like of shape (n,)
        The data classes.
    classifier : str, optional
        The classifier of the cells, as in :func:`~spatialize.gs.cat_esi.cat_esi_nongriddata`, with its
        parameters as keyword arguments. Default: ``"knn"``.
    p_process : {"mondrian", "mondrian-legacy", "voronoi"}, optional
        The partition process. Default: ``"mondrian"``.
    data_cond : bool, optional
        Voronoi only: nuclei among the data (True) or uniform in the box. Default: True.
    n_partitions : int, optional
        The number of partitions, the members of each law. Default: 200.
    alpha : float, optional
        The granularity of the partitions. Default: 0.8.
    seed : int, optional
        Seed of the partitions. Default: drawn at random.
    ordinal_order : list, optional
        The classes in their order, for ordinal data. Default: None (nominal classes).
    local_k : int, optional
        The nearest other data whose class shares enter each law. Default: 8.
    base : float, optional
        The base of the logarithms of the information readings. Default: e.
    best_params_found : dict, optional
        The output of a search's ``best_result()``; its keys override the arguments, except
        ``n_partitions``. Default: None.
    callback : callable, optional
        Where the progress and the messages go (:mod:`spatialize.logging`). Default: ``None``.

    Returns
    -------
    CategoricalPosteriorAudit
    """
    best = kwargs.get("best_params_found")
    if best is not None:
        kwargs = dict(kwargs)
        kwargs.update({k: v for k, v in best.items() if k in kwargs and k != "n_partitions"})
    kwargs["callback"] = resolve_callback(kwargs["callback"])
    points = np.asarray(points, dtype=np.float64)
    classes, codes = _encode(values, kwargs["ordinal_order"])
    members = _cat_members(points, codes, kwargs)
    return CategoricalPosteriorAudit(members, points, values, classes, ordinal=kwargs["ordinal_order"] is not None,
                                     local_k=kwargs["local_k"], seed=kwargs["seed"], base=kwargs["base"],
                                     callback=kwargs["callback"], cells=_cells_reader(points, kwargs),
                                     refit=lambda c: _cat_members(points, np.asarray(c, int), kwargs))

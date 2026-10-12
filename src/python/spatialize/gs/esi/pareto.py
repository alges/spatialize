"""Pareto hyperparameter optimisation for ESI.

For each parameter configuration this module jointly evaluates two objectives:

    ε̂   — Encoder error: the empirical robustness level, the largest divergence between the
            laws of two data sharing a cell, per partition, averaged over the partitions; the laws
            are fitted once for the search (by default from each datum's nearest neighbours).
    R_CV — Decoder error: standard k-fold / LOO cross-validation score.

The Pareto frontier over (ε̂, R_CV) is returned as an :class:`ESIParetoResult`,
which provides several strategies for selecting a single "best" configuration
when more than one frontier point exists.

Public API
----------
ESIParetoResult
    Result object, analogous to :class:`~spatialize.gs.esi.ESIGridSearchResult`.
ParetoOptimizer
    ESI Pareto optimiser.
esi_pareto_optimization
    Convenience function with ``@signature_overload`` defaults for ESI.
"""

from __future__ import annotations

import numpy as np
from itertools import permutations, product

import spatialize.gs.esi.scorefunction as sf
from spatialize.gs import lib_spatialize_facade, partitioning_process, local_interpolator as li, with_more_decoders
from spatialize.gs.esi._main import build_arg_list
from spatialize._parallel import map_chunks
from spatialize.empirical import FittedModelFactory
from spatialize.logging import singleton_null_callback, log_message
from spatialize import logging


# ---------------------------------------------------------------------------
# Scoring map (string → callable)
# ---------------------------------------------------------------------------

_COMMON_GRID_KEYS = {"n_partitions", "alpha"}

# Interpolator-specific parameter defaults (mirrors esi/_main.py)
_INTERP_DEFAULTS = with_more_decoders({
    li.IDW:          {"exponent": 2.0},
    li.KRIGING:      {"model": "spherical", "nugget": 0.1, "range": 5000.0, "sill": 1.0},
    li.ADAPTIVE_IDW: {"metric": "mae"},
})

# ===========================================================================
# EmpiricalRobustnessBound  (the empirical robustness level, per partition)
# ===========================================================================

def reference_laws(points, values, laws=None, n_neighbours=16, min_spread=0.05, point_model_name="kde",
                   nan_model_name="ignore", seed=None) -> list:
    """The estimated conditional law at each datum, fitted once and shared by every configuration.

    The empirical robustness level compares laws that do not depend on the partitions it judges.
    By default the law at a datum is that of the values of its ``n_neighbours`` nearest data, itself
    excluded, a local predictor whose law narrows to the location's as the data densify.

    Parameters
    ----------
    points, values : array-like
        The data.
    laws : None, array-like of shape (n, m) or callable, optional
        ``None`` for the nearest neighbours; a matrix whose row ``i`` holds ``m`` draws from the law
        at datum ``i`` (NaN for none); or a function ``(points, values) -> matrix``.
    n_neighbours : int, optional
        Neighbours of the default law (16).
    min_spread : float, optional
        The smallest standard deviation a law may have, as a fraction of the standard deviation of
        the data (0.05). Where the draws are all alike (a flat surface, a zero atom) the law would
        be a point mass, whose divergence from any other law is unbounded, while the bound on the
        encoder error assumes densities bounded above and below. The floor keeps every law a density
        at the scale of the data, so it does not depend on the units.
    point_model_name, nan_model_name, seed
        The density fitted to each row (:class:`~spatialize.empirical.FittedModelFactory`).

    Returns
    -------
    list
        Per datum, ``None`` when fewer than two draws remain, else ``(model, centre, low, high,
        scale)``: the fitted density, the mean of the draws, the interval the divergence integrates
        over and the standard deviation of the law. With ``"kde"`` the density keeps the mean of the
        draws and their variance, raised to the floor if smaller, its tails falling at the law's own
        scale.
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if laws is None:
        from scipy.spatial import cKDTree
        k = min(int(n_neighbours), len(values) - 1)
        _, nb = cKDTree(points).query(points, k=k + 1)
        draws = values[nb[:, 1:]]
    elif callable(laws):
        draws = np.asarray(laws(points, values), dtype=np.float64)
    else:
        draws = np.asarray(laws, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[0] != len(values):
        raise ValueError(f"the reference laws must be a matrix with one row per datum, got shape {draws.shape}")
    factory = FittedModelFactory(nan_model_name=nan_model_name, point_model_name=point_model_name, seed=seed)
    finite = values[np.isfinite(values)]
    floor = float(min_spread) * float(np.std(finite)) if len(finite) > 1 else 0.0
    result = []
    for row in draws:
        s = row[~np.isnan(row)]
        if len(s) < 2:
            result.append(None)
            continue
        centre = float(s.mean())
        own = float(np.std(s))
        sd = max(own, floor) or 1e-6 * max(1.0, abs(centre))
        if point_model_name == "kde":
            model = _variance_preserving_kde(s, centre, own, sd)
        else:
            model, _ = factory.create(s)
        result.append((model, centre, float(s.min()) - 6.0 * sd, float(s.max()) + 6.0 * sd, sd))
    return result


def _variance_preserving_kde(sample, centre, own, sd):
    """A Gaussian KDE with the sample's mean and the variance ``sd**2``: the draws pulled towards
    their mean by 1/√2, whose variance is then ``own**2 / 2``, and a bandwidth adding the rest. With
    ``sd == own`` the bandwidth is sd/√2, so the kernels add back the variance the pull removed; a
    larger ``sd`` (a floor) widens the kernels. Its tails fall at the scale of the law itself; the
    narrow kernels of a density-optimal bandwidth would make a divergence read their tails instead
    (two laws N(0, 1) and N(8, 1) would diverge by hundreds of nats where the laws diverge by 32).
    For a Gaussian sample the fitted law is the Gaussian."""
    from sklearn.neighbors import KernelDensity
    pulled = centre + (np.asarray(sample, dtype=np.float64) - centre) / np.sqrt(2.0)
    bandwidth = np.sqrt(max(sd ** 2 - own ** 2 / 2.0, (sd ** 2) / 2.0))
    return KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(pulled.reshape(-1, 1))


class EmpiricalRobustnessBound:
    """The empirical robustness level of an ensemble of partitions.

    For one partition, the largest divergence between the estimated laws at two data sharing a cell,

        ε̂_t = max_{u}  max_{i,j: η_t(x_i)=η_t(x_j)=u}  D_KL(p̂_i ‖ p̂_j),

    and for the ensemble the mean over its partitions, ε̂ = (1/T) Σ_t ε̂_t, the expected level of the
    random partition. The laws p̂_i are given (:func:`reference_laws`), the same for every
    configuration, so ε̂ reads the cells alone.

    Parameters
    ----------
    n_partitions, alpha, p_process, seed
        The partitions, those an ESI call with the same arguments draws.
    pair_strategy : str
        The pairs compared in each cell. ``"gaussian"`` (default), the pair whose laws, read as
        Gaussians of their mean and standard deviation, diverge the most; ``"exhaustive"``, every
        ordered pair, exact but quadratic in the data of a cell; ``"max_min"``, the data of largest
        and smallest law mean, both ways, blind to laws that differ in spread.
    support_sample_size : int
        Points of the grid each divergence integrates over.
    data_cond : bool
        For Voronoi, whether the nuclei are drawn among the data (default) or uniformly.
    """

    def __init__(
        self,
        n_partitions: int,
        alpha: float,
        p_process: str,
        seed: int,
        pair_strategy: str = "gaussian",
        support_sample_size: int = 500,
        data_cond: bool = True,
    ) -> None:
        self.n_partitions        = n_partitions
        self.alpha               = alpha
        self.p_process           = p_process
        self.data_cond           = data_cond
        self.seed                = seed
        self.pair_strategy       = pair_strategy
        self.support_sample_size = support_sample_size
        self.repeated            = False   # set by ParetoOptimizer: one of several estimates in a row

        # Populated by estimate()
        self.leaf_indexes:  np.ndarray | None = None
        self.per_partition: np.ndarray | None = None
        self.epsilon_hat:   float | None      = None

    def estimate(self, points: np.ndarray, laws: list) -> float:
        """ε̂ of these partitions for the given laws (one entry per datum, see :func:`reference_laws`)."""
        points = np.asarray(points, dtype=np.float32)
        alpha = -self.alpha if self.p_process == partitioning_process.VORONOI and not self.data_cond else self.alpha
        self.leaf_indexes = lib_spatialize_facade.cells(points, points, self.p_process, alpha,
                                                        self.n_partitions, self.seed)   # (n, T)

        # the pairs of each partition, and each distinct pair once
        centres = np.array([np.nan if law is None else law[1] for law in laws])
        scales = np.array([np.nan if law is None else law[4] for law in laws])
        by_partition, index = [], {}
        for t in range(self.n_partitions):
            cells = self.leaf_indexes[:, t]
            mine = []
            for k in np.unique(cells[cells >= 0]):
                members = np.where((cells == k) & ~np.isnan(centres))[0]
                if len(members) < 2:
                    continue
                for pair in self._select_pairs(members, centres, scales):
                    mine.append(index.setdefault(pair, len(index)))
            by_partition.append(mine)

        # the divergences take almost all the time and are independent, so they run in parallel under
        # the session settings; each is computed alone, so the result does not depend on the workers
        pairs = list(index)
        d = np.array(_divergences(laws, pairs, self.support_sample_size, self.repeated))
        self.per_partition = np.array([d[mine].max() if mine else 0.0 for mine in by_partition])
        self.epsilon_hat = float(self.per_partition.mean())
        return self.epsilon_hat

    def _select_pairs(self, members, centres, scales) -> list[tuple[int, int]]:
        """Ordered (i, j) pairs to evaluate in a cell, both directions since D_KL is asymmetric."""
        if self.pair_strategy == "exhaustive":
            return list(permutations(members.tolist(), 2))
        if self.pair_strategy == "gaussian":
            # the pair whose laws, read as Gaussians of their mean and spread, diverge the most
            m, sd = centres[members], scales[members]
            g = (np.log(sd[None, :] / sd[:, None])
                 + (sd[:, None] ** 2 + (m[:, None] - m[None, :]) ** 2) / (2.0 * sd[None, :] ** 2) - 0.5)
            i, j = np.unravel_index(int(np.argmax(g)), g.shape)
            return [(int(members[i]), int(members[j]))] if i != j else []
        i_max = int(members[int(np.argmax(centres[members]))])
        i_min = int(members[int(np.argmin(centres[members]))])
        if i_max == i_min:
            return []
        return [(i_max, i_min), (i_min, i_max)]


# ===========================================================================
# ParetoOptimizer
# ===========================================================================

class ParetoOptimizer:
    """Pareto hyperparameter optimiser for ESI.

    For each configuration in ``param_grid``:

    1. Computes ε̂ via :class:`EmpiricalRobustnessBound`, on laws fitted once
       for the whole search (:func:`reference_laws`).
    2. Runs LOO / k-fold cross-validation to obtain R_CV.

    Parameters
    ----------
    param_grid : dict
        ``{param_name: [values]}`` mapping.  Must contain ``"n_partitions"``
        and ``"alpha"``; interpolator-specific keys (e.g. ``"exponent"``) are
        forwarded to both the robustness bound and the CV call.
    local_interpolator : str
        ``"idw"`` (default), ``"kriging"``, or ``"adaptiveidw"``.
    p_process : str
        Partitioning process used for both the encoder (robustness bound) and
        decoder (CV) steps.
    data_cond : bool
        For Voronoi, whether the nuclei are drawn among the data (default) or uniformly.
    scoring : callable
        Decoder scoring function with signature
        ``(true_values, esi_samples) → float``.
        Default: :func:`~spatialize.gs.esi.scorefunction.neg_log_likelihood`.
    k : int or str
        CV folds.  ``"loo"`` or ``-1`` triggers leave-one-out.
    seed : int
        Random seed shared between encoder and decoder calls.
    folding_seed : int
        Secondary seed for k-fold fold assignment.
    pair_strategy : str
        Pairs compared in each cell (:class:`EmpiricalRobustnessBound`):
        ``"gaussian"`` (default), ``"exhaustive"`` or ``"max_min"``.
    reference_laws : None, array-like or callable
        The estimated law at each datum (:func:`reference_laws`); ``None`` for
        its ``n_neighbours`` nearest data.
    n_neighbours : int
        Neighbours of the default reference laws (16).
    min_spread : float
        Smallest standard deviation of a reference law, as a fraction of that of the data (0.05).
    point_model_name : str
        Density model fitted to each reference law: ``"kde"`` (default),
        ``"emm"``, ``"vim"``.
    nan_model_name : str
        NaN strategy for :class:`FittedModelFactory`: ``"ignore"``
        (default) or ``"replace"``.
    support_sample_size : int
        Points of the grid each divergence integrates over (default 500).
    callback : callable, optional
        Where the progress and the messages go. Default: ``None``, shown as the session settings
        ``display``, ``progress`` and ``verbosity`` say (:mod:`spatialize.session`). A callable
        receiving the messages of :mod:`spatialize.logging` sends them elsewhere, such as an
        application's own interface, while :func:`~spatialize.logging.singleton_null_callback`
        drops them.
    """

    def __init__(
        self,
        param_grid: dict,
        local_interpolator: str = "idw",
        p_process: str = partitioning_process.MONDRIAN,
        data_cond: bool       = True,
        scoring               = sf.neg_log_likelihood,
        k                     = 5,
        seed: int             = 0,
        folding_seed: int     = 0,
        pair_strategy: str    = "gaussian",
        reference_laws        = None,
        n_neighbours: int     = 16,
        min_spread: float     = 0.05,
        point_model_name: str = "kde",
        nan_model_name: str   = "ignore",
        support_sample_size: int = 500,
        fixed_interp_kwargs: dict | None = None,
        callback              = None,
    ) -> None:
        self.param_grid          = param_grid
        self.local_interpolator  = local_interpolator
        self.p_process           = p_process
        self.data_cond           = data_cond
        self.scoring             = scoring
        self.k                   = k
        self.seed                = seed
        self.folding_seed        = folding_seed
        self.pair_strategy       = pair_strategy
        self.reference_laws      = reference_laws
        self.n_neighbours        = n_neighbours
        self.min_spread          = min_spread
        self.point_model_name    = point_model_name
        self.nan_model_name      = nan_model_name
        self.support_sample_size = support_sample_size
        self.fixed_interp_kwargs = fixed_interp_kwargs or {}
        self.callback            = callback or singleton_null_callback

    # ------------------------------------------------------------------

    def fit(self, points: np.ndarray, values: np.ndarray) -> list:
        """Run the grid search and return an :class:`ESIParetoResult`.

        Parameters
        ----------
        points : array-like, shape (n, d)
        values : array-like, shape (n,)

        Returns
        -------
        list of dict
            One entry per evaluated configuration with keys ``"params"``,
            ``"epsilon"`` (ε̂), ``"decoder_error"`` (R_CV), ``"left_out"`` (the
            share of the data left out of R_CV) and ``"nan_members"`` (the
            share of NaN cross-validation members).
            Wrap with :class:`~spatialize.gs.esi.ESIParetoResult` to access
            the Pareto frontier and selection strategies.
        """
        points = np.asarray(points, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)

        scoring_fn = self.scoring

        n = len(points)
        k_val  = n if self.k in ("loo", -1) else int(self.k)
        method = "loo" if k_val == n else "kfold"

        cv_fn = lib_spatialize_facade.get_operator(points, self.local_interpolator, method, self.p_process)

        # the laws the encoder error compares, the same for every configuration
        laws = reference_laws(points, values, self.reference_laws, self.n_neighbours, self.min_spread,
                              self.point_model_name, self.nan_model_name, self.seed)

        configs   = list(_iter_param_grid(self.param_grid))
        n_configs = len(configs)
        self.callback(logging.progress.init(n_configs, 1, desc="Pareto search · encoder and decoder errors"))

        all_results: list[dict] = []
        for cfg in configs:
            n_partitions = cfg["n_partitions"]
            alpha        = cfg["alpha"]
            interp_kw    = {k: v for k, v in cfg.items()
                            if k not in _COMMON_GRID_KEYS}

            log_message(logging.logger.debug(
                f"[ParetoOptimizer] evaluating config: {cfg}"
            ))

            # ── 1. Encoder error ε̂: the cells of these partitions, read on the fixed laws ──
            erb = EmpiricalRobustnessBound(
                n_partitions        = n_partitions,
                alpha               = alpha,
                p_process           = self.p_process,
                data_cond           = self.data_cond,
                seed                = self.seed,
                pair_strategy       = self.pair_strategy,
                support_sample_size = self.support_sample_size,
            )
            erb.repeated = n_configs > 1
            eps_hat = erb.estimate(points, laws)

            # ── 2. Decoder error R_CV ────────────────────────────────────
            full_params = self._build_full_params(n_partitions, alpha, interp_kw)
            l_args = build_arg_list(points, values, points, full_params)
            if method == "kfold":
                l_args.insert(-2, k_val)
                l_args.insert(-2, self.folding_seed)
            _, cv_samples = cv_fn(*l_args)

            r_cv = float(scoring_fn(values, cv_samples))
            out, nan = sf.left_out(cv_samples, scoring_fn)

            all_results.append({
                "params":        cfg,
                "epsilon":       eps_hat,
                "decoder_error": r_cv,
                "left_out":      out,
                "nan_members":   nan,
            })

            self.callback(logging.progress.inform())

        self.callback(logging.progress.stop())
        return all_results

    # ------------------------------------------------------------------

    def _build_full_params(
        self,
        n_partitions: int,
        alpha: float,
        interp_kw: dict,
    ) -> dict:
        """Construct the params dict for ``build_arg_list``."""
        defaults = dict(_INTERP_DEFAULTS.get(self.local_interpolator, {}))
        defaults.update(self.fixed_interp_kwargs)
        defaults.update(interp_kw)
        return {
            "n_partitions":       n_partitions,
            "alpha":              alpha,
            "local_interpolator": self.local_interpolator,
            "p_process":          self.p_process,
            "data_cond":          self.data_cond,
            "seed":               self.seed,
            "callback":           singleton_null_callback,
            **defaults,
        }


# ===========================================================================
# Private helpers
# ===========================================================================

def _divergences(laws, pairs, support_sample_size, repeated=False) -> list:
    """D_KL(p̂_i ‖ p̂_j) for each pair, in order, on worker processes when the work repays them
    (:func:`spatialize._parallel.map_chunks`); each worker receives only the laws of its pairs."""
    def chunk_data(rows):
        chunk = [pairs[r] for r in rows]
        return chunk, {k: laws[k] for pair in chunk for k in pair}

    def divergences(rows, data):
        chunk, chunk_laws = data
        return [_kl_from_models(chunk_laws[i], chunk_laws[j], support_sample_size) for i, j in chunk]

    return list(map_chunks(divergences, len(pairs), data_for=chunk_data, repeated=repeated))


def _log_density(model, x):
    """The log-density of a fitted model at the points ``x`` (1-D). A Gaussian KDE is evaluated
    exactly from its data and bandwidth, in log space, so a far tail is a finite number and never
    an underflow to zero."""
    from scipy.special import logsumexp
    from sklearn.neighbors import KernelDensity
    if isinstance(model, KernelDensity) and model.kernel == "gaussian":
        data = np.asarray(model.tree_.data).ravel()
        h = float(model.bandwidth_)
        z = (x[:, None] - data[None, :]) / h
        return logsumexp(-0.5 * z * z, axis=1) - np.log(len(data) * h * np.sqrt(2.0 * np.pi))
    return np.asarray(model.score_samples(x.reshape(-1, 1)), dtype=np.float64).ravel()


def _kl_from_models(law_i: tuple, law_j: tuple, support_sample_size: int = 500) -> float:
    """D_KL(p̂_i ‖ p̂_j) between two laws of :func:`reference_laws`.

    The integral runs over the interval that holds p̂_i, on ``support_sample_size`` points, with
    both log-densities exact. Where p̂_j is far, its log-density is large and negative but finite,
    so the divergence of two laws that barely overlap is large and finite, with no floor and no
    ceiling, and it does not depend on the units of the variable.
    """
    model_i, _, low, high, _ = law_i
    model_j = law_j[0]
    if high <= low:
        return 0.0
    x, dx = np.linspace(low, high, support_sample_size, retstep=True)
    log_i = _log_density(model_i, x)
    log_j = _log_density(model_j, x)
    w = np.exp(log_i - log_i.max())
    w /= w.sum()                      # p̂_i on the grid, normalised to its mass there
    log_norm_i = log_i.max() + np.log(np.exp(log_i - log_i.max()).sum() * dx)
    return max(float(np.sum(w * ((log_i - log_norm_i) - log_j))), 0.0)


def _iter_param_grid(param_grid: dict):
    """Yield all combinations from a ``{key: [values]}`` dict."""
    keys = list(param_grid.keys())
    for combo in product(*[param_grid[k] for k in keys]):
        yield dict(zip(keys, combo))

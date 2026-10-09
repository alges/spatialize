"""Pareto hyperparameter optimisation for ESI.

For each parameter configuration this module jointly evaluates two objectives:

    ε̂   — Encoder error: empirical robustness bound,
            estimated via fitted continuous density models (KDE / GMM).
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
# EmpiricalRobustnessBound  (Algorithm 1 with fitted KL)
# ===========================================================================

class EmpiricalRobustnessBound:
    """Algorithm 1 using fitted continuous density models for KL estimation.

    Computes:

        ε̂ = max_{u ∈ U}  max_{i,j: η(x_i)=η(x_j)=u}  D_KL(p̂_i ‖ p̂_j)

    where each p̂_i is represented by the LOO ESI sample vector at x_i and
    KL is estimated by fitting independent KDE / GMM models to each sample
    vector and integrating over a shared grid.

    Parameters
    ----------
    n_partitions, alpha, local_interpolator, p_process, seed
        ESI encoder parameters. ε̂ is computed on the partitions of ``p_process``.
    data_cond : bool
        For Voronoi, whether the nuclei are drawn among the data (default) or uniformly.
    pair_strategy : str
        ``"max_min"`` (default) or ``"exhaustive"``.
    point_model_name : str
        ``"kde"`` (default), ``"emm"``, or ``"vim"``.
    nan_model_name : str
        ``"ignore"`` (default) or ``"replace"``.
    support_sample_size : int
        Number of grid points for the shared KL evaluation grid.
    """

    def __init__(
        self,
        n_partitions: int,
        alpha: float,
        local_interpolator: str,
        p_process: str,
        seed: int,
        pair_strategy: str = "max_min",
        point_model_name: str = "kde",
        nan_model_name: str = "ignore",
        support_sample_size: int = 500,
        interp_kwargs: dict | None = None,
        data_cond: bool = True,
    ) -> None:
        self.n_partitions       = n_partitions
        self.alpha              = alpha
        self.local_interpolator = local_interpolator
        self.p_process          = p_process
        self.data_cond          = data_cond
        self.seed               = seed
        self.pair_strategy      = pair_strategy
        self.support_sample_size = support_sample_size
        self.interp_kwargs      = interp_kwargs or {}
        self.repeated           = False   # set by ParetoOptimizer: one of several estimates in a row

        self._factory = FittedModelFactory(
            nan_model_name=nan_model_name,
            point_model_name=point_model_name,
            seed=seed,
        )

        # Populated by estimate()
        self.esi_samples:  np.ndarray | None = None
        self.leaf_indexes: np.ndarray | None = None
        self.epsilon_hat:  float | None      = None

    def estimate(self, points: np.ndarray, values: np.ndarray) -> float:
        """Run Algorithm 1 and return ε̂."""
        points = np.asarray(points, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)

        # Step 1 — LOO predictive distributions as ESI sample vectors.
        loo_fn = lib_spatialize_facade.get_operator(
            points, self.local_interpolator, "loo", self.p_process
        )
        params = self._build_params()
        l_args = build_arg_list(points, values, points, params)
        _, self.esi_samples = loo_fn(*l_args)          # shape (n, T)

        # Step 2 — the cell of each datum in each partition of step 1 (same partition, alpha as
        # passed to the engine, size and seed; the facade applies the session domain)
        self.leaf_indexes = lib_spatialize_facade.cells(
            points, points, self.p_process, l_args[3], self.n_partitions, self.seed
        )  # shape (n, T)

        # Step 3 — pre-fit one density model per training point (fit once, reuse).
        # Each point's KDE is fitted exactly once here; the inner loop then calls
        # _kl_from_models() which uses score_samples() directly, skipping the
        # EmpiricalModel Akima-interpolator layer and all redundant refitting.
        models = self._prefit_models(self.esi_samples)

        # Step 4 — the pairs that share a cell in some partition, each once
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for t in range(self.n_partitions):
            cells = self.leaf_indexes[:, t]
            for k in np.unique(cells):
                members = np.where(cells == k)[0]
                if len(members) < 2:
                    continue
                for (i, j) in self._select_pairs(members):
                    if (i, j) in seen:
                        continue
                    seen.add((i, j))
                    if models[i] is None or models[j] is None:
                        continue
                    pairs.append((i, j))

        # Step 5 — the largest divergence among them. The divergences take almost all the time and
        # are independent, so they run in parallel under the session settings; a maximum does not
        # depend on the order, so the result is the same with any number of workers.
        self.epsilon_hat = _max_divergence(models, pairs, self.support_sample_size, self.repeated)
        return self.epsilon_hat

    # ------------------------------------------------------------------

    def _build_params(self) -> dict:
        """Build the params dict for ``build_arg_list``."""
        defaults = dict(_INTERP_DEFAULTS.get(self.local_interpolator, {}))
        defaults.update(self.interp_kwargs)
        return {
            "n_partitions":       self.n_partitions,
            "alpha":              self.alpha,
            "local_interpolator": self.local_interpolator,
            "p_process":          self.p_process,
            "data_cond":          self.data_cond,
            "seed":               self.seed,
            "callback":           singleton_null_callback,
            **defaults,
        }

    def _select_pairs(self, members: np.ndarray) -> list[tuple[int, int]]:
        """Return ordered (i, j) pairs to evaluate for a cell.

        Both directions are returned because D_KL is asymmetric.
        """
        if self.pair_strategy == "exhaustive":
            return list(permutations(members.tolist(), 2))

        # max_min: pair the point with highest mean against lowest mean
        means = np.nanmean(self.esi_samples[members], axis=1)
        i_max = members[int(np.argmax(means))]
        i_min = members[int(np.argmin(means))]
        if i_max == i_min:
            return []
        return [(i_max, i_min), (i_min, i_max)]

    def _prefit_models(self, samples: np.ndarray) -> list:
        """Fit one density model per training point; return list of (model, d_min, d_max) or None.

        The factory returns an independent estimator per point, so the list can
        be safely reused across all KL pair evaluations.
        """
        result = []
        for idx in range(len(samples)):
            s = samples[idx].astype(np.float64)
            s = s[~np.isnan(s)]
            if len(s) < 2:
                result.append(None)
                continue
            model, _ = self._factory.create(s)   # a fresh model, its bandwidth from this sample
            d_min, d_max = float(s.min()), float(s.max())
            ext = (d_max - d_min) * 0.1
            result.append((model, d_min - ext, d_max + ext))
        return result


# ===========================================================================
# ParetoOptimizer
# ===========================================================================

class ParetoOptimizer:
    """Pareto hyperparameter optimiser for ESI.

    For each configuration in ``param_grid``:

    1. Computes ε̂ via :class:`EmpiricalRobustnessBound`.
    2. Runs LOO / k-fold cross-validation to obtain R_CV.

    When the CV method is LOO, the ESI samples produced by the encoder are
    reused for decoder scoring — no redundant C++ call is made.

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
        Pair-selection strategy for the robustness bound: ``"max_min"``
        (default) or ``"exhaustive"``.
    point_model_name : str
        Density model for fitted KL: ``"kde"`` (default), ``"emm"``,
        ``"vim"``.
    nan_model_name : str
        NaN strategy for :class:`FittedModelFactory`: ``"ignore"``
        (default) or ``"replace"``.
    support_sample_size : int
        Grid resolution for the shared KL evaluation grid (default 500).
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
        pair_strategy: str    = "max_min",
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

        # Only needed for k-fold; LOO samples come from the encoder for free.
        cv_fn = (
            None if method == "loo"
            else lib_spatialize_facade.get_operator(
                points, self.local_interpolator, "kfold", self.p_process
            )
        )

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

            # ── 1. Encoder error ε̂ ──────────────────────────────────────
            # Merge: fixed params first, per-config grid params override.
            merged_interp_kw = {**self.fixed_interp_kwargs, **interp_kw}
            erb = EmpiricalRobustnessBound(
                n_partitions        = n_partitions,
                alpha               = alpha,
                local_interpolator  = self.local_interpolator,
                p_process           = self.p_process,
                data_cond           = self.data_cond,
                seed                = self.seed,
                pair_strategy       = self.pair_strategy,
                point_model_name    = self.point_model_name,
                nan_model_name      = self.nan_model_name,
                support_sample_size = self.support_sample_size,
                interp_kwargs       = merged_interp_kw,
            )
            erb.repeated = n_configs > 1
            eps_hat = erb.estimate(points, values)

            # ── 2. Decoder error R_CV ────────────────────────────────────
            # LOO: encoder already produced LOO ESI samples — reuse directly.
            # k-fold: encoder uses LOO (for ε̂), decoder uses k-fold separately.
            if method == "loo":
                cv_samples = erb.esi_samples
            else:
                full_params = self._build_full_params(n_partitions, alpha, interp_kw)
                l_args = build_arg_list(points, values, points, full_params)
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

def _max_divergence(models, pairs, support_sample_size, repeated=False) -> float:
    """The largest D_KL(p̂_i ‖ p̂_j) over ``pairs``, on worker processes when the work repays them
    (:func:`spatialize._parallel.map_chunks`); each worker receives only the models of its pairs."""
    def chunk_data(rows):
        chunk = [pairs[r] for r in rows]
        return chunk, {k: models[k] for pair in chunk for k in pair}

    def divergences(rows, data):
        chunk, chunk_models = data
        return [_kl_from_models(chunk_models[i], chunk_models[j], support_sample_size) for i, j in chunk]

    best = 0.0
    for d in map_chunks(divergences, len(pairs), data_for=chunk_data, repeated=repeated):
        if d > best:
            best = d
    return best


def _density(model, x):
    """The density of a fitted model at the points ``x`` (1-D). A Gaussian KDE is evaluated exactly
    and vectorised from its data and bandwidth; scikit-learn's tree only pays off when it
    approximates, which would make the divergence depend on the units of the variable."""
    from sklearn.neighbors import KernelDensity
    if isinstance(model, KernelDensity) and model.kernel == "gaussian":
        data = np.asarray(model.tree_.data).ravel()
        h = float(model.bandwidth_)
        z = (x[:, None] - data[None, :]) / h
        return np.exp(-0.5 * z * z).sum(axis=1) / (len(data) * h * np.sqrt(2.0 * np.pi))
    return np.exp(model.score_samples(x.reshape(-1, 1))).ravel()


def _kl_from_models(
    model_tuple_i: tuple,
    model_tuple_j: tuple,
    support_sample_size: int = 500,
) -> float:
    """D_KL(p̂_i ‖ p̂_j) from two pre-fitted sklearn density models.

    Uses ``score_samples`` directly on the shared evaluation grid, bypassing
    the :class:`EmpiricalModel` Akima-interpolator layer and avoiding redundant
    model fitting when the same model is reused across multiple pairs.
    Called by :meth:`EmpiricalRobustnessBound.estimate` via the model cache
    built by :meth:`EmpiricalRobustnessBound._prefit_models`.
    """
    model_i, d_min_i, d_max_i = model_tuple_i
    model_j, d_min_j, d_max_j = model_tuple_j

    x_min = min(d_min_i, d_min_j)
    x_max = max(d_max_i, d_max_j)
    if x_max <= x_min:
        return 0.0

    x_shared, dx = np.linspace(x_min, x_max, support_sample_size, retstep=True)

    p_i = np.nan_to_num(_density(model_i, x_shared), nan=0.0).clip(0.0)
    p_j = np.nan_to_num(_density(model_j, x_shared), nan=0.0).clip(0.0)

    sum_i = p_i.sum() * dx
    sum_j = p_j.sum() * dx
    if sum_i == 0.0 or sum_j == 0.0:
        return 0.0

    p_i /= sum_i
    p_j /= sum_j

    mask = p_i > 0.0
    # a floor for p̂_j: the density of a uniform law on the grid carrying 1/(10 N) of the mass, a
    # density (1/length), so the divergence does not depend on the units of the variable
    p_j_floor = 1.0 / ((x_max - x_min) * support_sample_size * 10.0)
    p_j_safe = np.maximum(p_j[mask], p_j_floor)
    return max(float(np.sum(p_i[mask] * (np.log(p_i[mask]) - np.log(p_j_safe))) * dx), 0.0)


def _iter_param_grid(param_grid: dict):
    """Yield all combinations from a ``{key: [values]}`` dict."""
    keys = list(param_grid.keys())
    for combo in product(*[param_grid[k] for k in keys]):
        yield dict(zip(keys, combo))

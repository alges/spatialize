import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import ParameterGrid, KFold

import libspatialize as lsp
from spatialize import EstimationResult, GridSearchResult, SpatializeError
from spatialize._util import signature_overload
from spatialize._math_util import flatten_grid_data
from spatialize.logging import default_singleton_callback, singleton_null_callback
from spatialize import logging
from spatialize.logging import log_message

from .agg_functions import aggregate_with_mv, aggregate_with_ordinal_mv, categorical_feature_precision
from .classifiers import get_classifier_fns, SKLEARN_CLASSIFIER_PARAMS, SKLEARN_RANDOM_STATE_CLASSIFIERS
from .score_functions import resolve_scoring, SCORING_OPTIONS


# ─────────────────────────────────────────────────────────────
#  Aggregation resolution
# ─────────────────────────────────────────────────────────────

def _resolve_agg_fn(agg_function, ordinal_order=None):
    """
    Resolve *agg_function* to a bound ``callable(esi_samples) → estimation``.

    Accepts a string shorthand or any callable:
      - ``'mv'``  : majority vote; automatically selects ordinal variant when
                    *ordinal_order* is provided.
      - callable  : used as-is (backward-compatible).

    ``'btd'`` is only supported via ``CatESIResult.re_estimate`` because it
    needs metadata from the result object (categories, grid shape, etc.).
    """
    if callable(agg_function):
        return agg_function
    if agg_function == 'mv':
        if ordinal_order is not None:
            return lambda s: aggregate_with_ordinal_mv(s, ordinal_order)
        return aggregate_with_mv
    raise ValueError(
        f"Unknown agg_function '{agg_function}'. "
        "Use 'mv', a callable, or 'btd' via result.re_estimate()."
    )


# ─────────────────────────────────────────────────────────────
#  Value encoding helpers
# ─────────────────────────────────────────────────────────────

def _make_encoder(values):
    """
    Map arbitrary categorical labels to contiguous integer codes.

    Category identity is determined by ``str(v)`` (so ``1`` and ``"1"`` are
    the same category), but the *original* value is preserved in
    ``code_to_cat`` so that decoding returns the same type as the input.

    Returns
    -------
    encoded : np.ndarray, float64
    cat_to_code : dict  str(v) -> float code
    code_to_cat : dict  float code -> original value
    """
    seen = {}  # str(v) -> original value, first-seen wins
    for v in values:
        key = str(v)
        if key not in seen:
            seen[key] = v
    cat_to_code = {key: float(i) for i, key in enumerate(seen)}
    code_to_cat = {float(i): orig for i, orig in enumerate(seen.values())}
    encoded = np.array([cat_to_code[str(v)] for v in values], dtype=np.float64)
    return encoded, cat_to_code, code_to_cat


def _decode_samples(esi_samples, code_to_cat):
    """
    Convert integer-coded esi_samples (p × n) back to original labels.

    Vectorised: builds a flat lookup array then applies it in one numpy call
    instead of iterating over all (p × n) entries in Python.
    """
    n_codes = len(code_to_cat)
    # Build a lookup array: index i → category string (or None for NaN sentinel)
    lookup = np.empty(n_codes + 1, dtype=object)   # last slot = None (NaN)
    for code, cat in code_to_cat.items():
        idx = int(round(code))
        if 0 <= idx < n_codes:
            lookup[idx] = cat
    lookup[n_codes] = None   # NaN sentinel

    flat = esi_samples.ravel()
    nan_mask = np.isnan(flat.astype(np.float64))
    indices = np.where(nan_mask, n_codes,
                       np.clip(np.round(flat.astype(np.float64)).astype(int), 0, n_codes - 1))
    return lookup[indices].reshape(esi_samples.shape)


# ─────────────────────────────────────────────────────────────
#  Visualization helpers (adapted from categorical/cat.py)
# ─────────────────────────────────────────────────────────────

def _sort_categories(categories, custom_order=None):
    """Sort categories numerically or alphabetically (strings last)."""
    order_map = {item: i for i, item in enumerate(custom_order)} if custom_order else None

    def key(item):
        try:
            return (0, float(item))
        except (ValueError, TypeError):
            s = str(item)
            if order_map is not None:
                return (1, order_map.get(s, float('inf')))
            return (1, s)

    return sorted(list(categories), key=key)


def _category_colors(n, cmap='Accent'):
    if n <= 10:
        cm = plt.colormaps.get(cmap, plt.colormaps['Accent'])
    elif n <= 20:
        cm = plt.colormaps['tab20']
    elif n <= 40:
        cm = plt.colormaps['plasma']
    else:
        cm = plt.colormaps['turbo']
    return ['#%02x%02x%02x' % (int(c[0]*255), int(c[1]*255), int(c[2]*255))
            for c in (cm(t) for t in np.linspace(0, 1, n))]


def plot_categorical_points(points, values, nonnum_order=None, ax=None,
                             cmap='Accent', title='', legend_title=''):
    """
    Plot 2D scatter coloured by categorical labels.

    Parameters
    ----------
    points : (n, 2) coordinates
    values : (n,) categorical labels
    nonnum_order : list, optional custom sort order for non-numeric categories
    ax : matplotlib axes (created if None)
    cmap : colormap name (≤10 categories)
    title, legend_title : plot annotation strings
    """
    if points.shape[1] != 2:
        raise SpatializeError("plot_categorical_points only supports 2D data.")
    unique_cats = _sort_categories(np.unique(values), nonnum_order)
    colors = _category_colors(len(unique_cats), cmap)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    for i, cat in enumerate(unique_cats):
        mask = values == cat
        ax.scatter(points[mask, 0], points[mask, 1],
                   label=str(cat), color=colors[i], s=10)

    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_aspect('equal')
    ax.legend(title=legend_title, loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────
#  Result classes
# ─────────────────────────────────────────────────────────────

class CatESIResult(EstimationResult):
    """
    Result of a categorical ESI estimation.

    Methods mirror spatialize's ``ESIResult``, adapted for categorical data.
    """

    def __init__(self, estimation, esi_samples, griddata=False,
                 original_shape=None, xi=None, ordinal_order=None):
        super().__init__(estimation, griddata, original_shape, xi=xi)
        self._esi_samples = esi_samples   # shape (p, n_partitions), object dtype
        self._precision = None
        self.ordinal_order = ordinal_order  # dict {category: rank} or None

    # ── samples ──────────────────────────────────────────────

    def esi_samples(self, raw=False):
        """
        Raw ESI samples, shape (p, n_partitions) or (d1, d2, n_partitions) for griddata.
        """
        if self.griddata and not raw:
            n = self._esi_samples.shape[1]
            return self._esi_samples.reshape(tuple(list(self.original_shape) + [n]))
        return self._esi_samples

    # ── precision ────────────────────────────────────────────

    def precision(self):
        """
        Per-location precision: proportion of ESI samples matching the estimation.

        Returns
        -------
        np.ndarray of float in [0, 1], shape (p,) or (d1, d2) for griddata.
        """
        prec = categorical_feature_precision(self._esi_samples, self._estimation)
        if self.griddata:
            self._precision = prec.reshape(self.original_shape)
        else:
            self._precision = prec
        return self._precision

    # ── ordinal order ─────────────────────────────────────────

    def set_ordinal_order(self, ordinal_order):
        """
        Set the ordinal category order (dict mapping category → integer rank).

        After calling this, ``re_estimate('mv')`` will automatically use the
        ordinal median-vote aggregation instead of the nominal majority vote.

        Parameters
        ----------
        ordinal_order : dict, e.g. ``{'Low': 0, 'Medium': 1, 'High': 2}``
        """
        self.ordinal_order = ordinal_order
        return self  # fluent interface

    # ── re-estimation ─────────────────────────────────────────

    def re_estimate(self, agg_function='mv', **agg_kwargs):
        """
        Re-aggregate ESI samples with a new aggregation function.

        Parameters
        ----------
        agg_function : ``'mv'`` | ``'btd'`` | callable, default ``'mv'``
            ``'mv'``
                Majority vote. Automatically selects the ordinal median-vote
                variant when ``self.ordinal_order`` is set.
            ``'btd'``
                Dawid-Skene EM aggregation. Infers ``categories_list``,
                ``category_type``, and ``map_dimensions`` from the result;
                pass extra kwargs for ``spatial_constraints``, etc.
            callable
                Any function ``(esi_samples, ...) -> estimation``; used as-is.
        **agg_kwargs
            Forwarded to the aggregation function (e.g. ``spatial_constraints``
            for ``'btd'``).

        Examples
        --------
        Nominal (default)::

            result.re_estimate()
            result.re_estimate('mv')

        Ordinal — set the order once, then just call ``'mv'``::

            result.set_ordinal_order({'Low': 0, 'Medium': 1, 'High': 2})
            result.re_estimate('mv')

        BTD with spatial constraints::

            result.re_estimate('btd', spatial_constraints=[('A', 'B')])
        """
        if agg_function == 'btd':
            agg_fn = self._make_btd_fn(**agg_kwargs)
        else:
            agg_fn = _resolve_agg_fn(agg_function, self.ordinal_order)

        self._estimation = agg_fn(self._esi_samples)
        self._precision = None
        return self.estimation()

    def _make_btd_fn(self, **kwargs):
        """Build a BTD aggregation callable, inferring metadata from the result."""
        from .agg_functions import aggregate_with_btd  # may not be available

        # Infer categories from samples
        all_vals = [s for s in self._esi_samples.ravel() if s is not None]
        categories_list = kwargs.pop(
            'categories_list',
            list(dict.fromkeys(str(v) for v in all_vals))
        )
        # Infer category type from ordinal_order presence / n_categories
        category_type = kwargs.pop(
            'category_type',
            'ordinal' if self.ordinal_order is not None
            else 'binary' if len(categories_list) == 2
            else 'nominal'
        )
        ordinal_order_map = kwargs.pop('ordinal_order_map', self.ordinal_order)
        # For griddata, map_dimensions is available from original_shape
        map_dimensions = kwargs.pop(
            'map_dimensions',
            self.original_shape if self.griddata else None
        )

        def _btd(esi_samples):
            estimation, _, _ = aggregate_with_btd(
                esi_samples,
                category_type=category_type,
                categories_list=categories_list,
                ordinal_order_map=ordinal_order_map,
                map_dimensions=map_dimensions,
                **kwargs,
            )
            return estimation

        return _btd

    # ── visualisation ─────────────────────────────────────────

    def plot_estimation(self, ax=None, title='Estimation', cmap='Accent',
                        nonnum_order=None, **kwargs):
        """
        Scatter plot of the categorical estimation (2D only).

        Parameters
        ----------
        ax : matplotlib axes
        title : str
        cmap : colormap for category colours
        nonnum_order : list, custom category order for legend
        """
        est = self.estimation()
        xi = self._xi

        if xi is None:
            raise SpatializeError("xi not available; cannot plot.")

        # Flatten griddata for scatter
        if self.griddata:
            if xi.shape[0] != 2:
                raise SpatializeError("plot_estimation only supports 2D griddata.")
            coords = np.column_stack([xi[0].ravel(), xi[1].ravel()])
            flat_est = est.ravel()
        else:
            if xi.shape[1] != 2:
                raise SpatializeError("plot_estimation only supports 2D data.")
            coords = xi
            flat_est = est

        plot_categorical_points(coords, flat_est, nonnum_order=nonnum_order,
                                 ax=ax, cmap=cmap, title=title)

    def __repr__(self):
        est = self.estimation()
        unique = np.unique(est)
        return (f"CatESIResult\n"
                f"  categories : {list(unique)}\n"
                f"  n_locations: {est.shape[0] if not self.griddata else np.prod(self.original_shape)}\n"
                f"  n_partitions: {self._esi_samples.shape[1]}\n"
                f"Use .estimation(), .esi_samples(), .precision(), .plot_estimation()")


class CatESIGridSearchResult(GridSearchResult):
    """
    Result of a hyperparameter search for categorical ESI.
    """

    def __init__(self, search_result_data, classifier, ordinal_order=None, sklearn_classifier=None, random_state=None):
        super().__init__(search_result_data)
        self.classifier = classifier
        self.ordinal_order = ordinal_order
        self.sklearn_classifier = sklearn_classifier
        self.random_state = random_state

    def best_result(self, **kwargs):
        """Return a dict of best parameters suitable for passing to cat_esi_nongriddata."""
        b_param = self.best_params.sort_values(by='cv_error', ascending=True)
        row = pd.DataFrame(b_param.iloc[0]).to_dict()
        index = list(row.keys())[0]
        result = row[index]
        # pandas stores None as NaN and ints as float64; restore correct types
        def _restore(v):
            if pd.isna(v):
                return None
            if isinstance(v, float) and v.is_integer():
                return int(v)
            return v
        result = {k: _restore(v) for k, v in result.items()}
        result["result_data_index"] = index
        result["classifier"] = self.classifier
        # Use 'mv' string so _resolve_agg_fn automatically picks ordinal/nominal variant
        result["agg_function"] = 'mv'
        result["ordinal_order"] = self.ordinal_order
        if self.sklearn_classifier is not None:
            result["sklearn_classifier"] = self.sklearn_classifier
        if self.random_state is not None:
            result["random_state"] = self.random_state
        return result


# ─────────────────────────────────────────────────────────────
#  Internal C++ call
# ─────────────────────────────────────────────────────────────

@signature_overload(
    pivot_arg=("classifier", "knn_pca", "classifier"),
    common_args={
        "n_partitions": 300,
        "alpha": 0.8,
        "seed": np.random.randint(1000, 10000),
        "agg_function": 'mv',
        "ordinal_order": None,
        "callback": default_singleton_callback,
        "best_params_found": None,
    },
    specific_args={
        "knn_pca":      {"n_neighbors": None, "max_points": None, "n_cv_splits": None},
        "scikit-learn": {"sklearn_classifier": None},
        "knn":          {"n_neighbors": None},
        "svm":          {"C": None, "kernel": None, "gamma": None, "random_state": None},
        "rf":           {"n_estimators": None, "max_depth": None, "random_state": None},
        "dt":           {"max_depth": None, "min_samples_split": None, "random_state": None},
    },
)
def _call_custom_esi(points, values, xi, **kwargs):
    """
    Core call to ``libspatialize.estimation_custom_esi``.

    Handles encoding of string categories to integer codes, builds
    classifier callbacks, calls C++, decodes results, and applies
    the aggregation function.
    """
    # log_message(logging.logger.debug("calling estimation_custom_esi"))  # TODO: move to facade layer

    if kwargs["best_params_found"] is not None:
        try:
            best = kwargs["best_params_found"]["n_partitions"]
            log_message(logging.logger.debug(f"best number of partitions found: {best}"))
            del kwargs["best_params_found"]["n_partitions"]  # this param can be overwritten in all cases
        except KeyError:
            pass
        log_message(logging.logger.debug(f"using best params found: {kwargs['best_params_found']}"))
        for k in kwargs["best_params_found"]:
            try:
                kwargs[k] = kwargs["best_params_found"][k]
            except KeyError:
                pass

    #if kwargs.get("callback") is not singleton_null_callback:
    #    log_message(logging.logger.debug(f"calling estimation_custom_esi with kwargs: {kwargs}"))

    # Encode values to integer codes (handles both numeric and string labels)
    encoded_values, _, code_to_cat = _make_encoder(values)

    # Build classifier callbacks
    set_cell_params_fn, classifier_fn = get_classifier_fns(
        kwargs["classifier"],
        **{k: v for k, v in kwargs.items() if k != "classifier"},
    )

    try:
        _, esi_samples_raw = lsp.estimation_custom_esi(
            np.float64(points),
            encoded_values,
            kwargs["n_partitions"],
            kwargs["alpha"],
            kwargs["seed"],
            np.float64(xi),
            set_cell_params_fn,
            classifier_fn,
            kwargs["callback"],
        )
    except Exception as e:
        raise SpatializeError(e)

    # Decode integer codes back to original category labels
    esi_samples = _decode_samples(esi_samples_raw, code_to_cat)

    agg_fn = _resolve_agg_fn(kwargs["agg_function"], kwargs.get("ordinal_order"))
    estimation = agg_fn(esi_samples)
    return estimation, esi_samples


# ─────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────

def cat_esi_griddata(points, values, xi, **kwargs) -> CatESIResult:
    """
    Categorical ESI estimation on a regular grid.

    Parameters
    ----------
    points : (N, D) sample coordinates
    values : (N,) categorical labels
    xi : tuple of D grid arrays, as returned by np.meshgrid / np.mgrid
    classifier : {'knn_pca', 'scikit-learn'}, default 'knn_pca'
    agg_function : callable, default aggregate_with_mv
    n_partitions : int, default 300
    alpha : float, default 0.8
    seed : int
    callback : progress callback
    n_neighbors : int (knn_pca only) – fix k neighbours
    max_points : int (knn_pca only) – subsampling cap per cell
    n_cv_splits : int (knn_pca only) – CV splits for parameter search
    sklearn_classifier : sklearn estimator (scikit-learn only)

    Returns
    -------
    CatESIResult
    """
    ng_xi, original_shape = flatten_grid_data(xi)
    estimation, esi_samples = _call_custom_esi(points, values, ng_xi, **kwargs)
    return CatESIResult(estimation, esi_samples,
                        griddata=True, original_shape=original_shape, xi=xi,
                        ordinal_order=kwargs.get('ordinal_order'))


def cat_esi_nongriddata(points, values, xi, **kwargs) -> CatESIResult:
    """
    Categorical ESI estimation at arbitrary (non-grid) locations.

    Parameters
    ----------
    points : (N, D) sample coordinates
    values : (N,) categorical labels
    xi : (M, D) query coordinates
    classifier : {'knn_pca', 'scikit-learn'}, default 'knn_pca'
    agg_function : callable, default aggregate_with_mv
    n_partitions : int, default 300
    alpha : float, default 0.8
    seed : int
    callback : progress callback
    n_neighbors : int (knn_pca only)
    max_points : int (knn_pca only)
    n_cv_splits : int (knn_pca only)
    sklearn_classifier : sklearn estimator (scikit-learn only)

    Returns
    -------
    CatESIResult
    """
    estimation, esi_samples = _call_custom_esi(points, values, xi, **kwargs)
    return CatESIResult(estimation, esi_samples, xi=xi,
                        ordinal_order=kwargs.get('ordinal_order'))


@signature_overload(
    pivot_arg=("classifier", "knn_pca", "classifier"),
    common_args={
        "k": 5,                                # number of CV folds
        "n_partitions": [100, 300],
        "alpha": [0.7, 0.8, 0.9],
        "scoring": "f1_macro",
        "seed": np.random.randint(1000, 10000),
        "folding_seed": np.random.randint(1000, 10000),
        "agg_function": 'mv',
        "ordinal_order": None,
        "callback": default_singleton_callback,
        "griddata": False,
    },
    specific_args={
        "knn_pca": {
            "n_neighbors": [3, 5, 7],
            "max_points": [None],
            "n_cv_splits": [None],
        },
        "scikit-learn": {"sklearn_classifier": None},
        "knn": {"n_neighbors": [3, 5, 7]},
        "svm": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"], "gamma": ["scale"], "random_state": None},
        "rf":  {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "random_state": None},
        "dt":  {"max_depth": [None, 3, 5, 10], "min_samples_split": [2, 5], "random_state": None},
    },
)
def cat_esi_hparams_search(points, values, xi, **kwargs) -> CatESIGridSearchResult:
    """
    Hyperparameter search for categorical ESI via k-fold cross-validation.

    k-fold CV is implemented in Python (no C++ LOO/kfold for custom ESI).
    For each parameter combination, trains on k-1 folds and evaluates on
    the held-out fold using categorical accuracy.

    Parameters
    ----------
    points : (N, D)
    values : (N,) categorical labels
    xi : accepted for API consistency with cat_esi_griddata/cat_esi_nongriddata;
        not used during CV (held-out fold points serve as query locations)
    classifier : {'knn_pca', 'scikit-learn'}
    k : int, number of CV folds (default 5)
    n_partitions : list of int
    alpha : list of float
    scoring : str or callable, default ``'f1_macro'``
        Scoring metric for CV. String options: ``'accuracy'``, ``'f1_macro'``,
        ``'f1_weighted'``, ``'f1_micro'``, ``'precision_macro'``,
        ``'recall_macro'``, ``'cohen_kappa'``. Or any callable
        ``(true, pred) → error`` (lower is better).
    seed, folding_seed : int
    agg_function : callable
    griddata : bool
    n_neighbors : list of int (knn_pca only)
    max_points : list (knn_pca only)
    n_cv_splits : list (knn_pca only)
    sklearn_classifier : sklearn estimator (scikit-learn only)

    Returns
    -------
    CatESIGridSearchResult
    """
    log_message(logging.logger.debug("cat_esi_hparams_search: building parameter grid"))

    scorer = resolve_scoring(kwargs["scoring"])

    # Build parameter grid (common args + classifier-specific args)
    grid_params = {
        "n_partitions": kwargs["n_partitions"],
        "alpha": kwargs["alpha"],
    }
    if kwargs["classifier"] == "knn_pca":
        if kwargs.get("n_neighbors") is not None:
            grid_params["n_neighbors"] = kwargs["n_neighbors"]
        if kwargs.get("max_points") is not None:
            grid_params["max_points"] = kwargs["max_points"]
        if kwargs.get("n_cv_splits") is not None:
            grid_params["n_cv_splits"] = kwargs["n_cv_splits"]
    elif kwargs["classifier"] in SKLEARN_CLASSIFIER_PARAMS:
        for param in SKLEARN_CLASSIFIER_PARAMS[kwargs["classifier"]]:
            if kwargs.get(param) is not None:
                grid_params[param] = kwargs[param]

    param_grid = list(ParameterGrid(grid_params))

    k = kwargs["k"]
    n = points.shape[0]

    # Support LOO: k=-1 or k==n means leave-one-out
    if k == -1 or k == n:
        k = n

    rng = np.random.default_rng(kwargs["folding_seed"])
    kf = KFold(n_splits=k, shuffle=True, random_state=int(rng.integers(0, 9999)))
    fold_splits = list(kf.split(points))

    results = {}
    n_combos = len(param_grid)
    kwargs["callback"](logging.progress.init(n_combos, 1))

    for i, param_set in enumerate(param_grid):
        all_true = []
        all_pred = []

        for tr_idx, te_idx in fold_splits:
            tr_pts = np.ascontiguousarray(points[tr_idx])
            tr_vals = values[tr_idx]
            te_pts = np.ascontiguousarray(points[te_idx])
            te_vals = values[te_idx]

            combo = {
                "classifier": kwargs["classifier"],
                "n_partitions": param_set["n_partitions"],
                "alpha": param_set["alpha"],
                "seed": kwargs["seed"],
                "agg_function": kwargs["agg_function"],
                "ordinal_order": kwargs.get("ordinal_order"),
                "callback": singleton_null_callback,
                "best_params_found": None,
            }
            if kwargs["classifier"] == "knn_pca":
                combo["n_neighbors"] = param_set.get("n_neighbors")
                combo["max_points"] = param_set.get("max_points")
                combo["n_cv_splits"] = param_set.get("n_cv_splits")
            elif kwargs["classifier"] == "scikit-learn":
                combo["sklearn_classifier"] = kwargs["sklearn_classifier"]
            elif kwargs["classifier"] in SKLEARN_CLASSIFIER_PARAMS:
                for param in SKLEARN_CLASSIFIER_PARAMS[kwargs["classifier"]]:
                    combo[param] = param_set.get(param)
                if kwargs["classifier"] in SKLEARN_RANDOM_STATE_CLASSIFIERS:
                    combo["random_state"] = kwargs.get("random_state")

            try:
                estimation, _ = _call_custom_esi(tr_pts, tr_vals, te_pts, **combo)
                all_true.append(te_vals)
                all_pred.append(estimation)
            except Exception:
                pass  # skip failed folds

        # Score on all holdout predictions combined (mirrors spatialize's kfold CV structure)
        if all_true:
            results[i] = scorer(
                np.concatenate(all_true),
                np.concatenate(all_pred),
            )
        else:
            results[i] = 1.0  # all folds failed

        kwargs["callback"](logging.progress.inform())

    kwargs["callback"](logging.progress.stop())

    rows = []
    for i, param_set in enumerate(param_grid):
        row = {"cv_error": results[i], "classifier": kwargs["classifier"]}
        row.update(param_set)
        rows.append(row)

    result_data = pd.DataFrame(rows)
    return CatESIGridSearchResult(result_data, kwargs["classifier"],
                                  ordinal_order=kwargs.get("ordinal_order"),
                                  sklearn_classifier=kwargs.get("sklearn_classifier"),
                                  random_state=kwargs.get("random_state"))

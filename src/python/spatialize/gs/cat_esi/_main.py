import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid, KFold

from spatialize import EstimationResult, GridSearchResult, SpatializeError, session
from spatialize.gs import lib_spatialize_facade
from spatialize._util import signature_overload
from spatialize._math_util import flatten_grid_data
from spatialize.logging import default_singleton_callback, singleton_null_callback
from spatialize import logging
from spatialize.logging import log_message
from spatialize.viz import plot_categorical_colormap, plot_colormap_data, PlotStyle

from .agg_functions import aggregate_with_mv, aggregate_with_ordinal_mv, categorical_feature_precision, categorical_precision_cube
from .classifiers import get_classifier_fns, SKLEARN_CLASSIFIER_PARAMS, SKLEARN_RANDOM_STATE_CLASSIFIERS
from .score_functions import resolve_scoring


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
    arr = np.asarray(values)
    if arr.dtype.kind == 'f' and np.isnan(arr).any():
        raise ValueError(
            "Input 'values' contains NaN. Remove or impute NaN entries before calling cat_esi."
        )

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
    flat_float = flat.astype(np.float64)
    nan_mask = np.isnan(flat_float)
    indices = np.full(len(flat), n_codes, dtype=int)   # default: NaN sentinel slot
    valid = ~nan_mask
    indices[valid] = np.clip(np.round(flat_float[valid]).astype(int), 0, n_codes - 1)
    return lookup[indices].reshape(esi_samples.shape)


# ─────────────────────────────────────────────────────────────
#  Result classes
# ─────────────────────────────────────────────────────────────

class CatESIResult(EstimationResult):
    """Result of a categorical ESI estimation.

    Wraps the aggregated categorical estimation together with the raw
    per-partition ESI samples, and exposes precision, re-aggregation, and
    plotting methods. Methods mirror spatialize's :class:`~spatialize.result.EstimationResult`,
    adapted for categorical data — in particular, :meth:`precision` here
    returns a per-location agreement ratio in ``[0, 1]`` rather than an
    error metric.

    Parameters
    ----------
    estimation : ndarray
        Aggregated categorical estimation, one label per query location
        (or grid cell, for griddata).
    esi_samples : ndarray
        Raw per-partition category labels, shape ``(p, n_partitions)``
        where ``p`` is the number of query locations.
    griddata : bool, optional
        Whether `estimation` and `esi_samples` correspond to a regular
        grid. Default: ``False``.
    original_shape : tuple, optional
        Shape of the target grid before flattening, required when
        `griddata` is ``True``.
    xi : array_like, optional
        Query locations (or grid arrays, for griddata) used to produce
        the estimation.
    ordinal_order : dict, optional
        Mapping of category label to integer rank, e.g.
        ``{'Low': 0, 'Medium': 1, 'High': 2}``. When set, :meth:`re_estimate`
        uses ordinal median-vote aggregation instead of nominal majority
        vote. Default: ``None`` (nominal categories).
    """

    def __init__(self, estimation, esi_samples, griddata=False,
                 original_shape=None, xi=None, ordinal_order=None):
        """Store the estimation, raw samples, and griddata metadata.

        Parameters
        ----------
        estimation : ndarray
            Aggregated categorical estimation.
        esi_samples : ndarray
            Raw per-partition category labels, shape ``(p, n_partitions)``.
        griddata : bool, optional
            Whether the data corresponds to a regular grid. Default: ``False``.
        original_shape : tuple, optional
            Shape of the target grid before flattening, required when
            `griddata` is ``True``.
        xi : array_like, optional
            Query locations (or grid arrays, for griddata).
        ordinal_order : dict, optional
            Mapping of category label to integer rank. Default: ``None``.
        """
        super().__init__(estimation, griddata, original_shape, xi=xi)
        self._esi_samples = esi_samples   # shape (p, n_partitions), object dtype
        self._precision = None
        self.ordinal_order = ordinal_order  # dict {category: rank} or None

    # ── samples ──────────────────────────────────────────────

    def esi_samples(self, raw=False):
        """Return the raw, unaggregated per-partition ESI samples.

        Parameters
        ----------
        raw : bool, optional
            If ``True``, always return the flat ``(p, n_partitions)`` array,
            even for griddata results. If ``False`` (default) and the result
            is griddata, reshape to ``(d1, d2, ..., n_partitions)`` using
            `original_shape`.

        Returns
        -------
        ndarray
            Category labels per partition, shape ``(p, n_partitions)`` or
            ``(d1, d2, n_partitions)`` for griddata.
        """
        if self.griddata and not raw:
            n = self._esi_samples.shape[1]
            return self._esi_samples.reshape(tuple(list(self.original_shape) + [n]))
        return self._esi_samples

    # ── precision ────────────────────────────────────────────

    def precision(self, precision_function=categorical_feature_precision):
        """
        Per-location precision computed by *precision_function*.

        Parameters
        ----------
        precision_function : callable, optional
            A function with signature ``(estimation, esi_samples) → np.ndarray``
            that returns a per-location float metric (lower or higher = better
            depending on the function).  Matches the ``loss_function`` convention
            used by ``ESIResult.precision()``.

            Built-in options (importable from ``spatialize.gs.cat_esi``):

            - ``categorical_feature_precision`` *(default)* – proportion of ESI
              samples that disagree with the aggregated estimation at each
              location; values in [0, 1] where 1 means full disagreement
              (high uncertainty).

            Custom functions must accept ``(estimation, esi_samples)`` and
            return a 1-D array of length ``p`` (number of query locations).

        Returns
        -------
        np.ndarray of float, shape (p,) or (d1, d2) for griddata.
        """
        prec = precision_function(self._estimation, self._esi_samples)
        if self.griddata:
            self._precision = prec.reshape(self.original_shape)
        else:
            self._precision = prec
        return self._precision

    def precision_cube(self, precision_function=categorical_precision_cube):
        """
        Per-location, per-partition uncertainty — the unaggregated analogue of
        :meth:`precision`.

        Mirrors ``ESIResult.precision_cube()``: instead of collapsing across
        partitions, returns the full ``(p, n_partitions)`` array so that callers
        can apply their own aggregation or inspect the distribution per location.

        Parameters
        ----------
        precision_function : callable, optional
            Signature ``(estimation, esi_samples) → np.ndarray of shape (p, n)``.
            Default is ``categorical_precision_cube``: 1.0 where the partition
            prediction disagrees with the estimation, 0.0 otherwise.

        Returns
        -------
        np.ndarray, shape (p, n_partitions) or (d1, d2, n_partitions) for griddata.
        """
        cube = precision_function(self._estimation, self._esi_samples)
        if self.griddata:
            return cube.reshape(self.original_shape[0], self.original_shape[1], cube.shape[1])
        return cube

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

    def plot_estimation(self, ax=None, w=None, h=None, cmap=None, nonnum_order=None,
                        title='Estimation', theme='alges',
                        figsize=None, dpi=100, **kwargs):
        """
        Plot the categorical estimation (2D only).

        Uses ``plot_categorical_colormap`` for griddata and
        ``plot_categorical_scatter`` for non-griddata. The *theme* is applied
        via ``PlotStyle`` in both cases, mirroring ``ESIResult.plot_estimation``.

        Parameters
        ----------
        ax : matplotlib Axes, optional
        cmap : str | matplotlib Colormap | list of colours | None
            Colour source for the discrete category palette.  Accepts the
            same options as ``plot_colormap_data``:

            - ``None`` (default) – discrete swatches sampled from the active
              *theme*'s own colormap (see ``PlotStyle.resolve_cmap``), falling
              back to ``'Accent'``/``tab20``/``plasma``/``turbo`` by category
              count when ``theme`` is ``None``.
            - A **spatialize palette** name (e.g. ``'alges'``, ``'crest_r'``).
            - A **Scientific Colour Map** name (e.g. ``'batlow'``, ``'roma'``).
            - Any **matplotlib** colourmap name (e.g. ``'tab20'``, ``'viridis'``).
            - A matplotlib **Colormap object** (sampled at n evenly-spaced points).
            - A **list of hex/named colours** (cycled if shorter than n categories).
        nonnum_order : list, optional custom sort order for non-numeric categories
        title : str
        theme : str, PlotStyle theme. Default ``'alges'``.
        figsize, dpi : figure size and resolution (when *ax* is None)
        **kwargs : forwarded to the underlying plot function
        """
        est = self.estimation()
        xi = self._xi

        if self.griddata and len(xi) != 2:
            raise SpatializeError("plot_estimation only supports 2D griddata.")
        if not self.griddata and xi.shape[1] != 2:
            raise SpatializeError("plot_estimation only supports 2D data.")

        with PlotStyle(theme=theme) as style:
            resolved_cmap = cmap
            if resolved_cmap is None:
                n_categories = len(np.unique(np.asarray(est, dtype=object)))
                resolved_cmap = style.resolve_cmap('categorical', n_categories=n_categories)
            plot_categorical_colormap(
                est, cmap=resolved_cmap, nonnum_order=nonnum_order,
                ax=ax, w=w, h=h, griddata=self.griddata,
                xi_locations=xi,
                extent=self._get_extent(),
                title=title, figsize=figsize, dpi=dpi, **kwargs,
            )

    def plot_precision(self, ax=None, w=None, h=None, theme='alges', cmap=None, **imshow_args):
        """Plot the per-location agreement ratio of the estimation.

        Computes :meth:`precision` if it has not been computed yet, then
        renders it as a colormap image via
        :func:`~spatialize.viz.plot_colormap_data`. Values are in
        ``[0, 1]``, the fraction of ESI partition samples that disagree
        with the aggregated estimation at each location (not an error
        metric).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If ``None``, a new figure/axis is created.
        w : float, optional
            Width scale factor for the plot.
        h : float, optional
            Height scale factor for the plot.
        theme : str, optional
            Theme name. Available: ``'whitegrid'``, ``'darkgrid'``,
            ``'white'``, ``'dark'``, ``'alges'``, ``'minimal'``,
            ``'publication'``. Default: ``'alges'``.
        cmap : str, optional
            Colormap for the plot. If ``None``, uses the theme default or
            ``'bwr'``.
        **imshow_args
            Additional keyword arguments passed to
            :func:`~spatialize.viz.plot_colormap_data`.

        Raises
        ------
        SpatializeError
            If the result is not 2D.
        """
        if self._precision is None:
            self._precision = self.precision()

        xi = self._xi
        if self.griddata and len(xi) != 2:
            raise SpatializeError("plot_precision only supports 2D griddata.")
        if not self.griddata and xi.shape[1] != 2:
            raise SpatializeError("plot_precision only supports 2D data.")

        plot_imshow_args = imshow_args.copy()
        if not cmap:
            cmap = plot_imshow_args.pop('cmap', None)
        if 'extent' not in plot_imshow_args:
            extent = self._get_extent()
            if extent is not None:
                plot_imshow_args['extent'] = extent

        with PlotStyle(theme=theme, precision_cmap=cmap) as style:
            plot_colormap_data(self._precision, ax=ax, w=w, h=h, xi_locations=xi, griddata=self.griddata, cmap=style.precision_cmap, **plot_imshow_args)

    def quick_plot(self, w=None, h=None,
                   theme='alges',
                   estimation_cmap=None,
                   precision_cmap=None,
                   show=True,
                   **fig_args):

        """
        Side-by-side plot of estimation and precision.

        Parameters
        ----------
        w : float, optional
            Width scale factor passed to plot_estimation / plot_precision.
        h : float, optional
            Height scale factor passed to plot_estimation / plot_precision.
        theme : str
            PlotStyle theme. Default ``'alges'``.
        estimation_cmap : str or Colormap, optional
            Discrete colormap for the estimation panel. Defaults to swatches
            sampled from the theme's own colormap (see ``plot_estimation``).
        precision_cmap : str or Colormap, optional
            Colormap for the precision panel. Defaults to the theme's colormap.
        show : bool, optional
            If True (default), call plt.show() and return None. If False, return the figure.
        **fig_args :
            Extra keyword arguments forwarded to ``plt.figure()`` (e.g.
            ``figsize=(10, 8)``, ``dpi=120``).

        Returns
        -------
        None if show=True, otherwise matplotlib.figure.Figure.
        """
        xi = self._xi
        if xi is None:
            raise SpatializeError("xi not available; cannot quick_plot.")

        if self.griddata and len(xi) > 2:
            raise SpatializeError("quick_plot() for 3D+ griddata is not supported.")
        if not self.griddata and xi.shape[1] > 2:
            raise SpatializeError("quick_plot() for 3D+ data is not supported.")

        plot_fig_args = fig_args.copy()
        plot_fig_args.setdefault('figsize', (10,8))
        plot_fig_args.setdefault('dpi', 120)

        with PlotStyle(theme=theme, precision_cmap=precision_cmap) as style:
            fig = plt.figure(**plot_fig_args)
            gs = fig.add_gridspec(1, 2, wspace=0.5)
            ax1, ax2 = gs.subplots()

            resolved_estimation_cmap = estimation_cmap
            if resolved_estimation_cmap is None:
                n_categories = len(np.unique(np.asarray(self.estimation(), dtype=object)))
                resolved_estimation_cmap = style.resolve_cmap('categorical', n_categories=n_categories)

            ax1.set_title('Estimation')
            self.plot_estimation(ax1, w=w, h=h, theme=None, cmap=resolved_estimation_cmap)
            ax1.set_aspect('equal')

            ax2.set_title('Precision')
            self.plot_precision(ax2, w=w, h=h, theme=None, cmap=style.precision_cmap)
            ax2.set_aspect('equal')

        if show:
            plt.show()
        else:
            return fig

    _title = "Categorical ESI estimation"
    _methods = "estimation() · esi_samples() · precision() · quick_plot()"

    def _summary(self):
        from spatialize import _display
        from spatialize.result import _data_rows
        est = np.asarray(self.estimation(), dtype=object).ravel()
        labels, counts = np.unique([str(v) for v in est if v is not None], return_counts=True)
        rows = [[lab, int(c), f"{100 * c / max(len(est), 1):.1f} %"] for lab, c in zip(labels, counts)]
        members = np.asarray(self._esi_samples, dtype=object)
        ensemble = [("members per location", members.shape[1] if members.ndim == 2 else None),
                    ("categories", len(labels)), ("ordinal", "yes" if self.ordinal_order else "no")]
        return _display.Summary(
            self._title, blocks=[("kv", "Data", _data_rows(self)), ("kv", "Ensemble", ensemble),
                                 ("table", "Most frequent category per location", ["category", "locations", "share"], rows)],
            footer=self._methods)


class CatESIGridSearchResult(GridSearchResult):
    _title = "Categorical ESI hyperparameter search"

    """Result of a hyperparameter search for categorical ESI.

    Wraps the per-combination cross-validation scores produced by
    :func:`cat_esi_hparams_search`, together with the classifier and
    ordinal metadata needed to reconstruct a ready-to-use parameter
    set via :meth:`best_result`.

    Parameters
    ----------
    search_result_data : pandas.DataFrame
        One row per parameter combination, with a ``cv_error`` column and
        one column per searched hyperparameter.
    classifier : str
        Classifier used during the search, e.g. ``'knn_pca'`` or
        ``'scikit-learn'``.
    ordinal_order : dict, optional
        Mapping of category label to integer rank, propagated to the
        returned parameter dict so downstream calls reuse the same
        ordinal aggregation. Default: ``None``.
    sklearn_classifier : object, optional
        Fitted or unfitted scikit-learn estimator instance, required when
        `classifier` is ``'scikit-learn'``. Default: ``None``.
    random_state : int, optional
        Random state forwarded to scikit-learn classifiers that accept
        one (e.g. ``'svm'``, ``'rf'``, ``'dt'``). Default: ``None``.
    seed : int, optional
        ESI partitioning seed used during the search. Default: ``None``.
    """

    def __init__(self, search_result_data, classifier, ordinal_order=None, sklearn_classifier=None, random_state=None, seed=None):
        """Store the search results and classifier metadata.

        Parameters
        ----------
        search_result_data : pandas.DataFrame
            One row per parameter combination, with a ``cv_error`` column.
        classifier : str
            Classifier used during the search.
        ordinal_order : dict, optional
            Mapping of category label to integer rank. Default: ``None``.
        sklearn_classifier : object, optional
            scikit-learn estimator instance. Default: ``None``.
        random_state : int, optional
            Random state for scikit-learn classifiers. Default: ``None``.
        seed : int, optional
            ESI partitioning seed. Default: ``None``.
        """
        super().__init__(search_result_data)
        self.classifier = classifier
        self.ordinal_order = ordinal_order
        self.sklearn_classifier = sklearn_classifier
        self.random_state = random_state
        self.seed = seed

    def best_result(self, **kwargs):
        """Return the best-scoring parameter combination as a ready-to-use dict.

        Selects the row of `search_result_data` with the lowest ``cv_error``,
        restores correct Python types (pandas stores ``None`` as ``NaN`` and
        integers as ``float64``), and augments it with the classifier and
        ordinal metadata stored on this result.

        Parameters
        ----------
        **kwargs
            Accepted for API consistency; currently unused.

        Returns
        -------
        dict
            Best-performing hyperparameter combination, including
            ``result_data_index``, ``classifier``, ``agg_function`` (always
            ``'mv'``), ``ordinal_order``, and, when set, ``sklearn_classifier``,
            ``random_state``, and ``seed``. Suitable for unpacking directly
            into :func:`cat_esi_griddata` or :func:`cat_esi_nongriddata` via
            ``best_params_found``.
        """
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
        if self.seed is not None:
            result["seed"] = self.seed
        return result


# ─────────────────────────────────────────────────────────────
#  Internal C++ call
# ─────────────────────────────────────────────────────────────

def _engine_alpha(kwargs):
    """alpha as the engine takes it: negative for Voronoi with uniform nuclei."""
    alpha = kwargs["alpha"]
    return -alpha if kwargs.get("p_process") == "voronoi" and not kwargs.get("data_cond", True) else alpha


@signature_overload(
    pivot_arg=("classifier", "knn_pca", "classifier"),
    common_args={
        "n_partitions": 300,
        "alpha": 0.8,
        "p_process": "mondrian",
        "data_cond": True,
        "seed": None,
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
    Core call to ``libspatialize.run`` with the custom decoder.

    Handles encoding of string categories to integer codes, builds
    classifier callbacks, calls C++, decodes results, and applies
    the aggregation function.
    """
    if kwargs.get("seed") is None:
        kwargs["seed"] = np.random.randint(1000, 10000)  # generate per-call so each run is independent

    if kwargs["best_params_found"] is not None:
        kwargs["best_params_found"] = dict(kwargs["best_params_found"])  # copy to avoid mutating caller's dict
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

    # Encode values to integer codes (handles both numeric and string labels)
    encoded_values, _, code_to_cat = _make_encoder(values)

    # Build classifier callbacks
    set_cell_params_fn, classifier_fn = get_classifier_fns(
        kwargs["classifier"],
        **{k: v for k, v in kwargs.items() if k != "classifier"},
    )

    try:
        _, esi_samples_raw = lib_spatialize_facade.run(
            np.float64(points), encoded_values, np.float64(xi), kwargs["p_process"], "custom",
            {"post_creation": set_cell_params_fn, "estimation": classifier_fn},
            _engine_alpha(kwargs), kwargs["n_partitions"], kwargs["seed"], callback=kwargs["callback"])
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
    p_process : {"mondrian", "mondrian-raw", "voronoi"}, default "mondrian"
        Partition process, as in :func:`~spatialize.gs.esi.esi_griddata`.
    data_cond : bool, default True
        For ``"voronoi"``, whether the nuclei are drawn among the data or
        uniformly in the box.
    seed : int
    callback : progress callback
    n_neighbors : int (knn_pca only) – fix k neighbours
    max_points : int (knn_pca only) – subsampling cap per cell
    n_cv_splits : int (knn_pca only) – CV splits for parameter search
    sklearn_classifier : sklearn estimator (scikit-learn only)
    best_params_found : dict or None, optional
        Parameter dict typically obtained from
        :meth:`CatESIGridSearchResult.best_result`. When given, every key it
        contains **overrides** the corresponding argument passed at the call
        site, with one exception: ``n_partitions`` is ignored if present --
        the value passed at the call site (or its default of ``300``) is
        used instead. This is intentional: it lets you run the
        hyperparameter search cheaply with few partitions and then estimate
        with many. Default: ``None``.

        .. warning::
            :meth:`CatESIGridSearchResult.best_result` always injects
            ``result_data_index``, ``classifier``, ``agg_function``
            (always the string ``'mv'``) and ``ordinal_order`` into the dict
            it returns, plus ``sklearn_classifier``, ``random_state`` and
            ``seed`` when those were set on the search result. Passing that
            dict therefore silently overrides any of those arguments given
            at the call site.

    Returns
    -------
    CatESIResult

    Notes
    -----
    Unlike the continuous ESI path, `best_params_found` is **not** defensively
    copied before use: the ``n_partitions`` key is deleted from the very
    dictionary object you pass in. If you hold a reference to the dict
    returned by :meth:`CatESIGridSearchResult.best_result` and pass it to
    this function, ``n_partitions`` is removed from *your* dict as a side
    effect, and is no longer available for later inspection or reuse. Pass a
    copy (``dict(best)``) if you need the original to stay intact.

    See Also
    --------
    cat_esi_hparams_search : Grid search that produces a
        ``best_params_found`` dict for this function.
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
    p_process : {"mondrian", "mondrian-raw", "voronoi"}, default "mondrian"
        Partition process, as in :func:`~spatialize.gs.esi.esi_griddata`.
    data_cond : bool, default True
        For ``"voronoi"``, whether the nuclei are drawn among the data or
        uniformly in the box.
    seed : int
    callback : progress callback
    n_neighbors : int (knn_pca only)
    max_points : int (knn_pca only)
    n_cv_splits : int (knn_pca only)
    sklearn_classifier : sklearn estimator (scikit-learn only)
    best_params_found : dict or None, optional
        Parameter dict typically obtained from
        :meth:`CatESIGridSearchResult.best_result`. When given, every key it
        contains **overrides** the corresponding argument passed at the call
        site, with one exception: ``n_partitions`` is ignored if present --
        the value passed at the call site (or its default of ``300``) is
        used instead. This is intentional: it lets you run the
        hyperparameter search cheaply with few partitions and then estimate
        with many. Default: ``None``.

        .. warning::
            :meth:`CatESIGridSearchResult.best_result` always injects
            ``result_data_index``, ``classifier``, ``agg_function``
            (always the string ``'mv'``) and ``ordinal_order`` into the dict
            it returns, plus ``sklearn_classifier``, ``random_state`` and
            ``seed`` when those were set on the search result. Passing that
            dict therefore silently overrides any of those arguments given
            at the call site.

    Returns
    -------
    CatESIResult

    Notes
    -----
    Unlike the continuous ESI path, `best_params_found` is **not** defensively
    copied before use: the ``n_partitions`` key is deleted from the very
    dictionary object you pass in. If you hold a reference to the dict
    returned by :meth:`CatESIGridSearchResult.best_result` and pass it to
    this function, ``n_partitions`` is removed from *your* dict as a side
    effect, and is no longer available for later inspection or reuse. Pass a
    copy (``dict(best)``) if you need the original to stay intact.

    See Also
    --------
    cat_esi_hparams_search : Grid search that produces a
        ``best_params_found`` dict for this function.
    """
    estimation, esi_samples = _call_custom_esi(points, values, xi, **kwargs)
    return CatESIResult(estimation, esi_samples, xi=xi,
                        ordinal_order=kwargs.get('ordinal_order'))


@signature_overload(
    pivot_arg=("classifier", "knn_pca", "classifier"),
    common_args={
        "k": 5,                                # number of CV folds
        "cv": "engine",                        # "engine" (cells of one ensemble) or "refit" (per fold)
        "p_process": "mondrian",
        "data_cond": True,
        "n_partitions": [100, 300],
        "alpha": [0.7, 0.8, 0.9],
        "scoring": "f1_macro",
        "seed": None,
        "folding_seed": None,
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

    For each parameter combination, predicts every datum from the others
    (see ``cv``) and scores the predicted categories. The result also records
    the share of the data that got no category (``left_out``, members of
    cells without data under the session setting ``empty_cells="nan"``) and,
    with ``cv="engine"``, the share of undefined members (``nan_members``); a
    warning lists the configurations over the session setting
    ``max_left_out``.

    Parameters
    ----------
    points : (N, D)
    values : (N,) categorical labels
    xi : accepted for API consistency with cat_esi_griddata/cat_esi_nongriddata;
        not used during CV (held-out fold points serve as query locations)
    classifier : {'knn_pca', 'scikit-learn'}
    k : int, number of CV folds (default 5)
    cv : {"engine", "refit"}, default "engine"
        How the cross-validation is done. ``"engine"`` draws the partitions
        once on all the data and, in each cell, predicts the held-out data
        with the classifier trained on the cell's other data, as
        :func:`~spatialize.gs.esi.esi_hparams_search` does; leave-one-out
        (``k=-1``) and the session's empty-cell policies apply.
        ``"refit"`` trains the whole ensemble again on the data outside each
        fold, its partitions drawn on them, and predicts the fold. The two
        schemes can select different configurations.
    p_process, data_cond
        As in :func:`cat_esi_griddata`.
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
    if kwargs.get("seed") is None:
        kwargs["seed"] = np.random.randint(1000, 10000)          # ESI partitioning seed
    if kwargs.get("folding_seed") is None:
        kwargs["folding_seed"] = np.random.randint(1000, 10000)  # KFold shuffle seed

    log_message(logging.logger.debug("cat_esi_hparams_search: building parameter grid"))

    points = np.asarray(points)
    values = np.asarray(values)

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

    def combo_for(param_set):
        combo = {
            "classifier": kwargs["classifier"],
            "n_partitions": param_set["n_partitions"],
            "alpha": param_set["alpha"],
            "p_process": kwargs["p_process"],
            "data_cond": kwargs["data_cond"],
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
        return combo

    def score(true, pred):
        """The score on the data that got a category, with the share of those that did not."""
        pred = np.asarray(pred, dtype=object)
        got = np.array([x is not None for x in pred], dtype=bool)
        value = scorer(np.asarray(true)[got], pred[got]) if got.any() else 1.0
        return value, float(np.mean(~got))

    def engine_cv(param_set):
        # one ensemble on all the data; in each cell the held-out data are predicted by the classifier
        # trained on the cell's other data (or on those outside the fold), as in esi_hparams_search
        combo = combo_for(param_set)
        encoded, _, code_to_cat = _make_encoder(values)
        set_fn, clf_fn = get_classifier_fns(combo["classifier"], **{k_: v for k_, v in combo.items() if k_ != "classifier"})
        _, raw = lib_spatialize_facade.run(
            np.float64(points), encoded, np.float64(points), kwargs["p_process"], "custom",
            {"post_creation": set_fn, "estimation": clf_fn}, _engine_alpha(combo), combo["n_partitions"],
            kwargs["seed"], method="loo" if k == n else "kfold", k=0 if k == n else k,
            folding_seed=int(kwargs["folding_seed"]), callback=singleton_null_callback)
        members = _decode_samples(raw, code_to_cat)
        estimation = _resolve_agg_fn(kwargs["agg_function"], kwargs.get("ordinal_order"))(members)
        value, left = score(values, estimation)
        return value, left, float(np.mean(members == None))  # noqa: E711 (object array)

    def refit_cv(param_set):
        # the whole ensemble trained again on the data outside each fold, its partitions drawn on them
        rng = np.random.default_rng(kwargs["folding_seed"])
        kf = KFold(n_splits=k, shuffle=True, random_state=int(rng.integers(0, 9999)))
        all_true, all_pred = [], []
        for tr_idx, te_idx in kf.split(points):
            try:
                estimation, _ = _call_custom_esi(np.ascontiguousarray(points[tr_idx]), values[tr_idx],
                                                 np.ascontiguousarray(points[te_idx]), **combo_for(param_set))
                all_true.append(values[te_idx])
                all_pred.append(estimation)
            except Exception:
                pass  # skip failed folds
        if not all_true:
            warnings.warn(f"All folds failed for param set {param_set}; assigning worst score.",
                          RuntimeWarning, stacklevel=3)
            return 1.0, 1.0, np.nan
        value, left = score(np.concatenate(all_true), np.concatenate(all_pred))
        return value, left, np.nan

    if kwargs["cv"] not in ("engine", "refit"):
        raise ValueError(f"cv must be 'engine' or 'refit'; got {kwargs['cv']!r}")
    run_cv = engine_cv if kwargs["cv"] == "engine" else refit_cv

    results = {}
    n_combos = len(param_grid)
    kwargs["callback"](logging.progress.init(n_combos, 1, desc="categorical hyperparameter search"))
    for i, param_set in enumerate(param_grid):
        results[i] = run_cv(param_set)
        kwargs["callback"](logging.progress.inform())
    kwargs["callback"](logging.progress.stop())

    rows = []
    for i, param_set in enumerate(param_grid):
        row = {"cv_error": results[i][0], "left_out": results[i][1], "nan_members": results[i][2],
               "classifier": kwargs["classifier"], "p_process": kwargs["p_process"],
               "data_cond": kwargs["data_cond"]}
        row.update(param_set)
        rows.append(row)

    result_data = pd.DataFrame(rows)
    from spatialize.gs.esi import scorefunction as sf
    best = {int(np.argmin(result_data["cv_error"].to_numpy()))}
    sf.warn_left_out([(r["left_out"], 0.0 if np.isnan(r["nan_members"]) else r["nan_members"])
                      for _, r in result_data.iterrows()],
                     [", ".join(f"{key}={p[key]}" for key in p) for p in param_grid],
                     best, session.get("max_left_out"))
    return CatESIGridSearchResult(result_data, kwargs["classifier"],
                                  ordinal_order=kwargs.get("ordinal_order"),
                                  sklearn_classifier=kwargs.get("sklearn_classifier"),
                                  random_state=kwargs.get("random_state"),
                                  seed=kwargs.get("seed"))

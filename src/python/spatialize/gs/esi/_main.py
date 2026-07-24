import tempfile
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import ParameterGrid

from spatialize import SpatializeError, logging, GridSearchResult, EstimationResult
import spatialize.gs.esi.aggfunction as af
import spatialize.gs.esi.lossfunction as lf
import spatialize.gs.esi.scorefunction as sf
from spatialize._util import signature_overload
from spatialize._math_util import flatten_grid_data
from spatialize.gs import lib_spatialize_facade, partitioning_process, local_interpolator as li
from spatialize.logging import log_message, default_singleton_callback, singleton_null_callback
from spatialize.viz import plot_colormap_array, PlotStyle


class ESIGridSearchResult(GridSearchResult):
    """Result of a hyperparameter grid search for ESI.

    Wraps the cross-validation error obtained for every combination of
    parameters evaluated by :func:`esi_hparams_search`, and exposes the
    combination with the lowest error as a dict ready to pass to
    :func:`esi_griddata` / :func:`esi_nongriddata` via ``best_params_found``.

    Parameters
    ----------
    search_result_data : pandas.DataFrame
        One row per evaluated parameter combination, with a ``cv_error``
        column plus one column per searched hyperparameter (e.g.
        ``n_partitions``, ``alpha``, ``exponent``) and a
        ``local_interpolator`` column.
    p_process : str
        Partitioning process used during the search, ``"mondrian"`` or
        ``"voronoi"``.

    Attributes
    ----------
    search_result_data : pandas.DataFrame
        The raw per-combination results, as passed in.
    cv_error : pandas.DataFrame
        The ``cv_error`` column of `search_result_data`.
    best_params : pandas.DataFrame
        Subset of `search_result_data` whose ``cv_error`` equals the minimum
        observed value (there may be more than one row in case of ties).
    p_process : str
        Partitioning process used during the search.
    """
    def __init__(self, search_result_data, p_process):
        super().__init__(search_result_data)
        self.p_process = p_process

    def best_result(self, **kwargs):
        """Return the best-scoring parameter combination from the search.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility with
            :meth:`~spatialize.result.GridSearchResult.best_result`.

        Returns
        -------
        dict
            The row of `best_params` with the lowest ``cv_error`` (ties are
            broken by taking the first row), plus ``"result_data_index"``
            (its index in `search_result_data`), ``"agg_function"``
            (:func:`~spatialize.gs.esi.aggfunction.mean`), and ``"p_process"``.
            Ready to pass as ``best_params_found`` to :func:`esi_griddata` or
            :func:`esi_nongriddata`.

            .. warning::
                The returned dict always contains ``"agg_function"`` and
                ``"p_process"``, even if the search itself never varied
                them. Passing this dict as ``best_params_found`` therefore
                silently overrides any ``agg_function`` or ``p_process``
                given at the call site.
        """
        b_param = self.best_params.sort_values(by='cv_error', ascending=True)
        row = pd.DataFrame(b_param.iloc[0]).to_dict(index=True)
        index = list(row.keys())[0]
        result = row[index]
        result.update({"result_data_index": index,
                       "agg_function": af.mean,
                       "p_process": self.p_process})
        
        return result


class ESIParetoResult:
    """Results of a Pareto hyperparameter search for ESI.

    Analogous to :class:`ESIGridSearchResult` but carries two objectives per
    configuration — the encoder error ε̂ and the decoder error R_CV — and
    exposes the Pareto frontier between them.

    Attributes
    ----------
    all_results : list of dict
        One entry per evaluated parameter configuration.  Keys:
        ``"params"`` (dict), ``"epsilon"`` (float, ε̂), ``"decoder_error"``
        (float, R_CV).
    frontier : list of dict
        Non-dominated subset of ``all_results`` (lower is better for both
        objectives).
    p_process : str
        Partitioning process used during optimisation.
    local_interpolator : str
        Local interpolator used during optimisation.
    """

    def __init__(
        self,
        all_results: list,
        p_process: str,
        local_interpolator: str,
    ) -> None:
        """Initialize the Pareto result and compute its frontier.

        Parameters
        ----------
        all_results : list of dict
            One entry per evaluated parameter configuration, each with keys
            ``"params"`` (dict), ``"epsilon"`` (float), ``"decoder_error"``
            (float). Typically produced by
            :func:`esi_pareto_hparams_search`.
        p_process : str
            Partitioning process used during optimisation, ``"mondrian"``
            or ``"voronoi"``.
        local_interpolator : str
            Local interpolator used during optimisation, ``"idw"``,
            ``"kriging"``, or ``"adaptiveidw"``.
        """
        self.all_results = all_results
        self.p_process = p_process
        self.local_interpolator = local_interpolator
        self.frontier = self._compute_frontier(all_results)

    # ------------------------------------------------------------------
    # Public retrieval methods
    # ------------------------------------------------------------------

    def best_result(self, strategy: str = "min_decoder") -> dict:
        """Return the best configuration from the Pareto frontier.

        When the frontier contains a single point the strategy is irrelevant.
        For multi-point frontiers the following strategies are available:

        ``"min_decoder"`` *(default)*
            Select the frontier point with the lowest decoder error R_CV.
        ``"min_epsilon"``
            Select the frontier point with the lowest encoder error ε̂.
        ``"knee"``
            Geometric *knee* point: maximum perpendicular distance from the
            line connecting the two extreme frontier points (in normalised
            objective space).  Represents the most balanced trade-off.
        ``"utopia"``
            Closest frontier point to the *utopia* point (the vector of
            individual objective minima) in normalised objective space.

        Parameters
        ----------
        strategy : str
            One of ``"min_decoder"``, ``"min_epsilon"``, ``"knee"``,
            ``"utopia"``.

        Returns
        -------
        dict
            Configuration dict compatible with :func:`esi_griddata`
            ``best_params_found`` argument.  Contains all parameter keys from
            ``"params"`` plus ``"epsilon"``, ``"decoder_error"``,
            ``"agg_function"``, ``"p_process"``, and ``"local_interpolator"``.

            .. warning::
                All five extra keys are always injected, regardless of
                whether the search varied them. Passing this dict as
                ``best_params_found`` therefore silently overrides any
                ``agg_function``, ``p_process``, or ``local_interpolator``
                given at the call site (``epsilon`` and ``decoder_error``
                are not accepted parameters of :func:`esi_griddata` /
                :func:`esi_nongriddata` and are simply ignored there).

        Raises
        ------
        ValueError
            If the frontier is empty or an unknown strategy is specified.
        """
        if not self.frontier:
            raise ValueError("Pareto frontier is empty — no results were evaluated.")

        r = self.frontier[0] if len(self.frontier) == 1 else self._select_frontier_point(strategy)
        return self._format_result(r)

    def best_for_tau(self, tau: float):
        """Minimum decoder error subject to ε̂ ≤ *tau*.

        Searches across **all** evaluated configurations, not only the
        frontier, so this can return a dominated point when the constraint
        forces a sub-optimal encoder error.

        Parameters
        ----------
        tau : float
            Upper bound on the encoder error ε̂.

        Returns
        -------
        dict or None
            Best feasible configuration dict (same format as
            :meth:`best_result`), or ``None`` if no configuration satisfies
            the constraint.
        """
        feasible = [r for r in self.all_results if r["epsilon"] <= tau]
        if not feasible:
            return None
        return self._format_result(min(feasible, key=lambda r: r["decoder_error"]))

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(self, ax=None, show: bool = True, annotate: bool = False,
             theme: str = 'alges', scatter_color=None, frontier_color=None):
        """Scatter plot of all configs with the Pareto frontier highlighted.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  A new figure is created when ``None``.
        show : bool
            Call ``plt.show()`` when ``True`` (default).
        annotate : bool
            Annotate frontier points with their parameter values.
        theme : str, optional
            Theme name passed to :class:`~spatialize.viz.PlotStyle`.
            Available: ``'whitegrid'``, ``'darkgrid'``, ``'white'``, ``'dark'``,
            ``'alges'``, ``'minimal'``, ``'publication'``.  Default: ``'alges'``.
        scatter_color : str, optional
            Color for the all-configs scatter points.  Defaults to the theme's
            primary color (``style.color``).
        frontier_color : str, optional
            Color for the Pareto frontier line and markers.  Defaults to the
            first color of the theme's ``precision_cmap``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        with PlotStyle(theme=theme) as style:
            _scatter_color  = scatter_color  if scatter_color  is not None else style.color
            _frontier_color = frontier_color if frontier_color is not None else style.precision_cmap(0.15)

            if ax is None:
                _, ax = plt.subplots(figsize=(7, 5))

            eps_all = [r["epsilon"] for r in self.all_results]
            dec_all = [r["decoder_error"] for r in self.all_results]
            ax.scatter(eps_all, dec_all, color=_scatter_color, alpha=0.5, s=40,
                       label="all configs", zorder=2)

            if self.frontier:
                front_sorted = sorted(self.frontier, key=lambda r: r["epsilon"])
                eps_f = [r["epsilon"] for r in front_sorted]
                dec_f = [r["decoder_error"] for r in front_sorted]
                ax.plot(eps_f, dec_f, "o-", color=_frontier_color, lw=2, ms=7,
                        label="Pareto frontier", zorder=3)

                if annotate:
                    for r in front_sorted:
                        label = ", ".join(f"{k}={v}" for k, v in r["params"].items())
                        ax.annotate(
                            label, (r["epsilon"], r["decoder_error"]),
                            textcoords="offset points", xytext=(5, 5), fontsize=7,
                        )

            ax.set_xlabel("Encoder error ε̂")
            ax.set_ylabel("Decoder error R_CV")
            ax.set_title("Encoder–Decoder Pareto Frontier")
            ax.legend()

            if show:
                plt.tight_layout()
                plt.show()

        return ax

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_result(self, r: dict) -> dict:
        """Return a result dict compatible with ESI estimation ``best_params_found``."""
        result = dict(r["params"])
        result.update({
            "epsilon":            r["epsilon"],
            "decoder_error":      r["decoder_error"],
            "agg_function":       af.mean,
            "p_process":          self.p_process,
            "local_interpolator": self.local_interpolator,
        })
        return result

    def _select_frontier_point(self, strategy: str) -> dict:
        if strategy == "min_decoder":
            return min(self.frontier, key=lambda r: r["decoder_error"])
        if strategy == "min_epsilon":
            return min(self.frontier, key=lambda r: r["epsilon"])
        if strategy in ("knee", "utopia"):
            return self._geometric_selection(strategy)
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Choose from 'min_decoder', 'min_epsilon', 'knee', 'utopia'."
        )

    def _geometric_selection(self, strategy: str) -> dict:
        """Knee or utopia selection on the normalised frontier."""
        front = sorted(self.frontier, key=lambda r: r["epsilon"])
        eps = np.array([r["epsilon"] for r in front], dtype=float)
        dec = np.array([r["decoder_error"] for r in front], dtype=float)

        eps_range = eps.max() - eps.min()
        dec_range = dec.max() - dec.min()

        if eps_range == 0.0 or dec_range == 0.0:
            return min(front, key=lambda r: r["decoder_error"])

        eps_n = (eps - eps.min()) / eps_range
        dec_n = (dec - dec.min()) / dec_range

        if strategy == "utopia":
            dist = np.hypot(eps_n, dec_n)
            return front[int(np.argmin(dist))]

        p1 = np.array([eps_n[0], dec_n[0]])
        p2 = np.array([eps_n[-1], dec_n[-1]])
        line_vec = p2 - p1
        line_len = float(np.linalg.norm(line_vec))
        if line_len == 0.0:
            return front[0]
        line_unit = line_vec / line_len

        dists = [
            float(np.linalg.norm(
                np.array([en, dn]) - p1
                - np.dot(np.array([en, dn]) - p1, line_unit) * line_unit
            ))
            for en, dn in zip(eps_n, dec_n)
        ]
        return front[int(np.argmax(dists))]

    @staticmethod
    def _compute_frontier(results: list) -> list:
        """Return non-dominated configurations (lower is better for both objectives)."""
        dominated: set = set()
        for i, r_i in enumerate(results):
            if i in dominated:
                continue
            for j, r_j in enumerate(results):
                if i == j:
                    continue
                if (
                    r_j["epsilon"]       <= r_i["epsilon"] and
                    r_j["decoder_error"] <= r_i["decoder_error"] and
                    (r_j["epsilon"] < r_i["epsilon"] or
                     r_j["decoder_error"] < r_i["decoder_error"])
                ):
                    dominated.add(i)
                    break
        return [r for i, r in enumerate(results) if i not in dominated]


class ESIResult(EstimationResult):
    """Result of an ESI estimation.

    Returned by :func:`esi_griddata` and :func:`esi_nongriddata`, this class
    holds the aggregated estimate together with the full ensemble of `ESI
    samples` (one estimate per random partition) that produced it. It
    provides methods to display the aggregate estimate, inspect individual
    partition scenarios, recompute the estimate with a different aggregation
    function, and calculate precision (error) based on a loss function.

    Parameters
    ----------
    estimation : ndarray
        Aggregated point estimate at each location in `xi`, flattened to
        1D of length $N_{x^*}$.
    esi_samples : ndarray
        ESI samples, shape $(N_{x^*}, m)$ where $m$ is `n_partitions`. One
        column per random partition.
    griddata : bool, optional
        Whether `xi` was originally grid-shaped. Default: ``False``.
    original_shape : tuple, optional
        Shape to reshape flattened results back into when `griddata` is
        ``True``.
    xi : array_like, optional
        The interpolation locations passed to :func:`esi_griddata` or
        :func:`esi_nongriddata` (grid-shaped tuple/array if `griddata` is
        ``True``, otherwise an $N_{x^*} \\times D$ array).
    points : array_like, optional
        The original observed sample locations, used e.g. to calibrate
        ensemble widening against a spatial target variance.
    values : array_like, optional
        The original observed sample values, used together with `points`.

    Attributes
    ----------
    griddata : bool
        Whether `xi` is grid-shaped.
    original_shape : tuple or None
        Shape used to reshape flattened results for grid data.
    points : array_like or None
        The original observed sample locations, if provided.
    values : array_like or None
        The original observed sample values, if provided.
    """
    def __init__(self, estimation, esi_samples, griddata=False, original_shape=None, xi=None,
                 points=None, values=None):
        """Initialize the result with an aggregated estimate and its ESI samples.

        See the class docstring for parameter descriptions.
        """
        super().__init__(estimation, griddata, original_shape, xi=xi, points=points, values=values)
        self._esi_samples = esi_samples
        self._precision = None
        # query coordinates flattened to line up 1:1 with `esi_samples` rows, regardless of
        # whether `xi` is grid-shaped -- used to calibrate ensemble widening
        if xi is None:
            self._xi_flat = None
        elif griddata:
            self._xi_flat, _ = flatten_grid_data(xi)
        else:
            self._xi_flat = xi

    def precision(self, loss_function=lf.mse_loss):
        """Calculate the precision (error) between the estimate and the ESI samples.

        Applies `loss_function` to the estimate and the ensemble of ESI
        samples, then aggregates the result over the ensemble (see
        :meth:`precision_cube` for the non-aggregated per-sample version).

        Parameters
        ----------
        loss_function : callable, optional
            Function ``(estimation, esi_samples) -> array`` used to compute
            the error. Default: :func:`~spatialize.gs.esi.lossfunction.mse_loss`.

        Returns
        -------
        ndarray
            The precision of the estimation, of shape $N_{x^*}$ for
            non-gridded data, or `original_shape` for gridded data.
        """
        log_message(logging.logger.debug(f'applying "{loss_function}" loss function'))
        prec = loss_function(self._estimation, self._esi_samples)

        if self.griddata:
            self._precision = prec.reshape(self.original_shape)
        else:
            self._precision = prec

        return self._precision

    def precision_cube(self, loss_function=lf.mse_cube):
        """Apply a loss function to each ESI sample against the current estimate.

        Unlike :meth:`precision`, the result is not aggregated over the
        ensemble: it returns the full data `cube` of per-sample losses, with
        the same dimensions as the ESI samples cube.

        Parameters
        ----------
        loss_function : callable, optional
            Function ``(estimation, esi_samples) -> array`` used to compute
            the per-sample error. Default:
            :func:`~spatialize.gs.esi.lossfunction.mse_cube`.

        Returns
        -------
        ndarray
            The precision cube, shape $(N_{x^*}, m)$ for non-gridded data,
            or $(d_1, d_2, m)$ for gridded data, where $m$ = `n_partitions`.
        """
        log_message(logging.logger.debug(f'applying "{loss_function}" loss function'))
        prec = loss_function(self._estimation, self._esi_samples)
        if self.griddata:
            return prec.reshape(self.original_shape[0], self.original_shape[1], prec.shape[1])
        else:
            return prec

    def esi_samples(self, raw=False):
        """
        The central concept for dealing with ESI estimation results is the `ESI sample`. 
        In this sense, it should be noted that each random partition delivers an estimate 
        for each of the locations provided in the argument `xi` (for both gridded and 
        non-gridded data). The set of estimates for a particular partition is what in 
        Spatialize is considered an `ESI sample`.

        This method then returns the set of all ESI samples, one for each random partition,
        calculated for the estimation.

        Returns
        -------
        Array
            The ESI samples, an array of dimension $N_{x^*} \\times m$ 
            ($m$ = `n_partitions` in both function :func:`esi_griddata` and 
            :func:`esi_nongriddata`), for non-gridded data, and of dimension 
            $d_1 \\times d_2 \\times m$ for gridded data -- remember that, in this case,
            $d_1 \\times d_2 = N_{x^*}$
        """
        if self.griddata and not raw:
            N = self._esi_samples.shape[1]
            return self._esi_samples.reshape(tuple(list(self.original_shape) + [N]))
        else:
            return self._esi_samples

    def re_estimate(self, agg_function=af.mean):
        """Re-estimate the ESI samples using the given aggregation function.

        Recalculates the final estimate based on the aggregation function
        provided (e.g. by taking the mean of the ESI samples). This method
        updates the internal estimate in place; the next call to
        :meth:`~spatialize.result.EstimationResult.estimation` returns this
        new estimate.

        Parameters
        ----------
        agg_function : callable, optional
            Function ``(esi_samples) -> array`` used to aggregate the
            ensemble into a point estimate. Default:
            :func:`~spatialize.gs.esi.aggfunction.mean`.

        Returns
        -------
        ndarray
            The re-computed estimate (same as calling
            :meth:`~spatialize.result.EstimationResult.estimation`
            afterwards).
        """
        self._estimation = agg_function(self._esi_samples)
        return self.estimation()

    def plot_precision(self, ax=None, w=None, h=None, theme='alges', cmap=None, **imshow_args):
        """Plot the precision of the estimation.

        Computes :meth:`precision` with the default loss function if it has
        not been calculated yet, then renders it as a colormap image.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If ``None``, a new one is created.
        w : int, optional
            Width of the image, used to reshape non-gridded data.
        h : int, optional
            Height of the image, used to reshape non-gridded data.
        theme : str, optional
            Theme name. Available: ``'whitegrid'``, ``'darkgrid'``,
            ``'white'``, ``'dark'``, ``'alges'``, ``'minimal'``,
            ``'publication'``. Default: ``'alges'``.
        cmap : str, optional
            Colormap for the plot. If ``None``, uses the theme default or
            ``'bwr'``.
        **imshow_args
            Additional keyword arguments passed to ``matplotlib``'s
            ``imshow`` via the internal plotting routine.
        """
        if self._precision is None:
            self._precision = self.precision()

        plot_imshow_args = imshow_args.copy()
        if not cmap:
            cmap = plot_imshow_args.pop('cmap', None)
        if 'extent' not in plot_imshow_args:
            extent = self._get_extent()
            if extent is not None:
                plot_imshow_args['extent'] = extent

        with PlotStyle(theme=theme, precision_cmap=cmap) as style:
            self._plot_data(self._precision, ax, w, h, cmap = style.precision_cmap, **plot_imshow_args)
    
    def quick_plot(self, w=None, h=None,
                   theme = 'alges',
                   estimation_cmap = None,
                   precision_cmap = None,
                   show = True,
                   **fig_args):
        """Plot the estimation and its precision side by side.

        Only supported for 2D data; raises for 3D+ data.

        Parameters
        ----------
        w : int, optional
            Width of the images, used to reshape non-gridded data.
        h : int, optional
            Height of the images, used to reshape non-gridded data.
        theme : str, optional
            Theme name. Available: ``'whitegrid'``, ``'darkgrid'``,
            ``'white'``, ``'dark'``, ``'alges'``, ``'minimal'``,
            ``'publication'``. Default: ``'alges'``.
        estimation_cmap : str, optional
            Colormap for the estimation plot. If ``None``, uses the theme
            default or ``'coolwarm'``.
        precision_cmap : str, optional
            Colormap for the precision plot. If ``None``, uses the theme
            default or ``'bwr'``.
        show : bool, optional
            If ``True`` (default), call ``plt.show()`` and return ``None``.
            If ``False``, return the figure instead of displaying it.
        **fig_args
            Additional keyword arguments passed to ``plt.figure()``.

        Returns
        -------
        matplotlib.figure.Figure or None
            ``None`` if `show` is ``True``; otherwise the created figure.

        Raises
        ------
        SpatializeError
            If the data is 3D or higher-dimensional.
        """
        if self.griddata:
            if len(self._xi) > 2:
                raise SpatializeError("quick_plot() for 3D data is not supported")
        else:
            if self._xi.shape[-1] > 2:
                raise SpatializeError("quick_plot() for 3D data is not supported")

        plot_fig_args = fig_args.copy()
        plot_fig_args.setdefault('figsize', (10,8))
        plot_fig_args.setdefault('dpi', 120)

        with PlotStyle(theme=theme, cmap=estimation_cmap, precision_cmap=precision_cmap) as style:
            fig = plt.figure(**plot_fig_args)
            gs = fig.add_gridspec(1, 2, wspace=0.45)
            ax1, ax2 = gs.subplots()

            ax1.set_title('Estimation')
            self.plot_estimation(ax1, w=w, h=h, theme=None, cmap=style.cmap)
            ax1.set_aspect('equal')

            ax2.set_title('Precision')
            self.plot_precision(ax2, w=w, h=h, theme=None, cmap=style.precision_cmap)
            ax2.set_aspect('equal')

        if show:
            plt.show()
        else:
            return fig

    def preview_esi_samples(self, n_imgs=9, n_cols=3, title_prefix="ESI sample", title=None,
                            figsize=(10, 10), dpi=120, theme='alges', cmap=None, **imshow_args):
        """Visualize a preview of the ESI samples as a grid of colormap images.

        Displays a subset of the ESI samples using
        :func:`~spatialize.viz.plot_colormap_array`, laid out in a grid where
        each image corresponds to one ESI sample (i.e. one random
        partition's estimate).

        Parameters
        ----------
        n_imgs : int, optional
            Number of ESI samples (images) to display. Default: ``9``.
        n_cols : int, optional
            Number of columns in the grid layout. Default: ``3``.
        title_prefix : str, optional
            Prefix added to each subplot title (e.g., "ESI sample 1", "ESI
            sample 2"). Default: ``"ESI sample"``.
        title : str, optional
            Title for the entire plot.
        figsize : tuple, optional
            Width, height of the figure in inches. Default: ``(10, 10)``.
        dpi : int, optional
            Resolution of the figure in dots per inch. Default: ``120``.
        theme : str, optional
            Theme name. Available: ``'whitegrid'``, ``'darkgrid'``,
            ``'white'``, ``'dark'``, ``'alges'``, ``'minimal'``,
            ``'publication'``. Default: ``'alges'``.
        cmap : str, optional
            Colormap for the plot. If ``None``, uses the theme default or
            ``'coolwarm'``.
        **imshow_args
            Additional keyword arguments passed to
            :func:`~spatialize.viz.plot_colormap_array`.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the grid of ESI sample images.
        """
        # Retrieve cmap if specified within imshow_args
        plot_imshow_args = imshow_args.copy()
        if not cmap:
            cmap = plot_imshow_args.pop('cmap', None)

        with PlotStyle(theme=theme, cmap=cmap) as style:
            return plot_colormap_array(
                self.esi_samples(raw=True), 
                n_imgs=n_imgs,
                n_cols=n_cols, 
                norm_lims=True,
                xi_locations=self._xi_flat,
                reference_map=self.estimation(),
                title_prefix=title_prefix,
                title=title,
                figsize=figsize,
                dpi=dpi,
                cmap=style.cmap,
                **plot_imshow_args
                )

# ============================================= PUBLIC API ==========================================================
@signature_overload(pivot_arg=("local_interpolator", li.IDW, "local interpolator"),
                    common_args={"k": 10,
                                 "griddata": False,
                                 "p_process": partitioning_process.MONDRIAN,  # partitioning process
                                 "data_cond": [True, False],  # whether to condition the partitioning process on samples
                                 # -- valid only when 'p_process' is 'voronoi'.
                                 "n_partitions": [100],
                                 "alpha": list(np.flip(np.arange(0.70, 0.90, 0.01))),
                                 "scoring": sf.mae,
                                 "seed": np.random.randint(1000, 10000),
                                 "folding_seed": np.random.randint(1000, 10000),
                                 "callback": default_singleton_callback,
                                 },
                    specific_args={
                        li.IDW: {"exponent": list(np.arange(1.0, 15.0, 1.0))},
                        li.KRIGING: {"model": ["spherical", "exponential", "cubic", "gaussian"],
                                     "nugget": [0.0, 0.5, 1.0],
                                     "range": [10.0, 50.0, 100.0, 200.0],
                                     "sill": [0.9, 1.0, 1.1]},
                        li.ADAPTIVE_IDW: {"metric": ["mae"], "parallelize": False}
                    })
def esi_hparams_search(points, values, xi, **kwargs):
    """Perform a k-fold (or leave-one-out) cross-validation hyperparameter search for ESI.

    Evaluates ESI over the Cartesian product of the given hyperparameter
    lists, scoring each combination by cross-validation error, and returns
    an :class:`ESIGridSearchResult` from which the best combination can be
    extracted and fed straight into :func:`esi_griddata` /
    :func:`esi_nongriddata`.

    Parameters
    ----------
    points : array_like
        The input points. Contains the coordinates of known data points.
        This is an $N_s \\times D$ array, where $N_s$ is the number of data
        points, and $D$ is the number of dimensions.
    values : array_like
        The input values associated with each point in `points`. This must
        be a 1D array of length $N_s$.
    xi : array_like
        The interpolation points used for cross-validation. If the data are
        gridded (``griddata=True``), they correspond to an array of grids of
        $D$ components, each with the dimensions of one of the grid faces,
        $d_1 \\times d_2 = N_{x^*}$, as returned by ``numpy.mgrid`` in
        Numpy, or ``meshgrid`` in Matlab or R. If the data are not gridded,
        they are simply an $N_{x^*} \\times D$ array of locations. In both
        cases, $D$ coincides with the dimensionality of `points`.

    Other Parameters
    ----------------
    local_interpolator : str, optional
        Local interpolator to search over: ``"idw"`` (default),
        ``"kriging"``, or ``"adaptiveidw"``. Determines which of the
        interpolator-specific parameters below are accepted.
    k : int, optional
        Number of cross-validation folds. If `k` equals the number of
        points or is ``-1``, leave-one-out (LOO) cross-validation is used
        instead of k-fold. Default: ``10``.
    griddata : bool, optional
        Whether `xi` is grid-shaped (see `xi` above). Default: ``False``.
    p_process : str, optional
        Partitioning process: ``"mondrian"`` (default) or ``"voronoi"``.
    data_cond : list of bool, optional
        Whether to condition the partitioning process on the data samples;
        only used when `p_process` is ``"voronoi"``. Default: ``[True,
        False]``.
    n_partitions : list of int, optional
        Candidate ensemble sizes (number of spatial partitions) to search.
        Default: ``[100]``.
    alpha : list of float, optional
        Candidate partition-granularity values to search, each in ``[0,
        1)``; higher values produce smaller partition cells. Default:
        descending values from ``0.90`` to ``0.70``.
    scoring : callable, optional
        Cross-validation scoring function ``(true_values, esi_samples) ->
        float`` used to rank parameter combinations, e.g.
        :func:`~spatialize.gs.esi.scorefunction.mae`,
        :func:`~spatialize.gs.esi.scorefunction.neg_log_likelihood`, or
        :func:`~spatialize.gs.esi.scorefunction.crps`. Distribution-based
        scorers (`neg_log_likelihood`, `crps`) are automatically replaced
        with `mae` and a warning is issued if the smallest candidate in
        `n_partitions` is below 30. Default:
        :func:`~spatialize.gs.esi.scorefunction.mae`.
    seed : int, optional
        Random seed used for partitioning during estimation. Default: a
        random integer in ``[1000, 10000)``.
    folding_seed : int, optional
        Random seed used to assign points to folds when `k`-fold (not LOO)
        cross-validation is used. Default: a random integer in ``[1000,
        10000)``.
    callback : callable, optional
        Progress-reporting callback invoked as the search runs. Default:
        :func:`~spatialize.logging.default_singleton_callback`.
    exponent : list of float, optional
        *(IDW only)* Candidate IDW distance-decay exponents to search.
        Default: ``[1.0, 2.0, ..., 14.0]``.
    model : list of str, optional
        *(Kriging only)* Candidate variogram models to search, from
        ``"spherical"``, ``"exponential"``, ``"cubic"``, ``"gaussian"``.
    nugget : list of float, optional
        *(Kriging only)* Candidate variogram nugget values to search.
    range : list of float, optional
        *(Kriging only)* Candidate variogram range values to search.
    sill : list of float, optional
        *(Kriging only)* Candidate variogram sill values to search.
    metric : list of str, optional
        *(Adaptive IDW only)* Candidate local-optimization metrics to
        search. Default: ``["mae"]``.
    parallelize : bool, optional
        *(Adaptive IDW only)* Whether to parallelize the per-cell parameter
        optimization. Default: ``False``.

    Returns
    -------
    ESIGridSearchResult
        The grid search result, wrapping the cross-validation error for
        every evaluated combination and exposing the best one via
        :meth:`ESIGridSearchResult.best_result`.

    Examples
    --------
    .. code-block:: python

        from spatialize.gs.esi import esi_hparams_search, esi_griddata

        search_result = esi_hparams_search(
            points, values, (grid_x, grid_y),
            griddata=True,
            local_interpolator="idw",
            k=10,
            exponent=[1.0, 2.0, 3.0, 4.0],
            alpha=[0.7, 0.8, 0.9],
            n_partitions=[100, 300, 500],
        )

        result = esi_griddata(points, values, (grid_x, grid_y),
                              local_interpolator="idw",
                              n_partitions=100      # overwritten at call
                              best_params_found=search_result.best_result())
    """
    log_message(logging.logger.debug(f"searching best params ..."))

    method, k = "kfold", kwargs["k"]
    if k == points.shape[0] or k == -1:
        method = "loo"

    # get the cross validation function
    cross_validate = lib_spatialize_facade.get_operator(points, kwargs["local_interpolator"],
                                                        method, kwargs["p_process"])

    grid = {"n_partitions": kwargs["n_partitions"],
            "alpha": kwargs["alpha"]}

    if kwargs["p_process"] == partitioning_process.VORONOI:
        grid["data_cond"] = kwargs["data_cond"]

    if kwargs["local_interpolator"] == li.IDW:
        grid["exponent"] = kwargs["exponent"]

    if kwargs["local_interpolator"] == li.KRIGING:
        grid["model"] = kwargs["model"]
        grid["nugget"] = kwargs["nugget"]
        grid["range"] = kwargs["range"]
        grid["sill"] = kwargs["sill"]

    if kwargs["local_interpolator"] == li.ADAPTIVE_IDW:
        grid["metric"] = kwargs["metric"]

    # get the actual parameter grid
    param_grid = ParameterGrid(grid)

    _distribution_scorers = (sf.neg_log_likelihood, sf.crps)
    if kwargs["scoring"] in _distribution_scorers and min(kwargs["n_partitions"]) < 30:
        import warnings
        warnings.warn(
            f"'{kwargs['scoring'].__name__}' requires at least 30 partitions to fit a reliable "
            f"distribution, but the smallest value in 'n_partitions' is {min(kwargs['n_partitions'])}. "
            f"The scoring function has been replaced with 'mae'. "
            f"To use '{kwargs['scoring'].__name__}', set all values in 'n_partitions' to >= 30.",
            UserWarning,
            stacklevel=2,
        )
        kwargs["scoring"] = sf.mae

    if isinstance(xi, tuple):
        p_xi = deepcopy(xi)
    else:
        p_xi = xi.copy()

    if kwargs["griddata"]:
        p_xi, _ = flatten_grid_data(xi)

    # run the scenarios
    results = {}

    def run_scenario(i):
        param_set = param_grid[i].copy()        # dictionary with set 'i' of parameters to evaluate
        param_set["local_interpolator"] = kwargs["local_interpolator"]
        param_set["seed"] = kwargs["seed"]
        param_set["callback"] = singleton_null_callback
        param_set["p_process"] = kwargs["p_process"]

        if kwargs["p_process"] == partitioning_process.MONDRIAN:
            param_set["data_cond"] = True

        if kwargs["local_interpolator"] == li.ADAPTIVE_IDW:
            param_set["parallelize"] = kwargs["parallelize"]

        l_args = build_arg_list(points, values, p_xi, param_set)
        if method == "kfold":
            l_args.insert(-2, k)
            l_args.insert(-2, kwargs["folding_seed"])

        _, cv = cross_validate(*l_args)     # returns esi samples for the input data

        results[i] = kwargs["scoring"](values, cv)

        kwargs["callback"](logging.progress.inform())

    it = range(len(param_grid))
    kwargs["callback"](logging.progress.init(len(param_grid), 1))
    for i in it:
        run_scenario(i)
    kwargs["callback"](logging.progress.stop())

    # create a dataframe with all results
    result_data = pd.DataFrame(columns=list(grid.keys()) + ["cv_error"])
    c = 0
    for idx, v in results.items():
        d = {"cv_error": v,
             "local_interpolator": kwargs["local_interpolator"],
             }
        d.update(param_grid[idx])
        if not result_data.empty:
            result_data = pd.concat([result_data, pd.DataFrame(d, index=[c])])
        else:
            result_data = pd.DataFrame(d, index=[c])
        c += 1
    return ESIGridSearchResult(result_data, kwargs["p_process"])


def esi_griddata(points, values, xi, **kwargs):
    """
    Perform ESI estimation for grid data. This is the function used to make an estimate 
    with ESI in the case of sample data and unmeasured locations that are on a grid.

    Parameters
    ----------
    points :  Array of input data points
         The input points. Contains the coordinates of known data points. 
         This is an $N_s \\times D$ array, where $N_s$ is the number of data points, and
         $D$ is the number of dimensions.
    values : Array of values corresponding to input data points.
         The input values associated with each point in points. This must
         be a 1D array of length $N_s$. 
    xi : Array of points where estimation is desired.
         The interpolation points. If the data are gridded, they correspond to an 
         array of grids of $D$ components, each with the dimensions of one of the grid
         faces, $d_1 \\times d_2 = N_{x^*}$, where $N_{x^*}$ is the total number of 
         unmeasured locations to estimate. Each component of this array represents the
         coordinate matrix on the corresponding axis, as returned by the functions 
         ``numpy.mgrid`` in Numpy, or ``meshgrid`` in Matlab or R.

         If the data are not gridded, they are simply the locations at which to evaluate 
         the interpolation. It is then an $N_{x^*} \\times D$ array.

         In both cases, $D$ is the dimensionality of each location, which coincides with the
         dimensionality of the ``points``.

    Other Parameters
    ----------------
    local_interpolator : {"idw", "kriging", "adaptiveidw"}, optional
         Which local interpolator to use within each partition cell.
         Determines which of the interpolator-specific parameters below are
         accepted. Default: ``"idw"``.
    n_partitions : int, optional
         Number of spatial partitions in the ensemble. Default: ``500``.
    p_process : {"mondrian", "voronoi"}, optional
         Spatial partitioning process used to build the ensemble. Default:
         ``"mondrian"``.
    data_cond : bool, optional
         Whether to condition the partitioning process on the sample
         points. Valid only when ``p_process="voronoi"``. Default: ``True``.
    alpha : float, optional
         Controls partition size, in ``[0, 1]``; higher values produce
         smaller partitions. Default: ``0.8``.
    agg_function : callable, optional
         Function used to aggregate the ensemble of local estimates into a
         single value per location. Default:
         :func:`~spatialize.gs.esi.aggfunction.mean`.
    seed : int, optional
         Random seed for the partitioning process. Default: a random
         integer in ``[1000, 10000)``.
    callback : callable, optional
         Callback used to report estimation progress. Default:
         :func:`~spatialize.logging.default_singleton_callback`.
    best_params_found : dict or None, optional
         Parameter dict typically obtained from
         :meth:`ESIGridSearchResult.best_result` or
         :meth:`ESIParetoResult.best_result`. When given, every key it
         contains **overrides** the corresponding argument passed at the
         call site, with one exception: ``n_partitions`` is ignored if
         present -- the value passed at the call site (or its default of
         ``500``) is used instead. This is intentional: it lets you run the
         hyperparameter search cheaply with few partitions and then
         estimate with many. Default: ``None``.

         .. warning::
             :meth:`ESIGridSearchResult.best_result` always injects
             ``agg_function`` (:func:`~spatialize.gs.esi.aggfunction.mean`)
             and ``p_process`` into the dict it returns, so passing it
             silently overrides any ``agg_function`` or ``p_process`` given
             at the call site. :meth:`ESIParetoResult.best_result` injects
             those same two keys plus ``local_interpolator``, ``epsilon``,
             and ``decoder_error``.
    exponent : float, optional
         IDW distance-decay exponent. Only used when
         ``local_interpolator="idw"``. Default: ``2.0``.
    model : {"spherical", "exponential", "cubic", "gaussian"}, optional
         Variogram model. Only used when ``local_interpolator="kriging"``.
         Default: ``"spherical"``.
    nugget : float, optional
         Variogram nugget. Only used when ``local_interpolator="kriging"``.
         Default: ``0.1``.
    range : float, optional
         Variogram range. Only used when ``local_interpolator="kriging"``.
         Default: ``5000.0``.
    sill : float, optional
         Variogram sill. Only used when ``local_interpolator="kriging"``.
         Default: ``1.0``.
    metric : str, optional
         Error metric used for the per-cell LOO parameter optimization.
         Only used when ``local_interpolator="adaptiveidw"``. Default:
         ``"mae"``.
    parallelize : bool, optional
         Whether to parallelize the per-cell parameter optimization. Only
         used when ``local_interpolator="adaptiveidw"``. Default: ``False``.

    Returns
    -------
    ESIResult
         The estimation result.

    Notes
    -----
    Not every local interpolator is available at every dimensionality:
    2D data supports IDW, Kriging, Adaptive IDW, and Voronoi partitioning;
    3D data supports IDW, Kriging, and Adaptive IDW (Mondrian partitioning
    only); 4D and 5D data support IDW only.

    See Also
    --------
    esi_hparams_search : Grid search that produces a ``best_params_found``
        dict for this function.
    esi_pareto_hparams_search : Pareto-frontier search that produces a
        ``best_params_found`` dict for this function.

    Examples
    --------
    .. highlight:: python
    .. code-block:: python

        esi_griddata(points, values, (grid_x, grid_y),
                 local_interpolator="idw",
                 p_process="mondrian",
                 data_cond=False,
                 exponent=1.0,
                 n_partitions=500, alpha=0.985,
                 agg_function=af.mean)

    """
    ng_xi, original_shape = flatten_grid_data(xi)
    estimation, esi_samples = _call_libspatialize(points, values, ng_xi, **kwargs)
    return ESIResult(estimation, esi_samples, griddata=True, original_shape=original_shape, xi=xi,
                     points=points, values=values)


def esi_nongriddata(points, values, xi, **kwargs):
    """
    Perform ESI estimation for non-grid data. This function generates an estimate in ESI space,
    from a set of sample points (i.e. measured locations), at a set of unmeasured points at 
    arbitrary locations in space.

    Parameters
    ----------
    points :  Array of input data points
         The input points. Contains the coordinates of known data points. 
         This is an $N_s \\times D$ array, where $N_s$ is the number of data points, and
         $D$ is the number of dimensions.
    values : Array of values corresponding to input data points.
         The input values associated with each point in points. This must
         be a 1D array of length $N_s$. 
    xi : Array of points where estimation is desired.
         The interpolation points. If the data are gridded, they correspond to an 
         array of grids of $D$ components, each with the dimensions of one of the grid
         faces, $d_1 \\times d_2 = N_{x^*}$, where $N_{x^*}$ is the total number of 
         unmeasured locations to estimate. Each component of this array represents the
         coordinate matrix on the corresponding axis, as returned by the functions 
         ``numpy.mgrid`` in Numpy, or ``meshgrid`` in Matlab or R.

         If the data are not gridded, they are simply the locations at which to evaluate 
         the interpolation. It is then an $N_{x^*} \\times D$ array.

         In both cases, $D$ is the dimensionality of each location, which coincides with the
         dimensionality of the ``points``.

    Other Parameters
    ----------------
    local_interpolator : {"idw", "kriging", "adaptiveidw"}, optional
         Which local interpolator to use within each partition cell.
         Determines which of the interpolator-specific parameters below are
         accepted. Default: ``"idw"``.
    n_partitions : int, optional
         Number of spatial partitions in the ensemble. Default: ``500``.
    p_process : {"mondrian", "voronoi"}, optional
         Spatial partitioning process used to build the ensemble. Default:
         ``"mondrian"``.
    data_cond : bool, optional
         Whether to condition the partitioning process on the sample
         points. Valid only when ``p_process="voronoi"``. Default: ``True``.
    alpha : float, optional
         Controls partition size, in ``[0, 1]``; higher values produce
         smaller partitions. Default: ``0.8``.
    agg_function : callable, optional
         Function used to aggregate the ensemble of local estimates into a
         single value per location. Default:
         :func:`~spatialize.gs.esi.aggfunction.mean`.
    seed : int, optional
         Random seed for the partitioning process. Default: a random
         integer in ``[1000, 10000)``.
    callback : callable, optional
         Callback used to report estimation progress. Default:
         :func:`~spatialize.logging.default_singleton_callback`.
    best_params_found : dict or None, optional
         Parameter dict typically obtained from
         :meth:`ESIGridSearchResult.best_result` or
         :meth:`ESIParetoResult.best_result`. When given, every key it
         contains **overrides** the corresponding argument passed at the
         call site, with one exception: ``n_partitions`` is ignored if
         present -- the value passed at the call site (or its default of
         ``500``) is used instead. This is intentional: it lets you run the
         hyperparameter search cheaply with few partitions and then
         estimate with many. Default: ``None``.

         .. warning::
             :meth:`ESIGridSearchResult.best_result` always injects
             ``agg_function`` (:func:`~spatialize.gs.esi.aggfunction.mean`)
             and ``p_process`` into the dict it returns, so passing it
             silently overrides any ``agg_function`` or ``p_process`` given
             at the call site. :meth:`ESIParetoResult.best_result` injects
             those same two keys plus ``local_interpolator``, ``epsilon``,
             and ``decoder_error``.
    exponent : float, optional
         IDW distance-decay exponent. Only used when
         ``local_interpolator="idw"``. Default: ``2.0``.
    model : {"spherical", "exponential", "cubic", "gaussian"}, optional
         Variogram model. Only used when ``local_interpolator="kriging"``.
         Default: ``"spherical"``.
    nugget : float, optional
         Variogram nugget. Only used when ``local_interpolator="kriging"``.
         Default: ``0.1``.
    range : float, optional
         Variogram range. Only used when ``local_interpolator="kriging"``.
         Default: ``5000.0``.
    sill : float, optional
         Variogram sill. Only used when ``local_interpolator="kriging"``.
         Default: ``1.0``.
    metric : str, optional
         Error metric used for the per-cell LOO parameter optimization.
         Only used when ``local_interpolator="adaptiveidw"``. Default:
         ``"mae"``.
    parallelize : bool, optional
         Whether to parallelize the per-cell parameter optimization. Only
         used when ``local_interpolator="adaptiveidw"``. Default: ``False``.

    Returns
    -------
    ESIResult
         The estimation result.

    Notes
    -----
    Not every local interpolator is available at every dimensionality:
    2D data supports IDW, Kriging, Adaptive IDW, and Voronoi partitioning;
    3D data supports IDW, Kriging, and Adaptive IDW (Mondrian partitioning
    only); 4D and 5D data support IDW only.

    See Also
    --------
    esi_hparams_search : Grid search that produces a ``best_params_found``
        dict for this function.
    esi_pareto_hparams_search : Pareto-frontier search that produces a
        ``best_params_found`` dict for this function.
    """
    estimation, esi_samples = _call_libspatialize(points, values, xi, **kwargs)
    return ESIResult(estimation, esi_samples, xi=xi, points=points, values=values)


@signature_overload(
    pivot_arg=("local_interpolator", li.IDW, "local interpolator"),
    common_args={
        "p_process":            partitioning_process.MONDRIAN,
        "scoring":              sf.neg_log_likelihood,
        "k":                    5,
        "n_partitions":         [100, 200, 300],
        "alpha":                list(np.flip(np.arange(0.70, 0.90, 0.05))),
        "seed":                 np.random.randint(1000, 10000),
        "folding_seed":         np.random.randint(1000, 10000),
        "pair_strategy":        "max_min",
        "point_model_name":     "kde",
        "nan_model_name":       "ignore",
        "support_sample_size":  500,
        "callback":             default_singleton_callback,
    },
    specific_args={
        li.IDW:          {"exponent":  list(np.arange(1.0, 5.0, 1.0))},
        li.KRIGING:      {
            "model":  ["spherical", "exponential"],
            "nugget": [0.0, 0.5],
            "range":  [50.0, 200.0],
            "sill":   [0.9, 1.0],
        },
        li.ADAPTIVE_IDW: {"metric": ["mae"], "parallelize": False},
    },
)
def esi_pareto_hparams_search(points, values, **kwargs):
    """Pareto hyperparameter optimisation for ESI.

    Jointly minimises the encoder error ε̂ (empirical robustness bound via
    fitted density models) and the decoder error R_CV (cross-validation
    score), returning an :class:`ESIParetoResult` that exposes the Pareto
    frontier and several strategies for selecting a single best configuration.

    Parameters
    ----------
    points : array-like, shape (n, d)
        Training sample locations.
    values : array-like, shape (n,)
        Training sample values.
    local_interpolator : str, optional
        ``"idw"`` (default), ``"kriging"``, or ``"adaptiveidw"``.
    p_process : str, optional
        Partitioning process.  Default: ``"mondrian"``.
    scoring : str or callable, optional
        Decoder scoring function: ``"nll"`` (default), ``"crps"``,
        ``"rmse"``, ``"mae"``, or any callable
        ``(true_values, esi_samples) → float``.
    k : int or str, optional
        Number of CV folds or ``"loo"`` / ``-1`` for leave-one-out.
        Default: ``5``.
    n_partitions : list of int, optional
        Ensemble sizes to search.  Default: ``[100, 200, 300]``.
    alpha : list of float, optional
        Partition granularities to search.
    seed : int, optional
        Random seed (shared by encoder and decoder).
    folding_seed : int, optional
        Secondary seed for k-fold assignment.
    pair_strategy : str, optional
        Pair-selection for the robustness bound: ``"max_min"`` (default) or
        ``"exhaustive"``.
    point_model_name : str, optional
        Density model for fitted KL: ``"kde"`` (default), ``"emm"``,
        ``"vim"``.
    nan_model_name : str, optional
        NaN strategy: ``"ignore"`` (default) or ``"replace"``.
    support_sample_size : int, optional
        KL evaluation grid resolution (default 500).
    callback : callable, optional
        Progress callback.
    exponent : list of float, optional *(IDW only)*
        IDW exponents to search.
    model, nugget, range, sill : list *(Kriging only)*
        Variogram parameters to search.
    metric : list of str, optional *(Adaptive IDW only)*

    Returns
    -------
    ESIParetoResult

    Examples
    --------
    .. code-block:: python

        import spatialize.gs.esi.scorefunction as sf
        from spatialize.gs.esi import esi_pareto_hparams_search

        result = esi_pareto_hparams_search(
            points, values,
            local_interpolator="idw",
            n_partitions=[100, 200, 300],
            alpha=[0.75, 0.80, 0.85],
            exponent=[1.0, 2.0],
            scoring=sf.neg_log_likelihood,
            k=5,
        )

        # Best balanced trade-off
        best = result.best_result(strategy="knee")

        # Constrained: encoder error must be ≤ 0.5
        constrained = result.best_for_tau(tau=0.5)

        result.plot()
    """
    param_grid = {
        "n_partitions": kwargs["n_partitions"],
        "alpha":        kwargs["alpha"],
    }
    if kwargs["local_interpolator"] == li.IDW:
        param_grid["exponent"] = kwargs["exponent"]
    elif kwargs["local_interpolator"] == li.KRIGING:
        param_grid["model"]  = kwargs["model"]
        param_grid["nugget"] = kwargs["nugget"]
        param_grid["range"]  = kwargs["range"]
        param_grid["sill"]   = kwargs["sill"]
    elif kwargs["local_interpolator"] == li.ADAPTIVE_IDW:
        param_grid["metric"] = kwargs["metric"]

    # Fixed (non-searchable) interpolator kwargs — execution flags that are
    # forwarded as-is to every ESI call, not iterated over in the grid.
    fixed_interp_kwargs = {}
    if kwargs["local_interpolator"] == li.ADAPTIVE_IDW:
        fixed_interp_kwargs["parallelize"] = kwargs["parallelize"]

    _distribution_scorers = (sf.neg_log_likelihood, sf.crps)
    if kwargs["scoring"] in _distribution_scorers and min(kwargs["n_partitions"]) < 30:
        import warnings
        warnings.warn(
            f"'{kwargs['scoring'].__name__}' requires at least 30 partitions to fit a reliable "
            f"distribution, but the smallest value in 'n_partitions' is {min(kwargs['n_partitions'])}. "
            f"The scoring function has been replaced with 'mae'. "
            f"To use '{kwargs['scoring'].__name__}', set all values in 'n_partitions' to >= 30.",
            UserWarning,
            stacklevel=2,
        )
        kwargs["scoring"] = sf.mae

    from spatialize.gs.esi.pareto import ParetoOptimizer
    optimizer = ParetoOptimizer(
        param_grid          = param_grid,
        local_interpolator  = kwargs["local_interpolator"],
        p_process           = kwargs["p_process"],
        scoring             = kwargs["scoring"],
        k                   = kwargs["k"],
        seed                = kwargs["seed"],
        folding_seed        = kwargs["folding_seed"],
        pair_strategy       = kwargs["pair_strategy"],
        point_model_name    = kwargs["point_model_name"],
        nan_model_name      = kwargs["nan_model_name"],
        support_sample_size = kwargs["support_sample_size"],
        fixed_interp_kwargs = fixed_interp_kwargs,
        callback            = kwargs["callback"],
    )
    all_results = optimizer.fit(points, values)
    return ESIParetoResult(all_results, kwargs["p_process"], kwargs["local_interpolator"])


# =========================================== END of PUBLIC API ======================================================
@signature_overload(pivot_arg=("local_interpolator", li.IDW, "local interpolator"),
                    common_args={"n_partitions": 500,
                                 "p_process": partitioning_process.MONDRIAN,  # partitioning process
                                 "data_cond": True,  # whether to condition the partitioning process on samples
                                 # -- valid only when ‘p_process’ is ‘voronoi’.
                                 "alpha": 0.8,
                                 "agg_function": af.mean,
                                 "seed": np.random.randint(1000, 10000),
                                 "callback": default_singleton_callback,
                                 "best_params_found": None
                                 },
                    specific_args={
                        li.IDW: {"exponent": 2.0},
                        li.KRIGING: {"model": "spherical", "nugget": 0.1, "range": 5000.0, "sill": 1.0},
                        li.ADAPTIVE_IDW: {"metric": "mae", "parallelize": False}
                    })
def _call_libspatialize(points, values, xi, **kwargs):
    """
    Call the libspatialize library to perform ESI estimation.

    :param points: The input points.
    :param values: The input values.
    :param xi: The interpolation points.
    :param kwargs: Additional keyword arguments.
    :return: The estimation and ESI samples.
    """
    log_message(logging.logger.debug('calling libspatialize'))

    if not kwargs["best_params_found"] is None:
        kwargs["best_params_found"] = dict(kwargs["best_params_found"])  # copy to avoid mutating caller's dict
        try:
            best = kwargs["best_params_found"]["n_partitions"]
            log_message(logging.logger.debug(f"best number of partitions found: "
                                             f"{best}"))
            del kwargs["best_params_found"]["n_partitions"]  # this param can be overwritten all cases
        except KeyError:
            pass
        log_message(logging.logger.debug(f"using best params found: {kwargs['best_params_found']}"))
        for k in kwargs["best_params_found"]:
            try:
                kwargs[k] = kwargs["best_params_found"][k]
            except KeyError:
                pass

    # get the estimator function
    estimate = lib_spatialize_facade.get_operator(points, kwargs["local_interpolator"],
                                                  "estimate", kwargs["p_process"])

    # get the argument list
    l_args = build_arg_list(points, values, xi, kwargs)

    # run
    try:
        esi_model, esi_samples = estimate(*l_args)
    except Exception as e:
        raise SpatializeError(e)

    estimation = kwargs["agg_function"](esi_samples)

    return estimation, esi_samples


def _validate_alpha(alpha):
    """Raise ValueError if alpha >= 1; warn if alpha > 0.95."""
    if isinstance(alpha, (list, tuple)):
        for a in alpha:
            _validate_alpha(a)
        return
    if alpha >= 1.0:
        raise ValueError(
            f"alpha must be < 1 (got {alpha}). "
        )
    if alpha > 0.95:
        import warnings
        warnings.warn(
            f"alpha={alpha} is very close to 1. This produces very fine partitions "
            "and may cause slow computation.",
            UserWarning,
            stacklevel=3,
        )


def build_arg_list(points, values, xi, nonpos_args):
    """
    Build the argument list for the libspatialize function.

    :param points: The input points.
    :param values: The input values.
    :param xi: The interpolation points.
    :param nonpos_args: The non-positional arguments.
    :return: The argument list.
    """
    alpha = nonpos_args["alpha"]
    _validate_alpha(alpha)
    if nonpos_args["p_process"] == partitioning_process.VORONOI and not nonpos_args["data_cond"]:
        alpha *= -1

    # add initial common args
    l_args = [np.float32(points), np.float32(values),
              nonpos_args["n_partitions"], alpha, np.float32(xi), nonpos_args["callback"]]

    # add specific args
    if nonpos_args["local_interpolator"] == li.IDW:
        l_args.insert(-2, nonpos_args["exponent"])
        l_args.insert(-2, nonpos_args["seed"])

    if nonpos_args["local_interpolator"] == li.KRIGING:
        l_args.insert(-2, lib_spatialize_facade.get_kriging_model_number(nonpos_args["model"]))
        l_args.insert(-2, nonpos_args["nugget"])
        l_args.insert(-2, nonpos_args["range"])
        l_args.insert(-2, nonpos_args["sill"])
        l_args.insert(-2, nonpos_args["seed"])

    if nonpos_args["local_interpolator"] == li.ADAPTIVE_IDW:
        l_args.insert(-2, nonpos_args["seed"])
        l_args.insert(-2, nonpos_args.get("metric", "mae"))
        l_args.insert(-2, nonpos_args.get("parallelize", False))

    return l_args

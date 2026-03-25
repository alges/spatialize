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
    """
    A class to represent the result of a grid search for ESI.

    :param search_result_data: The search result data.
    :param agg_function_map: The aggregation function map.
    :param p_process: The partitioning process.
    """
    def __init__(self, search_result_data, p_process):
        super().__init__(search_result_data)
        self.p_process = p_process

    def best_result(self, **kwargs):
        """
        Get the best result from the grid search.

        :param kwargs: Additional keyword arguments.
        :return: The best result.
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
    """
    A class to represent the result of an ESI estimation. 

    As a result, this function also returns an object, which is an instance
    of the class :func:`ESIResult`, containing the preliminary estimate according to the provided
    arguments. This class provides a set of methods to display aspects of the result, such as
    the aggregate estimate, the scenarios of the different partitions, or a precision calculation
    based on some loss function.

    """
    def __init__(self, estimation, esi_samples, griddata=False, original_shape=None, xi=None):
        super().__init__(estimation, griddata, original_shape, xi=xi)
        self._esi_samples = esi_samples
        self._precision = None

    def precision(self, loss_function=lf.mse_loss):
        """
        Calculates the precision (or error) between the estimate and the ESI samples using the 
        specified loss function.

        :param loss_function: The loss function to use.
        :return: The precision of the estimation.
        """
        log_message(logging.logger.debug(f'applying "{loss_function}" loss function'))
        prec = loss_function(self._estimation, self._esi_samples)

        if self.griddata:
            self._precision = prec.reshape(self.original_shape)
        else:
            self._precision = prec

        return self._precision

    def precision_cube(self, loss_function=lf.mse_cube):
        """
        It applies a loss (error) function to each ESI sample with respect to the current estimate. 
        The difference with the :func:`precision` method is that it does not aggregate the result 
        over the total calculated losses, returning the total data `cube` whose dimensions are the 
        same as the ESI samples cube.

        :param loss_function: The loss function to use.
        :return: The precision cube of the estimation.
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
        """
        Re-estimate the ESI samples using the given aggregation function.
        It recalculates the final estimate based on the aggregation function provided 
        (e.g. by taking the mean of the ESI samples). This method updates the internal
        estimate and returns the new result. Then, the next time the :func:`estimation` 
        method is called, this is the estimate it will return.

        :param agg_function: The aggregation function to use.
        :return: The re-estimated ESI samples.
        """
        self._estimation = agg_function(self._esi_samples)
        return self.estimation()

    def plot_precision(self, ax=None, w=None, h=None, theme='alges', cmap=None, **imshow_args):
        """
        Plot the precision of the estimation.

        :param ax: The axis to plot on.
        :param w: The width of the plot.
        :param h: The height of the plot.
        :param theme: Theme name. Available: 'whitegrid', 'darkgrid', 'white',
            'dark', 'alges', 'minimal', 'publication'.
        :param cmap: Colormap for the plot. If None, uses theme default or 'bwr'.
        :param imshow_args: Additional imshow arguments to pass to the `_plot_data` function.
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
        """
        Quickly plot the estimation and precision.

        :param w: The width of the plot.
        :param h: The height of the plot.
        :param theme: Theme name. Available: 'whitegrid', 'darkgrid', 'white',
            'dark', 'alges', 'minimal', 'publication'.
        :param estimation_cmap: Colormap for the estimation plot. If None, uses theme default or 'coolwarm'.
        :param precision_cmap: Colormap for the precision plot. If None, uses theme default or 'bwr'.
        :param show: If True (default), call plt.show() and return None. If False, return the figure.
        :param fig_args: Additional figure arguments.
        :return: None if show=True, otherwise the figure.
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
        """
        Visualizes a preview of the ESI samples as a grid of colormap images.

        This method displays a subset of the ESI samples using the `plot_colormap_array` function.
        The ESI samples are visualized in a grid layout, where each image corresponds to one ESI sample.

        :param n_imgs: The number of ESI samples (images) to display. Defaults to 9.
        :param n_cols: The number of columns in the grid layout. Defaults to 3.
        :param title_prefix: A prefix to add to each subplot title (e.g., "ESI sample 1", "ESI sample 2").
        :param title: The title for the entire plot.
        :param figsize: Width, height of the figure in inches. Defaults to (10, 10).
        :param dpi: The resolution of the figure in dots-per-inch. Defaults to 120.
        :param theme: Theme name. Available: 'whitegrid', 'darkgrid', 'white',
            'dark', 'alges', 'minimal', 'publication'.
        :param cmap: Colormap for the plot. If None, uses theme default or 'coolwarm'.
        :param imshow_args: Additional imshow arguments to pass to the `plot_colormap_array` function.
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
                xi_locations=self._xi,
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
    """
    Perform a hyperparameter search for ESI.

    :param points: The input points.
    :param values: The input values.
    :param xi: The interpolation points.
    :param kwargs: Additional keyword arguments.
    :return: The grid search result.
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
    kwargs: dict
         Additional keyword arguments.
    
    Returns
    -------
    The result as :func:`ESIResult`.

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
    return ESIResult(estimation, esi_samples, griddata=True, original_shape=original_shape, xi=xi)


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
    kwargs: dict
         Additional keyword arguments.
    
    Returns
    -------
    ESIResult
        The result as :func:`ESIResult`.
    """
    estimation, esi_samples = _call_libspatialize(points, values, xi, **kwargs)
    return ESIResult(estimation, esi_samples, xi=xi)


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

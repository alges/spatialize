import tempfile
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import ParameterGrid

from spatialize import SpatializeError, logging, GridSearchResult, EstimationResult
import spatialize.gs.esi.aggfunction as af
import spatialize.gs.esi.lossfunction as lf
from spatialize._util import signature_overload, in_notebook
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
    def __init__(self, search_result_data, agg_function_map, p_process):
        super().__init__(search_result_data)
        self.agg_func_map = agg_function_map
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
                       "agg_function": self.agg_func_map[result["agg_func_name"]],
                       "p_process": self.p_process})
        return result


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

        with PlotStyle(theme=theme, precision_cmap=cmap) as style:
            self._plot_data(self._precision, ax, w, h, cmap = style.precision_cmap, **plot_imshow_args)
    
    def quick_plot(self, w=None, h=None,
                   theme = 'alges',
                   estimation_cmap = None,
                   precision_cmap = None,
                   **fig_args):
        """
        Quickly plot the estimation and precision.

        :param w: The width of the plot.
        :param h: The height of the plot.
        :param theme: Theme name. Available: 'whitegrid', 'darkgrid', 'white',
            'dark', 'alges', 'minimal', 'publication'.
        :param estimation_cmap: Colormap for the estimation plot. If None, uses theme default or 'coolwarm'.
        :param precision_cmap: Colormap for the precision plot. If None, uses theme default or 'bwr'.
        :param fig_args: Additional figure arguments.
        :return: The figure.
        """
        if self._xi.shape[1] > 2:
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

            if not in_notebook():
                return fig      # just in case you want to embed it somewhere else

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
                                 # -- valid only when ‘p_process’ is ‘voronoi’.
                                 "n_partitions": [100],
                                 "alpha": list(np.flip(np.arange(0.70, 0.90, 0.01))),
                                 "agg_function": {"mean": af.mean, "median": af.median},
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
                        li.ADAPTIVE_IDW: {}
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

    # get the actual parameter grid
    param_grid = ParameterGrid(grid)

    if isinstance(xi, tuple):
        p_xi = deepcopy(xi)
    else:
        p_xi = xi.copy()

    if kwargs["griddata"]:
        p_xi, _ = flatten_grid_data(xi)

    # run the scenarios
    results = {}

    def run_scenario(i):
        param_set = param_grid[i].copy()
        param_set["local_interpolator"] = kwargs["local_interpolator"]
        param_set["seed"] = kwargs["seed"]
        param_set["callback"] = singleton_null_callback
        param_set["p_process"] = kwargs["p_process"]

        if kwargs["p_process"] == partitioning_process.MONDRIAN:
            param_set["data_cond"] = True

        l_args = build_arg_list(points, values, p_xi, param_set)
        if method == "kfold":
            l_args.insert(-2, k)
            l_args.insert(-2, kwargs["folding_seed"])

        model, cv = cross_validate(*l_args)

        for agg_func_name, agg_func in kwargs["agg_function"].items():
            results[(agg_func_name, i)] = np.nanmean(np.abs(values - agg_func(cv)))

        kwargs["callback"](logging.progress.inform())

    it = range(len(param_grid))
    kwargs["callback"](logging.progress.init(len(param_grid), 1))
    for i in it:
        run_scenario(i)
    kwargs["callback"](logging.progress.stop())

    # create a dataframe with all results
    result_data = pd.DataFrame(columns=list(grid.keys()) + ["cv_error"])
    c = 0
    for k, v in results.items():
        d = {"agg_func_name": k[0],
             "cv_error": v,
             "local_interpolator": kwargs["local_interpolator"],
             }
        d.update(param_grid[k[1]])
        if not result_data.empty:
            result_data = pd.concat([result_data, pd.DataFrame(d, index=[c])])
        else:
            result_data = pd.DataFrame(d, index=[c])
        c += 1
    return ESIGridSearchResult(result_data, kwargs["agg_function"], kwargs["p_process"])


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
    return ESIResult(estimation, esi_samples, griddata=True, original_shape=original_shape)


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
                        li.KRIGING: {"model": 1, "nugget": 0.1, "range": 5000.0, "sill": 1.0},
                        li.ADAPTIVE_IDW: {}
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

    return l_args

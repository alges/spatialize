import multiprocessing

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

import spatialize.gs
from spatialize import SpatializeError, logging, GridSearchResult, EstimationResult
from spatialize._math_util import flatten_grid_data
from spatialize.logging import default_singleton_callback, singleton_null_callback, log_message
from spatialize.gs import lib_spatialize_facade


class IDWGridSearchResult(GridSearchResult):
    """Result of a hyperparameter grid search for plain IDW interpolation.

    Wraps the cross-validation error obtained for every combination of
    `radius` and `exponent` evaluated by :func:`idw_hparams_search`, and
    exposes the best combination as a dict ready to pass to
    :func:`idw_griddata` / :func:`idw_nongriddata` via ``best_params_found``.

    Parameters
    ----------
    search_result_data : pandas.DataFrame
        One row per evaluated parameter combination, with a ``cv_error``
        column plus ``radius`` and ``exponent`` columns.

    Attributes
    ----------
    search_result_data : pandas.DataFrame
        The raw per-combination results, as passed in.
    cv_error : pandas.DataFrame
        The ``cv_error`` column of `search_result_data`.
    best_params : pandas.DataFrame
        Subset of `search_result_data` whose ``cv_error`` equals the minimum
        observed value (there may be more than one row in case of ties).
    """
    def __init__(self, search_result_data):
        """Initialize the grid search result.

        See the class docstring for parameter descriptions.
        """
        super().__init__(search_result_data)

    def best_result(self, optimize_data_usage=False, **kwargs):
        """Return the best-scoring parameter combination from the search.

        Among the rows tied for minimum ``cv_error``, sorts by `radius` to
        break the tie.

        Parameters
        ----------
        optimize_data_usage : bool, optional
            If ``False`` (default), ties are broken by taking the largest
            `radius` (more data used per estimate). If ``True``, the
            smallest `radius` is preferred instead.
        **kwargs
            Unused; accepted for interface compatibility with
            :meth:`~spatialize.result.GridSearchResult.best_result`.

        Returns
        -------
        dict
            The selected row of `best_params`, plus
            ``"result_data_index"`` (its index in `search_result_data`).
            Ready to pass as ``best_params_found`` to :func:`idw_griddata`
            or :func:`idw_nongriddata`.
        """
        b_param = self.best_params.sort_values(by='radius', ascending=optimize_data_usage)
        row = pd.DataFrame(b_param.iloc[0]).to_dict(index=True)
        index = list(row.keys())[0]
        result = row[index]
        result.update({"result_data_index": index})
        return result


class IDWResult(EstimationResult):
    """Result of a plain IDW estimation.

    Returned by :func:`idw_griddata` and :func:`idw_nongriddata`. Adds no
    behaviour of its own; it inherits estimate access, plotting, and
    reshaping from :class:`~spatialize.result.EstimationResult`.
    """
    pass


def idw_hparams_search(points, values, xi,
                       k=10,
                       griddata=False,
                       radius=(0.1, 0.2, 0.5, 0.1, 0.2),
                       exponent=tuple(np.arange(0.8, 1.0, 0.1)),
                       folding_seed=np.random.randint(1000, 10000),
                       callback=default_singleton_callback
                       ):
    """Perform a k-fold (or leave-one-out) cross-validation hyperparameter search for plain IDW.

    Evaluates plain IDW interpolation over the Cartesian product of `radius`
    and `exponent`, scoring each combination by mean absolute cross-validation
    error, and returns an :class:`IDWGridSearchResult` from which the best
    combination can be extracted and fed straight into :func:`idw_griddata` /
    :func:`idw_nongriddata`.

    Parameters
    ----------
    points : array_like
        The input points. Contains the coordinates of known data points.
        This is an $N_s \times D$ array, where $N_s$ is the number of data
        points, and $D$ is the number of dimensions.
    values : array_like
        The input values associated with each point in `points`. This must
        be a 1D array of length $N_s$.
    xi : array_like
        The interpolation points used for cross-validation. If the data are
        gridded (``griddata=True``), they correspond to an array of grids of
        $D$ components, each with the dimensions of one of the grid faces,
        as returned by ``numpy.mgrid``. If the data are not gridded, they
        are simply an $N_{x^*} \times D$ array of locations.
    k : int, optional
        Number of cross-validation folds. If `k` equals the number of
        points or is ``-1``, leave-one-out (LOO) cross-validation is used
        instead of k-fold. Default: ``10``.
    griddata : bool, optional
        Whether `xi` is grid-shaped (see `xi` above). Default: ``False``.
    radius : tuple of float, optional
        Candidate `radius` values for the grid search -- the maximum
        distance within which neighboring points contribute to an
        estimate. Default: ``(0.1, 0.2, 0.5, 0.1, 0.2)``.
    exponent : tuple of float, optional
        Candidate IDW distance-decay `exponent` values for the grid search.
        Default: ``tuple(np.arange(0.8, 1.0, 0.1))``.
    folding_seed : int, optional
        Seed for the random number generator used to build the
        cross-validation folds (only used when `k`-fold, not LOO, is
        performed). Default: a random integer in ``[1000, 10000)``.
    callback : callable, optional
        Callback used to report search progress. Default:
        :func:`~spatialize.logging.default_singleton_callback`.

    Returns
    -------
    IDWGridSearchResult
        The grid search results, one row per evaluated ``(radius,
        exponent)`` combination.
    """
    log_message(logging.logger.debug(f"searching best params ..."))

    method = "kfold"
    if k == points.shape[0] or k == -1:
        method = "loo"

    # get the cross validation function
    cross_validate = lib_spatialize_facade.get_operator(points,
                                                        spatialize.gs.local_interpolator.IDW,
                                                        method,
                                                        spatialize.gs.PLAIN_INTERPOLATOR)

    grid = {"radius": radius,
            "exponent": exponent}

    # get the actual parameter grid
    param_grid = ParameterGrid(grid)

    p_xi = xi
    if griddata:
        p_xi, _ = flatten_grid_data(xi)

    # run the scenarios
    results = {}

    def run_scenario(i):
        param_set = param_grid[i].copy()

        l_args = [np.float32(points),
                  np.float32(values),
                  param_set["radius"],
                  param_set["exponent"],
                  singleton_null_callback]

        if method == "kfold":
            l_args.insert(-1, k)
            l_args.insert(-1, folding_seed)

        cv = cross_validate(*l_args)
        results[i] = np.nanmean(np.abs(values - cv))
        callback(logging.progress.inform())

    callback(logging.progress.init(len(param_grid), 1))
    it = range(len(param_grid))
    for i in it:
        run_scenario(i)
    callback(logging.progress.stop())

    # create a dataframe with all results
    result_data = pd.DataFrame(columns=list(grid.keys()) + ["cv_error"])
    for k, v in results.items():
        d = {"cv_error": v}
        d.update(param_grid[k])
        if not result_data.empty:
            result_data = pd.concat([result_data, pd.DataFrame(d, index=[k])])
        else:
            result_data = pd.DataFrame(d, index=[k])

    return IDWGridSearchResult(result_data)


def idw_griddata(points, values, xi, **kwargs):
    """Perform plain IDW interpolation on a regular grid.

    Parameters
    ----------
    points : array_like
        The input points. Contains the coordinates of known data points.
        This is an $N_s \times D$ array, where $N_s$ is the number of data
        points, and $D$ is the number of dimensions.
    values : array_like
        The input values associated with each point in `points`. This must
        be a 1D array of length $N_s$.
    xi : array_like
        The grid where interpolation is desired: an array of grids of $D$
        components, each with the dimensions of one of the grid faces, as
        returned by ``numpy.mgrid``.
    best_params_found : dict or None, optional
        Parameter dict typically obtained from
        :meth:`IDWGridSearchResult.best_result`. When given, it replaces the
        interpolation parameters entirely: only the ``radius`` and
        ``exponent`` keys are read, and any other key in the dict is
        ignored. **Both keys are required** -- a dict missing either one
        raises :exc:`KeyError` (see `Raises` below). The `radius` and
        `exponent` arguments passed explicitly at the call site are
        **silently ignored** when this dict is given; there is no
        per-key fallback to them. Default: ``None``.
    **kwargs
        Additional keyword arguments forwarded to the estimator: `radius`
        and `exponent` (see :func:`idw_hparams_search`), and `callback`.

    Returns
    -------
    IDWResult
        The interpolation results on the grid.

    Raises
    ------
    KeyError
        If `best_params_found` is given but lacks a ``radius`` or an
        ``exponent`` key. This is a bare :exc:`KeyError` raised while
        reading the dict; it is **not** wrapped in
        :exc:`~spatialize.SpatializeError`.

    Notes
    -----
    This is plain IDW, not an ensemble method, so there is no
    ``n_partitions`` concept here and no key of `best_params_found` is
    exempt from being applied.

    A dict returned by ``idw_hparams_search(...).best_result()`` always
    contains both ``radius`` and ``exponent``, so the requirement above
    only bites on hand-built dicts. A dict produced by
    :func:`~spatialize.gs.esi.esi_hparams_search` is **not** compatible
    here: it has no ``radius`` key and will raise :exc:`KeyError`.

    See Also
    --------
    idw_hparams_search : Grid search that produces a compatible
        ``best_params_found`` dict for this function.
    """
    ng_xi, original_shape = flatten_grid_data(xi)
    estimation = _call_libspatialize(points, values, ng_xi, **kwargs)
    return IDWResult(estimation, True, original_shape)


def idw_nongriddata(points, values, xi, **kwargs):
    """Perform plain IDW interpolation at scattered (non-gridded) locations.

    Parameters
    ----------
    points : array_like
        The input points. Contains the coordinates of known data points.
        This is an $N_s \times D$ array, where $N_s$ is the number of data
        points, and $D$ is the number of dimensions.
    values : array_like
        The input values associated with each point in `points`. This must
        be a 1D array of length $N_s$.
    xi : array_like
        The locations where interpolation is desired: an $N_{x^*} \times D$
        array.
    best_params_found : dict or None, optional
        Parameter dict typically obtained from
        :meth:`IDWGridSearchResult.best_result`. When given, it replaces the
        interpolation parameters entirely: only the ``radius`` and
        ``exponent`` keys are read, and any other key in the dict is
        ignored. **Both keys are required** -- a dict missing either one
        raises :exc:`KeyError` (see `Raises` below). The `radius` and
        `exponent` arguments passed explicitly at the call site are
        **silently ignored** when this dict is given; there is no
        per-key fallback to them. Default: ``None``.
    **kwargs
        Additional keyword arguments forwarded to the estimator: `radius`
        and `exponent` (see :func:`idw_hparams_search`), and `callback`.

    Returns
    -------
    IDWResult
        The interpolation results at the given locations.

    Raises
    ------
    KeyError
        If `best_params_found` is given but lacks a ``radius`` or an
        ``exponent`` key. This is a bare :exc:`KeyError` raised while
        reading the dict; it is **not** wrapped in
        :exc:`~spatialize.SpatializeError`.

    Notes
    -----
    This is plain IDW, not an ensemble method, so there is no
    ``n_partitions`` concept here and no key of `best_params_found` is
    exempt from being applied.

    A dict returned by ``idw_hparams_search(...).best_result()`` always
    contains both ``radius`` and ``exponent``, so the requirement above
    only bites on hand-built dicts. A dict produced by
    :func:`~spatialize.gs.esi.esi_hparams_search` is **not** compatible
    here: it has no ``radius`` key and will raise :exc:`KeyError`.

    See Also
    --------
    idw_hparams_search : Grid search that produces a compatible
        ``best_params_found`` dict for this function.
    """
    estimation = _call_libspatialize(points, values, xi, **kwargs)
    return IDWResult(estimation)


def _call_libspatialize(points, values, xi, radius=np.inf, exponent=1.0,
                        callback=default_singleton_callback,
                        best_params_found=None):
    """
    Call the libspatialize C++ library to perform IDW estimation.

    :param points: Array of input data points.
    :param values: Array of values corresponding to input data points.
    :param xi: Array of points where estimation is desired.
    :param radius: Radius parameter for IDW estimation.
    :param exponent: Exponent parameter for IDW estimation.
    :param callback: Callback function for logging progress.
    :param best_params_found: Dictionary containing best parameters found from previous searches (optional).
    :return: Estimation results from libspatialize library.
    """
    log_message(logging.logger.debug("running idw"))

    if best_params_found is None:
        rad = radius
        exp = exponent
    else:
        log_message(logging.logger.debug(f"using best params found: {best_params_found}"))
        rad, exp = best_params_found["radius"], best_params_found["exponent"]

    # get the estimator function
    estimate = lib_spatialize_facade.get_operator(points,
                                                  spatialize.gs.local_interpolator.IDW,
                                                  "estimate",
                                                  spatialize.gs.PLAIN_INTERPOLATOR)

    # get the argument list
    l_args = [np.float32(points), np.float32(values),
              rad, exp, np.float32(xi), callback]

    # run
    try:
        estimation = estimate(*l_args)
    except Exception as e:
        raise SpatializeError(e)

    return estimation

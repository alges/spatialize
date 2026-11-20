"""Plain (non-ensemble) kriging, with the same interface as :mod:`spatialize.gs.idw`.

The estimator is the univariate use of the :class:`~spatialize.gs._cokriging.Cokriging` engine: ordinary
or simple kriging with an anisotropic search neighborhood and a variogram of one or more nested
structures. Besides the estimate it returns the kriging variance.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from spatialize import SpatializeError, logging, GridSearchResult, EstimationResult
from spatialize._math_util import flatten_grid_data
from spatialize.logging import resolve_callback, singleton_null_callback, log_message
from spatialize.gs._cokriging import Cokriging

_MODELS = ("spherical", "exponential", "cubic", "gaussian")


class KrigingGridSearchResult(GridSearchResult):
    _title = "Kriging hyperparameter search"

    """Result of a hyperparameter grid search for plain kriging.

    Wraps the cross-validation error obtained for every `radius` evaluated by
    :func:`kriging_hparams_search`, and exposes the best one as a dict ready to pass to
    :func:`kriging_griddata` / :func:`kriging_nongriddata` via ``best_params_found``.

    Parameters
    ----------
    search_result_data : pandas.DataFrame
        One row per evaluated parameter combination, with a ``cv_error`` column plus a ``radius``
        column.

    Attributes
    ----------
    search_result_data : pandas.DataFrame
        The raw per-combination results, as passed in.
    cv_error : pandas.DataFrame
        The ``cv_error`` column of `search_result_data`.
    best_params : pandas.DataFrame
        Subset of `search_result_data` whose ``cv_error`` equals the minimum observed value (there
        may be more than one row in case of ties).
    """
    def __init__(self, search_result_data):
        """Initialize the grid search result. See the class docstring for parameter descriptions."""
        super().__init__(search_result_data)

    def best_result(self, optimize_data_usage=False, **kwargs):
        """Return the best-scoring parameter combination from the search.

        Parameters
        ----------
        optimize_data_usage : bool, optional
            If ``False`` (default), ties are broken by taking the largest `radius` (more data used
            per estimate). If ``True``, the smallest `radius` is preferred instead.
        **kwargs
            Unused; accepted for interface compatibility with
            :meth:`~spatialize.result.GridSearchResult.best_result`.

        Returns
        -------
        dict
            The selected row of `best_params`, plus ``"result_data_index"``. Ready to pass as
            ``best_params_found`` to :func:`kriging_griddata` or :func:`kriging_nongriddata`.
        """
        b_param = self.best_params.sort_values(by='radius', ascending=optimize_data_usage)
        row = pd.DataFrame(b_param.iloc[0]).to_dict(index=True)
        index = list(row.keys())[0]
        result = row[index]
        result.update({"result_data_index": index})
        return result


class KrigingResult(EstimationResult):
    _title = "Kriging estimation"

    """Result of a plain kriging estimation.

    Returned by :func:`kriging_griddata` and :func:`kriging_nongriddata`. It adds the kriging
    variance and the number of neighbors used at every location to
    :class:`~spatialize.result.EstimationResult`.

    Parameters
    ----------
    estimation : array_like
        Estimated values, flattened to 1D.
    variance : array_like
        Kriging variance at every location, flattened to 1D.
    neighbors : array_like
        Number of neighbors used at every location (``-1`` where the kriging system failed).
    """
    def __init__(self, estimation, variance=None, neighbors=None, griddata=False, original_shape=None, **kwargs):
        super().__init__(estimation, griddata, original_shape, **kwargs)
        self._variance = variance
        self._neighbors = neighbors

    def variance(self):
        """Return the kriging variance at locations `xi`, shaped like :meth:`estimation`."""
        return self._reshape(self._variance)

    def neighbors(self):
        """Return the number of neighbors used at locations `xi`, shaped like :meth:`estimation`."""
        return self._reshape(self._neighbors)

    def _reshape(self, data):
        if data is None:
            return None
        return data.reshape(self.original_shape) if self.griddata else data

    def plot_variance(self, ax=None, w=None, h=None, theme='alges', cmap=None, **imshow_args):
        """Plot the kriging variance, with the same arguments as :meth:`plot_estimation`."""
        from spatialize.viz import PlotStyle
        args = imshow_args.copy()
        if 'extent' not in args:
            extent = self._get_extent()
            if extent is not None:
                args['extent'] = extent
        with PlotStyle(theme=theme, cmap=cmap) as style:
            self._plot_data(self.variance(), ax, w, h, cmap=style.cmap, **args)


# ----------------------------------------------------------------------------------------------
# engine adapter

def _as3(value, d, name, pad):
    """A scalar or a sequence of up to 3 numbers as a float array of 3 (the engine is 3D)."""
    v = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if v.size == 1:
        v = np.repeat(v, d)
    if v.size not in (d, 3):
        raise SpatializeError(f"'{name}' must be a scalar or have {d} (or 3) components, got {v.size}")
    return np.concatenate([v, [pad] * (3 - v.size)]) if v.size < 3 else v


def _pad_coords(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] > 3:
        raise SpatializeError("kriging works on 1, 2 or 3 dimensional coordinates, as an N x D array")
    return np.concatenate([a, np.zeros((a.shape[0], 3 - a.shape[1]))], axis=1)


def _structures(d, model, nugget, sill, range, angles, structures):
    """The engine's variogram dict from the single structure arguments or a list of nested ones."""
    if structures is None:
        structures = [{"model": model, "sill": sill, "range": range, "angles": angles}]
    model_list = []
    for s in structures:
        t = str(s.get("model", model)).lower()
        if t not in _MODELS:
            raise SpatializeError(f"variogram model '{t}' not supported (available: {', '.join(_MODELS)})")
        r = s.get("range", range)
        if r is None:
            raise SpatializeError("the variogram 'range' is required")
        rr = _as3(r, d, "range", 1.0)
        if np.any(rr <= 0):
            raise SpatializeError("the variogram 'range' must be positive")
        model_list.append({"type": t,
                           "sill": [[float(s.get("sill", sill))]],
                           "ranges": rr,
                           "angles": _as3(s.get("angles", angles), 3, "angles", 0.0)})
    return {"nugget": [[float(nugget)]], "model": model_list}


def _default_radius(points):
    ext = np.ptp(np.asarray(points, dtype=np.float64), axis=0)
    return float(np.sqrt(np.sum(ext ** 2))) or 1.0


def _engine(points, values, radius, kriging_type, mean, nb_ndata, nb_angles, nb_octant, variogram, limits, top_cut):
    d = np.asarray(points).shape[1]
    params = {
        "samples_coords": _pad_coords(points),
        "samples_values": np.asarray(values, dtype=np.float64).reshape(-1, 1),
        "kriging_type": kriging_type,
        "nb_radius": _as3(radius, d, "radius", float(np.max(radius))),
        "nb_angles": _as3(nb_angles, 3, "nb_angles", 0.0),
        "nb_ndata": list(nb_ndata),
        "nb_octant": nb_octant,
        "variogram": variogram,
    }
    if kriging_type == "simple":
        params["kriging_mean"] = [np.nanmean(values) if mean is None else mean]
    if limits is not None:
        params["limits"] = limits
    if top_cut is not None:
        params["top_cut"] = [top_cut]
    return Cokriging(params)


def _run_engine(points, values, targets, target, **engine_args):
    """Run the engine and return ``(estimate, variance, neighbors)`` as 1D arrays."""
    eng = _engine(points, values, **engine_args)
    exec_params = {"target": target}
    if target == "scattered":
        exec_params["target_coords"] = _pad_coords(targets)
    pred, var, nbrs = eng(exec_params)
    return pred[:, 0], var[:, 0], nbrs[:, 0]


_DEFAULTS = dict(radius=None, kriging_type="ordinary", mean=None, nb_ndata=(1, 16), nb_angles=(0.0, 0.0, 0.0),
                 nb_octant=False, model="spherical", nugget=0.0, sill=1.0, range=None, angles=(0.0, 0.0, 0.0),
                 structures=None, limits=None, top_cut=None)


def _estimate(points, values, xi, callback=None, best_params_found=None, **kw):
    """Run kriging at `xi`, returning the flat estimate, variance and neighbors."""
    callback = resolve_callback(callback)
    log_message(logging.logger.debug("running kriging"))
    a = dict(_DEFAULTS)
    unknown = set(kw) - set(a)
    if unknown:
        raise SpatializeError(f"unknown kriging arguments: {', '.join(sorted(unknown))}")
    a.update(kw)

    points = np.asarray(points, dtype=np.float64)
    if a["radius"] is None or np.isinf(a["radius"]):
        a["radius"] = _default_radius(points)
    if best_params_found is not None:
        log_message(logging.logger.debug(f"using best params found: {best_params_found}"))
        a["radius"] = best_params_found["radius"]
    if a["range"] is None and a["structures"] is None:
        raise SpatializeError("the variogram 'range' (or a list of 'structures') is required")

    variogram = _structures(points.shape[1], a["model"], a["nugget"], a["sill"], a["range"], a["angles"],
                            a["structures"])
    try:
        return _run_engine(points, values, np.asarray(xi, dtype=np.float64), "scattered",
                           radius=a["radius"], kriging_type=a["kriging_type"], mean=a["mean"],
                           nb_ndata=a["nb_ndata"], nb_angles=a["nb_angles"], nb_octant=a["nb_octant"],
                           variogram=variogram, limits=a["limits"], top_cut=a["top_cut"])
    except SpatializeError:
        raise
    except Exception as e:
        raise SpatializeError(e)


_KRIGING_ARGS_DOC = """
    radius : float or array_like, optional
        Search radius, a scalar or one per axis (semi-axes of the search ellipsoid, rotated by
        `nb_angles`). Default: the diagonal of the data's bounding box, i.e. no restriction.
    kriging_type : {"ordinary", "simple"}, optional
        Default: ``"ordinary"``.
    mean : float, optional
        Known mean for simple kriging. Default: the mean of `values`.
    nb_ndata : (int, int), optional
        Minimum number of neighbors needed to estimate, and maximum number used (the nearest ones).
        Default: ``(1, 16)``.
    nb_angles : array_like, optional
        Rotation angles of the search ellipsoid, in degrees. Default: ``(0, 0, 0)``.
    nb_octant : bool, optional
        Take at most ``nb_ndata[1]`` neighbors per octant instead of overall. Default: ``False``.
    model : {"spherical", "exponential", "cubic", "gaussian"}, optional
        Variogram model of the single structure. Default: ``"spherical"``.
    nugget : float, optional
        Nugget effect. Default: ``0``.
    sill : float, optional
        Partial sill of the structure (the variance it adds above the nugget). Default: ``1``.
    range : float or array_like
        Range of the structure, a scalar or one per axis. Required, unless `structures` is given.
    angles : array_like, optional
        Rotation angles of the variogram's anisotropy ellipsoid, in degrees. Default: ``(0, 0, 0)``.
    structures : list of dict, optional
        Nested structures, each a dict with the keys ``model``, ``sill``, ``range`` and ``angles``;
        the ones missing take the values given above. Replaces the single structure.
    limits : (float, float), optional
        Values outside are discarded as missing. Default: no limits.
    top_cut : float, optional
        Values above are set to this value. Default: none.
"""


def kriging_hparams_search(points, values, xi,
                           k=10,
                           griddata=False,
                           radius=(0.1, 0.2, 0.5),
                           folding_seed=None,
                           callback=None,
                           **kriging_args
                           ):
    """Perform a k-fold (or leave-one-out) cross-validation search of the `radius` for plain kriging.

    Evaluates kriging over every candidate `radius`, scoring each by mean absolute cross-validation
    error, and returns a :class:`KrigingGridSearchResult` from which the best one can be extracted and
    fed straight into :func:`kriging_griddata` / :func:`kriging_nongriddata`. The variogram and the
    rest of the kriging arguments are fixed by the user (see below).

    Parameters
    ----------
    points : array_like
        The input points, an $N_s \\times D$ array.
    values : array_like
        The input values, a 1D array of length $N_s$.
    xi : array_like
        The interpolation points, as in :func:`~spatialize.gs.idw.idw_hparams_search` (kept for
        interface compatibility; the error is measured at the samples).
    k : int or str, optional
        Number of cross-validation folds. If `k` is ``"loo"``, ``-1`` or the number of points,
        leave-one-out is used. Default: ``10``.
    griddata : bool, optional
        Whether `xi` is grid-shaped. Default: ``False``.
    radius : tuple of float, optional
        Candidate search radii. Default: ``(0.1, 0.2, 0.5)``.
    folding_seed : int, optional
        Seed of the folds. Default: a random integer in ``[1000, 10000)``.
    callback : callable, optional
        Where the progress and the messages go, as in :func:`~spatialize.gs.idw.idw_hparams_search`.
    **kriging_args
        The kriging arguments held fixed during the search:
{ARGS}
    Returns
    -------
    KrigingGridSearchResult
        The search results, one row per evaluated `radius`.
    """
    callback = resolve_callback(callback)
    log_message(logging.logger.debug("searching best params ..."))
    kriging_args.pop("best_params_found", None)
    a = dict(_DEFAULTS)
    unknown = set(kriging_args) - set(a)
    if unknown:
        raise SpatializeError(f"unknown kriging arguments: {', '.join(sorted(unknown))}")
    a.update(kriging_args)
    if a["range"] is None and a["structures"] is None:
        raise SpatializeError("the variogram 'range' (or a list of 'structures') is required")

    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n = points.shape[0]
    if folding_seed is None:
        folding_seed = int(np.random.randint(1000, 10000))
    loo = k in ("loo", -1) or k == n
    if not loo:
        folds = np.random.RandomState(folding_seed).permutation(n) % int(k)

    variogram = _structures(points.shape[1], a["model"], a["nugget"], a["sill"], a["range"], a["angles"],
                            a["structures"])
    param_grid = ParameterGrid({"radius": radius})

    def engine_args(r):
        return dict(radius=r, kriging_type=a["kriging_type"], mean=a["mean"], nb_ndata=a["nb_ndata"],
                    nb_angles=a["nb_angles"], nb_octant=a["nb_octant"], variogram=variogram,
                    limits=a["limits"], top_cut=a["top_cut"])

    results = {}

    def run_scenario(i):
        r = param_grid[i]["radius"]
        try:
            if loo:
                cv, _, _ = _run_engine(points, values, None, "leave_one_out_sample", **engine_args(r))
            else:
                cv = np.full(n, np.nan)
                for f in range(int(k)):
                    test = folds == f
                    if test.any() and (~test).any():
                        cv[test], _, _ = _run_engine(points[~test], values[~test], points[test], "scattered",
                                                     **engine_args(r))
        except Exception as e:
            raise SpatializeError(e)
        results[i] = np.nanmean(np.abs(values - cv))
        callback(logging.progress.inform())

    callback(logging.progress.init(len(param_grid), 1, desc="Kriging hyperparameter search"))
    for i in range(len(param_grid)):
        run_scenario(i)
    callback(logging.progress.stop())

    result_data = pd.DataFrame([dict(param_grid[i], cv_error=v) for i, v in results.items()],
                               index=list(results.keys()), columns=["radius", "cv_error"])
    return KrigingGridSearchResult(result_data)


def kriging_griddata(points, values, xi, **kwargs):
    """Perform plain kriging on a regular grid.

    Parameters
    ----------
    points : array_like
        The input points, an $N_s \\times D$ array.
    values : array_like
        The input values, a 1D array of length $N_s$.
    xi : array_like
        The grid where estimation is desired: an array of grids of $D$ components, as returned by
        ``numpy.mgrid``.
    best_params_found : dict or None, optional
        Parameter dict typically obtained from :meth:`KrigingGridSearchResult.best_result`. Its
        ``radius`` replaces the one given explicitly; the other kriging arguments are not affected.
        Default: ``None``.
    callback : callable, optional
        Where the messages go, as in :func:`~spatialize.gs.idw.idw_griddata`.
    **kwargs
        The kriging arguments:
{ARGS}
    Returns
    -------
    KrigingResult
        The estimate and the kriging variance on the grid.

    Raises
    ------
    SpatializeError
        If the variogram is not well defined or the kriging system cannot be built.

    See Also
    --------
    kriging_hparams_search : Search of the `radius`, producing a compatible ``best_params_found``.
    """
    ng_xi, original_shape = flatten_grid_data(xi)
    est, var, nbrs = _estimate(points, values, ng_xi, **kwargs)
    return KrigingResult(est, var, nbrs, True, original_shape)


def kriging_nongriddata(points, values, xi, **kwargs):
    """Perform plain kriging at scattered (non-gridded) locations.

    Parameters
    ----------
    points : array_like
        The input points, an $N_s \\times D$ array.
    values : array_like
        The input values, a 1D array of length $N_s$.
    xi : array_like
        The locations where estimation is desired: an $N_{x^*} \\times D$ array.
    best_params_found : dict or None, optional
        As in :func:`kriging_griddata`.
    callback : callable, optional
        Where the messages go.
    **kwargs
        The kriging arguments, as in :func:`kriging_griddata`.

    Returns
    -------
    KrigingResult
        The estimate and the kriging variance at the given locations.

    See Also
    --------
    kriging_hparams_search : Search of the `radius`, producing a compatible ``best_params_found``.
    """
    est, var, nbrs = _estimate(points, values, xi, **kwargs)
    return KrigingResult(est, var, nbrs)


for _f in (kriging_hparams_search, kriging_griddata):
    _f.__doc__ = _f.__doc__.replace("{ARGS}", _KRIGING_ARGS_DOC.rstrip("\n"))

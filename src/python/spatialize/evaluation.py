"""Tools for evaluating the performance of spatial interpolators."""

import os
import sys
import functools
import importlib.resources as rs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spatialize.viz import PlotStyle, plot_colormap_data, plot_nongriddata, plot_scatter

from spatialize._math_util import flatten_grid_data
from spatialize.resources import data
from spatialize.logging import log_message
from spatialize import SpatializeError, logging

# ------------ Metrics ------------

# --- Continuous ---
def _mask_valid_points(true, pred):
    """Returns a boolean mask of positions where neither true nor pred is NaN."""
    return ~np.isnan(np.asarray(true, dtype=float)) & ~np.isnan(np.asarray(pred, dtype=float))

def error(true, pred):
    """Calculates point-wise signed error (pred - true)."""
    return np.asarray(pred) - np.asarray(true)

def absolute_error(true, pred):
    """Calculates point-wise absolute error for a prediction."""
    return np.abs(error(true, pred))

def bias(true, pred):
    """Calculates the mean signed error (bias). Positive = systematic over-prediction."""
    true, pred = np.asarray(true, dtype=float), np.asarray(pred, dtype=float)
    valid_mask = _mask_valid_points(true, pred)
    return np.mean(error(true[valid_mask], pred[valid_mask]))

def MAE(true, pred):
    """Calculates the Mean Absolute Error (MAE) metric for a prediction."""
    true, pred = np.asarray(true, dtype=float), np.asarray(pred, dtype=float)
    valid_mask = _mask_valid_points(true, pred)
    return np.mean(absolute_error(true[valid_mask], pred[valid_mask]))

def MSE(true, pred):
    """Calculates the Mean Squared Error (MSE) metric for a prediction."""
    true, pred = np.asarray(true, dtype=float), np.asarray(pred, dtype=float)
    valid_mask = _mask_valid_points(true, pred)
    return np.mean(error(true[valid_mask], pred[valid_mask]) ** 2)

def RMSE(true, pred):
    """Calculates the Root Mean Squared Error (RMSE) metric for a prediction."""
    return np.sqrt(MSE(true, pred))

def R2(true, pred):
    """Calculates the coefficient of determination (R²). Returns 0 if true is constant."""
    true, pred = np.asarray(true, dtype=float), np.asarray(pred, dtype=float)
    valid_mask = _mask_valid_points(true, pred)
    t, p = true[valid_mask], pred[valid_mask]
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - np.sum((t - p) ** 2) / ss_tot

def operational(metric):
    """Decorator to turn an error metric into its operational version (divided by dynamic range)."""
    @functools.wraps(metric)
    def inner_function(true, pred):
        true = np.asarray(true, dtype=float)
        dyn_range = np.nanmax(true) - np.nanmin(true)
        if dyn_range == 0:
            return np.nan
        return metric(true, pred) / dyn_range
    return inner_function

@operational
def op_error(true, pred):
    """Calculates point-wise operational error (normalized by dynamic range)."""
    return error(true, pred)

@operational
def op_mae(true, pred):
    """Calculates the Operational Mean Absolute Error (OpMAE)."""
    return MAE(true, pred)

@operational
def op_rmse(true, pred):
    """Calculates the Operational Root Mean Squared Error (OpRMSE)."""
    return RMSE(true, pred)

@operational
def op_mse(true, pred):
    """Calculates the Operational Mean Squared Error (OpMSE)."""
    return MSE(true, pred)


# --- Categorical ---
def _mask_valid_categorical(true, pred):
    """
    Returns a boolean mask of positions where neither true nor pred is missing.

    A value is considered missing if it is None or a float NaN.  String labels
    (including the literal string 'nan') are treated as valid categories.
    """
    def _is_valid(v):
        if v is None:
            return False
        try:
            return not np.isnan(v)   # catches float NaN
        except (TypeError, ValueError):
            return True              # non-numeric label — always valid
    return np.array([_is_valid(t) and _is_valid(p) for t, p in zip(true, pred)])

_ALLOWED_AVERAGES = {'weighted', 'macro', 'micro', None}

def _validate_average(average):
    """Raises ValueError if `average` is not a valid averaging strategy."""
    if average not in _ALLOWED_AVERAGES:
        raise ValueError(
            f"Invalid average={average!r}. Must be one of "
            f"{sorted(a for a in _ALLOWED_AVERAGES if a is not None)} or None.")

def accuracy(true, pred):
    """
    Calculates the accuracy (fraction of correct predictions) for categorical predictions.

    Positions where either true or pred is missing (None / NaN) are ignored.
    """
    true, pred = np.asarray(true, dtype=object), np.asarray(pred, dtype=object)
    mask = _mask_valid_categorical(true, pred)
    if not mask.any():
        return np.nan
    return np.mean(true[mask] == pred[mask])

def precision(true, pred, average='weighted'):
    """
    Calculates precision for categorical predictions.

    Missing values (None / NaN) in either array are excluded before computing.

    Parameters
    ----------
    true : array_like
        Ground truth labels.
    pred : array_like
        Predicted labels.
    average : str, optional
        Averaging strategy: 'weighted', 'macro', 'micro', or None (per-class).
        Default is 'weighted'.
    """
    _validate_average(average)
    from sklearn.metrics import precision_score
    true, pred = np.asarray(true, dtype=object), np.asarray(pred, dtype=object)
    mask = _mask_valid_categorical(true, pred)
    return precision_score(true[mask], pred[mask], average=average, zero_division=0)

def recall(true, pred, average='weighted'):
    """
    Calculates recall for categorical predictions.

    Missing values (None / NaN) in either array are excluded before computing.

    Parameters
    ----------
    true : array_like
        Ground truth labels.
    pred : array_like
        Predicted labels.
    average : str, optional
        Averaging strategy: 'weighted', 'macro', 'micro', or None (per-class).
        Default is 'weighted'.
    """
    _validate_average(average)
    from sklearn.metrics import recall_score
    true, pred = np.asarray(true, dtype=object), np.asarray(pred, dtype=object)
    mask = _mask_valid_categorical(true, pred)
    return recall_score(true[mask], pred[mask], average=average, zero_division=0)

def f1_score(true, pred, average='weighted'):
    """
    Calculates F1 score for categorical predictions.

    Missing values (None / NaN) in either array are excluded before computing.

    Parameters
    ----------
    true : array_like
        Ground truth labels.
    pred : array_like
        Predicted labels.
    average : str, optional
        Averaging strategy: 'weighted', 'macro', 'micro', or None (per-class).
        Default is 'weighted'.
    """
    _validate_average(average)
    from sklearn.metrics import f1_score as _sklearn_f1
    true, pred = np.asarray(true, dtype=object), np.asarray(pred, dtype=object)
    mask = _mask_valid_categorical(true, pred)
    return _sklearn_f1(true[mask], pred[mask], average=average, zero_division=0)


# Metrics requiring numeric vs. categorical `values`, used by loo_validation/kfold_validation
# to pick a categorical-aware default and to validate explicitly-passed metrics upfront.
_CONTINUOUS_METRICS = {MAE, MSE, RMSE, R2, bias, error, absolute_error,
                       op_mae, op_mse, op_rmse, op_error}
_CATEGORICAL_METRICS = {accuracy, precision, recall, f1_score}


# ------------ Cross-validation ------------

# --- LOO ---
def loo_validation(points, values, interpolation_function,
                   metrics=None, return_predictions=False,
                   suppress_output=True,
                   **interpolation_kwargs):
    """
    Leave-one-out cross-validation for any spatial interpolation method.

    At each iteration, one sample is held out and the interpolator is asked to
    predict at that location using the remaining samples.  Works with any callable
    that follows the signature ``f(points, values, xi, **kwargs) -> array``.

    Parameters
    ----------
    points : array_like, shape (n, d)
        Spatial coordinates of the data.
    values : array_like, shape (n,)
        Observed values at ``points``.
    interpolation_function : callable
        Function with signature ``f(points, values, xi, **kwargs) -> array``.
        Must return an array whose first element is the prediction at ``xi[0]``.
    metrics : list of callable, optional
        Metric functions to evaluate.  Each must have signature
        ``metric(true, pred) -> float``.  Defaults to ``[MAE, RMSE]`` for
        continuous ``values``, or ``[accuracy]`` when ``values`` is
        auto-detected as categorical (non-numeric dtype).
    return_predictions : bool, optional
        If True, also return the full predictions array.  Default is False.
    suppress_output : bool, optional
        If True, suppress stdout during each interpolation call.  Default is True.
    **interpolation_kwargs
        Additional keyword arguments forwarded to ``interpolation_function``.

    Returns
    -------
    results : dict
        Mapping of metric function name to its value computed over all LOO folds.
    predictions : ndarray, shape (n,)
        Only returned when ``return_predictions=True``.  Contains LOO predictions
        (NaN where prediction failed).

    Raises
    ------
    SpatializeError
        If any metric in ``metrics`` is incompatible with the detected
        (categorical vs. numeric) type of ``values``.

    Examples
    --------
    >>> def esi_predict(points, values, xi, **kwargs):
    ...     return esi_nongriddata(points, values, xi, n_partitions=300, **kwargs).estimation()
    >>>
    >>> results = loo_validation(points, values, esi_predict,
    ...                          metrics=[MAE, RMSE, R2],
    ...                          local_interpolator='idw', alpha=0.9, exponent=2.0)
    >>> print(results)
    {'MAE': 0.12, 'RMSE': 0.18, 'R2': 0.94}
    >>>
    >>> results, preds = loo_validation(points, values, esi_predict,
    ...                                 return_predictions=True,
    ...                                 local_interpolator='idw', alpha=0.9, exponent=2.0)
    """
    points = np.asarray(points)
    _values = np.asarray(values)
    _categorical = not np.issubdtype(_values.dtype, np.number)

    if metrics is None:
        metrics = [accuracy] if _categorical else [MAE, RMSE]
    else:
        wrong = _CONTINUOUS_METRICS if _categorical else _CATEGORICAL_METRICS
        bad = [m for m in metrics if m in wrong]
        if bad:
            raise SpatializeError(
                f"metrics {[m.__name__ for m in bad]} are incompatible with "
                f"{'categorical' if _categorical else 'numeric'} values.")

    if _categorical:
        values = _values.astype(object)
        predictions = np.full(len(values), None, dtype=object)
    else:
        values = _values.astype(float)
        predictions = np.full(len(values), np.nan)
    n_samples = len(points)

    for i in range(n_samples):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[i] = False

        xi = points[i:i+1]

        try:
            if suppress_output:
                with SuppressOutput():
                    pred = interpolation_function(
                        points[train_mask], values[train_mask], xi, **interpolation_kwargs)
            else:
                pred = interpolation_function(
                    points[train_mask], values[train_mask], xi, **interpolation_kwargs)
            predictions[i] = np.asarray(pred).flat[0]
        except Exception as e:
            log_message(logging.logger.warning(f"LOO failed at sample {i}: {e}"))

    results = {m.__name__: m(values, predictions) for m in metrics}

    if return_predictions:
        return results, predictions
    return results


# --- K-Fold ---
def kfold_validation(points, values, interpolation_function,
                     k=5, metrics=None, return_predictions=False,
                     suppress_output=True, seed=None,
                     **interpolation_kwargs):
    """
    K-fold cross-validation for any spatial interpolation method.

    Splits the data into ``k`` folds.  In each iteration, one fold is held out
    as the test set and the interpolator is fitted on the remaining folds.

    Parameters
    ----------
    points : array_like, shape (n, d)
        Spatial coordinates of the data.
    values : array_like, shape (n,)
        Observed values at ``points``.
    interpolation_function : callable
        Function with signature ``f(points, values, xi, **kwargs) -> array``.
    k : int, optional
        Number of folds.  Default is 5.
    metrics : list of callable, optional
        Metric functions, each with signature ``metric(true, pred) -> float``.
        Defaults to ``[MAE, RMSE]`` for continuous ``values``, or
        ``[accuracy]`` when ``values`` is auto-detected as categorical
        (non-numeric dtype).
    return_predictions : bool, optional
        If True, also return the out-of-fold predictions array.  Default is False.
    suppress_output : bool, optional
        If True, suppress stdout during each call.  Default is True.
    seed : int, optional
        Random seed for reproducible fold assignment.
    **interpolation_kwargs
        Additional keyword arguments forwarded to ``interpolation_function``.

    Returns
    -------
    results : dict
        Mapping of metric name to value, averaged across all folds.
    predictions : ndarray, shape (n,)
        Only returned when ``return_predictions=True``.  Out-of-fold predictions
        (NaN where prediction failed).

    Raises
    ------
    SpatializeError
        If any metric in ``metrics`` is incompatible with the detected
        (categorical vs. numeric) type of ``values``.

    Examples
    --------
    >>> results = kfold_validation(points, values, esi_predict,
    ...                            k=10, metrics=[MAE, RMSE, R2],
    ...                            local_interpolator='idw', alpha=0.9, exponent=2.0)
    """
    points = np.asarray(points)
    n_samples = len(points)
    _values = np.asarray(values)
    _categorical = not np.issubdtype(_values.dtype, np.number)

    if metrics is None:
        metrics = [accuracy] if _categorical else [MAE, RMSE]
    else:
        wrong = _CONTINUOUS_METRICS if _categorical else _CATEGORICAL_METRICS
        bad = [m for m in metrics if m in wrong]
        if bad:
            raise SpatializeError(
                f"metrics {[m.__name__ for m in bad]} are incompatible with "
                f"{'categorical' if _categorical else 'numeric'} values.")

    if _categorical:
        values = _values.astype(object)
        predictions = np.full(n_samples, None, dtype=object)
    else:
        values = _values.astype(float)
        predictions = np.full(n_samples, np.nan)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    folds = np.array_split(indices, k)

    for fold_idx in folds:
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[fold_idx] = False

        xi = points[fold_idx]

        try:
            if suppress_output:
                with SuppressOutput():
                    preds = interpolation_function(
                        points[train_mask], values[train_mask], xi, **interpolation_kwargs)
            else:
                preds = interpolation_function(
                    points[train_mask], values[train_mask], xi, **interpolation_kwargs)
            predictions[fold_idx] = np.asarray(preds).flatten()
        except Exception as e:
            log_message(logging.logger.warning(f"K-fold failed on a fold: {e}"))

    results = {m.__name__: m(values, predictions) for m in metrics}

    if return_predictions:
        return results, predictions
    return results


# ------------ Benchmark ------------
# --- Kriging ---
def kriging_prediction(points, values, xi, seed=42, griddata=False, **kwargs):
    from pykrige.rk import Krige
    if griddata: 
        _xi, original_shape = flatten_grid_data(xi)
    else:
        _xi = xi.copy()

    np.random.seed(seed)
    kriging = Krige(**kwargs)
    kriging.fit(points, values)
    est = kriging.predict(_xi)

    if griddata: 
        return est.reshape(original_shape)
    else:
        return est

def auto_krige(points, values, xi, griddata=False, seed=42, metric='mae'):
    """Perform automated Kriging parameter grid search + Kriging estimation."""
    from sklearn.model_selection import GridSearchCV
    from pykrige.rk import Krige

    _METRIC_MAP = {
        'mae':  'neg_mean_absolute_error',
        'mse':  'neg_mean_squared_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2':   'r2',
    }
    metric_key = metric.lower()
    if metric_key not in _METRIC_MAP:
        raise ValueError(f"metric '{metric}' not recognised. Use one of: {list(_METRIC_MAP)}")
    scoring = _METRIC_MAP[metric_key]

    if griddata:
        n_dims = len(xi)
    else:
        n_dims = xi.shape[-1]

    param_dict = {
    "variogram_model": ["linear", "power", "gaussian", "spherical"],
    "nlags": [4, 6, 8],
    "n_closest_points": [3, 5, 10, 15]
    }
    if n_dims==2:
        param_dict["method"] = ["ordinary", "universal"]
    elif n_dims==3:
        param_dict["method"] = ["ordinary3d", "universal3d"]
    else:
        raise SpatializeError("Kriging supported for 2D or 3D data only")

    grid_search = GridSearchCV(Krige(), param_dict, scoring=scoring, verbose=True)

    # Execute Kriging with optimized parameters
    with SuppressOutput():
        grid_search.fit(X = points, y = values)
    log_message(logging.logger.debug(f"best_params = {grid_search.best_params_}"))

    # pending, does kriging prediction work for both griddata and nongriddata or must we reshape griddata and then return back to original shape (as currently impldmented)?
    return kriging_prediction(points, values, xi, seed, griddata, **grid_search.best_params_)

# --- Scipy ---
def scipy_prediction(points, values, xi, method='nearest', griddata=False):
    """Returns scipy.griddata prediction."""
    from scipy.interpolate import griddata as scipy_griddata

    if griddata:
        _xi, original_shape = flatten_grid_data(xi)
    else:
        _xi = xi.copy()

    est = scipy_griddata(points, values, _xi, method=method)

    if griddata:
        return est.reshape(original_shape)
    else:
        return est

def auto_scipy_prediction(points, values, xi, metric='mae', griddata=False):
    """
    Select the best scipy interpolation method via LOO cross-validation and return its estimate.

    Evaluates 'nearest', 'linear', and 'cubic' scipy methods using LOO
    cross-validation with the specified metric, then runs the winner on the
    full dataset to produce the final prediction.

    Parameters
    ----------
    points : array_like, shape (n, d)
        Spatial coordinates of the data.
    values : array_like, shape (n,)
        Observed values at ``points``.
    xi : array_like
        Locations where the estimate is desired.
    metric : str, optional
        LOO metric used to rank methods.  One of 'mae', 'rmse', 'mse', 'r2'.
        Default is 'mae'.
    griddata : bool, optional
        If True, ``xi`` is a meshgrid tuple and output is reshaped accordingly.
        Default is False.

    Returns
    -------
    ndarray
        Predicted values at ``xi`` from the best-scoring method.
    """
    _METRIC_MAP = {'mae': MAE, 'rmse': RMSE, 'mse': MSE, 'r2': R2}
    _HIGHER_IS_BETTER = {'r2'}

    metric_key = metric.lower()
    if metric_key not in _METRIC_MAP:
        raise ValueError(f"metric '{metric}' not recognised. Use one of: {list(_METRIC_MAP)}")
    metric_fn = _METRIC_MAP[metric_key]

    best_method = None
    best_score = None

    for method in ('nearest', 'linear', 'cubic'):
        try:
            results = loo_validation(
                points, values, scipy_prediction,
                metrics=[metric_fn],
                method=method, griddata=griddata,
            )
            score = results[metric_fn.__name__]
            if best_score is None or (
                (metric_key in _HIGHER_IS_BETTER and score > best_score) or
                (metric_key not in _HIGHER_IS_BETTER and score < best_score)
            ):
                best_score = score
                best_method = method
        except Exception as e:
            log_message(logging.logger.warning(f"LOO failed for method '{method}': {e}"))

    if best_method is None:
        raise RuntimeError("All scipy methods failed during LOO validation.")

    log_message(logging.logger.debug(
        f"Best method: '{best_method}'  {metric_fn.__name__} = {best_score:.4f}"))
    return scipy_prediction(points, values, xi, method=best_method, griddata=griddata)


# ------------ Case Studies ------------

# Helper functions for case studies
def _calculate_extent(dataframe, x_col='X', y_col='Y'):
    """
    Calculate spatial extent from coordinate columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing spatial coordinates.
    x_col : str, optional
        Name of the X coordinate column. Default is 'X'.
    y_col : str, optional
        Name of the Y coordinate column. Default is 'Y'.

    Returns
    -------
    list
        Extent as [xmin, xmax, ymin, ymax].

    Examples
    --------
    >>> df = pd.DataFrame({'X': [0, 10, 5], 'Y': [0, 20, 10]})
    >>> _calculate_extent(df)
    [0, 10, 0, 20]
    """
    return [
        dataframe[x_col].min(),
        dataframe[x_col].max(),
        dataframe[y_col].min(),
        dataframe[y_col].max()
    ]


class SyntheticScenario:
    """
    Generate synthetic spatial interpolation scenarios with known ground truth.

    This class creates test scenarios for spatial interpolation methods by generating
    synthetic data with known mathematical functions. It supports both 2D and 3D cases,
    and can output data in griddata or non-griddata formats. The synthetic functions
    allow validation of interpolation methods against ground truth values.

    Parameters
    ----------
    n_dims : int, optional
        Dimensionality of the scenario (2 or 3). Default is 2.
    extent : list, optional
        Spatial extent of the domain.
        - For 2D: [x_min, x_max, y_min, y_max]
        - For 3D: [x_min, x_max, y_min, y_max, z_min, z_max]
        Default is [0, 1, 0, 1].
    griddata : bool, optional
        If True, generate regular grid format compatible with esi_griddata().
        If False, generate scattered points format for esi_nongriddata().
        Default is False.
    n_grid_points : int or tuple, optional
        Number of grid points per dimension. Can be:
        - int: Same number of points in all dimensions
        - tuple: Specific points per dimension (length must match n_dims)
        - None: Automatically set to (extent_max - extent_min + 1) per dimension
        Default is None.
    categorical : bool, optional
        If True, generate categorical (class label) outputs instead of continuous
        values.  Forces 2D only.  ``simulate_scenario`` will default to
        ``kind='nominal'`` and return string label arrays.  Default is False.

    Attributes
    ----------
    n_dims : int
        Number of spatial dimensions (2 or 3).
    extent : list
        Spatial extent of the domain.
    griddata : bool
        Format flag for output data structure.
    categorical : bool
        Whether the scenario produces categorical labels.
    n_grid_points : tuple
        Number of grid points per dimension.

    Examples
    --------
    Continuous 2D scenario:

    >>> scenario = SyntheticScenario(n_dims=2, extent=[0, 100, 0, 150], griddata=True)
    >>> points, values, xi, reference = scenario.simulate_scenario(n_samples=300, seed=42)
    >>> scenario.plot_2d_scenario(points, xi, reference, theme='publication')

    Categorical 2D scenario:

    >>> scenario = SyntheticScenario(categorical=True, n_grid_points=80)
    >>> points, values, xi, reference = scenario.simulate_scenario(n_samples=200, seed=0)
    >>> scenario.plot_2d_scenario(points, xi, reference)

    3D continuous scenario with custom grid resolution:

    >>> scenario = SyntheticScenario(n_dims=3, extent=[0, 10, 0, 10, 0, 5],
    ...                              griddata=False, n_grid_points=(50, 50, 25))
    >>> points, values, xi, reference = scenario.simulate_scenario(n_samples=500)

    See Also
    --------
    spatialize.gs.esi.esi_griddata : ESI for regular grids
    spatialize.gs.esi.esi_nongriddata : ESI for scattered points
    spatialize.gs.cat_esi.cat_esi_nongriddata : Categorical ESI for scattered points
    """
    ORDINAL_ORDER = {"Low": 0, "Medium": 1, "High": 2}

    def __init__(self,
                 n_dims=2,
                 extent=[0, 1, 0, 1],
                 griddata=False,
                 n_grid_points=None,
                 categorical=False):
        # Assertions
        assert len(extent)==2*n_dims, f"{2*n_dims} values expected for extent."
        assert n_dims in [2, 3], f"{n_dims} dimensions not supported. 2D and 3D available."
        assert not (categorical and n_dims != 2), "Categorical scenarios are only supported for 2D."

        self.n_dims = n_dims
        self.extent = extent
        self.griddata = griddata
        self.categorical = categorical

        # Number of grid points
        self.n_grid_points = n_grid_points or tuple([int(extent[i+1] - extent[i] + 1)
                                                       for i in range(0, len(extent), 2)])
        # If single number, apply to all dimensions
        if isinstance(self.n_grid_points, int):
            self.n_grid_points = tuple([self.n_grid_points] * n_dims)

    def create_regular_grid(self):
        """
        Create a regular grid for interpolation.

        Generates a regular grid covering the spatial extent defined in the scenario.
        The output format depends on the `griddata` flag.

        Returns
        -------
        ndarray
            Grid points in the format specified by `self.griddata`:
            - If griddata=True (2D): shape (2, nx, ny) for use with esi_griddata()
            - If griddata=False (2D): shape (nx*ny, 2) array of [x, y] coordinates
            - If griddata=True (3D): shape (3, nx, ny, nz) for use with esi_griddata()
            - If griddata=False (3D): shape (nx*ny*nz, 3) array of [x, y, z] coordinates

        Examples
        --------
        >>> scenario = SyntheticScenario(n_dims=2, extent=[0, 10, 0, 20],
        ...                              griddata=True, n_grid_points=50)
        >>> xi = scenario.create_regular_grid()
        >>> xi.shape
        (2, 50, 50)
        """
        if self.n_dims == 2:
            x_min, x_max, y_min, y_max = self.extent
            nx, ny = self.n_grid_points
            
            if self.griddata:
                return np.mgrid[x_min:x_max:nx*1j, y_min:y_max:ny*1j]
            
            x_coords = np.linspace(x_min, x_max, nx)
            y_coords = np.linspace(y_min, y_max, ny)
            return np.array([(x, y) for x in x_coords for y in y_coords])
        
        else:       # 3D
            x_min, x_max, y_min, y_max, z_min, z_max = self.extent
            nx, ny, nz = self.n_grid_points
            
            if self.griddata:
                return np.mgrid[x_min:x_max:nx*1j, 
                               y_min:y_max:ny*1j, 
                               z_min:z_max:nz*1j]
            
            x_coords = np.linspace(x_min, x_max, nx)
            y_coords = np.linspace(y_min, y_max, ny)
            z_coords = np.linspace(z_min, z_max, nz)
            return np.array([(x, y, z) for x in x_coords for y in y_coords for z in z_coords])
    
    # Sample data generation
    def _normalize_coords(self, *coords):
        """
        Normalize coordinates to [0, 1] range based on extent.

        Parameters
        ----------
        *coords : array_like
            Variable number of coordinate arrays to normalize.

        Returns
        -------
        list
            List of normalized coordinate arrays.
        """
        normalized = []
        for i, coord in enumerate(coords):
            min_val = self.extent[2*i]
            max_val = self.extent[2*i + 1]
            normalized.append((coord - min_val) / (max_val - min_val))
        return normalized
    
    def cubic_func_2d(self, x, y):
        """
        Generate 2D synthetic test function values.

        Computes a smooth, non-linear test function with multiple local extrema.
        The function is designed to test interpolation performance with complex
        spatial patterns. Coordinates are automatically normalized to [0, 1].

        Parameters
        ----------
        x : array_like
            X-coordinates (will be normalized to extent).
        y : array_like
            Y-coordinates (will be normalized to extent).

        Returns
        -------
        ndarray
            Function values at the given coordinates.

        Notes
        -----
        The function formula is:
        f(x,y) = x(1-x) * cos(4πx) * sin²(4πy²)

        where x and y are first normalized to [0,1].
        """
        x, y = self._normalize_coords(x, y)
        return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2
    
    def cubic_func_3d(self, x, y, z):
        """
        Generate 3D synthetic test function values.

        Computes a smooth, non-linear 3D test function with multiple local extrema.
        The function extends the 2D version by adding variation in the z-dimension.
        Coordinates are automatically normalized to [0, 1].

        Parameters
        ----------
        x : array_like
            X-coordinates (will be normalized to extent).
        y : array_like
            Y-coordinates (will be normalized to extent).
        z : array_like
            Z-coordinates (will be normalized to extent).

        Returns
        -------
        ndarray
            Function values at the given coordinates.

        Notes
        -----
        The function formula is:
        f(x,y,z) = x(1-x) * cos(4πx) * sin²(4πy²) * cos(4πz)

        where x, y, and z are first normalized to [0,1].
        """
        x, y, z = self._normalize_coords(x, y, z)
        return (x * (1 - x) * np.cos(4 * np.pi * x) *
                np.sin(4 * np.pi * y ** 2) ** 2 *
                np.cos(4 * np.pi * z))

    def nominal_gt(self, x, y):
        """
        Generate a 4-class categorical label map using a spiral-polar construction.

        Class boundaries are defined in polar coordinates using a superposition of a
        spiral phase and radial oscillations, producing irregular, multi-connected
        class regions that are challenging for spatial classifiers.  Coordinates are
        automatically normalized to [0, 1] based on the scenario extent.

        Parameters
        ----------
        x : array_like
            X-coordinates (will be normalized to extent).
        y : array_like
            Y-coordinates (will be normalized to extent).

        Returns
        -------
        ndarray of str
            Class labels ('A', 'B', 'C', or 'D') at the given coordinates.
        """
        x_norm, y_norm = self._normalize_coords(x, y)
        cx, cy = x_norm - 0.5, y_norm - 0.5
        r = np.sqrt(cx**2 + cy**2)
        theta = np.arctan2(cy, cx)

        spiral = theta + 3.5 * np.pi * r
        f = np.sin(spiral * 2.0) * np.cos(r * 4 * np.pi)
        g = np.cos(spiral * 3.0) * np.sin(r * 6 * np.pi + 1.0)
        combined = f + g

        return np.select(
            [combined < -0.7, combined < 0.05, combined < 0.7],
            ["A", "B", "C"], default="D",
        )

    def _terrain_z(self, x_norm, y_norm):
        """
        Evaluate the underlying continuous terrain field on normalised [0, 1] coordinates.

        The field is a superposition of Gaussian bumps and a sinusoidal ripple,
        designed to produce irregular, partially overlapping class regions when
        thresholded into ordinal categories.

        Parameters
        ----------
        x_norm, y_norm : array_like
            Coordinates already normalised to [0, 1].

        Returns
        -------
        ndarray of float
        """
        x, y = np.asarray(x_norm), np.asarray(y_norm)
        return (
            3.0 * np.exp(-((x - 0.2)**2 + (y - 0.8)**2) / 0.018)
          + 2.5 * np.exp(-((x - 0.8)**2 + (y - 0.2)**2) / 0.022)
          + 2.0 * np.exp(-((x - 0.55)**2 + (y - 0.55)**2) / 0.012)
          - 1.8 * np.exp(-((x - 0.5)**2  + (y - 0.85)**2) / 0.03)
          - 1.5 * np.exp(-((x - 0.15)**2 + (y - 0.15)**2) / 0.025)
          + 0.4 * np.sin(6 * np.pi * x) * np.cos(5 * np.pi * y)
        )

    def ordinal_gt(self, x, y):
        """
        Generate ordinal class labels from a synthetic terrain field.

        Thresholds the underlying ``_terrain_z`` field at its 33rd and 67th
        percentiles (computed from a dense reference grid) to produce three
        roughly balanced ordinal classes: 'Low', 'Medium', 'High'.  Coordinates
        are automatically normalised to [0, 1] based on the scenario extent.
        Thresholds are derived from the unit square so class boundaries are
        stable regardless of the sample set.

        Parameters
        ----------
        x : array_like
            X-coordinates (will be normalised to extent).
        y : array_like
            Y-coordinates (will be normalised to extent).

        Returns
        -------
        ndarray of str
            Class labels ('Low', 'Medium', or 'High') at the given coordinates.

        See Also
        --------
        SyntheticScenario.ORDINAL_ORDER : canonical integer ranking of the labels.
        """
        x_norm, y_norm = self._normalize_coords(x, y)

        # Fixed thresholds from a dense unit-square grid for stable class boundaries
        ref = np.linspace(0, 1, 300)
        rx, ry = np.meshgrid(ref, ref)
        z_ref = self._terrain_z(rx.ravel(), ry.ravel())
        q33, q67 = np.percentile(z_ref, [33, 67])

        z = self._terrain_z(x_norm, y_norm)
        return np.where(z < q33, "Low", np.where(z < q67, "Medium", "High"))

    def simulate_scenario(self, kind=None, n_samples=100, seed=None,
                          custom_func=None, custom_func_params=None):
        """
        Generate a complete synthetic interpolation scenario.

        Creates random sample points with known values and a reference grid with
        ground truth values for validation. This provides all inputs needed to
        test spatial interpolation methods.

        Parameters
        ----------
        kind : str, optional
            Type of synthetic function to use. Options:
            - 'cubic'   : smooth continuous function (``cubic_func_2d`` / ``cubic_func_3d``)
            - 'nominal' : 4-class spiral-polar categorical map (``nominal_gt``, 2D only)
            If None, defaults to 'nominal' when ``self.categorical=True``, else 'cubic'.
        n_samples : int, optional
            Number of random sample points to generate. Default is 100.
        seed : int, optional
            Random seed for reproducibility. If None, results will vary.
            Default is None.
        custom_func : callable, optional
            Custom function to use instead of built-in functions.
            Should accept coordinates as separate arguments (x, y) or (x, y, z).
            Default is None.
        custom_func_params : dict, optional
            Additional keyword arguments to pass to custom_func.
            Default is None.

        Returns
        -------
        points : ndarray, shape (n_samples, n_dims)
            Random sample point coordinates.
        values : ndarray, shape (n_samples,)
            Function values (float) or labels (str) at sample points.
        xi : ndarray
            Interpolation grid locations (format depends on self.griddata).
        reference_values : ndarray
            Ground truth values at grid locations for validation.

        Raises
        ------
        ValueError
            If kind is unrecognised and no custom_func is provided.

        Examples
        --------
        >>> scenario = SyntheticScenario(n_dims=2, extent=[0, 100, 0, 150], griddata=True)
        >>> points, values, xi, reference = scenario.simulate_scenario(n_samples=300, seed=42)
        >>> print(f"Samples: {points.shape}, Grid: {xi.shape}")
        Samples: (300, 2), Grid: (2, 101, 151)

        Categorical scenario — kind defaults to 'nominal':

        >>> scenario = SyntheticScenario(categorical=True, n_grid_points=80)
        >>> points, values, xi, reference = scenario.simulate_scenario(n_samples=200, seed=0)

        Using a custom function:

        >>> def linear_func(x, y, slope=1.0):
        ...     return slope * (x + y)
        >>> scenario = SyntheticScenario(n_dims=2)
        >>> points, values, xi, ref = scenario.simulate_scenario(
        ...     n_samples=50, custom_func=linear_func, custom_func_params={'slope': 2.0})
        """
        if seed is not None:
            np.random.seed(seed)

        # Resolve default kind
        if kind is None:
            kind = 'nominal' if self.categorical else 'cubic'

        # Grid to make estimates
        xi = self.create_regular_grid()

        # Random locations for the samples
        ranges = [self.extent[i+1] - self.extent[i] for i in range(0, len(self.extent), 2)]
        offsets = [self.extent[i] for i in range(0, len(self.extent), 2)]
        points = np.random.random((n_samples, self.n_dims)) * ranges + offsets

        # Select function
        if kind == 'cubic':
            func = self.cubic_func_2d if self.n_dims == 2 else self.cubic_func_3d
            func_params = {}
        elif kind == 'nominal':
            assert self.n_dims == 2, "'nominal' kind is only supported for 2D scenarios."
            func = self.nominal_gt
            func_params = {}
        elif kind == 'ordinal':
            assert self.n_dims == 2, "'ordinal' kind is only supported for 2D scenarios."
            func = self.ordinal_gt
            func_params = {}
        elif custom_func:
            func = custom_func
            func_params = custom_func_params or {}
        else:
            raise ValueError(f"kind '{kind}' not recognised. Use 'cubic', 'nominal', 'ordinal', or provide a custom_func.")

        values = func(*points.T, **func_params)

        if self.griddata:
            reference_values = func(*xi, **func_params)
        else:
            reference_values = func(*xi.T, **func_params)

        return points, values, xi, reference_values
    
    def plot_2d_scenario(self, points, values, xi, reference_values,
                         theme='alges', cmap=None, nonnum_order=None,
                         point_size=1.5, #point_color='white',
                         figsize=(5, 6), dpi=100, title='Reference values and data points'):
        """
        Visualize 2D scenario with reference values and sample points.

        Creates a plot showing the ground truth function values across the domain
        with sample point locations overlaid. Useful for visualizing synthetic
        scenarios before and after interpolation.

        Parameters
        ----------
        points : ndarray, shape (n_samples, 2)
            Sample point coordinates from simulate_scenario().
        values : ndarray, shape (n_samples,)
            Sample values (float or str) from simulate_scenario().  Used to color
            sample point markers when ``self.categorical=True``.
        xi : ndarray
            Grid locations from simulate_scenario() (format depends on griddata).
        reference_values : ndarray
            Ground truth values from simulate_scenario().
        theme : str, optional
            Visualization theme. Options: 'alges', 'minimal', 'publication', 'whitegrid'.
            Default is 'alges'.
        cmap : str or Colormap, optional
            Colormap. When None, uses the theme default — for continuous
            scenarios this is the theme's own sequential colormap; for
            categorical scenarios it is a set of discrete swatches sampled
            from that same theme colormap (see ``PlotStyle.resolve_cmap``),
            so both panels stay visually consistent with the active theme.
            Default is None.
        nonnum_order : list, optional
            Ordered list of category labels for ordinal scenarios (e.g.
            ``['Low', 'Medium', 'High']``).  Passed to the categorical plotting
            functions to control legend/color ordering.  Default is None.
        point_size : float, optional
            Size of sample point markers. Default is 1.5.
        point_color : str, optional
            Color of sample point markers for continuous scenarios. Default is 'white'.
        figsize : tuple, optional
            Figure size as (width, height) in inches. Default is (5, 6).
        dpi : int, optional
            Figure resolution in dots per inch. Default is 100.
        title : str, optional
            Plot title. Default is 'Reference values and data points'.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        ax : matplotlib.axes.Axes
            The created axes.

        Raises
        ------
        AssertionError
            If the scenario is not 2D.

        Examples
        --------
        Continuous scenario:

        >>> scenario = SyntheticScenario(n_dims=2, extent=[0, 100, 0, 150], griddata=True)
        >>> points, values, xi, reference = scenario.simulate_scenario(n_samples=200)
        >>> fig, ax = scenario.plot_2d_scenario(points, values, xi, reference, theme='publication')
        >>> plt.show()

        Nominal categorical scenario:

        >>> scenario = SyntheticScenario(categorical=True, n_grid_points=80)
        >>> points, values, xi, reference = scenario.simulate_scenario(n_samples=200, seed=0)
        >>> fig, ax = scenario.plot_2d_scenario(points, values, xi, reference)
        >>> plt.show()

        Ordinal categorical scenario:

        >>> scenario = SyntheticScenario(categorical=True, n_grid_points=80)
        >>> points, values, xi, reference = scenario.simulate_scenario(kind='ordinal', n_samples=200, seed=0)
        >>> fig, ax = scenario.plot_2d_scenario(points, values, xi, reference,
        ...                                     nonnum_order=list(SyntheticScenario.ORDINAL_ORDER))
        >>> plt.show()
        """
        assert self.n_dims == 2, "Only 2D scenarios supported."

        with PlotStyle(theme=theme, cmap=cmap) as style:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

            if self.categorical:
                # Resolve once and reuse for both panels so the reference map
                # and the overlaid sample points agree on class -> colour.
                n_categories = len(np.unique(np.asarray(reference_values, dtype=object)))
                resolved_cmap = cmap or style.resolve_cmap('categorical', n_categories=n_categories)

                plot_colormap_data(reference_values, xi_locations=xi, ax=ax,
                                   griddata=self.griddata, data_type='categorical',
                                   cmap=resolved_cmap, nonnum_order=nonnum_order,
                                   extent=self.extent)
                # Overlay sample points colored by their class label
                plot_scatter(points, values, cmap=resolved_cmap, data_type='categorical',
                             nonnum_order=nonnum_order, fig=fig, ax=ax,
                             s=point_size * 10, edgecolors='white',
                             linewidths=0.5, extent=self.extent, show_legend=False)
            else:
                vmin, vmax = np.nanmin(reference_values), np.nanmax(reference_values)
                try:
                    plot_colormap_data(reference_values, xi_locations=xi, ax=ax,
                                        griddata=self.griddata, cmap=style.cmap,
                                        vmin=vmin, vmax=vmax, extent=self.extent)
                except (ValueError, SpatializeError) as e:
                    log_message(logging.logger.debug(
                        f"plot_colormap_data failed ({e}); falling back to plot_nongriddata"))
                    plot_nongriddata(reference_values, xi_locations=xi, ax=ax,
                                     cmap=style.cmap, vmin=vmin, vmax=vmax, extent=self.extent)

                plot_scatter(points, values, cmap=style.cmap,
                             fig=fig, ax=ax,
                             vmin=vmin, vmax=vmax,
                             s=point_size * 10, #color=point_color,
                             extent=self.extent, edgecolors='white',
                             linewidths=0.5, show_colorbar=False)

            ax.set_title(title)
            return fig, ax


# pendiente: resolver tema con extent/ xlim/ylim

class PrecipitationCaseStudy:
    """
    Real-world precipitation case study with visualization utilities.

    This class provides a complete workflow for a 2.5D (x, y, elevation) precipitation
    interpolation case study. It loads precipitation measurements from weather stations,
    defines interpolation locations with elevation data, and provides formatted
    visualization methods for results.

    The case study covers three dates with varying precipitation patterns and is
    useful for demonstrating ESI methods on real-world environmental data.

    Note: besides from spatialize's base dependencies, this case study requires for xarray, geopandas and seaborn to be installed.

    Attributes
    ----------
    locs : pd.DataFrame
        Interpolation locations with columns ['X', 'Y', 'Z'] (UTM coordinates + elevation).
    data : pd.DataFrame
        Precipitation sample data with columns ['x', 'y', 'z'] plus date columns.
    study_area : geopandas.GeoDataFrame
        Shapefile boundary of the study area for plotting.
    dates : list
        List of date strings for the case study (typically 3 dates).
    locs_array : xarray.Dataset
        Interpolation locations as xarray for plotting.
    extent : list
        Spatial extent as [xmin, xmax, ymin, ymax].
    data_cmap : matplotlib.colors.Colormap
        Colormap for precipitation values.
    locs_cmap : matplotlib.colors.Colormap
        Colormap for elevation values.
    precision_cmap : matplotlib.colors.Colormap
        Colormap for precision/uncertainty values.

    Methods
    -------
    model_inputs()
        Get formatted inputs for ESI functions (points, values, xi).
    plot_input_data()
        Visualize precipitation samples for all dates.
    plot_interpolation_locations()
        Visualize interpolation grid with elevation.
    plot_esi_results(results, parameters)
        Create comprehensive 3x3 plot of inputs, estimates, and precision.
    plot_model_comparison(results, date, interpolators, names)
        Compare multiple interpolation methods side-by-side.

    Examples
    --------
    Basic workflow for the precipitation case study:

    >>> case_study = PrecipitationCaseStudy()
    >>> points, values, xi = case_study.model_inputs()
    >>>
    >>> # Run ESI for each date
    >>> results = {}
    >>> for date in case_study.dates:
    ...     result = esi_nongriddata(points[date], values[date], xi,
    ...                              local_interpolator='idw', exponent=2.0)
    ...     results[date] = result
    >>>
    >>> # Visualize results
    >>> case_study.plot_input_data()
    >>> case_study.plot_esi_results(results_df, parameters_df)

    Notes
    -----
    Data files are bundled with the package under ``spatialize/resources/data/``:
    - PP_locations.csv: Interpolation grid with elevation
    - PP_samples.csv: Precipitation measurements at stations
    - PP_Basin/Basin_UTM.shp: Study area boundary shapefile

    The data uses UTM coordinates (meters) and elevation in meters above sea level.
    Precipitation values are in millimeters (mm).

    See Also
    --------
    SyntheticScenario : Generate synthetic test scenarios
    spatialize.gs.esi.esi_nongriddata : ESI for scattered points
    """
    def __init__(self, locs_cmap=None, data_cmap=None, precision_cmap=None):
        self._load_data()
        self._setup_environment(locs_cmap, data_cmap, precision_cmap)

    def _load_data(self):
        """
        Load precipitation case study data files.

        Loads interpolation locations, precipitation samples, and study area
        boundary from the data directory.
        """
        import geopandas as gpd

        base = str(rs.files(data))

        # Interpolation locations for the 'xi' input:
        self.locs = pd.read_csv(os.path.join(base, 'PP_locations.csv'))

        # Precipitation data:
        self.data = pd.read_csv(os.path.join(base, 'PP_samples.csv'))

        # Outline of study area
        self.study_area = gpd.read_file(os.path.join(base, 'PP_Basin', 'Basin_UTM.shp'))

    def _setup_environment(self,
                           locs_cmap=None,
                           data_cmap=None,
                           precision_cmap=None):
        """
        Set up plot configurations and colormaps.

        Configures matplotlib settings for consistent visualization and creates
        custom colormaps for precipitation, elevation, and precision plots.

        Parameters
        ----------
        locs_cmap : matplotlib.colors.Colormap, optional
            Custom colormap for elevation. If None, uses seaborn blend. Default is None.
        data_cmap : matplotlib.colors.Colormap, optional
            Custom colormap for precipitation. If None, uses seaborn blend. Default is None.
        precision_cmap : matplotlib.colors.Colormap, optional
            Custom colormap for precision. If None, uses seaborn blend. Default is None.
        """
        # Matplotlib configuration
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': 'DejaVu Serif',
            'grid.alpha': 0.6,
            'axes.titleweight': 'demibold',
            'axes.titlesize': 11,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'figure.titleweight': 'bold',
            'figure.titlesize': 13,
            'savefig.bbox': 'tight'
            })
        
        # List of studied dates (to iterate on)
        self.dates = list(self.data.columns)[3:6]

        # For imshow plots
        self.locs_array = self.locs.set_index(['X', 'Y']).to_xarray()
        self.extent = _calculate_extent(self.locs)

        # Colormaps
        import seaborn as sns

        self.data_cmap = data_cmap or sns.color_palette("blend:#ffe370,#84ffb8,#38b2ff,#009abe,#006a83,#004252", as_cmap=True)
        self.data_cmap.set_under('#ffce06')
        self.data_cmap.set_over('#00242d')

        self.locs_cmap = locs_cmap or sns.color_palette("blend:#80a74f,#bddb97,#e3cfb2,#d6b78c,#c9a066,#ba8842,#946c35,#825f2e,#332512", as_cmap=True)

        self.precision_cmap = precision_cmap or sns.color_palette("blend:#4b7f52,#7dd181,#b6f9c9,#ffdf80,#f8a07f,#f57d4e,#d2430b", as_cmap=True)
        self.precision_cmap.set_over('#b93007')
    
    def model_inputs(self):
        """
        Get formatted inputs for ESI interpolation functions.

        Extracts and formats the data into the structure expected by ESI functions.
        Sample points and values are organized by date, while interpolation locations
        are shared across all dates.

        Returns
        -------
        points : dict
            Dictionary mapping date strings to sample coordinate arrays.
            Each array has shape (n_samples, 3) with columns [x, y, z].
        values : dict
            Dictionary mapping date strings to precipitation value arrays.
            Each array has shape (n_samples,) with precipitation in mm.
        xi : ndarray, shape (n_locations, 3)
            Interpolation locations with columns [X, Y, Z] (UTM + elevation).

        Examples
        --------
        >>> case_study = PrecipitationCaseStudy()
        >>> points, values, xi = case_study.model_inputs()
        >>> print(f"Dates: {list(points.keys())}")
        >>> print(f"Samples for first date: {points[case_study.dates[0]].shape}")
        >>> print(f"Interpolation locations: {xi.shape}")
        """
        # Interpolation locations:
        xi = self.locs[['X', 'Y', 'Z']].values

        # Samples:
        points = {}
        values = {}

        for date in self.dates:
            mask = ~np.isnan(self.data[date])        # Mask for non-NaN values
            points[date] = self.data[['x', 'y', 'z']][mask].values
            values[date] = self.data[date][mask].values

        return points, values, xi
    
    def _plot_format(self, ax):
        """
        Apply consistent formatting to precipitation case study plots.

        Sets axis limits, tick locations, and overlays the study area boundary.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to format.
        """
        ax.set_xticks(ticks=[250000, 310000, 370000, 430000], labels=['250000', '310000', '370000', '430000'])
        ax.set_yticks(ticks=[6200000, 6240000, 6280000, 6320000, 6360000], labels=['6200000', '6240000', '6280000', '6320000', '6360000'])
        ax.set_xlim([250000 - 6000, 430000 + 6000])
        ax.set_ylim([6200000, 6360000])

        self.study_area.boundary.plot(ax=ax, color='black', linewidth=0.5)

    def plot_input_data(self):
        """
        Plot precipitation sample data for all dates.

        Creates a figure with three subplots showing the spatial distribution of
        precipitation measurements for each date in the case study.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        axs : array of matplotlib.axes.Axes
            Array of axes for each date subplot.

        Examples
        --------
        >>> case_study = PrecipitationCaseStudy()
        >>> fig, axs = case_study.plot_input_data()
        >>> plt.show()
        """
        fig, axs = plt.subplots(1, 3, figsize=(9.8, 2.8), sharey=True, width_ratios=[1,1,1.2])#, layout='compressed')

        for i, date in enumerate(self.dates):
            cbar = (i==len(self.dates)-1)
            plot_scatter(self.data[['x', 'y']].values, self.data[date].values,
                         cmap=self.data_cmap, fig=fig, ax=axs[i],
                         xlabel=None, ylabel=None,
                         show_colorbar=cbar, vmin=0, vmax=100)
            self._plot_format(axs[i])
            axs[i].set_title(date)

        fig.suptitle('Precipitation Data')
        axs[0].set_ylabel('UTM North')
        axs[1].set_xlabel('UTM East')

        #cbar1 = plt.colorbar(axs[-1].collections[-1], ax=axs[2], aspect=10)
        #cbar1.set_label('Precipitation [mm]', fontsize=11)

        return fig, axs
    
    def plot_interpolation_locations(self):
        """
        Plot interpolation grid with elevation values.

        Visualizes the spatial distribution of interpolation locations colored by
        elevation, providing context for the topography of the study area.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        ax : matplotlib.axes.Axes
            The created axes.

        Examples
        --------
        >>> case_study = PrecipitationCaseStudy()
        >>> fig, ax = case_study.plot_interpolation_locations()
        >>> plt.show()
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), layout='compressed')

        plot_nongriddata(self.locs['Z'].values, xi_locations=self.locs[['X', 'Y']].values,
                         ax=ax, cmap=self.locs_cmap, vmin=0, vmax=6000, zorder=1, extent=self.extent, cbar_label='Elevation')
        self._plot_format(ax)

        fig.suptitle('Interpolation Locations')
        ax.set_ylabel('UTM North')
        ax.set_xlabel('UTM East')

        return fig, ax
    
    def plot_esi_results(self, results,
                         parameters=None,
                         local_interpolator='idw',
                         precision_function='Operational Error',
                         fig_title='ESI Results',
                         dpi=100):
        """
        Generate comprehensive 3x3 visualization of ESI results.

        Creates a publication-quality figure showing input data, ESI estimates, and
        precision maps for all three dates in the case study. Each row represents
        a different aspect (inputs, estimates, precision) and each column represents
        a different date.

        Parameters
        ----------
        results : pd.DataFrame
            DataFrame with MultiIndex columns (date, 'value'/'precision') and
            index columns ['X', 'Y'] matching interpolation locations.
        parameters : pd.DataFrame
            DataFrame with dates as index and parameter columns (e.g., 'alpha',
            'exponent' for IDW or 'model', 'nugget', 'range' for Kriging).
        local_interpolator : str, optional
            Type of local interpolator used ('idw' or 'kriging'). Affects parameter
            annotation formatting. Default is 'idw'.
        precision_function : str, optional
            Name of precision metric for colorbar label. Default is 'Operational Error'.
        fig_title : str, optional
            Overall figure title. Default is 'ESI Results'.
        dpi : int, optional
            Figure resolution in dots per inch. Default is 100.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        axs : ndarray of matplotlib.axes.Axes
            3x3 array of axes.

        Examples
        --------
        >>> case_study = PrecipitationCaseStudy()
        >>> # ... run ESI for all dates and organize results ...
        >>> fig, axs = case_study.plot_esi_results(results_df, params_df)
        >>> plt.savefig('esi_results.png', dpi=300, bbox_inches='tight')
        """
        locs_xy = self.locs[['X', 'Y']].values

        fig, axs = plt.subplots(3, 3, figsize=(11, 8.4), dpi=dpi, sharex=True, sharey=True, layout='compressed')
        fig.suptitle(fig_title, fontsize=14)

        for j, date in enumerate(self.dates):
            cbar = (j==len(self.dates)-1)

            axs[0, j].set_title(date, fontsize=12)
            for i in range(3):
                self._plot_format(axs[i, j])

            # Input data: elevation background + precipitation scatter
            plot_nongriddata(self.locs['Z'].values, xi_locations=locs_xy,
                             ax=axs[0, j], cmap=self.locs_cmap, vmin=0, vmax=6000,
                             show_colorbar=False, zorder=1, extent=self.extent)
            plot_scatter(self.data[['x', 'y']].values, self.data[date].values,
                         cmap=self.data_cmap, fig=fig, ax=axs[0, j],
                         show_colorbar=cbar, vmin=0, vmax=100,
                         edgecolors='white', linewidths=0.5, extent=self.extent)

            # ESI estimate
            plot_nongriddata(results[(date, 'value')].values, xi_locations=locs_xy,
                             ax=axs[1, j], cmap=self.data_cmap, vmin=0, vmax=100, 
                             show_colorbar=cbar, extent=self.extent)

            # Annotate parameters
            if parameters is not None:
                if local_interpolator == 'idw':
                    axs[1, j].text(252000, 6340000,
                                   f"alpha = {parameters.loc[date, 'alpha']}\nexp = {parameters.loc[date, 'exponent']}",
                                   fontsize=8.5)
                elif local_interpolator == 'kriging':
                    axs[1, j].text(252000, 6205000,
                                   f"alpha = {parameters.loc[date, 'alpha']}\nmodel = {parameters.loc[date, 'model']}\nnugget = {parameters.loc[date, 'nugget']}\nrange = {parameters.loc[date, 'range']}",
                                   fontsize=7.5)

            # Precision
            plot_nongriddata(results[(date, 'precision')].values, xi_locations=locs_xy,
                             ax=axs[2, j], cmap=self.precision_cmap, vmin=0, vmax=0.3, 
                             show_colorbar=cbar, extent=self.extent)

        # Titles and labels
        axs[0, 0].set_ylabel('Input Data', labelpad=20, fontweight='bold')
        axs[1, 0].set_ylabel('ESI Estimates', labelpad=20, fontweight='bold')
        axs[2, 0].set_ylabel('ESI Precision', labelpad=20, fontweight='bold')

        axs[2, 1].set_xlabel('UTM East', labelpad=8, fontsize=11)
        fig.text(0.035, 0.5, 'UTM North', rotation='vertical', fontsize=11, ha='center', va='center')

        return fig, axs
    
    def plot_model_comparison(self, results, date,
                              interpolators=['nearest', 'rbf', 'kriging', 'esi'],
                              names=['Nearest Neighbor', 'RBF', 'Kriging', 'ESI'],
                              dpi=100):
        """
        Compare multiple interpolation methods side-by-side for a single date.

        Creates a visualization comparing different interpolation approaches, showing
        the input data and elevation on the left and interpolation results from
        different methods on the right.

        Parameters
        ----------
        results : pd.DataFrame
            DataFrame with MultiIndex columns (interpolator, date, 'value') and
            index columns ['X', 'Y'].
        date : str
            Date to visualize (must be one of self.dates).
        interpolators : list of str, optional
            List of interpolator names matching first level of results columns.
            Default is ['nearest', 'rbf', 'kriging', 'esi'].
        names : list of str, optional
            Display names for each interpolator (same order as interpolators).
            Default is ['Nearest Neighbor', 'RBF', 'Kriging', 'ESI'].
        dpi : int, optional
            Figure resolution in dots per inch. Default is 100.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        subfigs : array of matplotlib.figure.SubFigure
            Array containing [input_subfig, results_subfig].
        ax0 : matplotlib.axes.Axes
            Axes for input data visualization.
        axs : ndarray of matplotlib.axes.Axes
            2xN array of axes for interpolation method results.

        Examples
        --------
        >>> case_study = PrecipitationCaseStudy()
        >>> # ... run multiple interpolation methods and organize results ...
        >>> fig, subfigs, ax0, axs = case_study.plot_model_comparison(
        ...     results_df, date='2021-01-15',
        ...     interpolators=['idw', 'kriging', 'esi'],
        ...     names=['IDW', 'Kriging', 'ESI'])
        >>> plt.show()
        """
        locs_xy = self.locs[['X', 'Y']].values
        n_models = len(interpolators)

        fig = plt.figure(layout='constrained', figsize=(8 + n_models, 5.8), dpi=dpi)
        fig.suptitle(f'Estimates for {date}', fontsize=16)
        subfigs = fig.subfigures(1, 2, width_ratios=[1, (2.2/3) * (n_models/2)], wspace=0.03)

        # === PRECIPITATION DATA + INTERPOLATION LOCATIONS ===
        ax0 = subfigs[0].add_subplot(111)
        self._plot_format(ax0)
        ax0.text(252000, 6340000, f"min = {self.data[date].min()}\nmax = {self.data[date].max()}",
                 fontsize=9, zorder=4)
        ax0.set_title('Inputs', fontsize=13, pad=8)

        plot_nongriddata(self.locs['Z'].values, xi_locations=locs_xy,
                         ax=ax0, cmap=self.locs_cmap, vmin=0, vmax=6000,
                         show_colorbar=False, extent=self.extent)
        plot_scatter(self.data[['x', 'y']].values, self.data[date].values,
                     cmap=self.data_cmap, fig=subfigs[0], ax=ax0,
                     show_colorbar=False, vmin=0, vmax=100,
                     edgecolors='white', linewidths=0.6, extent=self.extent)

        # === INTERPOLATORS ===
        import math
        axs = subfigs[1].subplots(nrows=2, ncols=math.ceil(n_models / 2), sharex='all', sharey='all')

        result_rows = results[results['date'] == date] if 'date' in results.columns else results
        for i, ax in enumerate(axs.flatten()):
            if i >= n_models:
                ax.set_visible(False)
                continue
            interp_vals = result_rows[(interpolators[i], date, 'value')].values \
                if (interpolators[i], date, 'value') in result_rows.columns \
                else result_rows[interpolators[i]].values
            plot_nongriddata(interp_vals, xi_locations=locs_xy, 
                             ax=ax, cmap=self.data_cmap, vmin=0, vmax=100,
                             show_colorbar=False, extent=self.extent)
            ax.set_title(names[i], fontsize=13, pad=8)
            self._plot_format(ax)
            ax.text(252000, 6340000,
                    f"min = {float(np.nanmin(interp_vals)):.1f}\nmax = {float(np.nanmax(interp_vals)):.1f}",
                    fontsize=8)

        # Colorbars
        cbar = plt.colorbar(ax0.collections[-1], ax=ax0, aspect=14, location='bottom', extend='both')
        cbar.set_label('Precipitation [mm]')
        cbar1 = plt.colorbar(ax0.images[-1], ax=ax0, aspect=14, location='bottom', pad=0.12)
        cbar1.set_label('Elevation [m]')

        # Labels
        ax0.set_xlabel('UTM East', fontsize=11)
        ax0.set_ylabel('UTM North', fontsize=11)
        axs[1, 1].set_xlabel('UTM East', fontsize=11)
        subfigs[1].supylabel('UTM North', fontsize=11)

        return fig, subfigs, ax0, axs

class SuppressOutput:
    """
    Context manager to temporarily suppress stdout output.

    Redirects stdout to /dev/null for the duration of the context, suppressing
    all print statements and stdout output. Useful for silencing verbose library
    functions during batch processing.

    Examples
    --------
    >>> with SuppressOutput():
    ...     print("This will not be displayed")
    ...     # Any function calls here will have stdout suppressed
    >>> print("This will be displayed normally")
    This will be displayed normally

    Warnings
    --------
    This suppresses ALL stdout within the context, including error messages.
    Use with caution and ensure proper error handling.
    """
    def __enter__(self):
        """Enter the context and redirect stdout to /dev/null."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self

    def __exit__(self, *args):
        """Exit the context and restore stdout."""
        sys.stdout.close()
        sys.stdout = self._original_stdout

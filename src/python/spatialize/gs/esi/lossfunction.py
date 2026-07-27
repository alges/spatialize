import numpy as np
from spatialize.gs.esi.aggfunction import mean, identity


def loss(agg_function):
    """Decorator turning a pointwise loss into a ``(estimation, esi_samples) -> array`` callable.

    Wraps a pointwise loss function of two point-estimate arguments (e.g.
    ``(x, y) -> (x - y) ** 2``) so that it can be used as the `loss_function`
    argument of :meth:`~spatialize.gs.esi.ESIResult.precision` /
    :meth:`~spatialize.gs.esi.ESIResult.precision_cube`. The decorated
    function computes the pointwise loss between the estimate and every ESI
    sample, then aggregates the per-partition losses with `agg_function`.

    Parameters
    ----------
    agg_function : callable
        Aggregation function ``(loss_cube) -> array`` applied over the
        ensemble/partition axis of the per-partition losses, e.g.
        :func:`~spatialize.gs.esi.aggfunction.mean` (to average the loss
        over partitions) or :func:`~spatialize.gs.esi.aggfunction.identity`
        (to leave it as a per-partition cube).

    Returns
    -------
    callable
        A decorator that, applied to a pointwise loss function, returns a
        callable instance implementing ``(estimation, esi_samples) -> array``.

    See Also
    --------
    spatialize.gs.esi.aggfunction : Aggregation functions usable as
        `agg_function` here.
    spatialize.gs.esi.scorefunction : Cross-validation scores computed
        directly from the raw ESI samples.
    """
    def outer_function(function):
        function_name = function.__name__
        module_name = function.__module__

        class inner_function:
            def __call__(self, estimation, esi_samples):
                return _apply_loss_function(estimation, esi_samples,
                                            function,
                                            agg_function)

            def __repr__(self):
                return f"<decorated--{module_name}.{function_name}>"

        return inner_function()

    return outer_function


@loss(mean)
def mse_loss(x, y):
    """Mean Squared Error between an ESI sample column and the estimate.

    Used as the `loss_function` argument of
    :meth:`~spatialize.gs.esi.ESIResult.precision` (the default), and
    internally by :func:`_apply_loss_function` where `x` is one column of
    `esi_samples` and `y` is the point estimate; the per-partition squared
    errors are then averaged over partitions via :func:`~spatialize.gs.esi.aggfunction.mean`.

    Parameters
    ----------
    x : array_like, shape (n_points,)
        Values from one ESI partition (one column of the ESI samples).
    y : array_like, shape (n_points,)
        The point estimate to compare against.

    Returns
    -------
    ndarray, shape (n_points,)
        The pointwise squared error ``(x - y) ** 2``.

    See Also
    --------
    spatialize.gs.esi.aggfunction : Aggregation functions used to collapse
        this loss over the ensemble.
    spatialize.gs.esi.scorefunction : Cross-validation scores computed
        directly from the raw ESI samples.
    """
    return (x - y) ** 2


@loss(mean)
def mae_loss(x, y):
    """Mean Absolute Error between an ESI sample column and the estimate.

    Used as the `loss_function` argument of
    :meth:`~spatialize.gs.esi.ESIResult.precision`; internally by
    :func:`_apply_loss_function`, where `x` is one column of `esi_samples`
    and `y` is the point estimate, with the per-partition absolute errors
    averaged over partitions via :func:`~spatialize.gs.esi.aggfunction.mean`.

    Parameters
    ----------
    x : array_like, shape (n_points,)
        Values from one ESI partition (one column of the ESI samples).
    y : array_like, shape (n_points,)
        The point estimate to compare against.

    Returns
    -------
    ndarray, shape (n_points,)
        The pointwise absolute error ``|x - y|``.

    See Also
    --------
    spatialize.gs.esi.aggfunction : Aggregation functions used to collapse
        this loss over the ensemble.
    spatialize.gs.esi.scorefunction : Cross-validation scores computed
        directly from the raw ESI samples.
    """
    return np.abs(x - y)


@loss(identity)
def mse_cube(x, y):
    """Mean Squared Error between an ESI sample column and the estimate, unaggregated.

    Same pointwise error as :func:`mse_loss`, but left unaggregated over
    partitions (via :func:`~spatialize.gs.esi.aggfunction.identity`) so it
    can be used as the `loss_function` argument of
    :meth:`~spatialize.gs.esi.ESIResult.precision_cube` to obtain the
    full per-partition loss cube.

    Parameters
    ----------
    x : array_like, shape (n_points,)
        Values from one ESI partition (one column of the ESI samples).
    y : array_like, shape (n_points,)
        The point estimate to compare against.

    Returns
    -------
    ndarray, shape (n_points,)
        The pointwise squared error ``(x - y) ** 2``.

    See Also
    --------
    spatialize.gs.esi.aggfunction : Aggregation functions used to collapse
        this loss over the ensemble.
    spatialize.gs.esi.scorefunction : Cross-validation scores computed
        directly from the raw ESI samples.
    """
    return mse_loss(x, y)


@loss(identity)
def mae_cube(x, y):
    """Mean Absolute Error between an ESI sample column and the estimate, unaggregated.

    Same pointwise error as :func:`mae_loss`, but left unaggregated over
    partitions (via :func:`~spatialize.gs.esi.aggfunction.identity`) so it
    can be used as the `loss_function` argument of
    :meth:`~spatialize.gs.esi.ESIResult.precision_cube` to obtain the
    full per-partition loss cube.

    Parameters
    ----------
    x : array_like, shape (n_points,)
        Values from one ESI partition (one column of the ESI samples).
    y : array_like, shape (n_points,)
        The point estimate to compare against.

    Returns
    -------
    ndarray, shape (n_points,)
        The pointwise absolute error ``|x - y|``.

    See Also
    --------
    spatialize.gs.esi.aggfunction : Aggregation functions used to collapse
        this loss over the ensemble.
    spatialize.gs.esi.scorefunction : Cross-validation scores computed
        directly from the raw ESI samples.
    """
    return mae_loss(x, y)


class OperationalErrorLoss:
    """Callable family of relative-MAE loss functions indexed by a dynamic range.

    Computes the Mean Absolute Error between the estimate and the ESI
    samples, normalized by a `dyn_range` (the estimate's operational
    dynamic range) so the resulting error is expressed as a fraction of
    that range rather than in the variable's raw units. Instances are used
    as the `loss_function` argument of
    :meth:`~spatialize.gs.esi.ESIResult.precision` /
    :meth:`~spatialize.gs.esi.ESIResult.precision_cube`.

    Parameters
    ----------
    dyn_range : float or None, optional
        The dynamic range to normalize by. If ``None``, it is computed at
        call time as the difference between the maximum and minimum of the
        estimate. Default: ``None``.
    use_cube : bool, optional
        If ``True``, the loss is left unaggregated over partitions (as in
        :func:`mse_cube` / :func:`mae_cube`), suitable for `precision_cube`.
        If ``False``, it is averaged over partitions (as in :func:`mse_loss`
        / :func:`mae_loss`), suitable for `precision`. Default: ``False``.

    See Also
    --------
    spatialize.gs.esi.aggfunction : Aggregation functions used to collapse
        this loss over the ensemble.
    spatialize.gs.esi.scorefunction : Cross-validation scores computed
        directly from the raw ESI samples.
    """
    def __init__(self, dyn_range=None, use_cube=False):
        """Store the normalization and aggregation configuration.

        Parameters
        ----------
        dyn_range : float or None, optional
            The dynamic range to normalize the MAE by. If ``None``, it is
            computed at call time from the estimate. Default: ``None``.
        use_cube : bool, optional
            If ``True``, return the unaggregated per-partition loss cube;
            if ``False``, average over partitions. Default: ``False``.
        """
        self.use_cube = use_cube
        self.dyn_range = dyn_range

    def __call__(self, estimation, esi_samples):
        """Compute the relative MAE between the estimate and the ESI samples.

        Parameters
        ----------
        estimation : array_like, shape (n_points,)
            The point estimate to compare against.
        esi_samples : array_like, shape (n_points, n_partitions)
            The raw ESI samples.

        Returns
        -------
        ndarray
            The relative MAE, aggregated over partitions to shape
            ``(n_points,)`` if `use_cube` is ``False``, or left as a cube of
            shape ``(n_points, n_partitions)`` if `use_cube` is ``True``.
        """
        dyn_range = self.dyn_range
        if dyn_range is None:
            dyn_range = np.abs(np.min(estimation) - np.max(estimation))

        def relative_mae(x, y):
            return np.abs(x - y) / dyn_range

        @loss(identity)
        def _op_error_cube(x, y):
            return relative_mae(x, y)

        @loss(mean)
        def _op_error_aggregated(x, y):
            return relative_mae(x, y)

        _op_error = _op_error_aggregated
        if self.use_cube:
            _op_error = _op_error_cube

        return _op_error(estimation, esi_samples)


def _apply_loss_function(estimation, esi_samples, loss_function, agg_function):
    loss = np.empty(esi_samples.shape)
    for i in range(loss.shape[1]):
        loss[:, i] = loss_function(esi_samples[:, i], estimation)

    return agg_function(loss)

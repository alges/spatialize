"""Co-estimation: a variable predicted from several others through two stages of ensembles
(experimental, see :mod:`spatialize.futures`).

The method works in two stages.

1. **Marginal stage.** Each variable, with its own data, is estimated by its own ensemble (Mondrian
   partitions and the decoder ``estimation``) at ``n_aux`` auxiliary locations drawn uniformly in
   the box of the data and the queries, with the Mondrian rate set on the box of the data. The members of each variable at each auxiliary location
   are reduced to one value by ``aggregation``.
2. **Joint stage.** A second ensemble of Mondrian partitions is drawn on the auxiliary locations.
   In each of its cells, ``co_estimation`` receives the values of every variable at the auxiliary
   locations of the cell and predicts one value at each query of the cell.

The result holds one member per partition of the joint stage at each query, a single predicted
variable.

The method is under study, with known limits.

- The theory of the library treats a single field, so the two stages have no derivation from it.
- The marginal laws are reduced to one value per auxiliary location before the joint stage, so the
  uncertainty of the first stage does not reach the result.
- The auxiliary locations are uniform in the box, whatever the sampling of the data.
- Only Mondrian partitions are used, outside :func:`~spatialize.gs.esi.esi_griddata`'s engine: the
  session settings (domain, threads, cells without data) do not apply, the computation running on one thread.
- Every variable must have the same number of data, at locations of its own.
- Cross-validation is marginal only, variable by variable (:func:`coesi_marginal_cv`).
"""
import numpy as np

import libspatialize as lsp

from spatialize import SpatializeError
from spatialize.futures import _experimental
from spatialize.gs.esi import ESIResult
from spatialize.logging import singleton_null_callback


def _stack(points, values):
    """The data of the variables as the engine takes them, (variables, n, d) and (variables, n)."""
    points = [np.asarray(p, dtype=np.float32) for p in points]
    values = [np.asarray(v, dtype=np.float32).ravel() for v in values]
    if len(points) != len(values) or not points:
        raise SpatializeError("points and values must give the same number of variables (at least one)")
    sizes = {len(v) for v in values} | {p.shape[0] for p in points}
    if len(sizes) != 1:
        raise SpatializeError("co-estimation needs the same number of data for every variable "
                              f"(got {sorted(sizes)})")
    if len({p.shape[1] for p in points}) != 1:
        raise SpatializeError("the locations of every variable must have the same number of coordinates")
    return np.stack(points), np.stack(values)


def coesi_nongriddata(points, values, xi, co_estimation, estimation, aggregation=np.nanmean,
                      co_post_creation=None, post_creation=None, n_partitions=100, alpha=0.8, seed=None,
                      n_aux=100, callback=None):
    """Co-estimation at a list of locations.

    Parameters
    ----------
    points : sequence of array_like
        The data locations of each variable, one array of shape :math:`(n, d)` per variable.
    values : sequence of array_like
        The data values of each variable, one array of shape :math:`(n,)` per variable.
    xi : array_like
        The locations to estimate, shape :math:`(m, d)`.
    co_estimation : callable
        ``co_estimation(aux_points, aux_values, queries, params) -> predictions``, the joint stage
        in one cell: ``aux_values`` has one column per variable, one row per auxiliary location of
        the cell. Returns one value per query.
    estimation : callable
        ``estimation(points, values, queries, params) -> predictions``, the decoder of the marginal
        stage, applied to each variable in each cell, as in :doc:`/development/python_decoders`.
    aggregation : callable, optional
        ``aggregation(members) -> float``, the reduction of a variable's members at an auxiliary
        location. Default: :func:`numpy.nanmean`.
    co_post_creation, post_creation : callable, optional
        The per-cell parameters of the joint and of the marginal stage,
        ``(points, values) -> params``. Default: none.
    n_partitions : int, optional
        The number of partitions of every ensemble. Default: 100.
    alpha : float, optional
        The granularity of the Mondrian partitions. Default: 0.8.
    seed : int, optional
        The seed of the run. Default: drawn at random.
    n_aux : int, optional
        The number of auxiliary locations of the joint stage. Default: 100.
    callback : callable, optional
        Where the progress and the messages go. Default: ``None``, shown as the session settings
        ``display``, ``progress`` and ``verbosity`` say (:mod:`spatialize.session`). A callable
        receiving the messages of :mod:`spatialize.logging` sends them elsewhere, such as an
        application's own interface, while :func:`~spatialize.logging.singleton_null_callback`
        drops them.

    Returns
    -------
    ESIResult
        The members of the joint stage at each location and their mean.
    """
    _experimental("Co-estimation (spatialize.futures.coesi)")
    smp, val = _stack(points, values)
    xi = np.asarray(xi, dtype=np.float32)
    seed = int(np.random.randint(1000, 10000)) if seed is None else int(seed)
    _, members = lsp.estimation_custom_coesi(
        smp, val, int(n_partitions), float(alpha), seed, xi, co_post_creation, co_estimation, post_creation,
        estimation, lambda m: float(aggregation(m)), callback or singleton_null_callback, n_aux=int(n_aux))
    members = np.asarray(members)
    return ESIResult(np.nanmean(members, axis=1), members, xi=xi)


def coesi_marginal_cv(points, values, method="loo", loo=None, kfold=None, post_creation=None,
                      n_partitions=100, alpha=0.8, seed=None, k=5, folding_seed=None, callback=None):
    """Cross-validation of the marginal stage of co-estimation, variable by variable.

    Parameters
    ----------
    points, values : sequence of array_like
        As in :func:`coesi_nongriddata`.
    method : {"loo", "kfold"}, optional
        Leave-one-out (default) or k-fold.
    loo : callable
        ``loo(points, values, params) -> predictions`` in one cell, one per datum; required for
        ``"loo"``.
    kfold : callable
        ``kfold(k, points, values, folds, params) -> predictions`` in one cell; required for
        ``"kfold"``.
    post_creation : callable, optional
        The per-cell parameters, ``(points, values) -> params``.
    n_partitions, alpha, seed : optional
        As in :func:`coesi_nongriddata`.
    k, folding_seed : int, optional
        The number of folds and the seed of their assignment, for ``"kfold"``.
    callback : callable, optional
        Where the progress and the messages go. Default: ``None``, shown as the session settings
        ``display``, ``progress`` and ``verbosity`` say (:mod:`spatialize.session`). A callable
        receiving the messages of :mod:`spatialize.logging` sends them elsewhere, such as an
        application's own interface, while :func:`~spatialize.logging.singleton_null_callback`
        drops them.

    Returns
    -------
    ndarray
        Shape :math:`(\\text{variables}, n, \\text{n\\_partitions})`, the cross-validation members of
        each datum of each variable.
    """
    _experimental("Co-estimation (spatialize.futures.coesi)")
    smp, val = _stack(points, values)
    seed = int(np.random.randint(1000, 10000)) if seed is None else int(seed)
    queries = smp[0]
    visitor = callback or singleton_null_callback
    if method == "loo":
        if loo is None:
            raise SpatializeError("method='loo' needs the function 'loo'")
        _, members = lsp.marginal_loo_custom_coesi(smp, val, int(n_partitions), float(alpha), seed, queries,
                                                   post_creation, loo, visitor)
    elif method == "kfold":
        if kfold is None:
            raise SpatializeError("method='kfold' needs the function 'kfold'")
        folding_seed = int(np.random.randint(1000, 10000)) if folding_seed is None else int(folding_seed)
        _, members = lsp.marginal_kfold_custom_coesi(smp, val, int(n_partitions), float(alpha), seed, int(k),
                                                     folding_seed, queries, post_creation, kfold, visitor)
    else:
        raise SpatializeError(f"method must be 'loo' or 'kfold'; got {method!r}")
    return np.asarray(members)

"""Facade between the Python API and the compiled extension ``libspatialize``.

The extension describes what it offers in ``libspatialize.catalog()``: every partition (encoder)
and decoder, the dimensions each supports and its parameters. Everything here derives from that
catalogue, so a partition or decoder registered in C++ is available to the Python API without
being listed again.

The public functions build their argument lists with ``build_arg_list`` (``gs/esi/_main.py``), in
the order ``[samples, values, n_partitions, alpha, <decoder arguments, seed>, (k, folding_seed,)
queries, callback]``. :meth:`lib_spatialize_facade.get_operator` returns a callable that takes that
list and runs ``libspatialize.run``, within the session settings (:mod:`spatialize.session`).
"""
import libspatialize as lsp

import numpy as np

from spatialize import SpatializeError, logging, session
from spatialize.logging import log_message


class local_interpolator:
    IDW, KRIGING, ADAPTIVE_IDW = "idw", "kriging", "adaptiveidw"


class partitioning_process:
    MONDRIAN, VORONOI = "mondrian", "voronoi"
    MONDRIAN_RAW = "mondrian-raw"  # the theory's Mondrian process (opt-in)


#: plain (non-ensemble) IDW, a separate engine outside the catalogue
PLAIN_INTERPOLATOR = "plain"

#: the extension's catalogue: partitions, decoders, methods
CATALOG = lsp.catalog()
_PARTITIONS = {p["name"]: p for p in CATALOG["partitions"]}
_DECODERS = {d["name"]: d for d in CATALOG["decoders"]}

_PLAIN = {"estimate": lsp.estimation_nn_idw, "loo": lsp.loo_nn_idw, "kfold": lsp.kfold_nn_idw}


def _dims_text(dims):
    lo, hi = dims
    return f"{lo} or more" if hi is None else (f"{lo}" if lo == hi else f"{lo} to {hi}")


def _check(kind, specs, name, d):
    if name not in specs:
        raise SpatializeError(f"{kind} '{name}' not supported (available: {', '.join(specs)})")
    lo, hi = specs[name]["dims"]
    if d < lo or (hi is not None and d > hi):
        raise SpatializeError(f"{kind} '{name}' is available for {_dims_text(specs[name]['dims'])} "
                              f"dimensions, not {d}")


def supports(partition, decoder, d):
    """Whether the partition and the decoder are both available in ``d`` dimensions."""
    try:
        _check("Partitioning process", _PARTITIONS, partition, d)
        _check("Local interpolator", _DECODERS, decoder, d)
    except SpatializeError:
        return False
    return True


class lib_spatialize_facade:
    get_partitions_using_esi = lsp.get_partitions_using_esi
    get_leaf_for_samples_using_esi = lsp.get_leaf_for_samples_using_esi

    @classmethod
    def get_operator(cls, points, local_interpolator, operation, partitioning_process):
        """The compiled operation for a decoder on a partition, taking the legacy argument list.

        Parameters
        ----------
        points : ndarray of shape (n, d)
            The samples, whose number of columns sets the dimension.
        local_interpolator : str
            A decoder of the catalogue.
        operation : {"estimate", "loo", "kfold"}
        partitioning_process : str
            A partition of the catalogue, or ``PLAIN_INTERPOLATOR`` for plain IDW.

        Raises
        ------
        SpatializeError
            If the combination is not available in this dimension or the operation is unknown.
        """
        d = int(points.shape[1])
        if operation not in CATALOG["methods"]:
            raise SpatializeError(f"Operation '{operation}' not supported (available: {', '.join(CATALOG['methods'])})")
        if partitioning_process == PLAIN_INTERPOLATOR:
            return _PLAIN[operation]
        _check("Partitioning process", _PARTITIONS, partitioning_process, d)
        _check("Local interpolator", _DECODERS, local_interpolator, d)
        log_message(logging.logger.debug(f"esi operation: {operation}; partition: {partitioning_process}; "
                                          f"local interpolator: {local_interpolator}"))
        return _in_session_domain(_through_run(partitioning_process, local_interpolator, operation),
                                  operation, queries_at=-2)

    @classmethod
    def get_custom_esi_operator(cls):
        """Estimation with a decoder given by Python callables, on the Mondrian partition.

        The returned callable takes ``(samples, values, n_partitions, alpha, seed, queries,
        post_creation, estimation, callback)``.
        """
        def call(samples, values, n_partitions, alpha, seed, queries, post_creation, estimation, callback):
            return _run("mondrian", "custom", "estimate", samples, values, queries, alpha, n_partitions, seed,
                        {"post_creation": post_creation, "estimation": estimation}, 0, 0, callback)
        return _in_session_domain(call, "estimate", queries_at=-4)

    @classmethod
    def run(cls, samples, values, queries, partition, decoder, params, alpha, n_partitions, seed,
            method="estimate", k=0, folding_seed=0, callback=None):
        """Members of any partition and decoder of the catalogue, within the session settings.

        Parameters
        ----------
        samples, values, queries : array_like
            Data locations (n, d), data values (n,) and query locations (m, d).
        partition, decoder : str
            Names in the catalogue.
        params : dict
            The decoder's parameters, as the catalogue lists them (named choices accepted).
        alpha : float
            Spatialize's granularity (Mondrian) or nuclei rate (Voronoi, negative for uniform nuclei).
        n_partitions, seed : int
        method : {"estimate", "loo", "kfold"}
        k, folding_seed : int
            For ``method="kfold"``.
        callback : callable, optional

        Returns
        -------
        tuple
            ``(None, members)``, with one row per query (``estimate``) or per datum.
        """
        d = int(np.shape(samples)[1])
        _check("Partitioning process", _PARTITIONS, partition, d)
        _check("Local interpolator", _DECODERS, decoder, d)

        def call(samples, values, queries):
            return _run(partition, decoder, method, samples, values, queries, alpha, n_partitions, seed,
                        dict(params), k, folding_seed, callback)
        return _in_session_domain(call, method, queries_at=2)(samples, values, queries)

    @classmethod
    def get_kriging_model_number(cls, model):
        """The number of a variogram model (1 spherical, 2 exponential, 3 cubic, 4 gaussian)."""
        choices = next(p for p in _DECODERS["kriging"]["params"] if p["name"] == "model")["choices"]
        return choices.index(model) + 1


def _run(partition, decoder, method, samples, values, queries, alpha, n_partitions, seed, params, k,
         folding_seed, visitor):
    return lsp.run(np.asarray(samples, np.float32), np.asarray(values, np.float32),
                   np.asarray(queries, np.float32), partition, float(alpha), int(n_partitions), int(seed),
                   decoder, params, method, int(k), int(folding_seed), visitor)


def _through_run(partition, decoder, operation):
    """A callable taking the legacy argument list, ``[samples, values, n_partitions, alpha,
    <decoder arguments, seed>, (k, folding_seed,) queries, callback]``, computed by ``run``.

    The decoder arguments are, in order, ``exponent`` (idw); ``model, nugget, range, sill``
    (kriging); and ``metric, parallelize`` after the seed (adaptive IDW).
    """
    def call(*args):
        samples, values, n_partitions, alpha = args[:4]
        queries, visitor = args[-2], args[-1]
        rest = list(args[4:-2])
        k = folding_seed = 0
        if operation == "kfold":
            k, folding_seed = rest[-2:]
            rest = rest[:-2]
        if decoder == local_interpolator.IDW:
            exponent, seed = rest
            params = {"exponent": exponent}
        elif decoder == local_interpolator.KRIGING:
            model, nugget, range_, sill, seed = rest
            params = {"model": model, "nugget": nugget, "range": range_, "sill": sill}
        elif decoder == local_interpolator.ADAPTIVE_IDW:
            seed, metric, parallelize = rest
            params = {"metric": metric, "parallelize": parallelize}
        else:
            raise SpatializeError(f"Local interpolator '{decoder}' has no argument list in the public API")
        return _run(partition, decoder, operation, samples, values, queries, alpha, n_partitions, seed, params,
                    k, folding_seed, visitor)

    return call


def _in_session_domain(function, operation, queries_at):
    """Wrap an operation so that its partitions are drawn on the session domain.

    The engine draws the partitions on the box of the samples and the queries. Adding the two
    opposite corners of the domain to the queries makes that box equal to the domain; the rows of
    the corners are then removed from an estimation's output. Without a session domain the
    operation is called unchanged.
    """
    def call(*args):
        samples = args[0]
        corners = session._domain_corners(int(np.shape(samples)[1]))
        if corners is None:
            return function(*args)
        args = list(args)
        queries = np.asarray(args[queries_at], dtype=np.float32)
        session._check_inside(corners, "data", samples)
        session._check_inside(corners, "queries", queries)
        args[queries_at] = np.vstack([queries, corners])
        estimation, members = function(*args)
        if operation == "estimate":
            members = members[:-2]
            if estimation is not None:
                estimation = estimation[:-2]
        return estimation, members

    return call

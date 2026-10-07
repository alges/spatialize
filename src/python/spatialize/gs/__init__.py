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
import warnings

import libspatialize as lsp

import numpy as np

from spatialize import SpatializeError, logging, session
from spatialize.logging import log_message


class local_interpolator:
    IDW, KRIGING, ADAPTIVE_IDW = "idw", "kriging", "adaptiveidw"
    SHARP_IDW = "sharpidw"                              # sharpened adaptive IDW
    CELLMEAN = "cellmean"                               # the mean of the cell's data
    DRAW = "draw"                                       # a datum drawn uniformly
    WDRAW_IDW, WDRAW_ADAPTIVE_IDW = "wdraw_idw", "wdraw_adaptiveidw"
    WDRAW_SHARP_IDW, WDRAW_KRIGING = "wdraw_sharpidw", "wdraw_kriging"


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

#: how the extension was built (OpenMP availability, threads)
BUILD_INFO = lsp.build_info()

OPENMP_HELP = """Spatialize was built without OpenMP, so it runs on a single thread. To run in parallel, install
an OpenMP runtime and reinstall Spatialize from source:
  - macOS:   brew install libomp
             pip install --force-reinstall --no-binary spatialize spatialize
  - Linux:   install the GNU OpenMP runtime (e.g. apt install libgomp1), then reinstall as above
  - Windows: build with Microsoft Visual C++, whose OpenMP support the build enables
To silence this warning, run on one thread on purpose: spatialize.session.set(parallel=False)."""

_warned_no_openmp = False


def _num_threads():
    """Threads for one call from the session settings: 1 when not parallel, otherwise
    ``num_threads`` (0 = all). Without OpenMP, running in parallel warns once and runs serially.
    """
    global _warned_no_openmp
    if not session.get("parallel"):
        return 1
    if not BUILD_INFO["openmp"]:
        if not _warned_no_openmp:
            warnings.warn(OPENMP_HELP, UserWarning, stacklevel=3)
            _warned_no_openmp = True
        return 1
    n = session.get("num_threads")
    return 0 if n is None else n


#: decoders whose public argument list predates the catalogue (see _through_run)
_LEGACY_ARGUMENTS = {"idw", "kriging", "adaptiveidw"}


def decoder_params(decoder):
    """Names of a decoder's parameters, in the catalogue's order (Python callables left out)."""
    return [p["name"] for p in _DECODERS[decoder]["params"] if p["type"] != "callable"]


def more_decoders(idw, kriging, adaptive, grid=False):
    """Defaults of the decoders beyond idw, kriging and adaptiveidw, from a function's own defaults.

    Parameters
    ----------
    idw, kriging, adaptive : dict
        The function's defaults (or search grids) for ``idw``, ``kriging`` and ``adaptiveidw``.
    grid : bool
        Whether the defaults are search grids (lists), as in the hyperparameter searches.
    """
    one = (lambda x: [x]) if grid else (lambda x: x)
    sharp = dict(adaptive, kappa_r=one(1.5), kappa_g=one(0.2), rho_max=one(3.0))
    return {
        local_interpolator.CELLMEAN: {}, local_interpolator.DRAW: {},
        local_interpolator.WDRAW_IDW: dict(idw),
        local_interpolator.WDRAW_KRIGING: dict(kriging, negative_weights=one("clip")),
        local_interpolator.SHARP_IDW: sharp,
        local_interpolator.WDRAW_ADAPTIVE_IDW: dict(adaptive),
        local_interpolator.WDRAW_SHARP_IDW: dict(sharp),
    }


def with_more_decoders(specific, grid=False):
    """A function's ``signature_overload`` defaults, extended to every decoder of the public API."""
    out = dict(specific)
    out.update(more_decoders(specific[local_interpolator.IDW], specific[local_interpolator.KRIGING], specific[local_interpolator.ADAPTIVE_IDW], grid))
    return out


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
                        {"post_creation": post_creation, "estimation": estimation}, 0, 0, callback, _num_threads())
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

        params = dict(params)
        threads = _num_threads()

        def call(samples, values, queries):
            return _run(partition, decoder, method, samples, values, queries, alpha, n_partitions, seed,
                        params, k, folding_seed, callback, threads)
        return _in_session_domain(call, method, queries_at=2)(samples, values, queries)

    @classmethod
    def cells(cls, samples, queries, partition, alpha, n_partitions, seed):
        """The cell of each query in each partition, within the session settings.

        Returns an int array of shape (queries, n_partitions); two queries share a cell of
        partition t exactly when their labels in column t are equal, -1 marking a query outside the
        box. The partitions are those ``run`` draws with the same arguments.
        """
        samples = np.asarray(samples, np.float32)
        queries = np.asarray(queries, np.float32)
        d = int(samples.shape[1])
        _check("Partitioning process", _PARTITIONS, partition, d)
        corners = _domain_corners(d)
        if corners is not None:
            _check_inside(corners, "data", samples)
            _check_inside(corners, "queries", queries)
            queries = np.vstack([queries, corners])
        labels = lsp.cells(samples, queries, partition, float(alpha), int(n_partitions), int(seed), _num_threads())
        return labels[:-2] if corners is not None else labels

    @classmethod
    def get_kriging_model_number(cls, model):
        """The number of a variogram model (1 spherical, 2 exponential, 3 cubic, 4 gaussian)."""
        choices = next(p for p in _DECODERS["kriging"]["params"] if p["name"] == "model")["choices"]
        return choices.index(model) + 1


def _run(partition, decoder, method, samples, values, queries, alpha, n_partitions, seed, params, k,
         folding_seed, visitor, num_threads):
    return lsp.run(np.asarray(samples, np.float32), np.asarray(values, np.float32),
                   np.asarray(queries, np.float32), partition, float(alpha), int(n_partitions), int(seed),
                   decoder, params, method, int(k), int(folding_seed), visitor, int(num_threads))


def _through_run(partition, decoder, operation):
    """A callable taking the legacy argument list, ``[samples, values, n_partitions, alpha,
    <decoder arguments, seed>, (k, folding_seed,) queries, callback]``, computed by ``run``.

    The decoder arguments are, in order, ``exponent`` (idw); ``model, nugget, range, sill``
    (kriging); ``metric`` after the seed (adaptive IDW); and for every other decoder its parameters in
    the catalogue's order, then the seed.
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
            seed, metric = rest
            params = {"metric": metric}
        else:
            *given, seed = rest
            params = dict(zip(decoder_params(decoder), given))
        return _run(partition, decoder, operation, samples, values, queries, alpha, n_partitions, seed, params,
                    k, folding_seed, visitor, _num_threads())

    return call


def _domain_corners(d):
    """The two opposite corners of the session domain as a float32 (2, d) array, or None."""
    box = session.get("domain")
    if box is None:
        return None
    if len(box) != d:
        raise SpatializeError(f"the session domain has {len(box)} coordinates but the data have {d}")
    return np.asarray(box, dtype=np.float32).T.copy()


def _check_inside(corners, what, arr):
    """Raise if some rows of ``arr`` lie outside the box spanned by ``corners``."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size and (np.any(arr < corners[0]) or np.any(arr > corners[1])):
        raise SpatializeError(f"some {what} lie outside the session domain {session.get('domain')}")


def _in_session_domain(function, operation, queries_at):
    """Wrap an operation so that its partitions are drawn on the session domain.

    The engine draws the partitions on the box of the samples and the queries. Adding the two
    opposite corners of the domain to the queries makes that box equal to the domain; the rows of
    the corners are then removed from an estimation's output. Without a session domain the
    operation is called unchanged.
    """
    def call(*args):
        samples = args[0]
        corners = _domain_corners(int(np.shape(samples)[1]))
        if corners is None:
            return function(*args)
        args = list(args)
        queries = np.asarray(args[queries_at], dtype=np.float32)
        _check_inside(corners, "data", samples)
        _check_inside(corners, "queries", queries)
        args[queries_at] = np.vstack([queries, corners])
        estimation, members = function(*args)
        if operation == "estimate":
            members = members[:-2]
            if estimation is not None:
                estimation = estimation[:-2]
        return estimation, members

    return call

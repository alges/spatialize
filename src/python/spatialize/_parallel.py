"""Parallel loops on the Python side (joblib), under the same session settings as the compiled code.

The work is split into chunks of items, about four per worker. A few items run serially first; the
rest goes to worker processes only when the time those items took, extrapolated to all of them,
exceeds the cost of starting the workers, a cost that almost vanishes while the workers of a
previous loop are still alive. Every item must give the same result wherever it runs (its
own seed, no shared state), so the choice changes only the run time.
"""
import os
import time

from spatialize import logging, session

#: items timed serially before deciding whether to start worker processes
PILOT_ITEMS = 20
#: estimated serial time, in seconds, above which the remaining items go to worker processes
MIN_PARALLEL_SECONDS = 3.0
#: the same while the workers of a previous loop are still alive (loky keeps idle workers 300 s)
MIN_PARALLEL_SECONDS_WARM = 0.3
_WARM_FOR = 240.0
_last_parallel = None


def _num_workers():
    """Worker processes from the session: 1 when not parallel, otherwise ``num_threads``, else
    ``OMP_NUM_THREADS``, else every processor. They need no OpenMP."""
    if not session.get("parallel"):
        return 1
    n = session.get("num_threads")
    if n:
        return n
    env = os.environ.get("OMP_NUM_THREADS", "")
    return int(env) if env.isdigit() and int(env) > 0 else (os.cpu_count() or 1)


def _chunks(items, n_workers):
    """``items`` (a range or list) split into about four chunks per worker."""
    size = max(1, -(-len(items) // (4 * n_workers)))
    return [items[a:a + size] for a in range(0, len(items), size)]


def map_chunks(work, n_items, data_for=None, callback=None, repeated=False, desc=None):
    """Results of ``work`` for items ``0 .. n_items - 1``, in order.

    Parameters
    ----------
    work : callable
        ``work(rows, data) -> list``, one result per item of ``rows`` (a range). It runs in worker
        processes, so it must be picklable (cloudpickle: closures are fine), and it should take what
        it needs from ``data`` rather than from large captured variables.
    n_items : int
        Number of items.
    data_for : callable, optional
        ``data_for(rows)`` gives the data a chunk needs, sent to its worker with it; by default
        ``None``. It keeps what each worker receives small.
    callback : callable, optional
        Progress callback (``logging.progress`` protocol), informed once per item.
    desc : str, optional
        The name of the progress run.
    repeated : bool, optional
        Whether the caller runs several such loops in a row (a search over configurations), whose
        workers then start once for all of them; the low threshold applies from the first loop.
    """
    data_for = data_for or (lambda rows: None)
    if callback is not None:
        callback(logging.progress.init(n_items, 1, desc=desc))

    def serial(rows):
        out = []
        for chunk in _chunks(rows, 25):  # small chunks, for the progress
            out += work(chunk, data_for(chunk))
            if callback is not None:
                for _ in chunk:
                    callback(logging.progress.inform())
        return out

    n_workers = _num_workers()
    pilot = range(0, min(PILOT_ITEMS, n_items))
    t0 = time.perf_counter()
    results = serial(pilot)
    rest = range(len(pilot), n_items)
    elapsed = time.perf_counter() - t0
    estimate = elapsed / max(len(pilot), 1) * len(rest)

    warm = _last_parallel is not None and time.monotonic() - _last_parallel < _WARM_FOR
    threshold = MIN_PARALLEL_SECONDS_WARM if warm or repeated else MIN_PARALLEL_SECONDS
    if n_workers == 1 or len(rest) < 2 or estimate < threshold:
        results += serial(rest)
    else:
        results += _parallel(work, rest, data_for, n_workers, callback)

    if callback is not None:
        callback(logging.progress.stop())
    return results


def _parallel(work, rows, data_for, n_workers, callback):
    # the chunks come back in order as they finish, so the progress is counted here, in the calling
    # thread; a worker that dies raises an error instead of leaving anything waiting
    global _last_parallel
    from joblib import Parallel, delayed
    results = []
    tasks = (delayed(work)(chunk, data_for(chunk)) for chunk in _chunks(rows, n_workers))
    for part in Parallel(n_jobs=n_workers, backend="loky", return_as="generator")(tasks):
        results += part
        if callback is not None:
            for _ in part:
                callback(logging.progress.inform())
    _last_parallel = time.monotonic()
    return results

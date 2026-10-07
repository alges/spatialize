"""Session settings, values that apply to every call until changed.

A setting is resolved at each call as the value given in an active :func:`override`, otherwise the
value given to :func:`set`, otherwise its built-in default. Every built-in default reproduces the
behaviour of Spatialize without a session, so a session changes no result unless the user sets it.
Each :class:`~spatialize.gs.esi.ESIResult` records the settings it was computed with in
``effective_config``, which :func:`~spatialize.data.save_result` keeps.

The settings are the following.

``domain`` (array-like of shape (d, 2), or ``None``)
    The box ``[[low, high], ...]`` on which the partitions are drawn, one row per coordinate. With
    ``None``, the default, the box is the smallest one containing the data and the queries of each
    call. The partitions then depend on the other locations requested, and so does the estimate at
    a location. With a fixed domain, the estimate at a location no longer depends on the other
    queries (for kriging, up to float32 rounding, since each cell solves its queries together). The
    hyperparameter searches then draw the same partitions as the estimation for the same seed. Every
    datum and query must lie in the domain.

``parallel`` (bool)
    Whether the compiled code runs on several threads. The default is ``True``. Results are the same
    bit for bit with any number of threads, so the setting changes only the run time. When
    Spatialize was built without OpenMP, a warning says once how to install it, after which the code
    runs on one thread.

``num_threads`` (positive int, or ``None``)
    The number of threads when ``parallel`` is true. With ``None``, the default, the runtime uses
    every processor, or the number set by the environment variable ``OMP_NUM_THREADS``.

Examples
--------
>>> import spatialize
>>> spatialize.session.set(domain=[[0, 100], [0, 50]])
>>> with spatialize.session.override(domain=None):     # restored on exit
...     pass
>>> spatialize.session.effective_config()
{'domain': ((0.0, 100.0), (0.0, 50.0)), 'parallel': True, 'num_threads': None}
>>> spatialize.session.reset()
"""
import contextlib
import contextvars

import numpy as np

from spatialize import SpatializeError

_DEFAULTS = {"domain": None, "parallel": True, "num_threads": None}

_global = {}
_overrides = contextvars.ContextVar("spatialize_session_overrides", default={})


def _validate_domain(value):
    if value is None:
        return None
    try:
        box = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise SpatializeError(f"domain must be an array of [low, high] pairs; got {value!r}") from None
    if box.ndim != 2 or box.shape[1] != 2:
        raise SpatializeError(f"domain must have shape (d, 2), one [low, high] per coordinate; got {box.shape}")
    if not np.all(np.isfinite(box)) or np.any(box[:, 0] >= box[:, 1]):
        raise SpatializeError("domain needs finite bounds with low < high on every coordinate")
    return tuple((float(lo), float(hi)) for lo, hi in box)


def _validate_parallel(value):
    if not isinstance(value, (bool, np.bool_)):
        raise SpatializeError(f"parallel must be True or False; got {value!r}")
    return bool(value)


def _validate_num_threads(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise SpatializeError(f"num_threads must be a positive integer or None; got {value!r}")
    return int(value)


_VALIDATORS = {"domain": _validate_domain, "parallel": _validate_parallel, "num_threads": _validate_num_threads}


def _validated(settings):
    unknown = [k for k in settings if k not in _DEFAULTS]
    if unknown:
        raise SpatializeError(f"unknown session setting(s) {sorted(unknown)}; known: {sorted(_DEFAULTS)}")
    return {k: _VALIDATORS[k](v) for k, v in settings.items()}


def set(**settings):
    """Set session values, kept until changed or reset."""
    _global.update(_validated(settings))


def reset(*names):
    """Return the named settings, or all of them, to their built-in defaults."""
    for name in (names or tuple(_global)):
        if name not in _DEFAULTS:
            raise SpatializeError(f"unknown session setting '{name}'")
        _global.pop(name, None)


def get(name):
    """The value of a setting in effect now."""
    if name not in _DEFAULTS:
        raise SpatializeError(f"unknown session setting '{name}'")
    overrides = _overrides.get()
    if name in overrides:
        return overrides[name]
    return _global.get(name, _DEFAULTS[name])


@contextlib.contextmanager
def override(**settings):
    """Set session values inside a ``with`` block only (safe across threads and notebook cells)."""
    token = _overrides.set({**_overrides.get(), **_validated(settings)})
    try:
        yield
    finally:
        _overrides.reset(token)


def effective_config():
    """Every setting with the value in effect now."""
    return {name: get(name) for name in _DEFAULTS}


def show():
    """Print the settings in effect, marking those different from the default."""
    for name, value in effective_config().items():
        mark = "" if value == _DEFAULTS[name] else "   (set)"
        print(f"{name} = {value!r}{mark}")


# --------------------------------------------------------------------------------------------------
# internal helpers for the facade


def _domain_corners(d):
    """The two opposite corners of the session domain as a float32 (2, d) array, or None."""
    box = get("domain")
    if box is None:
        return None
    if len(box) != d:
        raise SpatializeError(f"the session domain has {len(box)} coordinates but the data have {d}")
    return np.asarray(box, dtype=np.float32).T.copy()


def _check_inside(corners, what, arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size and (np.any(arr < corners[0]) or np.any(arr > corners[1])):
        raise SpatializeError(f"some {what} lie outside the session domain {get('domain')}")

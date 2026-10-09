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
    Whether the compiled code runs on several threads, and the Python loops on several processes
    (``joblib``): the encoder error of the Pareto search
    (:func:`~spatialize.gs.esi.esi_pareto_hparams_search`) and the simulations
    (:func:`~spatialize.gs.ess.ess_sample`), these only when the work repays starting the
    processes. The default is ``True``. Results are the same bit for bit with
    any number of threads or processes, so the setting changes only the run time. When Spatialize was built without OpenMP, a warning says once
    how to install it, after which the compiled code runs on one thread, the processes being
    unaffected.

``num_threads`` (positive int, or ``None``)
    The number of threads, or processes, when ``parallel`` is true. With ``None``, the default, the
    runtime uses every processor, or the number set by the environment variable ``OMP_NUM_THREADS``.

``empty_cells`` (``"nan"``, ``"mark"`` or ``"coarsen"``)
    What a member is when the cell of its location holds no datum, in estimation, leave-one-out and
    k-fold. Unlike the settings above, it changes **which law is estimated**, not only how it is
    computed, so it also changes the results of the hyperparameter searches.

    - ``"nan"``, the default, leaves the member NaN. Every reading drops it, so the law at a location
      is the one conditioned on its cell having data. Far from the data, where most cells are empty,
      it rests on few members. The cross-validation scores are then computed on the data whose cells
      keep other data, which favours partitions that are too fine.
    - ``"mark"`` gives the empty cell one value, shared by every location in that cell of that
      partition (see ``mark_source``). It follows the theory's law, which sends the weight of the
      partitions that leave a location without data to the law of the marks. Every member is
      defined, and two locations in one empty cell move together, as they should.
    - ``"coarsen"`` predicts the locations of an empty cell with the decoder, from a coarser cell
      that holds data. For Mondrian partitions it is the nearest ancestor of the cell in the tree
      whose region holds data, the cell an earlier cut would have left whole, so every location of
      the empty cell uses the same data. For Voronoi partitions each location goes to the nearest
      nucleus whose cell holds data, which is the Voronoi partition of the nuclei with data. The
      decoder's parameters are fitted on the coarser cell when it has none of its own (a Mondrian
      ancestor). Every member is defined, varying with the location inside the empty cell as the decoder's predictions do elsewhere.

    :meth:`~spatialize.gs.esi.ESIResult.empty_cell_fraction` tells, at each location, the share of
    partitions concerned.

``mark_source`` (``"local"``, ``"cells"`` or ``"data"``)
    Where the mark of an empty cell comes from, under ``empty_cells="mark"``. A cell holding data is drawn, which gives the mark as ``mark_value`` says.

    - ``"local"``, the default, draws among the ``mark_knn`` cells with data nearest to the empty
      cell, so the mark follows the level of the field around it.
    - ``"cells"`` draws among all the cells with data of the partition, each weighing the same, the
      block-mark model's single law of the marks for the whole field.
    - ``"data"`` draws one datum uniformly among all the data, so densely sampled zones weigh more.

    The empty cells of one partition draw their sources without repetition while candidates remain.
    In leave-one-out and k-fold the data held out take no part.

``mark_knn`` (positive int)
    The number of nearby cells with data among which ``mark_source="local"`` draws. The default is 8.

``mark_value`` (``"decoder"`` or ``"datum"``)
    What the cell drawn by ``mark_source`` gives as the mark of an empty cell.

    - ``"decoder"``, the default, is that cell's decoder prediction at the empty cell's centre (its
      nucleus, for Voronoi), the kind of value the decoder gives elsewhere: an observed value for the
      drawing decoders, a local average for the others.
    - ``"datum"`` is one of that cell's data, drawn uniformly, whatever the decoder. It is the
      block-mark model, in which a block carries one mark drawn from the law of the marks.
      ``mark_source="cells"`` with ``mark_value="datum"`` follows the model exactly.

    With ``mark_source="data"`` the mark is always a datum.

``max_left_out`` (number between 0 and 1)
    The share of the data that a hyperparameter search may leave out of its cross-validation score
    before it warns. Under ``empty_cells="nan"`` a datum alone in its cell (leave-one-out), or a cell
    whose data all fall in one fold (k-fold), gives NaN members, and the scores drop a datum with
    too few valid members: one for the mean absolute and squared errors, 30 for the negative
    log-likelihood, two for the CRPS. The data dropped are the isolated ones, the hardest to
    predict, so a fine partition is scored on the easy data only and looks better than it is.
    The searches record, for every configuration, the share of the data left out (``left_out``) and
    the share of NaN members (``nan_members``). They warn when some configurations leave out more
    than this share, listing all of them and marking the best. The default is 0.05. Under
    ``empty_cells="mark"`` or ``"coarsen"`` no datum is left out.

``display`` (``"auto"``, ``"terminal"``, ``"notebook"``, ``"plain"`` or ``"silent"``)
    How the progress bars and the messages look. ``"auto"``, the default, detects where Spatialize
    writes: a Jupyter kernel (JupyterLab, the classic notebook, the notebooks of VS Code and PyCharm,
    Colab) gets HTML bars and messages; a terminal, including the terminals of the IDEs and
    PyCharm's run console, gets live bars drawn with ``rich``; any other output (a file, a pipe, the
    log of a continuous integration) gets plain lines, a progress line at every quarter. The other
    values force one of these looks, ``"silent"`` showing nothing. All of them share the colours of
    the ``alges`` palette. The setting changes no result.

``verbosity`` (``"debug"``, ``"info"``, ``"warning"`` or ``"error"``)
    The lowest level of the messages shown. The default, ``"warning"``, shows the warnings and the
    errors, such as the data a search leaves out of its score or the calibration of the laws of the
    posterior analysis. ``"info"`` adds what the functions do, ``"debug"`` the details. Setting a
    level on ``spatialize.logging.log`` (``log.setLevel("DEBUG")``) takes precedence. The setting
    changes no result.

Examples
--------
>>> import spatialize
>>> spatialize.session.set(domain=[[0, 100], [0, 50]])
>>> with spatialize.session.override(domain=None):     # restored on exit
...     pass
>>> spatialize.session.effective_config()
{'domain': ((0.0, 100.0), (0.0, 50.0)), 'parallel': True, 'num_threads': None, 'empty_cells': 'nan', 'mark_source': 'local', 'mark_knn': 8, 'mark_value': 'decoder', 'max_left_out': 0.05, 'display': 'auto', 'verbosity': 'warning'}
>>> spatialize.session.reset()
"""
import contextlib
import contextvars

import numpy as np

from spatialize import SpatializeError

_DEFAULTS = {"domain": None, "parallel": True, "num_threads": None, "empty_cells": "nan",
             "mark_source": "local", "mark_knn": 8, "mark_value": "decoder", "max_left_out": 0.05,
             "display": "auto", "verbosity": "warning"}

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


def _validate_num_threads_like(name):
    def validate(value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise SpatializeError(f"{name} must be a positive integer; got {value!r}")
        return int(value)
    return validate


def _validate_share(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) \
            or not 0 <= value <= 1:
        raise SpatializeError(f"max_left_out must be a number between 0 and 1; got {value!r}")
    return float(value)


def _choice(name, options):
    def validate(value):
        if value not in options:
            raise SpatializeError(f"{name} must be one of {list(options)}; got {value!r}")
        return value
    return validate


_VALIDATORS = {"domain": _validate_domain, "parallel": _validate_parallel, "num_threads": _validate_num_threads,
               "empty_cells": _choice("empty_cells", ("nan", "mark", "coarsen")),
               "mark_source": _choice("mark_source", ("local", "cells", "data")),
               "mark_knn": _validate_num_threads_like("mark_knn"),
               "mark_value": _choice("mark_value", ("decoder", "datum")),
               "max_left_out": _validate_share,
               "display": _choice("display", ("auto", "terminal", "notebook", "plain", "silent")),
               "verbosity": _choice("verbosity", ("debug", "info", "warning", "error"))}


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
    """Show the settings in effect, marking those different from the default, in the look of
    ``display``."""
    from spatialize import _display
    _display.settings_table([(name, value, value != _DEFAULTS[name]) for name, value in effective_config().items()])


"""
Scoring functions for cat_esi_hparams_search cross-validation.

Each function takes ``(true_values, estimated_values)`` — both 1-D arrays of
categorical labels — and returns a scalar **error** (lower is better), so that
``GridSearchResult.best_params`` finds the minimum as the best configuration.

Naming mirrors sklearn's classification metrics, but the sign is flipped:
    cv_error = 1 - score   (for accuracy-based metrics)
    cv_error = score       (for native error metrics like brier)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    log_loss,
)


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _to_str_array(arr):
    return np.array([str(v) for v in arr])


# ─────────────────────────────────────────────────────────────
#  Scoring functions
# ─────────────────────────────────────────────────────────────

def accuracy(true_values, estimated_values):
    """Compute one minus the classification accuracy.

    This is the default scorer for :func:`~spatialize.gs.cat_esi.cat_esi_hparams_search`.

    Parameters
    ----------
    true_values : array_like
        Ground-truth categorical labels, 1-D.
    estimated_values : array_like
        Predicted categorical labels, 1-D, same length as `true_values`.

    Returns
    -------
    float
        ``1 - accuracy``, in ``[0, 1]``; lower is better.

    Notes
    -----
    Backed by ``sklearn.metrics.accuracy_score``: the fraction of matching
    labels between `true_values` and `estimated_values`, with no per-class
    averaging.
    """
    return 1.0 - accuracy_score(_to_str_array(true_values),
                                _to_str_array(estimated_values))


def f1_macro(true_values, estimated_values):
    """Compute one minus the macro-averaged F1 score.

    Parameters
    ----------
    true_values : array_like
        Ground-truth categorical labels, 1-D.
    estimated_values : array_like
        Predicted categorical labels, 1-D, same length as `true_values`.

    Returns
    -------
    float
        ``1 - f1_macro``, in ``[0, 1]``; lower is better.

    Notes
    -----
    Backed by ``sklearn.metrics.f1_score`` with ``average='macro'``: the F1
    score is computed independently for each class and then unweighted-averaged,
    so all classes count equally regardless of their support (size).
    """
    return 1.0 - f1_score(_to_str_array(true_values),
                          _to_str_array(estimated_values),
                          average='macro', zero_division=0)


def f1_weighted(true_values, estimated_values):
    """Compute one minus the support-weighted F1 score.

    Parameters
    ----------
    true_values : array_like
        Ground-truth categorical labels, 1-D.
    estimated_values : array_like
        Predicted categorical labels, 1-D, same length as `true_values`.

    Returns
    -------
    float
        ``1 - f1_weighted``, in ``[0, 1]``; lower is better.

    Notes
    -----
    Backed by ``sklearn.metrics.f1_score`` with ``average='weighted'``: the F1
    score is computed independently for each class and then averaged, weighted
    by each class's support (number of true instances), so frequent classes
    dominate the score.
    """
    return 1.0 - f1_score(_to_str_array(true_values),
                          _to_str_array(estimated_values),
                          average='weighted', zero_division=0)


def f1_micro(true_values, estimated_values):
    """Compute one minus the micro-averaged F1 score.

    Parameters
    ----------
    true_values : array_like
        Ground-truth categorical labels, 1-D.
    estimated_values : array_like
        Predicted categorical labels, 1-D, same length as `true_values`.

    Returns
    -------
    float
        ``1 - f1_micro``, in ``[0, 1]``; lower is better.

    Notes
    -----
    Backed by ``sklearn.metrics.f1_score`` with ``average='micro'``: true
    positives, false positives, and false negatives are pooled globally
    across classes before computing F1, which makes it numerically
    equivalent to accuracy for multi-class (single-label) problems.
    """
    return 1.0 - f1_score(_to_str_array(true_values),
                          _to_str_array(estimated_values),
                          average='micro', zero_division=0)


def precision_macro(true_values, estimated_values):
    """Compute one minus the macro-averaged precision.

    Parameters
    ----------
    true_values : array_like
        Ground-truth categorical labels, 1-D.
    estimated_values : array_like
        Predicted categorical labels, 1-D, same length as `true_values`.

    Returns
    -------
    float
        ``1 - precision_macro``, in ``[0, 1]``; lower is better.

    Notes
    -----
    Backed by ``sklearn.metrics.precision_score`` with ``average='macro'``:
    precision is computed independently for each class and then
    unweighted-averaged, so all classes count equally regardless of size.
    """
    return 1.0 - precision_score(_to_str_array(true_values),
                                 _to_str_array(estimated_values),
                                 average='macro', zero_division=0)


def recall_macro(true_values, estimated_values):
    """Compute one minus the macro-averaged recall.

    Parameters
    ----------
    true_values : array_like
        Ground-truth categorical labels, 1-D.
    estimated_values : array_like
        Predicted categorical labels, 1-D, same length as `true_values`.

    Returns
    -------
    float
        ``1 - recall_macro``, in ``[0, 1]``; lower is better.

    Notes
    -----
    Backed by ``sklearn.metrics.recall_score`` with ``average='macro'``:
    recall is computed independently for each class and then
    unweighted-averaged, which is equivalent to ``1 - balanced_accuracy``
    when classes have uniform priors.
    """
    return 1.0 - recall_score(_to_str_array(true_values),
                              _to_str_array(estimated_values),
                              average='macro', zero_division=0)


def cohen_kappa(true_values, estimated_values):
    """Compute one minus Cohen's kappa agreement coefficient.

    Parameters
    ----------
    true_values : array_like
        Ground-truth categorical labels, 1-D.
    estimated_values : array_like
        Predicted categorical labels, 1-D, same length as `true_values`.

    Returns
    -------
    float
        ``1 - kappa``, in ``[0, 2]``; lower is better. If only one class is
        present in the combined data, kappa is undefined and this returns
        ``2.0`` (treated as the worst case).

    Notes
    -----
    Backed by ``sklearn.metrics.cohen_kappa_score``, which corrects raw
    agreement for the agreement expected by chance — useful when class
    frequencies are imbalanced or for ordinal data where near-miss
    disagreements matter.
    """
    t = _to_str_array(true_values)
    e = _to_str_array(estimated_values)
    try:
        kappa = cohen_kappa_score(t, e)
    except ValueError:
        # Only one class present → κ undefined; treat as worst case
        kappa = -1.0
    # κ ∈ [-1, 1]; shift so lower is worse and clip to [0, 2]
    return 1.0 - kappa


# ─────────────────────────────────────────────────────────────
#  String-to-function resolver
# ─────────────────────────────────────────────────────────────

_REGISTRY = {
    'accuracy':        accuracy,
    'f1_macro':        f1_macro,
    'f1_weighted':     f1_weighted,
    'f1_micro':        f1_micro,
    'precision_macro': precision_macro,
    'recall_macro':    recall_macro,
    'cohen_kappa':     cohen_kappa,
}


def resolve_scoring(scoring):
    """
    Return a ``(true, estimated) → error`` callable.

    Accepts either a string name (see ``SCORING_OPTIONS``) or any callable.
    """
    if callable(scoring):
        return scoring
    if scoring in _REGISTRY:
        return _REGISTRY[scoring]
    raise ValueError(
        f"Unknown scoring '{scoring}'. "
        f"Available: {list(_REGISTRY.keys())}"
    )


SCORING_OPTIONS = list(_REGISTRY.keys())

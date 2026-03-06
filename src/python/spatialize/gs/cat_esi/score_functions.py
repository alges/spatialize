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
    """1 − accuracy.  Default scorer."""
    return 1.0 - accuracy_score(_to_str_array(true_values),
                                _to_str_array(estimated_values))


def f1_macro(true_values, estimated_values):
    """1 − macro-averaged F1.  Treats all classes equally regardless of size."""
    return 1.0 - f1_score(_to_str_array(true_values),
                          _to_str_array(estimated_values),
                          average='macro', zero_division=0)


def f1_weighted(true_values, estimated_values):
    """1 − weighted F1.  Weights each class by its support (class frequency)."""
    return 1.0 - f1_score(_to_str_array(true_values),
                          _to_str_array(estimated_values),
                          average='weighted', zero_division=0)


def f1_micro(true_values, estimated_values):
    """1 − micro-averaged F1.  Equivalent to accuracy for multi-class problems."""
    return 1.0 - f1_score(_to_str_array(true_values),
                          _to_str_array(estimated_values),
                          average='micro', zero_division=0)


def precision_macro(true_values, estimated_values):
    """1 − macro precision."""
    return 1.0 - precision_score(_to_str_array(true_values),
                                 _to_str_array(estimated_values),
                                 average='macro', zero_division=0)


def recall_macro(true_values, estimated_values):
    """1 − macro recall (= 1 − balanced accuracy for uniform priors)."""
    return 1.0 - recall_score(_to_str_array(true_values),
                              _to_str_array(estimated_values),
                              average='macro', zero_division=0)


def cohen_kappa(true_values, estimated_values):
    """1 − Cohen's κ.  Accounts for chance agreement; good for ordinal data."""
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

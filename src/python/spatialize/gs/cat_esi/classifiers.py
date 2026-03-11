"""
Classifier implementations for categorical ESI.

Provides:
  - knn_pca   : anisotropic PCA + KNN (built-in preset)
  - scikit-learn : adapter for any sklearn-compatible classifier

The public entry point is ``get_classifier_fns(classifier, **kwargs)``,
which returns the ``(set_cell_params_fn, classifier_fn)`` pair expected
by ``libspatialize.estimation_custom_esi``.
"""

import numpy as np
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
#  knn_pca classifier
# ═══════════════════════════════════════════════════════════════════

def _knn_pca_weighted_diffs(cell_points, cell_values, max_neighbors=10):
    """
    Weighted difference vectors for PCA anisotropy estimation.

    For each point, computes difference vectors to its nearest neighbours,
    weighted by categorical dissimilarity (1.0 if categories differ, 0.0 if same).
    """
    n = len(cell_points)
    k = min(max_neighbors, n - 1)
    if k <= 0:
        return np.empty((0, cell_points.shape[1]))

    # Brute-force pairwise squared distances — faster than KDTree for small cells
    diffs_all = cell_points[:, np.newaxis, :] - cell_points[np.newaxis, :, :]  # (n, n, D)
    sq_dists  = np.einsum('ijk,ijk->ij', diffs_all, diffs_all)                 # (n, n)
    np.fill_diagonal(sq_dists, np.inf)                                         # exclude self
    neighbor_idx = np.argpartition(sq_dists, k, axis=1)[:, :k]                # (n, k)

    # Categorical dissimilarity weight
    dissimilarity = (cell_values[neighbor_idx] != cell_values[:, np.newaxis]).astype(np.float64)

    # Weighted difference vectors, keep only dissimilar pairs
    neighbor_diffs = cell_points[neighbor_idx] - cell_points[:, np.newaxis, :]  # (n, k, D)
    weighted = (neighbor_diffs * dissimilarity[:, :, np.newaxis]).reshape(-1, cell_points.shape[1])
    return weighted[dissimilarity.ravel() > 0]


def _knn_pca_direction(cell_points, cell_values):
    """
    SVD on categorical boundary gradients to find the principal anisotropy direction.

    Returns (principal_direction, explained_variance_ratio, rotation_matrix).
    Returns (zeros, zeros, None) if SVD cannot be computed.
    """
    if cell_points.shape[0] < 2:
        raise ValueError("At least 2 points required.")

    cell_points_f = np.ascontiguousarray(cell_points, dtype=np.float64)
    cell_values_i = np.ascontiguousarray(cell_values, dtype=np.int64)

    max_neighbors = min(15, cell_points_f.shape[0] - 1)
    weighted_diffs = _knn_pca_weighted_diffs(cell_points_f, cell_values_i, max_neighbors)

    n_dim = cell_points.shape[1]
    # Need at least n_dim samples for a full-rank rotation matrix
    if weighted_diffs.shape[0] < n_dim:
        return np.zeros(n_dim), np.zeros(n_dim), None

    centered = weighted_diffs - weighted_diffs.mean(axis=0)  # zero-center before decomposition
    try:
        # full_matrices=False: Vt is (n_dim, n_dim) when n_samples >= n_dim
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros(n_dim), np.zeros(n_dim), None

    total_s2 = np.dot(S, S)
    evr = S ** 2 / total_s2 if total_s2 > 0 else np.zeros(len(S))  # proportion of variance per axis
    return Vt[0], evr, Vt  # principal direction, explained variance ratio, full rotation matrix


def _knn_pca_predict(cell_points, cell_values, cell_queries, cell_params):
    """
    Anisotropic KNN prediction using scipy KDTree.

    Transforms all points into the isotropic space defined by the rotation matrix
    and anisotropy scales, then performs standard Euclidean KNN search.

    cell_params layout: [rotation_matrix_flat (D²), anisotropy_scales (D), k_neighbors (1)]
    """
    if len(np.unique(cell_values)) < 2:
        return np.full(len(cell_queries), cell_values[0])
    n_dim = cell_points.shape[1]
    k_neighbors = int(round(cell_params[-1]))
    rotation_matrix   = cell_params[: n_dim * n_dim].reshape(n_dim, n_dim)
    anisotropy_scales = cell_params[n_dim * n_dim : n_dim * n_dim + n_dim]

    if len(cell_points) == 0 or k_neighbors == 0:
        return np.full(len(cell_queries), np.nan)

    k_eff = min(k_neighbors, len(cell_points))

    # Anisotropic distance: ||diag(s) @ R @ (a-b)||  =  Euclidean in transformed space
    transform = anisotropy_scales[:, np.newaxis] * rotation_matrix  # (D, D)
    train_t = cell_points @ transform.T   # (N, D)
    query_t = cell_queries @ transform.T  # (M, D)

    # Brute-force pairwise squared distances — faster than KDTree for small cells
    diffs    = query_t[:, np.newaxis, :] - train_t[np.newaxis, :, :]  # (M, N, D)
    sq_dists = np.einsum('ijk,ijk->ij', diffs, diffs)                 # (M, N)

    if k_eff == 1:
        # No majority vote needed for a single neighbour
        return cell_values[np.argmin(sq_dists, axis=1)]

    indices = np.argpartition(sq_dists, k_eff - 1, axis=1)[:, :k_eff]  # (M, k_eff)

    # Vectorised majority vote: category codes are 0-indexed integers stored as float64
    int_vals = np.round(cell_values[indices]).astype(np.int64)   # (M, k_eff)
    n_cats   = int(np.round(cell_values).astype(np.int64).max()) + 1
    # counts[i, c] = number of neighbours of query i with category code c
    counts      = (int_vals[:, :, np.newaxis] == np.arange(n_cats, dtype=np.int64)).sum(axis=1)
    return counts.argmax(axis=1).astype(cell_values.dtype)


def _set_cell_params_knn_pca(
    cell_points: np.ndarray,
    cell_values: np.ndarray,
    n_neighbors: Optional[int] = None,
    max_points: Optional[int] = None,
    n_cv_splits: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Estimate optimal anisotropic KNN parameters for a single ESI partition.

    Uses PCA to find the principal direction of categorical variation, then
    performs a grid search over anisotropy ratios and k values using random
    train/test CV splits.

    Parameters
    ----------
    cell_points : (N, D)
    cell_values : (N,) integer-coded categories
    n_neighbors : fixed k to use (skips k search if provided)
    max_points  : subsampling cap for parameter estimation
    n_cv_splits : number of random CV splits for parameter search

    Returns
    -------
    np.ndarray, flat: [rotation_matrix (D²), anisotropy_scales (D), k_neighbors (1)]
    """
    n_dim    = cell_points.shape[1]
    n_points = cell_points.shape[0]
    param_size = n_dim * n_dim + n_dim + 1

    def _default_params():
        p = np.empty(param_size, dtype=np.float64)
        p[: n_dim * n_dim] = np.eye(n_dim).flatten()
        p[n_dim * n_dim : n_dim * n_dim + n_dim] = np.ones(n_dim)
        p[-1] = 1.0
        return p

    if n_points < 2:
        return _default_params()

    # Derive a per-partition seed so that cells with the same n_eff don't share
    # identical splits. Mix the global seed with a hash of the cell's centroid.
    if seed is not None:
        centroid_bits = int(np.round(cell_points.mean(axis=0) * 1e6).astype(np.int64).sum())
        cell_seed = (seed * 2654435761 + centroid_bits) & 0xFFFFFFFF  # Knuth multiplicative hash
    else:
        cell_seed = None
    rng = np.random.default_rng(cell_seed)

    max_points_actual   = max_points   if max_points   is not None else min(n_points, 200)
    num_cv_splits_actual = n_cv_splits if n_cv_splits is not None else max(3, min(10, n_points // 10))

    if n_points > max_points_actual:
        idx = rng.choice(n_points, max_points_actual, replace=False)
        cell_points_s = np.ascontiguousarray(cell_points[idx], dtype=np.float64)
        cell_values_s = np.ascontiguousarray(cell_values[idx], dtype=np.int64)
        n_eff = max_points_actual
    else:
        cell_points_s = np.ascontiguousarray(cell_points, dtype=np.float64)
        cell_values_s = np.ascontiguousarray(cell_values, dtype=np.int64)
        n_eff = n_points

    _, explained_variance_ratio, pca_rotation = _knn_pca_direction(cell_points_s, cell_values_s)
    if pca_rotation is None:
        return _default_params()

    # k candidates
    if n_neighbors is not None:
        k_candidates = list(n_neighbors) if hasattr(n_neighbors, '__iter__') else [n_neighbors]
    else:
        n_test_k = max(3, min(5, n_eff // 10))
        all_k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        k_candidates = [k for k in all_k if k <= n_eff - 1][:n_test_k] or [1]

    # Anisotropy ratio candidates
    first_ev = explained_variance_ratio[0] if len(explained_variance_ratio) > 0 else 0.0
    if n_dim == 1:
        n_ratios = 2
    elif first_ev > 0.8:
        n_ratios = 7
    elif first_ev > 0.6:
        n_ratios = 5
    else:
        n_ratios = 3
    anisotropy_ratios = np.maximum(np.linspace(1.0, 4.0, max(2, n_ratios)), 1.0)

    max_train = int(n_eff * 0.7)
    max_test  = n_eff - max_train
    if max_train <= 0 or max_test <= 0:
        return _default_params()

    best_accuracy           = -1.0
    best_anisotropy_scales  = np.ones(n_dim, dtype=np.float64)
    best_k                  = int(k_candidates[0])

    max_k = int(k_candidates[-1])

    # Precompute PCA-rotated coordinates once — anisotropy is just axis scaling in this space
    pts_pca = cell_points_s @ pca_rotation.T  # (n_eff, D)

    # Generate splits once so every ratio is evaluated on the same folds (correctness + speed)
    splits = []
    for _ in range(num_cv_splits_actual):
        perm = rng.permutation(n_eff)
        splits.append((perm[:max_train], perm[max_train:]))

    for ratio in anisotropy_ratios:
        scales = np.ones(n_dim, dtype=np.float64)
        if n_dim >= 2:
            scales[0] = 1.0 / np.sqrt(ratio)
            scales[1] = np.sqrt(ratio)
        else:
            scales[0] = 1.0 / np.sqrt(ratio)
        scales = np.maximum(scales, 1e-9)

        # Scale PCA coordinates by the anisotropy vector (O(n_eff * D), no matrix multiply needed)
        pts_scaled = pts_pca * scales  # (n_eff, D)

        k_fold_accs = {k: [] for k in k_candidates}

        for tr_idx, te_idx in splits:
            tr_s    = pts_scaled[tr_idx]          # (n_train, D)
            tr_vals = cell_values_s[tr_idx]
            te_s    = pts_scaled[te_idx]          # (n_test, D)
            te_vals = cell_values_s[te_idx]

            if len(tr_s) == 0 or len(te_s) == 0:
                continue

            # Brute-force pairwise squared distances: faster than KDTree for small n
            # diffs: (n_test, n_train, D) → sq_dists: (n_test, n_train)
            diffs    = te_s[:, np.newaxis, :] - tr_s[np.newaxis, :, :]
            sq_dists = np.einsum('ijk,ijk->ij', diffs, diffs)

            k_eff = min(max_k, len(tr_s))
            # argpartition on squared distances (no sqrt needed for ranking)
            part_idx    = np.argpartition(sq_dists, min(k_eff - 1, sq_dists.shape[1] - 1), axis=1)
            all_indices = part_idx[:, :k_eff]  # (n_test, k_eff)

            for k in k_candidates:
                k_use = min(int(k), k_eff)
                neighbor_vals = tr_vals[all_indices[:, :k_use]]
                preds = np.array([
                    np.argmax(np.bincount(vals))
                    for vals in neighbor_vals
                ])
                k_fold_accs[k].append(np.mean(preds == te_vals))

        for k in k_candidates:
            if k_fold_accs[k]:
                acc = np.mean(k_fold_accs[k])
                if acc > best_accuracy:
                    best_accuracy          = acc
                    best_anisotropy_scales = scales.copy()
                    best_k                 = int(k)

    p = np.empty(param_size, dtype=np.float64)
    p[: n_dim * n_dim]                              = pca_rotation.flatten()
    p[n_dim * n_dim : n_dim * n_dim + n_dim]        = best_anisotropy_scales
    p[-1]                                            = float(best_k)
    return p


# ═══════════════════════════════════════════════════════════════════
#  Named scikit-learn classifier presets
# ═══════════════════════════════════════════════════════════════════

def _make_knn_fns(n_neighbors=None):
    """Wrap sklearn KNeighborsClassifier in the C++ callback protocol."""
    from sklearn.neighbors import KNeighborsClassifier
    _k = n_neighbors if n_neighbors is not None else 5

    def set_cell_params_fn(cell_points, cell_values):
        return np.array([], dtype=np.float64)

    def classifier_fn(cell_points, cell_values, cell_queries, cell_params):
        if len(cell_queries) == 0 or len(cell_points) == 0:
            return np.full(len(cell_queries), np.nan)
        if len(np.unique(cell_values)) < 2:
            return np.full(len(cell_queries), cell_values[0])
        try:
            clf = KNeighborsClassifier(n_neighbors=min(_k, len(cell_points)))
            clf.fit(cell_points, cell_values)
            return clf.predict(cell_queries).astype(np.float64)
        except Exception:
            return np.full(len(cell_queries), np.nan)

    return set_cell_params_fn, classifier_fn


def _make_svm_fns(C=None, kernel=None, gamma=None, random_state=None):
    """Wrap sklearn SVC in the C++ callback protocol."""
    from sklearn.svm import SVC
    _C      = C      if C      is not None else 1.0
    _kernel = kernel if kernel is not None else "rbf"
    _gamma  = gamma  if gamma  is not None else "scale"
    _rs     = random_state if random_state is not None else np.random.randint(0, 100000)

    def set_cell_params_fn(cell_points, cell_values):
        return np.array([], dtype=np.float64)

    def classifier_fn(cell_points, cell_values, cell_queries, cell_params):
        if len(cell_queries) == 0 or len(cell_points) == 0:
            return np.full(len(cell_queries), np.nan)
        if len(np.unique(cell_values)) < 2:
            return np.full(len(cell_queries), cell_values[0])
        try:
            clf = SVC(C=_C, kernel=_kernel, gamma=_gamma, random_state=_rs)
            clf.fit(cell_points, cell_values)
            return clf.predict(cell_queries).astype(np.float64)
        except Exception:
            return np.full(len(cell_queries), np.nan)

    return set_cell_params_fn, classifier_fn


def _make_rf_fns(n_estimators=None, max_depth=None, random_state=None):
    """Wrap sklearn RandomForestClassifier in the C++ callback protocol."""
    from sklearn.ensemble import RandomForestClassifier
    _n  = n_estimators if n_estimators is not None else 100
    _rs = random_state if random_state is not None else np.random.randint(0, 100000)

    def set_cell_params_fn(cell_points, cell_values):
        return np.array([], dtype=np.float64)

    def classifier_fn(cell_points, cell_values, cell_queries, cell_params):
        if len(cell_queries) == 0 or len(cell_points) == 0:
            return np.full(len(cell_queries), np.nan)
        if len(np.unique(cell_values)) < 2:
            return np.full(len(cell_queries), cell_values[0])
        try:
            clf = RandomForestClassifier(n_estimators=_n, max_depth=max_depth, random_state=_rs)
            clf.fit(cell_points, cell_values)
            return clf.predict(cell_queries).astype(np.float64)
        except Exception:
            return np.full(len(cell_queries), np.nan)

    return set_cell_params_fn, classifier_fn


def _make_dt_fns(max_depth=None, min_samples_split=None, random_state=None):
    """Wrap sklearn DecisionTreeClassifier in the C++ callback protocol."""
    from sklearn.tree import DecisionTreeClassifier
    _mss = min_samples_split if min_samples_split is not None else 2
    _rs  = random_state if random_state is not None else np.random.randint(0, 100000)

    def set_cell_params_fn(cell_points, cell_values):
        return np.array([], dtype=np.float64)

    def classifier_fn(cell_points, cell_values, cell_queries, cell_params):
        if len(cell_queries) == 0 or len(cell_points) == 0:
            return np.full(len(cell_queries), np.nan)
        if len(np.unique(cell_values)) < 2:
            return np.full(len(cell_queries), cell_values[0])
        try:
            clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=_mss, random_state=_rs)
            clf.fit(cell_points, cell_values)
            return clf.predict(cell_queries).astype(np.float64)
        except Exception:
            return np.full(len(cell_queries), np.nan)

    return set_cell_params_fn, classifier_fn


# Maps each named sklearn classifier to its specific hyperparameter names.
# Used in _main.py to build the hparams search grid and CV combo dicts.
SKLEARN_CLASSIFIER_PARAMS = {
    "knn": ["n_neighbors"],
    "svm": ["C", "kernel", "gamma"],
    "rf":  ["n_estimators", "max_depth"],
    "dt":  ["max_depth", "min_samples_split"],
}

# Classifiers that accept a random_state argument (single value, not searched over).
SKLEARN_RANDOM_STATE_CLASSIFIERS = {"svm", "rf", "dt"}


# ═══════════════════════════════════════════════════════════════════
#  Scikit-learn adapter
# ═══════════════════════════════════════════════════════════════════

def _make_sklearn_fns(sklearn_classifier):
    """
    Wrap a scikit-learn classifier in the C++ callback protocol.

    ``set_cell_params_fn`` returns an empty array (no-op);
    ``classifier_fn`` fits a fresh clone and predicts — stateless
    and safe across parallel partitions.
    """
    import copy

    def set_cell_params_fn(cell_points, cell_values):
        return np.array([], dtype=np.float64)

    def classifier_fn(cell_points, cell_values, cell_queries, cell_params):
        if len(cell_queries) == 0 or len(cell_points) == 0:
            return np.full(len(cell_queries), np.nan)
        if len(np.unique(cell_values)) < 2:
            return np.full(len(cell_queries), cell_values[0])
        try:
            fitted = copy.deepcopy(sklearn_classifier)
            fitted.fit(cell_points, cell_values)
            return fitted.predict(cell_queries)
        except Exception:
            return np.full(len(cell_queries), np.nan)

    return set_cell_params_fn, classifier_fn


# ═══════════════════════════════════════════════════════════════════
#  Public factory
# ═══════════════════════════════════════════════════════════════════

def get_classifier_fns(classifier: str, **kwargs) -> Tuple:
    """
    Return ``(set_cell_params_fn, classifier_fn)`` for the given classifier string.

    Parameters
    ----------
    classifier : {'knn_pca', 'scikit-learn', 'knn', 'svm', 'rf', 'dt'}
    **kwargs :
        For 'knn_pca'      : n_neighbors, max_points, n_cv_splits
        For 'scikit-learn' : sklearn_classifier
        For 'knn'          : n_neighbors
        For 'svm'          : C, kernel, gamma
        For 'rf'           : n_estimators, max_depth
        For 'dt'           : max_depth, min_samples_split
    """
    if classifier == "knn_pca":
        n_neighbors = kwargs.get("n_neighbors", None)
        max_points  = kwargs.get("max_points",  None)
        n_cv_splits = kwargs.get("n_cv_splits", None)
        seed        = kwargs.get("seed",        None)

        def set_cell_params_fn(cell_points, cell_values):
            return _set_cell_params_knn_pca(
                cell_points, cell_values,
                n_neighbors=n_neighbors,
                max_points=max_points,
                n_cv_splits=n_cv_splits,
                seed=seed,
            )

        return set_cell_params_fn, _knn_pca_predict

    elif classifier == "scikit-learn":
        sklearn_classifier = kwargs.get("sklearn_classifier")
        if sklearn_classifier is None:
            raise ValueError(
                "classifier='scikit-learn' requires the 'sklearn_classifier' argument."
            )
        return _make_sklearn_fns(sklearn_classifier)

    elif classifier == "knn":
        return _make_knn_fns(n_neighbors=kwargs.get("n_neighbors"))

    elif classifier == "svm":
        return _make_svm_fns(
            C=kwargs.get("C"),
            kernel=kwargs.get("kernel"),
            gamma=kwargs.get("gamma"),
            random_state=kwargs.get("random_state"),
        )

    elif classifier == "rf":
        return _make_rf_fns(
            n_estimators=kwargs.get("n_estimators"),
            max_depth=kwargs.get("max_depth"),
            random_state=kwargs.get("random_state"),
        )

    elif classifier == "dt":
        return _make_dt_fns(
            max_depth=kwargs.get("max_depth"),
            min_samples_split=kwargs.get("min_samples_split"),
            random_state=kwargs.get("random_state"),
        )

    else:
        raise ValueError(
            f"Unknown classifier '{classifier}'. "
            "Supported: 'knn_pca', 'scikit-learn', 'knn', 'svm', 'rf', 'dt'."
        )

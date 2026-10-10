import numpy as np
from collections import Counter
from typing import Any, Dict, List, Tuple

from . import _dawid_skene as _ds_cpp
from spatialize.logging import singleton_null_callback


# ─────────────────────────────────────────────────────────────
#  Majority vote (nominal)
# ─────────────────────────────────────────────────────────────

def aggregate_with_mv(esi_samples: np.ndarray) -> np.ndarray:
    """
    Aggregate ESI samples using a simple majority vote (mode).

    Parameters
    ----------
    esi_samples : np.ndarray, shape (p, n)
        p locations × n partitions of categorical values.

    Returns
    -------
    np.ndarray, shape (p,)
        Most-frequent category at each location.
    """
    if esi_samples.ndim != 2:
        raise ValueError("esi_samples must be a 2D array of shape (p, n).")

    n_locations, n_partitions = esi_samples.shape
    estimation = np.empty(n_locations, dtype=object)

    def majority_vote_mode(partition_samples):
        """Return the most frequent category across partitions for one location."""
        if len(partition_samples) == 0:
            return None
        counts = Counter(partition_samples)  # count occurrences of each label
        if not counts:
            return None
        # Iterate counts to find the winner; first-encountered wins on ties
        winning_category, max_count = None, 0
        for category, count in counts.items():
            if count > max_count:
                max_count         = count
                winning_category  = category
        return winning_category

    for location_idx in range(n_locations):
        row = [x for x in esi_samples[location_idx, :] if x is not None]
        estimation[location_idx] = majority_vote_mode(row)

    return estimation


# ─────────────────────────────────────────────────────────────
#  Weighted majority vote (nominal)
# ─────────────────────────────────────────────────────────────

def aggregate_with_weighted_mv(esi_samples: np.ndarray, weights) -> np.ndarray:
    """
    Aggregate ESI samples with a weighted majority vote: the category whose members carry the largest
    total weight.

    An alternative to the majority vote, which weighs every partition alike, for partitions known to
    be of unequal reliability, for instance by a validation score of each partition. With equal
    weights it is the majority vote. :func:`aggregate_with_btd` learns the reliabilities from the
    members instead.

    Parameters
    ----------
    esi_samples : np.ndarray, shape (p, n)
        p locations × n partitions of categorical values (``None`` for a member of an empty cell,
        which takes no part).
    weights : array_like, shape (n,)
        Non-negative weight of each partition.

    Returns
    -------
    np.ndarray, shape (p,)
        The category of largest total weight at each location (the first one met on ties, as the
        majority vote), ``None`` where no member is defined.
    """
    if esi_samples.ndim != 2:
        raise ValueError("esi_samples must be a 2D array of shape (p, n).")
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != esi_samples.shape[1]:
        raise ValueError(f"weights has {w.size} entries but there are {esi_samples.shape[1]} partitions.")
    if (w < 0).any() or not np.isfinite(w).all():
        raise ValueError("weights must be finite and non-negative.")
    estimation = np.empty(esi_samples.shape[0], dtype=object)
    for i, row in enumerate(esi_samples):
        totals = {}
        for x, wx in zip(row, w):
            if x is not None:
                totals[x] = totals.get(x, 0.0) + wx
        best, best_w = None, -1.0
        for category, total in totals.items():
            if total > best_w:
                best, best_w = category, total
        estimation[i] = best
    return estimation


# ─────────────────────────────────────────────────────────────
#  Median vote (ordinal)
# ─────────────────────────────────────────────────────────────

def aggregate_with_ordinal_mv(
    esi_samples: np.ndarray,
    ordinal_category_order: Dict[str, int],
) -> np.ndarray:
    """
    Aggregate ESI samples using a median vote for ordinal categories.

    Parameters
    ----------
    esi_samples : np.ndarray, shape (p, n)
        p locations × n partitions of ordinal categorical values.
    ordinal_category_order : dict
        Mapping from category string to integer rank, e.g.
        ``{'Low': 0, 'Medium': 1, 'High': 2}``.

    Returns
    -------
    np.ndarray, shape (p,)
        Median category at each location.
    """
    if esi_samples.ndim != 2:
        raise ValueError("esi_samples must be a 2D array of shape (p, n).")
    if not ordinal_category_order:
        raise ValueError("ordinal_category_order must be provided for ordinal aggregation.")

    n_locations, _ = esi_samples.shape

    # Normalize keys to strings so that int keys {0: 0, 1: 1} and string keys
    # {'Low': 0, 'High': 1} both match decoded sample values (which are str-compared).
    # rank_to_category preserves the original key type so the returned estimation
    # has the same type as the user's original category labels.
    str_keyed_order = {str(k): v for k, v in ordinal_category_order.items()}
    rank_to_category = {v: k for k, v in ordinal_category_order.items()}

    # Validate that every non-None value in the samples has a known rank
    unique_sample_vals = {v for v in esi_samples.ravel() if v is not None}
    for value in unique_sample_vals:
        if str(value) not in str_keyed_order:
            raise ValueError(
                f"Category '{value}' found in esi_samples but not in ordinal_category_order."
            )

    estimation = np.empty(n_locations, dtype=object)

    def ordinal_median_vote(partition_samples):
        """Return the median category across partitions for one location."""
        if len(partition_samples) == 0:
            return None

        # Convert categories to integer ranks, then sort to find the median
        ranks = [str_keyed_order[str(item)] for item in partition_samples]
        ranks.sort()
        n = len(ranks)

        if n % 2 == 1:
            # Odd number of partitions: the middle rank is unambiguous
            median_rank = ranks[n // 2]
        else:
            # Even number of partitions: average the two central ranks and find
            # the existing category closest to that average
            median_rank_lower = ranks[n // 2 - 1]
            median_rank_upper = ranks[n // 2]
            true_median_value = (median_rank_lower + median_rank_upper) / 2.0
            median_rank = min(str_keyed_order.values(),
                              key=lambda rank: abs(rank - true_median_value))

        return rank_to_category[median_rank]

    for location_idx in range(n_locations):
        row = [x for x in esi_samples[location_idx, :] if x is not None]
        estimation[location_idx] = ordinal_median_vote(row)

    return estimation


# ─────────────────────────────────────────────────────────────
#  Precision scoring
# ─────────────────────────────────────────────────────────────

def categorical_precision_cube(
    estimation: np.ndarray,
    esi_samples: np.ndarray,
) -> np.ndarray:
    """
    Per-location, per-partition uncertainty indicator (0-1 loss).

    Analogous to ``mse_cube`` / ``mae_cube`` in ESI: returns the unaggregated
    loss over all partitions rather than collapsing to a scalar per location.
    Higher values indicate higher uncertainty.

    - 1.0 where the partition prediction *disagrees* with the estimation.
    - 0.0 where it agrees.

    ``categorical_feature_precision`` is the mean of this cube along axis 1.

    Parameters
    ----------
    estimation : np.ndarray, shape (p,)
    esi_samples : np.ndarray, shape (p, n)

    Returns
    -------
    np.ndarray, shape (p, n), float in {0.0, 1.0}.
    """
    # 1 where a partition disagrees with the aggregated estimate, 0 where it agrees
    return (esi_samples != estimation[:, None]).astype(np.float64)


def categorical_feature_precision(
    estimation: np.ndarray,
    esi_samples: np.ndarray,
) -> np.ndarray:
    """
    Per-location uncertainty: proportion of ESI samples that disagree with
    the aggregated estimation (mean 0-1 loss over partitions).

    Analogous to ``mse_loss`` / ``mae_loss`` in ESI. Higher values indicate
    higher uncertainty.

    - Value → 0: all partitions agree (low uncertainty).
    - Value → 1: partitions spread across categories (high uncertainty).

    Equivalent to ``categorical_precision_cube(...).mean(axis=1)``.

    Parameters
    ----------
    estimation : np.ndarray, shape (p,)
        Aggregated category predictions (e.g. from majority vote).
    esi_samples : np.ndarray, shape (p, n)
        Full ensemble samples across all partitions.

    Returns
    -------
    np.ndarray, shape (p,), float in [0, 1].
    """
    return categorical_precision_cube(estimation, esi_samples).mean(axis=1)


# ─────────────────────────────────────────────────────────────
#  Dawid-Skene EM aggregation (compiled module spatialize.gs.cat_esi._dawid_skene)
# ─────────────────────────────────────────────────────────────


def aggregate_with_btd(
    esi_samples: np.ndarray,
    category_type: str = "nominal",
    categories_list: list = None,
    ordinal_order_map: Dict = None,
    map_dimensions: Tuple[int, ...] = None,
    spatial_constraints: List[Tuple[str, str]] = None,
    tolerance: float = 1e-4,
    max_iter: int = 100,
    spatial_penalty_factor: float = 1.0,
    seed: int = None,
    callback=None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Aggregate ESI samples using the Dawid-Skene EM model (C++ backend).

    Parameters
    ----------
    esi_samples : np.ndarray, shape (p, n)
    category_type : {'nominal', 'ordinal', 'binary'}
    categories_list : list of valid category strings
    ordinal_order_map : dict mapping category to rank (for ordinal type)
    map_dimensions : tuple of ints, spatial grid shape (e.g. (200, 300))
    spatial_constraints : list of (cat_a, cat_b) forbidden-adjacent pairs
    tolerance : EM convergence threshold
    max_iter : maximum EM iterations
    spatial_penalty_factor : float
        Weight of the spatial constraints (mean field): at each EM iteration a category loses, at each
        location, the factor times its neighbours' current probabilities of a category it may not
        touch. It is set by the user; 0 gives the non-spatial model.
    seed : int, optional
        Seed of the random initialisation of the confusion matrices; drawn at each call when
        ``None``. EM reaches a local maximum, so a fixed seed makes the result reproducible.
    callback : callable, optional
        Receives the messages and the progress of EM (one step per iteration) through Spatialize's
        protocol; by default they are shown as the session settings ``display``, ``progress`` and
        ``verbosity`` say. The threads come from the session (``parallel``, ``num_threads``); the
        result is the same for any number of threads. Ctrl-C stops between iterations.

    Returns
    -------
    estimation : np.ndarray of object, shape (p,)
        The most probable category at each location.
    probabilities : np.ndarray, shape (p, K)
        The posterior probability of each category at each location, the columns in the order of
        ``categories_list`` (sorted categories of the samples when it is ``None``).
    log_likelihood : float
        The final marginal log-likelihood.

    Notes
    -----
    A binary variable is a nominal variable with two categories: each partition then has a 2×2
    confusion matrix (its sensitivity and specificity), the binary model of truth discovery.

    Examples
    --------
    Basic usage after a ``cat_esi_nongriddata`` call::

        result = cat_esi_nongriddata(points, values, xi, classifier="knn_pca")
        samples = result.esi_samples(raw=True)
        categories = list(np.unique(values).astype(str))

        estimation, probs, ll = aggregate_with_btd(
            samples,
            categories_list=categories,
            map_dimensions=(200, 300),
            spatial_constraints=[('20.0', '29.0')],
            spatial_penalty_factor=8.9,
        )

    Update the result object with the BTD estimation::

        import functools
        btd_agg = functools.partial(
            aggregate_with_btd,
            categories_list=categories,
            map_dimensions=(200, 300),
            spatial_constraints=[('20.0', '29.0')],
            spatial_penalty_factor=8.9,
        )
        result.re_estimate(agg_function=btd_agg)
    """
    num_items = esi_samples.shape[0]
    if map_dimensions is not None and int(np.prod(map_dimensions)) != num_items:
        raise ValueError(
            f"Product of map_dimensions {map_dimensions} ({int(np.prod(map_dimensions))}) "
            f"must equal number of items ({num_items})."
        )
    # the members as category codes (-1: a member of an empty cell, no claim): no string claims
    import pandas as pd
    samples = np.asarray(esi_samples, dtype=object)
    defined = ~pd.isna(samples)
    code_names, inverse = np.unique(samples[defined].astype(str), return_inverse=True)
    codes = np.full(samples.shape, -1, dtype=np.int32)
    codes[defined] = inverse

    spatial_constraints_safe = spatial_constraints if spatial_constraints is not None else []
    from spatialize.gs import _num_threads
    from spatialize.logging import resolve_callback
    cpp_kwargs = dict(
        codes=codes,
        code_names=[str(c) for c in code_names],
        categories_list=categories_list,
        tolerance=tolerance,
        max_iter=max_iter,
        category_type=category_type,
        ordinal_order_map=ordinal_order_map,
        map_dimensions=tuple(int(d) for d in map_dimensions) if map_dimensions is not None else None,
        spatial_constraints=spatial_constraints_safe,
        spatial_penalty_factor=spatial_penalty_factor,
        seed=int(np.random.default_rng().integers(2 ** 31 - 1)) if seed is None else int(seed),
        visitor=resolve_callback(callback),
        num_threads=_num_threads(),
    )

    estimation, by_item, log_likelihood = _ds_cpp.aggregate_codes(**cpp_kwargs)
    cats = [str(c) for c in categories_list] if categories_list is not None else sorted(
        {str(c) for d in by_item.values() for c in d})
    probabilities = np.zeros((num_items, len(cats)))
    for item, d in by_item.items():
        i = int(item.split("_")[1]) - 1
        for k, c in enumerate(cats):
            probabilities[i, k] = d.get(c, 0.0)
    return np.array(estimation, dtype=object), probabilities, float(log_likelihood)

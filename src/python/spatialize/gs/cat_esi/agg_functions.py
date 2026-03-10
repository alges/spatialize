import numpy as np
from collections import Counter
from typing import Any, Dict, List, Tuple


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
        # Convert to strings so that mixed types (int, float, str) compare consistently
        string_samples = [str(x) for x in partition_samples]
        counts = Counter(string_samples)
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

def categorical_feature_precision(
    esi_samples: np.ndarray,
    esi_estimation: np.ndarray,
) -> np.ndarray:
    """
    Compute per-location precision as the proportion of ESI samples that
    match the aggregated estimation.

    Parameters
    ----------
    esi_samples : np.ndarray, shape (p, n)
    esi_estimation : np.ndarray, shape (p,)

    Returns
    -------
    np.ndarray, shape (p,), float – proportion in [0, 1].
    """
    if esi_samples.shape[0] != esi_estimation.shape[0]:
        raise ValueError("The p dimension of esi_samples and esi_estimation must match.")
    if esi_estimation.ndim != 1:
        raise ValueError("esi_estimation must be a 1-D array.")

    n_locations, n_partitions = esi_samples.shape
    precision_scores = np.empty(n_locations, dtype=np.float64)

    for location_idx in range(n_locations):
        # Compare as strings so numeric codes and string labels are handled uniformly
        estimated_category   = str(esi_estimation[location_idx])
        matching_partitions  = np.sum([str(x) == estimated_category
                                       for x in esi_samples[location_idx, :]])
        precision_scores[location_idx] = matching_partitions / n_partitions

    return precision_scores


# ─────────────────────────────────────────────────────────────
#  Optional: Dawid-Skene EM aggregation (requires compiled C++ module)
# ─────────────────────────────────────────────────────────────

try:
    import dawid_skene_em_cpp as _ds_cpp

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
    ) -> Tuple[np.ndarray, Any, float]:
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
        spatial_penalty_factor : weight for spatial constraints

        Returns
        -------
        (estimation, probabilities, log_likelihood)

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

        # Build item → spatial coordinates mapping required by the C++ model
        item_coordinates = {}
        if map_dimensions is not None:
            total_cells = int(np.prod(map_dimensions))
            if total_cells != num_items:
                raise ValueError(
                    f"Product of map_dimensions {map_dimensions} ({total_cells}) "
                    f"must equal number of items ({num_items})."
                )
            n_spatial_dims = len(map_dimensions)
            for item_idx in range(num_items):
                coords = [0] * n_spatial_dims
                remaining = item_idx
                # Decompose flat index into N-dimensional coordinates (row-major)
                for dim in range(n_spatial_dims - 1, -1, -1):
                    coords[dim]  = remaining % map_dimensions[dim]
                    remaining   //= map_dimensions[dim]
                item_coordinates[f"Item_{item_idx + 1}"] = tuple(coords)

        spatial_constraints_safe = spatial_constraints if spatial_constraints is not None else []

        # Only activate spatial model when all required inputs are present
        use_spatial_model = bool(map_dimensions and item_coordinates and spatial_constraints_safe)

        cpp_kwargs = dict(
            esi_samples=esi_samples.tolist(),
            categories_list=categories_list,
            tolerance=tolerance,
            max_iter=max_iter,
            category_type=category_type,
            ordinal_order_map=ordinal_order_map,
            map_dimensions=map_dimensions           if use_spatial_model else None,
            item_coordinates=item_coordinates       if use_spatial_model else {},
            spatial_constraints=spatial_constraints_safe if use_spatial_model else [],
        )
        if use_spatial_model:
            cpp_kwargs["spatial_penalty_factor"] = spatial_penalty_factor

        result = _ds_cpp.aggregate_with_btd_em(**cpp_kwargs)
        return np.array(result[0], dtype=object), result[1], result[2]

    def objective_function_for_spatial_penalty(
        penalty_factor_array: np.ndarray,
        esi_samples: np.ndarray,
        categories_list: list,
        map_dimensions: Tuple[int, ...],
        spatial_constraints: List[Tuple[str, str]],
        tolerance: float = 1e-4,
        max_iter: int = 100,
    ) -> float:
        """
        Objective function for scipy optimizers: returns the negative marginal
        log-likelihood for a given ``spatial_penalty_factor``.

        Intended to be passed to ``scipy.optimize.basinhopping`` or ``minimize``
        via the ``func`` / ``fun`` argument.

        Parameters
        ----------
        penalty_factor_array : np.ndarray, shape (1,)
            Current penalty factor value (wrapped in an array by scipy).
        esi_samples : np.ndarray, shape (p, n)
        categories_list : list of category strings
        map_dimensions : tuple of ints, spatial grid shape
        spatial_constraints : list of (cat_a, cat_b) forbidden-adjacent pairs
        tolerance : EM convergence threshold
        max_iter : maximum EM iterations

        Returns
        -------
        float – negative log-likelihood (minimise this to maximise likelihood)

        Examples
        --------
        Use directly with ``scipy.optimize.minimize``::

            from scipy.optimize import minimize
            result = minimize(
                objective_function_for_spatial_penalty,
                x0=[1.0],
                args=(samples, categories, (200, 300), [('20.0', '29.0')]),
                method='L-BFGS-B',
                bounds=[(1e-6, None)],
            )
            optimal_penalty = result.x[0]

        Prefer ``optimize_btd_spatial_penalty`` for the full basin-hopping workflow.
        """
        penalty_factor = float(penalty_factor_array[0])
        if penalty_factor <= 0:
            return np.inf
        try:
            _, _, log_likelihood = aggregate_with_btd(
                esi_samples=esi_samples,
                categories_list=categories_list,
                map_dimensions=map_dimensions,
                spatial_constraints=spatial_constraints,
                tolerance=tolerance,
                max_iter=max_iter,
                spatial_penalty_factor=penalty_factor,
                category_type="nominal",
            )
            return -log_likelihood
        except Exception:
            return np.inf

    def optimize_btd_spatial_penalty(
        esi_samples: np.ndarray,
        categories_list: list,
        map_dimensions: Tuple[int, ...],
        spatial_constraints: List[Tuple[str, str]],
        initial_penalty: float = 5.0,
        penalty_bounds: Tuple[float, float] = (1.0, 20.0),
        niter: int = 5,
        T: float = 2.0,
        stepsize: float = 1.0,
        tolerance: float = 1e-4,
        max_iter: int = 100,
    ) -> Tuple[float, float]:
        """
        Find the optimal ``spatial_penalty_factor`` for ``aggregate_with_btd``
        using basin-hopping global optimisation (scipy).

        Parameters
        ----------
        esi_samples : np.ndarray, shape (p, n)
        categories_list : list of category strings
        map_dimensions : tuple of ints, spatial grid shape (e.g. ``(200, 300)``)
        spatial_constraints : list of ``(cat_a, cat_b)`` forbidden-adjacent pairs
        initial_penalty : starting guess for the penalty factor
        penalty_bounds : ``(low, high)`` search bounds
        niter : number of basin-hopping iterations (more → wider search)
        T : basin-hopping temperature (higher → more uphill moves accepted)
        stepsize : maximum random jump size between basins
        tolerance : EM convergence threshold passed to ``aggregate_with_btd``
        max_iter : maximum EM iterations passed to ``aggregate_with_btd``

        Returns
        -------
        optimal_penalty_factor : float
        optimal_log_likelihood : float

        Examples
        --------
        Full workflow after estimation::

            result = cat_esi_nongriddata(points, values, xi, classifier="knn_pca")
            samples = result.esi_samples(raw=True)
            categories = list(np.unique(values).astype(str))

            optimal_penalty, log_ll = optimize_btd_spatial_penalty(
                samples,
                categories_list=categories,
                map_dimensions=(200, 300),
                spatial_constraints=[('20.0', '29.0')],
            )

            estimation, probs, ll = aggregate_with_btd(
                samples,
                categories_list=categories,
                map_dimensions=(200, 300),
                spatial_constraints=[('20.0', '29.0')],
                spatial_penalty_factor=optimal_penalty,
            )
        """
        from scipy.optimize import basinhopping

        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": [penalty_bounds],
            "options": {
                "disp": False,
                "maxiter": 50,
                "maxls": 20,
                "eps": 0.05,
                "gtol": 1e-4,
            },
            "args": (esi_samples, categories_list, map_dimensions,
                     spatial_constraints, tolerance, max_iter),
        }

        result = basinhopping(
            func=objective_function_for_spatial_penalty,
            x0=np.array([initial_penalty]),
            niter=niter,
            T=T,
            stepsize=stepsize,
            minimizer_kwargs=minimizer_kwargs,
        )

        return float(result.x[0]), float(-result.fun)

except ImportError:
    pass  # BTD functions not available without the compiled dawid_skene_em_cpp module

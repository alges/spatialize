import numpy as np
import math
import libspatialize as lsp
from spatialize import SpatializeError, logging
from spatialize.logging import log_message, default_singleton_callback, singleton_null_callback

# ============================================================================
# Shared utility functions
# ============================================================================

def _get_esi_estimates(points, values, xi, T, alpha_t, exp, seed, callback):
    """
    Get T spatial estimates for each location in xi using ESI-IDW.

    Parameters
    ----------
    points : array-like, shape (n_samples, n_dims)
        Sample locations. Must be 2D array.
    values : array-like, shape (n_samples,)
        Sample values corresponding to points.
    xi : array-like, shape (n_targets, n_dims)
        Target locations for estimation.
    T : int
        Number of spatial estimates (ensemble size). Must be > 0.
    alpha_t : float
        Spatial partition granularity parameter. Range: [0, 1].
        Higher values create smaller, more granular partitions.
    exp : float
        IDW exponent. Must be > 0.
    seed : int
        Random seed for reproducibility.
    callback : callable
        Callback function for progress reporting.

    Returns
    -------
    esi_samples : ndarray, shape (n_targets, T)
        T estimates for each target location.

    """
    points = np.asarray(points)
    values = np.asarray(values)
    xi = np.asarray(xi)

    _, esi_samples = lsp.estimation_esi_idw(
        points, values, T, alpha_t, exp, seed, xi, callback
    )
    return esi_samples


def _create_partitions(samples, M, alpha_m, seed):
    """
    Create M Mondrian partitions over the range of samples.

    Handles both 1D and 2D cases automatically based on input shape.

    Parameters
    ----------
    samples : ndarray
        Samples to partition.
        - For 1D: shape (n_samples,) or (n_samples, 1)
        - For 2D: shape (n_samples, 2)
    M : int
        Number of partition trees.
    alpha_m : float
        Range partition granularity parameter.
        Higher values create smaller, more granular partitions.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    partitions : list of ndarray
        List of M partitions as numpy arrays. Each partition contains
        cell information in format [leaf_idx, bounds...].
    """
    # Ensure samples is 2D for libspatialize
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    partitions = lsp.get_partitions_using_esi(
        samples, M, alpha_m, None, seed
    )

    # Convert to numpy arrays
    return [np.array(p) for p in partitions]


def _get_leaf_indexes(samples, M, alpha_m, seed):
    """
    Get leaf indexes for samples in the M partition trees.

    Handles both 1D and 2D cases automatically based on input shape.

    Parameters
    ----------
    samples : ndarray
        Samples to query.
        - For 1D: shape (n_samples,) or (n_samples, 1)
        - For 2D: shape (n_samples, 2)
    M : int
        Number of partition trees.
    alpha_m : float
        Range partition granularity parameter.
        Higher values create smaller, more granular partitions.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    indexes : ndarray, shape (n_samples, M)
        Leaf index for each sample in each tree.
    """
    # Ensure samples is 2D for libspatialize
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    return lsp.get_leaf_for_samples_using_esi(
        samples, M, alpha_m, None, seed
    )


class SpatialEntropy:
    """
    Computes Spatial Entropy of a regionalized variable using an ensemble scheme.

    Spatial entropy quantifies the uncertainty in the spatial distribution of a
    variable by combining:
    1. Ensemble Spatial Interpolation (ESI) to generate T spatial realizations
    2. Mondrian partitioning of the value range into M trees
    3. Differential entropy estimation via cell-based probability density

    The entropy h(U|X=x) measures the uncertainty about the value U at location x,
    averaged over M partition trees for stability.

    Attributes
    ----------
    T : int
        Number of spatial estimates (ensemble size)
    M : int
        Number of range partition trees
    alpha_t : float
        Spatial partition granularity parameter
    alpha_m : float
        Range partition granularity parameter
    seed : int
        Random seed
    progress_bar : bool
        Whether to display progress bars
    esi_samples : ndarray, optional
        Stored ESI samples from last calculation, shape (n_targets, T)

    Examples
    --------
    >>> entropy_estimator = SpatialEntropy(T=100, M=100)
    >>> entropies = entropy_estimator.calculate_entropy(points, values, xi)
    """
    def __init__(self,
                 T=100,
                 M=100,
                 alpha_t=0.9,
                 alpha_m=0.6,
                 seed=0,
                 callback=None
                 ):
        """
        Initialize Spatial Entropy estimator.

        Parameters
        ----------
        T : int, default=100
            Number of spatial estimates (ensemble size). Must be > 0.
        M : int, default=100
            Number of range partition trees. Must be > 0.
        alpha_t : float, default=0.9
            Spatial partition granularity. Range: [0, 1].
            Higher values create smaller spatial partitions.
        alpha_m : float, default=0.6
            Range partition granularity. Range: [0, 1].
            Higher values create smaller range partitions.
        seed : int, default=0
            Random seed for reproducibility.
        callback : callable, optional
            Callback function for progress reporting. If None, uses default progress callback.

        Raises
        ------
        SpatializeError
            If T or M are non-positive, or alpha values are outside [0, 1].
        """
        if T <= 0:
            raise SpatializeError(f"T must be positive, got {T}")
        if M <= 0:
            raise SpatializeError(f"M must be positive, got {M}")
        if not 0 <= alpha_t <= 1:
            raise SpatializeError(f"alpha_t must be in [0, 1], got {alpha_t}")
        if not 0 <= alpha_m <= 1:
            raise SpatializeError(f"alpha_m must be in [0, 1], got {alpha_m}")

        self.T = T
        self.M = M
        self.alpha_t = alpha_t
        self.alpha_m = alpha_m
        self.seed = seed
        self.callback = callback if callback is not None else default_singleton_callback

    def calculate_entropy(self, points, values, xi, exp=3.0):
        """
        Calculate ensemble spatial entropy for all points in xi.

        For each target location x_i, computes h(U|X=x_i) by:
        1. Generating T spatial estimates via ESI-IDW
        2. Partitioning the value range using M Mondrian trees
        3. Computing differential entropy from cell probabilities and sizes

        Parameters
        ----------
        points : array-like, shape (n_samples, n_dims)
            Sample locations.
        values : array-like, shape (n_samples,)
            Sample values.
        xi : array-like, shape (n_targets, n_dims)
            Target locations for entropy calculation.
        exp : float, default=3.0
            IDW exponent for interpolation. Must be > 0.

        Returns
        -------
        entropies : ndarray, shape (n_targets,)
            Spatial entropy estimates for each target location.

        Examples
        --------
        >>> se = SpatialEntropy(T=200, M=150)
        >>> h = se.calculate_entropy(sample_points, sample_values, target_points)
        >>> print(f"Mean entropy: {h.mean():.2f}")
        """
        log_message(logging.logger.debug(f'calculating spatial entropy for {len(xi) if hasattr(xi, "__len__") else "unknown"} target locations'))

        # Validate inputs
        xi = np.asarray(xi)
        if len(xi) == 0:
            raise SpatializeError("xi must contain at least one target location")

        log_message(logging.logger.debug(f'generating {self.T} ESI samples using alpha_t={self.alpha_t}, exponent={exp}'))

        print("Generating ESI samples...")
        # Get esi_samples for all points in xi (progress tracked by C++ code)
        esi_samples = _get_esi_estimates(points, values, xi, self.T, self.alpha_t, exp, self.seed, self.callback)
        self.esi_samples = esi_samples

        # Pre-allocate array for efficiency
        entropies = np.zeros(len(xi))

        log_message(logging.logger.debug(f'computing entropy using {self.M} partition trees with alpha_m={self.alpha_m}'))

        print("Computing entropy...")
        # Calculate entropy for each target location with progress tracking
        self.callback(logging.progress.init(len(xi), 1))

        for i in range(len(xi)):
            estimates = self.esi_samples[i]

            # Use different seed for each location to avoid identical partitions
            # but maintain reproducibility
            location_seed = self.seed + i

            # Partition samples range
            partitions = _create_partitions(estimates, self.M, self.alpha_m, location_seed)
            leaf_indexes = _get_leaf_indexes(estimates, self.M, self.alpha_m, location_seed)

            # Calculate entropy for each of the M trees
            entropies_per_tree = np.zeros(self.M)
            for m, (partition, leaf_index) in enumerate(zip(partitions, leaf_indexes.T)):
                # Count number of points per leaf
                leaf_idxs, counts = np.unique(leaf_index, return_counts=True)

                entropy_m = 0.0
                # Calculate entropy in tree m
                for j, n_jm in zip(leaf_idxs, counts):
                    # Get length of cell j from partition m (1D partitions: [leaf_idx, x_min, x_max])
                    _, x_min, x_max = partition[j]
                    length_jm = x_max - x_min

                    if length_jm > 0:
                        # Differential entropy: -sum p(j) * log2(p(j) / length(j))
                        p_jm = n_jm / self.T
                        cell_entropy = p_jm * math.log2(p_jm / length_jm)
                        entropy_m -= cell_entropy

                entropies_per_tree[m] = entropy_m

            # Get ensemble estimate for point i (average across all M trees)
            entropies[i] = np.nanmean(entropies_per_tree)

            # Update progress
            self.callback(logging.progress.inform())

        self.callback(logging.progress.stop())
        log_message(logging.logger.debug(f'spatial entropy calculation complete'))

        return entropies
 

class SpatialMutualInformation:
    """
    Computes Spatial Mutual Information between two regionalized variables using an ensemble scheme.

    Spatial mutual information quantifies the statistical dependence between two
    variables at spatial locations, computed as:

        I(U;V | X=x, Y=y) = h(U | X=x) + h(V | Y=y) - h(U,V | X=x, Y=y)

    where:
    - h(U | X=x): marginal entropy of variable U at location x
    - h(V | Y=y): marginal entropy of variable V at location y
    - h(U,V | X=x, Y=y): joint entropy of (U,V) at locations (x,y)

    The method ensures consistency by computing marginal entropies via marginalization
    of the joint 2D Mondrian partition.

    Attributes
    ----------
    T : int
        Number of spatial estimates (ensemble size)
    M : int
        Number of range partition trees
    alpha_t : float
        Spatial partition granularity parameter
    alpha_m : float
        Range partition granularity parameter
    seed : int
        Random seed
    progress_bar : bool
        Whether to display progress bars
    esi_samples_u : ndarray, optional
        Stored ESI samples for variable U, shape (n_targets, T)
    esi_samples_v : ndarray, optional
        Stored ESI samples for variable V, shape (n_targets, T)
    entropies_u : ndarray, optional
        Marginal entropies for U at each target location
    entropies_v : ndarray, optional
        Marginal entropies for V at each target location
    entropies_joint : ndarray, optional
        Joint entropies at each target location

    Examples
    --------
    >>> mi_estimator = SpatialMutualInformation(T=200, M=150)
    >>> mi = mi_estimator.calculate_mutual_information(
    ...     points_u, values_u, points_v, values_v, target_locations
    ... )
    >>> print(f"Average MI: {mi.mean():.3f}")
    """
    def __init__(self,
                 T=100,
                 M=100,
                 alpha_t=0.9,
                 alpha_m=0.6,
                 seed=0,
                 callback=None
                 ):
        """
        Initialize Spatial Mutual Information estimator.

        Parameters
        ----------
        T : int, default=100
            Number of spatial estimates (ensemble size). Must be > 0.
        M : int, default=100
            Number of range partition trees. Must be > 0.
        alpha_t : float, default=0.9
            Spatial partition granularity. Range: [0, 1].
            Higher values create smaller spatial partitions.
        alpha_m : float, default=0.6
            Range partition granularity. Range: [0, 1].
            Higher values create smaller range partitions.
        seed : int, default=0
            Random seed for reproducibility.
        callback : callable, optional
            Callback function for progress reporting. If None, uses default progress callback.

        Raises
        ------
        SpatializeError
            If T or M are non-positive, or alpha values are outside [0, 1].
        """
        if T <= 0:
            raise SpatializeError(f"T must be positive, got {T}")
        if M <= 0:
            raise SpatializeError(f"M must be positive, got {M}")
        if not 0 <= alpha_t <= 1:
            raise SpatializeError(f"alpha_t must be in [0, 1], got {alpha_t}")
        if not 0 <= alpha_m <= 1:
            raise SpatializeError(f"alpha_m must be in [0, 1], got {alpha_m}")

        self.T = T
        self.M = M
        self.alpha_t = alpha_t
        self.alpha_m = alpha_m
        self.seed = seed
        self.callback = callback if callback is not None else default_singleton_callback

    def _calculate_marginal_entropy_from_joint(self, partitions_2d, leaf_indexes_2d, marginal='u'):
        """
        Calculate marginal entropy by marginalizing a 2D joint partition.

        This ensures consistency between marginal and joint entropy calculations
        by deriving h(U) and h(V) from the same 2D partition used for h(U,V).

        Parameters
        ----------
        partitions_2d : list of ndarray
            2D Mondrian partitions from joint (U,V) distribution.
            Each partition cell has format: [leaf_idx, u_min, u_max, v_min, v_max].
        leaf_indexes_2d : ndarray, shape (T, M)
            Leaf assignments for 2D samples in each tree.
        marginal : {'u', 'v'}
            Which marginal to compute.

        Returns
        -------
        entropy : float
            Marginal entropy estimate averaged over M trees.
        """
        entropies_per_tree = np.zeros(self.M)

        for m, (partition, leaf_index) in enumerate(zip(partitions_2d, leaf_indexes_2d.T)):
            # Count number of points per 2D leaf
            leaf_idxs, counts = np.unique(leaf_index, return_counts=True)

            # Build marginal distribution by projecting 2D cells onto chosen axis
            marginal_cells = {}  # {(x_min, x_max): count}

            for j, n_jm in zip(leaf_idxs, counts):
                # Get bounds of 2D cell (format: [leaf_idx, u_min, u_max, v_min, v_max])
                _, u_min, u_max, v_min, v_max = partition[j]

                # Project onto chosen marginal axis
                if marginal == 'u':
                    cell_bounds = (u_min, u_max)
                else:  # marginal == 'v'
                    cell_bounds = (v_min, v_max)

                # Accumulate counts for this marginal cell
                if cell_bounds in marginal_cells:
                    marginal_cells[cell_bounds] += n_jm
                else:
                    marginal_cells[cell_bounds] = n_jm

            # Calculate marginal entropy from projected cells
            entropy_m = 0.0
            for (x_min, x_max), n_jm in marginal_cells.items():
                length_jm = x_max - x_min

                if length_jm > 0 and n_jm > 0:
                    # Differential entropy: -sum p(j) * log2(p(j) / length(j))
                    p_jm = n_jm / self.T
                    cell_entropy = p_jm * math.log2(p_jm / length_jm)
                    entropy_m -= cell_entropy

            entropies_per_tree[m] = entropy_m

        # Get ensemble estimate (average across all M trees)
        return np.nanmean(entropies_per_tree)

    def _calculate_joint_entropy(self, estimates_u, estimates_v, location_seed):
        """
        Calculate joint entropy h(U,V | X=x, Y=y) for a pair of target locations.

        Parameters
        ----------
        estimates_u : ndarray, shape (T,)
            T estimates for U at target location x.
        estimates_v : ndarray, shape (T,)
            T estimates for V at target location y.
        location_seed : int
            Random seed for this specific location.

        Returns
        -------
        entropy : float
            Joint entropy estimate.
        partitions : list of ndarray
            2D Mondrian partitions (returned for marginalization).
        leaf_indexes : ndarray
            Leaf assignments (returned for marginalization).
        """
        # Stack samples into 2D array for joint partitioning
        joint_samples = np.column_stack([estimates_u, estimates_v])

        # Partition joint samples range (2D)
        partitions = _create_partitions(joint_samples, self.M, self.alpha_m, location_seed)
        leaf_indexes = _get_leaf_indexes(joint_samples, self.M, self.alpha_m, location_seed)

        # Calculate joint entropy for each of the M trees
        entropies_per_tree = np.zeros(self.M)
        for m, (partition, leaf_index) in enumerate(zip(partitions, leaf_indexes.T)):
            # Count number of points per leaf
            leaf_idxs, counts = np.unique(leaf_index, return_counts=True)

            entropy_m = 0.0
            # Calculate entropy in tree m
            for j, n_jm in zip(leaf_idxs, counts):
                # Get area of cell j from partition m
                # 2D partitions have format: [leaf_idx, u_min, u_max, v_min, v_max]
                _, u_min, u_max, v_min, v_max = partition[j]
                area_jm = (u_max - u_min) * (v_max - v_min)

                if area_jm > 0:
                    # Differential entropy: -sum p(j) * log2(p(j) / area(j))
                    p_jm = n_jm / self.T
                    cell_entropy = p_jm * math.log2(p_jm / area_jm)
                    entropy_m -= cell_entropy

            entropies_per_tree[m] = entropy_m

        # Get ensemble estimate (average across all M trees)
        entropy = np.nanmean(entropies_per_tree)

        return entropy, partitions, leaf_indexes

    def calculate_mutual_information(self,
                                     points_u, values_u,
                                     points_v, values_v,
                                     xi, exp=3.0):
        """
        Calculate spatial mutual information I(U;V | X=x, Y=y) for all points in xi.

        For each target location, computes:
            I(U;V | X=x, Y=y) = h(U | X=x) + h(V | Y=y) - h(U,V | X=x, Y=y)

        where h denotes differential entropy. The method ensures consistency by
        computing marginal entropies via marginalization of the joint 2D partition.

        Parameters
        ----------
        points_u : array-like, shape (n_samples_u, n_dims)
            Sample locations for variable U.
        values_u : array-like, shape (n_samples_u,)
            Sample values for variable U.
        points_v : array-like, shape (n_samples_v, n_dims)
            Sample locations for variable V.
        values_v : array-like, shape (n_samples_v,)
            Sample values for variable V.
        xi : array-like, shape (n_targets, n_dims)
            Target locations (used for both U and V).
        exp : float, default=3.0
            IDW exponent for interpolation. Must be > 0.

        Returns
        -------
        mutual_information : ndarray, shape (n_targets,)
            Mutual information estimates for each target location.

        Examples
        --------
        >>> smi = SpatialMutualInformation(T=200, M=150)
        >>> mi = smi.calculate_mutual_information(
        ...     points_u, values_u, points_v, values_v, target_locations
        ... )
        >>> # Access intermediate results
        >>> print(f"Mean h(U): {smi.entropies_u.mean():.2f}")
        >>> print(f"Mean MI: {mi.mean():.3f}")
        """
        log_message(logging.logger.debug(f'calculating spatial mutual information for {len(xi) if hasattr(xi, "__len__") else "unknown"} target locations'))

        # Validate inputs
        xi = np.asarray(xi)
        if len(xi) == 0:
            raise SpatializeError("xi must contain at least one target location")

        log_message(logging.logger.debug(f'generating {self.T} ESI samples for variable U using alpha_t={self.alpha_t}, exponent={exp}'))

        print("Generating ESI samples for U...")
        # Get ESI estimates for both variables at all target locations (progress tracked by C++ code)
        estimates_u = _get_esi_estimates(points_u, values_u, xi, self.T, self.alpha_t, exp, self.seed, self.callback)
        self.esi_samples_u = estimates_u

        log_message(logging.logger.debug(f'generating {self.T} ESI samples for variable V using alpha_t={self.alpha_t}, exponent={exp}'))

        print("Generating ESI samples for V...")
        estimates_v = _get_esi_estimates(points_v, values_v, xi, self.T, self.alpha_t, exp, self.seed + 1, self.callback)
        self.esi_samples_v = estimates_v

        # Pre-allocate arrays for efficiency
        mutual_information = np.zeros(len(xi))
        entropies_u = np.zeros(len(xi))
        entropies_v = np.zeros(len(xi))
        entropies_joint = np.zeros(len(xi))

        log_message(logging.logger.debug(f'computing mutual information using {self.M} partition trees with alpha_m={self.alpha_m}'))

        print("Computing mutual information...")
        # Calculate MI for each target location with progress tracking
        self.callback(logging.progress.init(len(xi), 1))

        for i in range(len(xi)):
            # Get estimates for this location
            est_u = estimates_u[i]
            est_v = estimates_v[i]

            # Use different seed for each location to avoid identical partitions
            # but maintain reproducibility
            location_seed = self.seed + i + 2

            # Calculate joint entropy FIRST and get partitions
            # This returns the 2D partition that we'll marginalize for h(U) and h(V)
            h_uv, partitions_2d, leaf_indexes_2d = self._calculate_joint_entropy(
                est_u, est_v, location_seed
            )

            # Calculate marginal entropies by marginalizing the 2D joint partition
            # This ensures all three entropies come from the same partition structure
            h_u = self._calculate_marginal_entropy_from_joint(
                partitions_2d, leaf_indexes_2d, marginal='u'
            )
            h_v = self._calculate_marginal_entropy_from_joint(
                partitions_2d, leaf_indexes_2d, marginal='v'
            )

            # Store entropies
            entropies_u[i] = h_u
            entropies_v[i] = h_v
            entropies_joint[i] = h_uv

            # Calculate mutual information
            mi = h_u + h_v - h_uv
            mutual_information[i] = mi

            # Update progress
            self.callback(logging.progress.inform())

        self.callback(logging.progress.stop())

        # Store as attributes
        self.entropies_u = entropies_u
        self.entropies_v = entropies_v
        self.entropies_joint = entropies_joint

        log_message(logging.logger.debug(f'spatial mutual information calculation complete'))

        return mutual_information
 
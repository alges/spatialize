r"""Variogram inference from the ensemble (experimental).

The direction of a field's anisotropy and its correlation lengths, read from an ensemble estimate instead
of from the experimental variogram of the data. The direction comes from the structure tensor of the map
of the estimate; the lengths from the drawing members, each a field that keeps the texture of the data,
along each principal direction of the map.

The recipe of :func:`infer_variogram`, tested in the examples (notebook *Variogram inference*):

#. a Pareto search of adaptive IDW (leave-one-out, mean absolute error) over a few granularities, its knee;
#. the map of adaptive IDW at the knee on a grid, whose structure tensor gives the principal directions
   and their coherence;
#. the members of the weighted draw ``wdraw_adaptiveidw`` at the same granularity, each read through its
   directional correlation, :math:`1 - \gamma(h)/\sigma^2`, along each principal direction; the lag where
   it first falls below :math:`e^{-1}` is the member's correlation length there.

For an exponential covariance :math:`e^{-h/a}` the correlation length is the scale :math:`a`, the practical
range :math:`3a`.

The theory has no nugget of the field: under the block-mark model the covariance :math:`\kappa_2\, e(x, y)`
is continuous at the origin. The part a nugget plays falls to the residual weight :math:`1 - \sum_i w_i`,
the probability that a location's cell holds no datum, a property of the location and its data rather than
of the field, large where the data are thin. The result reports it at every node of the grid.

A measurement error of the data is bounded from above. With an independent error of variance :math:`\tau^2`
the covariance across distinct locations keeps the block-mark form :math:`\kappa_2\, e`, the error adding to
the variance at a location alone (Prop. 5.6.1, draft), so two data sharing a cell differ by their errors
and by the field's own variation inside the cell. The pooled within-cell variance of the data,

.. math::

   W(\lambda) = \mathbb E_\Pi \Big[ \frac{\sum_c \sum_{i \in c} (z_i - \bar z_c)^2}{\sum_c (m_c - 1)} \Big],

is therefore :math:`\tau^2` plus a non-negative term, the encoder error, which vanishes as the cells shrink
(Thm 7.4.4, draft). It is the order-two estimator of the theory (10.2.1) read where two locations coincide.
Its minimum over a ladder of granularities, among those where enough data still share a cell, is an upper
bound on :math:`\tau^2`, with a standard error from the independent partitions. With few data the cells
empty before the field's variation inside them vanishes, so the bound stays above the error.
"""
import numpy as np

from spatialize.futures import _experimental
from spatialize.result import Summarised

_E = float(np.exp(-1.0))


# ----------------------------------------------------------------------------- grid geometry
def _grid_geometry(xi):
    """For a rectilinear grid given as coordinate arrays (one per coordinate, all of the grid's shape),
    the coordinate that varies along each array axis and its signed step."""
    coords = [np.asarray(c, float) for c in xi]
    d = len(coords)
    shape = coords[0].shape
    if len(shape) != d:
        raise ValueError("the grid must have one array axis per coordinate")
    coord_of, step = [], []
    for a in range(d):
        if shape[a] < 2:
            raise ValueError("every axis of the grid needs at least two nodes")
        diffs = [float(np.mean(np.diff(c, axis=a))) for c in coords]
        c = int(np.argmax(np.abs(diffs)))
        if abs(diffs[c]) == 0.0:
            raise ValueError("the grid is degenerate along an axis")
        coord_of.append(c)
        step.append(diffs[c])
    if sorted(coord_of) != list(range(d)):
        raise ValueError("the grid's axes must follow the coordinates one to one")
    return np.array(coord_of), np.array(step)


def _filled(z):
    z = np.asarray(z, float)
    return np.where(np.isfinite(z), z, np.nanmean(z))


# ----------------------------------------------------------------------------- the structure tensor
def _structure_tensor(z, coord_of, step, sigma):
    """The structure tensor of a map in physical coordinates, averaged over the map."""
    from scipy.ndimage import gaussian_filter, sobel
    z = _filled(z)
    d = z.ndim
    norm = 2.0 * 4.0 ** (d - 1)                 # sobel = central difference × smoothing in the other axes
    grad = [None] * d
    for a in range(d):
        grad[coord_of[a]] = sobel(z, axis=a) / (norm * step[a])
    J = np.empty((d, d))
    for i in range(d):
        for j in range(i, d):
            J[i, j] = J[j, i] = float(gaussian_filter(grad[i] * grad[j], sigma).mean())
    return J


def _principal_axes(J):
    """Principal directions of continuity, from the longest (smallest gradient energy) to the shortest,
    as the rows of a matrix, and the coherence :math:`(\\mu_{max} - \\mu_{min}) / (\\mu_{max} + \\mu_{min})`."""
    mu, vectors = np.linalg.eigh(J)             # ascending: the first direction varies least
    axes = vectors.T.copy()
    for k in range(len(axes)):                  # a sign convention: the first non-zero component positive
        nz = np.flatnonzero(np.abs(axes[k]) > 1e-12)
        if nz.size and axes[k, nz[0]] < 0:
            axes[k] = -axes[k]
    total = mu[-1] + mu[0]
    coherence = float((mu[-1] - mu[0]) / total) if total > 0 else 0.0
    return axes, coherence


# ----------------------------------------------------------------------------- correlation lengths
def _lags(shape, step):
    extent = np.abs(step) * (np.array(shape) - 1)
    h0 = float(np.min(np.abs(step)))
    n = max(int(np.ceil(0.5 * extent.max() / h0)), 2)
    return h0 * np.arange(1, n + 1)


def _directional_semivariogram(z, direction, coord_of, step, lags):
    """Semivariogram of a map along a physical direction, the map shifted by bilinear (n-linear)
    interpolation, read over the pairs that stay inside the grid."""
    from scipy.ndimage import map_coordinates
    z = _filled(z)
    per_unit = np.array([direction[coord_of[a]] / step[a] for a in range(z.ndim)])   # array index per unit length
    idx = np.indices(z.shape).reshape(z.ndim, -1).astype(float)
    flat = z.ravel()
    upper = (np.array(z.shape) - 1)[:, None]
    gamma = np.full(len(lags), np.nan)
    for i, h in enumerate(lags):
        moved = idx + h * per_unit[:, None]
        inside = np.all((moved >= 0) & (moved <= upper), axis=0)
        if inside.sum() < 10:
            break
        shifted = map_coordinates(z, moved[:, inside], order=1)
        gamma[i] = 0.5 * float(np.mean((flat[inside] - shifted) ** 2))
    return gamma


def _efold(lags, gamma, variance):
    """Lag where the correlation :math:`1 - \\gamma/\\sigma^2` first falls below :math:`e^{-1}`, linearly
    interpolated, and whether the curve never did (the length is then at least the longest lag read)."""
    corr = 1.0 - gamma / variance
    ok = np.isfinite(corr)
    lags, corr = lags[ok], corr[ok]
    below = np.flatnonzero(corr < _E)
    if not below.size:
        return float(lags[-1]) if lags.size else np.nan, True
    k = below[0]
    h0, r0 = (0.0, 1.0) if k == 0 else (lags[k - 1], corr[k - 1])
    return float(h0 + (r0 - _E) / (r0 - corr[k]) * (lags[k] - h0)), False


def _lengths_at(gammas, variances, lags, level):
    """Each member's length along one direction where its correlation first falls below ``level``, and
    whether it never did."""
    out, censored = [], False
    for g, var in zip(gammas, variances):
        corr = 1.0 - g / var
        ok = np.isfinite(corr)
        h, c = lags[ok], corr[ok]
        below = np.flatnonzero(c < level)
        if not below.size:
            out.append(float(h[-1]) if h.size else np.nan)
            censored = True
            continue
        j = below[0]
        h0, r0 = (0.0, 1.0) if j == 0 else (h[j - 1], c[j - 1])
        out.append(float(h0 + (r0 - level) / (r0 - c[j]) * (h[j] - h0)))
    return np.array(out), censored


def _error_bound(points, values, extra, p_process="mondrian", alphas=(0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998),
                 n_partitions=100, seed=0, min_dof=30, min_shared=0.15):
    """The pooled within-cell variance of the data on a ladder of granularities and its minimum: an upper bound
    on the variance of a measurement error (see the module documentation). The ladder stops once fewer than
    ``min_dof`` degrees of freedom, or fewer than a share ``min_shared`` of the data, remain on average, the
    partitions growing costly as the cells shrink. Returns the bound, its standard error over the
    partitions, the granularity and the mean degrees of freedom there (NaN when the first granularity already
    keeps fewer)."""
    from spatialize.gs.partitions import cell_labels
    p = np.asarray(points, float)
    v = np.asarray(values, float)
    best = (np.nan, np.nan, np.nan, np.nan)
    box = np.vstack([p, np.asarray(extra, float)])

    def pooled(alpha, n):
        lab = cell_labels(p, box, p_process=p_process, alpha=float(alpha), n_partitions=int(n), seed=seed)[:len(p)]
        est, dof = [], []
        for t in range(lab.shape[1]):
            _, inv, cnt = np.unique(lab[:, t], return_inverse=True, return_counts=True)
            s1, s2 = np.bincount(inv, v), np.bincount(inv, v * v)
            d = int((cnt - 1).sum())
            if d > 0:
                est.append(float((s2 - s1 * s1 / cnt).sum()) / d)
                dof.append(d)
        return est, dof

    for a in sorted(alphas):
        # the degrees of freedom only fall as the cells shrink, and the partitions grow costly: a cheap probe
        # stops the ladder once too few data share a cell
        _, probe = pooled(a, 5)
        if not probe or np.mean(probe) < max(min_dof, min_shared * len(p)):
            break
        est, dof = pooled(a, n_partitions)
        if not est or np.mean(dof) < min_dof:
            break
        m = float(np.mean(est))
        if not np.isfinite(best[0]) or m < best[0]:
            best = (m, float(np.std(est, ddof=1) / np.sqrt(len(est))) if len(est) > 1 else np.nan, float(a),
                    float(np.mean(dof)))
    return best


# ----------------------------------------------------------------------------- the result
class VariogramInference(Summarised):
    r"""The variogram read from an ensemble: principal directions, correlation lengths and sill.

    Attributes
    ----------
    dim : int
        Dimension, 2 or 3.
    axes : ndarray of shape (dim, dim)
        Principal directions of continuity as unit vectors in the coordinates of the data, one per row,
        from the longest correlation (major) to the shortest (minor).
    coherence : float
        :math:`(\mu_{max} - \mu_{min}) / (\mu_{max} + \mu_{min})` of the map's structure tensor, from 0 (no
        preferred direction) towards 1 (strong anisotropy). Near 0 the directions are not determined and
        the lengths should agree.
    azimuth : float
        Angle of the major axis in the plane of the first two coordinates, in degrees counterclockwise from
        the first coordinate axis, in [0, 180) in 2D and [0, 360) in 3D.
    plunge : float or None
        In 3D, the angle of the major axis above the plane of the first two coordinates, in degrees
        (the axis taken with a non-negative third component); ``None`` in 2D.
    lengths : ndarray of shape (dim,)
        Correlation length along each principal direction, the mean over the members.
    intervals : ndarray of shape (dim, 2)
        The central interval of level ``level`` of the members' lengths along each direction, the spread
        of the ensemble (not a calibrated confidence interval).
    censored : ndarray of bool, shape (dim,)
        Whether some member's correlation never fell below :math:`e^{-1}` within the lags read along that
        direction, so its length is a lower bound.
    practical_ranges : ndarray of shape (dim,)
        ``3 * lengths``, the practical ranges of an exponential model.
    sill : float
        The variance of the data values.
    error_bound : float
        An upper bound on the variance of a measurement error of the data, the smallest pooled within-cell
        variance over a ladder of granularities (see the module documentation); NaN without the data.
    error_bound_share : float
        ``error_bound / sill``.
    error_bound_se : float
        Its standard error over the independent partitions, given the data. Taking the minimum over the
        granularities makes the bound slightly optimistic when several of them hold the same value, as on a
        field without structure inside the cells.
    error_bound_alpha, error_bound_dof : float
        The granularity where the bound was reached and the mean number of degrees of freedom there.
    residual_weight : ndarray or None
        The theory's residual weight at each node of the grid, the share of partitions in which the node's cell
        holds no datum (:meth:`~spatialize.gs.esi.ESIResult.empty_cell_fraction`): the part a nugget plays in a
        classical fit, as a property of the location and its data. ``None`` when the partitions are unknown.
    members_used : int
        Number of members read (those without undefined values).
    level : float
        Level of the intervals.
    alpha : float or None
        Granularity of the ensemble, the knee of the search when :func:`infer_variogram` ran it.
    curves : dict
        ``lags`` and, per direction, the semivariogram of the map (``map``) and the mean over the members
        (``members``), for plotting.
    """

    _title = "Variogram inference (experimental)"

    def __init__(self, axes, coherence, member_lengths, censored, sill, level, curves, alpha=None, search=None,
                 residual_weight=None, error_bound=(np.nan, np.nan, np.nan, np.nan)):
        self.axes = np.asarray(axes, float)
        self.dim = self.axes.shape[0]
        self.coherence = float(coherence)
        self.member_lengths = np.asarray(member_lengths, float)          # (members, dim)
        self.lengths = self.member_lengths.mean(axis=0)
        q = (1.0 - level) / 2.0
        self.intervals = np.quantile(self.member_lengths, [q, 1.0 - q], axis=0).T
        self.censored = np.asarray(censored, bool)
        self.practical_ranges = 3.0 * self.lengths
        self.sill = float(sill)
        self.residual_weight = None if residual_weight is None else np.asarray(residual_weight, float)
        self.error_bound, self.error_bound_se, self.error_bound_alpha, self.error_bound_dof = (float(x) for x in error_bound)
        self.error_bound_share = self.error_bound / self.sill if self.sill > 0 else np.nan
        self.level = float(level)
        self.members_used = int(self.member_lengths.shape[0])
        self.curves = curves
        self.alpha = alpha
        self.search = search
        major = self.axes[0]
        if self.dim == 2:
            self.azimuth = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
            self.plunge = None
        else:
            v = major if major[2] >= 0 else -major
            self.azimuth = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
            self.plunge = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))

    @property
    def anisotropy_ratios(self):
        """Each length over the major one, 1 for the major axis."""
        return self.lengths / self.lengths[0]

    def as_dict(self):
        """The model as plain numbers: ``axes``, ``azimuth``, ``plunge``, ``coherence``, ``lengths``,
        ``practical_ranges``, ``intervals``, ``sill``, ``error_bound``, ``error_bound_share``, ``error_bound_se``,
        ``alpha``."""
        return dict(axes=self.axes.tolist(), azimuth=self.azimuth, plunge=self.plunge, coherence=self.coherence,
                    lengths=self.lengths.tolist(), practical_ranges=self.practical_ranges.tolist(),
                    intervals=self.intervals.tolist(), sill=self.sill, error_bound=self.error_bound,
                    error_bound_share=self.error_bound_share, error_bound_se=self.error_bound_se, alpha=self.alpha)

    def _summary(self):
        from spatialize import _display
        names = ["major", "minor"] if self.dim == 2 else ["major", "intermediate", "minor"]
        direction = [("azimuth (°)", self.azimuth)] + ([("plunge (°)", self.plunge)] if self.plunge is not None else [])
        direction += [("coherence", self.coherence)]
        if self.coherence < 0.05:
            direction += [("note", "no clear direction")]
        model = [("members read", self.members_used), ("sill", self.sill), ("granularity (alpha)", self.alpha),
                 ("error variance, at most", self.error_bound), ("its share of the sill", self.error_bound_share),
                 ("its standard error", self.error_bound_se), ("bound at alpha", self.error_bound_alpha)]
        if self.residual_weight is not None:
            model += [("residual weight, mean", float(np.nanmean(self.residual_weight))),
                      ("residual weight, max", float(np.nanmax(self.residual_weight)))]
        rows = []
        for k, name in enumerate(names):
            lo, hi = self.intervals[k]
            rows.append([name, " ".join(f"{x:+.3f}" for x in self.axes[k]), self.lengths[k], lo, hi,
                         self.practical_ranges[k], "yes" if self.censored[k] else "no"])
        return _display.Summary(
            self._title, subtitle=f"{self.dim}D · exponential lengths",
            blocks=[("kv", "Direction", direction), ("kv", "Ensemble", model),
                    ("table", "Correlation lengths", ["axis", "direction", "length",
                                                      f"q{(1 - self.level) / 2 * 100:g}",
                                                      f"q{(1 + self.level) / 2 * 100:g}",
                                                      "practical range", "lower bound"], rows)],
            footer="axes · lengths · intervals · practical_ranges · error_bound · residual_weight · as_dict() · plot()")

    def plot(self, axes=None, show=False):
        """The semivariogram of the members (mean) and of the map along each principal direction, with the
        lengths marked.

        Parameters
        ----------
        axes : sequence of matplotlib Axes, optional
            One per direction. Default: a new figure.
        show : bool, optional
            Call ``plt.show()``. Default: ``False``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        names = ["major", "minor"] if self.dim == 2 else ["major", "intermediate", "minor"]
        if axes is None:
            fig, axes = plt.subplots(1, self.dim, figsize=(4.6 * self.dim, 3.6), sharey=True)
        else:
            fig = axes[0].figure
        h = self.curves["lags"]
        for k, ax in enumerate(axes):
            ax.plot(h, self.curves["members"][k], color="C3", label="members, mean")
            ax.plot(h, self.curves["map"][k], color="C0", ls="--", label="map")
            ax.axhline(self.sill, color="0.6", lw=0.8)
            ax.axvline(self.lengths[k], color="C3", lw=0.8, ls=":")
            ax.set_title(f"{names[k]} axis, length {self.lengths[k]:.3g}", fontsize=9)
            ax.set_xlabel("lag")
        axes[0].set_ylabel("semivariance")
        axes[0].legend(fontsize=8)
        if show:
            plt.show()
        return fig


# ----------------------------------------------------------------------------- readings
def read_variogram(result, members=None, values=None, sigma=2.0, level=0.9, seed=0):
    r"""Read the variogram from an ensemble estimate on a grid.

    The principal directions come from the structure tensor of the estimate's map; the correlation
    lengths from the members, read along each direction (see the module documentation).

    Parameters
    ----------
    result : ESIResult
        An estimate on a regular grid (:func:`~spatialize.gs.esi.esi_griddata`), in 2 or 3 dimensions,
        whose map gives the directions.
    members : ESIResult, optional
        An estimate on the same grid whose members give the lengths, preferably from a drawing decoder
        (``wdraw_adaptiveidw``), whose members keep the texture of the data. Default: the members of
        ``result`` (averaging members are smoother, so their lengths come out longer).
    values : array_like, optional
        The data values, whose variance is the sill. Default: those ``result`` carries.
    sigma : float, optional
        Smoothing of the structure tensor, in grid cells. Default: 2.
    level : float, optional
        Level of the intervals over the members. Default: 0.9.
    seed : int, optional
        Seed of the partitions the bound on the measurement error is read with. Default: 0.

    Returns
    -------
    VariogramInference

    Raises
    ------
    ValueError
        If the estimate is not on a grid of 2 or 3 dimensions, or no member is fully defined.
    """
    _experimental("Variogram inference")
    if not getattr(result, "griddata", False) or result._xi is None:
        raise ValueError("variogram inference reads an estimate on a regular grid (esi_griddata)")
    xi = result._xi
    if len(xi) not in (2, 3):
        raise ValueError("variogram inference works in 2 or 3 dimensions")
    coord_of, step = _grid_geometry(xi)
    z = np.asarray(result.estimation(), float)
    axes, coherence = _principal_axes(_structure_tensor(z, coord_of, step, sigma))
    source = members if members is not None else result
    mem = np.asarray(source.esi_samples(), float)
    if mem.shape[:-1] != z.shape:
        raise ValueError("the members must be on the estimate's grid")
    finite = [t for t in range(mem.shape[-1]) if np.isfinite(mem[..., t]).all()]
    if not finite:
        raise ValueError("no member is defined at every node of the grid; estimate with the session setting "
                         "empty_cells='coarsen' (spatialize.session) so that empty cells take a value")
    vals = values if values is not None else result.values
    sill = float(np.var(np.asarray(vals, float))) if vals is not None else float(np.nanvar(z))
    lags = _lags(z.shape, step)
    gammas, variances = [[] for _ in axes], []
    for t in finite:
        zt = mem[..., t]
        variances.append(float(np.var(zt)))
        for k, u in enumerate(axes):
            gammas[k].append(_directional_semivariogram(zt, u, coord_of, step, lags))
    lengths, censored = [], np.zeros(len(axes), bool)
    for k in range(len(axes)):
        lk, censored[k] = _lengths_at(gammas[k], variances, lags, _E)
        lengths.append(lk)
    try:
        residual = np.asarray(source.empty_cell_fraction(), float)
    except Exception:
        residual = None
    bound = (np.nan, np.nan, np.nan, np.nan)
    pts = getattr(result, "points", None)
    if pts is not None and vals is not None:
        process = (getattr(result, "partition", None) or {}).get("p_process", "mondrian")
        nodes = np.asarray(result._xi_flat, float)
        corners = np.array(np.meshgrid(*[[c.min(), c.max()] for c in nodes.T], indexing="ij")).reshape(nodes.shape[1], -1).T
        bound = _error_bound(pts, vals, corners, p_process=process, seed=seed)   # the corners fix the grid's box
    curves = dict(lags=lags, members=[np.nanmean(np.array(g), axis=0) for g in gammas],
                  map=[_directional_semivariogram(z, u, coord_of, step, lags) for u in axes])
    return VariogramInference(axes, coherence, np.array(lengths).T, censored, sill, level, curves,
                              alpha=(result.partition or {}).get("alpha") if hasattr(result, "partition") else None,
                              residual_weight=residual, error_bound=bound)


def _default_grid(points, grid_size):
    lo, hi = points.min(axis=0), points.max(axis=0)
    h = float((hi - lo).max()) / (grid_size - 1)
    axes = [lo[c] + h * np.arange(int(np.floor((hi[c] - lo[c]) / h)) + 1) for c in range(points.shape[1])]
    axes = [a if len(a) >= 2 else np.array([lo[c], lo[c] + h]) for c, a in enumerate(axes)]
    return tuple(np.meshgrid(*axes, indexing="ij"))


def infer_variogram(points, values, xi=None, grid_size=None, alpha=(0.8, 0.9, 0.95, 0.98), n_partitions=100,
                    seed=0, sigma=2.0, level=0.9, callback=None):
    r"""Infer the anisotropy and the correlation lengths of a field from its data, through the ensemble.

    Runs the recipe of the module documentation: a Pareto search of adaptive IDW over ``alpha`` (leave-one-
    out, mean absolute error), the map at the knee, the members of ``wdraw_adaptiveidw`` at the same
    granularity, and :func:`read_variogram` on them.

    Parameters
    ----------
    points : array_like of shape (n, d)
        Data locations, ``d`` = 2 or 3.
    values : array_like of shape (n,)
        Data values.
    xi : tuple of ndarray, optional
        The grid to read the maps on, as coordinate arrays (``np.meshgrid``). Default: a regular grid over
        the box of the data with ``grid_size`` nodes along its longest side.
    grid_size : int, optional
        Default: 64 in 2D, 24 in 3D.
    alpha : sequence of float, optional
        Granularities searched. Default: 0.8, 0.9, 0.95, 0.98.
    n_partitions : int, optional
        Partitions of the search, the map and the members. Default: 100.
    seed : int, optional
        Seed of the search and of both ensembles. Default: 0.
    sigma, level : float, optional
        As in :func:`read_variogram`.
    callback : callable, optional
        Where progress and messages go (:mod:`spatialize.logging`). Default: the session's settings.

    Returns
    -------
    VariogramInference
        With ``alpha`` the knee of the search and ``search`` the search's result.

    Notes
    -----
    The three ensembles run with the session setting ``empty_cells="coarsen"`` whatever the session says,
    since the lengths need members defined at every node of the grid.

    Tested on synthetic stationary Gaussian fields with exponential covariance in 2D (examples, notebook
    *Variogram inference*), where it recovered the direction within a few degrees and the lengths along and
    across the structures far better than directional experimental variograms of the data.
    """
    from spatialize.gs.esi import esi_griddata, esi_pareto_hparams_search
    import spatialize.gs.esi.scoring as scoring
    _experimental("Variogram inference")
    points = np.asarray(points, float)
    values = np.asarray(values, float)
    d = points.shape[1]
    if d not in (2, 3):
        raise ValueError("variogram inference works in 2 or 3 dimensions")
    if xi is None:
        xi = _default_grid(points, grid_size or (64 if d == 2 else 24))
    from spatialize import session
    with session.override(empty_cells="coarsen"):        # the lengths need members defined at every node
        search = esi_pareto_hparams_search(points, values, local_interpolator="adaptiveidw",
                                           n_partitions=[int(n_partitions)], alpha=list(alpha), k="loo",
                                           scoring=scoring.mae, seed=seed, callback=callback)
        knee = float(search.best_result(strategy="knee")["alpha"])
        estimate = esi_griddata(points, values, xi, local_interpolator="adaptiveidw", n_partitions=int(n_partitions),
                                alpha=knee, seed=seed, callback=callback)
        members = esi_griddata(points, values, xi, local_interpolator="wdraw_adaptiveidw",
                               n_partitions=int(n_partitions), alpha=knee, seed=seed, callback=callback)
    out = read_variogram(estimate, members=members, values=values, sigma=sigma, level=level, seed=seed)
    out.alpha, out.search = knee, search
    return out

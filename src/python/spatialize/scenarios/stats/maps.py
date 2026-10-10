"""Map functionals for the visual criteria.

A visual criterion ("the anisotropy is clearly visible", "no blocks along the axes", ...) is tested
through a number computed from a gridded map — a *map functional* — whose threshold is fixed before
any reference run. The functionals here use numpy and scipy only.

Maps are 2D arrays indexed ``[row = y, column = x]`` on a regular grid of the unit square (the
layout of :func:`spatialize.scenarios.generators.fields.grid` reshaped to ``(m, m)``).
"""
import numpy as np
from scipy.ndimage import gaussian_filter, sobel


def orientation_coherence(z, sigma=2.0):
    r"""Dominant elongation direction and its strength (coherence) of a map.

    Parameters
    ----------
    z : array_like of shape (m, m)
        The map.
    sigma : float, default 2.0
        Width, in pixels, of the Gaussian smoothing of the structure tensor.

    Returns
    -------
    theta : float
        Elongation direction in degrees in [0, 180), counter-clockwise from the x axis.
    coherence : float
        In [0, 1]: 0 for an isotropic map, 1 for perfectly parallel structures.

    Notes
    -----
    Structure tensor :math:`J = \langle G_\sigma * (\nabla z\, \nabla z^\top)\rangle` (Sobel
    gradients, Gaussian smoothing, mean over the grid); with eigenvalues
    :math:`\mu_1 \ge \mu_2`, coherence is :math:`(\mu_1-\mu_2)/(\mu_1+\mu_2)`. Gradients point
    across the structures, hence the 90° added to the dominant gradient direction. On the 40×40 truth
    fields of scenario S03 (anisotropy at 30°) it gives θ = 26–29° and coherence 0.70–0.76; an
    isotropic control gives coherence 0.05–0.09.
    """
    z = np.asarray(z, float)
    gx, gy = sobel(z, axis=1), sobel(z, axis=0)
    jxx, jyy, jxy = (gaussian_filter(a, sigma).mean() for a in (gx * gx, gy * gy, gx * gy))
    mu = np.linalg.eigvalsh([[jxx, jxy], [jxy, jyy]])
    coherence = float((mu[1] - mu[0]) / (mu[1] + mu[0])) if mu.sum() > 0 else 0.0
    theta = float((np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy)) + 90.0) % 180.0)
    return theta, coherence


def contrast_ratio(z, truth):
    """Spread of a map relative to the truth.

    Parameters
    ----------
    z, truth : array_like
        The map and the true field on the same grid (NaN ignored).

    Returns
    -------
    float
        ``nanstd(z) / nanstd(truth)``; below 1 for a washed-out map.
    """
    return float(np.nanstd(z) / np.nanstd(truth))


def axis_artifact_index(z, truth, tol_deg=5.0):
    """Share of gradient energy within ±``tol_deg`` of the coordinate axes, minus the truth's share.

    Parameters
    ----------
    z, truth : array_like of shape (m, m)
        The map and the true field.
    tol_deg : float, default 5.0
        Half-width of the angular band around each axis.

    Returns
    -------
    float
        Difference of shares (positive: more axis-aligned gradients than the truth).

    Notes
    -----
    Not used by the catalogue: it is confounded with smoothing (maps of rotation-invariant
    estimators get the same values as those of axis-aligned ones). Use :func:`axis_lock`.
    """
    def share(a):
        gx, gy = sobel(a, axis=1), sobel(a, axis=0)
        e = gx ** 2 + gy ** 2
        ang = np.degrees(np.arctan2(gy, gx)) % 90.0
        near = (ang <= tol_deg) | (ang >= 90.0 - tol_deg)
        return float(e[near].sum() / e.sum()) if e.sum() > 0 else 0.0
    return share(np.asarray(z, float)) - share(np.asarray(truth, float))


def axis_diagonal_log_ratio(z, r_min=2):
    """Spectral energy on the coordinate axes against the diagonals.

    Parameters
    ----------
    z : array_like of shape (m, m)
        The map (NaN are replaced by its mean).
    r_min : int, default 2
        Smallest frequency radius used (the lowest rings hold too few bins).

    Returns
    -------
    float
        Mean over the frequency rings of the log ratio of the mean power on the axes to the mean
        power on the diagonals. Axis-aligned edges raise it.

    Notes
    -----
    The map is centred and tapered with a Hann window — without it, the jump where the map wraps
    around puts energy on the axes. For each integer radius ``r`` from ``r_min`` to ``m // 2`` the
    bins with ``kx == 0`` or ``ky == 0`` are compared with those with ``abs(kx) == abs(ky)``.
    On its own the value is confounded with smoothing (diagonal bins sit off the integer radius),
    so use it through a difference between maps of equal smoothness, as :func:`axis_lock` does.
    """
    z = np.asarray(z, float)
    z = np.where(np.isfinite(z), z, np.nanmean(z))
    z = z - z.mean()
    w = np.hanning(z.shape[0])
    z = z * np.outer(w, w)
    n = z.shape[0]
    power = np.abs(np.fft.fft2(z)) ** 2
    k = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(k, k)
    ring = np.rint(np.hypot(kx, ky)).astype(int)
    axis, diag = (kx == 0) | (ky == 0), np.abs(kx) == np.abs(ky)
    ratios = []
    for r in range(r_min, n // 2):
        a, d = axis & (ring == r), diag & (ring == r)
        if a.any() and d.any():
            ratios.append(np.log(power[a].mean() / power[d].mean()))
    return float(np.mean(ratios))


def axis_lock(z, z_rotated):
    """Axis-locking of an estimator.

    Parameters
    ----------
    z : array_like of shape (m, m)
        Map of the estimator.
    z_rotated : array_like of shape (m, m)
        Map of the same estimator on the same data rotated about the domain centre, evaluated at the
        same physical points (so both maps are aligned pixel by pixel).

    Returns
    -------
    float
        ``axis_diagonal_log_ratio(z) - axis_diagonal_log_ratio(z_rotated)``.

    Notes
    -----
    Both maps share data, decoder and smoothing, so only artefacts tied to the coordinate axes
    remain: about 0 for a rotation-invariant partition (Voronoi), positive for axis-aligned cells
    (Mondrian). On scenario S03, single Mondrian members give +2.0 to +2.9 and medians of 100
    members about +0.3; Voronoi estimators give about 0.
    """
    return axis_diagonal_log_ratio(z) - axis_diagonal_log_ratio(z_rotated)


def level_set_iou(a_mask, b_mask):
    """Intersection over union of two regions.

    Parameters
    ----------
    a_mask, b_mask : array_like of bool
        The regions, on the same grid.

    Returns
    -------
    float
        ``|A ∩ B| / |A ∪ B|``; 1 when both are empty.
    """
    a, b = np.asarray(a_mask, bool), np.asarray(b_mask, bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def directional_range(z, theta_deg, max_lag=None):
    r"""Correlation length of a map along a direction, in pixels.

    Parameters
    ----------
    z : array_like of shape (m, m)
        The map, indexed ``[y, x]``.
    theta_deg : float
        Direction in degrees, counter-clockwise from the x axis (the convention of
        :func:`orientation_coherence`).
    max_lag : float, optional
        Largest lag tried, in pixels. Default: half the side of the map.

    Returns
    -------
    float
        The lag at which the correlation between the map and its copy shifted along the direction
        first falls below :math:`e^{-1}`, linearly interpolated between half-pixel lags; ``max_lag``
        when it never does.

    Notes
    -----
    The shifted copy is read by bilinear interpolation, over the pixels whose shifted position lies
    inside the map.
    """
    from scipy.ndimage import map_coordinates
    z = np.asarray(z, float)
    m = z.shape[0]
    max_lag = m / 2 if max_lag is None else float(max_lag)
    u = np.array([np.sin(np.radians(theta_deg)), np.cos(np.radians(theta_deg))])   # (dy, dx)
    yy, xx = np.mgrid[0:m, 0:m].astype(float)
    prev_h, prev_r = 0.0, 1.0
    for h in np.arange(0.5, max_lag + 1e-9, 0.5):
        y2, x2 = yy + h * u[0], xx + h * u[1]
        inside = (y2 >= 0) & (y2 <= m - 1) & (x2 >= 0) & (x2 <= m - 1)
        if inside.sum() < 10:
            break
        shifted = map_coordinates(z, [y2[inside], x2[inside]], order=1)
        a = z[inside]
        r = float(np.corrcoef(a, shifted)[0, 1]) if a.std() > 0 and shifted.std() > 0 else 0.0
        if r < np.exp(-1):
            return float(prev_h + (prev_r - np.exp(-1)) / (prev_r - r) * (h - prev_h))
        prev_h, prev_r = h, r
    return max_lag


def range_ratio(z, theta_deg):
    """Correlation length along ``theta_deg`` over that across it (:func:`directional_range`).

    Parameters
    ----------
    z : array_like of shape (m, m)
    theta_deg : float

    Returns
    -------
    float
        Above 1 for a map elongated along ``theta_deg``.
    """
    return directional_range(z, theta_deg) / directional_range(z, theta_deg + 90.0)


def roughness(z):
    r"""Roughness of a map: the mean squared difference between neighbouring pixels.

    Parameters
    ----------
    z : array_like of shape (m, m)

    Returns
    -------
    float
        :math:`\big(\overline{(\Delta_x z)^2} + \overline{(\Delta_y z)^2}\big) / (4\, \mathrm{var}\, z)`,
        1 for white noise, near 0 for a smooth map, larger where the map jumps between pixels.
    """
    z = np.asarray(z, float)
    v = np.nanvar(z)
    if v == 0:
        return 0.0
    return float((np.nanmean(np.diff(z, axis=1) ** 2) + np.nanmean(np.diff(z, axis=0) ** 2)) / (4 * v))

"""Map functionals for the visual criteria. numpy/scipy only.

Maps are 2D arrays indexed [row = y, column = x] on a regular grid of the unit square.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, sobel


def orientation_coherence(z, sigma=2.0):
    """Dominant elongation direction (degrees in [0, 180), from the x axis) and coherence in [0, 1].

    Structure tensor J = mean over the grid of G_σ * (∇z ∇zᵀ); with eigenvalues μ₁ ≥ μ₂,
    coherence = (μ₁ − μ₂)/(μ₁ + μ₂). Gradients point across the structure, hence the +90°.
    Calibrated on the S03 truth fields: θ = 26–29° for a 30° anisotropy, coherence 0.70–0.76;
    isotropic control 0.05–0.09.
    """
    z = np.asarray(z, float)
    gx, gy = sobel(z, axis=1), sobel(z, axis=0)
    jxx, jyy, jxy = (gaussian_filter(a, sigma).mean() for a in (gx * gx, gy * gy, gx * gy))
    mu = np.linalg.eigvalsh([[jxx, jxy], [jxy, jyy]])
    coherence = float((mu[1] - mu[0]) / (mu[1] + mu[0])) if mu.sum() > 0 else 0.0
    theta = float((np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy)) + 90.0) % 180.0)
    return theta, coherence


def angular_error(theta, target):
    """Smallest absolute difference between two axial directions (degrees, period 180)."""
    d = abs(theta - target) % 180.0
    return min(d, 180.0 - d)


def contrast_ratio(z, truth):
    """Standard deviation of the map over that of the truth (below 1: a washed-out map)."""
    return float(np.nanstd(z) / np.nanstd(truth))


def axis_artifact_index(z, truth, tol_deg=5.0):
    """Share of gradient energy within ±tol of the coordinate axes, minus the truth's share."""
    def share(a):
        gx, gy = sobel(a, axis=1), sobel(a, axis=0)
        e = gx ** 2 + gy ** 2
        ang = np.degrees(np.arctan2(gy, gx)) % 90.0
        near = (ang <= tol_deg) | (ang >= 90.0 - tol_deg)
        return float(e[near].sum() / e.sum()) if e.sum() > 0 else 0.0
    return share(np.asarray(z, float)) - share(np.asarray(truth, float))


def level_set_iou(a_mask, b_mask):
    """Intersection over union of two boolean masks (1 when both are empty)."""
    a, b = np.asarray(a_mask, bool), np.asarray(b_mask, bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0

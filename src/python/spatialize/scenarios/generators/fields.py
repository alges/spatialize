"""Reference generators of the truth fields.

The normative description of each scenario's truth lives in its ``scenario.yaml``; these functions
are the reference implementation used to materialise ``pinned`` data and, in ``fresh`` mode, to
draw new fields. Any implementation may regenerate fields with its own RNG from the same
description: no criterion depends on a particular realisation.
"""
import numpy as np


def grid(m):
    """Cell centres ((i+½)/m, (j+½)/m) of an m×m grid on the unit square, row-major (y rows)."""
    g = (np.arange(m) + 0.5) / m
    gx, gy = np.meshgrid(g, g)
    return np.c_[gx.ravel(), gy.ravel()]


def uniform_design(rng, n, dim=2):
    return rng.random((n, dim))


def _aniso_matrix(a1, a2, theta_deg):
    t = np.deg2rad(theta_deg)
    rot = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    return rot.T @ np.diag([a1 ** -2, a2 ** -2]) @ rot


def sgf_exponential(rng, points, a1, a2=None, theta_deg=0.0, jitter=1e-8):
    """Stationary Gaussian field with covariance exp(−√(hᵀAh)) (12.3.8–12.3.9), drawn JOINTLY at
    all ``points`` by Cholesky, so samples and grid belong to one realisation. a2=None → isotropic."""
    a2 = a1 if a2 is None else a2
    A = _aniso_matrix(a1, a2, theta_deg)
    h = points[:, None, :] - points[None, :, :]
    d = np.sqrt(np.maximum(np.einsum("ijk,kl,ijl->ij", h, A, h), 0.0))
    chol = np.linalg.cholesky(np.exp(-d) + jitter * np.eye(len(points)))
    return chol @ rng.standard_normal(len(points))


def vbm(rng, n_cells, mark_sampler, dim=2):
    """Voronoi block-mark field (Def 12.3.8): returns a function x -> Z(x)."""
    centres = rng.random((n_cells, dim))
    marks = mark_sampler(rng, n_cells)

    def field(x):
        d = ((np.asarray(x)[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        return marks[np.argmin(d, axis=1)]
    return field

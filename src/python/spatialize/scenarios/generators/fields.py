"""Reference generators of the truth fields and reference estimators.

The normative description of each scenario's truth lives in its ``scenario.yaml``; these functions
are the reference implementation used to materialise its pinned data
(:mod:`spatialize.scenarios.generators.materialise`). No criterion depends on a particular
realisation: another implementation could regenerate fields from the same description with its
own random numbers. :func:`simple_kriging_exponential` computes the reference map of criterion V5.
"""
import numpy as np


def grid(m):
    """Cell centres of an m×m grid on the unit square.

    Parameters
    ----------
    m : int
        Cells per side.

    Returns
    -------
    ndarray of shape (m*m, 2)
        Points ``((i+½)/m, (j+½)/m)``, row-major with rows along y, so that ``values.reshape(m, m)``
        is indexed ``[y, x]``.
    """
    g = (np.arange(m) + 0.5) / m
    gx, gy = np.meshgrid(g, g)
    return np.c_[gx.ravel(), gy.ravel()]


def uniform_design(rng, n, dim=2):
    """``n`` points uniformly distributed on the unit cube of dimension ``dim``.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    n, dim : int
        Number of points and dimension.

    Returns
    -------
    ndarray of shape (n, dim)
    """
    return rng.random((n, dim))


def _aniso_matrix(a1, a2, theta_deg):
    t = np.deg2rad(theta_deg)
    rot = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    return rot.T @ np.diag([a1 ** -2, a2 ** -2]) @ rot


def sgf_exponential(rng, points, a1, a2=None, theta_deg=0.0, jitter=1e-8):
    r"""Stationary Gaussian field with an anisotropic exponential covariance, drawn jointly.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    points : ndarray of shape (n, 2)
        Every location of the realisation (e.g. samples stacked over a grid).
    a1, a2 : float
        Ranges along and across the anisotropy direction; ``a2=None`` gives an isotropic field.
    theta_deg : float, default 0.0
        Direction of the long range, in degrees from the x axis.
    jitter : float, default 1e-8
        Diagonal added before the Cholesky factorisation.

    Returns
    -------
    ndarray of shape (n,)
        One realisation at ``points`` (zero mean, unit variance).

    Notes
    -----
    Covariance :math:`C(h) = \exp(-\sqrt{h^\top A h})` with
    :math:`A = R_\vartheta^\top \operatorname{diag}(a_1^{-2}, a_2^{-2}) R_\vartheta`. All points
    are drawn at once by Cholesky factorisation, so samples and grid belong to one realisation.
    """
    a2 = a1 if a2 is None else a2
    A = _aniso_matrix(a1, a2, theta_deg)
    h = points[:, None, :] - points[None, :, :]
    d = np.sqrt(np.maximum(np.einsum("ijk,kl,ijl->ij", h, A, h), 0.0))
    chol = np.linalg.cholesky(np.exp(-d) + jitter * np.eye(len(points)))
    return chol @ rng.standard_normal(len(points))


def simple_kriging_exponential(samples, values, queries, a1, a2=None, theta_deg=0.0, jitter=1e-8):
    r"""Simple kriging with the covariance of :func:`sgf_exponential` (known zero mean).

    The best linear predictor of the field when its covariance is known: the reference map of
    criterion V5 in scenario S03.

    Parameters
    ----------
    samples : ndarray of shape (n, 2)
        Data locations.
    values : ndarray of shape (n,)
        Data values.
    queries : ndarray of shape (q, 2)
        Prediction locations.
    a1, a2, theta_deg : float
        Covariance parameters, as in :func:`sgf_exponential`.
    jitter : float, default 1e-8
        Diagonal added to the data covariance matrix.

    Returns
    -------
    ndarray of shape (q,)
        :math:`C_{qs} C_{ss}^{-1} z`.
    """
    a2 = a1 if a2 is None else a2
    A = _aniso_matrix(a1, a2, theta_deg)

    def cov(p, q):
        h = p[:, None, :] - q[None, :, :]
        return np.exp(-np.sqrt(np.maximum(np.einsum("ijk,kl,ijl->ij", h, A, h), 0.0)))
    s = np.asarray(samples, float)
    c_ss = cov(s, s) + jitter * np.eye(len(s))
    return cov(np.asarray(queries, float), s) @ np.linalg.solve(c_ss, np.asarray(values, float))


def vbm(rng, n_cells, mark_sampler, dim=2):
    """Voronoi block-mark field: one mark per cell of a Voronoi tessellation.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator (nuclei uniform on the unit cube, then marks).
    n_cells : int
        Number of Voronoi cells.
    mark_sampler : callable
        ``mark_sampler(rng, n)`` returns ``n`` marks, e.g. ``lambda r, n: r.lognormal(0, 1, n)``.
    dim : int, default 2
        Dimension.

    Returns
    -------
    callable
        ``field(x)``: the mark of the nearest nucleus for each row of ``x``.
    """
    centres = rng.random((n_cells, dim))
    marks = mark_sampler(rng, n_cells)

    def field(x):
        d = ((np.asarray(x)[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        return marks[np.argmin(d, axis=1)]
    return field


def simple_kriging_exponential_local(samples, values, queries, k, a1, a2=None, theta_deg=0.0, jitter=1e-8):
    r"""Simple kriging as :func:`simple_kriging_exponential`, each query from its ``k`` nearest data.

    The cropped neighbourhood changes from one query to the next, so the map jumps where it does:
    the rough reference of criterion V9 in scenario S03.

    Parameters
    ----------
    samples, values, queries, a1, a2, theta_deg, jitter
        As for :func:`simple_kriging_exponential`.
    k : int
        Number of nearest data (Euclidean) each query uses.

    Returns
    -------
    ndarray of shape (q,)
    """
    from scipy.spatial import cKDTree
    s, v, q = np.asarray(samples, float), np.asarray(values, float), np.asarray(queries, float)
    _, idx = cKDTree(s).query(q, k=min(int(k), len(s)))
    idx = np.reshape(idx, (len(q), -1))
    return np.array([simple_kriging_exponential(s[i], v[i], q[j:j + 1], a1, a2, theta_deg, jitter)[0]
                     for j, i in enumerate(idx)])


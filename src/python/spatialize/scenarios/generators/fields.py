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


def preferential_design(rng, n_uniform, n_box, box, dim=2):
    """A preferential design: ``n_uniform`` points uniform on the unit cube, then ``n_box`` more uniform
    in the sub-box ``box``, as a survey sampled more densely where early results looked good.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    n_uniform, n_box : int
        Number of points over the whole cube and inside ``box``.
    box : array_like of shape (dim, 2)
        Lower and upper bounds of the preferred region along each coordinate.
    dim : int, default 2
        Dimension.

    Returns
    -------
    ndarray of shape (n_uniform + n_box, dim)
        The uniform points first.
    """
    box = np.asarray(box, float)
    inner = box[:, 0] + (box[:, 1] - box[:, 0]) * rng.random((n_box, dim))
    return np.vstack([rng.random((n_uniform, dim)), inner])


def ordinary_kriging_exponential(samples, values, queries, practical_range, nugget=0.0):
    r"""Global ordinary kriging under an isotropic exponential variogram of unit sill, the classical baseline
    of scenarios S05 and S06.

    The variogram is :math:`\gamma(h) = c_0 + (1 - c_0)(1 - e^{-3h/r})` for :math:`h > 0`, ``nugget``
    :math:`c_0` and practical range :math:`r` (the convention of the kriging decoder). The weights do not
    depend on the sill, which is therefore fixed at 1. They take either sign, so the map can leave the range
    of the data.

    Parameters
    ----------
    samples : ndarray of shape (n, 2)
        Data locations.
    values : ndarray of shape (n,)
        Data values.
    queries : ndarray of shape (q, 2)
        Prediction locations.
    practical_range, nugget : float
        Variogram parameters, the nugget as a share of the unit sill.

    Returns
    -------
    ndarray of shape (q,)
        The kriged values, the mean estimated under the constraint that the weights sum to 1.
    """
    s = np.asarray(samples, float)
    n = len(s)
    c1 = 1.0 - float(nugget)

    def cov(p, q):
        return c1 * np.exp(-3.0 * np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2) / float(practical_range))
    lhs = np.ones((n + 1, n + 1))
    lhs[:n, :n] = cov(s, s)
    np.fill_diagonal(lhs[:n, :n], 1.0)
    lhs[n, n] = 0.0
    rhs = np.ones((n + 1, len(queries)))
    rhs[:n] = cov(s, np.asarray(queries, float))
    return np.linalg.solve(lhs, rhs)[:n].T @ np.asarray(values, float)


def block_kriging_variance_exponential(samples, block_points, practical_range, nugget=0.0):
    r"""Ordinary-kriging variance of block averages under an isotropic exponential variogram of unit sill.

    The variogram is :math:`\gamma(h) = c_0 + (1 - c_0)(1 - e^{-3h/r})` for :math:`h > 0`, ``nugget``
    :math:`c_0` and practical range :math:`r` (the convention of the kriging decoder). The variance
    depends on the variogram and on the locations alone, never on the data values, which is the
    property of the kriging-variance proxy for confidence that scenario S09 compares with the law.

    Parameters
    ----------
    samples : ndarray of shape (n, 2)
        Data locations.
    block_points : list of ndarray of shape (p, 2)
        The points discretising each block.
    practical_range, nugget : float
        Variogram parameters, the nugget as a share of the unit sill.

    Returns
    -------
    ndarray of shape (len(block_points),)
        :math:`\bar C(B, B) - \sum_i \lambda_i \bar C(v_i, B) - \mu` for each block.
    """
    s = np.asarray(samples, float)
    n = len(s)
    c1 = 1.0 - float(nugget)

    def cov(p, q, same=False):
        c = c1 * np.exp(-3.0 * np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2) / float(practical_range))
        if same:
            np.fill_diagonal(c, 1.0)
        return c
    lhs = np.ones((n + 1, n + 1))
    lhs[:n, :n] = cov(s, s, same=True)
    lhs[n, n] = 0.0
    rhs = np.ones((n + 1, len(block_points)))
    within = np.empty(len(block_points))
    for j, g in enumerate(block_points):
        g = np.asarray(g, float)
        rhs[:n, j] = cov(s, g).mean(axis=1)
        within[j] = cov(g, g, same=True).mean()
    sol = np.linalg.solve(lhs, rhs)
    return within - (sol[:n] * rhs[:n]).sum(axis=0) - sol[n]


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


def mondrian_cells(rng, rate, dim=2):
    r"""The cells of the theory's Mondrian process of budget ``rate`` on the unit cube.

    A cell of linear dimension :math:`L` (the sum of its sides) is cut after an exponential time of rate
    :math:`L`, along an axis chosen with probability proportional to its side, at a uniform position,
    while the remaining budget allows; the two halves carry on independently. Two points share a cell
    with probability :math:`\exp(-\lambda \lVert x - y \rVert_1)`, and on the unit square the expected
    number of cells is :math:`(1 + \lambda)^2`.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    rate : float
        The budget :math:`\lambda`.
    dim : int, default 2
        Dimension.

    Returns
    -------
    list of tuple of ndarray
        The cells, as ``(lower corner, upper corner)``.
    """
    cells = []
    stack = [(np.zeros(dim), np.ones(dim), float(rate))]
    while stack:
        lo, hi, budget = stack.pop()
        sides = hi - lo
        t = rng.exponential(1.0 / sides.sum())
        if t > budget:
            cells.append((lo, hi))
            continue
        d = rng.choice(dim, p=sides / sides.sum())
        cut = rng.uniform(lo[d], hi[d])
        left_hi, right_lo = hi.copy(), lo.copy()
        left_hi[d] = right_lo[d] = cut
        stack.append((right_lo, hi, budget - t))
        stack.append((lo, left_hi, budget - t))
    return cells


def mbm(rng, rate, mark_sampler, dim=2):
    r"""Mondrian block-mark field: one mark per cell of the theory's Mondrian process of budget ``rate``
    on the unit cube (:func:`mondrian_cells`), the marks independent of the cells.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator (cells, then marks).
    rate : float
        The budget :math:`\lambda`; on the unit square the expected number of cells is
        :math:`(1 + \lambda)^2`.
    mark_sampler : callable
        ``mark_sampler(rng, n)`` returns ``n`` marks.
    dim : int, default 2
        Dimension.

    Returns
    -------
    callable
        ``field(x)``: the mark of the cell holding each row of ``x``.
    """
    cells = mondrian_cells(rng, rate, dim)
    marks = mark_sampler(rng, len(cells))
    lows = np.array([c[0] for c in cells])
    highs = np.array([c[1] for c in cells])

    def field(x):
        x = np.asarray(x, float)
        inside = np.all((x[:, None, :] >= lows[None]) & (x[:, None, :] < highs[None]), axis=2)
        # a point on the upper edge of the cube belongs to the cell that reaches it
        inside |= np.all((x[:, None, :] >= lows[None]) & (x[:, None, :] <= highs[None]), axis=2) & ~inside.any(axis=1)[:, None]
        return marks[np.argmax(inside, axis=1)]
    field.n_cells = len(cells)
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


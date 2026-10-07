"""Which locations share a cell of the encoder's random partitions, and the laws that follow.

The encoder of an ensemble estimator is a random partition of the domain (see the Theory pages,
*Encoders*). These functions draw the partitions an estimation would draw, with the same
``p_process``, ``alpha``, ``n_partitions`` and ``seed``, and read from them which locations share a
cell, the co-occurrence of sets of locations and the law of the groupings of a few locations. They
also give the closed forms the theory's Mondrian process follows, to compare with.

Examples
--------
>>> import numpy as np
>>> from spatialize.gs.partitions import co_occurrence, mondrian_co_occurrence
>>> pts = np.random.default_rng(0).random((50, 2))
>>> pair = [np.array([[0.40, 0.50], [0.50, 0.50]])]
>>> p_hat = co_occurrence(pts, pair, p_process="mondrian-raw", alpha=0.8, n_partitions=4000, seed=1)
>>> bool(abs(p_hat[0] - mondrian_co_occurrence(pair, rate=2.5)[0]) < 0.05)
True
"""
import numpy as np

from spatialize import SpatializeError
from spatialize._util import random_seed
from spatialize.gs import lib_spatialize_facade, partitioning_process

__all__ = ["cell_labels", "co_occurrence", "partition_law", "mondrian_co_occurrence",
           "poisson_line_partition_law"]


def _alpha(p_process, alpha, data_cond):
    # Spatialize's convention: a negative alpha draws the Voronoi nuclei uniformly in the box
    if p_process == partitioning_process.VORONOI and not data_cond:
        return -abs(alpha)
    return alpha


def cell_labels(points, xi, p_process="mondrian", alpha=0.8, n_partitions=500, seed=None, data_cond=True):
    """The cell of each location in each of the encoder's partitions.

    Parameters
    ----------
    points : array_like of shape (n, d)
        The data locations, which set the box (with ``xi``) and, for the data-conditioned Voronoi
        partition, the candidate nuclei.
    xi : array_like of shape (m, d)
        The locations to label.
    p_process : {"mondrian", "mondrian-raw", "voronoi"}, optional
        The partition process. Default: ``"mondrian"``.
    alpha : float, optional
        The granularity, as in :func:`~spatialize.gs.esi.esi_griddata`. Default: ``0.8``.
    n_partitions : int, optional
        Number of partitions drawn. Default: ``500``.
    seed : int, optional
        Seed of the partitions; with the same seed an estimation draws the same partitions.
        Default: a random integer drawn at each call.
    data_cond : bool, optional
        For ``"voronoi"``, whether the nuclei are drawn among the data. Default: ``True``.

    Returns
    -------
    ndarray of int, shape (m, n_partitions)
        Two locations share a cell of partition ``t`` exactly when their labels in column ``t`` are
        equal; ``-1`` marks a location outside the box.
    """
    seed = random_seed.factory() if seed is None else seed
    return lib_spatialize_facade.cells(np.asarray(points, np.float32), np.asarray(xi, np.float32), p_process,
                                       _alpha(p_process, alpha, data_cond), n_partitions, seed)


def co_occurrence(points, sets, **kwargs):
    """The share of partitions in which every location of each set lies in one cell.

    Parameters
    ----------
    points : array_like of shape (n, d)
        The data locations (see :func:`cell_labels`).
    sets : sequence of array_like, each of shape (k_i, d)
        The sets of locations.
    **kwargs
        ``p_process``, ``alpha``, ``n_partitions``, ``seed`` and ``data_cond``, as in
        :func:`cell_labels`. All the sets are labelled under the same partitions.

    Returns
    -------
    ndarray of shape (len(sets),)
        The estimated co-occurrence :math:`\\hat e(S)` of each set.
    """
    sets = [np.atleast_2d(np.asarray(S, np.float32)) for S in sets]
    labels = cell_labels(points, np.vstack(sets), **kwargs)
    out, start = [], 0
    for S in sets:
        block = labels[start:start + len(S)]
        start += len(S)
        out.append(float(np.mean(np.all((block == block[0]) & (block[0] >= 0), axis=0))))
    return np.array(out)


def partition_law(points, xi, **kwargs):
    """The frequencies of the groupings of a few locations into cells.

    Parameters
    ----------
    points : array_like of shape (n, d)
        The data locations (see :func:`cell_labels`).
    xi : array_like of shape (m, d)
        The locations to group; the number of groupings grows fast with ``m``, so ``m`` should stay
        small.
    **kwargs
        As in :func:`cell_labels`.

    Returns
    -------
    dict
        For every grouping observed, its frequency among the partitions. A grouping is a tuple with
        one entry per location, the index of the first location of its block, so ``(0, 0, 2)`` puts
        the first two locations in one cell and the third alone.
    """
    labels = cell_labels(points, xi, **kwargs)
    counts = {}
    for t in range(labels.shape[1]):
        col = labels[:, t]
        key = tuple(int(np.argmax(col == c)) for c in col)
        counts[key] = counts.get(key, 0) + 1
    n = labels.shape[1]
    return {k: c / n for k, c in sorted(counts.items(), key=lambda kv: -kv[1])}


def mondrian_co_occurrence(sets, rate):
    """The theory's co-occurrence of each set under the Mondrian process of rate ``rate``.

    Parameters
    ----------
    sets : sequence of array_like, each of shape (k_i, d)
    rate : float
        The rate :math:`\\lambda`; for Spatialize's ``alpha`` on a box :math:`H`,
        :math:`\\lambda = 1/(\\mu(H)(1-\\alpha))`.

    Returns
    -------
    ndarray of shape (len(sets),)
        :math:`\\exp(-\\lambda \\sum_c \\mathrm{range}_c(S))` for each set.
    """
    return np.array([float(np.exp(-rate * np.sum(np.ptp(np.atleast_2d(np.asarray(S, float)), axis=0))))
                     for S in sets])


def poisson_line_partition_law(xi, rate):
    """The theory's law of the groupings of points on a line under Poisson cuts of rate ``rate``.

    On a line the Mondrian process cuts at a Poisson process, so each gap :math:`g` between
    consecutive points is cut independently with probability :math:`1 - e^{-\\lambda g}`.

    Parameters
    ----------
    xi : array_like of shape (m,) or (m, 1)
        The points, in increasing order.
    rate : float

    Returns
    -------
    dict
        Grouping (as in :func:`partition_law`) to probability, over the :math:`2^{m-1}` groupings
        into intervals of consecutive points.
    """
    x = np.asarray(xi, float).ravel()
    if np.any(np.diff(x) <= 0):
        raise SpatializeError("the points must be in increasing order")
    keep = np.exp(-rate * np.diff(x))
    law = {}
    for cuts in np.ndindex(*(2,) * len(keep)):
        p = float(np.prod([1 - keep[g] if c else keep[g] for g, c in enumerate(cuts)]))
        lab, block = [0], 0
        for g, c in enumerate(cuts):
            if c:
                block = g + 1
            lab.append(block)
        law[tuple(lab)] = law.get(tuple(lab), 0.0) + p
    return dict(sorted(law.items(), key=lambda kv: -kv[1]))

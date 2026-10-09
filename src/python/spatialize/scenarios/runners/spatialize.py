"""Spatialize's own runner — the only module of the suite that imports spatialize's estimators.

The runner holds no list of decoders or partitions. It asks spatialize's facade
(:mod:`spatialize.gs`), whose catalogue the compiled extension declares, which combinations exist in
a dimension, then computes the members through the same facade the public API uses, so every
partition and decoder of the catalogue is tested the way users run it.

Decoder names are those of the catalogue (``"idw"``, ``"kriging"``, ``"adaptiveidw"``, ...). A
scenario must declare every decoder parameter explicitly, since the parameters are pre-registered
and the runner fills in no default. Rates are translated into spatialize's ``alpha`` by
:func:`alpha_from_rate` (Mondrian) and :func:`alpha_from_intensity` (Voronoi).
"""
import itertools

import numpy as np

from ..protocol import EstimatorSpec

#: Encoder profiles this runner implements: profile -> (spatialize's ``p_process``, ``data_cond``).
PROFILES = {
    "mondrian": ("mondrian", True),        # Spatialize's default Mondrian, the theory's process since 1.3
    "mondrian-theory": ("mondrian", True),  # the theory's Mondrian process, named for the shared suite
    "mondrian-legacy": ("mondrian-legacy", True),  # the Mondrian partition of Spatialize 1.2
    "voronoi": ("voronoi", False),         # Spatialize's Voronoi, nuclei uniform in the box (alpha < 0)
    "voronoi-data": ("voronoi", True),     # Spatialize's Voronoi, nuclei among the samples (alpha >= 0)
}

#: Empty-cell policies this runner implements (Spatialize's session setting ``empty_cells``).
POLICIES = ("nan", "mark", "coarsen")


def alpha_from_rate(rate, domain):
    r"""Spatialize's Mondrian granularity α from a Mondrian rate λ and a box H.

    Parameters
    ----------
    rate : float
        Rate (lifetime) λ of the Mondrian process.
    domain : sequence of (low, high)
        The box H. Spatialize measures it on the data, so the runner passes the box of the samples.

    Returns
    -------
    float
        :math:`\alpha = 1 - 1/(\lambda\,\mu(H))`, with :math:`\mu(H)` the sum of the box's sides
        (Egaña et al., 2021, eq. 8).

    Raises
    ------
    ValueError
        If the rate is too small for the box (α ≥ 1).

    Notes
    -----
    Inverts λ = 1/(μ(H)(1 − α)).
    """
    mu = float(sum(hi - lo for lo, hi in domain))
    alpha = 1.0 - 1.0 / (float(rate) * mu)
    if not alpha < 1.0:
        raise ValueError(f"rate {rate} gives alpha {alpha} >= 1")
    return alpha


def _granularity_box(samples, domain):
    """The box whose μ(H) spatialize turns into the Mondrian lifetime: the box of the data, or the
    domain when the data's box has no extent (a single datum)."""
    s = np.asarray(samples, np.float32)
    box = list(zip(s.min(axis=0).tolist(), s.max(axis=0).tolist()))
    return box if sum(hi - lo for lo, hi in box) > 0 else domain


def alpha_from_intensity(intensity, domain, n_samples):
    r"""Absolute value of spatialize's Voronoi α from a Poisson–Voronoi intensity.

    Parameters
    ----------
    intensity : float
        Intensity :math:`\lambda_V` of the Voronoi generators, per unit volume.
    domain : sequence of (low, high)
        The box H.
    n_samples : int
        Number of data, n.

    Returns
    -------
    float
        :math:`|\alpha| = 2\lambda_V |H| / n`; the sign (nuclei uniform in the box or among the data)
        is set by the encoder profile.

    Notes
    -----

    spatialize draws :math:`N \sim \max(1, \mathrm{Poisson}(0.5\,n\,|\alpha|))` nuclei, at most
    :math:`n`. Matching the expected number of generators of a Poisson process of intensity
    :math:`\lambda_V` on the box :math:`H`, :math:`0.5\,n\,|\alpha| = \lambda_V |H|` with
    :math:`|H|` its volume, gives :math:`|\alpha| = 2 \lambda_V |H| / n`. The sign (nuclei uniform
    in the box or among the samples) is set by the profile.
    """
    vol = float(np.prod([hi - lo for lo, hi in domain]))
    return 2.0 * float(intensity) * vol / int(n_samples)


class SpatializeRunner:
    """:class:`~spatialize.scenarios.protocol.Runner` for spatialize's compiled estimators.

    Supports the encoder profiles in :data:`PROFILES` with the empty-cell policies ``"nan"``,
    ``"mark"`` and ``"coarsen"`` (set through the session, :mod:`spatialize.session`), for every
    decoder the catalogue offers in the domain's dimension. The partition box is pinned to the
    scenario's domain by adding its corners as extra queries. :meth:`cells` gives the cell of each
    query in the partitions :meth:`members` draws, and :meth:`loo` the leave-one-out ensemble.

    Examples
    --------
    >>> from spatialize import scenarios
    >>> from spatialize.scenarios.runners.spatialize import SpatializeRunner
    >>> report = scenarios.run(scenarios.catalog(ids=["S12-edge-cases"]), SpatializeRunner())
    >>> report.passed
    True
    """

    name = "spatialize"

    def __init__(self):
        from spatialize import gs, session
        self._gs = gs
        self._session = session

    def profile(self, encoder):
        """Canonical encoder profile name.

        Parameters
        ----------
        encoder : str
            Profile as written in a scenario.

        Returns
        -------
        str
        """
        return encoder

    def supports(self, est: EstimatorSpec) -> bool:
        """Whether the catalogue offers the estimator's partition and decoder in its dimension."""
        if est.encoder not in PROFILES or est.empty_cells not in POLICIES:
            return False
        if est.empty_cells == "mark" and set(est.mark) != {"source", "knn", "value"}:
            return False
        return self._gs.supports(PROFILES[est.encoder][0], est.decoder, len(est.domain))

    def members(self, est, samples, values, queries, *, n_members, seed):
        r"""Ensemble of the estimator at the queries (see :meth:`Runner.members
        <spatialize.scenarios.protocol.Runner.members>`).

        Raises
        ------
        NotImplementedError
            If the estimator is not supported.
        ValueError
            If a decoder parameter is missing or unknown, or the Voronoi intensity is too large for
            spatialize (it needs :math:`\lambda_V |H| < n/2`).
        """
        if not self.supports(est):
            raise NotImplementedError(f"{est.encoder}/{est.decoder} in {len(est.domain)}D")
        return self._run(est, samples, values, queries, n_members, seed, "estimate")[: len(queries)]

    def loo(self, est, samples, values, *, n_members, seed):
        """Leave-one-out ensemble: each datum predicted from the other data, one member per partition.

        Returns
        -------
        ndarray of shape (n, n_members)
        """
        if not self.supports(est):
            raise NotImplementedError(f"{est.encoder}/{est.decoder} in {len(est.domain)}D")
        return self._run(est, samples, values, samples, n_members, seed, "loo")

    def _run(self, est, samples, values, queries, n_members, seed, method):
        partition, alpha, q = self._partition(est, samples, queries)
        policy = {"empty_cells": est.empty_cells}
        if est.empty_cells == "mark":
            policy.update(mark_source=est.mark["source"], mark_knn=int(est.mark["knn"]),
                          mark_value=est.mark["value"])
        try:
            with self._session.override(**policy):
                _, out = self._gs.lib_spatialize_facade.run(samples, values, q, partition, est.decoder, est.params,
                                                            alpha, int(n_members), int(seed), method=method)
        except RuntimeError as e:
            raise ValueError(f"{est.id}: {e}") from None
        return np.asarray(out)

    def cells(self, est, samples, queries, *, n_members, seed):
        """The cell of each query in each partition :meth:`members` draws with the same arguments.

        Returns
        -------
        ndarray of int, shape (q, n_members)
            Two queries share a cell of partition t exactly when their labels in column t are equal.
        """
        if est.encoder not in PROFILES:
            raise NotImplementedError(est.encoder)
        partition, alpha, q = self._partition(est, samples, queries)
        labels = self._gs.lib_spatialize_facade.cells(samples, q, partition, alpha, int(n_members), int(seed))
        return np.asarray(labels)[: len(queries)]

    def _partition(self, est, samples, queries):
        """Spatialize's partition name and alpha for the estimator, and the queries with the corners
        of the domain appended."""
        # Spatialize draws the partition on bbox(samples ∪ queries): pin it to the declared domain
        corners = np.array(list(itertools.product(*est.domain)), np.float32)
        q = np.vstack([np.asarray(queries, np.float32), corners])
        partition, data_cond = PROFILES[est.encoder]
        if partition.startswith("mondrian"):
            # spatialize measures the box of the data, except for the partition of version 1.2
            box = est.domain if partition == "mondrian-legacy" else _granularity_box(samples, est.domain)
            alpha = alpha_from_rate(est.rate, box)
        else:
            alpha = alpha_from_intensity(est.rate, est.domain, len(samples))
            if not alpha < 1.0:  # spatialize accepts |alpha| < 1 only: expected nuclei < n/2
                raise ValueError(f"{est.id}: intensity {est.rate} needs |alpha| = {alpha:.3g} >= 1 with "
                                 f"{len(samples)} samples (spatialize's Voronoi needs λ_V·|H| < n/2)")
            if not data_cond:
                alpha = -alpha  # spatialize's convention: negative alpha = nuclei uniform in the box
        return partition, alpha, q

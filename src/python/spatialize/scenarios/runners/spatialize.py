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
    "mondrian": ("mondrian", True),        # Spatialize's Mondrian partition
    "mondrian-theory": ("mondrian-raw", True),  # the theory's Mondrian process (p_process="mondrian-raw")
    "voronoi": ("voronoi", False),         # Spatialize's Voronoi, nuclei uniform in the box (alpha < 0)
    "voronoi-data": ("voronoi", True),     # Spatialize's Voronoi, nuclei among the samples (alpha >= 0)
}

def alpha_from_rate(rate, domain):
    r"""Spatialize's Mondrian granularity α from a Mondrian rate λ and a box H.

    Parameters
    ----------
    rate : float
        Rate (lifetime) λ of the Mondrian process.
    domain : sequence of (low, high)
        The box H.

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

    Supports the encoder profiles in :data:`PROFILES` with ``empty_cells="nan"``, for every decoder
    the catalogue offers in the domain's dimension. The partition box is pinned to the scenario's
    domain by adding its corners as extra queries.

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
        from spatialize import gs
        self._gs = gs

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
        if est.encoder not in PROFILES or est.empty_cells != "nan":
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
        # Spatialize draws the partition on bbox(samples ∪ queries): pin it to the declared domain
        corners = np.array(list(itertools.product(*est.domain)), np.float32)
        q = np.vstack([np.asarray(queries, np.float32), corners])
        partition, data_cond = PROFILES[est.encoder]
        if partition.startswith("mondrian"):
            alpha = alpha_from_rate(est.rate, est.domain)
        else:
            alpha = alpha_from_intensity(est.rate, est.domain, len(samples))
            if not alpha < 1.0:  # spatialize accepts |alpha| < 1 only: expected nuclei < n/2
                raise ValueError(f"{est.id}: intensity {est.rate} needs |alpha| = {alpha:.3g} >= 1 with "
                                 f"{len(samples)} samples (spatialize's Voronoi needs λ_V·|H| < n/2)")
            if not data_cond:
                alpha = -alpha  # spatialize's convention: negative alpha = nuclei uniform in the box
        try:
            _, out = self._gs.lib_spatialize_facade.run(samples, values, q, partition, est.decoder, est.params,
                                                        alpha, int(n_members), int(seed))
        except RuntimeError as e:
            raise ValueError(f"{est.id}: {e}") from None
        return np.asarray(out)[: len(queries)]

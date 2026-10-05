"""spatialize's own Runner — the only module of the suite that imports spatialize's estimators.

The runner holds no list of decoders. It dispatches through spatialize's own facade — the
operator table ``lib_spatialize_facade.function_hash_map`` and the argument builder
``build_arg_list`` used by ``esi_griddata``/``esi_nongriddata`` — so it supports exactly the
encoder/decoder/dimension combinations spatialize supports. A decoder added to the facade becomes
testable by naming it in a scenario's ``estimators``; nothing here changes.

Decoder names are spatialize's ``local_interpolator`` names (``"idw"``, ``"kriging"``,
``"adaptiveidw"``, ...), and a scenario must declare every decoder parameter explicitly (they are
pre-registered, so no default is filled in here).
"""
import itertools

import numpy as np

from ..protocol import EstimatorSpec

#: Encoder profiles this runner implements: profile -> (spatialize's ``p_process``, ``data_cond``).
PROFILES = {
    "mondrian": ("mondrian", True),                      # alias of mondrian/spatialize-v1
    "mondrian/spatialize-v1": ("mondrian", True),
    "voronoi/spatialize-v1-uniform": ("voronoi", False),  # nuclei uniform in the box (alpha < 0)
    "voronoi/spatialize-v1-data": ("voronoi", True),      # nuclei drawn among the samples (alpha >= 0)
}

#: Facade arguments the runner sets itself; scenario ``params`` may not override them.
_RESERVED = {"alpha", "n_partitions", "seed", "p_process", "local_interpolator", "callback",
             "data_cond"}


def alpha_from_rate(rate, domain):
    """spatialize's granularity α from the book's Mondrian rate λ and the box H (ESI paper, eq. 8).

    λ = 1/(μ(H)(1 − α)), μ(H) = Σ side lengths  ⇒  α = 1 − 1/(λ μ(H)).
    """
    mu = float(sum(hi - lo for lo, hi in domain))
    alpha = 1.0 - 1.0 / (float(rate) * mu)
    if not alpha < 1.0:
        raise ValueError(f"rate {rate} gives alpha {alpha} >= 1")
    return alpha


def alpha_from_intensity(intensity, domain, n_samples):
    r"""Absolute value of spatialize's Voronoi α from the book's Poisson–Voronoi intensity (Def 2.3.2).

    spatialize draws :math:`N \sim \max(1, \mathrm{Poisson}(0.5\,n\,|\alpha|))` nuclei, at most
    :math:`n`. Matching the expected number of generators of a Poisson process of intensity
    :math:`\lambda_V` on the box :math:`H`, :math:`0.5\,n\,|\alpha| = \lambda_V |H|` with
    :math:`|H|` its volume, gives :math:`|\alpha| = 2 \lambda_V |H| / n`. The sign (nuclei uniform
    in the box or among the samples) is set by the profile.
    """
    vol = float(np.prod([hi - lo for lo, hi in domain]))
    return 2.0 * float(intensity) * vol / int(n_samples)


class SpatializeRunner:
    """:class:`~spatialize.scenarios.protocol.Runner` for spatialize's compiled estimators."""

    name = "spatialize"

    def __init__(self):
        from spatialize import SpatializeError
        from spatialize.gs import lib_spatialize_facade
        from spatialize.gs.esi._main import build_arg_list
        self._error, self._facade, self._build_arg_list = SpatializeError, lib_spatialize_facade, build_arg_list

    def profile(self, encoder):
        """Encoder profile name (``"mondrian"`` is an alias of ``"mondrian/spatialize-v1"``)."""
        return "mondrian/spatialize-v1" if encoder == "mondrian" else encoder

    def _operator(self, est):
        """The facade's operator for this estimator, or None when spatialize does not provide it."""
        if est.encoder not in PROFILES or est.empty_cells != "nan":
            return None
        probe = np.zeros((1, len(est.domain)), np.float32)
        try:
            return self._facade.get_operator(probe, est.decoder, "estimate", PROFILES[est.encoder][0])
        except self._error:
            return None

    def supports(self, est: EstimatorSpec) -> bool:
        return self._operator(est) is not None

    def members(self, est, samples, values, queries, *, n_members, seed):
        op = self._operator(est)
        if op is None:
            raise NotImplementedError(f"{est.encoder}/{est.decoder} in {len(est.domain)}D")
        clash = _RESERVED & set(est.params)
        if clash:
            raise ValueError(f"{est.id}: params may not set {sorted(clash)}")
        # spatialize-v1 draws the partition on bbox(samples ∪ queries): pin it to the declared domain
        corners = np.array(list(itertools.product(*est.domain)), np.float32)
        q = np.vstack([np.asarray(queries, np.float32), corners])
        p_process, data_cond = PROFILES[est.encoder]
        if p_process == "mondrian":
            alpha = alpha_from_rate(est.rate, est.domain)
        else:
            alpha = alpha_from_intensity(est.rate, est.domain, len(samples))
            if not alpha < 1.0:  # spatialize accepts |alpha| < 1 only: expected nuclei < n/2
                raise ValueError(f"{est.id}: intensity {est.rate} needs |alpha| = {alpha:.3g} >= 1 with "
                                 f"{len(samples)} samples (spatialize's Voronoi needs λ_V·|H| < n/2)")
        args = dict(est.params, alpha=alpha, n_partitions=int(n_members), seed=int(seed),
                    p_process=p_process, local_interpolator=est.decoder, data_cond=data_cond,
                    callback=None)
        try:
            arg_list = self._build_arg_list(samples, values, q, args)
        except KeyError as e:
            raise ValueError(f"{est.id}: decoder '{est.decoder}' needs parameter {e} in the scenario") from None
        _, out = op(*arg_list)
        return np.asarray(out)[: len(queries)]

"""Spatialize's own runner — the only module of the suite that imports spatialize's estimators.

The runner holds no list of decoders. It first dispatches through spatialize's public facade (the
operator table ``lib_spatialize_facade.function_hash_map`` and the argument builder
``build_arg_list`` used by ``esi_griddata`` / ``esi_nongriddata``), so everything the public API
offers is tested the way users run it. Combinations the public API does not offer yet but the
compiled engine does (e.g. kriging on Voronoi partitions) go through ``libspatialize.run``; the
report marks them ``[run]``.

Decoder names are spatialize's ``local_interpolator`` names (``"idw"``, ``"kriging"``,
``"adaptiveidw"``, ...). A scenario must declare every decoder parameter explicitly, since the
parameters are pre-registered and the runner fills in no default. Rates are translated into spatialize's ``alpha``
by :func:`alpha_from_rate` (Mondrian) and :func:`alpha_from_intensity` (Voronoi).
"""
import itertools

import numpy as np

from ..protocol import EstimatorSpec

#: Encoder profiles this runner implements: profile -> (spatialize's ``p_process``, ``data_cond``).
PROFILES = {
    "mondrian": ("mondrian", True),        # Spatialize's Mondrian partition
    "voronoi": ("voronoi", False),         # Spatialize's Voronoi, nuclei uniform in the box (alpha < 0)
    "voronoi-data": ("voronoi", True),     # Spatialize's Voronoi, nuclei among the samples (alpha >= 0)
}

#: Decoders of the compiled engine's generic entry point ``libspatialize.run``.
_RUN_DECODERS = {"idw", "kriging", "adaptiveidw"}

#: Facade arguments the runner sets itself; scenario ``params`` may not override them.
_RESERVED = {"alpha", "n_partitions", "seed", "p_process", "local_interpolator", "callback",
             "data_cond"}


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

    Supports the encoder profiles in :data:`PROFILES` with ``empty_cells="nan"``, and every decoder
    the public facade or ``libspatialize.run`` offers for the domain's dimension. The partition box
    is pinned to the scenario's domain by adding its corners as extra queries.

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
        from spatialize import SpatializeError
        from spatialize.gs import lib_spatialize_facade
        from spatialize.gs.esi._main import build_arg_list
        import libspatialize
        self._error, self._facade, self._build_arg_list = SpatializeError, lib_spatialize_facade, build_arg_list
        self._lib = libspatialize

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

    def _operator(self, est):
        """The facade's operator for this estimator, or None when the facade does not provide it."""
        if est.encoder not in PROFILES or est.empty_cells != "nan":
            return None
        probe = np.zeros((1, len(est.domain)), np.float32)
        try:
            return self._facade.get_operator(probe, est.decoder, "estimate", PROFILES[est.encoder][0])
        except self._error:
            return None

    def _engine_supports(self, est):
        """Whether the compiled engine's generic entry point ``libspatialize.run`` provides it."""
        if est.encoder not in PROFILES or est.empty_cells != "nan" or not hasattr(self._lib, "run"):
            return False
        if est.decoder not in _RUN_DECODERS:
            return False
        return est.decoder != "adaptiveidw" or len(est.domain) in (2, 3)

    def route(self, est):
        """Path by which the runner reaches the estimator.

        Parameters
        ----------
        est : EstimatorSpec

        Returns
        -------
        {"facade", "run", None}
            ``"facade"``: the public API's dispatch (preferred); ``"run"``: ``libspatialize.run``,
            for combinations the public API does not offer yet; None: not supported.
        """
        if self._operator(est) is not None:
            return "facade"
        if self._engine_supports(est):
            return "run"
        return None

    def supports(self, est: EstimatorSpec) -> bool:
        """Whether :meth:`route` finds a way to run the estimator (see the protocol)."""
        return self.route(est) is not None

    def members(self, est, samples, values, queries, *, n_members, seed):
        r"""Ensemble of the estimator at the queries (see :meth:`Runner.members
        <spatialize.scenarios.protocol.Runner.members>`).

        Raises
        ------
        NotImplementedError
            If the estimator is not supported.
        ValueError
            If a decoder parameter is missing or reserved, or the Voronoi intensity is too large
            for spatialize (it needs :math:`\lambda_V |H| < n/2`).
        """
        route = self.route(est)
        if route is None:
            raise NotImplementedError(f"{est.encoder}/{est.decoder} in {len(est.domain)}D")
        clash = _RESERVED & set(est.params)
        if clash:
            raise ValueError(f"{est.id}: params may not set {sorted(clash)}")
        # Spatialize draws the partition on bbox(samples ∪ queries): pin it to the declared domain
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
        if route == "facade":
            out = self._members_facade(est, samples, values, q, alpha, p_process, data_cond, n_members, seed)
        else:
            out = self._members_run(est, samples, values, q, alpha, p_process, data_cond, n_members, seed)
        return np.asarray(out)[: len(queries)]

    def _members_facade(self, est, samples, values, q, alpha, p_process, data_cond, n_members, seed):
        args = dict(est.params, alpha=alpha, n_partitions=int(n_members), seed=int(seed),
                    p_process=p_process, local_interpolator=est.decoder, data_cond=data_cond,
                    callback=None)
        try:
            arg_list = self._build_arg_list(samples, values, q, args)
        except KeyError as e:
            raise ValueError(f"{est.id}: decoder '{est.decoder}' needs parameter {e} in the scenario") from None
        _, out = self._operator(est)(*arg_list)
        return out

    def _members_run(self, est, samples, values, q, alpha, p_process, data_cond, n_members, seed):
        params = dict(est.params)
        if est.decoder == "kriging" and isinstance(params.get("model"), str):
            params["model"] = self._facade.get_kriging_model_number(params["model"])
        if p_process == "voronoi" and not data_cond:
            alpha = -alpha  # spatialize's convention: negative alpha = nuclei uniform in the box
        try:
            _, out = self._lib.run(np.asarray(samples, np.float32), np.asarray(values, np.float32), q,
                                   p_process, float(alpha), int(n_members), int(seed), est.decoder, params)
        except RuntimeError as e:
            raise ValueError(f"{est.id}: {e}") from None
        return out

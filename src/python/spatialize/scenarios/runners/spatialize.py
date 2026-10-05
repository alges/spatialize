"""spatialize's own Runner — the only module of the suite that imports spatialize's estimators.

Calls the C++ extension directly (not the high-level facade) so the conformance target is the
compiled estimator and its contract stays stable across facade refactors.
"""
import itertools

import numpy as np

from ..protocol import EstimatorSpec

PROFILES = {"mondrian": "mondrian/spatialize-v1", "mondrian/spatialize-v1": "mondrian/spatialize-v1"}
DECODERS = {"idw", "kriging", "adaptive_idw"}
KRIGING_MODELS = {"spherical": 1, "exponential": 2, "cubic": 3, "gaussian": 4}


def alpha_from_rate(rate, domain):
    """spatialize's granularity α from the book's Mondrian rate λ and the box H (ESI paper, eq. 8):
    λ = 1/(μ(H)(1 − α)), μ(H) = Σ side lengths  ⇒  α = 1 − 1/(λ μ(H))."""
    mu = float(sum(hi - lo for lo, hi in domain))
    alpha = 1.0 - 1.0 / (float(rate) * mu)
    if not alpha < 1.0:
        raise ValueError(f"rate {rate} gives alpha {alpha} >= 1")
    return alpha


class SpatializeRunner:
    name = "spatialize"

    def __init__(self, lib=None):
        if lib is None:
            import libspatialize as lib
        self.lib = lib

    def profile(self, encoder):
        return PROFILES.get(encoder, encoder)

    def supports(self, est: EstimatorSpec) -> bool:
        if est.encoder not in PROFILES or est.empty_cells != "nan" or est.decoder not in DECODERS:
            return False
        dim = len(est.domain)
        return dim in (2, 3) if est.decoder in ("kriging", "adaptive_idw") else dim >= 1

    def members(self, est, samples, values, queries, *, n_members, seed):
        if not self.supports(est):
            raise NotImplementedError(f"{est.encoder}/{est.decoder}")
        dim = len(est.domain)
        # spatialize-v1 draws the partition on bbox(samples ∪ queries): pin it to the declared domain
        corners = np.array(list(itertools.product(*est.domain)), np.float32)
        q = np.vstack([np.asarray(queries, np.float32), corners])
        s, v = np.asarray(samples, np.float32), np.asarray(values, np.float32)
        a = alpha_from_rate(est.rate, est.domain)
        T, p, lib = int(n_members), est.params, self.lib
        if est.decoder == "idw":
            _, out = lib.estimation_esi_idw(s, v, T, a, float(p.get("exponent", 2.0)), int(seed), q, None)
        elif est.decoder == "kriging":
            model = KRIGING_MODELS[p.get("model", "exponential")]
            fn = getattr(lib, f"estimation_esi_kriging_{dim}d")
            _, out = fn(s, v, T, a, model, float(p.get("nugget", 0.0)), float(p.get("range", 0.3)),
                        float(p.get("sill", 1.0)), int(seed), q, None)
        else:  # adaptive_idw; parallelize=False until the OpenMP/GIL crash (bug #1) is fixed
            fn = getattr(lib, f"estimation_adaptive_esi_idw_{dim}d")
            _, out = fn(s, v, T, a, int(seed), p.get("metric", "mae"), False, q, None)
        return np.asarray(out)[: len(queries)]

"""The contract between the scenario suite and an implementation under test.

An implementation provides a :class:`Runner`. The suite asks it for ensembles (``members``) and
computes everything else itself — readings of the law, metrics, statistical tests and the error
budget — identically for every implementation.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class EstimatorSpec:
    """Implementation-independent description of an ensemble estimator.

    Parameters
    ----------
    id : str
        Name of the estimator inside its scenario (e.g. ``"idw"``).
    encoder : str
        Partition profile, e.g. ``"mondrian/spatialize-v1"`` or ``"voronoi/spatialize-v1-uniform"``
        (see the documentation, *Encoder profiles*).
    rate : float or None
        The encoder's rate in the book's terms: the Mondrian rate/budget λ, or the intensity λ_V
        of the Poisson–Voronoi generators, per unit volume. Runners derive
        their own parameters from it and from ``domain`` — for spatialize's Mondrian,
        ``alpha = 1 - 1/(λ·μ(H))``; for its Voronoi, ``|alpha| = 2·λ_V·|H|/n``.
    domain : sequence of (low, high)
        The box H the partition is drawn on.
    decoder : str
        Local interpolator, named as spatialize's ``local_interpolator`` (``"idw"``,
        ``"kriging"``, ``"adaptiveidw"``, ...); other implementations map these names to theirs.
    params : dict
        Decoder parameters, all of them explicit (e.g. ``{"exponent": 2.0}``); they are
        pre-registered in the scenario, so runners do not fill in defaults.
    empty_cells : str
        Empty-cell policy; ``"nan"`` (a data-free cell yields NaN) is the only one every runner
        must support.
    """
    id: str
    encoder: str
    rate: Any
    domain: Sequence[Sequence[float]]
    decoder: str
    params: Dict[str, Any] = field(default_factory=dict)
    empty_cells: str = "nan"


@runtime_checkable
class Runner(Protocol):
    """What an implementation must provide to be tested by the suite."""

    name: str

    def supports(self, estimator: EstimatorSpec) -> bool:
        """Whether this implementation provides the estimator (unsupported checks are skipped)."""
        ...

    def members(self, estimator: EstimatorSpec, samples: np.ndarray, values: np.ndarray,
                queries: np.ndarray, *, n_members: int, seed: int) -> np.ndarray:
        """Ensemble of the estimator at the queries: array of shape ``(n_queries, n_members)``.

        Members must be independent given the data (one partition draw each). ``seed`` makes a
        run reproducible for this implementation; no criterion depends on its value.
        """
        ...

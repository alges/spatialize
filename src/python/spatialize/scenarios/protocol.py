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
        Partition profile, e.g. ``"mondrian/spatialize-v1"`` or ``"voronoi/spatialize-v1"``
        (see the documentation, *Encoder profiles*).
    rate : float or None
        The book's Mondrian rate/budget λ (Def 2.3.1). Runners derive their own parameters from it
        and from ``domain`` — for spatialize, ``alpha = 1 - 1/(λ·μ(H))``.
    domain : sequence of (low, high)
        The box H the partition is drawn on.
    decoder : str
        Local interpolator: ``"cell_mean"``, ``"idw"``, ``"adaptive_idw"``, ``"kriging"``, ...
    params : dict
        Decoder parameters (e.g. ``{"exponent": 2.0}``).
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

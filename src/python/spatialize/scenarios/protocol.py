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
        Partition profile, e.g. ``"mondrian"``, ``"voronoi"`` or ``"voronoi-data"``
        (see the documentation, *Encoder profiles*).
    rate : float or None
        The encoder's rate in the book's terms: the Mondrian rate/budget λ, or the intensity λ_V
        of the Poisson–Voronoi generators, per unit volume. Runners derive
        their own parameters from it and from ``domain`` — for spatialize's Mondrian,
        ``alpha = 1 - 1/(λ·μ(D))`` with D the box of the data; for its Voronoi,
        ``|alpha| = 2·λ_V·|H|/n``.
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
        must support. Spatialize's runner also supports ``"mark"`` (the cell takes one value, drawn
        from a cell with data) and ``"coarsen"`` (the decoder predicts from a coarser cell with data).
    mark : dict
        How ``"mark"`` draws the value, every key explicit: ``source`` (``"local"``, among the
        ``knn`` cells with data nearest to the empty cell; ``"cells"``, among every cell with data;
        ``"data"``, one datum among all the data), ``knn`` and ``value`` (``"decoder"``, the
        prediction of the drawn cell's decoder at the empty cell; ``"datum"``, one of the drawn
        cell's data). Empty for the other policies.
    """
    id: str
    encoder: str
    rate: Any
    domain: Sequence[Sequence[float]]
    decoder: str
    params: Dict[str, Any] = field(default_factory=dict)
    empty_cells: str = "nan"
    mark: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Runner(Protocol):
    """Interface an implementation provides to be tested by the suite.

    The suite only asks a runner for ensembles; it computes every reading of the law, functional,
    test and decision itself, identically for every implementation. Any object with these members
    is a runner (structural typing); spatialize's is
    :class:`~spatialize.scenarios.runners.spatialize.SpatializeRunner`.

    Attributes
    ----------
    name : str
        Name of the implementation, shown in the report.

    Notes
    -----
    A runner may also provide ``cells(estimator, samples, queries, *, n_members, seed)``, returning
    an int array of shape ``(q, n_members)``: the label of each query's cell in each partition that
    ``members`` draws with the same arguments, two queries sharing a cell of partition ``t`` exactly
    when their labels in column ``t`` are equal. The checks that read cells (scenario P10) are
    skipped for a runner without it.

    It may also provide ``loo(estimator, samples, values, *, n_members, seed)``, returning an array of
    shape ``(n, n_members)``: each datum predicted from the other data, one member per partition, as
    the hyperparameter searches cross-validate. Scenario P11 needs it.

    It may also provide ``law_cdf(estimator, samples, values, queries, thresholds, *, reading,
    n_members, seed)``, returning an array of shape ``(q, k)``: the cumulative distribution function
    that the implementation's reading ``reading`` of the law gives at each query and threshold.
    Scenario P4 needs it and names the readings to test.
    """

    name: str

    def supports(self, estimator: EstimatorSpec) -> bool:
        """Whether this implementation provides the estimator.

        Parameters
        ----------
        estimator : EstimatorSpec
            Implementation-free description of the estimator.

        Returns
        -------
        bool
            ``False`` makes the checks of that estimator be reported as skipped.
        """
        ...

    def members(self, estimator: EstimatorSpec, samples: np.ndarray, values: np.ndarray,
                queries: np.ndarray, *, n_members: int, seed: int) -> np.ndarray:
        """Ensemble of the estimator at the queries.

        Parameters
        ----------
        estimator : EstimatorSpec
            The estimator (partition profile, rate, domain, decoder and its parameters).
        samples : ndarray of shape (n, d)
            Data locations.
        values : ndarray of shape (n,)
            Data values.
        queries : ndarray of shape (q, d)
            Prediction locations.
        n_members : int
            Number of ensemble members (partition draws).
        seed : int
            Seed making the run reproducible for this implementation; no criterion depends on its
            value.

        Returns
        -------
        ndarray of shape (q, n_members)
            One column per member. Members must be independent given the data (one partition draw
            each); with ``empty_cells="nan"`` a query in a cell without data is NaN.
        """
        ...

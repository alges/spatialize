"""Geostatistical test scenarios: an implementation-independent conformance suite.

Scenarios are geostatistical situations with a known truth, taken from the theory of ensemble
estimation over random partitions (Egaña, Díaz, Navarro, Ehrenfeld, *A General Theory of
Higher-Order Geostatistics*). Every acceptance criterion is a statistical test with a declared
level and power, or an almost-sure property; no realisation is compared draw for draw, so any
correct implementation of the same concepts — not only spatialize — can be tested by plugging in a
:class:`~spatialize.scenarios.protocol.Runner`.

Package layout
--------------
- :func:`catalog`, :func:`run`, :class:`Scenario`, :class:`Report` — load scenarios, run them under
  one error budget, read the outcomes (module ``spatialize.scenarios.engine``, which also holds
  the evaluators that turn a scenario file into checks).
- :mod:`spatialize.scenarios.protocol` — what an implementation must provide
  (:class:`~spatialize.scenarios.protocol.Runner`, :class:`~spatialize.scenarios.protocol.EstimatorSpec`).
- ``spatialize.scenarios.stats`` — test families (:mod:`~spatialize.scenarios.stats.families`),
  error budget (:mod:`~spatialize.scenarios.stats.budget`) and map functionals for the visual
  criteria (:mod:`~spatialize.scenarios.stats.maps`).
- ``spatialize.scenarios.generators`` — truth fields, reference estimators and the
  materialisation of pinned data.
- :mod:`spatialize.scenarios.runners.spatialize` — spatialize's own runner.
- :mod:`spatialize.scenarios.figures` — maps saved for human review.
- ``catalog/<scenario>/`` — one directory per scenario: ``scenario.yaml`` and its pinned data.

Examples
--------
From the command line (exit status 0 when every outcome is as expected, 1 otherwise)::

    python -m spatialize.scenarios                  # whole catalogue, fast mode
    python -m spatialize.scenarios --mode full      # release mode
    python -m spatialize.scenarios --list           # scenarios and their checks

From Python:

>>> from spatialize import scenarios
>>> from spatialize.scenarios.runners.spatialize import SpatializeRunner
>>> report = scenarios.run(scenarios.catalog(), SpatializeRunner(), mode="ci")
>>> print(report.table())

Importing this package does not load spatialize's C++ extension. PyYAML is required
(``pip install 'spatialize[scenarios]'``).
"""
#: Version of the suite, independent of spatialize's version.
VERSION = "0.1.0"

from .protocol import EstimatorSpec, Runner  # noqa: E402,F401
from .engine import Report, Scenario, catalog, run  # noqa: E402,F401

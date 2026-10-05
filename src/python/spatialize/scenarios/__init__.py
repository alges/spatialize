"""Geostatistical test scenarios: an implementation-independent conformance suite.

Scenarios are geostatistical situations with a known truth, taken from the theory of ensemble
estimation over random partitions (Egaña, Díaz, Navarro, Ehrenfeld, *A General Theory of
Higher-Order Geostatistics*). Every acceptance criterion is a statistical test with a declared
level and power; no realisation is compared draw for draw, so any correct implementation of the
same concepts — not only spatialize — can be tested by plugging in a :class:`~spatialize.scenarios.protocol.Runner`.

Example
-------
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

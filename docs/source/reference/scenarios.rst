spatialize.scenarios
====================

Conformance suite of geostatistical scenarios; see :doc:`../scenarios/index`.

.. automodule:: spatialize.scenarios

.. autofunction:: spatialize.scenarios.catalog

.. autofunction:: spatialize.scenarios.run

.. autoclass:: spatialize.scenarios.Scenario
   :members: verify_checksums, field, estimator

.. autoclass:: spatialize.scenarios.Report
   :members: passed, table

Protocol
--------

.. automodule:: spatialize.scenarios.protocol
   :members:

Statistical tests and error budget
----------------------------------

.. automodule:: spatialize.scenarios.stats.families
   :members:

.. automodule:: spatialize.scenarios.stats.budget
   :members:

Map functionals
---------------

.. automodule:: spatialize.scenarios.stats.maps
   :members:

Field generators
----------------

.. automodule:: spatialize.scenarios.generators.fields
   :members:

.. automodule:: spatialize.scenarios.generators.materialise
   :members:

Spatialize runner
-----------------

.. automodule:: spatialize.scenarios.runners.spatialize
   :members:

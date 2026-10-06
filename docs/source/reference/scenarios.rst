spatialize.scenarios
====================

API of the conformance suite of geostatistical scenarios. For what the suite tests and why, start
with :doc:`../scenarios/index`; for the format of a scenario, see
:doc:`../scenarios/file_format`.

.. automodule:: spatialize.scenarios

Running scenarios
-----------------

Load the catalogue, run it with a runner under one error budget, and read the outcomes.

.. autofunction:: spatialize.scenarios.catalog

.. autofunction:: spatialize.scenarios.run

.. autoclass:: spatialize.scenarios.Report
   :members: passed, table

.. autoclass:: spatialize.scenarios.engine.CheckOutcome
   :members: status, ok

.. autoclass:: spatialize.scenarios.Scenario
   :members: tier, file, verify_checksums, field, estimator

.. autofunction:: spatialize.scenarios.engine.check_ids

.. autofunction:: spatialize.scenarios.__main__.main

Evaluators
----------

An evaluator turns a scenario file into checks: it asks the runner for ensembles, computes the
readings of the law or the map functionals, and returns undecided outcomes;
:func:`~spatialize.scenarios.run` then
applies the error budget. A scenario names its evaluator in its ``evaluator`` key.

.. autofunction:: spatialize.scenarios.engine.eval_pair_cooccurrence

.. autofunction:: spatialize.scenarios.engine.eval_map_visual

.. autofunction:: spatialize.scenarios.engine.eval_edge_cases

Testing an implementation
-------------------------

.. automodule:: spatialize.scenarios.protocol
   :members:

Spatialize's runner
-------------------

.. automodule:: spatialize.scenarios.runners.spatialize
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

Truth fields and pinned data
----------------------------

.. automodule:: spatialize.scenarios.generators.fields
   :members:

.. automodule:: spatialize.scenarios.generators.materialise
   :members: materialise

Figures for review
------------------

.. automodule:: spatialize.scenarios.figures
   :members:

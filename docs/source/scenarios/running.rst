.. _scenarios-running:

#######################################
Running the suite and adding scenarios
#######################################

.. currentmodule:: spatialize.scenarios

Installation
============

The suite ships with Spatialize. Reading scenario descriptors needs PyYAML, installed by the
``scenarios`` extra:

.. code-block:: bash

   pip install "spatialize[scenarios]"

Running it on Spatialize
========================

From Python:

.. code-block:: python

   from spatialize import scenarios
   from spatialize.scenarios.runners.spatialize import SpatializeRunner

   report = scenarios.run(scenarios.catalog(), SpatializeRunner(), mode="ci")
   print(report.table())
   assert report.passed

:func:`catalog` selects scenarios by tier or id (``catalog(tier="T1")``,
``catalog(ids=["S03-anisotropic-field"])``). :func:`run` applies one Holm budget to all the
checks of the call, so the scenarios you run together share the error budget.

From the repository, as part of the test suite:

.. code-block:: bash

   python -m pytest -q tests/scenarios
   SPATIALIZE_SCENARIO_MODE=full python -m pytest -q tests/scenarios       # release mode
   SPATIALIZE_SCENARIO_SEED=12345 python -m pytest -q tests/scenarios      # reproduce a run

Each check appears as one test; a failure message carries the seed, the mode, the p-value, the
level and the details needed to reproduce it.

Running it on another implementation
====================================

An implementation is tested by providing a :class:`~spatialize.scenarios.protocol.Runner`: an object with a ``name``, a
``supports(estimator)`` method and a ``members(...)`` method returning the ensemble at the queries
as an array of shape ``(n_queries, n_members)``. Estimators are described in implementation-free
terms by an :class:`~spatialize.scenarios.protocol.EstimatorSpec` — encoder profile, the rate :math:`\lambda` and the domain,
decoder, decoder parameters and empty-cell policy — and each runner maps them to its own
parameters (for Spatialize, ``alpha`` from :math:`\lambda` and the domain; see :doc:`encoders`).

.. code-block:: python

   import numpy as np
   from spatialize.scenarios import EstimatorSpec

   class MyRunner:
       name = "mine"

       def supports(self, est: EstimatorSpec) -> bool:
           return est.decoder in {"idw"} and est.empty_cells == "nan"

       def members(self, est, samples, values, queries, *, n_members, seed):
           out = np.empty((len(queries), n_members))
           ...  # one independent partition draw per column
           return out

Requirements: members must be independent given the data (one partition draw each); empty cells
must give NaN under the ``"nan"`` policy; checks whose estimator is not supported are reported as
skipped. Everything else — readings of the law, functionals, tests and the error budget — is
computed by the suite, identically for every implementation.

Adding a scenario
=================

1. Create ``catalog/<ID>-<short-name>/scenario.yaml`` with ``id``, ``version``, ``tier``,
   ``evaluator``, ``book`` (the source in the theory), ``purpose``, ``domain``, the data or truth
   generator, ``estimators`` and ``checks``. Follow the two existing scenarios.
2. Pre-register every check: its family, its functional, its margin or bound, the minimum
   detectable effect :math:`\delta` and the sample sizes per mode, chosen with
   :func:`stats.budget.required_n` (or :func:`stats.budget.min_sign_test_fields` for sign tests) so
   that power is at least 0.9 under the budget of the whole catalogue.
3. Add a negative control when the family could pass vacuously (``expect: reject`` for the profile
   that must fail).
4. For pinned data, write the fields with
   :mod:`spatialize.scenarios.generators.materialise`, which stores ``data/*.npy`` and the
   ``CHECKSUMS.sha256`` manifest. Never edit a data file by hand: checksums are verified when the
   catalogue is loaded.
5. Thresholds are fixed **before** the first reference run and are not tuned afterwards; changing
   one is a new version of the scenario.
6. Run the suite in both modes and with several seeds, and record the calibration in the
   descriptor's ``provenance``.

Not part of the suite: the refactor guard
=========================================

The repository also has ``tests/refactor_guard``, a bitwise snapshot of the compiled library's
outputs on fixed inputs. It detects *any* change in the numbers during internal refactors and is
deliberately the opposite of this suite: it compares realisations, is specific to one platform and
build, and is not an acceptance criterion. It is not shipped with the package.

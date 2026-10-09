.. _scenarios-extending:

####################
Extending the suite
####################

.. currentmodule:: spatialize.scenarios

Testing another implementation
==============================

An implementation takes part through a :class:`~spatialize.scenarios.protocol.Runner`, an object with
a ``name``, a ``supports(estimator)`` method and a ``members(...)`` method returning the ensemble at
the queries as an array of shape ``(n_queries, n_members)``. An
:class:`~spatialize.scenarios.protocol.EstimatorSpec` describes each estimator in implementation-free
terms, through its encoder profile, the rate :math:`\lambda` with the domain, the decoder with its
parameters and the empty-cell policy. Each runner maps these to its own parameters, as Spatialize's
runner derives ``alpha`` from :math:`\lambda` and the domain (:doc:`encoders`).

.. code-block:: python

   import numpy as np
   from spatialize import scenarios
   from spatialize.scenarios import EstimatorSpec

   class MyRunner:
       name = "mine"

       def supports(self, est: EstimatorSpec) -> bool:
           return est.decoder in {"idw"} and est.empty_cells == "nan"

       def members(self, est, samples, values, queries, *, n_members, seed):
           out = np.empty((len(queries), n_members))
           ...  # one independent partition draw per column
           return out

   report = scenarios.run(scenarios.catalog(), MyRunner(), mode="ci")
   print(report.table())

A runner must meet three requirements.

- Members are independent given the data, with one partition draw each.
- Empty cells give NaN under the ``"nan"`` policy.
- Estimators it does not support are declared through ``supports``, so their checks are reported as
  skipped.

The suite computes everything else, from the readings of the law and the functionals to the tests
and the error budget, identically for every implementation.

A runner may also give the cells of its partitions, through a method
``cells(est, samples, queries, *, n_members, seed)`` returning an integer array of shape
``(n_queries, n_members)``. Two queries share a cell of partition :math:`t` exactly when their
labels in column :math:`t` are equal, for the partitions ``members`` draws with the same arguments.
The checks of scenario P10 that compare the locations of one cell need it. A runner without it has
them skipped. Spatialize's runner reads the cells with ``lib_spatialize_facade.cells``.

A runner may also give leave-one-out ensembles, through a method
``loo(est, samples, values, *, n_members, seed)`` returning an array of shape
``(n_samples, n_members)``, each datum predicted from the other data. Scenarios P11 and P12 need it, P12 together with ``cells``.

Testing a new spatialize decoder
================================

Spatialize's own runner (:class:`~spatialize.scenarios.runners.spatialize.SpatializeRunner`)
holds no list of decoders or partitions. It asks the facade between Python and C++ which
combinations the compiled extension's catalogue offers in a dimension
(:doc:`../development/architecture`), then computes the members through that same facade, so every
partition and decoder is tested the way users run it. A decoder registered in the catalogue is
available to the scenarios at once. A scenario tests it by naming it in its ``estimators`` with its
catalogue name and every parameter explicit, as below.

.. code-block:: yaml

   estimators:
     - {id: aidw, encoder: mondrian, rate: 5.0, decoder: adaptiveidw, params: {metric: mae}, empty_cells: nan}
     - {id: ok, encoder: mondrian, rate: 5.0, decoder: kriging,
        params: {model: exponential, nugget: 0.0, range: 0.3, sill: 1.0}, empty_cells: nan}

Decoder parameters are pre-registered, so the runner fills in no defaults. A missing parameter raises
an error naming it, while the parameters the runner sets itself (``alpha``, ``n_partitions``,
``seed``, ...) cannot be overridden. A combination that neither path provides (e.g. adaptive IDW in
4D) is reported as skipped.

Adding a scenario
=================

1. Create ``catalog/<ID>-<short-name>/scenario.yaml`` with ``id``, ``version``, ``tier``,
   ``evaluator``, ``book``, ``purpose``, ``domain``, the data or truth generator, ``estimators`` and
   ``checks`` (:doc:`file_format` describes every key). The field ``book`` holds the source in the
   theory, a ``topic`` by content with the draft's numbers under a dated key such as
   ``draft_2026_09``. The existing scenarios serve as models.
2. Pre-register every check with its family, its functional, its margin or bound, the minimum
   detectable effect :math:`\delta` and the sample sizes per mode. Choose the sizes with
   :func:`stats.budget.required_n` (or :func:`stats.budget.min_sign_test_fields` for sign tests) so
   that power reaches 0.9 under the budget of the whole catalogue.
3. Add a negative control (``expect: reject`` for the profile that must fail) when the family could
   pass vacuously. A check whose violation is impossible for a correct implementation (no infinite
   values, outputs within the data range, ...) belongs to the ``almost-sure`` family, decided exactly
   without spending error budget. A check listing several ``estimators`` produces one outcome per
   estimator, named ``<check>-<estimator>``.
4. For pinned data, write the fields with :mod:`spatialize.scenarios.generators.materialise`, which
   stores ``data/*.npy`` with the ``CHECKSUMS.sha256`` manifest. Never edit a data file by hand, since
   the checksums are verified whenever the catalogue is loaded.
5. Fix the thresholds before the first reference run, without tuning them afterwards. Changing one
   makes a new version of the scenario. A pre-registered check that fails on Spatialize documents a
   finding, so either fix the code or record the finding with ``known_failure: <reason>`` (reported
   as ``KNOWN``, :doc:`statistics`), never relaxing the threshold to make it pass.
6. Run the suite (:doc:`running`) in both modes with several seeds, recording the calibration in the
   descriptor's ``provenance``.

The refactor guard, outside the suite
=====================================

The repository also holds ``tests/refactor_guard``, a bitwise snapshot of the compiled library's
outputs on fixed inputs. It detects any change in the numbers during internal refactors. It works
the opposite way to this suite, comparing realisations on one platform and one build, so it serves
as no acceptance criterion and does not ship with the package.

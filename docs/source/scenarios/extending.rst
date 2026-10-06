.. _scenarios-extending:

#########################################################
Testing another implementation and adding scenarios
#########################################################

.. currentmodule:: spatialize.scenarios

Testing another implementation
==============================

An implementation is tested by providing a :class:`~spatialize.scenarios.protocol.Runner`: an
object with a ``name``, a ``supports(estimator)`` method and a ``members(...)`` method returning
the ensemble at the queries as an array of shape ``(n_queries, n_members)``. Estimators are
described in implementation-free terms by an
:class:`~spatialize.scenarios.protocol.EstimatorSpec` — encoder profile, the rate :math:`\lambda`
and the domain, decoder, decoder parameters and empty-cell policy — and each runner maps them to
its own parameters (for Spatialize, ``alpha`` from :math:`\lambda` and the domain; see
:doc:`encoders`).

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

Requirements: members must be independent given the data (one partition draw each); empty cells
must give NaN under the ``"nan"`` policy; checks whose estimator is not supported are reported as
skipped. Everything else — readings of the law, functionals, tests and the error budget — is
computed by the suite, identically for every implementation.

Testing a new spatialize decoder
================================

Spatialize's own runner (:class:`~spatialize.scenarios.runners.spatialize.SpatializeRunner`)
holds no list of decoders: it dispatches through the same operator table and argument builder as
:func:`~spatialize.gs.esi.esi_griddata`, so it supports exactly the encoder, decoder and dimension
combinations spatialize supports. Once a new decoder is available in spatialize, a scenario tests
it by naming it in its ``estimators``, with spatialize's ``local_interpolator`` name and every
parameter explicit:

.. code-block:: yaml

   estimators:
     - {id: aidw, encoder: mondrian, rate: 5.0, decoder: adaptiveidw, params: {metric: mae}, empty_cells: nan}
     - {id: ok, encoder: mondrian, rate: 5.0, decoder: kriging,
        params: {model: exponential, nugget: 0.0, range: 0.3, sill: 1.0}, empty_cells: nan}

Decoder parameters are pre-registered, so the runner fills in no defaults: a missing parameter is
an error naming it, and parameters the runner sets itself (``alpha``, ``n_partitions``,
``seed``, ...) cannot be overridden. A combination spatialize does not provide (e.g. kriging in
4D) is reported as skipped.

Adding a scenario
=================

1. Create ``catalog/<ID>-<short-name>/scenario.yaml`` with ``id``, ``version``, ``tier``,
   ``evaluator``, ``book`` (the source in the theory: a ``topic`` by content and the draft's
   numbers under a dated key such as ``draft_2026_09``), ``purpose``, ``domain``, the data or truth
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
6. Run the suite (:doc:`running`) in both modes and with several seeds, and record the calibration
   in the descriptor's ``provenance``.

Not part of the suite: the refactor guard
=========================================

The repository also has ``tests/refactor_guard``, a bitwise snapshot of the compiled library's
outputs on fixed inputs. It detects *any* change in the numbers during internal refactors and is
deliberately the opposite of this suite: it compares realisations, is specific to one platform and
build, and is not an acceptance criterion. It is not shipped with the package.

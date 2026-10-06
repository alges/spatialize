.. _scenarios-running:

##################
Running the tests
##################

.. currentmodule:: spatialize.scenarios

There are three equivalent ways to run the scenarios — the command line, Python, and pytest from
a source checkout. All three run the same checks under the same error budget and print the same
report.

Quick start
===========

.. code-block:: bash

   pip install "spatialize[scenarios]"      # spatialize + PyYAML
   python -m spatialize.scenarios           # run every scenario, fast mode

Expected output (the seed and the p-values change from run to run; the decisions do not):

.. code-block:: text

   runner=spatialize mode=ci seed=12345 α_suite=0.001 (Holm)
   scenario/check                         family          p-value     level  expect result
   E2-mondrian-pair-cooccurrence/c1       gof-closed     2.43e-87   9.1e-05  reject PASS  — k=8 N=3100 ...
   S03-anisotropic-field/v1a              equivalence    1.51e-12   0.00017  pass   PASS  — mean=-0.2454 ...
   S03-anisotropic-field/v1b              one-sided      3.35e-08   0.00025  pass   PASS  — mean=0.6183 ...
   ...                                    (one line per check)
   S03-anisotropic-field/v1b-vor-data     one-sided      4.93e-08   0.00033  pass   PASS  — mean=0.8004 ...

   11 passed, 0 failed, 0 skipped (reproduce with --mode ci --seed 12345)

The command exits with status **0** when every check passed and **1** when any check failed, so
it can be used directly in scripts and continuous integration.

Prerequisites
=============

**Installed package.** The suite ships inside the ``spatialize`` package (scenario descriptors and
their data included). The only extra dependency is PyYAML, installed by the ``scenarios`` extra:
``pip install "spatialize[scenarios]"``, or ``pip install PyYAML`` next to an existing
installation.

**Source checkout.** Build the compiled library in place, then put the sources on the path:

.. code-block:: bash

   pip install -r requirements.txt PyYAML pytest
   python setup.py build_ext --inplace --force
   export PYTHONPATH=.:src/python             # in-place build first, then the sources
   python -m spatialize.scenarios

``--force`` matters: ``build_ext`` does not track the C++ headers, so without it a change to a
header alone is not recompiled and you would be testing the old library.

.. note::

   On macOS with a conda Python, importing the compiled library can fail with
   ``Library not loaded: ...libomp.dylib``. Point the loader at Homebrew's OpenMP:
   ``export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib``.

From the command line
=====================

.. code-block:: text

   python -m spatialize.scenarios [--mode {ci,full}] [--seed N] [--tier {T1,T2,T3}]
                                  [--id SCENARIO ...] [--alpha A] [--list] [--version]

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Option
     - Meaning
   * - ``--mode ci``
     - Default. Each check is sized to detect a deviation of about 0.05 with power 0.9; the whole
       catalogue runs in about a minute on a multi-core machine. Use it on every change.
   * - ``--mode full``
     - Each check is sized to detect about 0.02 (larger ensembles, more members). Slow; use it
       before a release or to certify an implementation. Only a ``full`` pass supports claims at
       that precision (see :doc:`statistics`).
   * - ``--seed N``
     - Seed of the run. By default a fresh seed is drawn and printed; pass it back to reproduce a
       run exactly.
   * - ``--tier T``
     - Run only the scenarios of one tier (``T1`` encoder law, ``T2`` estimator properties,
       ``T3`` geostatistical scenarios).
   * - ``--id SCENARIO``
     - Run only this scenario; repeat the option for several. ``--list`` shows the identifiers.
   * - ``--alpha A``
     - Family-wise error rate of the run (default :math:`10^{-3}`). Changing it is meant for
       studying the suite, not for making a run pass.
   * - ``--save-maps DIR``
     - Save the maps computed by the run (arrays and figures) under ``DIR`` for human review; see
       :ref:`scenarios-maps`. Does not change any decision.
   * - ``--list``
     - List the scenarios and their checks, and exit.

Exit status: ``0`` all checks passed (skipped checks do not count as failures), ``1`` at least one
check failed, ``2`` usage error (e.g. an unknown scenario).

Examples:

.. code-block:: bash

   python -m spatialize.scenarios --list
   python -m spatialize.scenarios --tier T1
   python -m spatialize.scenarios --id S03-anisotropic-field --mode full
   python -m spatialize.scenarios --mode ci --seed 12345      # reproduce a reported run
   python -m spatialize.scenarios --save-maps maps/                # also save the maps

.. important::

   The Holm budget covers **the checks of one run**. Running a subset (``--tier``, ``--id``) gives
   each check a larger share of the budget than in a full run, so a check can pass alone and fail
   in the full catalogue when its p-value is close to its level. The full catalogue run is the
   reference.

From Python
===========

.. code-block:: python

   from spatialize import scenarios
   from spatialize.scenarios.runners.spatialize import SpatializeRunner

   selected = scenarios.catalog()                      # or catalog(tier="T3"), catalog(ids=[...])
   report = scenarios.run(selected, SpatializeRunner(), mode="ci", seed=None)
   print(report.table())
   assert report.passed

   for o in report.outcomes:                           # one outcome per check
       print(o.scenario, o.check, o.result.p_value, o.level, o.passed)

:func:`catalog` loads the scenarios and verifies the checksums of their data; :func:`run`
evaluates them, applies Holm's procedure to all the checks of the call and returns a
:class:`Report`.

.. _scenarios-maps:

Looking at the maps
===================

Scenarios with visual criteria compute maps (for instance the median map of S03 on each replicate
field). They are not kept unless asked for:

.. code-block:: bash

   python -m spatialize.scenarios --id S03-anisotropic-field --save-maps maps/

or, from Python, ``scenarios.run(..., save_maps="maps/")``. This writes, under
``maps/<scenario>/``:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - File
     - Content
   * - ``<estimator>_<k>.png``
     - Field *k*: the truth with the sample locations (left) and the estimated map (right), on one
       colour scale; the declared orientation in white and the measured one in red; orientation
       :math:`\theta` and coherence :math:`c` of each map in the titles.
   * - ``<estimator>_summary.png``
     - All fields at once, five per row: each truth above its estimated map.
   * - ``compare_<k>.png``
     - Field *k*: the truth next to the map of every estimator of the scenario, on one colour scale
       (when the scenario has more than one estimator).
   * - ``<estimator>_point_<k>.npy``, ``<estimator>_truth_<k>.npy``
     - The arrays themselves (rows = :math:`y`, columns = :math:`x`, on the scenario's grid), to
       analyse further without rerunning.

The maps are those the checks were computed from in the same run, so each figure matches its line
of the report; with ``--seed`` they are reproducible. Saving needs matplotlib (a dependency of
spatialize) and never changes a decision: figures are for human review only (:doc:`visual`).

.. figure:: /_static/scenarios/S03_compare_13.png
   :width: 100%
   :alt: Truth and the five median maps of S03 on field 13

   ``compare_13.png`` of S03 (``ci`` mode, seed 12345): the truth with its samples and the median
   map of each estimator, on one colour scale; white is the declared orientation, red the measured
   one.

With pytest (source checkout)
=============================

``tests/scenarios/test_conformance.py`` runs the whole catalogue once and reports each check as a
separate test, so failures show up in the usual pytest summary. It puts the in-place build and
``src/python`` on the path itself:

.. code-block:: bash

   python -m pytest -q tests/scenarios
   SPATIALIZE_SCENARIO_MODE=full python -m pytest -q tests/scenarios      # full mode
   SPATIALIZE_SCENARIO_SEED=12345 python -m pytest -q tests/scenarios # reproduce a run
   python -m pytest -q -s tests/scenarios                                # also print the report

A failing test's message carries everything needed to reproduce it: seed, mode, family, p-value,
Holm level, expectation and details.

In continuous integration
=========================

Every push to ``develop`` builds the wheel, installs it, and runs
``python -m spatialize.scenarios --mode ci`` (workflow ``.github/workflows/test-linux.yml``). A
non-zero exit status fails the job; the seed is in the job log.

Reading the report
==================

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Column
     - Meaning
   * - ``scenario/check``
     - Scenario identifier and check identifier, as in the scenario's ``scenario.yaml``.
   * - ``family``
     - Test family, which fixes the statistic and the pass rule (:doc:`statistics`).
   * - ``p-value``
     - p-value of the check's test in this run.
   * - ``level``
     - Level Holm assigned to this p-value within the run (:math:`\alpha/m`,
       :math:`\alpha/(m-1)`, ... by rank).
   * - ``expect``
     - ``pass`` for an ordinary check; ``reject`` for a **negative control**, which passes when
       its test rejects (it shows the test has the power to see a known deviation).
   * - ``result``
     - ``PASS``, ``FAIL`` or ``SKIPPED`` (the runner does not provide that estimator), followed by
       the check's details: sample sizes, statistics, estimates and margins.

What to do when a check fails
=============================

1. **Reproduce it** with the printed seed and mode. Runs are deterministic given the seed.
2. **Do not reroll the seed until it passes.** With a correct implementation a run fails
   spuriously with probability at most :math:`10^{-3}`, so a failure is strong evidence of a real
   change. Rerunning with new seeds until one passes discards exactly that evidence.
3. **Read the details**: which quantity moved, by how much, and in which direction. Run the
   scenario in ``--mode full`` for a more precise estimate of the deviation.
4. If the change is intended (for example, a deliberate change of the partition process), the
   affected scenario must be revised — a new scenario version with its expectation updated and
   justified — never its threshold relaxed to accommodate the run.
5. A **negative control** that fails (its test did not reject) means the test lost power; the
   passes of that family are not trustworthy until it is fixed.

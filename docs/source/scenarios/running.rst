.. _scenarios-running:

##################
Running the tests
##################

.. currentmodule:: spatialize.scenarios

The scenarios run in three equivalent ways, from the command line, from Python, or with pytest in a
source checkout. All three run the same checks under the same error budget, printing the same report.

Quick start
===========

.. code-block:: bash

   pip install "spatialize[scenarios]"      # spatialize + PyYAML
   python -m spatialize.scenarios           # run every scenario, fast mode

The seed and the p-values change from run to run, while the decisions stay the same. A typical
output looks like this.

.. code-block:: text

   runner=spatialize mode=ci seed=12345 α_suite=0.001 (Holm)
   scenario/check                                    family          p-value     level  expect result
   E2-mondrian-pair-cooccurrence/c1                  gof-closed     2.43e-87   3.2e-05  reject PASS  — k=8 N=3100 ...
   S03-anisotropic-field/v1a                         equivalence    7.86e-23   7.1e-05  pass   PASS  — mean=-0.3027 se=0.472 ...
   S03-anisotropic-field/v1b                         one-sided      6.41e-12   0.00017  pass   PASS  — mean=0.6097 se=0.0116 ...
   ...                                               (one line per check)
   S03-anisotropic-field/v5-idw                      one-sided             1     0.001  pass   KNOWN  — mean=0.8461 se=0.00607 ...
   S12-edge-cases/e4-m-aidw                          almost-sure           1     exact  pass   PASS  — 0 violations ...

   61 passed, 0 failed, 3 known failures, 0 skipped (reproduce with --mode ci --seed 12345)

The command exits with status **0** when every check passed (recorded known failures included, see
:doc:`statistics`) and **1** when any check failed, so it can be used directly in scripts and
continuous integration.

Prerequisites
=============

**Installed package.** The suite ships inside the ``spatialize`` package, scenario descriptors and
their data included. Its only extra dependency, PyYAML, comes with the ``scenarios`` extra
(``pip install "spatialize[scenarios]"``), or with ``pip install PyYAML`` next to an existing
installation.

**Source checkout.** Build the compiled library in place, then put the sources on the path.

.. code-block:: bash

   pip install -r requirements.txt PyYAML pytest
   python setup.py build_ext --inplace --force
   export PYTHONPATH=.:src/python             # in-place build first, then the sources
   python -m spatialize.scenarios

``build_ext`` does not track the C++ headers, so without ``--force`` a change to a header alone is not
recompiled and you would be testing the old library.

.. note::

   On macOS with a conda Python, importing the compiled library can fail with
   ``Library not loaded: ...libomp.dylib``. Pointing the loader at Homebrew's OpenMP with
   ``export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib`` solves it.

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
       catalogue runs in about six minutes on a multi-core machine. Use it on every change.
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

The exit status is ``0`` when every outcome is as expected (skipped checks do not count as
failures), ``1`` when at least one check failed and ``2`` on a usage error (e.g. an unknown
scenario). Some examples follow.

.. code-block:: bash

   python -m spatialize.scenarios --list
   python -m spatialize.scenarios --tier T1
   python -m spatialize.scenarios --id S03-anisotropic-field --mode full
   python -m spatialize.scenarios --mode ci --seed 12345      # reproduce a reported run
   python -m spatialize.scenarios --save-maps maps/                # also save the maps

.. important::

   The Holm budget covers the checks of one run. Running a subset (``--tier``, ``--id``) gives each
   check a larger share of the budget than a full run does, so a check whose p-value is close to its
   level can pass alone while failing in the full catalogue. The run of the full catalogue serves as
   the reference.

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

:func:`catalog` loads the scenarios, verifying the checksums of their data. :func:`run` evaluates
them, applies Holm's procedure to all the checks of the call and returns a :class:`Report`.

.. _scenarios-maps:

Looking at the maps
===================

Scenarios with visual criteria compute maps (for instance the median map of S03 on each replicate
field), which are kept only on request.

.. code-block:: bash

   python -m spatialize.scenarios --id S03-anisotropic-field --save-maps maps/

From Python the same request reads ``scenarios.run(..., save_maps="maps/")``. Either form writes the
following files under ``maps/<scenario>/``.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - File
     - Content
   * - ``<estimator>_<k>.png``
     - Field *k*, with the truth and its sample locations (left) next to the estimated map (right) on
       one colour scale, the declared and the measured orientations drawn through the centre, and
       the orientation :math:`\theta` and coherence :math:`c` of each map in the titles.
   * - ``<estimator>_summary.png``
     - All fields at once, five per row, each truth above its estimated map.
   * - ``compare_<k>.png``
     - Field *k*, with the truth next to the map of every estimator of the scenario on one colour
       scale (when the scenario has more than one estimator).
   * - ``<estimator>_point_<k>.npy``, ``<estimator>_truth_<k>.npy``
     - The arrays themselves (rows = :math:`y`, columns = :math:`x`, on the scenario's grid), to
       analyse further without rerunning.

The checks were computed from these same maps in the same run, so each figure matches its line of
the report, and ``--seed`` reproduces them. Saving needs matplotlib, a dependency of Spatialize, and
never changes a decision, since figures serve human review only (:doc:`visual`).

.. figure:: /_static/scenarios/S03_compare_13.png
   :width: 100%
   :alt: Truth and the seven median maps of S03 on field 13

   ``compare_13.png`` of S03 (``ci`` mode, seed 12345), with the truth and its samples next to the
   median map of each estimator on one colour scale and the declared and the measured orientations
   drawn through the centre.

With pytest (source checkout)
=============================

``tests/scenarios/test_conformance.py`` runs the whole catalogue once and reports each check as a
separate test, so failures show up in the usual pytest summary. It puts the in-place build and
``src/python`` on the path itself.

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
     - Scenario identifier and check identifier, as in the scenario's ``scenario.yaml``. A runner
       may append the route by which it reached the estimator, in brackets. Spatialize's runner
       has a single route and appends nothing.
   * - ``family``
     - Test family, which fixes the statistic and the pass rule (:doc:`statistics`).
   * - ``p-value``
     - p-value of the check's test in this run.
   * - ``level``
     - Level Holm assigned to this p-value within the run (:math:`\alpha/m`,
       :math:`\alpha/(m-1)`, ... by rank); ``exact`` for almost-sure checks, which are decided
       exactly and spend none of the budget (:doc:`statistics`).
   * - ``expect``
     - ``pass`` for an ordinary check; ``reject`` for a **negative control**, which passes when
       its test rejects (it shows the test has the power to see a known deviation).
   * - ``result``
     - ``PASS``, ``FAIL``, ``SKIPPED`` (the runner does not provide that estimator), ``KNOWN`` (a
       recorded known failure that failed, as expected; it does not fail the run) or ``XPASS`` (a
       known failure that passed: the run fails until the record is updated), followed by the
       check's details: sample sizes, statistics, estimates and margins.

What to do when a check fails
=============================

1. **Reproduce it** with the printed seed and mode. Runs are deterministic given the seed.
2. **Do not reroll the seed until it passes.** With a correct implementation a run fails
   spuriously with probability at most :math:`10^{-3}`, so a failure is strong evidence of a real
   change. Rerunning with new seeds until one passes discards exactly that evidence.
3. **Read the details**, which say which quantity moved, by how much and in which direction. Running
   the scenario in ``--mode full`` gives a more precise estimate of the deviation.
4. If the change is intended (for example, a deliberate change of the partition process), the
   affected scenario gets a new version with its expectation updated and justified. Its threshold is
   never relaxed to accommodate the run.
5. A **negative control** that fails (its test did not reject) shows that the test lost power, so
   the passes of that family cannot be trusted until it is fixed.

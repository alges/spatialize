.. _testing:

#######
Testing
#######

Spatialize is tested in three layers, each answering a different question.

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Layer
     - Question
     - Where
   * - Example scripts
     - Does the library build, install and run end to end?
     - ``examples/scripted_examples/``
   * - Conformance scenarios
     - Does the estimator behave as the theory says, statistically?
     - ``spatialize.scenarios``, ``tests/scenarios`` (:doc:`../scenarios/index`)
   * - Refactor guard
     - Did an internal change alter any output, even by one bit?
     - ``tests/refactor_guard``

Building for tests
==================

.. code-block:: bash

   python setup.py build_ext --inplace --force

``build_ext`` does not track the headers under ``include/spatialize/``, so after editing one only
``--force`` makes the change reach the compiled library. On macOS with a conda Python, setting
``DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib`` makes the in-place build load Homebrew's OpenMP
runtime.

Example scripts
===============

The scripts in ``examples/scripted_examples/`` use the public API end to end. Every push to
``develop`` builds the wheel, installs it and runs ``esi_3d_nongriddata.py``.

Conformance scenarios
=====================

The conformance scenarios are statistical acceptance tests on geostatistical situations with a known
truth (:doc:`../scenarios/running`). They run in CI after the example script.

.. code-block:: bash

   PYTHONPATH=.:src/python python -m spatialize.scenarios          # or: python -m pytest tests/scenarios

Refactor guard
==============

The refactor guard keeps bitwise snapshots of the outputs of the compiled entry points on fixed
inputs, which show that a pure refactor changes nothing.

.. code-block:: bash

   python -m pytest -q tests/refactor_guard

- The snapshots describe the code they were taken from, defects included. A behaviour changed on
  purpose needs only the affected cases regenerated, in the same commit as the change, with
  ``python tests/refactor_guard/make_snapshots.py --only CASE ...``.
- Floating-point results depend on the compiler, the math library and the OpenMP runtime, so the
  snapshots are kept per platform (``snapshots/<platform>/``), the test being skipped where none
  exist. On a new platform they are generated from a commit before the change to be checked.
- The guard refuses to run against a binary older than the sources (see "Building for tests").
- It checks numbers, not log or progress messages, and does not ship with the package.

The guard and the scenarios complement each other. The guard detects any change at all, on one
platform only, with no notion of right or wrong. The scenarios accept any correct implementation
while rejecting wrong behaviour, though a change too small to matter statistically escapes them.

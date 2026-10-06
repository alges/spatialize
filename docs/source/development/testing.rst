.. _testing:

#######
Testing
#######

Three layers, each answering a different question.

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

``--force`` is required after editing headers: ``build_ext`` does not track
``include/spatialize/*.hpp``, so without it a header-only change is not recompiled. On macOS with a
conda Python, set ``DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib`` so the in-place build loads
Homebrew's OpenMP runtime.

Example scripts
===============

The scripts in ``examples/scripted_examples/`` are runnable end-to-end uses of the public API. Every
push to ``develop`` builds the wheel, installs it and runs ``esi_3d_nongriddata.py``.

Conformance scenarios
=====================

Statistical acceptance tests on geostatistical situations with a known truth; see
:doc:`../scenarios/running`. They run in CI after the example script:

.. code-block:: bash

   PYTHONPATH=.:src/python python -m spatialize.scenarios          # or: python -m pytest tests/scenarios

Refactor guard
==============

Bitwise snapshots of the outputs of the compiled entry points on fixed inputs, used to prove that a
pure refactor changes nothing:

.. code-block:: bash

   python -m pytest -q tests/refactor_guard

- The snapshots describe the code they were taken from, including its defects. When a behaviour is
  changed on purpose, regenerate only the affected cases, in the same commit as the change:
  ``python tests/refactor_guard/make_snapshots.py --only CASE ...``.
- Floating-point results depend on the compiler, the math library and the OpenMP runtime, so
  snapshots are per platform (``snapshots/<platform>/``); the test is skipped where none exist. On a
  new platform, generate them from a commit *before* the change to be checked.
- The guard refuses to run against a binary older than the sources (see "Building for tests").
- It checks numbers, not log or progress messages, and it is not shipped with the package.

The guard and the scenarios are complementary: the guard detects any change at all, but only on
one platform and with no notion of right or wrong; the scenarios accept any correct implementation
and reject wrong behaviour, but cannot see a change too small to matter statistically.

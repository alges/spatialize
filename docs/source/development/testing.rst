.. _testing:

#######
Testing
#######

Spatialize is tested in four layers, each answering a different question.

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Layer
     - Question
     - Where
   * - Example scripts
     - Does the library build, install and run end to end?
     - ``examples/scripted_examples/``
   * - Unit tests
     - Do the properties that must hold exactly hold, function by function?
     - ``tests/unit``
   * - Guard checks
     - Did an internal change alter any output, even by one bit?
     - ``tests/guard``
   * - Conformance scenarios
     - Does the estimator behave as the theory says, statistically?
     - ``spatialize.scenarios``, ``tests/scenarios`` (:doc:`../scenarios/index`)

``make test`` runs the example script, the unit tests with the guard checks, then the scenarios,
stopping at the first layer that fails. The scenarios stay apart from the other tests, since the
suite is meant to test other implementations of the theory as well.

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

Unit tests
==========

The unit tests check properties that must hold exactly on this platform's build, grouped by topic
in ``tests/unit``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - file
     - what it checks
   * - ``test_partitions.py``
     - the cells read by ``cells`` are those of the estimators' partitions and of the Mondrian leaf
       function
   * - ``test_decoders.py``
     - the sharpened decoder without its factors is the adaptive one, and the adaptive decoders do
       not depend on the units of the coordinates
   * - ``test_empty_cells.py``
     - the policies ``nan``, ``mark`` and ``coarsen`` (:doc:`../theory/blockmark`)
   * - ``test_custom_decoder.py``
     - a decoder written in Python reproduces the built-in one it implements
   * - ``test_scores.py``
     - the density readings do not depend on the units, and a scorer leaves out the data it cannot score
   * - ``test_posterior.py``
     - the readings of the posterior analysis (:doc:`../theory/posterior`)
   * - ``test_information.py``
     - the marginal entropies of the mutual information
   * - ``test_summaries.py``
     - the summaries of the results report the same figures as text, HTML and ``rich``

.. code-block:: bash

   python -m pytest -q tests/unit

They import the in-place build and the sources of ``src/python`` (``tests/unit/conftest.py``), and
share their data with the guard checks (``tests/guard/cases.py``).

Guard checks
============

The guard checks keep bitwise snapshots of the outputs of the compiled entry points and of the
public functions on fixed inputs, which show that a pure refactor changes nothing.

.. code-block:: bash

   python -m pytest -q tests/guard

- The snapshots describe the code they were taken from, defects included. A behaviour changed on
  purpose needs only the affected cases regenerated, in the same commit as the change, with
  ``python tests/guard/make_snapshots.py --only CASE ...``.
- Floating-point results depend on the compiler, the math library and the OpenMP runtime, so the
  snapshots are kept per platform (``snapshots/<platform>/``), the checks being skipped where none
  exist. On a new platform they are generated from a commit before the change to be checked.
- The checks refuse to run against a binary older than the sources (see "Building for tests").
- They check numbers, not log or progress messages, and do not ship with the package.

The guard checks and the scenarios complement each other. The guard checks detect any change at all, on one
platform only, with no notion of right or wrong. The scenarios accept any correct implementation
while rejecting wrong behaviour, though a change too small to matter statistically escapes them.

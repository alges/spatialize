.. _scenarios:

#################
Conformance Tests
#################

.. currentmodule:: spatialize.scenarios

Spatialize is not a toy library. It is meant for real work, such as estimating resources, mapping
contamination, rainfall or hydrogeological properties, and quantifying how uncertain those maps are,
where its results feed decisions with real consequences. A practitioner who relies on it needs more
than examples that run. They need evidence that the estimators behave as the theory says they must,
in the situations that matter in practice, release after release.

The conformance tests provide that evidence, which makes them the statistical safety signature of
Spatialize. Each test sets up a geostatistical situation with a known truth and decides a
pre-registered acceptance criterion with a statistical test of stated error rate and power. The whole
suite runs automatically on every change pushed to the development branch, and anyone can run it on
their own machine with one command. Its findings stay public. A defect a test exposed was fixed, with
the test left in place to keep watch over it, while an estimator that falls short of a criterion by
its nature has the shortfall recorded as a *known failure*, with its reason, instead of a relaxed
criterion.

The package ``spatialize.scenarios`` implements the suite as a catalogue of such situations together
with the statistical machinery that decides whether an implementation of ensemble estimation over
random partitions behaves as the theory says it must. Spatialize runs it on itself, and any other
implementation of the same concepts can run it through a small adapter (a
:class:`~spatialize.scenarios.protocol.Runner`).

How to run the tests
====================

.. code-block:: bash

   pip install "spatialize[scenarios]"     # spatialize + PyYAML
   python -m spatialize.scenarios          # run the whole catalogue (fast mode)

The command prints a report with one line per check, exiting with status 0 when every check passed
and 1 otherwise. :doc:`running` covers the options (``--mode full``, ``--seed``, ``--tier``, ``--id``,
``--list``), running from Python, with pytest or from a source checkout, how to read the report and
what to do when a check fails.

Why scenarios, and why statistics
=================================

The estimators of Spatialize are random objects. An ensemble estimate combines many random
partitions of the domain (the *encoder*), a local model fitted inside each cell (the *decoder*) and,
for some decoders, random draws inside the cell. Two correct implementations, or the same
implementation before and after an internal change of its random number generator or of the order of
a floating-point sum, produce different numbers while estimating the same law.

The suite therefore never compares realisations. It compares laws, in three tiers:

- the law of the partition process (e.g. the probability that two locations share a cell),
- the propositions the theory proves about the ensemble (e.g. a draw decoder never leaves the values
  of its cell),
- the behaviour of complete estimators on geostatistical situations whose truth is known (e.g. an
  anisotropy that must be visible in the map).

Every acceptance criterion is a statistical test with a stated statistic, null hypothesis, level and
power (:doc:`statistics`). A check that holds *almost surely*, since a single violation is impossible
under the theory, forms the limiting case of a test.

The theory
==========

The scenarios come from the theory of ensemble estimation over random partitions developed by
Egaña, Díaz, Navarro and Ehrenfeld (*A General Theory of Higher-Order Geostatistics: Random
Partitions and Distributional Inference for Spatial Fields*) and from the original ensemble spatial
interpolation paper (Egaña et al., 2021, *Natural Resources Research* 30(5), 3777–3793). Each
scenario names the part of the theory it comes from. The theory is still a draft, so the
documentation cites it by content. Each scenario file also records the section, equation and figure
numbers of the draft it was written against (field ``book``, key ``draft_2026_09``), and no other
place keeps those numbers. The vocabulary used throughout is as follows:

- **encoder**: the random partition of the domain (Mondrian or Voronoi), whose law is fixed before
  any data are seen;
- **decoder**: the local interpolator applied inside a cell (cell mean, IDW, adaptive IDW,
  kriging, draw decoders);
- **ensemble members**: the decoder's predictions under each partition draw; their empirical
  distribution at a location is the estimator's *predictive law* there;
- **block-mark model**: the idealisation in which each cell carries one mark drawn from a common
  law, independently of the partition, under which the theory's identities hold exactly.

Tiers
=====

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Tier
     - What it tests
     - Target
   * - **T1** Encoder law
     - the law of the partition process (co-occurrence probabilities, partition laws)
     - closed forms and exact laws
   * - **T2** Estimator properties
     - propositions about the ensemble law and the decoders
     - theorems and identities
   * - **T3** Geostatistical scenarios
     - end-to-end behaviour on situations with a known truth
     - relations between estimators, reference laws, visual criteria

The suite carries its own version, independent of Spatialize's
(:data:`~spatialize.scenarios.VERSION`), so an external implementation can pin the version it conforms
to.

.. toctree::
   :maxdepth: 2

   running
   statistics
   encoders
   visual
   catalog
   file_format
   extending

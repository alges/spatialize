.. _scenarios:

#################
Conformance Tests
#################

.. currentmodule:: spatialize.scenarios

Spatialize is not a toy library. It is meant to be used in real work — estimating resources,
mapping contamination, rainfall or hydrogeological properties, and quantifying how uncertain those
maps are — where its results feed decisions with real consequences. A practitioner who relies on
it needs more than examples that run: they need evidence that the estimators behave as the theory
says they must, on the situations that matter in practice, and that this keeps being true release
after release.

These conformance tests are that evidence: **the statistical safety signature of Spatialize**.
Each test is a geostatistical situation with a known truth and a pre-registered acceptance
criterion, decided by a statistical test with a stated error rate and power. The whole suite runs
automatically on every change pushed to the development branch, and anyone can run it on their own
machine with one command. Its findings are public: when
a test exposed a defect, the defect was fixed and the test kept watch over it; when an estimator
falls short of a criterion by its nature, the shortfall is recorded as a *known failure* with its
reason, never hidden by relaxing the criterion.

Technically, ``spatialize.scenarios`` is a **conformance suite**: a catalogue of geostatistical
situations with a known truth, together with the statistical machinery that decides whether an
implementation of ensemble estimation over random partitions behaves as the theory says it must.
Spatialize runs it on itself, and any other implementation of the same concepts can run it too, by
providing a small adapter (a :class:`~spatialize.scenarios.protocol.Runner`).

How to run the tests
====================

.. code-block:: bash

   pip install "spatialize[scenarios]"     # spatialize + PyYAML
   python -m spatialize.scenarios          # run the whole catalogue (fast mode)

The command prints a report with one line per check and exits with status 0 when every check
passed and 1 otherwise. Options (``--mode full``, ``--seed``, ``--tier``, ``--id``, ``--list``),
running from Python or with pytest, running from a source checkout, and how to read the report and
act on a failure are explained in :doc:`running`.

Why scenarios, and why statistics
=================================

The estimators of Spatialize are random objects. An ensemble estimate is built from many random
partitions of the domain (the *encoder*), a local model fitted inside each cell (the *decoder*),
and, for some decoders, random draws inside the cell. Two correct implementations — or the same
implementation before and after an internal change of its random number generator or of the order
of a floating-point sum — produce **different numbers** while estimating **the same law**.

So the suite never compares realisations. It compares *laws*:

- the law of the partition process (e.g. the probability that two locations share a cell),
- the propositions the theory proves about the ensemble (e.g. a draw decoder never leaves the values
  of its cell),
- the behaviour of complete estimators on geostatistical situations whose truth is known (e.g. an
  anisotropy that must be visible in the map).

Every acceptance criterion is therefore a **statistical test** with a stated statistic, null
hypothesis, level and power (see :doc:`statistics`). A check that holds *almost surely* — one
violation is impossible under the theory — is the limiting case of a test.

The theory
==========

The scenarios come from the theory of ensemble estimation over random partitions developed by
Egaña, Díaz, Navarro and Ehrenfeld (*A General Theory of Higher-Order Geostatistics: Random
Partitions and Distributional Inference for Spatial Fields*) and from the original ensemble spatial
interpolation paper (Egaña et al., 2021, *Natural Resources Research* 30(5), 3777–3793). Each
scenario names the part of the theory it comes from. The theory is still a draft, so the
documentation cites it by content; each scenario file also records the section, equation and
figure numbers of the draft it was written against (field ``book``, key ``draft_2026_09``), the
only place where those numbers live. Vocabulary used throughout:

- **encoder**: the random partition of the domain (Mondrian or Voronoi), whose *law* is fixed
  before seeing any data;
- **decoder**: the local interpolator applied inside a cell (cell mean, IDW, adaptive IDW,
  kriging, draw decoders);
- **ensemble members**: the decoder's predictions under each partition draw; their empirical
  distribution at a location is the estimator's *predictive law* there;
- **block-mark model**: the idealisation in which each cell carries one mark drawn from a common
  law, independently of the partition — the model under which the theory's identities are exact.

Tiers
=====

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Tier
     - What it tests
     - Target
   * - **T1** Encoder law
     - the law of the partition process: co-occurrence probabilities, partition laws
     - closed forms and exact laws
   * - **T2** Estimator properties
     - propositions about the ensemble law and the decoders
     - theorems and identities
   * - **T3** Geostatistical scenarios
     - end-to-end behaviour on situations with a known truth
     - relations between estimators, reference laws, visual criteria

The suite is versioned independently of Spatialize (:data:`~spatialize.scenarios.VERSION`), so an external implementation
can pin the version it conforms to.

.. toctree::
   :maxdepth: 2

   running
   statistics
   encoders
   visual
   catalog
   file_format
   extending

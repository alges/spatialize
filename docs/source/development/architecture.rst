.. _architecture:

############
Architecture
############

Spatialize offers a Python API (``spatialize``) over a compiled C++ core, the extension module
``libspatialize``, built with pybind11 from ``src/c++/libspatialize.cpp`` and the header-only engine
in ``include/spatialize/``.

The encoder–decoder split
=========================

Ensemble spatial interpolation combines two independent ingredients, around which the C++ engine is
organised.

- The **encoder** is a random partition of the domain, drawn many times.
- The **decoder** is a local interpolator applied inside each cell of a partition.

Each ensemble member is the decoder's prediction under one partition draw, so the members at a
location form the estimator's predictive law there.

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Concept
     - C++
     - Header
   * - one partition
     - ``Partition`` (``MondrianTree``, with or without ``raw``, and ``VoronoiTree``)
     - ``partition.hpp``, ``partitions/mondrian.hpp``, ``partitions/voronoi.hpp``
   * - local interpolator
     - ``Decoder`` (``IDWDecoder``, ``KrigingDecoder``, ``AdaptiveIDWDecoder``, ``CustomDecoder``)
     - ``decoder.hpp``, ``decoders/idw.hpp``, ``decoders/kriging.hpp``, ``decoders/adaptive_idw.hpp``,
       ``decoders/custom.hpp``
   * - the ensemble
     - ``Ensemble``, one loop for estimation, leave-one-out and k-fold, with ``ESI`` drawing a Mondrian
       forest and ``VORONOI`` a Voronoi forest
     - ``ensemble.hpp``

A ``Partition`` exposes its cells only through ``n_leaves()``, ``search_leaf(point)`` and the samples
of each cell (``samples_by_leaf``). A ``Decoder`` implements ``leaf_estimation``, ``leaf_loo`` and
``leaf_kfold`` on the samples of one cell. It may also implement ``fit()``, run once after the forest
is drawn to compute per-cell parameters, as adaptive IDW does to fit its exponent and anisotropy.
Since the loop knows neither the partition process nor the decoder, any decoder runs on any
partition.

The headers follow the same roles under ``include/spatialize/``, with the three interfaces at the top
(``partition.hpp``, ``decoder.hpp``, ``ensemble.hpp``), one file per partition process in
``partitions/`` and one per decoder in ``decoders/``. No file holds a particular combination, so
Voronoi with kriging comes from ``partitions/voronoi.hpp`` together with ``decoders/kriging.hpp``,
assembled at run time. Two separate engines keep their own folders. Co-estimation lives in
``coesi/custom_coesi.hpp``, whose ``CUSTOM_COESI`` runs one ``CUSTOM_ESI`` ensemble per variable, while
the non-ensemble nearest-neighbour IDW lives in ``nn/``. Shared utilities stay at the top
(``utils.hpp``, ``kdtree.hpp``, ``callback*.hpp``), together with ``grad_descent.hpp``, the optimiser
used by adaptive IDW.

From Python to C++
==================

The compiled extension declares what it offers in a *catalogue* (``src/c++/registry.hpp``), where
every partition and decoder is registered once, with the dimensions it supports and its
parameters. ``libspatialize.catalog()`` returns it as plain Python data, and the facade between
the Python API and the extension (``spatialize/gs/__init__.py``) is built from it, so a partition
or decoder registered in C++ needs no second list in Python.

1. A public function (``esi_griddata``, ``esi_nongriddata``, the hyperparameter searches, ...)
   resolves its defaults and calls ``lib_spatialize_facade.get_operator``, which checks the
   partition, the decoder and the dimension against the catalogue.
2. ``build_arg_list`` (``spatialize/gs/esi/_main.py``) builds the positional arguments, all
   ``float32``, which the operator maps onto ``libspatialize.run``, applying the session settings
   (:mod:`spatialize.session`).
3. ``run`` validates the arrays and the parameters, then calls the internal engine
   ``run_ensemble``, which proceeds in four stages.

   - It computes the box of samples and queries and, for Mondrian, the lifetime
     :math:`\lambda = 1/(\mu(H)(1-\alpha))`, with :math:`\mu(H)` the sum of the box's sides.
   - It draws the forest with a ``std::mt19937`` seeded by ``seed``.
   - It fits the decoder, continuing the same generator.
   - It runs estimation, leave-one-out or k-fold, returning ``(None, members)`` with one column per
     partition.

The generic entry point ``libspatialize.run``
=============================================

``libspatialize.run`` exposes the engine directly, for any partition and decoder of the catalogue.

.. code-block:: python

   import libspatialize as lib

   _, members = lib.run(samples, values, queries,            # float32 arrays
                        partition="voronoi", alpha=0.5,       # a partition of lib.catalog()
                        forest_size=300, seed=42,
                        decoder="kriging",                    # a decoder of lib.catalog()
                        params={"model": "exponential", "nugget": 0.0, "range": 0.3, "sill": 1.0},
                        method="estimate")                    # "estimate" | "loo" | "kfold"

- ``params`` holds the decoder's parameters as the catalogue lists them. A missing required
  parameter, an unknown one or an unsupported dimension raises an error.
- ``alpha`` keeps its meaning from the public API, the normalised granularity above for Mondrian and
  the nuclei rate for Voronoi, where negative values place the nuclei uniformly in the box instead of
  at sample locations.
- ``method="kfold"`` uses ``k`` and ``folding_seed``.

As a low-level function it applies neither defaults nor session settings, returning raw members.
``lib_spatialize_facade.run`` is the same call within the session settings.

Supported combinations
======================

Every partition works with every decoder, in the dimensions both support.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Catalogue name
     - Dimensions
     - Role
   * - ``mondrian``
     - 1 or more
     - Spatialize's Mondrian partition (the default)
   * - ``mondrian-raw``
     - 1 or more
     - the theory's Mondrian process (opt-in)
   * - ``voronoi``
     - 1 or more
     - Voronoi partition, nuclei among the samples or uniform in the box
   * - ``idw``
     - 1 or more
     - inverse distance weighting
   * - ``kriging``
     - 1 or more
     - ordinary kriging with a fixed variogram
   * - ``adaptiveidw``
     - 2 or 3
     - IDW with exponent and anisotropy fitted in each cell
   * - ``custom``
     - 1 or more
     - Python callables on each cell (categorical ESI and user decoders)
   * - ``cellmean``
     - 1 or more
     - the mean of the cell's data
   * - ``draw``
     - 1 or more
     - a datum of the cell drawn uniformly
   * - ``wdraw_idw``
     - 1 or more
     - a datum drawn with probability proportional to 1/d^p
   * - ``wdraw_kriging``
     - 1 or more
     - a datum drawn with probability proportional to its kriging weight, made non-negative
   * - ``sharpidw``
     - 2 or 3
     - the sharpened adaptive IDW, with residual-boosted weights and an exponent raised with the
       cell's gradient
   * - ``wdraw_adaptiveidw``
     - 2 or 3
     - a datum drawn with probability proportional to its adaptive IDW weight
   * - ``wdraw_sharpidw``
     - 2 or 3
     - a datum drawn with probability proportional to its sharpened weight

The decoders that draw take their random numbers from the run's seed, the tree and the query alone,
so a draw depends neither on the number of threads nor on the other queries. With one seed they see
the same partitions as the averaging decoders.

The public functions offer the averaging decoders ``idw``, ``kriging`` and ``adaptiveidw``.
Categorical ESI uses ``custom``. The other decoders are reached through ``run`` until the public
functions take them. Plain
IDW (``spatialize.gs.idw``) is a separate engine, outside the catalogue.

Changes of results
==================

The engine reproduces earlier results bit for bit, except where a defect was corrected on purpose.
The following corrections change numbers.

- **IDW weights** are :math:`1/d^p` everywhere, in estimation, leave-one-out and k-fold, on Mondrian
  and Voronoi partitions, with a datum at distance 0 taking all the weight. Earlier versions used
  :math:`1/(1+d^p)` in leave-one-out and k-fold, so cross-validation scored a different decoder from
  the one that predicts. The Voronoi IDW used it everywhere, which on a unit-scale domain comes close
  to a cell mean.
- **Kriging leave-one-out** applies the weights to the other data of the cell, while earlier versions
  mixed in the held-out datum's own value.
- **k-fold** gives NaN for a sample whose cell has no other sample to train on, where earlier versions
  returned 0.0.
- **Adaptive IDW** returns the datum's value at a query placed on it.
- **Voronoi with uniform nuclei** (negative ``alpha``) draws one uniform value per coordinate of each
  nucleus, where earlier versions drew the coordinates :math:`d` times over and kept the last draw.
  The law of the partition is the same, but a given seed yields a different partition.
- **Co-estimation** (``CUSTOM_COESI``) estimates each variable from its own data. Earlier versions
  used the first variable's data for all of them.

Reproducibility under parallel execution
========================================

- **Seeds.** The seeds determine a run fully. The forest, then the decoder's ``fit``, draw from one
  ``std::mt19937`` in a fixed order, while k-fold uses its own ``folding_seed``.
- **Parallel trees.** Estimation, leave-one-out and k-fold run the trees of the ensemble in
  parallel under OpenMP, each tree writing only its own column of the result, so the output is
  bitwise the same for any number of threads. A decoder whose ``thread_safe()`` returns false,
  such as ``CustomDecoder`` with its Python callbacks, keeps the trees serial. An exception raised
  inside a tree is rethrown once the loop ends.
- **Number of threads.** The session settings ``parallel`` and ``num_threads``
  (:mod:`spatialize.session`) decide it. The facade passes it to ``run`` as ``num_threads``, where 0
  leaves the OpenMP default of every processor, or ``OMP_NUM_THREADS``. ``libspatialize.build_info()``
  tells whether the extension was built with OpenMP. Without it, a request to run in parallel warns
  once, with installation instructions, then runs on one thread.
- **Parallel adaptive IDW.** The per-cell fitting of adaptive IDW also runs under OpenMP. Each tree
  receives its own generator, seeded before the parallel region, so parallel and serial runs give
  bitwise identical results.
- **Linear algebra.** Eigen is compiled with ``EIGEN_DONT_PARALLELIZE``. A matrix product split
  across threads changes the order of summation, which on a nearly singular kriging system can change
  the rank the pseudo-inverse keeps, so the results would depend on the number of threads.
- **Python from C++.** Progress reports, log messages and the Ctrl-C check call into Python, which
  requires the GIL. Inside a parallel region only the calling thread, identified by its OS thread
  id, may make such calls, the worker threads updating only atomic counters. Any new parallel code
  must follow the same rule.

Adding a decoder
================

1. Subclass ``Decoder`` in its own file under ``include/spatialize/decoders/``, implementing
   ``leaf_estimation``, ``leaf_loo`` and ``leaf_kfold`` on the samples of one cell. Each call also
   receives a ``CellContext`` with the tree, the cell and the run's seed, from which a decoder that
   draws derives its random numbers. Implement ``fit``
   if the decoder needs per-cell parameters, storing them in ``partition->leaf_params``, and draw
   any randomness from the generator passed to ``fit`` only. The ``leaf_*`` methods run on several
   cells at once, so they must not modify shared state. A decoder that cannot meet this overrides
   ``thread_safe()`` to return false.
2. Register it in the catalogue (``src/c++/registry.hpp``), with its dimensions, its parameters and
   a factory. ``run``, the facade and the scenario runner then know it.
3. Test it with the conformance scenarios, by naming it in a scenario's estimators (:doc:`testing`).
4. Give it a place in the public functions' argument lists (``build_arg_list`` and the defaults of
   ``signature_overload``), then document it.

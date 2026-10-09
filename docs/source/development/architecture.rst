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
assembled at run time. Two separate engines keep their own folders. Co-estimation, experimental (``spatialize.futures.coesi``), lives in ``coesi/custom_coesi.hpp``, whose
``CUSTOM_COESI`` runs one ``CUSTOM_ESI`` ensemble per variable on the former engine, while
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
``libspatialize.cells`` draws the same partitions for the same arguments and returns the cell of
every query in each of them, which :mod:`spatialize.gs.partitions` reads.
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
     - 1, 2 or 3
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
     - 1, 2 or 3
     - the sharpened adaptive IDW, with residual-boosted weights and an exponent raised with the
       cell's gradient
   * - ``wdraw_adaptiveidw``
     - 1, 2 or 3
     - a datum drawn with probability proportional to its adaptive IDW weight
   * - ``wdraw_sharpidw``
     - 1, 2 or 3
     - a datum drawn with probability proportional to its sharpened weight

The decoders that draw take their random numbers from the run's seed, the tree and the query alone,
so a draw depends neither on the number of threads nor on the other queries. With one seed they see
the same partitions as the averaging decoders.

Cells without data are handled once, in ``Ensemble`` (``include/spatialize/empty_cells.hpp``), so
every decoder on every partition follows the session setting ``empty_cells``, which ``run`` takes
with ``mark_source``, ``mark_knn`` and ``mark_value``. The decoder is never called on an empty cell
itself.

- Under ``"mark"``, ``fill_marks`` draws for each empty cell a source cell holding data, among the
  ``mark_knn`` nearest to the cell's ``leaf_point`` or among all of them, without repetition within
  a partition, then takes the source's decoder prediction at that point, or one of its data. The
  random numbers depend only on the seed, the tree and a key (the cell in estimation, the held-out
  datum in leave-one-out, the cell and the fold in k-fold).
- Under ``"coarsen"``, ``fill_coarse`` asks the partition for the coarser cell with data,
  ``Partition::coarser`` (the nearest ancestor for ``MondrianTree``, which keeps parent pointers,
  the nearest nucleus with data for ``VoronoiTree``), and predicts there with the decoder. An
  ancestor has no fitted parameters, so ``Decoder::fit_cell`` fits them on its data, with a seed
  derived from the run's seed, the region and the key.

In both, the held-out data of leave-one-out and k-fold take no part. A new partition implements
``leaf_point`` and ``coarser``, and a new decoder with per-cell parameters implements ``fit_cell``.

The public functions offer every decoder of the catalogue, ``custom`` included, whose Python
functions are passed as keyword arguments (:doc:`python_decoders`). Their argument lists keep the
historical order for ``idw``, ``kriging`` and ``adaptiveidw``, while the other decoders pass their
parameters in the catalogue's order, functions included, followed by the seed. Plain
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
- **Python loops.** The loops written in Python, the divergences of the Pareto encoder error, the
  simulations of ESS and the ranking of SPA, run on worker processes through
  ``spatialize._parallel.map_chunks``, with the same session settings. The items go in chunks of
  about four per worker, each chunk receiving only the data it needs. A few items run serially first,
  the rest going to the workers only when the time those items took, extrapolated to all of them,
  exceeds the cost of starting the workers. Each item carries its own seed, so the result does not
  depend on where it runs. New Python loops should use the same function. The ranking of SPA
  became a direct reading in version 1.3 and no longer uses it.

Progress and messages
=====================

The compiled code and the Python functions report through one protocol of JSON messages,
``spatialize.logging`` (``{"message": {...}}`` for a log line, ``{"progress": {"init": n, "step": s,
"desc": ...}}``, ``{"progress": {"token": ...}}`` and ``{"progress": "done"}`` for a run), sent to the
``callback`` of each function. The default callback shows them through ``spatialize._display``, in the
look the session setting ``display`` chooses.

- **Where the output goes.** With ``display="auto"``, :func:`spatialize._display.environment`
  detects a Jupyter kernel (HTML through ``IPython.display``, a progress bar updated in place), a
  terminal, including the terminals of the IDEs and PyCharm's run console (``rich``, with a live bar
  replaced by a summary line when the run ends), or any other output (plain lines on standard error,
  one at every quarter of a run). All three use the colours of the ``alges`` palette.
- **Names of the runs.** A run takes its name from the ``desc`` of its ``init`` message, otherwise
  from the engine's announcement just before it, ``"[C++|mondrian/idw] computing estimates"`` giving
  "computing estimates · mondrian/idw". New progress loops in Python pass a ``desc``. Runs may nest,
  each shown on its own.
- **Messages.** They go through Spatialize's own logger, ``spatialize.logging.log``, which does not
  propagate to the root logger and is not configured on import, so the logging of the application is
  left alone. The session setting ``verbosity`` sets the lowest level shown (warnings by default),
  unless a level was set on ``log`` itself.

Adding a decoder
================

There are two routes, each with its own page. :doc:`python_decoders` writes a decoder as Python
functions (optionally compiled with numba), with nothing to compile, passed as
``local_interpolator="custom"``. :doc:`cpp_decoders` adds a ``Decoder`` subclass to the library and
registers it in the catalogue, which runs on every core and draws random numbers reproducibly.

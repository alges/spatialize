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
     - ``Partition`` (``MondrianTree``, ``VoronoiTree``)
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

1. A public function (``esi_griddata``, ``esi_nongriddata``, the hyperparameter searches, ...)
   resolves its defaults and calls ``lib_spatialize_facade.get_operator``
   (``spatialize/gs/__init__.py``), which picks a compiled entry point from
   ``function_hash_map[dimension][partition + interpolator][estimate | loo | kfold]``.
2. ``build_arg_list`` (``spatialize/gs/esi/_main.py``) builds its positional arguments, all
   ``float32``.
3. The entry point (e.g. ``estimation_esi_idw``) validates the arrays, then calls the internal engine
   ``run_ensemble``, which proceeds in four stages.

   - It computes the box of samples and queries and, for Mondrian, the lifetime
     :math:`\lambda = 1/(\mu(H)(1-\alpha))`, with :math:`\mu(H)` the sum of the box's sides.
   - It draws the forest with a ``std::mt19937`` seeded by ``seed``.
   - It fits the decoder, continuing the same generator.
   - It runs estimation, leave-one-out or k-fold, returning ``(None, members)`` with one column per
     partition.

Every estimation entry point goes through ``run_ensemble``, so all of them share one implementation
of the loop, the partitions and the decoders.

The generic entry point ``libspatialize.run``
=============================================

``libspatialize.run`` exposes the engine directly, for any partition and decoder.

.. code-block:: python

   import libspatialize as lib

   _, members = lib.run(samples, values, queries,            # float32 arrays
                        partition="voronoi", alpha=0.5,       # "mondrian" | "voronoi"
                        forest_size=300, seed=42,
                        decoder="kriging",                    # "idw" | "kriging" | "adaptiveidw"
                        params={"model": 2, "nugget": 0.0, "range": 0.3, "sill": 1.0},
                        method="estimate")                    # "estimate" | "loo" | "kfold"

- ``params`` holds the decoder's parameters, ``exponent`` for ``idw``, then ``model``
  (1 spherical, 2 exponential, 3 cubic, 4 gaussian), ``nugget``, ``range`` and ``sill`` for
  ``kriging``, and ``metric`` (``"mae"`` or ``"mse"``) with ``parallelize`` for ``adaptiveidw``. A
  missing required parameter raises an error.
- ``alpha`` keeps its meaning from the public API, the normalised granularity above for Mondrian and
  the nuclei rate for Voronoi, where negative values place the nuclei uniformly in the box instead of
  at sample locations.
- ``method="kfold"`` uses ``k`` and ``folding_seed``.

As a low-level function it applies no defaults, returning raw members. For every combination that
also has a dedicated entry point, it returns exactly the same numbers given the same arguments. The
public Python API does not use it yet.

Supported combinations
======================

.. list-table::
   :header-rows: 1
   :widths: 20 20 30 30

   * - Partition
     - Decoder
     - Public API (``local_interpolator``, dimensions)
     - ``libspatialize.run``
   * - Mondrian
     - IDW
     - ``idw``, 2–5D
     - any dimension
   * - Mondrian
     - kriging
     - ``kriging``, 2–3D
     - any dimension
   * - Mondrian
     - adaptive IDW
     - ``adaptiveidw``, 2–3D
     - 2–3D
   * - Voronoi
     - IDW
     - ``idw`` with ``p_process="voronoi"``, 2D
     - any dimension
   * - Voronoi
     - kriging, adaptive IDW
     - —
     - as for Mondrian

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

Reproducibility under parallel execution
========================================

- **Seeds.** The seeds determine a run fully. The forest, then the decoder's ``fit``, draw from one
  ``std::mt19937`` in a fixed order, while k-fold uses its own ``folding_seed``.
- **Parallel adaptive IDW.** With ``parallelize=True`` the per-cell fitting runs under OpenMP. Each
  tree receives its own generator, seeded before the parallel region, so parallel and serial runs
  give bitwise identical results.
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
   ``leaf_estimation``, ``leaf_loo`` and ``leaf_kfold`` on the samples of one cell. Implement ``fit``
   if the decoder needs per-cell parameters, storing them in ``partition->leaf_params``, and draw
   any randomness from the generator passed to ``fit`` only.
2. Register it in ``run`` (``src/c++/libspatialize.cpp``) with its parameters.
3. Test it with the conformance scenarios, which reach any decoder the facade or ``run`` knows
   (:doc:`testing`).
4. Expose it in the Python facade (``function_hash_map``, ``build_arg_list``, defaults of the public
   functions), then document it.

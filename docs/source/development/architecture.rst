.. _architecture:

############
Architecture
############

Spatialize has a Python API (``spatialize``) over a compiled C++ core (the extension module
``libspatialize``, built with pybind11 from ``src/c++/libspatialize.cpp`` and the header-only
engine in ``include/spatialize/``).

Encoder and decoder
===================

Ensemble spatial interpolation combines two independent ingredients, and the C++ engine is
organised around that split:

- the **encoder** — a random partition of the domain, drawn many times;
- the **decoder** — a local interpolator applied inside each cell of a partition.

Each ensemble member is the decoder's prediction under one partition draw; the members at a
location form the estimator's predictive law there.

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Concept
     - C++
     - Header
   * - one partition
     - ``Partition``: ``MondrianTree``, ``VoronoiTree``
     - ``partition.hpp``, ``abstract_esi.hpp``, ``abstract_voronoi.hpp``
   * - local interpolator
     - ``Decoder``: ``IDWDecoder``, ``VoronoiIDWDecoder``, ``KrigingDecoder``,
       ``AdaptiveIDWDecoder``, ``CustomDecoder``
     - ``decoder.hpp``, ``esi_idw.hpp``, ``voronoi_idw.hpp``, ``esi_kriging.hpp``,
       ``adaptive_esi_idw.hpp``, ``custom_esi.hpp``
   * - the ensemble
     - ``Ensemble`` (one loop for estimation, leave-one-out and k-fold); ``ESI`` draws a Mondrian
       forest, ``VORONOI`` a Voronoi forest
     - ``ensemble.hpp``

A ``Partition`` exposes its cells only through ``n_leaves()``, ``search_leaf(point)`` and the
samples of each cell (``samples_by_leaf``). A ``Decoder`` implements ``leaf_estimation``,
``leaf_loo`` and ``leaf_kfold`` on the samples of one cell, and optionally ``fit()``, run once
after the forest is drawn to compute per-cell parameters (adaptive IDW fits its exponent and
anisotropy there). Because the loop knows neither the partition process nor the decoder, any
decoder runs on any partition.

The classes named after the original estimators — ``ESI_IDW``, ``VORONOI_IDW``, ``ESI_Kriging``,
``ADAPTIVE_ESI_IDW``, ``CUSTOM_ESI`` — are kept as fixed combinations of a partition and a decoder.
Co-estimation (``CUSTOM_COESI``) and the non-ensemble nearest-neighbour IDW (``NN_IDW``) are
separate engines.

From Python to C++
==================

1. A public function (``esi_griddata``, ``esi_nongriddata``, the hyperparameter searches, ...)
   resolves its defaults and calls ``lib_spatialize_facade.get_operator``
   (``spatialize/gs/__init__.py``), which picks a compiled entry point from
   ``function_hash_map[dimension][partition + interpolator][estimate | loo | kfold]``.
2. ``build_arg_list`` (``spatialize/gs/esi/_main.py``) builds its positional arguments, all
   ``float32``.
3. The entry point (e.g. ``estimation_esi_idw``) validates the arrays and calls the internal
   engine ``run_ensemble``, which

   - computes the box of samples and queries; for Mondrian, the lifetime
     :math:`\lambda = 1/(\mu(H)(1-\alpha))` with :math:`\mu(H)` the sum of the box's sides;
   - draws the forest with a ``std::mt19937`` seeded by ``seed``;
   - fits the decoder, continuing the same generator;
   - runs estimation, leave-one-out or k-fold, and returns ``(None, members)`` with one column
     per partition.

Every estimation entry point goes through ``run_ensemble``, so they all share one implementation
of the loop, the partitions and the decoders.

The generic entry point ``libspatialize.run``
=============================================

``libspatialize.run`` exposes the engine directly, for any partition and decoder:

.. code-block:: python

   import libspatialize as lib

   _, members = lib.run(samples, values, queries,            # float32 arrays
                        partition="voronoi", alpha=0.5,       # "mondrian" | "voronoi"
                        forest_size=300, seed=42,
                        decoder="kriging",                    # "idw" | "kriging" | "adaptiveidw"
                        params={"model": 2, "nugget": 0.0, "range": 0.3, "sill": 1.0},
                        method="estimate")                    # "estimate" | "loo" | "kfold"

- ``params``: ``idw`` → ``exponent``; ``kriging`` → ``model`` (1 spherical, 2 exponential,
  3 cubic, 4 gaussian), ``nugget``, ``range``, ``sill``; ``adaptiveidw`` → ``metric``
  (``"mae"`` or ``"mse"``), ``parallelize``. A missing required parameter is an error.
- ``alpha`` has the same meaning as in the public API: for Mondrian, the normalised granularity
  above; for Voronoi, the nuclei rate, with negative values for nuclei placed uniformly in the box
  instead of at sample locations.
- ``method="kfold"`` uses ``k`` and ``folding_seed``.

It is a low-level function: it does no default handling and returns raw members. For the
combinations that also have a dedicated entry point, it returns exactly the same numbers with the
same arguments — except Voronoi with ``"idw"``, see below. The public Python API does not use it
yet.

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
     - ``idw`` with ``p_process="voronoi"``, 2D (Voronoi IDW kernel)
     - any dimension (standard IDW kernel)
   * - Voronoi
     - kriging, adaptive IDW
     - —
     - as for Mondrian

Behaviours kept for reproducibility
===================================

The engine reproduces the results of earlier versions bit for bit. Two behaviours that follow
from this are worth knowing when comparing numbers:

- The Voronoi IDW of the public API weights the data by :math:`1/(1+d^p)`, while the Mondrian IDW
  uses :math:`1/d^p` in estimation (a query on a datum takes its value). This is the
  ``VoronoiIDWDecoder``; ``libspatialize.run`` with ``decoder="idw"`` uses the standard
  ``IDWDecoder`` on both partitions.
- Leave-one-out and k-fold for IDW weight the data by :math:`1/(1+d^p)`.

Reproducibility and threads
===========================

- **Seeds.** A run is fully determined by its seed(s): the forest, then the decoder's ``fit``, draw
  from one ``std::mt19937`` in a fixed order; k-fold uses its own ``folding_seed``.
- **Parallel adaptive IDW.** With ``parallelize=True`` the per-cell fitting runs under OpenMP. Each
  tree receives its own generator, seeded before the parallel region, so parallel and serial runs
  give bitwise identical results.
- **Python from C++.** Progress reports, log messages and the Ctrl-C check call into Python and
  therefore need the GIL. Inside a parallel region only the calling thread (identified by its OS
  thread id) may do so; worker threads only update atomic counters. Any new parallel code must
  follow the same rule.

Adding a decoder
================

1. Subclass ``Decoder`` and implement ``leaf_estimation``, ``leaf_loo`` and ``leaf_kfold`` (on the
   samples of one cell); implement ``fit`` if the decoder needs per-cell parameters, storing them
   in ``partition->leaf_params``. Use only the generator passed to ``fit`` for randomness.
2. Register it in ``run`` (``src/c++/libspatialize.cpp``) with its parameters.
3. Test it: the conformance scenarios run any decoder known to the facade (:doc:`testing`).
4. Expose it in the Python facade (``function_hash_map``, ``build_arg_list``, defaults of the
   public functions) and document it.

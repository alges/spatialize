.. _changes:

#############
Release notes
#############

1.3.0 (in development)
======================

Some results differ from version 1.2.0. The section *Results that change* lists where and why. A result computed with 1.2.0 keeps its value once saved, every result recording the settings it was computed with in its ``effective_config``.

New
---

- **Theory pages** without code, on the method and how to choose its pieces (:doc:`theory/index`),
  including the block-mark model and the treatment of cells without data (:doc:`theory/blockmark`).
- **Decoders.** The cell mean (``cellmean``), the uniform draw (``draw``), the weighted draws
  (``wdraw_idw``, ``wdraw_kriging``, ``wdraw_adaptiveidw``, ``wdraw_sharpidw``) and the sharpened
  adaptive IDW (``sharpidw``, with ``kappa_r``, ``kappa_g`` and ``rho_max``), described in
  :doc:`theory/decoders`.
- **Partitions.** The theory's Mondrian process, ``p_process="mondrian-raw"``. Every partition now works in any dimension, adaptive IDW, the sharpened decoder and their draws in one, two and three dimensions (on a line only the exponent is fitted).
- **Session settings** (:mod:`spatialize.session`). A fixed partition ``domain``; ``parallel`` and
  ``num_threads`` for every parallel computation; ``empty_cells``, with ``mark_source``,
  ``mark_knn`` and ``mark_value``, for the cells that hold no datum.
- **Cells without data.** The default, ``empty_cells="nan"``, keeps the behaviour of 1.2.0.
  ``empty_cells="mark"`` gives each such cell one value shared by its locations, drawn by default from the nearby cells through their decoder, or following the
  block-mark model exactly. :meth:`~spatialize.gs.esi.ESIResult.empty_cell_fraction` reports, at
  each location, the share of partitions concerned.
- **Partition laws** (:mod:`spatialize.gs.partitions`). The cells of given locations, the
  co-occurrence of sets and the law of groupings, with the closed forms of the Mondrian process.
- **Spatial entropy and mutual information** (:doc:`reference/esmi`).
- **Pareto search on any partition.** The encoder error is computed on Mondrian and Voronoi
  partitions alike, with ``data_cond`` for Voronoi.
- **Parallel computation.** The trees of the ensemble run in parallel, as do the Python loops of the
  Pareto search, the simulations and the ranking of the data, with results the same bit for bit for
  any number of threads or processes.
- **Conformance tests** (:doc:`scenarios/index`), runnable with ``python -m spatialize.scenarios``,
  which prints its progress as it runs.
- :func:`~spatialize.empirical.silverman_bandwidth`, the bandwidth of the kernel density estimates.

Results that change
-------------------

- **Adaptive IDW and the sharpened decoder.** Their weights were :math:`1/(10^{-10} + d^p)`, which
  gave many nearby data the same weight with large exponents and vanished for large coordinates,
  where most members came out undefined (more than half of them with coordinates in thousands). The
  weights are now relative to the nearest datum, which removes both effects. The results change
  slightly with small coordinates and greatly with large ones. With exact weights the sharpened map
  is no better than the adaptive one on the anisotropic test field (:ref:`scenario-S03`).
- **Adaptive IDW in three dimensions.** The per-cell fit compared candidates with one of their
  angles instead of their error, so it chose poor parameters. On the test data of the internal checks
  the leave-one-out error fell by about 45 %.
- **Kernel density estimates.** Their bandwidth ignored the spread of the sample, which made the
  simulations with ``point_model_name="kde"``, the ``neg_log_likelihood`` score and the Pareto
  encoder error depend on the units of the variable (samples with a spread of 0.01 were simulated
  44 times too wide). The bandwidth now follows Silverman's rule on each sample, the densities being evaluated exactly.
- **Pareto encoder error.** Besides the bandwidth, two details made it depend on the units, the
  floor under the second density and an underflow that could turn a pair's divergence into 0.
- **Simulation.** ``FittedModelFactory`` now ignores the undefined members by default
  (``nan_model_name="ignore"``); it used to replace them with the median, which narrowed the local
  laws where cells are often empty.
- **Mutual information.** The marginal entropies counted the projections of overlapping cells as
  separate bins, which overstated them, so independent variables showed a mutual information
  between 0.8 and 2.6 bits. The marginals are now integrated out of the joint density, and the members
  of cells without data are left out of the entropies.
- **IDW on Voronoi partitions** uses the weights :math:`1/d^p`, as on Mondrian partitions (it used
  :math:`1/(1 + d^p)`, close to a cell mean on small domains).
- **Cross-validation.** Kriging's leave-one-out applied its weights to the wrong data, and k-fold
  gave 0.0 instead of an undefined value in cells left with a single datum.
- **Smaller corrections.** Adaptive IDW returns the datum at a data location; kriging no longer
  depends on the number of threads; co-estimation estimates each variable from its own data; a
  failed simulation at a location gives undefined scenarios and one warning; default seeds are drawn
  at each call.

Removed
-------

- The ``parallelize`` argument of the ESI functions, and ``n_jobs`` of
  :func:`~spatialize.gs.ess.ess_sample` and
  :meth:`~spatialize.gs.spa.PosteriorSampleAnalyzer.rank_samples`. The session settings
  ``parallel`` and ``num_threads`` replace them.
- The dedicated entry points of the compiled module (``estimation_esi_*`` and their relatives).
  Every ensemble computation goes through ``libspatialize.run``, described in
  :doc:`development/architecture`.

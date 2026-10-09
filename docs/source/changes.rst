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
- **Decoders written in Python.** ``local_interpolator="custom"`` takes the functions of a decoder
  (``estimation``, and optionally ``post_creation``, ``loo`` and ``kfold``) in the estimation
  functions and the searches, on any partition and in any dimension. Leave-one-out and k-fold are
  derived from ``estimation`` when not given.
- **Categorical estimation on every partition.** The categorical functions take ``p_process`` and
  ``data_cond``. Their search cross-validates on the cells of one ensemble by default, as the
  continuous search does (``cv="engine"``), or by training the ensemble again for each fold
  (``cv="refit"``, the previous behaviour, which can select differently).
- **Partitions.** The theory's Mondrian process, ``p_process="mondrian-raw"``. Every partition now works in any dimension, adaptive IDW, the sharpened decoder and their draws in one, two and three dimensions (on a line only the exponent is fitted).
- **Session settings** (:mod:`spatialize.session`). A fixed partition ``domain``; ``parallel`` and
  ``num_threads`` for every parallel computation; ``empty_cells``, with ``mark_source``,
  ``mark_knn`` and ``mark_value``, for the cells that hold no datum.
- **Cells without data.** The default, ``empty_cells="nan"``, keeps the behaviour of 1.2.0.``empty_cells="mark"`` gives each such cell one value shared by its locations, drawn by default
  from the nearby cells through their decoder, or following the block-mark model exactly.
  ``empty_cells="coarsen"`` predicts its locations with the decoder from a coarser cell holding data,
  the nearest ancestor for Mondrian partitions or the nearest nucleus with data for Voronoi ones.
  The hyperparameter searches record, per configuration, the share of the data left out of the
  cross-validation score and the share of undefined members, warning when configurations leave out
  more than the session setting ``max_left_out``. :meth:`~spatialize.gs.esi.ESIResult.empty_cell_fraction` reports, at
  each location, the share of partitions concerned.
- **Partition laws** (:mod:`spatialize.gs.partitions`). The cells of given locations, the
  co-occurrence of sets and the law of groupings, with the closed forms of the Mondrian process.
- **Pareto search on any partition.** The encoder error is computed on Mondrian and Voronoi
  partitions alike, with ``data_cond`` for Voronoi.
- **Parallel computation.** The trees of the ensemble run in parallel, as do the Python loops of the
  Pareto search, the simulations and the ranking of the data, with results the same bit for bit for
  any number of threads or processes.
- **Progress and messages** in one look, chosen by the session setting ``display``: HTML in Jupyter
  notebooks, live bars drawn with ``rich`` in terminals and IDEs, plain lines in logs and files, all
  in the colours of the ``alges`` palette. Each progress bar names its task and shows the count, the
  elapsed time and the time left, then a summary line. The setting ``verbosity`` sets the lowest
  level of the messages shown. ``spatialize.session.show()`` uses the same look.
- **Conformance tests** (:doc:`scenarios/index`), runnable with ``python -m spatialize.scenarios``,
  which prints its progress as it runs. Scenario P10 holds the empty-cell policies to what they
  declare (:ref:`scenario-P10`). Scenario P11 shows the selection bias of a cross-validation score
  that drops undefined members (:ref:`scenario-P11`).
- :func:`~spatialize.empirical.silverman_bandwidth`, the bandwidth of the kernel density estimates.
- **Posterior analysis of the data** (:doc:`reference/spa`). :func:`~spatialize.gs.spa.posterior_audit`
  builds the law of each datum from the other data, with any decoder and partition, on the session
  domain or the box of the data, and reports the share of partitions each law rests on
  (:attr:`~spatialize.gs.spa.PosteriorAudit.support`). Each datum gets its position in its law, a
  tail probability, a level, a log score and a flag controlling the false discovery rate
  (Benjamini–Hochberg), after the laws are widened, spread by one fitted factor and given tails
  (Student-t kernels by default, or generalized Pareto). ``calibration()`` reports how well the laws
  are calibrated, with plots of the calibration, of the surprise on the map and of each datum's law
  (:doc:`theory/posterior`). The partitions also give each datum a declustering weight, the share of
  the domain it represents, for declustered summaries of the values, while the shift and the coherence of
  the neighbours, which tell an isolated error from an unrepresented part of the domain, the
  proportional effect and the co-located data complete the review. Scenario P12 holds the analysis to
  planted errors, a raised patch and a preferential design (:ref:`scenario-P12`). The functions of version 1.2 work on top of it.

Under study
-----------

A new module, :mod:`spatialize.futures` (:doc:`reference/futures`), holds features whose theory or
design is still being settled, so their interface and results may change without notice. Each
warns once per session when first used.

- **Spatial entropy and mutual information**, ``spatialize.futures.esmi``.
- **Co-estimation**, ``spatialize.futures.coesi``, a variable predicted from several others through
  two stages of ensembles, with a Python interface for the first time and the number of auxiliary
  locations as a parameter.

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
- **Categorical search.** :func:`~spatialize.gs.cat_esi.cat_esi_hparams_search` now cross-validates
  on the cells of one ensemble by default, so it can select differently than before;
  ``cv="refit"`` restores the previous scheme.
- **IDW on Voronoi partitions** uses the weights :math:`1/d^p`, as on Mondrian partitions (it used
  :math:`1/(1 + d^p)`, close to a cell mean on small domains).
- **Cross-validation.** Kriging's leave-one-out applied its weights to the wrong data, and k-fold
  gave 0.0 instead of an undefined value in cells left with a single datum.
- **Posterior analysis.** The law of each datum no longer contains the datum. Version 1.2 added the
  value to its own members on purpose, so that the law always reached the datum and its tail
  probability stayed away from 0, at the price of capping its surprise. A model of the tails now
  does that work. ``sample_quantiles`` and ``sample_entropy`` change accordingly.
  :meth:`~spatialize.gs.spa.PosteriorSampleAnalyzer.rank_samples` now places each datum by the
  central intervals of *probability* of its law, as the theory describes, where version 1.2 used
  intervals holding a share of the entropy of a fitted density, whose probability differed from the
  share (about 0.55 for 0.5 and 0.93 for 0.9). It no longer runs on several processes, having
  become a direct reading. A datum whose law fails to fit
  keeps NaN readings and one warning counts them, where it used to be dropped silently.
- **Laws outside their range.** The cumulative distribution function of an
  :class:`~spatialize.empirical.EmpiricalModel` is 0 below its grid and 1 above, and its density 0
  outside, where both were undefined (NaN). A datum far above its law now reads 1.
- **Warnings are shown.** The messages of Spatialize stood at the error level, so its warnings, such
  as the data a search leaves out of its score, were hidden. Warnings and errors are now shown by
  default (session setting ``verbosity``). Spatialize no longer configures the root logger of the
  application on import, its messages going through its own logger, ``spatialize.logging.log``.
- **Smaller corrections.** Adaptive IDW returns the datum at a data location; kriging no longer
  depends on the number of threads; co-estimation estimates each variable from its own data; a
  failed simulation at a location gives undefined scenarios and one warning; default seeds are drawn
  at each call; the plotting methods of the posterior analysis return their figure, without
  showing it; the reference pages of :func:`~spatialize.gs.esi.esi_hparams_search`,
  :func:`~spatialize.gs.esi.esi_pareto_hparams_search`,
  :func:`~spatialize.gs.cat_esi.cat_esi_hparams_search` and
  :func:`~spatialize.gs.spa.cv_sample_pred_posterior` show their documentation, which their
  decorator dropped.

Removed
-------

- The dependency on ``tqdm``, the progress bars being drawn with ``rich`` and, in notebooks, with
  HTML.

- The ``parallelize`` argument of the ESI functions, and ``n_jobs`` of
  :func:`~spatialize.gs.ess.ess_sample` and
  :meth:`~spatialize.gs.spa.PosteriorSampleAnalyzer.rank_samples`. The session settings
  ``parallel`` and ``num_threads`` replace them.
- The dedicated entry points of the compiled module (``estimation_esi_*`` and their relatives).
  Every ensemble computation goes through ``libspatialize.run``, described in
  :doc:`development/architecture`.

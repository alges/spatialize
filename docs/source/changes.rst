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
- **Dawid–Skene aggregation of categorical ensembles**, an alternative to the majority vote, which
  stays the default: :func:`~spatialize.gs.cat_esi.aggregate_with_btd`, with ``re_estimate('btd')``
  on a result, weighs the partitions by their reliability, for nominal, ordinal and binary variables,
  with a spatial variant for forbidden adjacencies (:func:`~spatialize.gs.cat_esi.optimize_btd_spatial_penalty`
  chooses its penalty). It is compiled with Spatialize, runs on the session's threads with the same
  result for any number of them, reports its progress and messages like the other functions, and
  stops on Ctrl-C between iterations. ``CatESIResult.class_probabilities`` gives the probability of
  each category read from the members.
- **Categorical estimation on every partition.** The categorical functions take ``p_process`` and
  ``data_cond``. Their search cross-validates on the cells of one ensemble by default, as the
  continuous search does (``cv="engine"``), or by training the ensemble again for each fold
  (``cv="refit"``, the previous behaviour, which can select differently).
- **Partitions.** Every partition now works in any dimension, adaptive IDW, the sharpened decoder and their draws in one, two and three dimensions (on a line only the exponent is fitted).
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
  level of the messages shown, and ``progress`` turns the bars off while keeping the warnings.
  ``spatialize.session.show()`` uses the same look. The ``callback`` of every function now defaults to
  ``None``, meaning the session's look, a callable remaining the way to send the progress and the
  messages elsewhere.
- **Summaries of the results**, in the manner of statistical packages. Every estimation, search,
  simulation, Pareto search and posterior analysis shows what was estimated and how, the data, the
  ensemble and the statistics of the estimate (count, undefined values, quartiles, mean, extremes,
  standard deviation), or its best configurations, as text when printed, as HTML in notebooks and in
  colour in terminals, with ``summary()`` and ``show()`` (:doc:`reference/result`). The summaries
  replace the short text of version 1.2. The ESI results now record their decoder, its parameters
  and the aggregation.
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

- **The default Mondrian partition is the theory's Mondrian process.** ``p_process="mondrian"``, the
  default, now draws the Mondrian process of the theory. The whole box may stay one cell. Each cut
  chooses its axis with probability proportional to its side, so the partitions follow the closed
  forms of the theory. Up to version 1.2 the default always cut the whole box and drew the
  axis uniformly, which near ``alpha`` = 1 multiplied the cells into slivers (at least 92 000 against
  2 400 on the unit square at ``alpha`` = 0.99, up to 9 GB for one partition). That partition remains
  as ``p_process="mondrian-legacy"`` and reproduces the results of version 1.2 exactly. For the same
  ``alpha`` the new default holds about half as many cells, so its partitions are coarser. A search
  of hyperparameters chooses ``alpha`` again (:doc:`theory/encoders`). The spatial entropy and
  co-estimation of :mod:`spatialize.futures` use the theory's process as well.
- **The Mondrian rate comes from the box of the data.** ``alpha`` now sets the rate
  :math:`\lambda = 1/(\mu(D)(1-\alpha))` on the box :math:`D` of the data, while the partitions are
  still drawn on the box of the data and the queries. Up to version 1.2 the rate came from the box of
  the data and the queries, so asking for estimates over a larger region coarsened the cells
  everywhere. With the theory's Mondrian process, which is consistent under restriction, the law of
  the estimate at a location no longer depends on the other locations requested. Results change only
  where queries lie outside the box of the data, as on most grids. The partition of version 1.2,
  ``"mondrian-legacy"``, keeps the rate of version 1.2 (:doc:`theory/encoders`).

- **Adaptive IDW and the sharpened decoder.** Their weights were :math:`1/(10^{-10} + d^p)`, which
  gave many nearby data the same weight with large exponents and vanished for large coordinates,
  where most members came out undefined (more than half of them with coordinates in thousands). The
  weights are now relative to the nearest datum, which removes both effects. The results change
  slightly with small coordinates and greatly with large ones. With exact weights the sharpened map
  is no better than the adaptive one on the anisotropic test field (:ref:`scenario-S03`).
- **Adaptive IDW in three dimensions.** The per-cell fit compared candidates with one of their
  angles instead of their error, so it chose poor parameters. Once corrected, its search over the six
  parameters (exponent, three angles, two anisotropy ratios) evaluated the 728 neighbours of every
  point at each move, with angles moving by one degree, which made the fit impractical on cells with
  hundreds of data. The fit now moves one parameter at a time, from coarse steps to fine ones (15,
  5 then 1 degree for the angles), from ten random starts. On the drill holes of the Andes it is 24
  to 49 times faster, its leave-one-out error being 2 % to 8 % lower than that of the corrected
  full search, and the bundled 3D example runs in two and a half minutes. Results in one and two
  dimensions are unchanged.
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
- **OpenMP runtime too old for the compiler.** On macOS inside a conda environment whose
  ``llvm-openmp`` is older than the compiler, the extension built but failed to load
  (``symbol not found ... ___kmpc_dispatch_deinit``). The installation now stops before building
  with the fix, ``conda install "llvm-openmp>=19"``, and the import explains it when the runtime is
  replaced afterwards (:doc:`troubleshooting`).
- **Warnings are shown.** The messages of Spatialize stood at the error level, so its warnings, such
  as the data a search leaves out of its score, were hidden. Warnings and errors are now shown by
  default (session setting ``verbosity``). Spatialize no longer configures the root logger of the
  application on import, its messages going through its own logger, ``spatialize.logging.log``.
- **Maps at scattered locations.** The plots of an estimate at locations given as a list
  (``esi_nongriddata``, ``quick_plot``) reshaped the values with ``w`` and ``h`` as given, so a
  swapped pair, as in the bundled examples, scrambled the map, and locations that did not fill a
  grid were cut to a smaller one. Locations on a regular grid now decide the shape and the order
  themselves (``w`` the number of distinct x, ``h`` of distinct y), a disagreeing ``w``/``h`` being
  ignored with a warning, and other locations are drawn as points.
- **Ctrl-C** stopped a run only at the end of a partition, which with adaptive IDW in 3D could take
  many minutes, and an interrupted fit raised a ``RuntimeError``. A run now stops within one cell,
  inside the fit of adaptive IDW within a fraction of a second, with a ``KeyboardInterrupt``. The
  progress bars of the compiled loops now advance with every thread, so their time remaining no
  longer overstates a long fit many times over.
- **Faster kriging.** In each cell kriging now solves its system once for the values, where it
  computed the weights of every location. A cell with many locations is several times faster, 2.7 s
  against about 9 s for 3 000 data and 20 000 locations, which brings the answer to Ctrl-C from about
  nine seconds to about five. Leave-one-out and k-fold also solve their systems directly. The
  estimates change only by float32 rounding, at most :math:`10^{-6}` relative.
- **Fitted laws.** The cumulative distribution function of
  :class:`~spatialize.empirical.EmpiricalModel` and its inverse are interpolated with PCHIP, which
  keeps them monotone and within :math:`[0, 1]`. Akima's interpolation could fall below 0 next to
  a sharp step. A Gaussian mixture with a near-degenerate component no longer makes the inverse
  fail, and a model built from a fitted kernel density (``skl_model``) spans its range, where its
  grid had collapsed to one point. Building scenario P4 brought out the first two. Between the
  points of the model's grid the values change slightly.
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
- The progress classes of version 1.2 in :mod:`spatialize.logging`, ``AsyncProgressHandler``,
  ``AsyncProgressCounter``, ``AsyncProgressBar`` and ``SingletonAsyncProgressCounter``. The session
  settings ``display`` and ``progress`` choose how the progress is shown, and ``DisplayProgress``
  is the callback that shows it.

- The ``parallelize`` argument of the ESI functions, and ``n_jobs`` of
  :func:`~spatialize.gs.ess.ess_sample` and
  :meth:`~spatialize.gs.spa.PosteriorSampleAnalyzer.rank_samples`. The session settings
  ``parallel`` and ``num_threads`` replace them.
- The dedicated entry points of the compiled module (``estimation_esi_*`` and their relatives).
  Every ensemble computation goes through ``libspatialize.run``, described in
  :doc:`development/architecture`.

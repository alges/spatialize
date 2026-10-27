.. _changes:

#############
Release notes
#############

1.4.0 (in development)
======================

Results that change
-------------------

- **The folds of k-fold cross-validation** (``k`` above zero in the searches, the posterior analysis
  and plain IDW). The folds were drawn from a number in [0, 1) that the engine truncated to the seed 0,
  so every ``folding_seed`` gave the same folds, the default drawn per call included. The folds now
  follow ``folding_seed``. Every k-fold score changes, with the searches that rank configurations by one.
  Leave-one-out and estimation are unchanged.
- **Plain IDW** (``idw_griddata``, ``idw_nongriddata``, ``idw_hparams_search``). Its leave-one-out
  skipped the neighbour at the held-out datum's position in the list of neighbours instead of the
  datum itself, so the datum kept its weight in its own prediction, and k-fold read the folds the same
  way. The cross-validation errors came out far too low (the leave-one-out matched the true one at
  0.3 % of the data), which steered the search towards small radii and large exponents. Each neighbour
  is now read by its own identity. Its weights, :math:`1/(1+d^p)`, depended on the units of the
  coordinates, close to a plain mean on a unit square. They are now those of the ensemble's IDW decoder,
  :math:`1/d^p` relative to the nearest datum. Every plain IDW result changes.
- **IDW and its weighted draw with small or large coordinates** (``idw``, ``wdraw_idw``). Their
  weights :math:`1/d^p` were computed in single precision, so with coordinates near :math:`10^{-3}`
  and an exponent of 14 every weight was infinite, every IDW member undefined and every draw the last
  datum of its cell, while with coordinates in thousands and a large exponent they vanished. The weights
  are now relative to the nearest datum, as in the adaptive decoders since 1.3, which removes both
  effects. At ordinary scales the weighted draws are unchanged and IDW changes by float rounding
  (relative differences near :math:`10^{-6}`), its estimation a little faster.
- **The cumulative distribution function of a fitted law** (``EmpiricalModel.cdf`` and its inverse).
  Its interpolant could return 1 + 2·10⁻¹⁶ or step down by 10⁻¹⁶, which failed scenario
  :ref:`P4 <scenario-P4>` at some seeds (6 of its checks, 22 times in 80 seeds). The values now stay in
  their range and never decrease, so they change by an ulp at most, where they did not.
- **Two checks of the conformance tests at their level.** :ref:`E2 <scenario-E2>` read its eight
  proportions from the same partitions and tested them as independent, so it rejected the theory's
  Mondrian process at about twice its level (10 % of seeds at 0.05). It now uses their covariance, which
  the closed form gives. :ref:`E3 <scenario-E3>` reads each triple with a seed of its own, which makes
  its proportions independent for any implementation (12 % of seeds before, 6 % after). The negative
  controls still reject, far below the level of the catalogue.

Renamed
-------

- **Two checks of the conformance tests** give one outcome per estimator, as every other check listing
  several estimators does, so a runner can look each one up by the identifier
  :func:`~spatialize.scenarios.engine.check_ids` gives it. ``support`` of
  :ref:`S04 <scenario-S04>` becomes ``support-draw`` and ``support-wdraw``. ``data-range`` of
  :ref:`S05 <scenario-S05>` becomes ``data-range-idw``, ``data-range-aidw`` and ``data-range-wdraw``.

1.3.0 (2026-10-21)
==================

Some results differ from version 1.2.0. The section *Results that change* lists where and why. A result computed with 1.2.0 keeps its value once saved, every result recording the settings it was computed with in its ``effective_config``.

New
---

- **Theory pages** without code, on the method and how to choose its pieces (:doc:`theory/index`),
  including the block-mark model and the treatment of cells without data (:doc:`theory/blockmark`).
- **Decoders.** The cell mean (``cellmean``), the uniform draw (``draw``), the weighted draws
  (``wdraw_idw``, ``wdraw_kriging``, ``wdraw_adaptiveidw``, ``wdraw_sharpidw``) and the sharpened
  adaptive IDW (``sharpidw``, with ``kappa_r``, ``kappa_g`` and ``rho_max``), described in
  :doc:`theory/decoders`.
- **The coupling of the draws** (``coupling``), a parameter of every drawing decoder.
  ``"location"``, the default, draws each location of a cell on its own, as before. ``"cell"`` gives
  the cell one level per tree, read on each location's weighted law, so block means, areas and curves
  read from the members keep the spread of the draws. The law at each location is the same in both
  modes (:doc:`theory/decoders`).
- **Decoders written in Python.** ``local_interpolator="custom"`` takes the functions of a decoder
  (``estimation``, and optionally ``post_creation``, ``loo`` and ``kfold``) in the estimation
  functions and the searches, on any partition and in any dimension. Leave-one-out and k-fold are
  derived from ``estimation`` when not given.
- **Two categorical data sets.** :func:`~spatialize.data.load_lithology_golden`, the lithology of a
  real geologic map near Golden, Colorado (USGS State Geologic Map Compilation, public domain), with six
  lithologies, four ordered eras and the whole map as the truth; and
  :func:`~spatialize.data.load_facies_pyrcz`, synthetic wells with two facies and continuous properties
  (GeoDataSets by M. J. Pyrcz, MIT licence, shipped with the files), one set sampled preferentially.
- **Two exhaustive real fields.** :func:`~spatialize.data.load_elevation_walker_lake` gives the
  elevation of the Walker Lake area, Nevada, on a 260 × 300 grid (USGS 3D Elevation Program, public
  domain). :func:`~spatialize.data.load_reflectance_sundarbans` gives the near-infrared reflectance of the
  Sundarbans mangrove forest on a 400 × 580 grid, its tidal channels a field of sharp, winding boundaries
  (Landsat 9, USGS, public domain). Each returns the whole grid as the truth with samples drawn from it.
- **Readings for an asymmetric loss.** ``functionals.AsymmetricLoss(over, under, kind)`` reads the
  members by the functional a use's costs select, ``over`` for each unit of over-prediction and
  ``under`` for each unit of under-prediction, at the level ``under / (over + under)``. A linear loss
  selects the quantile, a squared loss the expectile, a zero-one loss the mode.
  ``functionals.Expectile`` and ``functionals.ExpectedShortfall`` (the mean beyond the value at risk,
  either tail) are new readings of their own, and ``scoring.asymmetric_loss`` charges the same
  loss in cross-validation, so a search chooses the ensemble the use prefers.
- **Variogram inference from the ensemble** (experimental, :mod:`spatialize.futures.variogram`).
  :func:`~spatialize.futures.variogram.infer_variogram` reads the direction of a field's anisotropy and its
  correlation lengths from the data, in two or three dimensions, the direction from the structure tensor of
  the adaptive-IDW map at the knee of a Pareto search and the lengths from the members of the weighted draw,
  with the theory's residual weight at every node in place of a nugget and an upper bound on a measurement
  error of the data, the smallest within-cell variance of the data over a ladder of granularities.
  :meth:`~spatialize.gs.esi.ESIResult.variogram` reads them from an estimate already on a grid. Scenario
  :ref:`S14 <scenario-S14>` checks it against the field itself (:doc:`theory/other`).
- **Widened members keeping their index.** :func:`~spatialize.gs.ess.widen_members` widens the members
  of an ESI estimation member by member, with the laws of the widening of ``ess_sample``, each member
  keeping its column. The members' dependence between locations survives, so the law of a block
  average, read from the widened members column by column, keeps the cancellation between locations
  that the average of the locations' quantiles gives away (:doc:`theory/ess`, scenario
  :ref:`S10 <scenario-S10>`).
- **The Maipo precipitation as a data set.** :func:`~spatialize.data.load_precipitation_maipo` gives
  the daily precipitation of 69 rain gauges of the Maipo basin, central Chile, on three days of July
  2018, with the 2 km cells of the basin, the data of
  :class:`~spatialize.evaluation.PrecipitationCaseStudy` without its plotting dependencies.
- **Dawid–Skene aggregation of categorical ensembles**, an alternative to the majority vote, which
  stays the default: :func:`~spatialize.gs.cat_esi.aggregate_with_btd`, with ``re_estimate('btd')``
  on a result, weighs the partitions by their reliability, for nominal, ordinal and binary variables,
  with a spatial variant that keeps forbidden adjacencies apart by mean field. The members reach the
  compiled code as integer codes. A weighted majority vote,
  :func:`~spatialize.gs.cat_esi.aggregate_with_weighted_mv` (``re_estimate('wmv', weights=...)``),
  takes reliabilities known in advance. It is compiled with Spatialize, runs on the session's threads with the same
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
  proportional effect and the co-located data complete the review. The information readings, the log
  score of each datum, the entropy of its law and their difference, come in nats, bits or any other
  base (``base``). ``calibration(information=True)`` adds a check of the whole density of the laws
  through the mean excess surprise, and ``evidence()`` with ``plot_evidence()`` tells how much the other
  data say about each datum, through the support of its law and the information it gains over the
  distribution of all the data, a reading that does not depend on the units. ``null_flags()`` measures
  the false flags on the data at hand, auditing replicates free of errors drawn from the laws, with
  p-values read through the number of flags and through the most surprising datum.
  ``cross(covariate)`` reads each datum given a second variable measured at the same locations, which
  finds shifted or swapped values that the neighbours alone do not show. ``recommend()`` turns the
  readings into a list of actions, the data to measure again, groups of suspects to check, conflicting
  duplicates, places to sample more and flags that only a neighbouring suspect caused, found by cleaning
  the suspects and reading the data again, which ``plot_recommendations()`` draws on the map of the
  data. Categorical data have their own audit,
  :func:`~spatialize.gs.spa.cat_posterior_audit`, whose laws of the classes mix the members, the nearest
  data and all the data. A single category carrying little information, its main reading is an order of
  review (``review()``), with the same calibration, evidence, ``cross``, ``null_flags``,
  ``recommend()`` and maps. Scenario P12 holds the analysis to
  planted errors, a raised patch and a preferential design (:ref:`scenario-P12`). The functions of version 1.2 work on top of it.
- **The warning near alpha = 1** (``alpha > 0.95``) now comes only on ``"mondrian-legacy"``, whose
  cells multiply there. The theory's Mondrian process and Voronoi grow their cells with alpha without
  that explosion, so fine granularities such as 0.98 are ordinary values.
- **A faster cell search.** Finding the cell of a location in a Mondrian tree copied the location at
  every level of the tree, and the tree's box at every call. It now reads them in place, with the same
  results bit for bit. Estimating 40 000 locations with 200 partitions takes 0.11 s instead of 0.31 s
  with the uniform draw, and about 10 % less with the adaptive IDW.
- **Searches no longer stall on threads.** numpy's BLAS and the engine's OpenMP each keep a pool of
  threads that spin for a while when idle. Where Python and the engine take turns, as in a search,
  the two pools starved each other, so a run of minutes could take tens of minutes, more often in
  Jupyter. BLAS now runs on one thread while the engine runs, keeping all its threads outside. A
  tutorial with a Pareto search over 90 variograms took 163 s in Jupyter, against more than 30
  minutes before.
- **``post_creation`` only on cells with data.** A decoder written in Python had its
  ``post_creation`` called on the cells without data too, so a fit dividing by the number of data
  failed on fine partitions. It now sees only cells with data, as the developer guide says.
- **Saving a result on a grid.** ``save_result`` wrote the estimate of a gridded result as a table with
  one column per grid row under a single header, and failed. It now writes the estimate and the
  locations flat, and ``load_result`` rebuilds the grid.
- **The quick plot of simulations on a grid.** ``ESSResult.quick_plot`` drew the fields of a gridded
  result as points with coordinates taken from the grid array itself, and failed. It now draws them on
  the grid.
- **plot_nongriddata without xarray.** It laid the values on their lattice through xarray, which
  Spatialize does not depend on, so it failed in a plain installation. It now uses NumPy.
- **Scores by name in the Pareto search.** ``esi_pareto_hparams_search`` took ``scoring="nll"``,
  ``"crps"``, ``"rmse"`` or ``"mae"`` in its documentation only, failing on any name. It now takes them.
- **Locations without members.** Under ``empty_cells="nan"`` a location far from the data can have no
  defined member. The mean, median and percentile aggregations give it NaN without numpy's warning
  about an empty slice; ``empty_cell_fraction()`` reports such locations.
- **Leave-one-out by name.** Every cross-validated search and the posterior analysis take
  ``k="loo"`` for leave-one-out, as the Pareto search already did, besides ``-1`` and the number of
  data.
- **The posterior analysis of categories repeats with its seed.** ``cat_posterior_audit`` passed its
  seed to the partitions but not to the classifiers, so the cross-validation inside ``knn_pca`` and the
  random state of ``rf`` and ``dt`` drew from the operating system, and two audits with the same seed
  gave different members, laws and lists of actions. The seed now reaches the classifiers, as it already
  did in ``cat_esi_griddata`` and ``cat_esi_nongriddata``.

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

- **Cross-validation of the adaptive decoders** (``adaptiveidw``, ``sharpidw`` and their weighted
  draws and means). Each cell fits its parameters by its own leave-one-out error, and the held-out
  predictions reused them, so every datum had helped choose the parameters that predicted it. Their
  leave-one-out and k-fold errors came out too low (4.96 mm against 6.14 mm with the gauge removed on
  the Maipo rain of 6 July, 24.7 m against a true 35.5 m on Walker Lake), which also misled the
  searches. The cell's data are now split in two halves, each predicted with the parameters fitted on
  the other, and cross-validation fits a cell in full only where a coarser cell or a mark needs it. The
  scores are honest and the searches faster (a Pareto search of adaptive IDW on Walker Lake, 37 s
  against 62 s). Estimation is unchanged.
- **The CRPS score** (``scoring.crps``, ``scoring="crps"``) integrated the cumulative distribution
  of a density fitted to the members, over the grid of that density only, and returned infinity, without
  saying why, when the fit failed. It is now the exact score of the members themselves,
  :math:`\mathrm{E}|X - y| - \tfrac12\,\mathrm{E}|X - X'|`, which keeps an atom among the members,
  such as the dry gauges the drawing decoders reproduce. Searches scored by the CRPS change.
- **The entropy of a fitted law** (``EmpiricalModel.entropy``). The density was divided by its sum
  before the integral, which returned neither the differential nor the discrete entropy, 0.04 for a
  normal law of unit spread whose entropy is 1.42. It is now the differential entropy, negative for a
  law narrower than a unit interval. The two entropy intervals keep their intervals, chosen by shares
  of the entropy of the law on its support grid, and report differential entropies.
- **Spatial mutual information** (``spatialize.futures.esmi.SpatialMutualInformation``,
  experimental). The ensembles of the two variables were drawn with different seeds, so their members
  were independent and the mutual information came out near zero whatever the dependence. The new
  default, ``method="colocated"``, pairs each datum of one variable with the other predicted at its
  location and weighs the pairs by the partitions, so it reads the variables' own relation, a
  dependence that is not monotone included. On two variables linked over one half of a region and
  independent over the other, its ranks follow the true link with a correlation of 0.89, against 0.47
  before. The former estimate remains as ``method="members"``, now on shared partitions with a
  permutation correction (``bias_correction=True``).
- **Widening chosen once per variable** (``FittedModelFactory(widening="auto")`` in ``ess_sample``). The
  choice between a Gamma and a skew-normal widening was made at each location, by the sign of its
  members, so a location whose members were all positive, in a variable that also takes negative values,
  got a sharply peaked Gamma law. ``ess_sample`` and the posterior analysis (``PosteriorAudit``,
  whose default it is) now decide it once, by the sign of the data.
- **Spatial entropy by simulation** (``SpatialEntropy(method="ess")``, experimental): the entropy of the
  law ensemble spatial simulation fits at each location, widened to the spread of the nearest data,
  which follows the field with an averaging decoder.
- **Simulation with Gaussian mixtures** (``ess_sample`` with ``point_model_name="emm"`` or ``"vim"``).
  Every location's mixture took the factory's seed, so the draws at all locations shared one random
  stream, and a simulated field followed the cells of the partitions instead of being drawn location
  by location. Each location now takes its own seed, as the kernel density did.
- **The variogram of the kriging decoders** (``kriging``, ``wdraw_kriging``). The nugget scaled the
  whole covariance, the value at the origin included, so ordinary kriging, whose weights ignore a
  constant factor, never saw it. It is now the jump of the variogram at the origin, in the units of
  the sill. The spherical and cubic models rose back towards full correlation beyond the range, which
  gave distant data the weight of coincident ones. They now stay at the sill there. Every kriging
  estimate with a nonzero nugget, or with the spherical or cubic model and data farther apart than the
  range, changes.
- **The anisotropic classifier of categorical ESI** (``knn_pca``) stretched the distances along the
  boundaries between categories instead of across them, so cross-validation always chose the isotropic
  metric. It now stretches them across, and chooses an anisotropic metric where the categories lie in
  bands or elongated patches (:doc:`theory/categorical`). Categorical estimates with ``knn_pca``
  change.

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
- **Pareto encoder error.** The empirical robustness level ε̂ fitted the laws it compares again for
  each configuration, from that configuration's own cross-validated predictions, and took the largest
  divergence over the cells of all its partitions. It read the decoder and the size of the cells more
  than the partition. It grew with the IDW exponent, with the rate and with ``n_partitions``, while a
  floor under the second density capped every divergence at 14.73 nats. Version 1.2.0 hid most of
  this behind its wide kernels. ε̂ now compares laws fitted once for the whole search, by default
  those of each datum's 16 nearest data (``reference_laws``, ``n_neighbours``), read as kernel
  densities that keep their mean and variance, with a floor of spread (``min_spread``, 5 % of the
  standard deviation of the data) for data whose neighbours are all alike, such as a flat surface. It takes the largest divergence per partition and
  averages it over the partitions, the divergence being computed in log space with no floor. It no
  longer depends on the decoder or on ``n_partitions``, falling as the cells shrink. A search also
  runs faster, since ε̂ needs no leave-one-out ensemble (9 s against 3.4 s for 21 configurations on
  300 data). The default
  ``pair_strategy`` is ``"gaussian"``, the pair of each cell whose laws, read as Gaussians, diverge
  the most. The former default, ``"max_min"``, compared only the data of extreme means, missing
  laws that differ in spread.
- **Empty cells coarsened with an adaptive decoder.** Under ``empty_cells="coarsen"`` a Mondrian
  ancestor standing for an empty cell fitted the adaptive decoder's parameters on its own data, a
  search over hundreds of data near the root of a tree, run once per partition on one thread. With a
  query far from the data, or a large ``domain``, a call took minutes. The ancestor now takes the
  parameters already fitted in its cell with the most data, all of them usable, so that a held-out
  datum never shapes them. The members of the empty cells change. The same calls take about two
  seconds.
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

Renamed
-------

- Three modules of :mod:`spatialize.gs.esi`, so that each is named for what it holds.
  ``aggfunction`` is now :mod:`~spatialize.gs.esi.functionals`, the readings of the members reported as
  the map, ``lossfunction`` is now :mod:`~spatialize.gs.esi.loss` and ``scorefunction`` is now
  :mod:`~spatialize.gs.esi.scoring`. Their categorical counterparts follow,
  ``spatialize.gs.cat_esi.agg_functions`` becoming ``spatialize.gs.cat_esi.functionals`` and
  ``spatialize.gs.cat_esi.score_functions`` becoming ``spatialize.gs.cat_esi.scoring``. The old names
  still import, giving the new modules with a ``FutureWarning``, until version 1.4. The examples import
  them as ``fn``, ``lf`` and ``sc``.

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

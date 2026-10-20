.. _scenarios-file-format:

#######################
Scenario file reference
#######################

Each scenario occupies a directory ``spatialize/scenarios/catalog/<id>/`` holding a
``scenario.yaml`` descriptor, the normative definition of the scenario, together with ``data/*.npy``
files and a ``CHECKSUMS.sha256`` manifest when its data are pinned. The tables below list every key
the descriptor may contain.

Values that differ between the two modes are written as ``{ci: <value>, full: <value>}``, while a
plain value applies to both.

Top-level keys
==============

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - key
     - type
     - meaning
   * - ``id``
     - string
     - Identifier, equal to the directory name (e.g. ``S03-anisotropic-field``).
   * - ``version``
     - integer
     - Version of the scenario. Any change to its data, estimators or criteria is a new version;
       released pinned data never change within a version.
   * - ``tier``
     - ``T1`` | ``T2`` | ``T3``
     - Encoder law, estimator properties, or geostatistical scenario (:doc:`index`).
   * - ``evaluator``
     - string
     - Which evaluator turns the scenario into checks: ``pair_cooccurrence``, ``partition_law``,
       ``map_visual``, ``edge_cases``, ``draw_laws``, ``locality``, ``empty_cells``,
       ``cv_selection``, ``posterior_audit``, ``cat_posterior_audit``, ``mark_law``, ``convergence``,
       ``error_floor``, ``atom_at_zero``, ``selection_curve``, ``simulation_texture``, ``granularity``, ``quantile_choice``, ``confidence_classes``,
       ``law_validity`` or
       ``uncorrelated_covariance`` (below).
   * - ``book``
     - mapping
     - Source in the theory: ``topic`` (by content) and the draft's section, equation and figure
       numbers under a dated key such as ``draft_2026_09`` — the only place where those numbers
       live, since the theory is still a draft.
   * - ``purpose``
     - text
     - What the scenario tests, in words.
   * - ``domain``
     - mapping
     - ``box``: list of ``[low, high]`` per coordinate. The partitions are drawn on this box and rates
       are defined on it.
   * - ``truth``
     - mapping
     - The true field: ``generator`` and its parameters (below).
   * - ``data``
     - mapping
     - Sampling design, queries and replicate fields (below).
   * - ``estimators_T``
     - per mode
     - Number of ensemble members per estimator and field.
   * - ``estimators``
     - list
     - The estimators under test (below).
   * - ``checks``
     - list
     - The pre-registered checks (below).
   * - ``provenance``
     - mapping
     - Calibration notes and the history of each version: what was fixed before which run, and why
       anything changed.

Estimators
==========

.. code-block:: yaml

   estimators:
     - {id: krig, encoder: mondrian, rate: 5.0, decoder: kriging,
        params: {model: exponential, nugget: 0.0, range: 0.604, sill: 1.0}, empty_cells: nan}

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - key
     - meaning
   * - ``id``
     - Name of the estimator inside the scenario; checks refer to it.
   * - ``encoder``
     - Partition profile (:doc:`encoders`), one of ``mondrian`` (Spatialize's default Mondrian
       partition, the theory's process since version 1.3), ``mondrian-theory`` (the theory's Mondrian
       process), ``mondrian-legacy`` (the Mondrian partition of Spatialize 1.2,
       ``"mondrian-legacy"``), ``voronoi`` (Spatialize's Voronoi partition with uniform nuclei) or
       ``voronoi-data`` (Spatialize's Voronoi partition with nuclei at the data).
   * - ``rate``
     - Mondrian rate :math:`\lambda`, or Voronoi intensity :math:`\lambda_V` per unit volume, on the
       scenario's ``domain``. Never Spatialize's ``alpha``: runners derive their own parameters.
   * - ``decoder``
     - Local interpolator, with Spatialize's ``local_interpolator`` names: ``idw``, ``kriging``,
       ``adaptiveidw``.
   * - ``params``
     - Every decoder parameter, explicitly (they are pre-registered; runners fill in no defaults):
       ``idw`` — ``exponent``; ``kriging`` — ``model`` (``spherical``, ``exponential``, ``cubic``,
       ``gaussian``), ``nugget``, ``range``, ``sill``; ``adaptiveidw`` — ``metric``
       (``mae``/``mse``).
   * - ``empty_cells``
     - Policy for cells without data; ``nan`` (the cell yields NaN) is the only one every runner
       must support. Spatialize's runner also supports ``mark`` and ``coarsen``
       (:doc:`../theory/blockmark`).
   * - ``mark``
     - With ``empty_cells: mark``, how the mark is drawn, every key explicit: ``source``
       (``local``, ``cells`` or ``data``), ``knn`` (the number of nearby cells of ``local``) and
       ``value`` (``decoder`` or ``datum``), Spatialize's session settings ``mark_source``,
       ``mark_knn`` and ``mark_value``.
   * - ``reference``
     - For the evaluator ``empty_cells``, the estimator with the same encoder and decoder under
       ``nan``, run with the same seed, whose NaN members mark the queries of empty cells.

Checks
======

The following keys are common to every evaluator.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - key
     - meaning
   * - ``id``
     - Identifier of the check (with ``estimators``, outcomes are named ``<id>-<estimator>``).
   * - ``title``
     - What the check claims, in words.
   * - ``family``
     - Test family (:doc:`statistics`): ``gof-closed``, ``identity``, ``almost-sure``,
       ``equivalence``, ``one-sided``, ``paired-relation``, ``two-sample``.
   * - ``estimator`` / ``estimators``
     - The estimator the check reads, or a list of them (one outcome each).
   * - ``expect``
     - Optional, per encoder profile, ``pass`` (default) or ``reject`` for a negative control, e.g.
       ``{mondrian-theory: pass, mondrian-legacy: reject}``. The profiles ``mondrian-theory`` and
       ``voronoi-theory`` name the theory's processes. Spatialize provides the first as its default
       ``"mondrian"``.
   * - ``known_failure``
     - Optional reason: the check records a known defect. It is reported ``KNOWN`` while it fails
       and ``XPASS`` — failing the run — once it passes (:doc:`statistics`).

Evaluator ``pair_cooccurrence``
-------------------------------

This evaluator reads the partition law through the estimator (scenario E2), with a single datum and
empty cells as NaN.

- ``data.datum`` — location of the single datum (value 1).
- ``data.directions`` — directions of displacement (normalised to unit :math:`\ell_1` length).
- ``data.displacements`` — distances from the datum.
- per check: ``estimator`` or ``estimators``, ``n_members`` (per mode) and ``delta`` (per mode,
  the minimum detectable effect the sample size was chosen for).

Evaluator ``map_visual``
------------------------

This evaluator decides visual criteria on median maps over pinned replicate fields (scenario S03).

- ``truth``: ``generator: sgf_exponential`` with ``a1``, ``a2`` (ranges) and ``theta_deg``.
- ``data``: ``design`` (``type: uniform``, ``n``), ``grid`` (queries on an ``m``×``m`` grid of cell
  centres), ``mode: pinned``, ``fields`` (per mode), ``generator_seed``.
- per check, ``functional`` and its parameters:

  .. list-table::
     :header-rows: 1
     :widths: 28 72

     * - ``functional``
       - parameters
     * - ``orientation``
       - ``margin_deg``: TOST of the orientation error within ±margin (V1).
     * - ``coherence_ratio``
       - ``min_ratio``: coherence of the map over the truth's, above it (V1).
     * - ``axis_lock``
       - ``rotation_deg``; ``max_value`` (median map below it) or ``margin`` (within ±margin of 0)
         (V4).
     * - ``axis_lock_members``
       - ``rotation_deg``, ``members_checked``, ``min_value``: single members above it (V4 power
         check).
     * - ``contrast_ratio_reference``
       - ``min_ratio``: spread of the map over that of simple kriging with the true covariance,
         above it (V5).

Evaluator ``error_floor``
-------------------------

This evaluator reads the convergence of the ensemble mean and its error against the truth over pinned
replicate fields (scenario S02).

- ``truth``: ``generator: mbm``, the theory's Mondrian process of budget ``rate`` on the unit square
  with one mark per cell, ``marks: {gaussian: [mean, sd]}`` or ``{lognormal: [mu, sigma]}``.
- ``data``: ``design`` (``type: uniform``, ``n``), ``grid``, ``mode: pinned``, ``fields`` (per
  mode), ``sizes`` (the ensemble sizes), ``replicates`` (independent ensembles per size, per mode),
  ``generator_seed``.
- per check, ``kind`` (``spread``, identity, with ``bootstrap``; ``floor``, equivalence, with
  ``margin``) and ``estimator``.

Evaluator ``atom_at_zero``
--------------------------

This evaluator reads a field with an atom at zero through the probability of a dry location, over
pinned replicate fields (scenario S04).

- ``truth``: ``generator: vbm``, a Voronoi block-mark field of ``n_cells`` cells with uniform nuclei,
  ``marks: {zero_inflated: {p_zero, lognormal: [mu, sigma]}}`` (or ``{gaussian: [mean, sd]}``,
  ``{lognormal: [mu, sigma]}``, ``{centred_lognormal: [mu, sigma]}``, lognormal marks minus their
  sample mean).
- ``data``: ``design``, ``grid``, ``mode: pinned``, ``fields`` (per mode), ``dry_below`` (a member
  below it counts as dry), ``generator_seed``.
- per check, ``kind`` (``dry_fraction``, ``contrast``, ``dry_iou``, ``dry_edges``, each a paired
  relation of ``estimator`` against ``against`` in the direction ``better``; ``support``, almost sure,
  over ``estimators``).

Evaluator ``selection_curve``
-----------------------------

This evaluator reads the support effect on the share of the domain above a cut-off, over pinned
replicate fields (scenario S08).

- ``truth``: ``generator: vbm`` (see ``atom_at_zero``).
- ``data``: ``design``, ``grid``, ``mode: pinned``, ``fields`` (per mode), ``generator_seed``.
- per check, ``kind: share_above``, ``estimator``, ``quantile`` (the truth's quantile used as the
  cut-off) and ``better``, the direction in which the share of the map of means departs from the
  truth's (paired relation).

Evaluator ``simulation_texture``
--------------------------------

This evaluator reads fields simulated from the ensemble against the truth through their local
histograms, over pinned replicate fields (scenario S01), with the runner's optional method ``simulate``
(:doc:`extending`).

- ``truth``: ``generator: vbm`` (see ``atom_at_zero``).
- ``data``: ``design``, ``grid``, ``mode: pinned``, ``fields`` (per mode), ``n_simulated`` (simulated
  fields per replicate field), ``windows`` (windows per field), ``generator_seed``.
- per check, ``kind: local_histograms``, ``estimator`` and ``window`` (the side of the windows, in
  pixels).

Evaluator ``granularity``
-------------------------

This evaluator compares the same estimator at two granularities through readings of its median map,
over pinned replicate fields (scenario S11).

- ``truth``: ``generator: sgf_exponential`` (see ``map_visual``).
- ``data``: ``design``, ``grid``, ``mode: pinned``, ``fields`` (per mode), ``rotation_deg`` (the
  rotation of the data for the axis-locking), ``n_simulated`` (fields drawn from the widened law, for
  its spread) and ``generator_seed``.
- per check, ``kind`` and its keys:

  - ``law_spread`` (equivalence), ``estimator``, ``against`` and ``margin``: the ratio of the mean
    standard deviations of the widened laws lies within a factor ``1 + margin``. It needs the runner's
    optional method ``simulate`` (:doc:`extending`);
  - ``error``, ``axis_lock`` and ``diamond`` (paired relation), ``estimator``, ``against``, ``better``
    and optionally ``members`` and ``against_members``, the number of members of each median map
    (default ``estimators_T``);
  - ``contrast`` (one-sided), ``estimator`` and ``above``.

Evaluator ``quantile_choice``
-----------------------------

This evaluator reads the quantile an asymmetric loss chooses, at each location and over blocks, on
the members widened member by member, over pinned replicate fields (scenario S10). It needs the
runner's optional method ``widened_members`` (:doc:`extending`).

- ``truth``: ``generator: vbm`` (see ``atom_at_zero``).
- ``data``: ``design``, ``grid``, ``mode: pinned``, ``fields`` (per mode), ``block`` (the side of the
  blocks, in grid cells), ``level_step`` (the step of the levels searched for the optimal one),
  ``widening`` (the family of the widening) and ``generator_seed``.
- per check, ``kind``, ``estimator`` and its keys:

  - ``optimal_level`` (paired relation), ``cost``, ``against_cost`` and ``better``, the cost of
    over-estimation relative to under-estimation;
  - ``optimal_level_near`` (equivalence), ``cost`` and ``margin``;
  - ``block_gap`` (paired relation), ``level`` and ``better``;
  - ``block_loss`` (paired relation), ``level``, the cost being :math:`(1 - \text{level})/\text{level}`.

Evaluator ``confidence_classes``
--------------------------------

This evaluator classifies blocks by the confidence the law of their average gives, against classes
cut from the kriging variance, over pinned replicate fields (scenario S09). It needs the runner's
optional method ``widened_members`` (:doc:`extending`).

- ``truth``: ``generator: vbm`` (see ``atom_at_zero``).
- ``data``: ``design`` (``type: preferential`` with ``n_uniform`` data over the domain and ``n_box``
  more uniform in ``box``, or ``type: uniform`` with ``n``), ``grid``, ``mode: pinned``, ``fields``
  (per mode), ``block`` (the side of the blocks, in grid cells), ``thresholds`` (the relative
  half-widths bounding the first two classes), ``widening``, ``proxy`` (``practical_range`` and
  ``nugget`` of the exponential variogram of the kriging variance) and ``generator_seed``.
- per check, ``kind``, ``estimator`` and its keys:

  - ``ordered`` (paired relation), ``better_class`` and ``worse_class``;
  - ``against_proxy`` (paired relation), no further key.

Evaluator ``edge_cases``
------------------------

This evaluator decides almost-sure checks on degenerate designs (scenario S12).

- ``truth``: ``generator: vbm_edge_cases``, ``n_cells``, ``marks: {lognormal: [mu, sigma]}``.
- ``data``: ``design`` (``n`` points uniform in ``box``), ``duplicates`` (``same_value``,
  ``different_value``), ``queries`` (``grid``, ``on_data``), ``fields`` (per mode),
  ``generator_seed``.
- per check: ``kind`` (``runs``, ``finite``, ``convex``, ``exact``), ``estimators`` (list) and, for
  ``convex`` and ``exact``, ``tolerance`` (relative to the data range).

Evaluator ``partition_law``
---------------------------

This evaluator reads the law of the partition of a few locations through the estimator (scenarios
E1, E3, E4, E4b, E5 and E6), its estimators using the ``cellmean`` decoder.

- ``data.points`` — locations on a line (``line_groupings``, ``interval_only``,
  ``conditional_independence``); ``data.sets`` — sets of locations (``set_cooccurrence``);
  ``data.spacings`` and ``data.origin`` — four equally spaced points (``fourth_cumulant``);
  ``data.n``, ``data.distances``, ``data.generator_seed`` and ``data.origin`` (``voronoi_line``) or
  ``data.centre`` (``isotropy``) — filler data uniform in the domain and the distances read.
- per check: ``kind`` (``line_groupings``, ``interval_only``, ``set_cooccurrence``,
  ``fourth_cumulant``, ``conditional_independence`` with ``target`` ``poisson`` or ``forced_cut``,
  ``voronoi_line`` with ``rho``, ``isotropy``), ``estimators`` and ``n_members`` (per mode).
- ``voronoi_line`` and ``isotropy`` read the cells through the runner's optional method ``cells``,
  each distance with its own seed.

Evaluator ``locality``
----------------------

This evaluator compares the law at one location estimated with two sets of other queries (scenario
P6).

- ``data``: ``n`` data uniform in the domain, ``location``, the numbers ``few`` and ``many`` of other
  queries, ``beyond`` (how far past the domain the ``beyond`` queries reach) and ``generator_seed``.
- per check: ``kind`` (``inside`` or ``beyond``) and ``estimators``.

Evaluator ``empty_cells``
-------------------------

This evaluator decides properties of the empty-cell policies (scenario P10). Every estimator is
run with one seed, so all of them see the same partitions.

- ``truth``: ``generator: smooth_plus_noise``, drawn by the evaluator, not pinned.
- ``data``: ``n`` data uniform in ``box``, a part of the domain, ``grid`` (queries on a
  ``grid``×``grid`` lattice of cell centres over the domain) and ``generator_seed``.
- per check, ``kind`` and ``estimators``, each policy estimator naming its ``reference``:

  .. list-table::
     :header-rows: 1
     :widths: 22 78

     * - ``kind``
       - claim
     * - ``nan_iff_empty``
       - Under ``nan``, a member is NaN exactly when no datum lies in the query's cell.
     * - ``unchanged``
       - Where the reference is finite, the member equals it, up to ``tolerance`` × data range.
     * - ``filled``
       - No member is NaN.
     * - ``one_per_cell``
       - In each partition, the queries of one empty cell share one value.
     * - ``observed``
       - Every member at a query of an empty cell is a data value.
     * - ``cell_mean``
       - Every member at a query of an empty cell is the mean of the data of a cell of the same
         partition, up to ``tolerance`` × data range.
     * - ``data_law``
       - With ``location``, goodness of fit (``gof-closed``). At the query nearest to it, the
         members of the partitions where its cell is empty are uniform over the data values.

  The kinds ``nan_iff_empty``, ``one_per_cell`` and ``cell_mean`` read the cells of the partitions
  through the runner's optional method ``cells`` (:doc:`extending`). A runner without it has them
  skipped.

Evaluator ``cv_selection``
--------------------------

This evaluator reads leave-one-out ensembles under the empty-cell policies (scenario P11), through
the runner's optional method ``loo`` (:doc:`extending`). Every estimator is run with one seed.

- ``truth``: ``generator: smooth_plus_noise``, drawn by the evaluator, not pinned.
- ``data``: ``centres`` of Gaussian clusters of ``per_cluster`` data with standard deviation
  ``spread``, ``scattered`` data uniform in the domain, the ``noise`` of the values and
  ``generator_seed``.
- per check, ``kind`` and its keys:

  .. list-table::
     :header-rows: 1
     :widths: 22 78

     * - ``kind``
       - keys and test
     * - ``loo_filled``
       - ``estimators``: no leave-one-out member is NaN (almost-sure).
     * - ``loo_unchanged``
       - ``estimators``, each naming its ``reference`` under ``nan``, and ``tolerance``: where the
         reference is finite, the member equals it (almost-sure).
     * - ``selection``
       - ``estimator`` (under ``nan``), ``reference`` (defined at every datum) and ``min_share``.
         The data with fewer than ``min_share`` × T finite members of the estimator are left out.
         The errors of the reference at the data kept and left out are compared by a
         Kolmogorov–Smirnov test (two-sample), run as a negative control.

Evaluator ``posterior_audit``
-----------------------------

This evaluator reads the data against the laws the other data give (scenario P12), through the
runner's optional methods ``loo`` and ``cells`` (:doc:`extending`), with the readings of
:class:`~spatialize.gs.spa.PosteriorAudit`.

- ``data``: ``n`` data uniform in the domain, their ``noise``, the ``patch`` (``side`` and
  ``shift``), the ``preferential`` design (half of the data where the field exceeds ``above``),
  optionally ``cross``, two variables at the same locations (``shared``, the standard deviation of
  the component of the sample they share, ``noise``, the error of each, and ``shift``, the planted
  shift of a value of the first, which also has two values swapped), the false discovery rate
  ``q``, ``fields`` (per mode) and ``generator_seed``.
- per check, ``kind`` (``errors_found``, ``clean_flags``, ``shift_patch``, ``shift_error``,
  ``declustering``, ``weights``, ``errors_recommended``, ``cross_power``), ``estimator``, for the one-sided kinds on
  shares ``bound``, and for ``cross_power`` the ``cross_estimator`` whose partitions weigh the pairs.

Evaluator ``cat_posterior_audit``
---------------------------------

This evaluator reads categorical data against the laws of the classes the other data give (scenario
P13), through the runner's optional methods ``cat_loo`` and ``cells`` (:doc:`extending`), with the
readings of :class:`~spatialize.gs.spa.CategoricalPosteriorAudit`. Its estimators name a classifier as
their ``decoder``.

- ``data``: ``n`` data uniform in the domain, the number of ``classes`` (cuts of a latent field at its
  quantiles), the ``noise`` of the latent field, the ``covariate_noise``, the number of ``mislabels``
  (two distant data of different classes are swapped besides), the ``review_share`` of the list to check
  first, ``fields`` (per mode) and ``generator_seed``.
- per check, ``kind`` (``errors_ranked``, ``tail_calibrated``, ``cross_ranked``), ``estimator``,
  ``bound`` and, for ``cross_ranked``, the ``cross_estimator`` whose partitions weigh the data.

Evaluator ``draw_laws``
-----------------------

This evaluator decides the laws of decoders that return a draw (scenarios P2 and P3), each read
against a *reference* decoder run with the same seed.

- ``truth``: ``generator: smooth_plus_noise``, drawn by the evaluator, not pinned.
- ``data``: ``n`` data and ``queries`` queries, uniform in the domain, and ``generator_seed``.
- per check, ``kind`` and its keys:

  .. list-table::
     :header-rows: 1
     :widths: 22 78

     * - ``kind``
       - keys and test
     * - ``support``
       - ``estimators``: every finite member is a data value (almost-sure).
     * - ``mean``
       - ``estimator``, ``reference``: the mean of the difference is 0 at each query (identity).
     * - ``variance``
       - ``estimator``, ``reference``: the mean squared difference equals the weighted dispersion
         (identity).
     * - ``frequencies``
       - ``estimator``, ``reference`` (the cell mean): the counts of each datum match their
         expectation (gof-closed).

Evaluator ``mark_law``
----------------------

This evaluator decides the law of the members under ``empty_cells="mark"`` (scenarios P5, P8 and
P9). The estimators use the decoder ``draw`` with marks drawn as data, so that every member is one
datum. Its probability under each partition then follows from the cells, read through the runner's
method ``cells``. It is uniform over the data of the location's cell when that cell holds data.
Otherwise it is the mark law, one cell with data drawn uniformly then one of its data
(``"cells"``), or one datum drawn uniformly (``"data"``).

- ``truth``: ``generator: smooth_plus_noise`` (P5, P8) or a step field (P9), drawn by the
  evaluator, not pinned.
- ``data``: ``n`` data uniform in ``box`` (default the domain), an optional ``cluster`` (``n`` data
  uniform in its ``box``), the ``locations`` read (P5, P8) and ``generator_seed``.
- per check, ``kind``, ``estimators`` and its keys:

  .. list-table::
     :header-rows: 1
     :widths: 22 78

     * - ``kind``
       - keys and test
     * - ``member_law``
       - ``law`` (``cells`` or ``data``): at each location, the count of each datum against the sum
         over the partitions of its probability (gof-closed).
     * - ``weights_sum``
       - ``tolerance``. The estimator, or the one its entry names with ``reference``, is the cell
         mean under ``"nan"``. Run on each datum's indicator, its weights sum to one with the share
         of partitions whose cell is empty (almost-sure).
     * - ``residual_grows``
       - ``near``, ``far`` (indices into ``locations``): the location ``far`` has an empty cell more
         often, paired by partition (paired-relation).
     * - ``pair_product``
       - ``pair`` (two indices), optional ``shuffle``. The mean product of the two members minus its
         expectation given the cells is zero (identity). A shared empty cell gives the second moment
         of the mark law, distinct cells the product of the means. ``shuffle`` takes the second
         member from the next partition, a negative control.
     * - ``preferential``
       - ``estimator_b``, ``far``, ``field`` (``base``, ``jump`` inside the cluster's box,
         ``noise``), ``fields``. Over the fields, the marks of ``estimator_b`` lie farther from the
         field's spatial mean than those of the estimator, read at ``far`` in the partitions where
         its cell is empty (paired-relation).

Evaluator ``convergence``
-------------------------

This evaluator decides that the spread of the ensemble mean falls as :math:`T^{-1/2}` (scenario
P1).

- ``data``: ``n`` data and ``queries`` queries, uniform in the domain, and ``generator_seed``.
- per check, ``sizes`` (the ensemble sizes), ``replicates`` (independent ensembles per size, per
  mode) and ``bootstrap`` (resamples for the standard error of the slope). The slope of the mean
  log spread against :math:`\log T` must be :math:`-1/2` (identity).

Evaluator ``law_validity``
--------------------------

This evaluator decides that each reading of the law is a cumulative distribution function
(scenario P4), through the runner's optional method ``law_cdf``.

- ``data``: ``n`` data and ``queries`` queries, uniform in the domain, with positive values, and
  ``generator_seed``.
- per check, ``reading`` (named by the runner), ``thresholds`` (their number, over the data range
  widened by its length on each side) and ``tolerance`` (the decrease allowed). A value outside
  :math:`[0, 1]`, a NaN, a decrease or a reading that cannot be built is a violation
  (almost-sure).

Evaluator ``uncorrelated_covariance``
-------------------------------------

This evaluator decides the covariance of cell means of an uncorrelated field against its closed
form on a line (scenario P7).

- ``domain``: ``[0, 1]``, one dimension.
- ``data``: ``n`` data on the grid :math:`(k + 1/2)/n`, the grid point ``origin`` and
  ``generator_seed``. The values are iid standard normal, drawn anew for each field.
- per check, ``offsets`` (grid steps from the origin) and ``fields`` (per mode). The estimator's
  ``rate`` enters the closed form. The mean product of the members at the origin and at each offset
  is tested against it (identity).

Data files
==========

Pinned data are written by :func:`spatialize.scenarios.generators.materialise.materialise`
(``python -m spatialize.scenarios.generators.materialise <id>``) and verified against
``CHECKSUMS.sha256`` whenever the catalogue is loaded.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - file
     - content
   * - ``data/samples_<k>.npy``
     - data locations of field ``k``, float32, shape (n, d)
   * - ``data/values_<k>.npy``
     - data values, float32, shape (n,)
   * - ``data/truth_<k>.npy``
     - the true field at the queries (``map_visual``)
   * - ``data/queries_<k>.npy``
     - the queries, when they depend on the field (``edge_cases``)
   * - ``data/exact_<k>.npy``
     - (query index, datum value) for the queries placed on data (``edge_cases``)

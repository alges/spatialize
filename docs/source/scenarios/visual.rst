.. _scenarios-visual:

################
Visual criteria
################

.. currentmodule:: spatialize.scenarios.stats.maps

A practitioner judges several properties of an estimate by looking at a map, such as a clearly
visible anisotropy, blocks left by axis-aligned partitions, a sharp or diffuse boundary, or high
values that connect into bodies. Two correct implementations produce different pixels, so the suite
never compares pixels. It turns each property into a *map functional*, a number computed from a
gridded map whose threshold or relation is fixed before any reference run. The threshold is
pre-registered in the scenario file, versioned with it and never tuned to make a run pass. The
functional is then tested across replicate fields like any other check.

Which map is read
=================

Every criterion names the map it reads, since the readings of the estimator look different:

- ``point`` — a point functional of the law at each location (median, mean, quantile);
- ``member`` — one ensemble member, i.e. a field drawn under a single partition;
- ``simulated`` — a field from ensemble spatial simulation;
- ``exceedance`` — a map of exceedance probabilities; ``spread`` — a map of interval widths;
- ``truth`` — the true field, which the scenario knows.

Map functionals
===============

.. list-table::
   :header-rows: 1
   :widths: 24 50 26

   * - Functional
     - Definition
     - What it sees
   * - orientation, coherence (:func:`orientation_coherence`)
     - structure tensor :math:`J = \langle G_\sigma * (\nabla Z\,\nabla Z^\top) \rangle`; with
       eigenvalues :math:`\mu_1 \ge \mu_2`, coherence :math:`(\mu_1-\mu_2)/(\mu_1+\mu_2) \in [0,1]`;
       the elongation direction is orthogonal to the dominant gradient
     - whether an anisotropy is visible, and in which direction
   * - axis-artefact index (:func:`axis_artifact_index`)
     - share of gradient energy within ±5° of the coordinate axes, minus the truth's share
     - not used: confounded with smoothing (see below)
   * - axis-locking (:func:`axis_lock`)
     - spectral energy on the coordinate axes against the diagonals (Hann-tapered, per frequency
       ring; :func:`axis_diagonal_log_ratio`) of the map, minus the same for the estimator run on
       the data rotated 45° and evaluated at the same points
     - blocks and terraces tied to the coordinate axes (criterion V4); 0 for rotation-invariant
       partitions
   * - contrast ratio (:func:`contrast_ratio`)
     - standard deviation of the map over that of the truth
     - washed-out maps
   * - contrast against the best linear predictor
     - standard deviation of the map over that of the simple-kriging map computed with the true
       covariance from the same data
       (:func:`spatialize.scenarios.generators.fields.simple_kriging_exponential`)
     - maps more washed out than any good linear estimate needs to be (criterion V5)
   * - level-set IoU (:func:`level_set_iou`)
     - intersection over union of a thresholded map and the true region
     - shapes of plumes, dry regions, exceeded areas
   * - planned
     - directional range ratio, roughness, edge sharpness, connectivity (components and Euler
       characteristic over thresholds), halo index, level-curve shape of an estimated covariance
     - see the catalogue

The axis-artefact index is confounded with smoothing. Maps of rotation-invariant (Voronoi) estimators
get the same values as Mondrian ones, while the index barely changes between eight and a hundred
ensemble members. Criterion V4 therefore uses axis-locking.

*Calibration of orientation and coherence.* On six truth fields of scenario :ref:`S03 <scenario-S03>`
(40×40 grid, :math:`\sigma = 2`) with an anisotropy at 30°, the orientations fall between 26° and 29°,
with coherences between 0.70 and 0.76. An isotropic control field gives coherences between 0.05 and
0.09 with random orientations, so the functional separates the two cases cleanly.

Figures for human review
========================

Figures let people spot what no criterion anticipated. They never serve as an acceptance criterion.
A problem a figure reveals leads to a new pre-registered criterion, not to a reference image.

A run saves the maps it computes when asked to, with ``--save-maps DIR`` on the command line or
``save_maps=DIR`` in :func:`spatialize.scenarios.run` (:ref:`scenarios-maps`). For each scenario,
estimator and replicate field, it writes the arrays together with a figure showing the truth with its
sample locations next to the estimated map on one colour scale. The declared and the measured
orientations are drawn through the centre, with the functionals in the titles. A summary figure
shows all fields at once. The figures use the maps the checks were computed from in the same run,
without recomputing anything, so a figure always matches the report line it illustrates.

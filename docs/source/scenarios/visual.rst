.. _scenarios-visual:

################
Visual criteria
################

.. currentmodule:: spatialize.scenarios.stats.maps

Some properties of an estimate are what a practitioner *sees* in a map: an anisotropy that is
clearly visible, blocks left by axis-aligned partitions, a boundary that is sharp or diffuse,
high values that connect into bodies. The suite tests them without comparing pixels — two correct
implementations produce different pixels — by turning each one into a **map functional**: a number
computed from a gridded map, with a threshold or relation **fixed before any reference run**
(pre-registered in the scenario file and versioned; never tuned to make a run pass). The
functional is then tested like any other check, across replicate fields.

Which map is read
=================

Every criterion names its map, because the readings of the estimator look different:

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
     - blocks and terraces aligned with the axes
   * - contrast ratio (:func:`contrast_ratio`)
     - standard deviation of the map over that of the truth
     - washed-out maps
   * - level-set IoU (:func:`level_set_iou`)
     - intersection over union of a thresholded map and the true region
     - shapes of plumes, dry regions, exceeded areas
   * - planned
     - directional range ratio, roughness, edge sharpness, connectivity (components and Euler
       characteristic over thresholds), halo index, level-curve shape of an estimated covariance
     - see the catalogue

*Calibration of orientation and coherence* (scenario :ref:`S03 <scenario-S03>`, 40×40 grid,
:math:`\sigma = 2`): the six truth fields of an anisotropy at 30° give orientations 26–29° and
coherences 0.70–0.76; an isotropic control field gives coherences 0.05–0.09 and random
orientations. The functional separates the two cases cleanly.

Figures for human review
========================

Figures are meant for people, to spot what no criterion anticipated. They are never an acceptance
criterion: when a figure reveals a problem, the remedy is a new pre-registered criterion, not a
reference image.

A run saves the maps it computes when asked to (``--save-maps DIR`` on the command line,
``save_maps=DIR`` in :func:`spatialize.scenarios.run`; see :ref:`scenarios-maps`). For each scenario,
estimator and replicate field it writes the arrays and a figure of the truth (with the sample
locations) next to the estimated map, on one colour scale, with the declared orientation (white)
and the measured one (red) drawn through the centre and the functionals in the titles; a summary
figure shows all fields at once. The maps are the ones the checks were computed from, in the same
run — nothing is recomputed — so a figure always matches the report line it illustrates.

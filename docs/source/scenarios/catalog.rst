.. _scenarios-catalog:

#########
Catalogue
#########

The catalogue lists the situations Spatialize is held to, organised in three tiers, the law of the
partition process (T1), propositions about the estimator (T2) and complete geostatistical situations
(T3), plus the visual criteria attached to the scenarios. Each tier's page describes every scenario
in full, with what it claims, where the claim comes from, the truth, the data, the estimators, each
pre-registered check with its test family, how its targets are derived, what it depends on and, for
implemented scenarios, its results and history.

Each implemented scenario lives in ``spatialize/scenarios/catalog/<id>/``, holding its normative
definition in a ``scenario.yaml`` descriptor (:doc:`file_format`) next to its pinned data,
``data/*.npy`` files with a ``CHECKSUMS.sha256`` manifest.

.. toctree::
   :maxdepth: 2

   catalog_encoder_law
   catalog_estimator_properties
   catalog_geostatistical
   catalog_visual_criteria

Status of every scenario
========================

The table gives every scenario of the specification, with what it needs before it can be
implemented. Since the theory is still a draft, target values taken from its examples are derived
independently before a scenario is implemented, and computed by the evaluator where possible.

- **implemented** — in the catalogue and run on every change;
- **ready** — needs only new evaluators or generators;
- **negative control** — its target needs a process Spatialize does not implement, so on Spatialize
  it can only be a test expected to reject;
- **blocked / partly ready** — needs a feature not yet in Spatialize or an external baseline.

.. list-table::
   :header-rows: 1
   :widths: 7 45 18 30

   * - id
     - claim
     - status
     - depends on
   * - 
     - **T1 — encoder law**
     - 
     - 
   * - :ref:`E1 <scenario-E1>`
     - exact law of the partition of four points on a line
     - **implemented**
     - —
   * - :ref:`E2 <scenario-E2>`
     - pair co-occurrence of the Mondrian process
     - **implemented**
     - —
   * - :ref:`E3 <scenario-E3>`
     - co-occurrence of three points
     - **implemented**
     - —
   * - :ref:`E4 <scenario-E4>`
     - Poisson–Voronoi co-occurrence (isotropy in 2D, decay in 1D)
     - needs a reading method
     - a single datum gives a single Voronoi nucleus, so E2's reading does not apply
   * - :ref:`E5 <scenario-E5>`
     - fourth joint cumulant of a block-mark field on a line
     - **implemented**
     - —
   * - :ref:`E6 <scenario-E6>`
     - conditional covariance under one uniform cut
     - negative control
     - targets on the theory's Mondrian; derive the value independently
   * - 
     - **T2 — estimator properties**
     - 
     - 
   * - :ref:`P1 <scenario-P1>`
     - spread across ensembles decreases as :math:`T^{-1/2}`
     - ready
     - —
   * - :ref:`P2 <scenario-P2>`
     - weighted-draw decoder: draws are data values; their mean is the IDW estimate
     - **implemented**
     - —
   * - :ref:`P3 <scenario-P3>`
     - draw decoder: frequencies match the cell's values
     - **implemented**
     - —
   * - :ref:`P4 <scenario-P4>`
     - estimated CDFs are monotone and within [0, 1]
     - ready
     - —
   * - :ref:`P5 <scenario-P5>`
     - weights on the data plus residual weight sum to 1
     - blocked
     - empty-cell policy and its diagnostic (phase 3)
   * - :ref:`P6 <scenario-P6>`
     - the law at a location does not depend on the other queries
     - **implemented**
     - —
   * - :ref:`P7 <scenario-P7>`
     - covariance of an uncorrelated field against its closed form
     - ready
     - derive the closed form independently
   * - :ref:`P8 <scenario-P8>`
     - empty cells share one mark
     - blocked
     - empty-cell policy (phase 3)
   * - :ref:`P9 <scenario-P9>`
     - cell-weighted marks are unbiased under preferential sampling
     - blocked
     - empty-cell policy (phase 3)
   * - 
     - **T3 — geostatistical scenarios**
     - 
     - 
   * - :ref:`S01 <scenario-S01>`
     - simulation keeps the geometry of the field
     - ready
     - —
   * - :ref:`S02 <scenario-S02>`
     - error stops decreasing beyond some ensemble size
     - ready
     - —
   * - :ref:`S03 <scenario-S03>`
     - anisotropic field (visual criterion V1)
     - **implemented**
     - new decoders are added as they land
   * - :ref:`S04 <scenario-S04>`
     - zero-inflated field
     - partly ready
     - draw decoders (phase 2) for its main claims
   * - :ref:`S05 <scenario-S05>`
     - heavy-tailed field
     - partly ready
     - ordinary kriging baseline (external dependency)
   * - :ref:`S06 <scenario-S06>`
     - non-stationary field, order relations of the estimated law
     - partly ready
     - indicator kriging baseline (external dependency)
   * - :ref:`S07 <scenario-S07>`
     - exceedance areas
     - ready
     - derive the area-variance identity independently
   * - :ref:`S08 <scenario-S08>`
     - support effect on tonnage curves
     - ready
     - —
   * - :ref:`S09 <scenario-S09>`
     - resource categories under preferential sampling
     - ready
     - preferential design generator
   * - :ref:`S10 <scenario-S10>`
     - optimal quantile levels
     - ready
     - —
   * - :ref:`S11 <scenario-S11>`
     - granularity and covariance
     - ready
     - —
   * - :ref:`S12 <scenario-S12>`
     - edge cases: few data, duplicates, queries on data and outside the data box
     - **implemented**
     - —
   * - :ref:`S13 <scenario-S13>`
     - connectivity of high-value bodies
     - ready
     - connectivity functional
   * - 
     - **V — visual criteria**
     - 
     - 
   * - V1
     - anisotropy visible (S03)
     - **implemented**
     - —
   * - V2
     - anisotropy against a fitted isotropic kriging (S03)
     - blocked
     - ordinary kriging baseline (external dependency)
   * - V3
     - anisotropy in members and simulations (S03)
     - ready
     - —
   * - V4
     - Mondrian blocks averaged away (S03); S11
     - **implemented** on S03
     - S11 for the coarse-partition case
   * - V5
     - contrast: maps not more washed out than the best linear predictor (S03); the "false cure"
       of coarse partitions (S11)
     - **implemented** on S03
     - S11 for the "false cure"
   * - V6
     - sharp dry-region boundaries (S04)
     - blocked
     - draw decoders (phase 2)
   * - V7
     - halos around extreme values (S05)
     - partly ready
     - ordinary kriging baseline for the halo comparison
   * - V8
     - exceedance regions (S07)
     - ready
     - —
   * - V9
     - roughness against a global simple-kriging reference
     - ready
     - reference computed in numpy
   * - V10
     - connectivity of high-value bodies (S13)
     - ready
     - connectivity functional
   * - V11
     - shape of estimated covariance level curves
     - ready
     - —


Example run
===========

The excerpt below comes from a run in ``ci`` mode with seed 12345, which takes about seven minutes
on a multi-core machine.

.. code-block:: text

   runner=spatialize mode=ci seed=12345 α_suite=0.001 (Holm)
   scenario/check                               family          p-value     level  expect result
   E2-mondrian-pair-cooccurrence/c1-idw         gof-closed     2.43e-87   3.1e-05  reject PASS  — k=8 N=3100 ...
   E2-mondrian-pair-cooccurrence/c1-idw-theory  gof-closed        0.212   0.00025  pass   PASS  — k=8 N=3100 ...
   S03-anisotropic-field/v1a                    equivalence    1.09e-42   6.3e-05  pass   PASS  — mean=-0.2803 se=0.351 ...
   S03-anisotropic-field/v1b                    one-sided      1.07e-20   0.00014  pass   PASS  — mean=0.6084 se=0.00868 ...
   ...
   S03-anisotropic-field/v1a-vor-uniform-aidw   equivalence    2.34e-50   4.2e-05  pass   PASS  — mean=-1.048 se=0.254 ...
   ...
   S03-anisotropic-field/v4a-idw                one-sided      1.01e-17   0.00017  pass   PASS  — mean=0.3424 se=0.0145 ...
   S03-anisotropic-field/v4b-idw                one-sided      5.03e-56     4e-05  pass   PASS  — mean=2.554 se=0.0609 ...
   ...
   S03-anisotropic-field/v4c-vor-uniform-krig   equivalence    8.78e-63   3.6e-05  pass   PASS  — mean=-0.0167 se=0.0094 ...
   ...
   S03-anisotropic-field/v5-idw                 one-sided             1     0.001  pass   KNOWN  — mean=0.8512 se=0.00422 ...
   ...
   S03-anisotropic-field/v5-krig                one-sided      1.01e-43   5.9e-05  pass   PASS  — mean=0.9575 se=0.00201 ...
   S03-anisotropic-field/v5-vor-uniform         one-sided         0.896   0.00033  pass   KNOWN  — mean=0.8954 se=0.0036 ...
   ...
   S12-edge-cases/e1-m-idw                      almost-sure           1     exact  pass   PASS  — 0 violations ...
   ...
   S12-edge-cases/e4-vd-idw                     almost-sure           1     exact  pass   PASS  — 0 violations ...
   ...

   62 passed, 0 failed, 3 known failures, 0 skipped (reproduce with --mode ci --seed 12345)
   maps saved under /Users/alvaro/tmp/spatialize-scenario-maps

The ``level`` column is the Holm level each p-value was compared with (:doc:`statistics`).
``exact`` marks almost-sure checks, which spend no error budget, and ``KNOWN`` a recorded known
failure (:doc:`statistics`).


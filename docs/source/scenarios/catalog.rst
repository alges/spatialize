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
     - Poisson–Voronoi co-occurrence (decay in 1D, isotropy in 2D as E4b)
     - **implemented**
     - —
   * - :ref:`E5 <scenario-E5>`
     - fourth joint cumulant of a block-mark field on a line
     - **implemented**
     - —
   * - :ref:`E6 <scenario-E6>`
     - conditional covariance under one uniform cut, through the Mondrian process and the partition of version 1.2
     - **implemented**
     - —
   * - 
     - **T2 — estimator properties**
     - 
     - 
   * - :ref:`P1 <scenario-P1>`
     - spread across ensembles decreases as :math:`T^{-1/2}`
     - **implemented**
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
     - **implemented**
     - —
   * - :ref:`P5 <scenario-P5>`
     - weights on the data plus residual weight sum to 1
     - **implemented**
     - —
   * - :ref:`P6 <scenario-P6>`
     - the law at a location does not depend on the other queries
     - **implemented**
     - —
   * - :ref:`P7 <scenario-P7>`
     - covariance of an uncorrelated field against its closed form
     - **implemented**
     - —
   * - :ref:`P8 <scenario-P8>`
     - empty cells share one mark
     - **implemented**
     - —
   * - :ref:`P9 <scenario-P9>`
     - cell-weighted marks lie closer to the spatial law under preferential sampling
     - **implemented**
     - —
   * - :ref:`P10 <scenario-P10>`
     - each empty-cell policy does what it declares
     - **implemented**
     - —
   * - :ref:`P11 <scenario-P11>`
     - a score that drops undefined members leaves out the hardest data
     - **implemented**
     - —
   * - :ref:`P12 <scenario-P12>`
     - posterior analysis finds planted errors and declusters a preferential design
     - **implemented**
     - the false discovery rate on clean fields (known failure)
   * - 
     - **T3 — geostatistical scenarios**
     - 
     - 
   * - :ref:`S01 <scenario-S01>`
     - simulation keeps the geometry of the field
     - ready · priority 4
     - runner method for simulated fields
   * - :ref:`S02 <scenario-S02>`
     - error stops decreasing beyond some ensemble size
     - ready · priority 1
     - Mondrian block-mark generator
   * - :ref:`S03 <scenario-S03>`
     - anisotropic field (visual criteria V1, V4 and V5, the order of the decoders)
     - **implemented**
     - kriging baselines for V2 and the comparison with ordinary kriging
   * - :ref:`S04 <scenario-S04>`
     - zero-inflated field
     - ready · priority 2
     - zero-inflated mark sampler; edge-sharpness functional (V6)
   * - :ref:`S05 <scenario-S05>`
     - heavy-tailed field
     - partly ready · priority 9
     - ordinary-kriging baseline (V7), writable in NumPy
   * - :ref:`S06 <scenario-S06>`
     - non-stationary field, order relations of the estimated law
     - partly ready · priority 10
     - indicator-kriging baseline (negative control), writable in NumPy
   * - :ref:`S07 <scenario-S07>`
     - exceedance areas
     - ready · priority 8
     - closed form of the area variance; simulated fields (S01)
   * - :ref:`S08 <scenario-S08>`
     - support effect on tonnage curves
     - ready · priority 3
     - tonnage functional
   * - :ref:`S09 <scenario-S09>`
     - resource categories under preferential sampling
     - ready · priority 7
     - preferential-design generator; classification rule
   * - :ref:`S10 <scenario-S10>`
     - optimal quantile levels
     - ready · priority 6
     - loss-optimal quantile readings
   * - :ref:`S11 <scenario-S11>`
     - granularity and covariance
     - ready · priority 5
     - covariance-shape reading (V11)
   * - :ref:`S12 <scenario-S12>`
     - edge cases: few data, duplicates, queries on data and outside the data box
     - **implemented**
     - —
   * - :ref:`S13 <scenario-S13>`
     - connectivity of high-value bodies
     - ready · priority 11
     - connectivity functional; elongated-body generator; simulated fields (S01)
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
     - blocked · priority 8
     - ordinary-kriging baseline with a fitted isotropic variogram, writable in NumPy
   * - V3
     - anisotropy in members and simulations (S03)
     - **implemented** on members · priority 1 for simulated fields
     - simulated fields (S01)
   * - V4
     - Mondrian blocks averaged away (S03); S11
     - **implemented** on S03 · priority 4 on S11
     - S11
   * - V5
     - contrast: maps not more washed out than the best linear predictor (S03); the "false cure"
       of coarse partitions (S11)
     - **implemented** on S03 · priority 5 on S11
     - S11
   * - V6
     - sharp dry-region boundaries (S04)
     - ready · priority 3
     - edge-sharpness functional (with S04)
   * - V7
     - halos around extreme values (S05)
     - partly ready · priority 9
     - halo index; ordinary-kriging baseline (V2)
   * - V8
     - exceedance regions (S07)
     - ready · priority 7
     - pre-set IoU level (with S07)
   * - V9
     - roughness against a global simple-kriging reference (S03)
     - **implemented**
     - —
   * - V10
     - connectivity of high-value bodies (S13)
     - ready · priority 10
     - connectivity functional (with S13)
   * - V11
     - shape of estimated covariance level curves
     - ready · priority 6
     - covariance-shape reading (with S11)


Example run
===========

The excerpt below comes from a run in ``ci`` mode with seed 12345, which takes about thirty minutes
on a multi-core machine, most of them in S03, and saves its maps for review.

.. code-block:: text

   runner=spatialize mode=ci seed=12345 α_suite=0.001 (Holm)
   scenario/check                                  family          p-value     level  expect result
   E2-mondrian-pair-cooccurrence/c1-idw            gof-closed     2.43e-87   1.4e-05  reject PASS  — k=8 N=3100 ...
   E2-mondrian-pair-cooccurrence/c1-idw-theory     gof-closed        0.212     5e-05  pass   PASS  — k=8 N=3100 ...
   S03-anisotropic-field/v1a                       equivalence    1.09e-42   2.3e-05  pass   PASS  — mean=-0.2803 se=0.351 ...
   S03-anisotropic-field/v1b                       one-sided      1.07e-20   3.3e-05  pass   PASS  — mean=0.6084 se=0.00868 ...
   ...
   S03-anisotropic-field/v1a-vor-uniform-aidw      equivalence    6.78e-51   1.8e-05  pass   PASS  — mean=-1.086 se=0.249 ...
   ...
   S03-anisotropic-field/v4a-idw                   one-sided      1.01e-17   3.6e-05  pass   PASS  — mean=0.3424 se=0.0145 ...
   S03-anisotropic-field/v4b-idw                   one-sided      5.03e-56   1.8e-05  pass   PASS  — mean=2.554 se=0.0609 ...
   ...
   S03-anisotropic-field/v4c-vor-uniform-krig      equivalence    8.78e-63   1.6e-05  pass   PASS  — mean=-0.0167 se=0.0094 ...
   ...
   S03-anisotropic-field/v5-idw                    one-sided             1     0.001  pass   KNOWN  — mean=0.8512 se=0.00422 ...
   S03-anisotropic-field/v5-krig                   one-sided      1.01e-43   2.2e-05  pass   PASS  — mean=0.9575 se=0.00201 ...
   S03-anisotropic-field/v5-vor-uniform            one-sided         0.896    0.0001  pass   KNOWN  — mean=0.8954 se=0.0036 ...
   ...
   S03-anisotropic-field/v1b-draw                  one-sided             1   0.00017  pass   KNOWN  — mean=0.406 se=0.0162 ...
   ...
   S03-anisotropic-field/v4a-sharp                 one-sided       6.3e-15   3.7e-05  pass   PASS  — mean=0.293 se=0.0219 ...
   S03-anisotropic-field/order-cov-sharp           paired-relation   1.63e-43   2.3e-05  pass   PASS  — paired t: mean=0.0261 ...
   S03-anisotropic-field/order-cov-wdraw           paired-relation   6.21e-73   1.5e-05  pass   PASS  — paired t: mean=0.1278 ...
   S03-anisotropic-field/order-cov-draw            paired-relation   4.93e-75   1.4e-05  pass   PASS  — paired t: mean=0.157 ...
   S03-anisotropic-field/order-rmse-sharp          paired-relation      0.989   0.00014  pass   KNOWN  — paired t: mean=-0.001145 ...
   ...
   S12-edge-cases/e1-m-idw                         almost-sure           1     exact  pass   PASS  — 0 violations ...
   ...
   S12-edge-cases/e4-vd-idw                        almost-sure           1     exact  pass   PASS  — 0 violations ...
   ...

   161 passed, 0 failed, 6 known failures, 0 skipped (reproduce with --mode ci --seed 12345)
   maps saved under /Users/alvaro/tmp/spatialize-scenario-maps

The ``level`` column is the Holm level each p-value was compared with (:doc:`statistics`).
``exact`` marks almost-sure checks, which spend no error budget, and ``KNOWN`` a recorded known
failure (:doc:`statistics`).


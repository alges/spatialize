.. _scenarios-catalog-visual:

####################################
Visual criteria in the catalogue
####################################

What a practitioner *sees* in a map — a visible anisotropy, blocks along the axes, a sharp boundary,
halos around extremes, connected bodies — is tested through pre-registered map functionals
(:doc:`visual`). Each criterion belongs to a scenario, where it is described in full.

.. list-table::
   :header-rows: 1
   :widths: 6 24 10 40 20

   * - id
     - claim
     - scenario
     - criterion
     - status
   * - V1
     - anisotropy direction and strength visible in the median map
     - :ref:`S03 <scenario-S03>`
     - orientation (TOST ±10°) and coherence ratio (> 0.5)
     - **implemented**
   * - V2
     - the ensemble keeps the short axis that a fitted isotropic kriging blurs
     - :ref:`S03 <scenario-S03>`
     - coherence of the ensemble map > that of ordinary kriging with a fitted isotropic variogram (paired)
     - blocked (ordinary-kriging baseline) · priority 8
   * - V3
     - the anisotropy is in the fields, not only in the mean
     - :ref:`S03 <scenario-S03>`
     - coherence and range ratio of members and simulated fields equivalent to the truth's
     - ready · priority 1
   * - V4
     - Mondrian blocks averaged away
     - :ref:`S03 <scenario-S03>`, :ref:`S11 <scenario-S11>`
     - axis-locking of the median map below 0.5, with a power check on single members and a Voronoi calibration
     - **implemented** on S03; S11 · priority 4
   * - V5
     - maps not washed out; the "false cure" of coarse partitions
     - :ref:`S03 <scenario-S03>`, :ref:`S11 <scenario-S11>`
     - contrast against simple kriging with the true covariance above 0.9 (S03); artefacts shrink without RMSE improving (S11)
     - **implemented** on S03; S11 · priority 5
   * - V6
     - sharp dry-region boundaries
     - :ref:`S04 <scenario-S04>`
     - IoU of :math:`p_{dry}>1/2` with the dry region and edge sharpness, weighted draw > draw
     - ready · priority 3
   * - V7
     - no halos around extreme values
     - :ref:`S05 <scenario-S05>`
     - halo index of ordinary kriging negative around top-decile cells; ensemble maps never below the data minimum
     - partly ready (kriging baseline) · priority 9
   * - V8
     - exceedance regions
     - :ref:`S07 <scenario-S07>`
     - IoU of :math:`P(Z>\ell)>1/2` with the true exceeded region above a pre-set level; member regions bracket the true area
     - ready · priority 7
   * - V9
     - roughness
     - S12 design
     - ensemble roughness at most that of a global simple-kriging reference; cropped-neighbourhood kriging rougher (control)
     - ready · priority 2
   * - V10
     - connectivity of high-value bodies
     - :ref:`S13 <scenario-S13>`
     - Euler-characteristic curves of members and simulated fields equivalent to the truth's; the mean map departs
     - ready (connectivity functional) · priority 10
   * - V11
     - shape of the estimated covariance
     - :ref:`S11 <scenario-S11>`
     - iso-lines diamond-shaped with coarse partitions, closer to circular with fine ones
     - ready · priority 6

Pending criteria by priority
============================

The order puts first the criteria of scenarios already implemented, S03 and S12, whose fields and
evaluators exist. The others follow the priority of their scenario (:doc:`catalog_geostatistical`),
the criteria needing a kriging baseline coming last but one.

#. **V3** (S03), the anisotropy in members and simulated fields.

   - The coherence of single members, read with the existing ``orientation_coherence``, against the
     truth's, by equivalence (TOST) with a margin fixed before the first run.
   - A directional range ratio functional, the ratio of the ranges of the empirical variogram along
     and across the anisotropy's direction, computed on the grid in NumPy.
   - The simulated fields wait for the runner method for simulated fields of S01. The members' part
     can be added first.

#. **V9** (S12), roughness.

   - A roughness functional, for instance the mean squared difference between neighbouring grid
     cells relative to the map's variance.
   - The global simple-kriging reference, which the suite computes
     (``simple_kriging_exponential``).
   - The control, a simple kriging restricted to a cropped neighbourhood of each location, written in
     NumPy, which must come out rougher.

#. **V6** (S04), sharp dry-region boundaries, with S04.

   - The overlap through the existing ``level_set_iou``. The draw decoders are in the catalogue.
   - An edge-sharpness functional, such as the mean gradient of :math:`p_{dry}` across the true dry
     boundary.

#. **V4** on S11, Mondrian blocks averaged away with coarse partitions, with S11.

   - The existing ``axis_lock`` on S11's two granularities, with the power check on single members and
     the Voronoi calibration of S03.

#. **V5** on S11, the "false cure", with S11.

   - The existing ``axis_artifact_index`` and ``contrast_ratio`` at 8 and 200 members, with the RMSE
     of S11. The criterion holds when the artefacts shrink while the RMSE does not improve.

#. **V11** (S11), the shape of the estimated covariance, with S11.

   - The covariance of the members at grid offsets, its iso-lines read through
     ``axis_diagonal_log_ratio`` applied to that covariance. The expected readings are diamonds
     (:math:`\ell_1`) with coarse partitions and near-circles with fine ones.

#. **V8** (S07), exceedance regions, with S07.

   - The existing ``level_set_iou`` of :math:`P(Z > \ell) > 1/2` with the true exceeded region, its
     pre-set level fixed before the first run.
   - The bracketing of the true area by the members' exceeded areas, shared with S07's coverage
     check.

#. **V2** (S03), the anisotropy against a fitted isotropic kriging.

   - An ordinary-kriging baseline with an isotropic variogram fitted to the data, by a rule fixed
     before the first run. It can be written in NumPy, as the suite's simple kriging is, and serves V7
     too.
   - The coherence of the ensemble's map against the baseline's, paired over S03's 80 fields.

#. **V7** (S05), no halos around extreme values, with S05.

   - A halo index, the mean of the map minus the data minimum in a ring around the top-decile cells,
     negative when a map dips below the data.
   - The ordinary-kriging baseline of V2.

#. **V10** (S13), the connectivity of high-value bodies, with S13.

   - The Euler-characteristic functional over thresholds and the generator of elongated bodies of
     S13, with the runner method for simulated fields of S01.

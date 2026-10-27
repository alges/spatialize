.. _scenarios-catalog-visual:

####################################
Visual criteria in the catalogue
####################################

The features a practitioner *sees* in a map — a visible anisotropy, blocks along the axes, a sharp
boundary, halos around extremes, connected bodies — are tested through pre-registered map functionals
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
     - **implemented** (adaptive IDW passes; IDW and kriging within the cell known failures)
   * - V3
     - the anisotropy is in the fields, not only in the mean
     - :ref:`S03 <scenario-S03>`
     - coherence and range ratio of members and simulated fields equivalent to the truth's
     - **implemented** on members (adaptive decoders pass, four known failures) and on simulated fields (adaptive IDW keeps the ranges, three known failures)
   * - V4
     - Mondrian blocks averaged away
     - :ref:`S03 <scenario-S03>`, :ref:`S11 <scenario-S11>`
     - axis-locking of the median map below 0.5, with a power check on single members and a Voronoi calibration (S03); stronger with coarse partitions than with fine ones (S11)
     - **implemented**
   * - V5
     - maps not washed out; the "false cure" of coarse partitions
     - :ref:`S03 <scenario-S03>`, :ref:`S11 <scenario-S11>`
     - contrast against simple kriging with the true covariance above 0.9 (S03); with coarse partitions the axis-locking falls from 8 to 200 members while the error stays above that of fine partitions with 8, whose contrast ratio exceeds 0.8 (S11)
     - **implemented**
   * - V6
     - sharp dry-region boundaries
     - :ref:`S04 <scenario-S04>`
     - IoU of :math:`p_{dry}>1/2` with the dry region and edge sharpness, weighted draw > draw
     - **implemented** on S04
   * - V7
     - no halos around extreme values
     - :ref:`S05 <scenario-S05>`
     - share of a 3-pixel ring around the top-decile cells where ordinary kriging falls below the smallest datum above 0 and above the share beyond the ring; members of the averaging and drawing decoders never below it
     - **implemented**
   * - V8
     - exceedance regions
     - :ref:`S07 <scenario-S07>`
     - IoU of :math:`P(Z>\ell)>1/2` with the true exceeded region above 0.5; the members' band of the area covers the true area (S07's coverage)
     - **implemented**
   * - V9
     - roughness
     - :ref:`S03 <scenario-S03>`
     - map roughness at most that of global simple kriging for the averaging decoders, above it by design for the adaptive ones; simple kriging from the 8 nearest data rougher (control)
     - **implemented**
   * - V10
     - connectivity of high-value bodies
     - :ref:`S13 <scenario-S13>`
     - Euler curve of the members of the weighted draw coupled by cell equivalent to the truth's (within 2) and closer to it than the map of means; simulated fields break the bodies
     - **implemented**
   * - V11
     - shape of the estimated covariance
     - :ref:`S11 <scenario-S11>`
     - the median map's covariance more diamond-shaped with coarse partitions than with fine ones
     - **implemented**

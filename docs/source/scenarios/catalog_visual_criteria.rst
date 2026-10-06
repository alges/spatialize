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
     - blocked (ordinary-kriging baseline)
   * - V3
     - the anisotropy is in the fields, not only in the mean
     - :ref:`S03 <scenario-S03>`
     - coherence and range ratio of members and simulated fields equivalent to the truth's
     - ready
   * - V4
     - Mondrian blocks averaged away
     - :ref:`S03 <scenario-S03>`, :ref:`S11 <scenario-S11>`
     - axis-locking of the median map below 0.5, with a power check on single members and a Voronoi calibration
     - **implemented** on S03
   * - V5
     - maps not washed out; the "false cure" of coarse partitions
     - :ref:`S03 <scenario-S03>`, :ref:`S11 <scenario-S11>`
     - contrast against simple kriging with the true covariance above 0.9 (S03); artefacts shrink without RMSE improving (S11)
     - **implemented** on S03
   * - V6
     - sharp dry-region boundaries
     - :ref:`S04 <scenario-S04>`
     - IoU of :math:`p_{dry}>1/2` with the dry region and edge sharpness, weighted draw > draw
     - blocked (draw decoders)
   * - V7
     - no halos around extreme values
     - :ref:`S05 <scenario-S05>`
     - halo index of ordinary kriging negative around top-decile cells; ensemble maps never below the data minimum
     - partly ready (kriging baseline)
   * - V8
     - exceedance regions
     - :ref:`S07 <scenario-S07>`
     - IoU of :math:`P(Z>\ell)>1/2` with the true exceeded region above a pre-set level; member regions bracket the true area
     - ready
   * - V9
     - roughness
     - S12 design
     - ensemble roughness at most that of a global simple-kriging reference; cropped-neighbourhood kriging rougher (control)
     - ready
   * - V10
     - connectivity of high-value bodies
     - :ref:`S13 <scenario-S13>`
     - Euler-characteristic curves of members and simulated fields equivalent to the truth's; the mean map departs
     - ready (connectivity functional)
   * - V11
     - shape of the estimated covariance
     - :ref:`S11 <scenario-S11>`
     - iso-lines diamond-shaped with coarse partitions, closer to circular with fine ones
     - ready

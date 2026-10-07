.. _theory-esi:

###############################
Ensemble spatial interpolation
###############################

Ensemble spatial interpolation (ESI) is the estimator of :doc:`sde` read as an interpolator, the
use most users meet first. Given data at scattered locations, it returns at each location of a grid
or of a list the members of the ensemble, from which every reading follows.

Maps
====

The map is one reading of the members at each location, the mean by default. The median suits
skewed variables and is the safer report for heavy-tailed ones. A quantile suits a use where
over- and under-prediction cost differently (:doc:`sde`). No variogram is needed, since the partition supplies the spatial coupling while the decoder adapts to the data inside each cell.

Uncertainty
===========

The same members give the uncertainty of the map, location by location.

- The spread of the members around the reported value measures how much the estimate depends on
  the partition.
- The share of members above a threshold estimates the probability of exceeding it.
- Two quantiles of the members bound an interval.

With an averaging decoder these readings capture the uncertainty that comes from the partition and
understate the rest, so intervals read from the raw members are too narrow. A drawing decoder
(:doc:`decoders`) or the widened laws of simulation (:doc:`ess`) correct for it.

Coherence
=========

Every reading comes from one ensemble, so the readings at a location agree with one another. The
probability of exceeding a threshold never rises with the threshold. An interval never inverts.
Across locations, the members are fields, so a question about many locations at once, an area or a
block average, is answered from the members directly.

Variables
=========

The same construction handles continuous variables in any dimension the decoder supports, and
categorical variables, whose decoders are classifiers fitted inside each cell (:doc:`other`).

In Spatialize
=============

:func:`~spatialize.gs.esi.esi_griddata` estimates on a grid and
:func:`~spatialize.gs.esi.esi_nongriddata` at a list of locations. Their result holds the members
(``esi_samples``), the reported map (``estimation``) and a measure of its uncertainty
(``precision``), with plotting helpers (:doc:`../reference/esi`).

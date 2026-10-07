.. _theory-ess:

###############################
Ensemble spatial simulation
###############################

Simulation draws whole fields that honour the estimated laws, for questions whose answer needs many
plausible versions of the field, a flow through a porous medium, a mine plan tested against grade
uncertainty, a risk computed over a region.

Fields from marginals
=====================

Ensemble spatial simulation (ESS) fits a law to the members at each location, then draws every
location independently from its own law. Drawing independently sounds like a recipe for noise, yet
the fields it produces carry the geometry of the variable. Neighbouring locations are estimated
from almost the same data, so their laws almost coincide. Two independent draws from two nearly
equal laws come back nearly equal. The structure lies in the slow drift of the laws across space,
not in the draws.

The draws carry the family of local laws, not the full joint law. The members of the
ensemble carry more, since each is a field computed under one partition, with the dependence
between locations the partition induces. For a question about many locations at once, such as the
area above a limit, the members give the right spread of the answer, while independently simulated
fields understate it.

Widening
========

The members at a location vary only as far as the cells around it vary, so their spread understates
how much the variable changes near that location. The missing part is the variance of the local
means, which the ensemble averages out. Simulation therefore widens each local law to a target
variance, read from the data nearest to the location. Each member is replaced by a small law around
it, chosen in a family suited to the variable, Gamma laws for positive variables and skew-normal
laws otherwise, so that the mixture of these laws keeps the mean of the members and reaches the
target variance. A location whose members already spread enough is left as it is.

A drawing decoder (:doc:`decoders`) puts the dispersion of the cells into the members themselves,
so it needs less widening.

Fitting the local laws
======================

The law at each location is estimated from its members, by a kernel density estimate or by a
mixture of Gaussians. The kernel estimate follows the members closely. A mixture summarises them
with a few components, which helps when the members show a few distinct values.

In Spatialize
=============

:func:`~spatialize.gs.ess.ess_sample` simulates from the result of an ESI estimation. The fitted laws
and the widening are set by :class:`~spatialize.empirical.FittedModelFactory`, whose
``point_model_name`` chooses the density (``"kde"``, ``"emm"``, ``"vim"``) and whose ``widening``
and ``widening_knn`` set the widening and the number of nearest data its target variance is read
from (:doc:`../reference/ess`, :doc:`../reference/empirical`).

.. _theory-other:

#################
Further readings
#################

The ensemble supports other readings, each with its own module. The analysis of the data against
the laws the other data give has its own page (:doc:`posterior`), and so do categorical variables
(:doc:`categorical`).

Information measures
====================

The differential entropy of the predictive law at a location,

.. math::

   h(v) = -\int \hat f_v(z) \log \hat f_v(z)\, dz,

summarises its uncertainty in one number without assuming the law is symmetric or unimodal. For two
variables :math:`U` and :math:`V` measured over the same region, the mutual information of their
local laws,

.. math::

   I(U; V)(x) = h_U(x) + h_V(x) - h_{U,V}(x),

measures how much knowing one tells about the other, location by location, and vanishes when they
are independent there. Spatialize estimates the joint density of the members of :math:`U` and :math:`V` with Mondrian
partitions of the plane of values, uniform within each cell, and integrates it over one variable to
obtain each marginal,

.. math::

   \hat f_U(u) = \sum_{c\,:\,u \in c_U} \frac{p_c}{|c_U|},

with :math:`p_c` the share of members in cell :math:`c` and :math:`c_U` its side along :math:`U`. The
three entropies then come from one density, so the estimate is never negative. Members of cells without data, undefined under ``empty_cells="nan"`` (:doc:`blockmark`), carry no
value and are left out, the shares :math:`p_c` being taken over the valid members. A location with
fewer than two valid members gets no entropy (NaN). For the mutual information, a pair counts only
when both of its members are valid.

Spatialize reads these laws in two ways. The entropy can come from the members themselves, their
density estimated with random partitions of their range (``method="partition"``), or from the law
ensemble spatial simulation fits at each location, widened to the spread of the nearest data
(``method="ess"``, :doc:`ess`). With an averaging decoder the members' spread follows the granularity,
so only the second follows the field. A drawing decoder gives members whose spread is that of the data,
so both read it.

The mutual information needs the joint law of :math:`U` and :math:`V` at a location, which neither
ensemble gives, since each is built on its own data. The default (``method="colocated"``) pairs the
two variables where one of them is measured, the datum of one with the other predicted there by its
own ensemble, and weighs each pair, for a target location, by the share of partitions in which the
pair's location and the target share a cell, so the partitions act as adaptive neighbourhoods. The
mutual information is then that of a kernel density of the weighted pairs, minus that of the same pairs
with :math:`V` permuted, the bias of the density under independence. The pairs hold the variables' own
relation, so independent variables give a value near zero and a dependence that is not monotone is
seen too. The alternative (``method="members"``) reads the members of both ensembles, drawn on shared
partitions, through random partitions of the plane of values. The members are averages, which blur the
relation of the variables, and they co-vary through the size of the cells they share whatever that
relation, so this estimate sees a link faintly and a non-monotone one hardly at all.

In Spatialize
=============

The entropy and mutual information are the estimators of
:doc:`../reference/futures`, where they are experimental: their interface and results may still
change.

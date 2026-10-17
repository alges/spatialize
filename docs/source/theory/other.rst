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

The members of :math:`U` and of :math:`V` at a location co-vary through the cells they share, so the
two ensembles are drawn on the same partitions, with one seed, one box holding both data sets and the
queries, and, for the Mondrian process, the granularity that gives both the same rate. With partitions
drawn independently the members would be independent whatever the dependence of the variables, and
the mutual information near zero. Voronoi partitions with nuclei among the data cannot be shared, so
they need ``data_cond=False``. A density estimated from a finite number of members puts a positive
mutual information even on independent variables, the more so the finer the partition of the values,
so the estimate subtracts, at each location, the mutual information of the same members with those of
:math:`V` permuted.

In Spatialize
=============

The entropy and mutual information are the estimators of
:doc:`../reference/futures`, where they are experimental: their interface and results may still
change.

.. _theory-other:

#################
Further readings
#################

The ensemble supports other readings, each with its own module. The analysis of the data against
the laws the other data give has its own page (:doc:`posterior`).

Categorical variables
=====================

A categorical variable :math:`Z(v) \in \{1, \dots, K\}`, a rock type or a land use, is estimated
with the same partitions and a classifier :math:`g` as decoder, fitted inside each cell. The members are categories, whose shares estimate the probability of each category,

.. math::

   \hat\pi_k(v) = \frac1T \sum_{t=1}^T \mathbf 1\{\hat z^{(t)}(v) = k\},
   \qquad \sum_k \hat\pi_k(v) = 1,

the reported category being the most frequent one. A nearest-neighbour classifier with an
anisotropic metric is built in, and any classifier with the scikit-learn interface can take its
place.

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

In Spatialize
=============

Categorical estimation is the functions of
:doc:`../reference/cat_esi`, and the entropy and mutual information the estimators of
:doc:`../reference/futures`, where they are experimental: their interface and results may still
change.

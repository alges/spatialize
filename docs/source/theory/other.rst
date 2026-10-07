.. _theory-other:

#################
Further readings
#################

The ensemble supports other readings, each with its own module.

Posterior analysis of the data
==============================

Cross-validation predicts each datum :math:`z_i` from the others, which gives at every datum a
predictive law :math:`\hat F_{-i}` built without it. The position of the datum in that law tells how surprising it is. Spatialize sorts the data by the central intervals of increasing mass
:math:`\alpha_1 < \alpha_2 < \dots` of :math:`\hat F_{-i}`,

.. math::

   I_{\alpha} = \big[\hat q_{(1-\alpha)/2},\ \hat q_{(1+\alpha)/2}\big],

placing each datum at the level of the widest interval that leaves it out. A datum outside even the
widest interval lies in the far tail of its law, which points to data that disagree with their
surroundings, such as transcription errors, a change of support or a genuinely anomalous place.

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
three entropies then come from one density, so the estimate is never negative.

In Spatialize
=============

Posterior analysis is :func:`~spatialize.gs.spa.cv_sample_pred_posterior` and its
``rank_samples`` (:doc:`../reference/spa`), categorical estimation the functions of
:doc:`../reference/cat_esi`, and the entropy and mutual information the estimators of
:doc:`../reference/esmi`.

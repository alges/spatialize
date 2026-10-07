.. _theory-other:

#################
Further readings
#################

The ensemble supports other readings, each with its own module.

Posterior analysis of the data
==============================

Cross-validation predicts each datum from the others, which gives at every datum a predictive law
built without it. Comparing the datum with that law tells how surprising the datum is, a reading
useful to find data that disagree with their surroundings, such as transcription errors, a change
of support or a genuinely anomalous place. Spatialize sorts the data into levels by how far into the tails of its law each datum falls, using
central intervals of increasing mass.

Categorical variables
=====================

A categorical variable, a rock type or a land use, is estimated with the same partitions and a
classifier as decoder, fitted inside each cell. The members at a location are categories, whose shares
estimate the probability of each category there. A nearest-neighbour classifier with an
anisotropic metric is built in, and any classifier with the scikit-learn interface can take its
place.

Information measures
====================

The entropy of the predictive law at a location summarises its uncertainty in one number that does
not assume the law is symmetric or unimodal. For two variables measured over the same region, the
mutual information between their local laws measures how much knowing one tells about the other,
location by location.

In Spatialize
=============

Posterior analysis is :func:`~spatialize.gs.spa.cv_sample_pred_posterior`
(:doc:`../reference/spa`), categorical estimation the functions of :doc:`../reference/cat_esi`, and
entropy and mutual information the estimators of :mod:`spatialize.gs.esmi`.

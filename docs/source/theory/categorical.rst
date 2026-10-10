.. _theory-categorical:

#####################
Categorical variables
#####################

A categorical variable :math:`Z(v) \in \{1, \dots, K\}` takes one of finitely many values, a rock
type, a facies, a land use. The ensemble estimates it with the same partitions as a continuous
variable. Only the decoder changes, a classifier fitted inside each cell. This page describes the law
the members give, the classifiers, and three ways of reading one category from the members, with
their assumptions.

The law of the classes
======================

Each partition :math:`t` gives a member :math:`\hat z^{(t)}(v)`, the category its cell's classifier
predicts at :math:`v`. The shares of the categories among the members estimate the probability of
each one,

.. math::

   \hat\pi_k(v) = \frac{1}{T_v} \sum_{t} \mathbf 1\{\hat z^{(t)}(v) = k\},
   \qquad \sum_k \hat\pi_k(v) = 1,

with :math:`T_v` the number of members defined at :math:`v`. This is the categorical form of the
predictive law of a location (:doc:`esi`). It carries more than a map of categories. The
probability of a class gives its own map, and the entropy :math:`-\sum_k \hat\pi_k \log \hat\pi_k` or
the share of members disagreeing with the reported category measures how settled the estimate is.

A cell without data has no classifier to fit. Under ``empty_cells="nan"`` its members are undefined
and take no part in the shares, so :math:`T_v` counts the members defined at :math:`v`. Under
``"mark"`` or ``"coarsen"`` they carry a category like the others (:doc:`blockmark`).

The classifiers
===============

Any classifier fitted on the data of a cell can serve as decoder. Spatialize builds in a nearest-
neighbour classifier with an anisotropic metric (``knn_pca``). In each cell it finds the direction
along which the category changes least, from a principal component analysis of the differences
between neighbouring data, and measures distances in a frame rotated to that direction and
stretched across it. The stretch and the number of neighbours are chosen by cross-validation inside
the cell. Plain nearest neighbours, support vector machines, random forests, decision trees and any
classifier with the scikit-learn interface can take its place.

Reading one category
====================

Three readings turn the members into one category per location. They differ in how much they trust
each partition.

**Majority vote** (the default). The category of largest share, :math:`\arg\max_k \hat\pi_k(v)`,
every partition weighing the same. For an ordinal variable, whose categories have an order, the
median category replaces the mode, so that a location torn between the two ends of the order does not
report one of them.

**Weighted majority vote.** When the reliability of each partition is known in advance, for instance
from a validation score, each member counts with its partition's weight :math:`w_t`, and the reported
category maximises :math:`\sum_t w_t \mathbf 1\{\hat z^{(t)}(v) = k\}`. Equal weights give the majority
vote.

**Dawid–Skene.** When the reliabilities are unknown, the Dawid–Skene model learns them from the
members, treating the partitions as annotators. Partition :math:`t` reports category :math:`l` when
the true one is :math:`k` with probability :math:`\theta^{(t)}_{kl}`, its confusion matrix, and the
true categories have prior shares :math:`\pi_k`. The posterior of the true category at :math:`v` is

.. math::

   P(z(v) = k \mid \text{members}) \propto \pi_k \prod_{t} \theta^{(t)}_{k\,\hat z^{(t)}(v)},

the product running over the members defined at :math:`v`. EM alternates this posterior with new
estimates of :math:`\pi` and of the confusion matrices until the log-likelihood settles. Partitions
that agree with the consensus get matrices close to the identity and weigh more. A group of partitions
biased towards one category is thereby discounted where the majority vote would follow it. The
reliabilities are global, so deciding one location informs the others through the partitions that
voted on it.

Kinds of variables
------------------

A binary variable is the case :math:`K = 2`, each confusion matrix holding a sensitivity and a
specificity. An ordinal variable adds to each confusion matrix pseudo-counts that decrease with the
distance between categories in the order, so that confusions between neighbouring categories are
expected more than confusions between distant ones. A nominal variable has no order.

Categories that may not touch
-----------------------------

Geology often forbids some contacts, basalt against gneiss for instance. On a grid of locations, the
spatial variant of Dawid–Skene keeps forbidden categories apart. At each iteration, category
:math:`k` at location :math:`v` loses

.. math::

   \beta \sum_{u \sim v} \; \sum_{l\,:\,(k, l) \in F} r_u(l)

from its log-posterior, with :math:`u \sim v` the grid neighbours of :math:`v`, :math:`F` the forbidden
pairs, :math:`r_u(l)` the probability of category :math:`l` at :math:`u` in the previous iteration, and
:math:`\beta \ge 0` the strength of the constraint. This is a mean-field reading of a Potts-like
interaction. The penalty follows the neighbours' state, so a location next to a forbidden category
moves away from it, while the same tie next to an allowed category stays as it is. The strength
:math:`\beta` is set by the user, :math:`\beta = 0` giving the non-spatial model.

Reproducibility
---------------

EM reaches a local maximum from a random start of the confusion matrices. A fixed seed makes the
result reproducible. It is also the same for any number of threads.

Choosing a reading
==================

The majority vote suits partitions of comparable quality, which is the ensemble's default
assumption. The weighted vote suits reliabilities measured beforehand. Dawid–Skene suits partitions
of unequal and unknown quality, for instance coarse and fine partitions mixed, or classifiers that
fail in some cells. Its spatial variant suits maps whose geology forbids contacts. Whatever the
reading, the probabilities of the classes remain the fuller answer, the reported category being one
summary of them.

In Spatialize
=============

Categorical estimation is the functions of :doc:`../reference/cat_esi`: the class probabilities
(``CatESIResult.class_probabilities``), the majority and median votes, the weighted vote
(:func:`~spatialize.gs.cat_esi.aggregate_with_weighted_mv`) and Dawid–Skene
(:func:`~spatialize.gs.cat_esi.aggregate_with_btd`, ``re_estimate('btd')``). Two bundled data sets
serve to try them: the lithology of a geologic map near Golden, Colorado, with the whole map as the
truth (:func:`~spatialize.data.load_lithology_golden`), and synthetic wells with two facies
(:func:`~spatialize.data.load_facies_pyrcz`).

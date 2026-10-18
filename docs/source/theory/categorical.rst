.. _theory-categorical:

#####################
Categorical variables
#####################

A categorical variable :math:`Z(v) \in \{1, \dots, K\}` takes one of finitely many values at each
location. Land cover, soil type, habitat, urban zoning, rock type and facies are nominal examples. The
presence or absence of a species or of a contaminant is binary. Classes of pollution, of flood risk or
of soil drainage are ordinal, their categories having an order. The ensemble estimates such a
variable with the same partitions as a continuous one. Only the decoder changes, a classifier fitted
inside each cell. This page describes the law the members give, its uncertainty, the classifiers and
their choice, and three ways of reading one category from the members, with their assumptions.

The law of the classes
======================

Each partition :math:`t` gives a member :math:`\hat z^{(t)}(v)`, the category its cell's classifier
predicts at :math:`v`. The shares of the categories among the members estimate the probability of
each one,

.. math::

   \hat\pi_k(v) = \frac{1}{T_v} \sum_{t} \mathbf 1\{\hat z^{(t)}(v) = k\},
   \qquad \sum_k \hat\pi_k(v) = 1,

with :math:`T_v` the number of members defined at :math:`v`. This is the categorical form of the
predictive law of a location (:doc:`esi`). Each indicator :math:`\mathbf 1\{Z(v) = k\}` is estimated by
an ensemble whose members are 0 or 1, so the shares inherit what the theory says of ensemble means.
They are frequencies of the classifiers' predictions, not calibrated probabilities. Whether a share of
0.7 means that the category is right seven times in ten is a property of the data and of the
classifier, which cross-validation can check.

The law carries more than a map of categories. The probability of each class gives its own map. The
spread of the law measures how settled the estimate is, through the share of members disagreeing
with the reported category,

.. math::

   d(v) = 1 - \hat\pi_{\hat k(v)}(v),

or through the entropy :math:`-\sum_k \hat\pi_k(v) \log \hat\pi_k(v)`. The disagreement of each
partition separately, a 0 or 1 per location and partition, gives a cube from which other summaries
can be computed.

Cells without data
------------------

A cell holding no datum has no classifier to fit. Under ``empty_cells="nan"`` its members are undefined
and take no part in the shares, so :math:`T_v` counts the members defined at :math:`v`, and a location
whose members are all undefined gets no category. Under ``"mark"`` such a cell receives the category of
a cell with data, one per cell, and under ``"coarsen"`` the classifier of the nearest coarser cell with
data predicts it (:doc:`blockmark`). The hyperparameter search reports the share of the data left
without a category, and warns above the session's ``max_left_out``.

What categories cannot fix
--------------------------

In the block-mark model (:doc:`blockmark`) a field is a random partition of the domain into blocks with
one value drawn independently in each. With continuous values, two locations show the same value only if
they share a block, so the values record how the partition groups the locations. A category takes few
values, and separate blocks often draw the same one, so two locations with the same category may share a
block or lie in two. The law of two categories fixes how often pairs and triples of locations share a
block, never how often four or more do. Two random groupings that put pairs and triples together equally
often, one sometimes joining four locations into one block and the other never, give the categories
exactly the same law. Each further category recovers only a few more orders. Categorical data therefore
cannot fix the shapes of the units beyond pairs and triples. The partitions and the classifier supply
that structure, and the disagreement among the members measures the freedom it leaves.

The classifiers
===============

Any classifier fitted on the data of a cell can serve as decoder. A cell whose data all share one
category predicts that category.

The anisotropic nearest neighbours (``knn_pca``)
------------------------------------------------

The built-in classifier, the default, measures distances in a frame adapted to the boundaries
between categories in each cell.

1. **The direction of the boundaries.** For each datum, the differences to its nearest neighbours of
   another category, at most 15 of them, point across the boundaries between categories. The first
   right singular vector of these differences, centred, is the direction :math:`n` in which the
   category changes most, normal to the boundaries. The second gives the direction along them.
2. **The metric.** Coordinates are rotated to these directions and scaled, by :math:`\sqrt r` across
   the boundaries and :math:`1/\sqrt r` along them, for an anisotropy ratio :math:`r \ge 1`. A
   neighbour along a boundary then counts as nearer than one at the same distance across it, which
   suits categories laid in bands or elongated patches.
3. **The choice of** :math:`r` **and of the number of neighbours** :math:`k`. A grid of ratios between 1
   and 4 and a few odd values of :math:`k` are compared. The grid holds 3 ratios when the first
   direction explains at most 60 % of the variance of the differences, 5 up to 80 %, and 7 above. They
   are compared by repeated random splits of the cell's data, 70 % to train and 30 % to test. The pair of
   best accuracy is kept. Cells with more than 200 data are subsampled for this search. The splits
   are seeded from the run's seed and the cell's centroid, so that the choice is reproducible.
4. **The prediction.** The :math:`k` nearest data in that metric vote, the most frequent category
   winning.

Without enough differences to estimate a direction, the cell falls back to isotropic distances with
one neighbour. In one dimension the ratio has no effect. Before version 1.3 the scaling was the
reverse, shrinking distances across the boundaries, so that cross-validation always chose
:math:`r = 1` and the classifier was isotropic.

Other classifiers
-----------------

Plain nearest neighbours (``knn``), support vector machines (``svm``), random forests (``rf``) and
decision trees (``dt``) are available by name, with their own hyperparameters. Any classifier with
the scikit-learn interface can be passed. A Python callable can serve as a fully custom classifier.

Choosing the hyperparameters
----------------------------

The categorical search compares combinations of the partition granularity, of the number of
partitions and of the classifier's hyperparameters by k-fold cross-validation, scoring the predicted
categories. The scores are oriented so that lower is better: one minus the accuracy, one minus the
macro, weighted or micro F1, one minus the macro precision or recall, and one minus Cohen's kappa,
which discounts the agreement expected by chance and so suits imbalanced classes. By default the
partitions are drawn once on all the data and each held-out datum is predicted by its cell's
classifier trained on the cell's other data (``cv="engine"``), as the continuous search does. With
``cv="refit"`` the ensemble is trained again for each fold. The categorical search has no Pareto
variant yet, because the encoder error of the continuous search has no categorical counterpart in
Spatialize.

Reading one category
====================

Three readings turn the members of a location into one category. They differ in how much they trust
each partition.

Majority vote
-------------

The default reports the category of largest share,

.. math::

   \hat k(v) = \arg\max_k \hat\pi_k(v),

every partition weighing the same, the first category met winning a tie. For an ordinal variable the
median category replaces the mode, so that a location torn between the two ends of the order does not
report one of them. With an even number of members the median is the category whose rank lies
nearest to the mean of the two central ranks.

Weighted majority vote
----------------------

When the reliability of each partition is known in advance, for instance from a validation score,
each member counts with its partition's weight :math:`w_t \ge 0`,

.. math::

   \hat k(v) = \arg\max_k \sum_t w_t \, \mathbf 1\{\hat z^{(t)}(v) = k\}.

Equal weights give the majority vote.

Dawid–Skene
-----------

When the reliabilities are unknown, the Dawid–Skene model learns them from the members, treating the
partitions as annotators of unequal skill. Partition :math:`t` reports category :math:`l` when the
true one is :math:`k` with probability :math:`\theta^{(t)}_{kl}`, its confusion matrix, independently
of the other partitions given the true category, and the true categories have prior shares
:math:`\pi_k`. Writing :math:`l_{vt}` for the member of partition :math:`t` at :math:`v`, the
log-likelihood of the members is

.. math::

   \log L(\pi, \theta) = \sum_v \log \sum_k \pi_k \prod_{t} \theta^{(t)}_{k\, l_{vt}},

the product running over the members defined at :math:`v`. EM maximises it by alternating two steps.

- **E-step.** The posterior of the true category at each location,

  .. math::

     r_v(k) = \frac{\pi_k \prod_t \theta^{(t)}_{k\, l_{vt}}}{\sum_{k'} \pi_{k'} \prod_t \theta^{(t)}_{k'\, l_{vt}}}.

- **M-step.** New prior shares and confusion matrices from these posteriors,

  .. math::

     \pi_k = \frac{\sum_v r_v(k)}{\sum_{k'} \sum_v r_v(k')}, \qquad
     \theta^{(t)}_{kl} = \frac{\sum_{v\,:\,l_{vt} = l} \big(r_v(k) + c_{kl}\big)}
                              {\sum_{l'} \sum_{v\,:\,l_{vt} = l'} \big(r_v(k) + c_{kl'}\big)},

  with :math:`c_{kl} = 0` for a nominal variable.

The confusion matrices start close to the identity, a diagonal drawn between 0.7 and 0.8 with the
rest spread evenly, and the prior shares start uniform. The steps repeat until the log-likelihood
changes by less than a tolerance or a maximum number of iterations is reached. The reported category
is the one of largest posterior. The posteriors themselves are the probabilities of the classes under
the model.

Partitions that agree with the consensus get matrices close to the identity and weigh more in the
posterior. A group of partitions biased towards one category is thereby discounted where the majority
vote would follow it. The confusion matrices are global, so deciding one location informs the others
through the partitions that voted on it. The start close to the identity keeps the labels of the
categories from being permuted, a symmetry of the likelihood. EM reaches a local maximum, so the
result depends on the random start, which a seed fixes. Each iteration costs a few passes over the
members, proportional to their number times :math:`K`.

The model assumes the partitions independent given the true category. The members of an ensemble are
not, since every partition is fitted to the same data, and their agreement overstates the evidence.
With tens of partitions the posteriors are therefore close to 0 or 1 almost everywhere, even where
the members are split. The reported category is a sound reading. For the probability of each class,
the shares :math:`\hat\pi_k(v)` of the members are the measure to use.

Kinds of variables
------------------

A **nominal** variable has no order. A **binary** variable is the case :math:`K = 2`, each confusion
matrix holding a sensitivity and a specificity. An **ordinal** variable adds to each member's entry the
pseudo-count

.. math::

   c_{kl} = \frac{0.1}{d_{kl}^2 + 1},

with :math:`d_{kl}` the distance between the ranks of :math:`k` and :math:`l` in the order, so that
confusions between neighbouring categories are expected more than confusions between distant ones.

Categories that may not touch
-----------------------------

Some pairs of categories should not be adjacent. Water does not lie next to bare rock in a land-cover
map without a shore between them, an industrial zone is kept apart from a residential one, and some
habitats need a transition between them. On a grid of locations, the spatial variant of Dawid–Skene
keeps forbidden categories apart. The neighbours of a location are the adjacent grid cells along each
axis, four in two dimensions and six in three. At each iteration after the first, category :math:`k`
at location :math:`v` loses

.. math::

   \beta \sum_{u \sim v} \; \sum_{l\,:\,(k, l) \in F} r_u(l)

from its log-posterior, with :math:`u \sim v` the neighbours of :math:`v`, :math:`F` the forbidden pairs,
:math:`r_u(l)` the posterior of category :math:`l` at :math:`u` in the previous iteration, and
:math:`\beta \ge 0` the strength of the constraint. This is a mean-field treatment of a Potts-like
interaction between neighbours. The penalty follows the neighbours' state, so a location next to a
forbidden category moves away from it, while the same tie next to an allowed category stays as it is.

The penalty competes with the evidence of the members. It moves a location whose members are split,
or whose posterior is close to a tie, and leaves alone a location the members support strongly. With
the posteriors close to 0 or 1, as above, it acts on few locations. Removing a contact also needs a
third category with support between the two forbidden ones. A location whose members name only
those two, on leaving one category, joins the other, which shifts the contact by one cell.

The forbidden pairs are given as pairs of categories, written as in the data, each pair forbidding
both orders, for instance water with urban and water with cropland in a land-cover map. The grid is
the one of a gridded result, or its shape is given with the members in the grid's row-major order.
The strength :math:`\beta` is set by the user, :math:`\beta = 0` giving the non-spatial model. The
log-likelihood that EM reports is the one of the non-spatial model, which does not see the penalty,
so it cannot choose :math:`\beta`. The effect of the constraint is read on the map instead, through the
contacts that remain.

Reproducibility and computation
-------------------------------

A fixed seed makes the result of Dawid–Skene reproducible. The E-step, the M-step and the
log-likelihood run on the session's threads, with their sums in a fixed order, so that the result is
also the same for any number of threads. The progress of EM is shown as for the other functions, and
Ctrl-C stops it between iterations.

Choosing a reading
==================

The majority vote suits partitions of comparable quality, the ensemble's default assumption. The
weighted vote suits reliabilities measured beforehand. Dawid–Skene suits partitions of unequal and
unknown quality, such as coarse and fine partitions mixed, or classifiers that fail in some cells. Its
spatial variant suits maps whose categories may not all touch. Whatever the reading, the
probabilities of the classes remain the fuller answer, the reported category being one summary of
them.

In Spatialize
=============

Categorical estimation is the functions of :doc:`../reference/cat_esi`: the class probabilities
(``CatESIResult.class_probabilities``), the disagreement (``precision``, ``precision_cube``), the
majority and median votes, the weighted vote
(:func:`~spatialize.gs.cat_esi.aggregate_with_weighted_mv`), Dawid–Skene
(:func:`~spatialize.gs.cat_esi.aggregate_with_btd`, ``re_estimate('btd')``) and the search
(:func:`~spatialize.gs.cat_esi.cat_esi_hparams_search`). Two bundled data sets serve to try them: the
lithology of a geologic map near Golden, Colorado, with the whole map as the truth
(:func:`~spatialize.data.load_lithology_golden`), and synthetic wells with two facies
(:func:`~spatialize.data.load_facies_pyrcz`).

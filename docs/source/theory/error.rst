.. _theory-error:

###############################
Error and the choice of a model
###############################

Two sources of error
====================

An ensemble estimate errs for two separate reasons, each of which the theory measures.

- The *encoder error* comes from the cells. A cell pools data on the assumption that their laws
  are alike, and where the field changes inside a cell that assumption fails. It depends on the
  partition alone, whatever local model is used.
- The *decoder error* comes from the local model, which can describe the law of a cell's data more
  or less well.

The total error, measured as the excess log-loss of the predictive law, is exactly the sum of the
two. They pull in opposite directions as the cells shrink. Finer cells are more homogeneous, which
lowers the encoder error, but they hold fewer data, which raises the decoder error. The best
granularity lies in between, and finding it is the job of model selection.

What cross-validation sees
==========================

Cross-validation predicts each datum from the others and scores the result, which is how
Spatialize compares configurations. Its score is bounded below by the encoder error of the
partition in use, and no tuning of the local model can lower that floor. A search that minimises
the cross-validation score alone therefore picks the best decoder for each partition, while the
encoder error of the partition stays hidden in the floor.

The theory turns this into a rule. The encoder error can be bounded from the data, by comparing the
predictive laws at pairs of data that share a cell, since a cell whose data have very different laws
is pooling unlike places. Model selection then minimises the cross-validation score among the
configurations whose estimated encoder error stays below a level :math:`\tau` of the user's
choosing, so robustness is a constraint and not one more term to trade away.

The Pareto frontier
===================

Each configuration, a granularity, a number of partitions and the decoder's parameters, gives a
pair of errors, one for the encoder and one for the decoder. A configuration is *Pareto-optimal*
when no other has both errors at most as large and one smaller. The Pareto-optimal configurations form a frontier. Solving the constrained problem for every level :math:`\tau` traces that frontier exactly.
The frontier need not be convex, so adding the two errors with a weight and minimising misses parts
of it, which is why the constrained form is used.

In practice the frontier is read in one of three ways.

- The configuration of lowest decoder error, when the partition is trusted.
- The *knee*, where lowering one error starts to cost much more of the other.
- The best configuration for a given level :math:`\tau` of encoder error, when a robustness level is
  required.

Scoring a predictive law
========================

Cross-validation can score the predictions by an absolute or squared error, which judges the map.
It can also score the whole predictive law, by its log-likelihood at the held-out datum or by the
continuous ranked probability score, which judge the intervals as well. Scores of the law need
enough partitions to estimate it at each datum, a few tens at least.

In Spatialize
=============

:func:`~spatialize.gs.esi.esi_hparams_search` searches by cross-validation alone, over
``n_partitions``, ``alpha`` and the decoder's parameters, with k-fold or leave-one-out.
:func:`~spatialize.gs.esi.esi_pareto_hparams_search` estimates both errors for each configuration and returns the frontier. Its ``best_result`` reads
the frontier by lowest decoder error (``"min_decoder"``) or by knee (``"knee"``), while
``best_for_tau`` picks the best configuration under a level :math:`\tau` (:doc:`../reference/esi`). The scoring functions are in
:mod:`spatialize.gs.esi.scorefunction`.

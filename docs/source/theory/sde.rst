.. _theory-sde:

#################################
Spatial distributional estimation
#################################

The recipe
==========

The estimator is simple enough to describe in a sentence. You cut the domain into cells at random,
fit a cheap local model inside each cell, read off a prediction at every location, and repeat many
times. After :math:`T` repetitions each location holds not one prediction but :math:`T` of them, one per cut. That cloud is the estimate, a distribution and not a number. Anyone who has met bagging
in machine learning will recognise the move, an ensemble of cheap models whose spread is kept
instead of averaged away.

The two halves of the recipe carry names that the rest of these pages use.

- The *encoder* is the random partition. It maps a location to the cell it falls in, and so
  decides which data the local model may use there. Spatialize offers several partition processes
  (:doc:`encoders`).
- The *decoder* is the local model. It predicts at a location from the data of its cell alone, by
  averaging them with some weights or by drawing one of them (:doc:`decoders`).

Each repetition gives one *member* of the ensemble, a value at every location computed under one
partition. The members at a location form its *predictive law*.

Why the laws fit together
=========================

The laws at different locations are not computed separately. Two neighbouring locations fall in
the same cell in most partitions, so they are predicted from almost the same data, while two
distant locations rarely share a cell. The overlap of the cells from one partition to the next
ties the predictions at different places together, so the members form a joint law of the field,
with its own dependence between locations. The probability that a set of locations shares a cell,
which only depends on the partition process, carries that dependence at every order, pairs, triples
and beyond, where kriging uses the covariance of pairs alone.

Every law built this way is a proper law, since it is the empirical law of :math:`T` values, so the
order violations of indicator kriging cannot occur.

Reading the law
===============

The law at a location answers many questions, each a different reading of it.

- Its mean or its median gives a map.
- A quantile gives a conservative or an optimistic figure.
- The share of members above a threshold estimates the probability of exceeding it.
- Two quantiles bound an interval.

Which single number to report depends on what an error costs. Under a squared loss the best report
is the mean, under an absolute loss the median. When over-predicting costs :math:`c` times as much
as under-predicting, the best report is the quantile at level :math:`1/(1+c)`, which needs no
variance and so stays meaningful for heavy-tailed variables, where the mean misleads. Choosing the
aggregation of the members is choosing that loss.

Questions also differ by the object they concern.

- A question about one location, a probability of exceedance, a quantile, the mass of an interval,
  is read from the law at that location.
- A question about the map taken whole, an area above a limit, a grade–tonnage curve, the average
  over a block, is read from the members, each a whole field under one partition. Assembling it
  from per-location figures, for instance by averaging the quantiles of the points of a block,
  assumes that all the locations move together and overstates the risk.

How many partitions
===================

The number of partitions :math:`T` controls the Monte Carlo error of every reading, which shrinks
like :math:`T^{-1/2}`. A few hundred partitions usually make it negligible against the error of
the estimate itself. More partitions do not change what the ensemble estimates, so they cannot
remove a bias that comes from the partition scale or from the local model.

What the ensemble converges to
==============================

With many partitions and data dense enough for the cells to fill, the ensemble converges to a
definite law. That law describes how the predictions of this one realisation of the field vary as
the cells move, not how the field would vary from one realisation to another. In practice the
spread of the members reflects the partition, so it tends to understate the variability of the
field near a location. Simulation corrects for it by widening the local laws (:doc:`ess`), and the
choice of granularity balances the errors it introduces (:doc:`error`).

In Spatialize
=============

:func:`~spatialize.gs.esi.esi_griddata` and :func:`~spatialize.gs.esi.esi_nongriddata` compute the
ensemble, with ``n_partitions`` partitions of granularity ``alpha``. The members are
``esi_samples()`` of the result, and ``agg_function`` (:doc:`../reference/functions`) chooses the
reading reported as the estimate.

.. _theory-problem:

###########################
Estimating a law, not a map
###########################

Suppose you have measured a variable at a few hundred places, the copper grade of drill cores, the
rainfall at gauges, the concentration of a pollutant in wells, and you want to say something about
it everywhere else. The classical answer is a map, one best value per location. A map answers some
questions well, such as "what is the grade here, roughly?", and others not at all.

Many practical questions are about the uncertainty and not about the best value.

- *What is the probability that the concentration here exceeds the legal limit?* That is a
  question about the tail of the law at one location.
- *How large is the area above the limit?* That is a question about many locations at once, whose answer depends on how their values move together.
- *How much ore lies above a cut-off grade?* Again a question about the joint behaviour of many
  locations, whose answer a map of averages gets wrong by smoothing the extremes away.

A single map carries none of this, so the object to estimate is the *law* of the variable at every
location, together with the way the laws at different locations depend on one another.

What the classical tools see
============================

Kriging, the workhorse of geostatistics, predicts with weights computed from the covariance, a
description of how values at two locations vary together. The covariance only involves pairs of
locations. Two fields can share it while differing in their connectivity, in the skewness of their
values or in how often extremes occur together, and kriging cannot tell them apart. Its prediction
is the best linear one under its own model, which is a statement about averages and pairs.

The usual extension to laws, indicator kriging, estimates the probability of exceeding each of
several thresholds with a separate kriging system. The systems know nothing of one another, so
their answers need not fit together. The estimated probability of exceeding a threshold can come
out larger than that of exceeding a lower one, or fall outside [0, 1], and nothing ties the laws at
different locations into a joint law.

Two requirements follow. The method Spatialize implements meets both by construction.

- At every location the estimate is a proper law, with no order violations to repair afterwards.
- The laws at different locations come from one shared mechanism, so that they knit into a joint
  law of the whole field.

In Spatialize
=============

The estimators of the next pages return, at each location, the members of an ensemble, from which
any reading of the law is computed (see :doc:`sde`). The conformance tests check the coherence
claims above on simulated fields (:doc:`../scenarios/index`).

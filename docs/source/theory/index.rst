.. _theory:

######
Theory
######

Spatialize implements a way of estimating a spatial variable that returns, at every location, a
whole law of possible values instead of a single number. These pages explain the ideas behind it,
which come from the book *A General Theory of Higher-Order Geostatistics* (Egaña, Díaz, Navarro and
Ehrenfeld), and say how to choose among the pieces Spatialize offers. They carry no code. Each page
ends with a pointer to the functions that implement it.

.. toctree::
   :maxdepth: 1

   problem
   sde
   encoders
   decoders
   blockmark
   error
   esi
   ess
   posterior
   categorical
   other

The pages build on one another in that order. A reader in a hurry can read the second, then the
pages on encoders and decoders, which hold the practical advice.

Conformance tests
=================

The theory states what the estimators must do, from the law of the random partitions to the
behaviour of a complete estimator on a field whose truth is known. Spatialize checks these statements
on itself with a public suite of conformance tests. Each test sets up a situation, decides a criterion
registered before its first run with a statistical test of stated level and power, and runs again on
every change. The catalogue lists every claim tested, the part of the theory it comes from and its
result, the known failures included, so a reader can see which statements of these pages hold in
Spatialize as it stands.

.. toctree::
   :maxdepth: 2

   ../scenarios/index

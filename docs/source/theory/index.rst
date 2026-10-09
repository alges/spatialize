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
   other

The pages build on one another in that order. A reader in a hurry can read the second, then the
pages on encoders and decoders, which hold the practical advice.

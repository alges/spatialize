.. _theory-encoders:

########
Encoders
########

The encoder decides which data the local model sees at each location, by cutting the domain into
cells at random. Different partition processes cut in different ways, which fixes how the estimate couples nearby locations. Choosing a process is choosing which spatial symmetry to
assume.

Granularity
===========

Every process has a granularity, which Spatialize exposes as ``alpha``. Fine cells hold few data,
so the local model sees only nearby values, which follows the field closely but leaves each model
with little to work with. Coarse cells hold many data, so each model is well fed but averages over
places that may differ. The best granularity balances the two (:doc:`error`). The
hyperparameter searches look for it.

For the Mondrian processes, ``alpha`` lies in :math:`[0, 1)` and sets the rate of the process,
:math:`\lambda = 1/(\mu(H)(1-\alpha))`, with :math:`\mu(H)` the sum of the sides of the box
:math:`H` the cells are drawn on. Values near 0 give a few large cells, values near 1 many small
ones. For the Voronoi processes, ``|alpha|`` sets the expected number of cells relative to the
number of data.

The box
=======

The cells are drawn on a box. By default it is the smallest box holding the data and the
locations to estimate, so asking for estimates over a larger region changes the partitions, and
with them the estimate at the other locations. Setting a fixed domain makes the estimate at a
location independent of the other locations requested, which matters when estimates made in
separate runs must be compared.

Mondrian
========

The Mondrian process cuts the box with straight lines parallel to the axes, one after another,
each cut splitting the cell it falls in, until a budget runs out. The cells are rectangles, or boxes
in higher dimensions. The process is named after the painter, whose canvases its cells resemble.
Two locations share a cell less and less often as they move apart, with a probability that depends
on the distance measured axis by axis, so its level curves are diamonds and not circles.

Spatialize's Mondrian, the default, departs from the textbook process in two details. It always cuts the box at least once, choosing the direction of each cut uniformly among the axes, whatever the shape of the cell. On a square domain the difference is modest. On an elongated
domain the uniform choice makes the cells elongated too, so the estimate couples locations along
the long side more strongly than across it.

Prefer it as the general default. It is fast, it works in any dimension, and Spatialize's published
results were obtained with it. Each member shows rectangular blocks, which the ensemble averages away, so that a few hundred partitions leave only a faint trace of the axes in
the averaged map.

The theory's Mondrian process
=============================

``mondrian-raw`` is the Mondrian process as the theory defines it. The whole box may stay a single cell. Each cut chooses its direction with probability proportional to the side it cuts, so long sides attract cuts, which keeps the cells from inheriting the shape of the domain. The probability that two
locations share a cell is exactly :math:`\exp(-\lambda \lVert x - y \rVert_1)`, the same in every
direction once distance is measured axis by axis.

Prefer it on elongated domains, where the default Mondrian stretches its cells, and whenever the
partition law itself matters, for instance when the estimate is compared with the theory's closed
forms or used to study the latent geometry of a field.

Voronoi with uniform nuclei
===========================

This process scatters points, called nuclei, uniformly over the box and gives each location to its
nearest nucleus. The cells are convex polygons with no preferred direction, so the probability that
two locations share a cell depends only on the distance between them. With ``alpha < 0``
Spatialize draws a random number of nuclei, on average half the number of data times ``|alpha|``.

Prefer it when the field is anisotropic at an angle to the axes, or when the direction of its
continuity is unknown, since its cells impose no direction of their own. Its averaged maps carry no
trace of the axes, which helps when they are read by eye.

Voronoi with nuclei at the data
===============================

With ``alpha >= 0`` the nuclei are chosen among the data locations, so the cells are small where
the data are dense and large where they are sparse. The resolution of the estimate then follows the
sampling. This is the default of Spatialize's Voronoi partition.

Prefer it when the sampling is strongly clustered, so that the estimate is detailed where the data
allow it and coarse elsewhere. Its partitions depend on where the data were taken, so they have no
law of their own to compare with the theory.

In Spatialize
=============

The public functions take the process as ``p_process`` (``"mondrian"``, ``"mondrian-raw"`` or
``"voronoi"``), with ``data_cond`` choosing the nuclei of the Voronoi partition and ``alpha`` the
granularity (:func:`~spatialize.gs.esi.esi_griddata`). A fixed domain is a session setting
(:mod:`spatialize.session`). The conformance tests measure the partition laws, the theory's
Mondrian process matching its closed form and the default one departing from it
(:doc:`../scenarios/encoders`).

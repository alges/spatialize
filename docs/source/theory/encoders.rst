.. _theory-encoders:

########
Encoders
########

The encoder cuts the domain into cells at random, in a way that fixes the co-occurrence
:math:`e(S)` of :doc:`sde`, hence how the estimate couples nearby locations. Choosing a process is
choosing which spatial symmetry to assume.

The box and the granularity
===========================

The cells are drawn on a box :math:`H = \prod_{c=1}^d [a_c, b_c]`, with sides
:math:`\ell_c = b_c - a_c` and :math:`\mu(H) = \sum_c \ell_c`. By default :math:`H` is the smallest
box holding the data and the locations to estimate, so asking for estimates over a larger region
changes the partitions. A fixed domain removes that dependence (:mod:`spatialize.session`).

Every process has a granularity, exposed as ``alpha``. For the Mondrian processes,
:math:`\alpha \in [0, 1)` sets the rate

.. math::

   \lambda(\alpha) = \frac{1}{\mu(H)\,(1-\alpha)},

so :math:`\alpha = 0` gives the coarsest partitions, :math:`\lambda = 1/\mu(H)`, and
:math:`\alpha \to 1` ever finer ones. For the Voronoi processes, :math:`|\alpha|` sets the expected
number of cells relative to the number of data :math:`n`. Fine cells follow the field closely but
leave each local model few data, coarse cells feed the local models well but average over places
that may differ. The best granularity balances the two (:doc:`error`).

The Mondrian process
====================

The Mondrian process of rate :math:`\lambda` on :math:`H` is drawn by a recursion started from
:math:`H` with budget :math:`\lambda`. Given a box with sides :math:`\ell_1, \dots, \ell_d` and
budget :math:`\beta`,

1. draw :math:`E \sim \operatorname{Exp}\big(\sum_c \ell_c\big)`;
2. if :math:`E > \beta` the box is a cell;
3. otherwise choose an axis :math:`c` with probability :math:`\ell_c / \sum_{c'} \ell_{c'}`, cut the
   box at a uniform position along it, and apply the recursion to each half with budget
   :math:`\beta - E`.

Long sides attract cuts, since a large box waits less for its next cut, with the axis then chosen in proportion to the sides. The cells are boxes, and the co-occurrence of a set :math:`S` has the closed
form

.. math::

   \begin{aligned}
   e(S) &= \exp\Big(-\lambda \sum_{c=1}^d \operatorname{range}_c(S)\Big),
   \\
   e(\{x, y\}) &= \exp\big(-\lambda \lVert x - y \rVert_1\big),
   \end{aligned}

with :math:`\operatorname{range}_c(S)` the extent of :math:`S` along axis :math:`c`. Distance is
measured axis by axis, so the level curves of :math:`e(\{0, h\})` are diamonds and not circles.

``mondrian``, the default, is this process exactly. Its partitions follow the closed forms above,
which makes the estimate comparable with them and with the latent geometry of a field. Each
member shows rectangular blocks, which the ensemble averages away, so that a few hundred partitions
leave only a faint trace of the axes in the averaged map.

The Mondrian partition of version 1.2
-------------------------------------

Up to version 1.2 the default Mondrian partition departed from the recursion in two details. The
whole box was always cut, as if :math:`E = 0` at the root. The axis was chosen uniformly, with
probability :math:`1/d`, whatever the sides. Each child still waited
:math:`\operatorname{Exp}\big(\sum_c \ell_c\big)` for its own cut. It remains available as
``"mondrian-legacy"``, to reproduce earlier results.

On a square domain the difference is modest at coarse granularities, the co-occurrence lying
somewhat below :math:`e(\{x,y\})`, and the partition holding about twice as many cells for the same
``alpha``. On an elongated domain the uniform choice of axis makes the cells elongated too, so
locations along the long side are coupled more strongly than across it.

At fine granularities the uniform choice of axis multiplies the cells. A thin cell keeps its long
side, so it still waits little for its next cut, which half of the time falls across its short side
and makes it thinner. The number of cells then grows much faster than the theory's
:math:`(1 + \lambda)^2` on the unit square. With 120 data on the unit square, at ``alpha`` = 0.979
(:math:`\lambda = 24`) a partition has at least 4 500 cells, against about 600 for the theory's
process. At ``alpha`` = 0.990 (:math:`\lambda = 48`) it has at least 92 000, against about 2 400,
most of them slivers holding no datum. Time and memory follow, one partition taking 1.7 s and
0.9 GB at ``alpha`` = 0.993 and 20 s and 9 GB at ``alpha`` = 0.995.

On the drill holes of the Andes bundled with Spatialize (a box of 400 by 600), with the same
``alpha`` the partition of version 1.2, being the finer one, gave the lower leave-one-out error
of IDW up to ``alpha`` = 0.9, and the theory's process from ``alpha`` = 0.95, with adaptive IDW from
0.9. At ``alpha`` = 0.98 it held 5 357 cells against 615, for ten times the time. Since a search chooses ``alpha``,
the theory's process became the default in version 1.3.

The Voronoi processes
=====================

A Voronoi partition places :math:`N` points :math:`g_1, \dots, g_N`, the *nuclei*, and gives each
location to its nearest nucleus,

.. math::

   C_k = \big\{x \in H : \lVert x - g_k \rVert \le \lVert x - g_{k'} \rVert \ \text{for every } k'\big\}.

The cells are convex polygons, polyhedra in higher dimensions, and nothing in the construction
prefers a direction. Spatialize draws

.. math::

   N = \min\big(n, \max(1, N_0)\big), \qquad N_0 \sim \operatorname{Poisson}\big(\tfrac12\, n\, |\alpha|\big),

and places the nuclei in one of two ways.

Uniform nuclei
--------------

With :math:`\alpha < 0` the nuclei are uniform in :math:`H`, which is the theory's Poisson–Voronoi
process restricted to the box, with intensity :math:`\lambda_V = \tfrac12 n|\alpha| / |H|` per unit
volume. The co-occurrence of two locations depends only on :math:`\lVert x - y \rVert`, so its level
curves are circles.

Prefer it when the field is anisotropic at an angle to the axes, or when the direction of its
continuity is unknown, since the cells impose no direction of their own. Its averaged maps carry no
trace of the axes.

Nuclei at the data
------------------

With :math:`\alpha \ge 0` the nuclei are drawn among the data locations, so the cells are small where
the data are dense and large where they are sparse. This is the default of the Voronoi partition.

Prefer it when the sampling is strongly clustered, so that the resolution follows the data. Its
partitions depend on where the data were taken, so they have no law of their own to compare with the
theory.

Summary
=======

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - ``p_process``
     - co-occurrence of two locations
     - prefer it for
   * - ``mondrian``
     - exactly :math:`e^{-\lambda \lVert x-y\rVert_1}`
     - the default, in any dimension
   * - ``mondrian-legacy``
     - close to :math:`e^{-\lambda \lVert x-y\rVert_1}`
     - reproducing the results of version 1.2
   * - ``voronoi``, ``alpha < 0``
     - a function of :math:`\lVert x-y \rVert`
     - oblique or unknown anisotropy, maps read by eye
   * - ``voronoi``, ``alpha >= 0``
     - depends on the sampling
     - clustered sampling

In Spatialize
=============

The public functions take ``p_process``, ``alpha`` and, for Voronoi, ``data_cond``
(:func:`~spatialize.gs.esi.esi_griddata`). A fixed domain is a session setting
(:mod:`spatialize.session`). The partitions can be studied on their own, through the cells of given
locations, the co-occurrence of sets and the law of groupings, together with the closed forms above
(:doc:`../reference/partitions`). The conformance tests measure the co-occurrence of each process against
these laws (:doc:`../scenarios/encoders`).

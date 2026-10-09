.. _scenarios-encoders:

#################
Encoder profiles
#################

A scenario states which partition process it targets, since the suite distinguishes the process of
the theory from the processes actually implemented.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Profile
     - Definition
   * - the theory's Mondrian process
     - The Mondrian process of rate (budget) :math:`\lambda` on a box :math:`H`, as defined by Roy
       and Teh (2009), by the theory and by Algorithm 1 of the ESI paper. It draws
       :math:`E \sim \mathrm{Exp}(\mu(H))` with :math:`\mu(H) = \sum_i (b_i - a_i)`. If :math:`E`
       exceeds the remaining budget the box becomes a cell. Otherwise it cuts an axis chosen with
       probability proportional to its side length, at a uniform position, recursing with budget
       :math:`\lambda - E`. Its co-occurrence is
       :math:`e(S) = \exp(-\lambda \sum_c \mathrm{range}_c(S))`. Spatialize implements it as its
       default partition, ``p_process="mondrian"`` since version 1.3 (profiles ``mondrian`` and
       ``mondrian-theory`` in the scenario files). The box is the bounding box of samples and
       queries, or the session domain when one is set (:doc:`../reference/session`).
   * - the Mondrian partition of Spatialize 1.2
     - The default Mondrian partition of Spatialize up to version 1.2, now
       ``p_process="mondrian-legacy"`` (profile ``mondrian-legacy``). It always splits the root
       box, choosing the cut axis uniformly among the dimensions. A child's time is its parent's plus
       :math:`\mathrm{Exp}(\mu(\text{child}))`, with splitting continuing while the time stays
       below :math:`\lambda`.
   * - the theory's Poisson–Voronoi partition
     - The Poisson–Voronoi partition of intensity :math:`\lambda_V`, as defined in the theory. Its
       generators form a homogeneous Poisson process of intensity :math:`\lambda_V` per unit volume,
       each location belonging to the cell of its nearest generator.
   * - Spatialize's Voronoi partition with uniform nuclei
     - Spatialize's Voronoi partition with ``alpha < 0``, with :math:`N \sim \max(1, \mathrm{Poisson}(0.5\,n\,|\alpha|))`
       nuclei, at most :math:`n` (the number of samples), placed uniformly in the box. Apart from
       the truncation of :math:`N`, this is a Poisson process restricted to the box, i.e. the
       theory's Poisson–Voronoi partition without generators outside :math:`H`. Two dimensions.
   * - Spatialize's Voronoi partition with nuclei at the data
     - Spatialize's Voronoi partition with ``alpha >= 0`` (its default, data-conditioned), with the same number
       of nuclei drawn among the sample locations (with replacement). Its law depends on the
       sampling design, so it has no counterpart in the theory. Two dimensions.

From the Mondrian rate to ``alpha``
===================================

Spatialize exposes a normalised granularity :math:`\alpha \in [0, 1)` (Egaña et al., 2021, eq. 8),
which differs from the rate :math:`\lambda`. The Mondrian budget is derived from it for the box
:math:`H` the partition is drawn on, as

.. math::

   \begin{gathered}
   \lambda(\alpha) = \frac{1}{\mu(H)\,(1-\alpha)}, \qquad \mu(H) = \sum_i (b_i - a_i),
   \\
   \text{equivalently}\quad \alpha = 1 - \frac{1}{\lambda\,\mu(H)} .
   \end{gathered}

The same :math:`\alpha` therefore gives different rates on different boxes. The value
:math:`\alpha = 0` gives the coarsest partition (:math:`\lambda = 1/\mu(H)`), while
:math:`\alpha \to 1` gives ever finer ones, and on the unit square :math:`\lambda = 1/(2(1-\alpha))`.
Scenarios declare the rate :math:`\lambda` together with the domain :math:`H`. Spatialize's runner
derives ``alpha`` from both (:func:`spatialize.scenarios.runners.spatialize.alpha_from_rate`),
pinning the box to the declared domain by adding the domain's corners as extra queries.

From the Voronoi intensity to ``alpha``
=======================================

Voronoi scenarios declare the theory's intensity :math:`\lambda_V`, never ``alpha``. Spatialize's
``alpha`` sets the expected number of nuclei, :math:`0.5\,n\,|\alpha|`, relative to the number of
samples :math:`n`. Matching it to the expected number of generators of the theory's process on
:math:`H`, :math:`\lambda_V\,|H|` with :math:`|H|` the volume of the box, gives

.. math::

   |\alpha| = \frac{2\,\lambda_V\,|H|}{n},

with the sign set by the profile (negative for uniform nuclei, positive for nuclei at the data)
(:func:`spatialize.scenarios.runners.spatialize.alpha_from_intensity`). The same scenario therefore
means the same partition law whatever the number of samples. Spatialize accepts only
:math:`|\alpha| < 1`, so a scenario must keep :math:`\lambda_V |H| < n/2`, the runner reporting an
error otherwise. For :math:`N` to stay close to Poisson, :math:`\lambda_V |H|` should also remain
well below :math:`n`, where the truncation at :math:`n` would act.

Measured deviation of the Mondrian partition of version 1.2
===========================================================

The table gives the probability that a location at displacement :math:`h` from the centre of the
box shares the centre's cell, estimated from 20 000 trees with :math:`\alpha = 0.8`.

.. list-table::
   :header-rows: 1

   * - box
     - direction
     - :math:`h = 0.05`
     - 0.1
     - 0.2
     - 0.3
     - closed form at 0.3
   * - 1×1 (:math:`\lambda = 2.5`)
     - axis
     - 0.847
     - 0.728
     - 0.554
     - 0.422
     - 0.472
   * - 1×1
     - diagonal (:math:`\ell_1 = h`)
     - 0.835
     - 0.708
     - 0.509
     - 0.372
     - 0.472
   * - 4×1 (:math:`\lambda = 1`)
     - long side
     - 0.961
     - 0.921
     - 0.853
     - 0.786
     - 0.741
   * - 4×1
     - short side
     - 0.839
     - 0.726
     - 0.561
     - 0.441
     - 0.741

On a square the co-occurrence lies 4–21 % below the closed form, depending on the direction beyond
the :math:`\ell_1` distance. On an elongated box the uniform choice of axis makes the partition
strongly anisotropic. The theory's process, the default since version 1.3, matches the closed form
in the same measurement on both boxes, in every direction (:math:`|z| < 2` over 20 000 trees). The
closed-form scenarios target it, and serve as negative controls (:doc:`statistics`) on the
partition of version 1.2, profile ``mondrian-legacy``.

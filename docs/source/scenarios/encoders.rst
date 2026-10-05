.. _scenarios-encoders:

#################
Encoder profiles
#################

A scenario states which partition process it targets, because the suite distinguishes the
process of the theory from the processes actually implemented.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Profile
     - Definition
   * - ``mondrian/book``
     - The Mondrian process of rate (budget) :math:`\lambda` on a box :math:`H` (Roy & Teh 2009;
       the theory's Def 2.3.1; the ESI paper's Algorithm 1): draw
       :math:`E \sim \mathrm{Exp}(\mu(H))`, :math:`\mu(H) = \sum_i (b_i - a_i)`; if :math:`E` exceeds
       the remaining budget the box is a cell; otherwise cut an axis chosen with probability
       proportional to its side length, at a uniform position, and recurse with budget
       :math:`\lambda - E`. Its co-occurrence is :math:`e(S) = \exp(-\lambda \sum_c
       \mathrm{range}_c(S))`.
   * - ``mondrian/spatialize-v1``
     - Spatialize's current Mondrian: the root box is always split; the cut axis is chosen
       **uniformly** among the dimensions; a child's time is its parent's plus
       :math:`\mathrm{Exp}(\mu(\text{child}))`; splitting continues while the time is below
       :math:`\lambda`. The box is the bounding box of samples and queries.
   * - ``voronoi/spatialize-v1``
     - Spatialize's Voronoi partition: the number of nuclei is :math:`\max(1, \mathrm{Poisson}(0.5\,n\,|\alpha|))`,
       at most :math:`n`; for :math:`\alpha \ge 0` the nuclei are random sample points
       (data-conditioned), for :math:`\alpha < 0` uniform points in the box.

``alpha`` is not the rate :math:`\lambda`
=========================================

Spatialize exposes a normalised granularity :math:`\alpha \in [0, 1)` (Egaña et al., 2021, eq. 8)
from which the Mondrian budget is derived for the box :math:`H` the partition is drawn on:

.. math::

   \lambda(\alpha) = \frac{1}{\mu(H)\,(1-\alpha)}, \qquad \mu(H) = \sum_i (b_i - a_i),
   \qquad\text{equivalently}\qquad \alpha = 1 - \frac{1}{\lambda\,\mu(H)} .

The same :math:`\alpha` gives different rates on different boxes: :math:`\alpha = 0` is the coarsest
partition (:math:`\lambda = 1/\mu(H)`) and :math:`\alpha \to 1` gives ever finer ones; on the unit
square :math:`\lambda = 1/(2(1-\alpha))`. Scenarios declare the rate :math:`\lambda` together with
the domain :math:`H`; Spatialize's runner derives ``alpha`` from both
(:func:`spatialize.scenarios.runners.spatialize.alpha_from_rate`) and pins the box to the declared
domain by adding the domain's corners as extra queries.

Measured deviation of ``mondrian/spatialize-v1``
=================================================

Probability that a location at displacement :math:`h` from the centre of the box shares the
centre's cell, 20 000 trees, :math:`\alpha = 0.8`:

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

On a square the co-occurrence is 4–21 % below the closed form and depends on direction beyond the
:math:`\ell_1` distance; on an elongated box the uniform choice of axis makes the partition strongly
anisotropic. Closed-form scenarios therefore target ``mondrian/book``; run on
``mondrian/spatialize-v1`` they are **negative controls** (:doc:`statistics`). Changing Spatialize's
default process would change every existing result, so it is not done; an opt-in profile closer to
the theory is a possible future addition.

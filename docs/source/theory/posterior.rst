.. _theory-posterior:

##############################
Posterior analysis of the data
##############################

Classical exploratory analysis studies the data before any model, with summaries and the expert
knowledge of the analyst, who decides which values look wrong. Posterior analysis reads each datum
against what the rest of the data say about it. The evidence comes from the data themselves,
through the partitions and the decoder, with no variogram or model chosen beforehand. It serves the
quality control of a set of data, pointing to the values their surroundings do not support.

The law of a datum
==================

Write :math:`z_1, \dots, z_n` for the data at the locations :math:`x_1, \dots, x_n`, and
:math:`\Pi_1, \dots, \Pi_T` for the partitions of the ensemble, :math:`C_t(x)` being the cell of
:math:`\Pi_t` that holds :math:`x`. Cross-validation predicts each datum from the other data of its
cell, with the decoder :math:`g`,

.. math::

   m_i^{(t)} = g\big(x_i;\ \{(x_j, z_j) : j \ne i,\ x_j \in C_t(x_i)\}\big),
   \qquad t \in \mathcal T_i,

over the partitions :math:`\mathcal T_i` in which the cell of :math:`x_i` holds another datum
(:doc:`error`). The members give the law of the datum built without it,

.. math::

   \hat F_{-i}(z) = \frac{1}{|\mathcal T_i|} \sum_{t \in \mathcal T_i} \mathbf 1\{m_i^{(t)} \le z\}.

The datum takes no part in its law, so a value its neighbours do not support keeps all of its
surprise.

An isolated datum often sits alone in its cell, and under ``empty_cells="nan"`` such a partition
gives it no member. The *support* of the datum,

.. math::

   s_i = \frac{|\mathcal T_i|}{T},

estimates the probability that its cell holds another datum. Its complement :math:`1 - s_i` is the
residual weight of the block-mark model (:doc:`blockmark`), the share of the law the data cannot
reach. The policies ``"mark"`` and ``"coarsen"`` give every datum a member in every partition,
:math:`s_i = 1`.

Reading one datum
=================

Write :math:`\tilde F_i` for the law the readings use, :math:`\hat F_{-i}` corrected as the section
on calibration describes, and :math:`\tilde f_i` for its density. The position of the datum in its
law,

.. math::

   u_i = \tilde F_i(z_i),

tells how far into its tails the datum falls. Its two-sided tail probability,

.. math::

   p_i = \min\{1,\ 2\min(u_i,\ 1 - u_i)\},

is the p-value of the hypothesis that the datum was drawn from its law. When it was, :math:`u_i` is
uniform on :math:`[0, 1]` and :math:`\Pr(p_i \le a) = a`. The level of a datum is the widest
central interval

.. math::

   I_\alpha = \big[\tilde F_i^{-1}\big(\tfrac{1-\alpha}{2}\big),\ \tilde F_i^{-1}\big(\tfrac{1+\alpha}{2}\big)\big]

that leaves it out, for a few probabilities :math:`\alpha`. Since :math:`z_i \notin I_\alpha` exactly
when :math:`u_i < (1 - \alpha)/2` or :math:`u_i > (1 + \alpha)/2`, a datum drawn from its law falls
outside :math:`I_\alpha` with probability :math:`1 - \alpha`.

The log score of the datum,

.. math::

   \ell_i = -\log \tilde f_i(z_i),

measures its surprise with the strictly proper scoring rule the hyperparameter searches use
(:doc:`error`). Its expectation under the law is the entropy of the law,

.. math::

   \mathbb E_{Z \sim \tilde F_i}\big[-\log \tilde f_i(Z)\big] = H(\tilde F_i),

estimated on the sample :math:`y_{i1}, \dots, y_{iN}` of the law by
:math:`\hat H_i = -\frac1N \sum_j \log \tilde f_i(y_{ij})`. The *excess surprise*
:math:`\ell_i - \hat H_i` is near 0 for a datum typical of its law and large for one its law does
not expect, in nats.

Many data at once
=================

With :math:`n` data, a share :math:`1 - \alpha` of them falls outside :math:`I_\alpha` by chance
alone, even when every law is right. Listing the data of the outer level would therefore list about
:math:`(1 - \alpha)\,n` clean data. Spatialize flags the data with the Benjamini–Hochberg procedure,
which sorts the p-values :math:`p_{(1)} \le \dots \le p_{(n)}` and flags the :math:`k` smallest,

.. math::

   k = \max\Big\{j : p_{(j)} \le \frac{j}{n}\, q\Big\},

none when no index qualifies. When the p-values of the :math:`n_0` clean data are uniform and
independent, or positively dependent, the false discovery rate, the expected share of clean data
among the flagged ones, satisfies

.. math::

   \mathbb E\Big[\frac{V}{\max(R, 1)}\Big] \le \frac{n_0}{n}\, q \le q,

with :math:`R` the number of flags and :math:`V` the clean data among them. When every datum is clean, :math:`V = R`, so the bound says that some flag appears with probability at most :math:`q`.

Calibration comes first
=======================

The guarantee rests on laws whose probabilities are right, which the laws read by cross-validation
are not. Each member averages the data of a cell, so the members spread less than the data do
around them. Read as they are, the laws put many more data in their tails than their probabilities
say. The tails of the laws are also too light, since a few hundred members say little about values
beyond them. Counting the members cannot give a tail probability below
:math:`2/(|\mathcal T_i| + 1)`, while the smallest p-value must fall below :math:`q/n` for a single
flag, :math:`1.7 \cdot 10^{-4}` with :math:`n = 300` and :math:`q = 0.05`. Spatialize corrects the
laws in four steps before reading them.

**Scale.** The values may be read through a monotone map :math:`\phi` fitted to the data,
:math:`\phi(z) = z` by default. The Yeo–Johnson map,

.. math::

   \phi_\lambda(z) =
   \begin{cases}
   \big((1 + z)^\lambda - 1\big)/\lambda, & z \ge 0,\ \lambda \ne 0, \\
   \log(1 + z), & z \ge 0,\ \lambda = 0, \\
   -\big((1 - z)^{2-\lambda} - 1\big)/(2 - \lambda), & z < 0,\ \lambda \ne 2, \\
   -\log(1 - z), & z < 0,\ \lambda = 2,
   \end{cases}

with :math:`\lambda` fitted by maximum likelihood, makes a skewed variable more symmetric while
keeping how far an extreme value lies. Normal scores map the datum of rank :math:`r` to
:math:`\Phi^{-1}\big((r - \tfrac12)/n\big)`, linearly between the data and past them, which makes the
data normal but brings every extreme value close to the largest score, so a gross error stands out
less. The positions :math:`u_i` do not depend on a monotone map by themselves, only through the
corrections below, which act on its scale.

**Widening.** The members of each datum are widened to the spread of its :math:`k` nearest other
data :math:`N_k(i)` (12 by default), the ensemble widening of simulation (:doc:`ess`), with a robust
target variance,

.. math::

   \tau_i^2 = \Big(1.4826\ \operatorname{med}_{j \in N_k(i)} \big|z_j - \operatorname{med}_{l \in N_k(i)} z_l\big|\Big)^2,

the squared scaled median absolute deviation, so an erroneous neighbour does not widen a law enough
to hide another error. Each member becomes the mean of a component, a Gamma law for non-negative
values or a skew-normal law otherwise, whose shape follows the skewness of the neighbours, with
the variance

.. math::

   \nu_i = \tau_i^2 - \operatorname{Var}\big(m_i^{(t)}\big)

that the members lack, so that the mixture has mean and variance those of the members plus
:math:`\nu_i`, its variance being :math:`\tau_i^2`. A law already as wide as its target is left as it
is.

**One factor.** Every law's sample is spread around its median by one factor :math:`c`,

.. math::

   y_{ij} \mapsto \operatorname{med}_l y_{il} + c\,\big(y_{ij} - \operatorname{med}_l y_{il}\big),

with :math:`c` the solution of

.. math::

   \frac1n \sum_{i=1}^n \mathbf 1\big\{u_i(c) \in [0.05,\ 0.95]\big\} = 0.9,

found by bisection, so that 90 % of the data fall inside the central 90 % interval of their law. A
factor above 1 tells that the widened laws were still too narrow.

**Tails.** The law is carried past its sample :math:`y_{i1}, \dots, y_{iN}` by a kernel density,

.. math::

   \tilde f_i(z) = \frac{1}{N h_i} \sum_{j=1}^N K\Big(\frac{z - y_{ij}}{h_i}\Big),
   \qquad
   \tilde F_i(z) = \frac1N \sum_{j=1}^N \mathcal K\Big(\frac{z - y_{ij}}{h_i}\Big),

with :math:`\mathcal K` the distribution function of the kernel :math:`K` and Silverman's
bandwidth from the sample's own spread,
:math:`h_i = 0.9 \min\big(\hat\sigma_i,\ \mathrm{IQR}_i/1.34\big)\, N^{-1/5}`. The kernel is
Student's :math:`t` with :math:`\nu = 3` degrees of freedom by default, whose tails decay as
:math:`|u|^{-(\nu + 1)}`, so a value far beyond every member gets a p-value as small as its distance
warrants. Gaussian kernels give the lightest tails. The alternative ``tails="gpd"`` standardises each
law by its median and interquartile range, :math:`w_{ij} = (y_{ij} - \operatorname{med}_i)/\mathrm{IQR}_i`,
pools the excesses over the 90 % quantile of every law, :math:`w_{ij} - \omega_i^{+}`, and fits to them
one generalized Pareto law of shape :math:`\xi` and scale :math:`\beta` by maximum likelihood, the
lower tail likewise. Above the threshold the tail is

.. math::

   \Pr(W_i > w) = 0.1\,\Big(1 + \xi\, \frac{w - \omega_i^{+}}{\beta}\Big)^{-1/\xi},
   \qquad w > \omega_i^{+},

the share of the sample being used below it.

Version 1.2 controlled the tails another way, adding the datum to its own sample. The law then
always reached the datum, which kept its tail probability away from 0, at the price of the surprise
the analysis looks for. The tail model replaces that device.

Reading the calibration
-----------------------

The calibration is read on the data themselves, through the coverage of the central intervals,

.. math::

   \hat C(\alpha) = \frac1n \sum_{i=1}^n \mathbf 1\big\{u_i \in \big[\tfrac{1-\alpha}{2},\ \tfrac{1+\alpha}{2}\big]\big\},
   \qquad
   z(\alpha) = \frac{\hat C(\alpha) - \alpha}{\sqrt{\alpha(1 - \alpha)/n}},

compared with the probability :math:`\alpha` through its binomial standard error, and through the
Kolmogorov–Smirnov distance of the positions :math:`u_i` from the uniform law. Spatialize reports
:math:`\hat C(\alpha)` before and after the factor, and calls the tails of the laws too narrow when
:math:`z(\alpha) < -3` for some :math:`\alpha \ge 0.9`, too wide when :math:`z(\alpha) > 3`. The
tails govern the flags, so they carry the verdict. The centre, the intervals of smaller probability,
is judged apart, since a law may be too wide in its centre with tails of the right weight, as on
the drill holes of the Andes data bundled with Spatialize, whose 50 % intervals held 68 % of the
data and whose 90 % and 99 % intervals held 90.5 % and 98.5 %.

What the corrections achieve
----------------------------

The corrections were measured on synthetic fields, without a guarantee for other data. Each field
holds 300 data with values :math:`\sin(2\pi x) + 0.5\cos(2\pi y)` plus Gaussian noise of standard
deviation 0.1, or the exponential of that sum with noise 0.3 (a lognormal field). The data are
placed uniformly, or half of them in four clusters. Each field is read clean and with three planted
errors, a value multiplied by 10, a value shifted by 3 standard deviations and a value shifted down
by 2.5 (multiplied by 0.1 on the lognormal field). Eight fields of each kind were read with IDW on
Mondrian partitions (``alpha`` = 0.85, 300 partitions), flags at :math:`q = 0.05`.

.. list-table:: Clean fields with a false flag, and planted errors found
   :header-rows: 1
   :widths: 32 17 17 17 17

   * - setting
     - Gaussian, uniform
     - Gaussian, clustered
     - lognormal, uniform
     - lognormal, clustered
   * - Gaussian kernels, no correction
     - 8/8 · 23/24
     - 8/8 · 23/24
     - 8/8 · 23/24
     - 8/8 · 24/24
   * - Gaussian kernels
     - 8/8 · 23/24
     - 8/8 · 21/24
     - 7/8 · 15/24
     - 6/8 · 14/24
   * - Student-t kernels (default)
     - 5/8 · 22/24
     - 4/8 · 17/24
     - 1/8 · 12/24
     - 4/8 · 12/24
   * - generalized Pareto tails
     - 0/8 · 19/24
     - 3/8 · 13/24
     - 5/8 · 5/24
     - 7/8 · 6/24
   * - Student-t, Yeo–Johnson scale
     - 5/8 · 22/24
     - 5/8 · 20/24
     - 3/8 · 5/24
     - 4/8 · 6/24
   * - Student-t, normal scores
     - 4/8 · 20/24
     - 3/8 · 13/24
     - 3/8 · 4/24
     - 2/8 · 3/24

Each cell gives the clean fields with at least one flag, out of 8, and the planted errors flagged,
out of 24. Without correction the laws flag between 140 and 1 041 clean data over the eight fields
with planted errors, and the coverage of their central 99 % interval falls to between 0.63 and 0.91.
With the corrections that coverage lies between 0.97 and 0.99.

On a clean field every flag is a false discovery, so the procedure promises that some flag appears
on at most 5 % of clean fields. No setting keeps that promise on every kind of field, the tails of
the laws remaining too light at the 99 % level. Heavier tails trade false flags for power. Clustered
data and skewed values are the hardest cases. The calibration of the laws is therefore part of the
result. Spatialize reports it, with a verdict, in whose light the flags are to be read.

An error or an unrepresented place
==================================

A value its neighbours do not support may be an error, a sample switched in the laboratory or a
decimal point misplaced, or it may record a part of the domain the other data do not represent, a
thin seam or a stream that only runs in flood. Nothing in the number tells the two apart, since the
difference lies in the provenance of the datum.

The neighbours of a surprising datum give some evidence, through their own positions
:math:`u_j`. Its *shift* is their mean signed position,

.. math::

   \sigma_i = \frac1k \sum_{j \in N_k(i)} (2u_j - 1) \in [-1,\ 1],

over its :math:`k` nearest other data (8 by default), near 0 when the neighbours are drawn from their
laws. An erroneous value takes part in the laws of its neighbours, pulling them towards it, so the
neighbours fall on the other side, which gives a datum far above its law a negative shift. Data of a part of
the domain the laws do not represent are surprised together, a datum above its law having a positive
shift. Inside such a part, a datum's law is built from neighbours of the same part, so the datum
itself is hardly surprised while its neighbourhood is. The shift and the datum's own position,
read together, tell the two cases apart.

On replicate fields of 300 data with a raised square of about 12 data and one isolated error
(scenario :ref:`scenario-P12`), the shift of the square's data exceeded that of the other data by
0.22 (standard deviation 0.12 over 20 fields), while the isolated error, always in the far upper
tail of its law, had a shift of -0.35 (standard deviation 0.19).

The *coherence*, the share of the neighbours on the same side as the datum,

.. math::

   \kappa_i = \frac1k \sum_{j \in N_k(i)} \mathbf 1\big\{\operatorname{sign}(u_j - \tfrac12) = \operatorname{sign}(u_i - \tfrac12)\big\},

follows a binomial law of :math:`k` trials of probability :math:`\tfrac12`, divided by :math:`k`, when
the neighbours are drawn from their laws independently of the datum. It is a weaker reading of the
same evidence, the square's data exceeding the others by 0.06 to 0.10 only, since the data of the
square are themselves hardly surprised. Posterior analysis gives the evidence and an order, from the
most to the least surprising datum. The analyst, who knows the provenance, decides.

How much of the domain each datum represents
============================================

Data are often taken where the values are high or of interest, so the plain summaries of the values
lean towards those values. Classical declustering weighs each datum by the area it represents, with
a grid of cells whose size the analyst chooses. The partitions give that weight without a choice.
Each partition shares the domain among the data, every cell giving its area equally to the data it
holds, the area of the cells without data being shared out in proportion. With :math:`n_t(C)` the
number of data in the cell :math:`C` of :math:`\Pi_t` and :math:`|C|` its area, the *declustering
weight* of datum :math:`i` is

.. math::

   w_i = \frac1T \sum_{t=1}^T \frac{|C_t(x_i)|}{n_t\big(C_t(x_i)\big)\ \sum_{C \in \Pi_t,\ n_t(C) > 0} |C|},
   \qquad \sum_{i=1}^n w_i = 1.

The areas are estimated by the share of :math:`M` locations drawn uniformly in the box of the
partitions (20 000 by default) that each cell holds. A datum in a dense cluster shares small cells
with many others and weighs little, an isolated datum in a large cell weighs much. The declustered
summaries weigh the values by :math:`w`,

.. math::

   \bar z_w = \sum_i w_i z_i,
   \qquad
   s_w^2 = \sum_i w_i (z_i - \bar z_w)^2,

the quantiles interpolating the cumulative weights at the midpoints of the sorted values,
:math:`\sum_{j \le r} w_{(j)} - \tfrac12 w_{(r)}`.

On a field with half of its 300 data taken where the values exceed 0.8, the plain mean of the
values was 0.60 and the declustered one 0.20, for a field whose mean over the domain is close to 0
apart from a raised patch of the example. The data of the preferential half weighed on average half
as much as the others.

Two further readings complete the picture. The *proportional effect* compares the width of each
datum's law with its centre through the rank correlation of Spearman between the medians and the
widths of the central 90 % intervals, both in the units of the values. A strong positive correlation
shows laws that widen with the values, as for skewed variables, which a transformed scale may then
suit. Co-located data, the pairs with :math:`\lVert x_i - x_j \rVert \le \varepsilon` for a tolerance
:math:`\varepsilon`, are listed with their values, since two different values at one location cannot
both be right while each sees the other as a neighbour.

In Spatialize
=============

Posterior analysis is :func:`~spatialize.gs.spa.posterior_audit`, whose result
:class:`~spatialize.gs.spa.PosteriorAudit` holds the readings of every datum
(:doc:`../reference/spa`).

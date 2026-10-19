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

estimated from :math:`N` draws :math:`y^*_{i1}, \dots, y^*_{iN}` of the density
:math:`\tilde f_i` itself, as many as the sample of the law holds,

.. math::

   \hat H_i = -\frac1N \sum_{j=1}^N \log \tilde f_i\big(y^*_{ij}\big).

The draws come from the density, not from the sample it is built on, since the density read at the
points of its own sample counts each point's own kernel and comes out too high. The *excess
surprise* :math:`e_i = \ell_i - \hat H_i` then has mean 0 for a datum drawn from its law, while a
datum its law does not expect has a large one. The logarithms are natural by default, the readings
being in nats. With ``base=2`` they are in bits, one bit less of entropy for each halving of the
spread of a law of given shape.

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
   \qquad w > \omega_i^{+}.

Between the two thresholds the law follows the kernel density of the sample with Gaussian kernels,
rescaled to hold the probability 0.8. The law spliced from the three parts has one density, which
the positions, the p-values, the log scores and the entropy all read. With a negative shape
:math:`\xi` a tail ends at :math:`\omega_i^{+} - \beta/\xi`, beyond which the law gives no value at
all.

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

The information check
----------------------

The coverage reads a few intervals of every law. The log score, the strictly proper score the
hyperparameter searches minimise (:doc:`error`), reads its whole density, each part weighted by how
probable the law makes it. For a datum drawn from its law the excess surprise :math:`e_i` has
mean 0 and the variance

.. math::

   \operatorname{Var}(e_i) = v_i \Big(1 + \frac1N\Big),
   \qquad
   v_i = \operatorname{Var}_{Z \sim \tilde f_i}\big[\log \tilde f_i(Z)\big],

the first term from the datum, the second from the draws that estimate the entropy, :math:`v_i`
being read on the same draws. The information check compares the mean excess with its standard
error,

.. math::

   \bar e = \frac1n \sum_{i=1}^n e_i,
   \qquad
   z_e = \frac{\bar e}{\frac1n \big(\sum_i \operatorname{Var}(e_i)\big)^{1/2}},

and calls the laws too narrow or off-centre when :math:`z_e > 3`, too wide when :math:`z_e < -3`.
A positive mean excess says that the data are less probable under their laws than the laws expect,
because the laws are too narrow, centred away from the data or too light in their tails. The
standard error treats the data as independent, as the binomial errors of the coverage do, while
neighbouring laws share data. A few gross errors can carry the mean on their own, their excess
growing with their distance to the law, slowly with Student-t kernels and as its square with
Gaussian ones, so the excess of each datum tells whether the mean comes from the bulk of the data. A datum beyond
the end of a bounded Pareto tail has an infinite excess, its law giving no value there. Such data
are counted apart, outside the mean, and any of them makes the laws too narrow.

The corrections on synthetic fields
-----------------------------------

The corrections were measured on synthetic fields, without a guarantee for other data. Each field
holds 300 data with values :math:`\sin(2\pi x) + 0.5\cos(2\pi y)` plus Gaussian noise of standard
deviation 0.1, or the exponential of that sum with noise 0.3 (a lognormal field). The data are
placed uniformly in the unit square, or half of them in four clusters, each datum of a cluster
drawn from a normal law of standard deviation 0.04 around one of four centres placed uniformly in
:math:`[0.1, 0.9]^2`. Each field is read clean and with three planted errors, a value multiplied by
10, a value shifted by 3 standard deviations and a value shifted down by 2.5 (multiplied by 0.1 on
the lognormal field). Eight fields of each kind were read with IDW on Mondrian partitions
(``alpha`` = 0.85, 300 partitions), flags at :math:`q = 0.05` (measured with version 1.3, October
2026).

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
     - 8/8 · 24/24
     - 8/8 · 24/24
     - 8/8 · 24/24
   * - Gaussian kernels
     - 8/8 · 23/24
     - 7/8 · 22/24
     - 5/8 · 13/24
     - 8/8 · 17/24
   * - Student-t kernels (default)
     - 1/8 · 23/24
     - 0/8 · 22/24
     - 1/8 · 11/24
     - 2/8 · 16/24
   * - generalized Pareto tails
     - 2/8 · 22/24
     - 2/8 · 19/24
     - 4/8 · 10/24
     - 6/8 · 15/24
   * - Student-t, Yeo–Johnson scale
     - 1/8 · 23/24
     - 0/8 · 21/24
     - 1/8 · 2/24
     - 0/8 · 3/24
   * - Student-t, normal scores
     - 4/8 · 18/24
     - 0/8 · 15/24
     - 1/8 · 4/24
     - 2/8 · 10/24

Each cell gives the clean fields with at least one flag, out of 8, and the planted errors flagged,
out of 24. Without correction the laws flag between 212 and 1 091 clean data over the eight clean
fields of a kind, and the coverage of their central 99 % interval falls to between 0.57 and 0.91.
With the default corrections that coverage lies between 0.96 and 1.

On a clean field every flag is a false discovery, so the procedure promises that some flag appears
on at most 5 % of clean fields. The default comes close, with 4 of the 32 clean fields flagged, and
no setting keeps the promise on every kind of field. Heavier tails trade false flags for power.
Skewed values are the hardest case, where the planted errors are found least often. A transformed
scale loses most of them there, a decimal error looking ordinary once the values are brought
towards normal.

The information check sees what the coverage misses. With Gaussian kernels the coverage called the
tails calibrated on 26 of the 32 clean fields, while the mean excess surprise exceeded 3 standard
errors on 19 of them, the light tails of the kernels making the few data beyond them very
improbable. With the default Student-t kernels the information check stayed between -3.5 and 2.1 on
every clean field. With the generalized Pareto tails it exceeded 3 on 2 of them, both lognormal, the
kind of field where those tails flag clean data most often. The calibration of the laws is therefore part of the
result. Spatialize reports it, with a verdict, in whose light the flags are to be read.

Flags of clean replicates
-------------------------

Since no setting keeps the promise of the procedure everywhere, Spatialize measures the false flags
on the data at hand. It draws replicates free of errors, every datum :math:`z^*_i` drawn on its own
from its law :math:`\tilde F_i`, the law the readings use, audits each replicate with the same
partitions, decoder and settings, and counts its flags. Every one of them is a false discovery. The
replicates keep the centres of the laws and the spread around them, the spatial pattern coming from
the centres.

An erroneous datum pulls the laws of its neighbours towards it, so replicates drawn from those laws
vary more than clean data would, which inflates their flags. The replicates are therefore drawn
after each flagged datum is replaced by the median of its law, the laws being computed again. With
:math:`F` the flags of the data and :math:`F_1, \dots, F_R` those of :math:`R` replicates, the mean
of the :math:`F_r` estimates the false flags the data would get without errors, the share of
replicates with some flag the chance that clean data get one, and

.. math::

   \hat p = \frac{1 + \#\{r : F_r \ge F\}}{R + 1}

is the Monte Carlo p-value of the hypothesis that the data are as clean as the replicates. The
count weighs every flag alike, so one datum a hundred spreads away counts as one flag. Read through
the most surprising datum instead, with :math:`Q` the smallest Benjamini–Hochberg adjusted p-value
of the data and :math:`Q_r` that of replicate :math:`r`, the same hypothesis has the p-value

.. math::

   \hat p_{\min} = \frac{1 + \#\{r : Q_r \le Q\}}{R + 1},

which weighs how far the strongest datum lies. Neither can fall below :math:`1/(R + 1)`.

On the fields of the table, four of each kind, with 20 replicates each, the replicates predicted the
share of clean fields with a false flag, 0.20 against an observed 0.19 with the default Student-t
kernels and 0.92 against 0.94 with Gaussian kernels, and the mean number of false flags, 0.29
against 0.25 and 3.0 against 3.1. On the same fields with three planted errors, the replicates drawn
from the laws as they are gave 0.47 fields with a flag among the clean data, while those drawn
after the flagged data were replaced gave 0.23, the rate of the clean fields. Drawing the replicates
as a law's median plus the residual of another datum at random overestimated the false flags
twofold to fivefold, since the residuals of the errors went into the replicates.

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
0.23 (standard deviation 0.13 over 20 fields), while the isolated error, always in the far upper
tail of its law, had a shift of -0.36 (standard deviation 0.19).

The *coherence*, the share of the neighbours on the same side as the datum,

.. math::

   \kappa_i = \frac1k \sum_{j \in N_k(i)} \mathbf 1\big\{\operatorname{sign}(u_j - \tfrac12) = \operatorname{sign}(u_i - \tfrac12)\big\},

follows a binomial law of :math:`k` trials of probability :math:`\tfrac12`, divided by :math:`k`, when
the neighbours are drawn from their laws independently of the datum. It is a weaker reading of the
same evidence, the square's data exceeding the others by 0.06 to 0.10 only, since the data of the
square are themselves hardly surprised. Posterior analysis gives the evidence and an order, from the
most to the least surprising datum. The analyst, who knows the provenance, decides.

The evidence behind each law
============================

A law may rest on little evidence in two ways. An isolated datum often sits alone in its cell, so
its law rests on few partitions, a low support :math:`s_i`. A datum whose neighbours disagree gets a
law nearly as wide as the distribution of all the data, even when every partition defines it. The
entropy measures that width without assuming a shape. The *information gain*

.. math::

   g_i = H(\tilde f_0) - H(\tilde f_i)

compares it with the entropy of the distribution of all the data, :math:`\tilde f_0` being the
kernel density of the values :math:`z_1, \dots, z_n` with the kernels of the laws, its entropy
estimated from draws as for the laws. An entropy depends on the units of the values, shifting by
:math:`\log a` when they are multiplied by :math:`a`, while the gain, a difference of two
entropies, does not, so gains can be compared across variables. In bits, a gain of one halves the
spread of a law of given shape. A gain near 0, or below it, says that the other data tell no more
about the datum than the distribution of all the data does. The laws are read after their
widening and the factor of the calibration, which lower every gain.

Support and gain separate the two kinds of weak evidence. An isolated datum has a low support,
a datum among disagreeing neighbours a low gain with a high support.

A second variable
=================

Two variables measured on one sample share what the sample holds, a mineral grain, a patch of
polluted soil, the water of one bottle. The neighbours of the sample do not see that part, which
widens the law of a datum built from them. The other variable at the same location sees it, so a
value plausible among its neighbours may be implausible given the other variable, a shifted value,
two values swapped between samples or a unit misread in one variable.

The law of :math:`u_i` given the covariate :math:`v_i` is built from the pairs :math:`(u_j, v_j)`
of the other data. Each pair weighs

.. math::

   c_{ij} = \pi_{ij}\, \exp\Big(-\frac{(v_j - v_i)^2}{2 h_i^2}\Big),

with :math:`\pi_{ij}` the share of partitions, of granularity 0.9 by default, in which
:math:`x_j` shares the cell of :math:`x_i`, and :math:`h_i` the larger of Silverman's bandwidth of
the covariate and the distance from :math:`v_i` to its 20th nearest :math:`v_j`, so that the law
always rests on a few pairs. A line fitted to the pairs by weighted least squares gives the centre
:math:`\hat u_i = \sum_j \ell_j u_j` at :math:`v_i`, and its residuals :math:`r_j` the shape,

.. math::

   \tilde f(u \mid v_i) = \sum_j \frac{c_{ij}}{\sum_k c_{ik}}\, \frac1{b_i}\,
   K_\nu\Big(\frac{u - \hat u_i - r_j}{b_i}\Big),
   \qquad
   b_i^2 = h_{r,i}^2 + s_i^2 \sum_j \ell_j^2,

with :math:`K_\nu` the Student-t kernel of 10 degrees of freedom, :math:`h_{r,i}` Silverman's
bandwidth of the weighted residuals and :math:`s_i^2` their variance. The last term, the variance
of the fitted centre, widens the law where the pairs do not reach :math:`v_i`, at the extremes of
the covariate, where the line would otherwise be extended beyond its data. The datum is then read
as in the rest of the analysis, by its position, its tail probability, a Benjamini–Hochberg flag
and its log score.

An error among the pairs pulls the line of its neighbours, widening their residuals, which hides
their own errors. The data the audit of the values flags therefore take no part in the others'
laws. Nor do, in a further reading, the data flagged given the covariate, the readings being
repeated until the flags no longer change, three times at most. A robust line, which gives no
weight to the pairs far from it, fails here. It treats the smaller of two
branches of a relation as errors, so with a link through a square it flagged 14 of 16 clean
fields.

The covariate gives no power where it says nothing about the datum. The *gain*, the entropy of the
law of :math:`u` among the pairs the partitions weigh minus that of the law given :math:`v_i`,
measures what it says. It is near 0 for an unrelated covariate, while it exceeds the mutual
information of the two variables at a location, since the first law also holds the variation of the
field within the neighbourhood.

On synthetic fields of 300 data, the two variables sharing a component of the sample of standard
deviation 0.5 besides measurement errors of 0.15, the reading flagged 1 of 64 clean fields. With a
linear link it found a value shifted by 1 in 13 of 16 fields and each of two swapped values in 8
and 10 of 16, where the audit of the values alone found none of them. Five of its 50 flags fell on
clean data, a share the false discovery rate allows once true errors are flagged. A link through the square of the
shared component tells its size and not its sign, which left little power. A link whose sign
changes across the domain was followed only in part, the partitions of 300 data mixing the two
relations near their border. Kernels of 5 degrees of freedom lost power, while Gaussian kernels
flagged a quarter of the clean fields with a linear link.

From readings to actions
========================

The readings combine into one list of actions. A gross error pulls the laws of its neighbours
towards it, so neighbours far on the other side of their laws get flagged with it, while a second
error beside it may hide. The suspects are therefore found by cleaning. Each flagged datum is
replaced by the median of its law. With the laws computed again, the original values are read
against them. A datum's law never holds the datum, so every datum is then read against neighbours
free of the suspects. The flags of that reading become the suspects, the cleaning repeating until
they no longer change. The laws keep the spread factor of the audit through the cleaning, so they
change through the data alone. A datum flagged at first but not after cleaning was surprising only
beside a suspect.

A part of the domain the other data do not represent cannot be told from errors by the flags. Its
data support each other, so the ensemble represents it, flagging at most its edge. On 40 synthetic
fields with two planted errors and a raised patch, only 9 patches had a datum flagged, one or two
each, as for an error. The cleaning kept 58 of the 60 errors and found 3 more that their neighbours
hid, while it cleared 6 of the 12 clean data flagged beside an error.

The list therefore recommends measuring a suspect again, never removing it. A measurement taken
again confirms a true value, that of a part of the domain the other data do not represent included,
while it corrects an error. Neighbouring suspects surprised the same way form a group to check, a
batch of samples or such a part. A datum inconsistent with a second variable measured with it, or
co-located with another of a different value, is listed too, as is a law resting on little evidence,
a place to sample more. The effect of measuring again is read on the cross-validated predictions of
the other data, with the suspects replaced by the medians of their laws. With the replicates of
:meth:`~spatialize.gs.spa.PosteriorAudit.null_flags`, the list says whether its flags exceed what
clean data get or give an order of review only.

The share of the domain behind each datum
=========================================

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

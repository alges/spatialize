.. _theory-other:

#################
Further readings
#################

The ensemble supports other readings, each with its own module. The analysis of the data against
the laws the other data give has its own page (:doc:`posterior`), and so do categorical variables
(:doc:`categorical`).

Information measures
====================

The differential entropy of the predictive law at a location,

.. math::

   h(v) = -\int \hat f_v(z) \log \hat f_v(z)\, dz,

summarises its uncertainty in one number without assuming the law is symmetric or unimodal. For two
variables :math:`U` and :math:`V` measured over the same region, the mutual information of their
local laws,

.. math::

   I(U; V)(x) = h_U(x) + h_V(x) - h_{U,V}(x),

measures how much knowing one tells about the other, location by location, and vanishes when they
are independent there. Spatialize estimates the joint density of the members of :math:`U` and :math:`V` with Mondrian
partitions of the plane of values, uniform within each cell, and integrates it over one variable to
obtain each marginal,

.. math::

   \hat f_U(u) = \sum_{c\,:\,u \in c_U} \frac{p_c}{|c_U|},

with :math:`p_c` the share of members in cell :math:`c` and :math:`c_U` its side along :math:`U`. The
three entropies then come from one density, so the estimate is never negative. Members of cells without data, undefined under ``empty_cells="nan"`` (:doc:`blockmark`), carry no
value and are left out, the shares :math:`p_c` being taken over the valid members. A location with
fewer than two valid members gets no entropy (NaN). For the mutual information, a pair counts only
when both of its members are valid.

Spatialize reads these laws in two ways. The entropy can come from the members themselves, their
density estimated with random partitions of their range (``method="partition"``), or from the law
ensemble spatial simulation fits at each location, widened to the spread of the nearest data
(``method="ess"``, :doc:`ess`). With an averaging decoder the members' spread follows the granularity,
so only the second follows the field. A drawing decoder gives members whose spread is that of the data,
so both read it.

The mutual information needs the joint law of :math:`U` and :math:`V` at a location, which neither
ensemble gives, since each is built on its own data. The default (``method="colocated"``) pairs the
two variables where one of them is measured, the datum of one with the other predicted there by its
own ensemble, and weighs each pair, for a target location, by the share of partitions in which the
pair's location and the target share a cell, so the partitions act as adaptive neighbourhoods. The
mutual information is then that of a kernel density of the weighted pairs, minus that of the same pairs
with :math:`V` permuted, the bias of the density under independence. The pairs hold the variables' own
relation, so independent variables give a value near zero and a dependence that is not monotone is
seen too. The alternative (``method="members"``) reads the members of both ensembles, drawn on shared
partitions, through random partitions of the plane of values. The members are averages, which blur the
relation of the variables, and they co-vary through the size of the cells they share whatever that
relation, so this estimate sees a link faintly and a non-monotone one hardly at all.

The variogram read from the ensemble
====================================

Classical geostatistics fits a variogram to the data, one direction at a time, before any map is
drawn. The ensemble can run the other way round, reading the variogram from the estimate once it
exists. Three readings of one run give it, the directions of anisotropy from the map, the correlation
lengths from the members, and a bound on the measurement error from the data and the partitions.

The setting
-----------

The data are :math:`z_1, \dots, z_n` at the locations :math:`x_1, \dots, x_n` of a domain
:math:`D \subset \mathbb R^d`, with :math:`d = 2` or :math:`3`, and :math:`G` is a regular grid of nodes
over :math:`D`. A Pareto search of adaptive IDW (:doc:`error`) chooses a granularity, the knee of its
frontier, at which two ensembles run on :math:`G`. The first, with adaptive IDW, gives the map
:math:`\hat z(g)`, the mean of its members at each node. The second, with the weighted draw of the same
decoder, gives the members :math:`z^{(1)}, \dots, z^{(T)}`, each a field over :math:`G` drawn under its
own partition, so that given the data they are independent replicates of the fitted law
(:doc:`esi`).

The target is a variogram with geometric anisotropy,

.. math::

   \gamma(h) = \sigma^2 \big( 1 - \rho\big( \lVert \Lambda^{-1} U^{\top} h \rVert \big) \big),
   \qquad \Lambda = \mathrm{diag}(L_1, \dots, L_d),

with :math:`U` the orthogonal matrix whose columns :math:`u_1, \dots, u_d` are the principal directions,
:math:`L_1 \ge \dots \ge L_d` the correlation lengths along them and :math:`\rho` a correlation function
with :math:`\rho(0) = 1`. For the exponential model :math:`\rho(r) = e^{-r}`, so :math:`L_k` is the scale
along :math:`u_k` and :math:`3 L_k` the practical range, where the correlation has fallen to
:math:`e^{-3} \approx 0.05`.

The directions
--------------

The structure tensor of the map is the average over the grid of the smoothed outer product of its
gradient,

.. math::

   J = \frac{1}{|G|} \sum_{g \in G} \big( K_\sigma * \nabla \hat z \, \nabla \hat z^{\top} \big)(g),

with the gradient read by finite differences in the coordinates of the domain and :math:`K_\sigma` a
Gaussian kernel of width :math:`\sigma` grid cells. Its eigendecomposition
:math:`J = \sum_k \mu_k\, v_k v_k^{\top}`, with :math:`\mu_1 \le \dots \le \mu_d`, gives the principal
directions, :math:`v_1` the direction along which the map varies least.

The reason is a property of stationary fields. For a field with covariance :math:`C` that is twice
differentiable at the origin, :math:`\mathbb E[\nabla Z \nabla Z^{\top}] = -\nabla\nabla^{\top} C(0)`, which
under the geometric anisotropy above is proportional to :math:`U \Lambda^{-2} U^{\top}`. Its eigenvectors
are the principal directions, the smallest eigenvalue belonging to the longest length. The map is a
smoothed version of the field, its covariance twice differentiable even where the field's is not. An
isotropic smoothing leaves those eigenvectors in place, so :math:`v_k \approx u_k`. The eigenvalues do not
carry over, since the smoothing changes them, which is why the lengths are read elsewhere. The coherence

.. math::

   c = \frac{\mu_d - \mu_1}{\mu_d + \mu_1} \in [0, 1]

measures how far the tensor departs from a multiple of the identity, 0 when no direction is preferred, the
directions being then undetermined.

The lengths
-----------

Along a direction :math:`u` the semivariogram of member :math:`t` at lag :math:`h` is

.. math::

   \gamma_t(h; u) = \frac{1}{2\,|G_h|} \sum_{g \in G_h} \big( z^{(t)}(g) - z^{(t)}(g + h u) \big)^2,

where :math:`G_h` holds the nodes whose shift :math:`g + hu` stays inside the grid, the shifted value being
read by linear interpolation between nodes. With :math:`s_t^2` the variance of the member over the grid,
its correlation along :math:`u` is :math:`r_t(h; u) = 1 - \gamma_t(h; u)/s_t^2`. Its correlation length is
the first lag where that correlation falls below :math:`e^{-1}`,

.. math::

   L_{t,k} = \inf \{ h > 0 : r_t(h; v_k) < e^{-1} \},

interpolated linearly between the two lags around the crossing. The length along :math:`v_k` is the mean
:math:`L_k = T^{-1} \sum_t L_{t,k}`, and the quantiles of the :math:`L_{t,k}` over the members give an
interval, which describes the spread of the ensemble without being a calibrated confidence interval. The
lags run to half the extent of the grid, so a member whose correlation never crosses :math:`e^{-1}` within
them gives only a lower bound, which the reading flags.

The members are read instead of the map because the map is a mean. Averaging over partitions smooths the
field, which lengthens its correlation, above all across the structures, while a drawing member places
data values in the cells of one partition and so keeps the variability of the data. For the exponential
model the crossing at :math:`e^{-1}` is exactly :math:`L_k`, since :math:`\rho(L_k / L_k) = e^{-1}`. The
variogram read is then

.. math::

   \hat\gamma(h) = \hat\sigma^2 \big( 1 - \exp( - \lVert \hat\Lambda^{-1} \hat U^{\top} h \rVert ) \big),

with :math:`\hat U = (v_1, \dots, v_d)`, :math:`\hat\Lambda = \mathrm{diag}(L_1, \dots, L_d)` and
:math:`\hat\sigma^2` the variance of the data.

The residual weight
-------------------

The theory has no nugget of the field. Under the block-mark model the covariance
:math:`\kappa_2\, e(\{x, y\})` is continuous at the origin, since :math:`e(\{x, y\}) \to 1` as the locations
coalesce. The part a nugget plays in a classical fit falls to the residual term of the single-point law
(:doc:`blockmark`),

.. math::

   F_v = \sum_i w_i(v)\, \delta_{z_i} + \Big( 1 - \sum_i w_i(v) \Big)\, \nu,

   1 - \sum_i w_i(v) = \Pr_\Pi\big( \text{the cell of } v \text{ holds no datum} \big),

the weight the law gives to the mark law :math:`\nu` because the data say nothing about :math:`v`. It is a
property of the location and its data, large where the data are thin, small where they are dense. The
reading estimates it at every node as the share of the members' partitions in which the node's cell holds
no datum.

A bound on the measurement error
--------------------------------

Suppose each datum carries an error, :math:`z_i = Y(x_i) + \varepsilon_i`, with the
:math:`\varepsilon_i` independent of one another, of the field and of the partitions, centred and of
variance :math:`\tau^2`. Under the block-mark model with such an error the covariance across distinct
locations keeps its form :math:`\kappa_2\, e(\{x, y\})`, while the error adds to the variance at a single
location, :math:`\operatorname{Var} Z(x) = \kappa_2 + \tau^2`. The order-two estimator of the covariance
(:math:`K_Q(x, y) = \kappa_2\, e(\{x, y\})` for :math:`x \ne y`) therefore misses the error everywhere
except where the two locations coincide, which leaves the data sharing a cell as the only place to read it.

Fix a partition :math:`\pi` and a cell :math:`c` holding the data of indices :math:`I_c`, with
:math:`m_c = |I_c|` and mean :math:`\bar z_c`. Writing :math:`z_i = Y_i + \varepsilon_i`, the sum of squared
deviations splits into the field's part, the error's part and a cross term,

.. math::

   \sum_{i \in I_c} (z_i - \bar z_c)^2
   = \sum_{i \in I_c} (Y_i - \bar Y_c)^2 + \sum_{i \in I_c} (\varepsilon_i - \bar\varepsilon_c)^2
   + 2 \sum_{i \in I_c} (Y_i - \bar Y_c)(\varepsilon_i - \bar\varepsilon_c).

Given :math:`\pi` and the field, the cross term has expectation zero, the errors being centred and
independent of both, while :math:`\mathbb E \sum_{i \in I_c} (\varepsilon_i - \bar\varepsilon_c)^2
= (m_c - 1)\, \tau^2`, the expectation of a sum of squared deviations of :math:`m_c` independent variables
of variance :math:`\tau^2`. Pooling the cells that hold two data or more,

.. math::

   W(\pi) = \frac{\sum_c \sum_{i \in I_c} (z_i - \bar z_c)^2}{\sum_c (m_c - 1)},

   \mathbb E\big[ W(\pi) \mid \pi \big] = \tau^2 + D(\pi),

   D(\pi) = \frac{\sum_c \mathbb E \sum_{i \in I_c} (Y_i - \bar Y_c)^2}{\sum_c (m_c - 1)} \ge 0 .

The term :math:`D(\pi)` is the field's own variation inside the cells, the encoder error of
:doc:`error` read on the data. Under the block-mark model the field is constant in each cell, so
:math:`D = 0` and :math:`W(\pi)` is unbiased for :math:`\tau^2`. On any other field :math:`D \ge 0`, so

.. math::

   \tau^2 \le W(\lambda) := \mathbb E_\Pi\big[ W(\Pi) \big]

at every granularity :math:`\lambda`, an inequality that needs no model of the variogram. As the cells
shrink, the field's variation inside them vanishes, which is the fine-cell sufficiency of :doc:`error`, so
:math:`W(\lambda) \downarrow \tau^2` as :math:`\lambda \to \infty`.

The reading estimates :math:`W(\lambda)` by the mean of :math:`W(\pi_t)` over :math:`T` partitions drawn
independently, whose standard deviation over :math:`\sqrt T` is its standard error, the partitions being
independent given the data. It does so on a ladder of increasing granularities and reports the smallest
value as the bound. The ladder stops once the cells hold too few shared data, fewer than 30 degrees of
freedom :math:`\sum_c (m_c - 1)` or fewer than 15 % of the data, the estimate then resting on few
pairs while the partitions grow costly. Two caveats go with the bound. Taking a minimum over several
granularities makes it slightly optimistic when they hold the same value, as on a field without
structure inside the cells. With a few hundred data the cells empty before :math:`D(\lambda)` vanishes, so
the bound stays loose, about a fifth of the variance on data without error in two dimensions.

Limits measured
---------------

A study on synthetic and real fields measured the limits of the reading. The direction is reliable when
the map's coherence is clear. In two dimensions the lengths match those of the field itself at the
resolution of the data, while a length shorter than the spacing of the data comes out long, the members
being unable to resolve it. In three dimensions the directions hold within a few degrees, whereas the
lengths are rough. A measurement error shortens the lengths, the members' correlation starting below 1,
by about a quarter when the error takes 0.3 of the variance. A correlation that never falls below
:math:`e^{-1}` within half the domain, as on a field with a trend, gives only a lower bound.

In Spatialize
=============

The entropy and mutual information are the estimators of
:doc:`../reference/futures`, where they are experimental, their interface and results liable to
change. The variogram read from the ensemble is :func:`~spatialize.futures.variogram.infer_variogram`,
experimental as well, while :meth:`~spatialize.gs.esi.ESIResult.variogram` reads it from an estimate
already on a grid.

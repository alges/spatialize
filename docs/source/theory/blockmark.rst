.. _theory-blockmark:

#####################
The block-mark model
#####################

The estimator mixes two things, the geometry of the partitions and the values found inside the
cells. The block-mark model is the simplest field in which the two are kept apart, which is why the
theory reads the estimator through it. Through it the theory writes the law the estimator converges
to, the *higher-order predictive law*, and says what a member should be at a location whose cell
holds no datum, which is how Spatialize fills such cells.

The estimator and its limit
===========================

Recall the estimator of :doc:`sde`. From data :math:`O = \{(v_i, z_i)\}_{i \le n}` it draws
:math:`T` partitions :math:`\Pi^{(1)}, \dots, \Pi^{(T)}`, predicts at a location :math:`v` from
the data of its cell in each one, and reports the empirical law of those predictions,

.. math::

   \hat z^{(t)}(v) = g\big(v;\, O_{\Pi^{(t)}}(v)\big),
   \qquad
   \hat F_v^{(T)}(z) = \frac1T \sum_{t=1}^T \mathbf 1\big\{\hat z^{(t)}(v) \le z\big\},

with :math:`g` the decoder and :math:`O_\Pi(v)` the data in the cell of :math:`v`. As :math:`T`
grows, :math:`\hat F_v^{(T)}` converges to the law of :math:`g(v; O_\Pi(v))` over the draws of the
partition, with the data fixed,

.. math::

   F_v(z) = \lim_{T \to \infty} \hat F_v^{(T)}(z)
   = \Pr_\Pi\big(g(v;\, O_\Pi(v)) \le z\big).

Two questions remain open. What is :math:`F_v`, written in terms of the data and the partition
law? What is a member when the cell of :math:`v` holds no datum, so that :math:`O_\Pi(v)` is
empty, leaving the decoder nothing to predict from? The block-mark model answers both.

The model
=========

Draw a random partition :math:`\Pi` of the domain from a partition process, and give each of its
cells, the *blocks*, one value, its *mark*,

.. math::

   m_B \sim \nu \quad\text{for every block } B \text{ of } \Pi,

the marks independent of one another and of the partition. The field takes at each location the
mark of its block,

.. math::

   Z(v) = m_{B(v)},

with :math:`B(v)` the block containing :math:`v`. Two locations in one block take the same value,
while two locations in different blocks take independent values. The geometry, which locations
share a block, lives entirely in the partition law. The values live entirely in the *mark law*
:math:`\nu`.

.. figure:: /_static/theory/block_mark.png
   :width: 100%
   :alt: Block-mark fields on a Mondrian and on a Voronoi partition, and data on one partition

   Panels (a) and (b) show one draw of the block-mark field on a Mondrian and on a Voronoi
   partition, each block painted with its own mark. Panel (c) shows data on one partition. A block
   holding data reveals its mark through them, while the hatched blocks hold none, so their marks
   are unknown.

The model is an idealisation. A real decoder gives more than one value per cell, while a real field varies inside a block. Its value lies in what it isolates. The dependence between locations, of
every order, comes only from their sharing blocks, through the co-occurrence of :doc:`sde`. For
two and three locations the joint cumulant of order :math:`k` is exactly

.. math::

   \kappa(Z(v_1), \dots, Z(v_k)) = \kappa_k\, e(\{v_1, \dots, v_k\}),

with :math:`\kappa_k` the :math:`k`-th cumulant of :math:`\nu`, so the partition law plays the
role a covariance plays for a Gaussian field, at every order.

The model and the decoders
--------------------------

The uniform draw decoder of :doc:`decoders` is the block-mark model read from the data. It returns
a datum of the cell drawn uniformly. When the data of a cell share one mark, every draw returns it,
so the estimate at :math:`v` is the mark of its block whenever that block holds data. The weighted
draws follow the model more loosely, the averaging decoders more loosely still, since they combine
several data. All of them agree on one point, the one this page is about. A decoder needs data, so
none of them says anything about a block that holds none.

The higher-order predictive law
===============================

Take the decoder that reads the model, the uniform draw, and data :math:`z_i = Z(v_i)`, :math:`i = 1, \dots, n`, and a location :math:`v` that is not a
datum. On each draw of the partition exactly one of the following happens. Either :math:`v` shares
its block with some data, and then its value is their common mark, an observed value, or it shares
its block with none, and then its value is a mark never seen, a fresh draw from :math:`\nu`. Order
the data and write :math:`A_i` for the event that :math:`v` shares its block with :math:`v_i` and
with no earlier datum, and :math:`A_0` for the event that it shares its block with no datum. These
events are disjoint and cover every draw. Their probabilities

.. math::

   w_i(v) = \Pr(A_i), \qquad 1 - \sum_{i=1}^n w_i(v) = \Pr(A_0),

give the limit :math:`F_v` of the estimator in closed form, the *higher-order predictive law*,

.. math::

   F_v = \lim_{T \to \infty} \hat F_v^{(T)}
   = \sum_{i=1}^n w_i(v)\, \delta_{z_i} + \Big(1 - \sum_{i=1}^n w_i(v)\Big)\, \nu .

It mixes point masses at the observed values, the draws on which :math:`v` meets data, with the
mark law, the draws on which it meets none. The second term answers the second question above. When the cell of :math:`v` is empty, its value
is a mark drawn from :math:`\nu`. The weights are not the pairwise
co-occurrences :math:`e(\{v, v_i\})`, since the events "shares with :math:`v_1`" and "shares with
:math:`v_2`" overlap. Counting each draw once gives an inclusion–exclusion over the earlier data,

.. math::

   \begin{aligned}
   w_1 &= e(\{v, v_1\}), \qquad
   w_2 = e(\{v, v_2\}) - e(\{v, v_1, v_2\}),
   \\
   w_i &= \sum_{S \subseteq \{v_1, \dots, v_{i-1}\}} (-1)^{|S|}\, e(\{v, v_i\} \cup S),
   \end{aligned}

so the weights use co-occurrences of every order, which a covariance cannot supply. That is the
sense in which the law is of higher order. The ensemble never computes the weights. It realises them
by drawing the partitions, each draw contributing one member to :math:`\hat F_v^{(T)}`.

The *residual weight* :math:`1 - \sum_i w_i(v)` is the probability that the block of :math:`v`
holds no datum. It is a property of the location, near 0 inside the sample, growing as :math:`v`
moves away from the data and as the blocks get smaller. On that share of the partitions the data say nothing about :math:`v`, so the law falls back on :math:`\nu`, the law of a value the sample
does not reach.

The joint law of several locations
==================================

Several locations :math:`u_1, \dots, u_K` without data are read together from the same draws.
On a draw, the locations that share a block with data take that block's mark, an observed value.
The locations that share a block without data take that block's mark too, which is unknown but
**one value for the whole block**, drawn from :math:`\nu`. Distinct blocks without data carry
independent marks. The joint law is the mixture of these configurations over the partition law, and
the covariance between two locations without data is

.. math::

   \operatorname{Cov}\big(Z(u_1), Z(u_2)\big) = \kappa_2\, e(\{u_1, u_2\}),

the probability that they share a block, times the variance of the marks. If each location received
its own draw from :math:`\nu` instead, two locations in one empty block would come out independent, which would distort every quantity that involves several locations at once, such as the area above a threshold, a block average or a simulated field.

How Spatialize fills cells without data
=======================================

The session setting ``empty_cells`` (:mod:`spatialize.session`) chooses what a member is when the
cell of its location holds no datum.

The marks, ``"mark"``
---------------------

This policy realises the residual term of the law. The member of partition :math:`t` at :math:`v`
becomes

.. math::

   \hat z^{(t)}(v) =
   \begin{cases}
   g\big(v;\, O_{\Pi^{(t)}}(v)\big) & \text{if the cell } C \text{ of } v \text{ holds data},\\
   m_{t,C} & \text{if it holds none},
   \end{cases}

so that each cell :math:`C` without data receives one value :math:`m_{t,C}`, its mark, and every
location in it takes that value as its member. Three session settings say where the value
comes from (:mod:`spatialize.session`). A cell holding data, the *source*, is drawn first, then it
gives the value.

1. **The source, ``mark_source``.**

   - ``"local"``, the default, draws uniformly among the ``mark_knn`` cells with data nearest to
     :math:`C`, nearness being measured between the cells' centres (their nuclei, for Voronoi). The
     mark then follows the level of the field around the empty cell, which suits a field whose level
     changes across the domain. It departs from the model, whose marks share one law over the whole
     field.
   - ``"cells"`` draws uniformly among all the cells with data of partition :math:`t`. Each cell
     weighs the same, as each block carries one mark in the model, so a zone sampled densely, which
     holds many data but few cells, does not dominate the marks.
   - ``"data"`` draws no cell but one datum uniformly among all the data, so densely sampled zones
     weigh more.

2. **The value, ``mark_value``.**

   - ``"decoder"``, the default, is the source's decoder prediction at the centre of :math:`C`. The
     mark is then the kind of value the decoder gives elsewhere, an observed value for the drawing
     decoders, a local average for the averaging ones, so the members of empty cells are neither
     rougher nor smoother than the others.
   - ``"datum"`` is one of the source's data, drawn uniformly, whatever the decoder. With
     ``mark_source="cells"`` it estimates the mark law by

     .. math::

        \hat\nu_t = \frac{1}{|\mathcal C_t|} \sum_{C' \in \mathcal C_t} \frac{1}{n_{C'}} \sum_{i \in C'} \delta_{z_i},

     with :math:`\mathcal C_t` the cells of partition :math:`t` holding data and :math:`n_{C'}` the
     number of data in :math:`C'`, which is the block-mark model exactly.

3. **No repetition within a partition.** The empty cells of one partition, taken in a fixed order,
   draw distinct sources while candidates remain, so two empty cells do not copy one value, which
   would couple them as if they formed one block. Empty cells of different partitions draw
   independently.
4. **Reproducible draws.** The random numbers of a mark depend only on the seed of the run, the
   partition and the cell. A mark does not change with the number of threads, nor with the other
   locations requested.
5. **Cross-validation.** In leave-one-out, a datum alone in its cell leaves that cell empty when it
   is held out, so it is predicted by a mark drawn without it, read at its own location. In k-fold,
   a fold that takes every datum of a cell leaves it empty, and the held-out data of that cell share
   one mark drawn from the data outside the fold. The data held out never serve as marks.

The members of cells with data are those of the decoder, unchanged. Under ``"mark"`` every member
is therefore defined. With ``mark_source="cells"`` and ``mark_value="datum"`` the ensemble's law at
:math:`v` estimates the whole of :math:`F_v`, the residual term included. The other combinations
keep the structure of the law, one shared value per empty cell carrying the residual weight, while
drawing that value from the neighbourhood or through the decoder.

Leaving them undefined, ``"nan"``
---------------------------------

The default leaves the member of an empty cell undefined (NaN), :math:`\hat z^{(t)}(v) =` NaN when
the cell of :math:`v` holds no datum, which every reading drops. The law estimated at :math:`v` is
then the one conditioned on the cell having data,

.. math::

   F_v^{\text{nan}} = \frac{\sum_i w_i(v)\, \delta_{z_i}}{\sum_i w_i(v)},

which renormalises the data's weights and ignores the residual term. Near the data the two laws
agree, since the residual weight is small. Far from the data they part. :math:`F_v^{\text{nan}}`
rests on the few partitions that reach some datum and keeps the values of the nearest data, while
:math:`F_v` turns towards the mark law, a spread that grows with the distance to the sample.

Coarser cells, ``"coarsen"``
----------------------------

This policy fills an empty cell with the decoder, from the data of a coarser cell. The member of
partition :math:`t` at :math:`v` becomes

.. math::

   \hat z^{(t)}(v) =
   \begin{cases}
   g\big(v;\, O_{\Pi^{(t)}}(v)\big) & \text{if the cell } C \text{ of } v \text{ holds data},\\
   g\big(v;\, O \cap C^\ast(v)\big) & \text{if it holds none},
   \end{cases}

with :math:`C^\ast(v)` the coarser cell, which depends on the partition process.

- **Mondrian partitions.** :math:`C^\ast` is the nearest ancestor of :math:`C` in the tree of cuts
  whose region holds data, the cell that an earlier stop of the recursion would have left whole.
  It is the same for every location of :math:`C`, so they are all predicted from the same data.
- **Voronoi partitions.** Each location goes to the nearest nucleus whose cell holds data, and
  :math:`C^\ast(v)` is that cell. The locations of one empty cell may go to different cells, which
  amounts to the Voronoi partition of the nuclei with data.

A Mondrian ancestor is not a cell of the partition, so the decoders with parameters fitted per cell
(adaptive IDW, the sharpened decoder, their draws and the decoders written in Python) fit them on
the ancestor's data first. The fit takes its random numbers from the seed of the run, the partition
and the region, so the result does not depend on the number of threads. In leave-one-out, a datum
alone in its cell is predicted from the coarser cell without it. In k-fold, the held-out data of a
cell emptied by their fold are predicted from the coarser cell with the data outside the fold.

The policy departs from the model. It does not draw a fresh value for the empty cell, so the
residual term of :math:`F_v` is not realised. It replaces the partition locally by a coarser one
around the empty cell, the members of that cell continuing the decoder's predictions of the data
nearby. The members then vary with the location inside the empty cell, as the decoder's predictions
do elsewhere, while under ``"mark"`` every location of the cell takes one value. Prefer
``"coarsen"`` when a smooth continuation of the field beyond the data is wanted, and ``"mark"``
when the law far from the data should show how little the data say there.

The share of members concerned
-------------------------------

The result of an estimation reports, at each location, the share of partitions whose cell held no
datum, ``empty_cell_fraction()``, an estimate of the residual weight :math:`1 - \sum_i w_i(v)`. It
does not depend on the policy. Under ``"nan"`` it is also the share of undefined members.

The effect on model selection
-----------------------------

Cross-validation predicts each datum from the others, so a datum alone in its cell faces an empty
cell. Under ``"nan"`` its members are dropped, so the scores are computed on the data that keep
neighbours, the easy ones, and partitions finer than the data can support look better than they
are. Under ``"mark"`` and ``"coarsen"`` those data are predicted, by marks or from a coarser cell, which the scores count.

The following example measures the effect. The field is :math:`z = \sin x + y/5 + \epsilon`, with
:math:`\epsilon` Gaussian noise of standard deviation 0.1, sampled at 200 locations drawn uniformly
in the square :math:`[0, 10]^2`. The estimator is IDW with exponent 2 on Mondrian partitions, with 100 partitions and seed 1. The score is the mean absolute error of leave-one-out, at four
values of ``alpha``. The script ``docs/theory_example_empty_cells.py`` of the repository reproduces
the table.

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - policy
     - ``alpha`` 0.8
     - 0.9
     - 0.95
     - 0.98
   * - ``"nan"``
     - 0.182
     - 0.172
     - 0.158
     - 0.160
   * - ``"mark"``, local source, decoder value (the defaults)
     - 0.184
     - 0.180
     - 0.179
     - 0.184
   * - ``"mark"``, ``"cells"`` source, ``"datum"`` value (the model)
     - 0.185
     - 0.187
     - 0.233
     - 0.457
   * - ``"coarsen"``
     - 0.182
     - 0.174
     - 0.169
     - 0.181

Under ``"nan"`` the finest partitions look nearly as good as any. The local marks predict an
isolated datum from the cells around it, much as a coarser partition would, so they penalise fine
partitions mildly. Coarser cells do the same with the decoder, so they penalise fine partitions
mildly too. The marks of the model come from anywhere in the field, so they penalise fine
partitions strongly, as the residual weight of the theory does.

In Spatialize
=============

The policy is the session setting ``empty_cells`` (``"nan"``, ``"mark"`` or ``"coarsen"``), with
``mark_source``, ``mark_knn`` and ``mark_value`` for the marks (:mod:`spatialize.session`). The residual weight is
:meth:`~spatialize.gs.esi.ESIResult.empty_cell_fraction`. The decoders that read the model inside
the cells, the uniform and weighted draws, are in :doc:`decoders`.

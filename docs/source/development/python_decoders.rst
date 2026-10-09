.. _python-decoders:

#############################
Writing a decoder in Python
#############################

A decoder is the local model that predicts inside one cell of a partition from the data the cell
holds (:doc:`../theory/decoders`). Spatialize offers two ways to add one.

- **In Python**, the subject of this page. The decoder is a few Python functions that Spatialize
  calls on every cell of every partition. Nothing is compiled, and with numba the functions run as
  fast as compiled code on one thread. It suits prototypes, research and decoders built on other
  Python libraries, such as scikit-learn.
- **In C++** (:doc:`cpp_decoders`). The decoder becomes part of the library, runs on every core,
  and can draw random numbers reproducibly. It suits decoders meant to stay.

Both routes give a decoder that works with every partition, in any dimension, in the estimation
functions and in the hyperparameter searches. The script
``examples/scripted_examples/custom_esi.py`` of the repository holds the code of this page.

Step 1. The functions
=====================

A Python decoder is the decoder named ``"custom"``, given by up to four functions. Only the first is
required.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - function
     - role
   * - ``estimation(points, values, queries, params)``
     - The prediction at the ``queries`` of one cell from its data. Returns one value per query.
   * - ``post_creation(points, values)``
     - The parameters of one cell, computed once from its data after the partitions are drawn, and
       handed to the other functions as ``params``. Optional.
   * - ``loo(points, values, params)``
     - Leave-one-out inside one cell, the prediction at each datum from the others. Returns one
       value per datum. Optional (Step 4).
   * - ``kfold(k, points, values, folds, params)``
     - k-fold inside one cell, the prediction at each datum from the data of other folds. Returns
       one value per datum. Optional (Step 4).

The arrays are NumPy arrays of float32.

- ``points`` has shape :math:`(m, d)`, the locations of the :math:`m` data of the cell, and
  ``values`` shape :math:`(m,)`, their values.
- ``queries`` has shape :math:`(q, d)`, the locations to predict in the cell.
- ``params`` is a one-dimensional array, the output of ``post_creation``, or empty without it.
- ``folds`` has shape :math:`(m,)`, the fold of each datum, from 0 to :math:`k - 1`.

The functions return a one-dimensional array of numbers, of any float type. A NaN means that the
function has no prediction for that location. Returning another number of values raises an error
that names the function.

Step 2. A first decoder
=======================

Inverse distance weighting with exponent 2 makes a decoder to start with, since its results can
be compared with the built-in ``"idw"``.

.. code-block:: python

   import numpy as np

   def idw_python(points, values, queries, params):
       p = params[0] if params.size else 2.0
       d = np.sqrt(((queries[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1))
       out = np.empty(len(queries))
       for i in range(len(queries)):
           at = d[i] == 0
           if at.any():                    # a query on a datum takes its value
               out[i] = values[at].mean()
               continue
           w = d[i] ** -p
           out[i] = (w * values).sum() / w.sum()
       return out

It is passed to the estimation functions with ``local_interpolator="custom"``.

.. code-block:: python

   from spatialize.gs.esi import esi_nongriddata

   result = esi_nongriddata(points, values, xi, local_interpolator="custom",
                            estimation=idw_python, n_partitions=100, alpha=0.9)

The result is an ordinary :class:`~spatialize.gs.esi.ESIResult`, with the members, the map and
its readings. Every other argument keeps its meaning, so ``p_process`` chooses the partition and
``n_partitions`` the size of the ensemble.

Step 3. Parameters per cell
===========================

A decoder that adapts to each cell computes its parameters once per cell in ``post_creation``,
instead of at every call of ``estimation``. Spatialize calls it once for each cell of each partition,
stores the result and hands it to the other functions as ``params``.

.. code-block:: python

   def exponent_from_spread(points, values):
       # a larger exponent where the values of the cell vary more
       return np.array([1.0 + min(np.std(values), 3.0)])

   result = esi_nongriddata(points, values, xi, local_interpolator="custom",
                            post_creation=exponent_from_spread, estimation=idw_python)

The parameters are numbers in a one-dimensional array, so they cannot hold a fitted Python object.
A decoder built on a model that must be trained, such as a scikit-learn classifier, trains it
inside ``estimation`` on the data it receives. Categorical estimation does so
(:doc:`../reference/cat_esi`).

Step 4. Cross-validation
========================

The hyperparameter searches and the encoder error of the Pareto search predict each datum from
the others. Without ``loo`` and ``kfold``, Spatialize derives them from ``estimation``.

- **Leave-one-out.** Each datum of a cell is predicted by ``estimation``, called with the other
  data of the cell and the datum's location as the only query, one call per datum.
- **k-fold.** The data of each fold present in the cell are predicted by ``estimation``, called
  with the data of the cell outside that fold, one call per fold.

Both use the parameters ``post_creation`` computed on the whole cell, as the built-in decoders do.
A datum alone in its cell, or a fold that takes all of a cell's data, has no prediction (NaN) unless
the session setting ``empty_cells`` fills it (Step 7).

The derived versions are exact, while their cost grows with the number of data per cell. A decoder
that can predict all the held-out data of a cell at once writes its own ``loo`` or ``kfold``,
returning one value per datum of the cell in their order.

.. code-block:: python

   def idw_loo(points, values, params):
       p = params[0] if params.size else 2.0
       d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1))
       np.fill_diagonal(d, np.inf)            # a datum does not predict itself
       w = d ** -p
       return (w @ values) / w.sum(axis=1)

The searches take the functions once, the same for every configuration they try, while the usual
hyperparameters (``n_partitions``, ``alpha``) vary.

.. code-block:: python

   from spatialize.gs.esi import esi_hparams_search

   search = esi_hparams_search(points, values, xi, local_interpolator="custom",
                               estimation=idw_python, loo=idw_loo, k=-1,
                               alpha=(0.7, 0.8, 0.9), n_partitions=(100,))
   best = search.best_result()

Step 5. Speed with numba
========================

Spatialize calls the functions once for each cell of each partition, which with a few hundred
partitions means tens of thousands of calls. Plain Python spends most of that time in the
interpreter. Compiling the functions with numba's ``@njit`` removes it.

.. code-block:: python

   from numba import njit

   @njit(cache=True)
   def idw_numba(points, values, queries, params):
       p = params[0] if params.size > 0 else 2.0
       out = np.empty(queries.shape[0])
       for i in range(queries.shape[0]):
           sw, swv, exact, n_exact = 0.0, 0.0, 0.0, 0
           for j in range(points.shape[0]):
               d2 = 0.0
               for c in range(points.shape[1]):
                   diff = points[j, c] - queries[i, c]
                   d2 += diff * diff
               if d2 == 0.0:
                   exact += values[j]
                   n_exact += 1
                   continue
               w = d2 ** (-p / 2)
               sw += w
               swv += w * values[j]
           out[i] = exact / n_exact if n_exact > 0 else swv / sw
       return out

A compiled function is passed exactly as a Python one. A few rules make numba effective.

- Work on the arrays with explicit loops and NumPy functions, without Python lists, dictionaries
  or objects, which numba cannot compile efficiently.
- Return a NumPy array, created with ``np.empty`` or ``np.zeros``.
- ``cache=True`` keeps the compiled code between sessions. A first call on small arrays compiles it before the
  estimation starts, so the timing of the estimation excludes compilation.
- ``post_creation`` can be compiled too, as long as it returns an array.

The following times come from ``custom_esi.py`` on the bundled 2D drill-hole data, 400 data and
60 000 locations, with 100 Mondrian partitions and ``alpha`` = 0.9, on a 28-core machine. The three
decoders give the same members up to :math:`4 \cdot 10^{-6}`.

.. list-table::
   :header-rows: 1
   :widths: 50 25

   * - decoder
     - time
   * - built-in ``"idw"``, on every core
     - 0.30 s
   * - built-in ``"idw"``, on one thread
     - 4.30 s
   * - ``"custom"`` compiled with numba
     - 4.32 s
   * - ``"custom"`` in plain Python
     - 24.6 s

With numba the Python decoder runs as fast as the compiled one on one thread. What remains of the
difference is parallelism. Python functions need Python's interpreter lock, so the cells of a Python decoder are computed one at a time, which the session settings
``parallel`` and ``num_threads`` cannot speed up. A decoder that must run on every core is written in C++
(:doc:`cpp_decoders`).

Step 6. Behaviour in the engine
===============================

A Python decoder goes through the same engine as the built-in ones, with a few consequences.

- **Any partition and dimension.** It works with every partition process (``p_process``) and in
  every dimension, since the functions receive the locations as they are.
- **Session settings.** The fixed ``domain`` applies. The settings ``parallel`` and ``num_threads``
  have no effect, as said above. Ctrl-C stops the computation at the end of the cell in progress.
- **Errors.** An exception raised by a function stops the estimation and reaches the caller as a
  ``SpatializeError`` carrying its message.

Step 7. Cells without data
==========================

Spatialize never calls the functions on a cell without data. Such cells follow the session setting
``empty_cells`` (:doc:`../theory/blockmark`).

- Under ``"nan"``, the default, their members are NaN.
- Under ``"mark"``, each empty cell takes the prediction of ``estimation`` on a source cell with
  data, at the empty cell's centre, or one of that cell's data with ``mark_value="datum"``.
- Under ``"coarsen"``, ``estimation`` predicts at the locations of the empty cell from the data of
  a coarser cell. For a Mondrian partition the coarser cell is an ancestor in the tree of cuts, for
  which Spatialize calls ``post_creation`` on the ancestor's data first.

Step 8. Randomness
==================

The functions receive no seed and no identifier of the partition or the cell. A decoder that draws
random numbers, such as a decoder that samples a datum, gives results that change from one run to
the next, and with them the members and the searches. Decoders written in Python are best kept
deterministic. Drawing decoders belong in C++, where each cell receives the run's seed, the
partition and the cell, from which the built-in draws derive reproducible random numbers
(:doc:`cpp_decoders`).

Step 9. Checking a decoder
==========================

Two checks catch most mistakes.

- **Against a known decoder.** A Python version of a built-in decoder should reproduce its members
  up to float rounding, as the IDW of this page does. The same comparison through
  ``esi_hparams_search`` checks the cross-validation.
- **On cases with known answers.** A decoder that returns the mean of the cell should give, under
  every partition, members equal to the means of the cells that
  :func:`~spatialize.gs.partitions.cell_labels` returns.

The conformance tests (:doc:`../scenarios/index`) name their estimators by catalogue name and
parameters in a file, so they apply to decoders of the catalogue, the built-in ones and those added
in C++.

When to move to C++
===================

A Python decoder becomes a candidate for C++ when it is meant to be part of the library, when it
must run on every core, or when it draws random numbers. The C++ route starts from the same
functions, ``estimation``, ``loo`` and ``kfold`` becoming the methods of a class
(:doc:`cpp_decoders`).

.. _cpp-decoders:

##########################
Writing a decoder in C++
##########################

A decoder written in C++ becomes part of the library. It runs on every core, it can draw random
numbers reproducibly, and the catalogue makes it known to ``run``, the Python functions, the
searches and the conformance tests at once. Writing a decoder in Python (:doc:`python_decoders`)
needs no compilation and suits prototypes. This page follows one example from the header to the
tests, a decoder that predicts the median of the cell's data. Its code was compiled and run while
this page was written.

Step 1. The interface
=====================

Every decoder derives from ``Decoder`` (``include/spatialize/decoder.hpp``). The engine hands it the
data of one cell at a time, through indices into the arrays of all the data.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - method
     - role
   * - ``leaf_estimation``
     - The prediction at the locations of one cell (``locations_id``, indices into ``locations``)
       from its data (``samples_id``, indices into ``coords`` and ``values``). One value per
       location.
   * - ``leaf_loo``
     - The prediction at each datum of the cell from the other data of the cell, one value per
       datum of ``samples_id``, in its order.
   * - ``leaf_kfold``
     - The prediction at each datum from the data of the cell in other folds, ``folds`` giving the
       fold of every datum.
   * - ``fit`` (optional)
     - Per-cell parameters, computed once for every cell of every partition after the forest is
       drawn, stored in ``partition->leaf_params`` and handed to the methods above as ``params``.
   * - ``fit_cell`` (optional)
     - The parameters of one region from given data, which ``empty_cells="coarsen"`` asks for a
       Mondrian ancestor. A decoder that overrides ``fit`` overrides it too.
   * - ``thread_safe`` (optional)
     - Whether cells may be decoded on several threads at once. ``true`` by default.

Every method also receives a ``CellContext`` with the partition (``tree``), the cell (``cell``) and
the run's ``seed``. A decoder that draws random numbers derives them from these and from the index
of the location, with the counter-based generator of ``decoders/draw.hpp``. Its draws then depend
neither on the number of threads nor on the other locations requested.

Step 2. The header
==================

A decoder lives in its own header under ``include/spatialize/decoders/``. The median decoder fits
in one page.

.. code-block:: cpp

   // include/spatialize/decoders/median.hpp
   #ifndef _SPTLZ_DECODERS_MEDIAN_
   #define _SPTLZ_DECODERS_MEDIAN_

   #include <algorithm>
   #include <cmath>
   #include <vector>
   #include "spatialize/decoder.hpp"

   namespace sptlz{
     // The median of the cell's data, the same at every location of the cell.
     class MedianDecoder: public Decoder {
       protected:
         static float median_of(std::vector<float> v){
           if(v.empty()) return(NAN);
           size_t h = v.size() / 2;
           std::nth_element(v.begin(), v.begin() + h, v.end());
           float upper = v[h];
           if(v.size() % 2 == 1) return(upper);
           float lower = *std::max_element(v.begin(), v.begin() + h);
           return(0.5f * (lower + upper));
         }

       public:
         // every location of the cell gets the median of the cell's data
         std::vector<float> leaf_estimation(std::vector<std::vector<float>> *coords, std::vector<float> *values,
                                            std::vector<int> *samples_id, std::vector<std::vector<float>> *locations,
                                            std::vector<int> *locations_id, std::vector<float> *params,
                                            const CellContext &cell){
           std::vector<float> v;
           for(int i: *samples_id) v.push_back(values->at(i));
           return(std::vector<float>(locations_id->size(), median_of(v)));
         }

         // each datum gets the median of the other data of its cell
         std::vector<float> leaf_loo(std::vector<std::vector<float>> *coords, std::vector<float> *values,
                                     std::vector<int> *samples_id, std::vector<float> *params, const CellContext &cell){
           std::vector<float> result;
           for(size_t i=0; i<samples_id->size(); i++){
             std::vector<float> v;
             for(size_t j=0; j<samples_id->size(); j++) if(j != i) v.push_back(values->at(samples_id->at(j)));
             result.push_back(median_of(v));
           }
           return(result);
         }

         // each datum gets the median of the data of its cell outside its fold
         std::vector<float> leaf_kfold(int k, std::vector<std::vector<float>> *coords, std::vector<float> *values,
                                       std::vector<int> *folds, std::vector<int> *samples_id, std::vector<float> *params,
                                       const CellContext &cell){
           std::vector<float> result;
           for(int i: *samples_id){
             std::vector<float> v;
             for(int j: *samples_id) if(folds->at(j) != folds->at(i)) v.push_back(values->at(j));
             result.push_back(median_of(v));
           }
           return(result);
         }
     };
   }

   #endif

Three rules keep a decoder correct inside the engine.

- The ``leaf_*`` methods run on several cells at once, on different threads, so they read the data
  and their own members but never modify shared state. A decoder that cannot meet this rule
  overrides ``thread_safe()`` to return ``false``, and its cells are then decoded one at a time.
- A method returns exactly one value per location (``leaf_estimation``) or per datum (``leaf_loo``,
  ``leaf_kfold``), NaN where it has no prediction.
- Randomness comes only from the ``CellContext``, as in Step 1, or from the generator passed to
  ``fit``.

Step 3. The catalogue
=====================

The catalogue (``src/c++/registry.hpp``) is the single place where a decoder is registered. Three
additions make the median decoder known everywhere.

.. code-block:: cpp

   // 1. the header, with the other decoders
   #include "spatialize/decoders/median.hpp"

   // 2. its entry, appended at the end of the list in decoders()
   specs.push_back({"cellmedian", "The median of the cell's data.",
      1, ANY, true,      // dimensions from 1 to any; thread-safe
      {},                // parameters (none)
      nullptr});

   // 3. its factory, with the others, at the entry's position in the list
   specs[11].make = [](int d, const py::dict &p, Method m)->sptlz::Decoder*{
     return(new sptlz::MedianDecoder());
   };

The entry gives the name, a one-line description, the lowest and highest dimension (``ANY`` for no
limit), whether the decoder is thread-safe, and its parameters. A parameter is described by its
name, its type (``"float"``, ``"int"``, ``"bool"``, ``"str"``, ``"choice"`` or ``"callable"``), a
description, whether it is required, its default when it is not, and the accepted names of a
``"choice"``. The IDW entry, for instance, declares ``{"exponent", "float", "the power p of the
distance", true, py::none(), {}}``, and its factory reads the value with
``param<float>(decoders()[0], p, "exponent")``. The catalogue checks the parameters a call gives
against the entry before the factory runs, so a factory receives valid values.

Step 4. Build
=============

.. code-block:: bash

   make            # or: python setup.py build_ext --inplace

The new decoder appears in ``libspatialize.catalog()["decoders"]``, and ``spatialize.gs.supports``
knows its dimensions.

Step 5. The Python functions
============================

The public functions take a decoder's parameters as keyword arguments, with defaults declared per
decoder. One line in ``more_decoders`` (``src/python/spatialize/gs/__init__.py``) gives the defaults
of the new decoder, here none.

.. code-block:: python

   "cellmedian": {},

A decoder with parameters lists each with its default wrapped in ``one(...)``, as the entries next to
it do, for instance ``"mydecoder": dict(power=one(2.0))``. The estimation functions then receive the
value, the hyperparameter searches a one-element list of candidates. Everything else follows from
the catalogue. The arguments reach ``run``
in the catalogue's order, the searches vary the parameters that are not functions, and the
conformance runner checks the dimensions. Adding the name to the class ``local_interpolator`` of the
same module is optional, a convenience for users.

The decoder can then be used like any other.

.. code-block:: python

   from spatialize.gs.esi import esi_nongriddata, esi_hparams_search

   result = esi_nongriddata(points, values, xi, local_interpolator="cellmedian",
                            n_partitions=100, alpha=0.8)
   search = esi_hparams_search(points, values, xi, local_interpolator="cellmedian",
                               griddata=False, k=5, alpha=(0.7, 0.8), n_partitions=(100,))

Step 6. Per-cell parameters
===========================

A decoder whose parameters depend on each cell's data, such as adaptive IDW, overrides ``fit``. The
engine calls it once after drawing the forest, and it loops over the partitions and their cells,
storing each cell's parameters in ``partition->leaf_params[cell]``. Its randomness, if any, comes
from the generator ``fit`` receives. The loop may run in parallel over the partitions, provided
that only the calling thread talks to Python (progress and Ctrl-C), as ``AdaptiveIDWDecoder::fit``
does.

Such a decoder also overrides ``fit_cell``, which fits one region from a given list of data with a
given seed. The policy ``empty_cells="coarsen"`` uses it for a Mondrian ancestor, a region that is
not a cell of the partition and so has no stored parameters.

Step 7. Tests
=============

Three layers check a new decoder (:doc:`testing`).

- **Properties that must hold exactly.** The unit tests of ``tests/unit`` gather them,
  for instance that the median decoder gives, in every partition, the median of the data of each
  cell, read with :func:`~spatialize.gs.partitions.cell_labels`.
- **Bitwise snapshots.** Adding the decoder to the list of cases in
  ``tests/guard/cases.py`` and running
  ``python tests/guard/make_snapshots.py --only CASE`` pins its outputs, so a later
  refactor that changes them is caught.
- **Conformance scenarios.** A scenario names its estimators by catalogue name and parameters in
  its ``scenario.yaml``, for instance ``{id: med, encoder: mondrian, rate: 5.0, decoder: cellmedian,
  params: {}}``, which subjects the decoder to the scenario's statistical checks.

Step 8. Documentation
=====================

A decoder meant for users is documented in four places: the list of ``local_interpolator`` values
in the docstrings of :func:`~spatialize.gs.esi.esi_griddata` and
:func:`~spatialize.gs.esi.esi_nongriddata`, the theory page on decoders (:doc:`../theory/decoders`),
with its intuition and when to prefer it, the table of the catalogue in :doc:`architecture`, and the
release notes (:doc:`../changes`).

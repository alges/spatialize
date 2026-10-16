.. _datasets:

*********
Data sets
*********

Spatialize ships the data sets its tutorials use, each loaded by one function of
:mod:`spatialize.data` with no download and no extra dependency. Several are exhaustively known
fields, a whole grid returned as the truth with samples drawn from it, so an estimate can be judged
against the truth everywhere, a check real data rarely allow.

.. list-table::
   :header-rows: 1
   :widths: 28 22 18 32

   * - loader
     - variable
     - truth
     - source and licence
   * - :func:`~spatialize.data.load_elevation_walker_lake`
     - elevation, continuous, 2D
     - 260 × 300 grid
     - USGS 3D Elevation Program, public domain
   * - :func:`~spatialize.data.load_reflectance_sundarbans`
     - near-infrared reflectance, continuous, 2D
     - 400 × 580 grid
     - Landsat 9, USGS, public domain
   * - :func:`~spatialize.data.load_simulated_anisotropic_data`
     - Gaussian, anisotropic, 2D
     - 100 realisations on a 200 × 200 grid
     - simulated for Spatialize
   * - :func:`~spatialize.data.load_lithology_golden`
     - lithology and era, categorical (nominal, ordinal), 2D
     - 120 × 120 grid
     - USGS State Geologic Map Compilation, public domain, via Macrostrat, CC-BY 4.0
   * - :func:`~spatialize.data.load_facies_pyrcz`
     - two facies with porosity, permeability and impedance, 2D
     - none, wells only
     - GeoDataSets by M. J. Pyrcz, MIT licence (shipped)
   * - :func:`~spatialize.data.load_precipitation_maipo`
     - daily rain with an atom at zero, 2D or 2.5D with elevation
     - none, gauges only
     - Maipo basin rain gauges, three days of July 2018
   * - :func:`~spatialize.data.load_drill_holes_andes_2D`
     - copper and gold grades, rock type, 2D
     - none, drill holes only
     - an Andean copper deposit
   * - :func:`~spatialize.data.load_drill_holes_andes_3D`
     - copper and gold grades, rock type, 3D
     - none, drill holes only
     - an Andean copper deposit

Exhaustive fields
=================

.. currentmodule:: spatialize.data

.. autofunction:: load_elevation_walker_lake

.. autofunction:: load_reflectance_sundarbans

.. autofunction:: load_simulated_anisotropic_data

.. autofunction:: load_lithology_golden

Sampled data
============

.. autofunction:: load_facies_pyrcz

.. autofunction:: load_precipitation_maipo

.. autofunction:: load_drill_holes_andes_2D

.. autofunction:: load_drill_holes_andes_3D

Bundled resources
=================

The files behind the loaders live in the package ``spatialize.resources``, under ``data``, one
folder or file per data set, with the licence of each third-party set beside it and the scripts that
rebuilt the public ones in the repository's ``tools/datasets``. The same package holds the Scientific
Colour Maps of Fabio Crameri (``resources/ScientificColourMaps``), perceptually uniform colour maps
readable by colour-blind viewers, which :mod:`spatialize.viz` uses, with their licence and their
documentation shipped beside them.

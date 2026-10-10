# Lithology of the Front Range foothills near Golden, Colorado

A 120 × 120 grid (21.3 km × 27.5 km, x and y in km from the south-west corner at
105.35° W, 39.60° N) of the geologic map units of the window 105.35°–105.10° W, 39.60°–39.85° N,
grouped into six lithologies (gneiss, granite, sandstone, shale, basalt, alluvium) and four eras
(Proterozoic, Paleozoic, Mesozoic, Cenozoic). Water bodies are left without a lithology.

- `grid.csv`: one row per grid cell (`x`, `y`, `unit`, `lithology`, `era`, `age_ma`, the mid-age of the
  unit's interval).
- `units.csv`: the 53 map units (name, generalised lithology, era, ages, lithology description, original
  map source).

**Source.** Horton, J.D., San Juan, C.A., and Stoeser, D.B., 2017, The State Geologic Map Compilation
(SGMC) geodatabase of the conterminous United States: U.S. Geological Survey Data Series 1052,
https://doi.org/10.3133/ds1052 — for this window, from Green, G.N., 1992, The Digital Geologic Map of
Colorado. Retrieved through the Macrostrat API (https://macrostrat.org, source 133, CC-BY 4.0 for the
Macrostrat compilation).

**Licence.** The SGMC is a work of the U.S. Geological Survey, in the public domain in the United States.
The Macrostrat compilation is under CC-BY 4.0: cite Macrostrat (Peters, S.E., Husson, J.M., and
Czaplewski, J., 2018, Macrostrat: a platform for geological data integration and deep-time Earth
crust research, Geochemistry, Geophysics, Geosystems 19, 1393–1409) and the SGMC above.

**Reproduce.** `tools/datasets/fetch_sgmc_window.py` then `tools/datasets/make_sgmc_lithology.py` in the
Spatialize repository (2026-10-10).

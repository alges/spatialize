# Elevation of the Walker Lake area, Nevada

A 260 × 300 grid of 100 m cells (26 km × 30 km, x and y in km from the south-west corner at
339 000 m E, 4 270 000 m N, UTM zone 11N, WGS 84) of the ground elevation around Walker Lake, Nevada, with
the lake's flat surface (about 1 194 m), the Wassuk Range to the west (up to 3 418 m) and the hills to
the east.

- `grid.csv`: one row per cell centre (`x`, `y`, `elevation` in m), row-major with rows along y.

The window has the size of the Walker Lake data set of Isaaks and Srivastava (1989), which was derived
from a digital elevation model of the same area. This file holds the elevation itself, not their
transformed variables U and V.

**Source.** U.S. Geological Survey, 3D Elevation Program (3DEP), 1/3 arc-second digital elevation
model, through the 3DEP elevation image service
(https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer), bilinear resampling.

**Licence.** A work of the U.S. Geological Survey, in the public domain in the United States. Cite it
as "U.S. Geological Survey, 3D Elevation Program".

**Reproduce.** `tools/datasets/fetch_walker_lake_elevation.py` in the Spatialize repository
(2026-10-11).

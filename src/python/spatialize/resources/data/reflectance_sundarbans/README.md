# Near-infrared reflectance of the Sundarbans, Bangladesh

A 400 × 580 grid of 60 m cells (24 km × 34.8 km, x and y in km from the south-west corner at
699 600 m E, 2 418 000 m N, UTM zone 45N, WGS 84) of the near-infrared surface reflectance of the
Sundarbans mangrove forest. Water absorbs the near infrared while the mangroves reflect it, so the tidal
channels draw a field with sharp, winding boundaries at every scale.

- `grid.csv`: one row per cell centre (`x`, `y`, `reflectance`, a fraction), row-major with rows
  along y.

**Source.** Landsat 9 OLI-2, Collection 2 Level-2 surface reflectance, band 5 (near infrared), scene
LC09_L2SP_138045_20221114_02_T1 (14 November 2022, cloud cover 0.01 %), U.S. Geological Survey, read
through the Microsoft Planetary Computer data API, bilinear resampling. The Level-2 scale is applied
(digital number × 2.75e-5 − 0.2).

**Licence.** Landsat data are in the public domain (U.S. Geological Survey). Cite them as "Landsat 9
image courtesy of the U.S. Geological Survey".

**Reproduce.** `tools/datasets/fetch_sundarbans_landsat.py` in the Spatialize repository (2026-10-11).

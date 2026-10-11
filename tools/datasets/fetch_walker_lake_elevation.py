"""Build the bundled data set of `spatialize.data.load_elevation_walker_lake`: the elevation of the
Walker Lake area, Nevada, from the U.S. Geological Survey's 3D Elevation Program (public domain).

    python fetch_walker_lake_elevation.py src/python/spatialize/resources/data/elevation_walker_lake

Needs numpy and Pillow. Asks the 3DEP image service for a 260 × 300 grid of 100 m cells in UTM zone 11N
(WGS 84), bilinear resampling, and writes `grid.csv` (x, y in km from the south-west corner, elevation
in m). The window has the size of Isaaks and Srivastava's Walker Lake data set; the values are the
elevation itself, not their transformed variables.
"""
import csv
import io
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

X0, Y0 = 339000.0, 4270000.0                  # south-west corner, UTM 11N (EPSG:32611)
NX, NY, CELL = 260, 300, 100.0
URL = ("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
       "?bbox={x0},{y0},{x1},{y1}&bboxSR=32611&imageSR=32611&size={nx},{ny}&format=tiff&pixelType=F32"
       "&interpolation=RSP_BilinearInterpolation&f=image")


def main(out):
    url = URL.format(x0=X0, y0=Y0, x1=X0 + NX * CELL, y1=Y0 + NY * CELL, nx=NX, ny=NY)
    with urllib.request.urlopen(url, timeout=120) as r:
        z = np.array(Image.open(io.BytesIO(r.read())), dtype=float)
    assert z.shape == (NY, NX), z.shape
    z = z[::-1]                               # image rows run north to south; grid rows south to north
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "grid.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "elevation"])
        for j in range(NY):
            for i in range(NX):
                w.writerow([f"{(i + 0.5) * CELL / 1000:.2f}", f"{(j + 0.5) * CELL / 1000:.2f}", f"{z[j, i]:.1f}"])
    print(f"{NX}×{NY} cells, elevation {z.min():.1f}–{z.max():.1f} m", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])

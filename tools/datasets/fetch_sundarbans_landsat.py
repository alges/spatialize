"""Build the bundled data set of `spatialize.data.load_reflectance_sundarbans`: the near-infrared
reflectance of a window of the Sundarbans mangrove forest, Bangladesh, from a Landsat 9 scene (U.S.
Geological Survey, public domain).

    python fetch_sundarbans_landsat.py src/python/spatialize/resources/data/reflectance_sundarbans

Needs numpy. Reads band 5 (near infrared, `nir08`) of the Collection 2 Level-2 surface reflectance
scene LC09_L2SP_138045_20221114_02_T1 (cloud cover 0.01 %) through the Microsoft Planetary Computer
data API, on a 400 × 580 grid of 60 m cells in UTM zone 45N (WGS 84), bilinear resampling, and writes
`grid.csv` (x, y in km from the south-west corner, reflectance). The scale of Collection 2 Level-2
surface reflectance is applied (× 2.75e-5 − 0.2). Water is dark and the mangroves bright, so the
channels draw a field with sharp, winding boundaries.
"""
import csv
import io
import os
import sys
import urllib.request

import numpy as np

SCENE = "LC09_L2SP_138045_20221114_02_T1"
X0, Y0 = 699600.0, 2418000.0                  # south-west corner, UTM 45N (EPSG:32645)
NX, NY, CELL = 400, 580, 60.0
URL = ("https://planetarycomputer.microsoft.com/api/data/v1/item/bbox/{x0},{y0},{x1},{y1}/{nx}x{ny}.npy"
       "?collection=landsat-c2-l2&item={scene}&assets=nir08&resampling=bilinear"
       "&coord_crs=EPSG:32645&dst_crs=EPSG:32645")


def main(out):
    url = URL.format(x0=X0, y0=Y0, x1=X0 + NX * CELL, y1=Y0 + NY * CELL, nx=NX, ny=NY, scene=SCENE)
    with urllib.request.urlopen(url, timeout=300) as r:
        a = np.load(io.BytesIO(r.read()), allow_pickle=False)
    band, mask = a[0].astype(float), a[-1]
    assert band.shape == (NY, NX) and (mask == 255).all(), (band.shape, np.unique(mask))
    refl = (band * 2.75e-5 - 0.2)[::-1]       # image rows run north to south; grid rows south to north
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "grid.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "reflectance"])
        for j in range(NY):
            for i in range(NX):
                w.writerow([f"{(i + 0.5) * CELL / 1000:.2f}", f"{(j + 0.5) * CELL / 1000:.2f}", f"{refl[j, i]:.4f}"])
    print(f"{NX}×{NY} cells, reflectance {refl.min():.4f}–{refl.max():.4f}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])

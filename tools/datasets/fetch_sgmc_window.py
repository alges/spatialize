"""Fetch the geologic map units of a window of the USGS State Geologic Map Compilation through the
Macrostrat API (one point query per unit not yet covered), and rasterise them on a grid.

Run outside the library (needs shapely):  python fetch_sgmc_window.py OUT.json
The output keeps, per grid cell, the map unit's id, name, generalised lithology string and age, so
that the lithology classes can be chosen afterwards (see make_sgmc_lithology.py).
"""
import json
import sys
import time
import urllib.request

from shapely.geometry import shape, Point

LON0, LON1, LAT0, LAT1 = -105.35, -105.10, 39.60, 39.85   # Front Range foothills near Golden, Colorado
NX = NY = 120
URL = "https://macrostrat.org/api/v2/geologic_units/map?lat={lat}&lng={lon}&scale=medium&format=geojson"


def unit_at(lon, lat):
    with urllib.request.urlopen(URL.format(lat=lat, lon=lon), timeout=60) as r:
        data = json.load(r)["success"]["data"]
    feats = data["features"] if isinstance(data, dict) else data
    return feats[0] if feats else None


def main(out):
    units, polys, grid = {}, [], []
    for j in range(NY):
        lat = LAT0 + (j + 0.5) * (LAT1 - LAT0) / NY
        row = []
        for i in range(NX):
            lon = LON0 + (i + 0.5) * (LON1 - LON0) / NX
            p = Point(lon, lat)
            hit = next((mid for mid, poly in polys if poly.contains(p)), None)
            if hit is None:
                f = unit_at(lon, lat)
                time.sleep(0.2)
                if f is None:
                    row.append(None)
                    continue
                pr = f["properties"]
                hit = pr["map_id"]
                if hit not in units:
                    units[hit] = {k: pr.get(k) for k in ("map_id", "source_id", "name", "lith", "liths",
                                                         "best_int_name", "t_age", "b_age", "comments")}
                    polys.append((hit, shape(f["geometry"])))
            row.append(hit)
        grid.append(row)
        print(f"row {j + 1}/{NY}: {len(units)} units", file=sys.stderr, flush=True)
    json.dump({"window": [LON0, LON1, LAT0, LAT1], "nx": NX, "ny": NY, "units": units, "grid": grid},
              open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1])

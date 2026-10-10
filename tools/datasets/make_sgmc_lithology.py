"""Build the bundled lithology data set of `spatialize.data.load_lithology_golden` from the window
fetched by fetch_sgmc_window.py (USGS State Geologic Map Compilation, through Macrostrat).

    python make_sgmc_lithology.py sgmc_window.json src/python/spatialize/resources/data/lithology_golden

Writes `grid.csv` (one row per cell of the 120×120 grid: x, y in km, the map unit, its generalised
lithology, era and age) and `units.csv` (the map units). The lithology classes group the units by
rock type; the era is ordinal (Proterozoic < Paleozoic < Mesozoic < Cenozoic). Water is no datum.
"""
import csv
import json
import math
import os
import sys

# map unit name (prefix) → generalised lithology
LITHOLOGY = [
    ("Felsic and hornblendic gneisses", "gneiss"),
    ("Biotitic gneiss, schist, and migmatite", "gneiss"),
    ("Granitic rocks of 1700", "granite"),
    ("Granitic rocks of 1400", "granite"),
    ("Laramide intrusive rocks", "granite"),
    ("Fountain Fm", "sandstone"),
    ("Lykins", "sandstone"),
    ("Dakota Group", "sandstone"),
    ("Laramie Fm and Fox Hills", "sandstone"),
    ("Denver and Arapahoe", "sandstone"),
    ("Upper part of Dawson Arkose", "sandstone"),
    ("Pierre Shale", "shale"),
    ("Colorado Group", "shale"),
    ("Basaltic flows", "basalt"),
    ("Modern alluvium", "alluvium"),
    ("Gravels and alluviums", "alluvium"),
    ("Older gravels and alluviums", "alluvium"),
    ("Bouldery gravel", "alluvium"),
]


def lithology(name):
    for prefix, cls in LITHOLOGY:
        if name.startswith(prefix):
            return cls
    return None   # water


def era(b_age):
    """The era of the unit's base age (Ma)."""
    if b_age is None:
        return None
    a = float(b_age)
    return "Proterozoic" if a > 541 else "Paleozoic" if a > 252 else "Mesozoic" if a > 66 else "Cenozoic"


def main(src, out):
    d = json.load(open(src))
    lon0, lon1, lat0, lat1 = d["window"]
    nx, ny = d["nx"], d["ny"]
    kx = 111.32 * math.cos(math.radians((lat0 + lat1) / 2))   # km per degree of longitude
    ky = 110.57                                               # km per degree of latitude
    os.makedirs(out, exist_ok=True)
    units = d["units"]
    with open(os.path.join(out, "grid.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "unit", "lithology", "era", "age_ma"])
        for j, row in enumerate(d["grid"]):
            for i, mid in enumerate(row):
                u = units[str(mid)]
                cls = lithology(u["name"])
                x = (i + 0.5) * (lon1 - lon0) / nx * kx
                y = (j + 0.5) * (lat1 - lat0) / ny * ky
                age = 0.5 * (float(u["t_age"]) + float(u["b_age"])) if u["t_age"] is not None else ""
                w.writerow([f"{x:.4f}", f"{y:.4f}", mid, cls or "", era(u["b_age"]) if cls else "",
                            f"{age:.1f}" if cls else ""])
    with open(os.path.join(out, "units.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "name", "lithology", "era", "interval", "top_age_ma", "base_age_ma", "lith", "source"])
        for mid, u in sorted(units.items(), key=lambda t: int(t[0])):
            cls = lithology(u["name"])
            w.writerow([mid, u["name"], cls or "", era(u["b_age"]) if cls else "", u["best_int_name"] or "",
                        u["t_age"] or "", u["b_age"] or "", u["lith"] or "", (u["comments"] or "").replace("\n", " ")])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

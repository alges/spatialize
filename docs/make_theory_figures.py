"""Figures of the theory pages, generated beforehand into docs/source/_static/theory/.

Run from the repository root with the extension built:

    PYTHONPATH=.:src/python python docs/make_theory_figures.py

The figures are deterministic (fixed seeds), so rerunning the script only changes them when the code
they show changes.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spatialize.gs.partitions import cell_labels

OUT = os.path.join(os.path.dirname(__file__), "source", "_static", "theory")
CMAP = "viridis"


def _grid(m):
    g = (np.arange(m) + 0.5) / m
    gx, gy = np.meshgrid(g, g)
    return np.c_[gx.ravel(), gy.ravel()].astype(np.float32)


def _one_partition(points, grid, p_process, alpha, seed, data_cond=True):
    """Labels of the grid and of the points in one partition (the box is the unit square)."""
    corners = np.array([[0, 0], [1, 1]], np.float32)
    box_points = np.vstack([points, corners]) if len(points) else corners
    labels = cell_labels(box_points, np.vstack([grid, points]) if len(points) else grid, p_process=p_process,
                         alpha=alpha, n_partitions=1, seed=seed, data_cond=data_cond)[:, 0]
    return labels[:len(grid)], labels[len(grid):]


def _paint(labels, rng):
    """One mark per cell, independent across cells: the block-mark field."""
    cells = np.unique(labels)
    marks = dict(zip(cells, rng.normal(size=len(cells))))
    return np.array([marks[c] for c in labels])


def block_mark(path):
    m = 300
    grid = _grid(m)
    rng = np.random.default_rng(20261007)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)

    # (a), (b): one draw of the block-mark field on each partition process
    # the uniform Voronoi nuclei are as many as the data scaled by alpha, so 60 reference points (which
    # take no other part) set their number
    reference = rng.uniform(0, 1, (60, 2)).astype(np.float32)
    for ax, (title, proc, alpha) in zip(axes[:2], [("(a) Mondrian partition", "mondrian-raw", 0.86),
                                                   ("(b) Voronoi partition, uniform nuclei", "voronoi", 0.6)]):
        labels, _ = _one_partition(reference, grid, proc, alpha, seed=7, data_cond=False)
        ax.imshow(_paint(labels, rng).reshape(m, m), origin="lower", extent=(0, 1, 0, 1), cmap=CMAP,
                  vmin=-2.5, vmax=2.5, interpolation="nearest")
        ax.set_title(title, fontsize=11)

    # (c): the data on one partition; cells with data show their value, empty cells get one mark each
    data = rng.uniform(0.05, 0.95, (25, 2)).astype(np.float32)
    values = rng.normal(size=len(data))
    labels, at_data = _one_partition(data, grid, "mondrian-raw", 0.9, seed=11)
    field = np.full(len(grid), np.nan)
    for c in np.unique(labels):
        inside = labels == c
        holders = np.where(at_data == c)[0]
        if len(holders):
            field[inside] = values[holders[0]]
    empty = np.isnan(field).reshape(m, m)
    ax = axes[2]
    ax.imshow(np.ma.masked_invalid(field.reshape(m, m)), origin="lower", extent=(0, 1, 0, 1), cmap=CMAP,
              vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax.contourf(np.linspace(0, 1, m), np.linspace(0, 1, m), empty.astype(float), levels=[0.5, 1.5],
                colors="none", hatches=["///"])
    ax.scatter(data[:, 0], data[:, 1], s=22, c="white", edgecolors="black", linewidths=0.8)
    ax.set_title("(c) Data on one partition; hatched: no datum", fontsize=11)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def home(path):
    """The method in one strip, on the anisotropic field of the conformance tests: the data, a few
    members of the ensemble, the median map and the probability of exceeding a threshold."""
    from matplotlib.patches import FancyArrowPatch
    from spatialize.gs.esi import esi_nongriddata
    from spatialize.resources import ScientificColourMaps as SCM
    from spatialize.scenarios import catalog, generators

    sc = catalog(ids=["S03-anisotropic-field"])["S03-anisotropic-field"]
    field = sc.field(13)
    m = 120
    grid = generators.grid(m)
    result = esi_nongriddata(np.asarray(field["samples"]), np.asarray(field["values"]), grid,
                             local_interpolator="adaptiveidw", p_process="mondrian", alpha=0.8,
                             n_partitions=300, seed=4, callback=lambda *a, **k: None)
    members = result.esi_samples(raw=True)
    median = np.nanmedian(members, axis=1).reshape(m, m)
    exceed = np.nanmean(members > 1.0, axis=1).reshape(m, m)
    truth = np.asarray(field["truth"]).reshape(sc.spec["data"]["grid"], -1)
    cmap, vmin, vmax = SCM.batlow, -2.2, 2.2
    grey = "#8a8f98"

    W, H = 15.0, 3.9
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_alpha(0)
    side, gap, bottom = 0.165, 0.085, 0.05          # square panels: side in width units
    tall = side * W / H                               # the same side in height units
    left = (1 - 4 * side - 3 * gap) / 2

    def square(x, y, image, cmap_=cmap, lo=vmin, hi=vmax, smooth=True, scale=1.0):
        ax = fig.add_axes([x, y, side * scale, tall * scale])
        ax.imshow(image, origin="lower", extent=(0, 1, 0, 1), cmap=cmap_, vmin=lo, vmax=hi, aspect="auto",
                  interpolation="bilinear" if smooth else "nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for edge in ax.spines.values():
            edge.set_edgecolor("#c9ced6"); edge.set_linewidth(1.6)
        return ax

    def title(k, text):
        fig.text(left + k * (side + gap) + side / 2, bottom + tall + 0.05, text, ha="center", va="bottom",
                 color=grey, fontsize=17)

    # (1) the data, over a faint view of the field they come from
    ax = square(left, bottom, truth)
    ax.images[0].set_alpha(0.5)
    pts, vals = np.asarray(field["samples"]), np.asarray(field["values"])
    ax.scatter(pts[:, 0], pts[:, 1], c=vals, cmap=cmap, vmin=vmin, vmax=vmax, s=10, edgecolors="white",
               linewidths=0.35)
    title(0, "the data")

    # (2) members of the ensemble, stacked like cards: one partition and its local models each
    dx, dy = 0.015, 0.015 * W / H
    x1 = left + side + gap
    card = (side - 2 * dx) / side                     # three cards fit the slot of one panel
    for j, t in enumerate((7, 3, 0)):
        k = 2 - j                                     # the back card first, up and to the right
        square(x1 + k * dx, bottom + k * dy, members[:, t].reshape(m, m), smooth=False, scale=card)
    title(1, "300 partitions, a model per cell")

    # (3) the median of the members, (4) the probability of exceeding 1
    square(left + 2 * (side + gap), bottom, median)
    title(2, "the map: their median")
    square(left + 3 * (side + gap), bottom, exceed, cmap_=SCM.lajolla, lo=0, hi=1)
    title(3, "the law: P(Z > 1)")

    for k in range(3):
        x0 = left + k * (side + gap) + side + (0.022 if k == 1 else 0.012)
        fig.patches.append(FancyArrowPatch((x0, bottom + tall / 2), (left + (k + 1) * (side + gap) - 0.012 - (0.012 if k == 0 else 0),
                                                                     bottom + tall / 2),
                                           transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22,
                                           color=grey, linewidth=2))
    fig.savefig(path, dpi=150, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    block_mark(os.path.join(OUT, "block_mark.png"))
    home(os.path.join(os.path.dirname(OUT), "home.png"))
    print("written to", OUT)

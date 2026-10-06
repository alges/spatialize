"""Maps of a run, saved for human review (never an acceptance criterion).

Requested with ``run(..., save_maps=DIR)`` or ``python -m spatialize.scenarios --save-maps DIR``.
For each scenario and estimator, every replicate field gets its arrays (``.npy``) and a figure of
the truth and the estimated map on one colour scale, with the measured functionals overlaid; a
summary figure shows all fields at once. matplotlib is imported only here, when maps are requested.
"""
import os

import numpy as np


def _arrow(ax, theta_deg, color, label=None):
    """Axial direction through the centre of the unit square."""
    t = np.radians(theta_deg)
    dx, dy = 0.4 * np.cos(t), 0.4 * np.sin(t)
    ax.plot([0.5 - dx, 0.5 + dx], [0.5 - dy, 0.5 + dy], color=color, lw=2.0, label=label)


def _panel(ax, z, title, vmin, vmax, theta, target=None, samples=None):
    im = ax.imshow(z, origin="lower", extent=(0, 1, 0, 1), vmin=vmin, vmax=vmax, cmap="viridis")
    if samples is not None:
        ax.scatter(samples[:, 0], samples[:, 1], s=2, c="k", alpha=0.35, linewidths=0)
    if target is not None:
        _arrow(ax, target, "white", "target")
    _arrow(ax, theta, "red", "measured")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def save_map_fields(directory, scenario_id, estimator_id, fields, target_theta=None):
    """Save the maps of one scenario/estimator.

    Parameters
    ----------
    directory : str
        Root output directory; files go to ``<directory>/<scenario_id>/``.
    scenario_id, estimator_id : str
        Used in file names and titles.
    fields : list of dict
        One entry per replicate field with keys ``point`` and ``truth`` (2D arrays, rows = y) and
        ``theta``, ``coherence``, ``theta_truth``, ``coherence_truth`` (floats); optionally
        ``samples`` (n×2 sample locations, drawn on the truth panel).
    target_theta : float, optional
        Declared orientation of the truth, drawn as a reference.

    Returns
    -------
    list of str
        Paths of the files written.
    """
    from matplotlib.figure import Figure

    out_dir = os.path.join(directory, scenario_id)
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def save(fig, name):
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=110, bbox_inches="tight")
        written.append(path)

    for k, f in enumerate(fields):
        for name in ("point", "truth"):
            path = os.path.join(out_dir, f"{estimator_id}_{name}_{k}.npy")
            np.save(path, f[name])
            written.append(path)
        vmin = float(np.nanmin([np.nanmin(f["truth"]), np.nanmin(f["point"])]))
        vmax = float(np.nanmax([np.nanmax(f["truth"]), np.nanmax(f["point"])]))
        fig = Figure(figsize=(8.4, 4.0))
        a1, a2 = fig.subplots(1, 2)
        _panel(a1, f["truth"], f"truth  θ={f['theta_truth']:.1f}°  c={f['coherence_truth']:.2f}",
               vmin, vmax, f["theta_truth"], target_theta, f.get("samples"))
        im = _panel(a2, f["point"], f"{estimator_id} (median)  θ={f['theta']:.1f}°  c={f['coherence']:.2f}",
                    vmin, vmax, f["theta"], target_theta)
        fig.colorbar(im, ax=[a1, a2], shrink=0.85)
        a2.legend(loc="lower right", fontsize=7, framealpha=0.6)
        fig.suptitle(f"{scenario_id} — field {k}", fontsize=10)
        save(fig, f"{estimator_id}_{k}.png")

    n = len(fields)
    ncol = min(n, 5)
    nrow = int(np.ceil(n / ncol))
    fig = Figure(figsize=(2.3 * ncol, 4.6 * nrow), layout="constrained")
    axes = fig.subplots(2 * nrow, ncol, squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for k, f in enumerate(fields):
        r, c = 2 * (k // ncol), k % ncol
        vmin = float(np.nanmin([np.nanmin(f["truth"]), np.nanmin(f["point"])]))
        vmax = float(np.nanmax([np.nanmax(f["truth"]), np.nanmax(f["point"])]))
        for ax in (axes[r, c], axes[r + 1, c]):
            ax.set_axis_on()
        _panel(axes[r, c], f["truth"], f"truth {k}: {f['theta_truth']:.0f}°, c={f['coherence_truth']:.2f}",
               vmin, vmax, f["theta_truth"], target_theta)
        _panel(axes[r + 1, c], f["point"], f"{estimator_id} {k}: {f['theta']:.0f}°, c={f['coherence']:.2f}",
               vmin, vmax, f["theta"], target_theta)
    fig.suptitle(f"{scenario_id} — {estimator_id}: truth (upper) and median map (lower) per field", fontsize=10)
    save(fig, f"{estimator_id}_summary.png")
    return written


def save_comparison(directory, scenario_id, by_estimator, target_theta=None):
    """Save, per field, the truth next to the map of every estimator (one colour scale per field).

    Parameters
    ----------
    directory : str
        Root output directory; files go to ``<directory>/<scenario_id>/compare_<k>.png``.
    scenario_id : str
        Used in file names and titles.
    by_estimator : dict of str to list of dict
        For each estimator id, its fields as in :func:`save_map_fields`.
    target_theta : float, optional
        Declared orientation of the truth, drawn as a reference.

    Returns
    -------
    list of str
        Paths of the files written.
    """
    from matplotlib.figure import Figure

    out_dir = os.path.join(directory, scenario_id)
    os.makedirs(out_dir, exist_ok=True)
    names = list(by_estimator)
    n_fields = min(len(v) for v in by_estimator.values())
    written = []
    for k in range(n_fields):
        first = by_estimator[names[0]][k]
        maps_k = [first["truth"]] + [by_estimator[e][k]["point"] for e in names]
        vmin = float(np.nanmin([np.nanmin(z) for z in maps_k]))
        vmax = float(np.nanmax([np.nanmax(z) for z in maps_k]))
        fig = Figure(figsize=(2.8 * (len(names) + 1), 3.6), layout="constrained")
        axes = fig.subplots(1, len(names) + 1, squeeze=False)[0]
        _panel(axes[0], first["truth"], f"truth\nθ={first['theta_truth']:.0f}°  c={first['coherence_truth']:.2f}",
               vmin, vmax, first["theta_truth"], target_theta, first.get("samples"))
        for ax, e in zip(axes[1:], names):
            f = by_estimator[e][k]
            im = _panel(ax, f["point"], f"{e}\nθ={f['theta']:.0f}°  c={f['coherence']:.2f}",
                        vmin, vmax, f["theta"], target_theta)
        fig.colorbar(im, ax=list(axes), shrink=0.8)
        axes[-1].legend(*axes[1].get_legend_handles_labels(), loc="lower right", fontsize=7, framealpha=0.6)
        fig.suptitle(f"{scenario_id} — field {k}: truth and median maps", fontsize=10)
        path = os.path.join(out_dir, f"compare_{k}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        written.append(path)
    return written

"""Posterior analysis of the data: each datum read against the law the other data give it.

Run from the repository with ``python examples/scripted_examples/spa_example.py``. The figures are
written to a folder of the system's temporary directory, printed at the end.
"""
import os
import tempfile

import matplotlib

matplotlib.use("Agg")

from spatialize import logging
from spatialize.data import load_drill_holes_andes_2D
from spatialize.gs.esi import esi_hparams_search
from spatialize.gs.spa import posterior_audit

logging.log.setLevel("INFO")

samples, locations, _, _ = load_drill_holes_andes_2D()
points = samples[['x', 'y']].values
values = samples[['cu']].values[:, 0]
xi = locations[['x', 'y']].values


def main():
    # a cheap search for the decoder's parameters, then the audit with many partitions
    search = esi_hparams_search(points, values, xi, local_interpolator="idw", griddata=False, k=10,
                                exponent=[1.0, 2.0, 4.0], alpha=(0.7, 0.8, 0.9), n_partitions=[50], seed=1500)
    audit = posterior_audit(points, values, local_interpolator="idw", n_partitions=300, seed=1500,
                            best_params_found=search.best_result())

    # calibration first: the flags are to be read in its light
    calibration = audit.calibration()
    print(calibration["coverage"].round(3).to_string(index=False))
    print(f"spread factor {calibration['spread_factor']:.2f}; tails: {calibration['verdict']}, "
          f"centre: {calibration['centre']}")

    # the readings of every datum, the most surprising first
    table = audit.table().sort_values("tail_p")
    print(table.head(10).round(4).to_string())
    print(f"{int(table.flag.sum())} data flagged at a false discovery rate of 5 %")

    # how much of the domain each datum represents
    print(audit.declustered().round(4).to_string())

    # co-located data with different values
    print(audit.duplicates().head(5).to_string())

    out = os.path.join(tempfile.gettempdir(), "spatialize-examples", "spa_example")
    os.makedirs(out, exist_ok=True)
    for name, fig in (("calibration", audit.plot_calibration()), ("surprise_map", audit.plot_map()),
                      ("value_vs_position", audit.plot_value_pit()),
                      ("most_surprising", audit.plot_datum(int(table.index[0]))),
                      ("declustered", audit.plot_declustered()),
                      ("proportional_effect", audit.plot_proportional_effect())):
        fig.savefig(os.path.join(out, f"spa_{name}.png"), dpi=100)
    print(f"figures saved in {out}")


if __name__ == '__main__':
    main()

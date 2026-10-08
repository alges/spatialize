"""The example of the theory page on the block-mark model (section "The effect on model selection").

Leave-one-out mean absolute error of IDW for four granularities under each treatment of the cells
without data. Run from the repository root with the extension built:

    PYTHONPATH=.:src/python python docs/theory_example_empty_cells.py
"""
import numpy as np
from spatialize import session
from spatialize.gs.esi import esi_hparams_search

rng = np.random.default_rng(0)
points = rng.uniform(0, 10, (200, 2))
values = np.sin(points[:, 0]) + points[:, 1] / 5 + 0.1 * rng.normal(size=200)
xi = rng.uniform(0, 10, (50, 2))

policies = {"nan": dict(empty_cells="nan"),
            "mark, defaults": dict(empty_cells="mark"),
            "mark, the model": dict(empty_cells="mark", mark_source="cells", mark_value="datum")}
for name, policy in policies.items():
    with session.override(**policy):
        errors = [esi_hparams_search(points, values, xi, local_interpolator="idw", griddata=False, k=-1,
                                     alpha=(alpha,), exponent=(2.0,), n_partitions=(100,), seed=1,
                                     callback=lambda *a, **k: None).best_result()["cv_error"]
                  for alpha in (0.8, 0.9, 0.95, 0.98)]
    print(f"{name:16s}", " ".join(f"{e:.3f}" for e in errors))

import sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src/python")   # run from the repository root
import spatialize
from spatialize.gs.esi import esi_nongriddata
spatialize.session.set(empty_cells="coarsen", progress=False)
rng = np.random.default_rng(0); P = rng.random((2000, 2)); V = np.sin(6 * P[:, 0]) + rng.normal(0, .1, 2000)
g = (np.arange(250) + .5) / 250; X = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
extra = dict(coupling=sys.argv[1]) if len(sys.argv) > 1 else {}
for li, kw in (("draw", {}), ("wdraw_idw", dict(exponent=2.0)), ("wdraw_sharpidw", {})):
    best = []
    for rep in range(3):
        t = time.time(); r = esi_nongriddata(P, V, X, local_interpolator=li, p_process="voronoi", alpha=0.9, n_partitions=100, seed=0, **kw, **extra); best.append(time.time() - t)
    print(li, extra, f"{min(best):.2f} s", flush=True)

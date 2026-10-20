import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
import spatialize
from spatialize.gs.esi import esi_nongriddata
spatialize.session.set(empty_cells="coarsen", progress=False)

def vbm(seed, m, n_cells, sigma, n, design="uniform"):
    g = (np.arange(m) + 0.5) / m; xi = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    rng = np.random.default_rng(seed); nu = rng.random((n_cells, 2)); mk = rng.lognormal(0, sigma, n_cells)
    f = lambda x: mk[np.argmin(((x[:, None] - nu[None]) ** 2).sum(-1), 1)]
    p = rng.random((n, 2)) if design == "uniform" else np.vstack([rng.random((150, 2)), 0.25 + 0.5 * rng.random((150, 2))])
    return p, f(p), f(xi), xi

def inside(t, mem, axis=-1): lo, hi = np.nanquantile(mem, [0.05, 0.95], axis=axis); return (lo <= t) & (t <= hi)
def blocks(m, k): side = np.arange(m) // k; return (side[None, :] + (m // k) * side[:, None]).ravel(), (m // k) ** 2

for coupling in ("location", "cell"):
    print("==", coupling, flush=True)
    # 16: area over twenty fields
    cov = []; widths = []
    for seed in range(20):
        p, v, t, xi = vbm(seed, 36, 30, 1.0, 350)
        lim = np.quantile(t, 0.8)
        mem = esi_nongriddata(p, v, xi, local_interpolator="draw", p_process="voronoi", alpha=0.9, n_partitions=300, seed=0, coupling=coupling).esi_samples()
        ar = (mem > lim).mean(0); cov.append(inside(np.mean(t > lim), ar)); widths.append(np.diff(np.quantile(ar, [.05, .95]))[0])
    print(f"16 area coverage {np.mean(cov):.2f}, mean band width {np.mean(widths):.3f}", flush=True)
    # 17 part 1: curve
    p, v, t, xi = vbm(0, 36, 30, 0.8, 350)
    mem = esi_nongriddata(p, v, xi, local_interpolator="draw", p_process="voronoi", alpha=0.9, n_partitions=300, seed=0, coupling=coupling).esi_samples()
    cuts = np.linspace(np.quantile(t, .02), np.quantile(t, .95), 22)
    sh = np.stack([(mem > c).mean(0) for c in cuts]); tr = np.array([(t > c).mean() for c in cuts])
    gm = np.stack([np.where((mem > c).sum(0) > 0, (mem * (mem > c)).sum(0) / np.maximum((mem > c).sum(0), 1), np.nan) for c in cuts]); tg = np.array([t[t > c].mean() for c in cuts])
    print(f"17 curve: share inside at {inside(tr, sh).sum()}/22, mean inside at {inside(tg, gm).sum()}/22", flush=True)
    # 17 part 2: blocks on the preferential design
    p, v, t, xi = vbm(1, 36, 30, 0.8, 300, "preferential")
    mem = esi_nongriddata(p, v, xi, local_interpolator="draw", p_process="voronoi", alpha=0.9, n_partitions=300, seed=0, coupling=coupling).esi_samples()
    b, nb = blocks(36, 3); bm = np.stack([mem[b == j].mean(0) for j in range(nb)]); tb = np.array([t[b == j].mean() for j in range(nb)])
    q05, q50, q95 = np.quantile(bm, [.05, .5, .95], 1); h = (q95 - q05) / (2 * q50); err = np.abs(bm.mean(1) - tb) / tb
    cat = np.where(h <= .15, 0, np.where(h <= .3, 1, 2))
    print("17 blocks: coverage", np.mean(inside(tb, bm)).round(3), "by class", [np.mean(inside(tb, bm)[cat == k]).round(2) for k in range(3)],
          "counts", np.bincount(cat, minlength=3), "median error", [np.median(err[cat == k]).round(3) for k in range(3)], flush=True)
    # 11: blocks under the two asymmetric uses
    p, v, t, xi = vbm(0, 40, 40, 0.9, 300)
    def loss(r, z, o, u): return np.mean(o * np.clip(r - z, 0, None) + u * np.clip(z - r, 0, None))
    b, nb = blocks(40, 4); tb = np.array([t[b == j].mean() for j in range(nb)])
    for (o, u), rec in (((5., 1.), dict(local_interpolator="draw", p_process="voronoi", alpha=0.9)),
                        ((1., 5.), dict(local_interpolator="wdraw_sharpidw", p_process="voronoi", alpha=0.8))):
        mem = esi_nongriddata(p, v, xi, n_partitions=300, seed=0, coupling=coupling, **rec).esi_samples()
        bm = np.stack([mem[b == j].mean(0) for j in range(nb)]); lv = u / (o + u); pq = np.quantile(mem, lv, 1)
        print(f"11 use ({o:g},{u:g}): block coverage {np.mean(inside(tb, bm)):.3f}, loss block q {loss(np.quantile(bm, lv, 1), tb, o, u):.3f}, "
              f"mean of point q {loss(np.array([pq[b == j].mean() for j in range(nb)]), tb, o, u):.3f}, point quantile loss {loss(pq, t, o, u):.3f}", flush=True)

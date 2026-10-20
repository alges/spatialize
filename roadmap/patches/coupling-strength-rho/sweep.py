import numpy as np, warnings
warnings.filterwarnings("ignore")
exec(open(__file__.replace("sweep.py", "measure.py")).read().split("for coupling in")[0])
modes = [("location", {}), *[(f"rho {r}", dict(coupling="cell", rho=r)) for r in (0.25, 0.5, 0.75, 0.9)], ("cell", dict(coupling="cell"))]
for name, kw in modes:
    z, pit = [], []
    for seed in range(40):
        p, v, t, xi = vbm(seed, 36, 30, 1.0, 350); lim = np.quantile(t, 0.8)
        mem = esi_nongriddata(p, v, xi, local_interpolator="draw", p_process="voronoi", alpha=0.9, n_partitions=300, seed=0, **kw).esi_samples()
        ar = (mem > lim).mean(0); a = np.mean(t > lim); z.append((a - ar.mean()) / ar.std()); pit.append(np.mean(ar < a) + .5 * np.mean(ar == a))
    z, pit = np.array(z), np.array(pit)
    p, v, t, xi = vbm(0, 36, 30, 0.8, 350)
    mem = esi_nongriddata(p, v, xi, local_interpolator="draw", p_process="voronoi", alpha=0.9, n_partitions=300, seed=0, **kw).esi_samples()
    cuts = np.linspace(np.quantile(t, .02), np.quantile(t, .95), 22)
    curve = inside(np.array([(t > c).mean() for c in cuts]), np.stack([(mem > c).mean(0) for c in cuts])).sum()
    p, v, t, xi = vbm(1, 36, 30, 0.8, 300, "preferential")
    mem = esi_nongriddata(p, v, xi, local_interpolator="draw", p_process="voronoi", alpha=0.9, n_partitions=300, seed=0, **kw).esi_samples()
    b, nb = blocks(36, 3); bm = np.stack([mem[b == j].mean(0) for j in range(nb)]); tb = np.array([t[b == j].mean() for j in range(nb)])
    blk17 = np.mean(inside(tb, bm))
    p, v, t, xi = vbm(0, 40, 40, 0.9, 300); b, nb = blocks(40, 4); tb = np.array([t[b == j].mean() for j in range(nb)])
    mem = esi_nongriddata(p, v, xi, local_interpolator="wdraw_sharpidw", p_process="voronoi", alpha=0.8, n_partitions=300, seed=0, **kw).esi_samples()
    bm = np.stack([mem[b == j].mean(0) for j in range(nb)]); blk11 = np.mean(inside(tb, bm))
    print(f"{name:9s} 16: sd(z) {z.std():.2f} outside 5-95 {np.mean((pit<.05)|(pit>.95)):.2f} | 17 curve {curve}/22, blocks {blk17:.2f} | 11 blocks (wdraw_sharpidw) {blk11:.2f}", flush=True)

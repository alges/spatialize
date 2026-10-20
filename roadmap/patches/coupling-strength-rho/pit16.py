import numpy as np, warnings
warnings.filterwarnings("ignore")
exec(open(__file__.replace("pit16.py", "measure.py")).read().split("for coupling in")[0])
for coupling in ("location", "cell"):
    pit, z = [], []
    for seed in range(40):
        p, v, t, xi = vbm(seed, 36, 30, 1.0, 350)
        lim = np.quantile(t, 0.8)
        mem = esi_nongriddata(p, v, xi, local_interpolator="draw", p_process="voronoi", alpha=0.9, n_partitions=300, seed=0, coupling=coupling).esi_samples()
        ar = (mem > lim).mean(0); a = np.mean(t > lim)
        pit.append(np.mean(ar < a) + 0.5 * np.mean(ar == a)); z.append((a - ar.mean()) / ar.std())
    pit = np.array(pit); z = np.array(z)
    print(coupling, "PIT quartile counts", np.histogram(pit, bins=4, range=(0, 1))[0], "outside 5-95", np.mean((pit < .05) | (pit > .95)).round(3),
          "sd of standardised error", z.std().round(2), "mean", z.mean().round(2), flush=True)

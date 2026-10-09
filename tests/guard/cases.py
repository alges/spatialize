"""Cases of the guard checks (bitwise snapshots of the C++ extension).

Every function exported by ``libspatialize`` is called with fixed inputs, parameters and seeds.
The inputs are small geostatistical datasets (piecewise-constant Voronoi block-mark fields, the
book's Def 12.3.8) so the snapshots exercise realistic cell occupancies, including empty cells.

This guard is internal to spatialize: it is NOT part of the shared scenario suite and NOT an
acceptance criterion. Its only job is to prove that a pure refactor changes no output, bit for bit,
on the machine that produced the snapshots.
"""
import numpy as np

GENERATOR_SEED = 20261005


def _block_mark_field(rng, dim, n_samples, n_cells, n_queries):
    """Voronoi block-mark field on the unit cube: one lognormal mark per cell (Def 12.3.8)."""
    centres = rng.random((n_cells, dim))
    marks = rng.lognormal(0.0, 0.8, n_cells)

    def field(x):
        d = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        return marks[np.argmin(d, axis=1)]

    samples = rng.random((n_samples, dim))
    if dim == 2:
        g = (np.arange(n_queries) + 0.5) / n_queries
        gx, gy = np.meshgrid(g, g)
        queries = np.c_[gx.ravel(), gy.ravel()]
    else:
        queries = rng.random((n_queries, dim))
    return (samples.astype(np.float32), field(samples).astype(np.float32),
            queries.astype(np.float32))


def datasets():
    """Inputs of every case, regenerated deterministically (and stored inside each snapshot)."""
    rng = np.random.default_rng(GENERATOR_SEED)
    s2, v2, q2 = _block_mark_field(rng, 2, n_samples=80, n_cells=12, n_queries=15)  # 15x15 grid
    s3, v3, q3 = _block_mark_field(rng, 3, n_samples=80, n_cells=12, n_queries=60)
    # drawn after the others, so the 2D and 3D inputs stay those of the earlier snapshots
    s1, v1, q1 = _block_mark_field(rng, 1, n_samples=80, n_cells=12, n_queries=60)
    return {"2d": (s2, v2, q2), "3d": (s3, v3, q3), "1d": (s1, v1, q1)}


# --- decoders written in Python for the custom ESI bindings: the cell mean (block-mark decoder) ---
def _post_creation(coords, values):
    return np.array([values.mean() if len(values) else np.nan], dtype=np.float32)


def _estimation(coords, values, queries, params):
    return np.full(len(queries), values.mean(), dtype=np.float32)


def _loo(coords, values, params):
    n = len(values)
    if n < 2:
        return np.full(n, np.nan, dtype=np.float32)
    return ((values.sum() - values) / (n - 1)).astype(np.float32)


def _kfold(k, coords, values, folds, params):
    out = np.full(len(values), np.nan, dtype=np.float32)
    for f in np.unique(folds):
        test, train = folds == f, folds != f
        if train.any():
            out[test] = values[train].mean()
    return out


def _aggregation(values):
    return float(np.mean(values))


T, ALPHA, EXP, SEED, K, FSEED = 20, 0.7, 2.0, 1234, 5, 77
T_ADAPTIVE = 5                          # adaptive fitting is slow
KRIG = dict(model=2, nugget=0.1, range=0.3, sill=1.0)


def cases(lib):
    """(name, dataset key, callable(samples, values, queries) -> output) for every exported function.

    The adaptive decoder runs on one thread, which gives the same numbers as several.
    """
    m, n_, r_, s_ = KRIG["model"], KRIG["nugget"], KRIG["range"], KRIG["sill"]
    c = []
    add = lambda name, ds, fn: c.append((name, ds, fn))

    add("get_partitions_using_esi", "2d", lambda s, v, q: lib.get_partitions_using_esi(s, T, ALPHA, None, SEED))
    add("get_leaf_for_samples_using_esi", "2d", lambda s, v, q: lib.get_leaf_for_samples_using_esi(s, T, ALPHA, None, SEED))

    add("estimation_nn_idw", "2d", lambda s, v, q: lib.estimation_nn_idw(s, v, 0.3, EXP, q, None))
    add("loo_nn_idw", "2d", lambda s, v, q: lib.loo_nn_idw(s, v, 0.3, EXP, None))
    add("kfold_nn_idw", "2d", lambda s, v, q: lib.kfold_nn_idw(s, v, 0.3, EXP, K, FSEED, None))

    # The ensemble cases go through libspatialize.run. They keep the names, and the snapshots, of the
    # dedicated entry points they replaced (removed 2026-10-07), which run reproduces bit for bit.
    def run(partition, alpha, forest_size, decoder, params, method, num_threads=0):
        return lambda s, v, q: lib.run(s, v, q, partition, alpha, forest_size, SEED, decoder, params, method, K, FSEED,
                                       None, num_threads)
    custom = {"estimate": dict(post_creation=_post_creation, estimation=_estimation),
              "loo": dict(post_creation=_post_creation, loo=_loo),
              "kfold": dict(post_creation=_post_creation, kfold=_kfold)}
    krig = dict(model=m, nugget=n_, range=r_, sill=s_)
    for method, pre in (("estimate", "estimation"), ("loo", "loo"), ("kfold", "kfold")):
        add(f"{pre}_esi_idw", "2d", run("mondrian", ALPHA, T, "idw", {"exponent": EXP}, method))
        # the Mondrian partition of Spatialize 1.2, kept as "mondrian-legacy": its snapshots are those
        # the default partition gave before it became the theory's process (2026-10-09)
        add(f"{pre}_esi_idw_legacy", "2d", run("mondrian-legacy", ALPHA, T, "idw", {"exponent": EXP}, method))
        for d in ("2d", "3d"):
            add(f"{pre}_esi_kriging_{d}", d, run("mondrian", ALPHA, T, "kriging", krig, method))
        for tag, a in (("dc", 0.5), ("nodc", -0.5)):      # data-conditioned / not (sign of alpha)
            add(f"{pre}_voronoi_idw_{tag}", "2d", run("voronoi", a, T, "idw", {"exponent": EXP}, method))
        for d in ("2d", "3d"):
            add(f"{pre}_adaptive_esi_idw_{d}", d, run("mondrian", ALPHA, T_ADAPTIVE, "adaptiveidw",
                                                      {"metric": "mae"}, method, num_threads=1))
        add(f"{pre}_custom_esi", "2d", run("mondrian", ALPHA, T, "custom", custom[method], method))

    # the decoders of phase 2 (draws, cell mean, sharpened adaptive IDW), pinned from their first version
    phase2 = [("cellmean", {}, T), ("draw", {}, T), ("wdraw_idw", {"exponent": EXP}, T),
              ("wdraw_kriging", dict(krig, negative_weights="clip"), T),
              ("sharpidw", {}, T_ADAPTIVE), ("wdraw_adaptiveidw", {}, T_ADAPTIVE), ("wdraw_sharpidw", {}, T_ADAPTIVE)]
    for decoder, params, forest_size in phase2:
        for method, pre in (("estimate", "estimation"), ("loo", "loo"), ("kfold", "kfold")):
            add(f"{pre}_{decoder}", "2d", run("mondrian", ALPHA, forest_size, decoder, params, method))

    # the adaptive decoders on a line, where only the exponent is fitted (since 2026-10-07)
    for decoder in ("adaptiveidw", "sharpidw", "wdraw_adaptiveidw", "wdraw_sharpidw"):
        for method, pre in (("estimate", "estimation"), ("loo", "loo"), ("kfold", "kfold")):
            add(f"{pre}_{decoder}_1d", "1d", run("mondrian", ALPHA, T_ADAPTIVE, decoder, {}, method))

    # co-estimation: two variables observed at the same locations (3D samples array: variables x n x d)
    co = lambda s, v: (np.stack([s, s]), np.stack([v, np.log(v)]).astype(np.float32))
    add("estimation_custom_coesi", "2d", lambda s, v, q: lib.estimation_custom_coesi(*co(s, v), T, ALPHA, SEED, q, None, lambda c_, v_, l_, p_: np.full(len(l_), v_.mean(), np.float32), _post_creation, _estimation, _aggregation, None))
    add("marginal_loo_custom_coesi", "2d", lambda s, v, q: lib.marginal_loo_custom_coesi(*co(s, v), T, ALPHA, SEED, q, _post_creation, _loo, None))
    add("marginal_kfold_custom_coesi", "2d", lambda s, v, q: lib.marginal_kfold_custom_coesi(*co(s, v), T, ALPHA, SEED, K, FSEED, q, _post_creation, _kfold, None))
    return c


def normalise(out):
    """Turn a binding's return value into a dict of numpy arrays (tuples/lists flattened)."""
    if isinstance(out, tuple):
        res = {}
        for i, o in enumerate(out):
            if o is not None:
                for k, a in normalise(o).items():
                    res[f"{i}.{k}"] = a
        return res
    if isinstance(out, list):            # get_partitions: ragged list of trees -> one array per tree
        return {f"tree{i}": np.asarray(t, dtype=np.float32) for i, t in enumerate(out)}
    return {"out": np.asarray(out)}

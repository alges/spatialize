"""Cases of the public-API guard: bitwise snapshots of the Python functions users call.

The internal cases (``cases.py``) pin the compiled entry points. These pin the public API above
them (``spatialize.gs.*``), so that a change of the facade between Python and C++ — its dispatch,
argument lists and defaults — can be shown to leave every result unchanged. Every seed is explicit;
the inputs are the datasets of ``cases.py``.
"""
import os
import sys
import warnings

import numpy as np

_REPO_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "python"))

T, ALPHA, EXP, SEED, K, FSEED = 20, 0.7, 2.0, 1234, 5, 77
T_ADAPTIVE = 5
KRIG = dict(model="exponential", nugget=0.1, range=0.3, sill=1.0)
NULL = dict(callback=lambda msg: None)


def _esi(result):
    return {"estimation": np.asarray(result.estimation(), dtype=np.float64),
            "samples": np.asarray(result.esi_samples(raw=True))}


def _frame(df):
    return {"table": df.select_dtypes("number").to_numpy(dtype=np.float64)}


def public_cases():
    """(name, dataset key, callable(samples, values, queries) -> dict of arrays)."""
    if _REPO_PYTHON not in sys.path:  # the public API of this checkout, never an installed one
        sys.path.insert(0, _REPO_PYTHON)
    from spatialize.gs.esi import esi_nongriddata, esi_griddata, esi_hparams_search, esi_pareto_hparams_search
    from spatialize.gs.cat_esi import cat_esi_nongriddata
    from spatialize.gs.idw import idw_nongriddata, idw_hparams_search
    from spatialize.gs.spa import cv_sample_pred_posterior
    from spatialize.gs.esmi import SpatialEntropy

    c = []
    add = lambda name, ds, fn: c.append((f"api.{name}", ds, fn))
    common = dict(n_partitions=T, alpha=ALPHA, seed=SEED, **NULL)

    def nongrid(li, **kw):
        return lambda s, v, q: _esi(esi_nongriddata(s, v, q, local_interpolator=li, **common, **kw))

    add("esi_nongriddata_idw", "2d", nongrid("idw", exponent=EXP))
    add("esi_nongriddata_idw_3d", "3d", nongrid("idw", exponent=EXP))
    add("esi_nongriddata_kriging", "2d", nongrid("kriging", **KRIG))
    add("esi_nongriddata_kriging_3d", "3d", nongrid("kriging", **KRIG))
    add("esi_nongriddata_voronoi_dc", "2d", nongrid("idw", exponent=EXP, p_process="voronoi", data_cond=True))
    add("esi_nongriddata_voronoi_nodc", "2d", nongrid("idw", exponent=EXP, p_process="voronoi", data_cond=False))
    add("esi_nongriddata_mondrian_raw", "2d", nongrid("idw", exponent=EXP, p_process="mondrian-raw"))
    for d in ("2d", "3d"):
        add(f"esi_nongriddata_adaptive_{d}", d, lambda s, v, q: _esi(esi_nongriddata(
            s, v, q, local_interpolator="adaptiveidw", n_partitions=T_ADAPTIVE, alpha=ALPHA, seed=SEED, **NULL)))

    def grid(s, v, q):
        gx, gy = np.mgrid[0:1:15j, 0:1:15j]
        return _esi(esi_griddata(s, v, (gx, gy), local_interpolator="idw", exponent=EXP, **common))
    add("esi_griddata_idw", "2d", grid)

    def search(li, k, **kw):
        return lambda s, v, q: _frame(esi_hparams_search(
            s, v, q, local_interpolator=li, k=k, n_partitions=[T], alpha=[0.6, ALPHA], seed=SEED,
            folding_seed=FSEED, **NULL, **kw).cv_error)
    add("esi_hparams_search_idw_kfold", "2d", search("idw", K, exponent=[1.0, EXP]))
    add("esi_hparams_search_idw_loo", "2d", search("idw", -1, exponent=[1.0, EXP]))
    add("esi_hparams_search_kriging", "2d", search("kriging", K, model=["exponential"], nugget=[0.1],
                                                    range=[0.3], sill=[1.0]))
    add("esi_hparams_search_voronoi", "2d", search("idw", K, exponent=[EXP], p_process="voronoi"))

    def pareto(s, v, q):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = esi_pareto_hparams_search(s, v, local_interpolator="idw", n_partitions=[30], alpha=[ALPHA],
                                          exponent=[EXP], k=K, seed=SEED, folding_seed=FSEED, **NULL)
        return {"errors": np.array([[x["epsilon"], x["decoder_error"]] for x in r.all_results])}
    add("esi_pareto_hparams_search", "2d", pareto)

    def spa(s, v, q):
        r = cv_sample_pred_posterior(s, v, q, local_interpolator="idw", exponent=EXP, k=K, n_partitions=T,
                                     alpha=ALPHA, seed=SEED, folding_seed=FSEED, **NULL)
        return {"posterior": np.asarray(r.post_result)}
    add("cv_sample_pred_posterior", "2d", spa)

    def esmi(s, v, q):
        h = SpatialEntropy(T=T, M=10, alpha_t=ALPHA, alpha_m=0.6, seed=SEED, callback=NULL["callback"],
                           exponent=EXP).calculate_entropy(s, v, q[:20])
        return {"entropy": np.asarray(h, dtype=np.float64)}
    add("spatial_entropy", "2d", esmi)

    def cat(s, v, q):
        cats = np.where(v > np.median(v), "high", "low")
        r = cat_esi_nongriddata(s, cats, q, n_partitions=T, alpha=ALPHA, seed=SEED, **NULL)
        return {"estimation": np.asarray(r.estimation()).astype(str),
                "samples": np.asarray(r.esi_samples(raw=True)).astype(str)}
    add("cat_esi_nongriddata", "2d", cat)

    from spatialize.gs.partitions import cell_labels, partition_law
    for proc, a in (("mondrian", ALPHA), ("mondrian-raw", ALPHA), ("voronoi", 0.5)):
        add(f"cell_labels_{proc}", "2d", lambda s, v, q, proc=proc, a=a: {
            "labels": cell_labels(s, q[:30], p_process=proc, alpha=a, n_partitions=T, seed=SEED).astype(np.int64)})

    add("idw_nongriddata", "2d", lambda s, v, q: {"estimation": np.asarray(
        idw_nongriddata(s, v, q, radius=0.3, exponent=EXP, **NULL).estimation(), dtype=np.float64)})
    add("idw_hparams_search", "2d", lambda s, v, q: _frame(idw_hparams_search(
        s, v, q, k=K, radius=(0.2, 0.3), exponent=(1.0, EXP), folding_seed=FSEED, **NULL).cv_error))
    return c

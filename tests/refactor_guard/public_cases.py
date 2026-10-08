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
    from spatialize.futures.esmi import SpatialEntropy, SpatialMutualInformation

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

    def with_marks(source, value="decoder"):
        def fn(s, v, q):
            from spatialize import session
            q_wide = np.vstack([q, q * 1.6 - 0.3]).astype(np.float32)   # locations beyond the data, in empty cells
            with session.override(empty_cells="mark", mark_source=source, mark_value=value):
                return _esi(esi_nongriddata(s, v, q_wide, local_interpolator="idw", exponent=EXP, **common))
        return fn
    add("esi_nongriddata_mark_local", "2d", with_marks("local"))
    add("esi_nongriddata_mark_cells", "2d", with_marks("cells"))
    add("esi_nongriddata_mark_data", "2d", with_marks("data"))
    add("esi_nongriddata_mark_blockmark", "2d", with_marks("cells", "datum"))

    def coarsened(li, **kw):
        def fn(s, v, q):
            from spatialize import session
            q_wide = np.vstack([q, q * 1.6 - 0.3]).astype(np.float32)   # locations beyond the data, in empty cells
            with session.override(empty_cells="coarsen"):
                return _esi(esi_nongriddata(s, v, q_wide, local_interpolator=li, **kw, **common))
        return fn
    add("esi_nongriddata_coarsen_idw", "2d", coarsened("idw", exponent=EXP))
    add("esi_nongriddata_coarsen_voronoi", "2d", coarsened("idw", exponent=EXP, p_process="voronoi", data_cond=False))
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

    def pareto(**kw):
        def fn(s, v, q):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = esi_pareto_hparams_search(s, v, local_interpolator="idw", n_partitions=[30], alpha=[ALPHA],
                                              exponent=[EXP], k=K, seed=SEED, folding_seed=FSEED, **kw, **NULL)
            return {"errors": np.array([[x["epsilon"], x["decoder_error"]] for x in r.all_results])}
        return fn
    add("esi_pareto_hparams_search", "2d", pareto())
    # the encoder error on a Voronoi partition, possible since it reads the cells through cells()
    add("esi_pareto_hparams_search_voronoi", "2d", pareto(p_process="voronoi", data_cond=False))

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

    def esmi_voronoi(s, v, q):
        h = SpatialEntropy(T=T, M=10, alpha_t=0.5, alpha_m=0.6, seed=SEED, callback=NULL["callback"],
                           p_process="voronoi", data_cond=False, exponent=EXP).calculate_entropy(s, v, q[:20])
        return {"entropy": np.asarray(h, dtype=np.float64)}
    add("spatial_entropy_voronoi", "2d", esmi_voronoi)

    def mutual_information(s, v, q):
        smi = SpatialMutualInformation(T=T, M=10, alpha_t=ALPHA, alpha_m=0.6, seed=SEED,
                                       callback=NULL["callback"], exponent=EXP)
        mi = smi.calculate_mutual_information(s, v, s, np.log(v), q[:20])
        return {"mi": np.asarray(mi, np.float64), "h_u": np.asarray(smi.entropies_u, np.float64),
                "h_joint": np.asarray(smi.entropies_joint, np.float64)}
    add("spatial_mutual_information", "2d", mutual_information)

    def coesi(s, v, q):
        from spatialize.futures.coesi import coesi_nongriddata
        import cases as _c
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = coesi_nongriddata([s, s], [v, np.log(v)], q, n_aux=60,
                                  co_estimation=lambda c_, v_, l_, p_: np.full(len(l_), v_[:, 0].mean() - v_[:, 1].mean(), np.float32),
                                  estimation=_c._estimation, post_creation=_c._post_creation,
                                  n_partitions=T, alpha=ALPHA, seed=SEED)
        return {"members": np.asarray(r.esi_samples(raw=True), np.float32)}
    add("coesi_nongriddata", "2d", coesi)

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

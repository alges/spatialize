"""Summaries of the results: text, HTML and rich renderings, with the figures they report."""
import numpy as np

import cases
import snapshot_lib as sl

LIB = sl.load_lib()
DATA = cases.datasets()


def _renders(obj):
    from rich.console import Console
    text, page = repr(obj), obj._repr_html_()
    console = Console(record=True, width=100, force_terminal=False)
    console.print(obj)
    return text, page, console.export_text()


def test_estimation_and_search_summaries_report_their_figures():
    """The summary of an ESI estimation reports its decoder, partition, number of partitions and
    the statistics of the estimate, in the same figures as text, HTML and rich; a search reports its
    configurations and its best cross-validation error."""
    from spatialize import session, _display
    from spatialize.gs.esi import esi_nongriddata, esi_hparams_search
    s, v, q = DATA["2d"]
    with session.override(display="silent"):
        r = esi_nongriddata(s, v, q, local_interpolator="idw", exponent=2.0, n_partitions=cases.T,
                            alpha=cases.ALPHA, seed=cases.SEED)
        search = esi_hparams_search(s, v, q, local_interpolator="idw", griddata=False, k=5, exponent=[1.0, 2.0],
                                    alpha=[cases.ALPHA], n_partitions=[cases.T], seed=cases.SEED)
    est = np.asarray(r.estimation(), float)
    for out in _renders(r):
        for figure in ("idw", "mondrian", str(cases.T), _display.fmt(float(np.nanmean(est))),
                       _display.fmt(float(np.nanmax(est)))):
            assert figure in out
    best = _display.fmt(float(search.search_result_data["cv_error"].min()))
    for out in _renders(search):
        assert best in out and "2" in out


def test_summary_of_the_posterior_analysis():
    """The summary of the posterior analysis reports the data, the calibration verdict and the
    number of flags."""
    from spatialize.gs.spa import posterior_audit
    s, v, _ = DATA["2d"]
    audit = posterior_audit(s, v, n_partitions=cases.T, alpha=cases.ALPHA, seed=cases.SEED,
                            callback=lambda *a, **k: None)
    verdict = audit.calibration()["verdict"]
    for out in _renders(audit):
        assert str(len(v)) in out and verdict in out and "flagged" in out

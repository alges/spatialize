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
    audit = posterior_audit(s, v, n_partitions=cases.T, alpha=cases.ALPHA, seed=cases.SEED)
    verdict = audit.calibration()["verdict"]
    for out in _renders(audit):
        assert str(len(v)) in out and verdict in out and "flagged" in out


def test_progress_setting_hides_the_bars_but_not_the_warnings(capsys):
    """With the session setting progress=False no bar is drawn, while warnings are still shown; with
    progress=True the bars of the plain display appear."""
    from spatialize import session
    from spatialize.gs.esi import esi_nongriddata
    from spatialize.logging import log_message, logger
    s, v, q = DATA["2d"]
    with session.override(display="plain", progress=False):
        esi_nongriddata(s, v, q, local_interpolator="idw", exponent=2.0, n_partitions=cases.T, seed=cases.SEED)
        log_message(logger.warning("a warning that must show"))
    err = capsys.readouterr().err
    assert "computing estimates" not in err and "a warning that must show" in err
    with session.override(display="plain", progress=True):
        esi_nongriddata(s, v, q, local_interpolator="idw", exponent=2.0, n_partitions=cases.T, seed=cases.SEED)
    assert "computing estimates" in capsys.readouterr().err

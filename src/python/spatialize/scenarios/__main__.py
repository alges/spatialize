"""Command line entry point: ``python -m spatialize.scenarios``.

Runs the catalogue (or part of it) with spatialize's runner under one Holm budget, prints the
report and exits with status 0 when every check passed, 1 when any check failed and 2 on a usage
error. See the documentation, *Running the suite*.
"""
import argparse
import sys

from . import VERSION, catalog, run
from .stats.budget import ALPHA_SUITE

_STATUS_STYLE = {"PASS": "success", "FAIL": "error", "KNOWN": "warning", "XPASS": "critical", "SKIPPED": "debug"}


def _styled():
    """The rich console and colours when the output is a terminal (Spatialize's ``display``), else
    None: the plain text of the report then goes to logs and files unchanged."""
    try:
        from spatialize import _display
        if _display.mode() != "terminal":
            return None
        return _display._rich_console(), _display.COLOURS
    except Exception:
        return None


def _report_styled(styled):
    """The rich console and colours for the report, which goes to standard output: styled only when
    standard output is itself a terminal, so that a report piped to a file arrives as plain text
    (the progress, on standard error, keeps its own style)."""
    if styled is None or not sys.stdout.isatty():
        return None
    from rich.console import Console
    from spatialize import _display
    return Console(file=sys.stdout, highlight=False, soft_wrap=False, theme=_display._theme()), styled[1]


def _progress_printer(styled):
    if styled is None:
        return lambda msg: print(msg, file=sys.stderr, flush=True)
    console, c = styled
    from rich.markup import escape

    def show(msg):
        text = escape(msg)
        if msg.startswith("["):                              # a scenario starts
            console.print(f"[bold {c['brand']}]{text}[/]")
        elif msg.strip().endswith(" s") and " done in " in msg:
            console.print(f"[{c['success']}]✔[/] [dim]{text.strip()}[/]")
        elif ": p = " in msg or "skipped" in msg:
            console.print(f"[dim]{text}[/]")
        else:
            console.print(f"[dim]{text}[/]")
    return show


def _print_report(report, styled):
    if styled is None:
        print(report.table())
        return
    console, c = styled
    from rich.table import Table
    from rich import box
    from rich.markup import escape
    table = Table(title=f"[bold {c['brand']}]spatialize[/] [dim]· conformance · runner {report.runner} · mode "
                        f"{report.mode} · seed {report.seed} · α_suite {report.alpha:g} (Holm)[/]",
                  title_justify="left", title_style="", box=box.SIMPLE_HEAD,
                  header_style=f"bold {c['brand_dark']}", pad_edge=False)
    for name, kw in (("scenario/check", {}), ("family", {}), ("p-value", {"justify": "right"}),
                     ("level", {"justify": "right"}), ("expect", {}), ("result", {}), ("detail", {"overflow": "fold"})):
        table.add_column(name, **kw)
    for o in report.outcomes:
        style = c[_STATUS_STYLE[o.status]]
        if o.skipped:
            table.add_row(escape(f"{o.scenario}/{o.check}"), "", "", "", "", f"[{style}]SKIPPED[/]",
                          f"[dim]{escape(o.skipped)}[/]")
            continue
        level = "exact" if o.result.family == "almost-sure" else f"{o.level:.2g}"
        table.add_row(escape(f"{o.scenario}/{o.check}"), o.result.family, f"{o.result.p_value:.3g}", level,
                      o.expect, f"[bold {style}]{o.status}[/]", f"[dim]{escape(o.result.detail)}[/]")
    console.print(table)
    if report.durations:
        from .engine import duration
        times = Table(title=f"[bold {c['brand']}]time per scenario[/]", title_justify="left", title_style="",
                      box=box.SIMPLE_HEAD, header_style=f"bold {c['brand_dark']}", pad_edge=False)
        times.add_column("scenario")
        times.add_column("time", justify="right")
        times.add_column("share", justify="right")
        for k, t in report.durations.items():
            share = t / report.total if report.total > 0 else 0.0
            times.add_row(escape(k), duration(t), f"[dim]{100 * share:.1f} %[/]")
        times.add_section()
        times.add_row("[bold]total[/]", f"[bold]{duration(report.total)}[/]", "")
        if report.paused() > 0:
            times.add_row("wall clock", duration(report.wall),
                          f"[{c['warning']}]the computer slept or the run was paused for {duration(report.paused())}[/]")
        console.print(times)


def main(argv=None):
    """Command line entry point (``python -m spatialize.scenarios``).

    Parameters
    ----------
    argv : list of str, optional
        Arguments (default: ``sys.argv[1:]``); see ``--help`` and the documentation page *Running
        the tests*.

    Returns
    -------
    int
        Exit status: 0 when every outcome is as expected (passed, or a recorded known failure),
        1 when any check failed or unexpectedly passed, 2 on a usage error.
    """
    parser = argparse.ArgumentParser(
        prog="python -m spatialize.scenarios",
        description="Run spatialize's geostatistical conformance scenarios.")
    parser.add_argument("--mode", choices=("ci", "full"), default="ci",
                        help="ci: minimum detectable effect ~0.05, fast (default); "
                             "full: ~0.02, slow (release)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed of the run (default: fresh, printed in the report)")
    parser.add_argument("--tier", choices=("T1", "T2", "T3"), default=None,
                        help="run only the scenarios of this tier")
    parser.add_argument("--id", dest="ids", action="append", default=None, metavar="SCENARIO",
                        help="run only this scenario (repeatable)")
    parser.add_argument("--alpha", type=float, default=ALPHA_SUITE,
                        help=f"family-wise error rate of the run (default {ALPHA_SUITE:g})")
    parser.add_argument("--save-maps", metavar="DIR", default=None,
                        help="save the maps computed by the run (arrays and figures) under DIR")
    parser.add_argument("--report", metavar="FILE", default=None,
                        help="also write the report, as plain text, to FILE")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the report, without the progress of the run (on stderr)")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    parser.add_argument("--version", action="version", version=f"spatialize.scenarios {VERSION}")
    args = parser.parse_args(argv)

    selected = catalog(tier=args.tier, ids=args.ids)
    if args.ids:
        unknown = sorted(set(args.ids) - set(selected))
        if unknown:
            parser.error(f"unknown scenario(s): {', '.join(unknown)}")

    if args.list:
        for sc in selected.values():
            checks = ", ".join(c["id"] for c in sc.spec["checks"])
            print(f"{sc.id:34s} {sc.tier}  checks: {checks}")
        return 0
    if not selected:
        parser.error("no scenario selected")

    from .runners.spatialize import SpatializeRunner  # loads the compiled library
    styled = _styled()
    if not args.quiet:
        start = (f"seed {args.seed if args.seed is not None else '(fresh, in the report)'}, mode {args.mode}; "
                 "progress on stderr, report at the end")
        if styled is None:
            print(start, file=sys.stderr, flush=True)
        else:
            styled[0].print(f"[bold {styled[1]['brand']}]spatialize[/] [dim]· conformance · {start}[/]")
    report = run(selected, SpatializeRunner(), mode=args.mode, seed=args.seed, alpha=args.alpha,
                 save_maps=args.save_maps, progress=None if args.quiet else _progress_printer(styled))
    out = _report_styled(styled)
    _print_report(report, out)
    count = {k: sum(1 for o in report.outcomes if o.status == k) for k in ("PASS", "FAIL", "KNOWN", "XPASS", "SKIPPED")}
    known = f", {count['KNOWN']} known failures" if count["KNOWN"] else ""
    xpass = f", {count['XPASS']} unexpectedly passed (update their known_failure)" if count["XPASS"] else ""
    from .engine import duration
    summary = (f"{count['PASS']} passed, {count['FAIL']} failed{known}{xpass}, {count['SKIPPED']} skipped "
               f"in {duration(report.total)} (reproduce with --mode {report.mode} --seed {report.seed})")
    if out is None:
        print("\n" + summary)
    else:
        console, c = out
        mark, colour = ("✔", c["success"]) if report.passed else ("✖", c["error"])
        console.print(f"\n[bold {colour}]{mark}[/] [bold {c['brand']}]spatialize[/] [dim]·[/] {summary}")
    if args.save_maps:
        print(f"maps saved under {args.save_maps}")
    if args.report:
        with open(args.report, "w") as f:
            f.write(report.table() + "\n\n" + summary + "\n")
        print(f"report written to {args.report}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())

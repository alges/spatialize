"""Command line entry point: ``python -m spatialize.scenarios``.

Runs the catalogue (or part of it) with spatialize's runner under one Holm budget, prints the
report and exits with status 0 when every check passed, 1 when any check failed and 2 on a usage
error. See the documentation, *Running the suite*.
"""
import argparse
import sys

from . import VERSION, catalog, run
from .stats.budget import ALPHA_SUITE


def main(argv=None):
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
    report = run(selected, SpatializeRunner(), mode=args.mode, seed=args.seed, alpha=args.alpha)
    print(report.table())
    n_fail = sum(1 for o in report.outcomes if not o.skipped and not o.passed)
    n_skip = sum(1 for o in report.outcomes if o.skipped)
    n_pass = len(report.outcomes) - n_fail - n_skip
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped "
          f"(reproduce with --mode {report.mode} --seed {report.seed})")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())

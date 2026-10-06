"""Scenario loading, evaluation and reporting.

A run evaluates one or more scenarios with a :class:`~spatialize.scenarios.protocol.Runner`,
collects every check's p-value, applies Holm's procedure to the whole run (α_suite = 1e-3) and
decides each check with its family's pass rule (see ``stats.families``). Checks may carry an
``expect`` per encoder profile: ``reject`` marks a *negative control* — the run is correct when the
test rejects (it proves the test has the power to see a known deviation).
"""
import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .protocol import EstimatorSpec, Runner
from .stats import budget, families, maps
from . import generators

CATALOG = os.path.join(os.path.dirname(__file__), "catalog")
MODES = ("ci", "full")


# ----------------------------------------------------------------------------- loading
def _yaml():
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError("spatialize.scenarios needs PyYAML: pip install 'spatialize[scenarios]'") from e
    return yaml


@dataclass
class Scenario:
    """One scenario of the catalogue: its descriptor and its directory.

    Attributes
    ----------
    id : str
        Scenario identifier (e.g. ``"S03-anisotropic-field"``).
    path : str
        Directory holding ``scenario.yaml`` and, for pinned data, ``data/`` and
        ``CHECKSUMS.sha256``.
    spec : dict
        The parsed ``scenario.yaml``: the normative definition of the scenario.
    """
    id: str
    path: str
    spec: dict

    @property
    def tier(self):
        return self.spec["tier"]

    def file(self, rel):
        return os.path.join(self.path, rel)

    def verify_checksums(self):
        """Check the pinned data files against ``CHECKSUMS.sha256``; raise ``ValueError`` on mismatch."""
        path = self.file("CHECKSUMS.sha256")
        if not os.path.exists(path):
            return
        for line in open(path):
            digest, rel = line.split()
            h = hashlib.sha256(open(self.file(rel), "rb").read()).hexdigest()
            if h != digest:
                raise ValueError(f"{self.id}: checksum mismatch for {rel}")

    def field(self, k):
        """Pinned data of field k: dict with samples, values, truth (if any)."""
        d = {}
        for name in ("samples", "values", "truth"):
            f = self.file(f"data/{name}_{k}.npy")
            if os.path.exists(f):
                d[name] = np.load(f)
        return d

    def estimator(self, est, encoder_profile=None):
        """The :class:`~spatialize.scenarios.protocol.EstimatorSpec` of an ``estimators`` entry."""
        return EstimatorSpec(id=est["id"], encoder=est["encoder"], rate=est.get("rate"),
                             domain=tuple(map(tuple, self.spec["domain"]["box"])),
                             decoder=est["decoder"], params=dict(est.get("params", {})),
                             empty_cells=est.get("empty_cells", "nan"))


def catalog(tier=None, ids=None):
    """Load the scenarios of the catalogue, optionally filtered.

    Parameters
    ----------
    tier : str, optional
        Keep only scenarios of this tier (``"T1"``, ``"T2"`` or ``"T3"``).
    ids : sequence of str, optional
        Keep only scenarios with these identifiers.

    Returns
    -------
    dict of str to Scenario
        Scenarios by identifier, in catalogue order. The checksums of pinned data are verified
        on loading.
    """
    yaml = _yaml()
    out = {}
    for name in sorted(os.listdir(CATALOG)):
        p = os.path.join(CATALOG, name, "scenario.yaml")
        if not os.path.exists(p):
            continue
        spec = yaml.safe_load(open(p))
        sc = Scenario(spec["id"], os.path.dirname(p), spec)
        if (tier and sc.tier != tier) or (ids and sc.id not in ids):
            continue
        sc.verify_checksums()
        out[sc.id] = sc
    return out


# ----------------------------------------------------------------------------- evaluators
@dataclass
class CheckOutcome:
    """Outcome of one check: its test result, its Holm level and the decision."""
    scenario: str
    check: str
    title: str
    result: families.TestResult
    expect: str = "pass"            # "pass" | "reject" (negative control)
    level: float = float("nan")
    rejected: bool = False
    passed: Optional[bool] = None
    skipped: str = ""


def _mode_value(v, mode):
    return v[mode] if isinstance(v, dict) and mode in v else v


def _expect(check, profile):
    exp = check.get("expect", {})
    return exp.get(profile, exp.get("default", "pass")) if isinstance(exp, dict) else exp


def eval_pair_cooccurrence(sc: Scenario, runner: Runner, mode: str, seed: int,
                           save_maps=None) -> List[CheckOutcome]:
    """Pair co-occurrence e({datum, query}) read from the members: with a single datum
    and empty cells as NaN, a member is finite exactly when the query shares the datum's cell."""
    s = sc.spec
    centre = np.asarray(s["data"]["datum"], float)
    hs = np.asarray(s["data"]["displacements"], float)
    out = []
    for check in s["checks"]:
        est = sc.estimator(next(e for e in s["estimators"] if e["id"] == check["estimator"]))
        profile = runner.profile(est.encoder) if hasattr(runner, "profile") else est.encoder
        title = f"{check['title']} [{profile}]"
        if not runner.supports(est):
            out.append(CheckOutcome(sc.id, check["id"], title, families.TestResult("gof-closed", np.nan, 1.0, "not_reject"),
                                    skipped=f"{runner.name} does not support {est.encoder}/{est.decoder}"))
            continue
        q, l1 = [], []
        for d in s["data"]["directions"]:
            u = np.asarray(d, float); u = u / np.abs(u).sum()
            for h in hs:
                q.append(centre + h * u); l1.append(h)
        q = np.asarray(q, np.float32)
        n = int(_mode_value(check["n_members"], mode))
        m = runner.members(est, centre[None, :].astype(np.float32), np.ones(1, np.float32), q, n_members=n, seed=seed)
        p_hat = np.isfinite(m).mean(axis=1)
        p0 = np.exp(-est.rate * np.asarray(l1))
        r = families.gof_proportions(p_hat, p0, n)
        out.append(CheckOutcome(sc.id, check["id"], title, r, expect=_expect(check, profile)))
    return out


def eval_map_visual(sc: Scenario, runner: Runner, mode: str, seed: int,
                    save_maps=None) -> List[CheckOutcome]:
    """Visual criteria on point maps over K pinned fields (documentation: Visual criteria).

    With ``save_maps`` (a directory), the maps of every field are also saved for human review.
    """
    s = sc.spec
    m_grid = s["data"]["grid"]
    queries = generators.grid(m_grid).astype(np.float32)
    K = int(_mode_value(s["data"]["fields"], mode))
    est_cache: Dict[str, list] = {}
    maps_by_est: Dict[str, list] = {}
    out = []
    for check in s["checks"]:
        est = sc.estimator(next(e for e in s["estimators"] if e["id"] == check["estimator"]))
        profile = runner.profile(est.encoder) if hasattr(runner, "profile") else est.encoder
        title = f"{check['title']} [{profile}]"
        if not runner.supports(est):
            out.append(CheckOutcome(sc.id, check["id"], title, families.TestResult("visual", np.nan, 1.0, "reject"),
                                    skipped=f"{runner.name} does not support {est.encoder}/{est.decoder}"))
            continue
        if est.id not in est_cache:
            T = int(_mode_value(s["estimators_T"], mode))
            rows, saved = [], []
            for k in range(K):
                f = sc.field(k)
                mem = runner.members(est, f["samples"], f["values"], queries, n_members=T, seed=seed + k)
                point = np.nanmedian(mem, axis=1).reshape(m_grid, m_grid)
                truth = f["truth"].reshape(m_grid, m_grid)
                (th, c), (th_t, c_t) = maps.orientation_coherence(point), maps.orientation_coherence(truth)
                rows.append(((th, c), (th_t, c_t)))
                saved.append(dict(point=point, truth=truth, theta=th, coherence=c, theta_truth=th_t,
                                  coherence_truth=c_t, samples=np.asarray(f["samples"])))
            est_cache[est.id] = rows
            if save_maps:
                from .figures import save_map_fields
                save_map_fields(save_maps, sc.id, est.id, saved, target_theta=s["truth"].get("theta_deg"))
                maps_by_est[est.id] = saved
        rows = est_cache[est.id]
        target = float(s["truth"]["theta_deg"])
        if check["functional"] == "orientation":
            err = [((th - target + 90.0) % 180.0) - 90.0 for (th, _), _ in rows]
            r = families.tost_mean(err, -check["margin_deg"], check["margin_deg"])
        elif check["functional"] == "coherence_ratio":
            ratio = [c / ct for (_, c), (_, ct) in rows]
            r = families.mean_greater(ratio, check["min_ratio"])
        else:
            raise ValueError(f"unknown functional {check['functional']}")
        out.append(CheckOutcome(sc.id, check["id"], title, r, expect=_expect(check, profile)))
    if save_maps and len(maps_by_est) > 1:
        from .figures import save_comparison
        save_comparison(save_maps, sc.id, maps_by_est, target_theta=s["truth"].get("theta_deg"))
    return out


EVALUATORS = {"pair_cooccurrence": eval_pair_cooccurrence, "map_visual": eval_map_visual}


# ----------------------------------------------------------------------------- run
@dataclass
class Report:
    """Result of a :func:`run`.

    Attributes
    ----------
    runner, mode, seed : str, str, int
        The runner's name, the mode (``"ci"`` or ``"full"``) and the seed of the run.
    outcomes : list of CheckOutcome
        One outcome per check, with its p-value, Holm level and decision.
    alpha : float
        Family-wise error rate of the run.
    """
    runner: str
    mode: str
    seed: int
    outcomes: List[CheckOutcome] = field(default_factory=list)
    alpha: float = budget.ALPHA_SUITE

    @property
    def passed(self):
        """Whether every non-skipped check passed."""
        return all(o.passed for o in self.outcomes if not o.skipped)

    def table(self):
        """Plain-text table of the outcomes, with p-values, Holm levels and decisions."""
        w = max([len("scenario/check")] + [len(f"{o.scenario}/{o.check}") for o in self.outcomes])
        lines = [f"runner={self.runner} mode={self.mode} seed={self.seed} α_suite={self.alpha:g} (Holm)",
                 f"{'scenario/check':{w}s}  {'family':12s} {'p-value':>10s} {'level':>9s}  {'expect':6s} result"]
        for o in self.outcomes:
            name = f"{o.scenario}/{o.check}"
            if o.skipped:
                lines.append(f"{name:{w}s}  SKIPPED: {o.skipped}")
                continue
            res = "PASS" if o.passed else "FAIL"
            lines.append(f"{name:{w}s}  {o.result.family:12s} {o.result.p_value:10.3g} {o.level:9.2g}  {o.expect:6s} {res}  — {o.result.detail}")
        return "\n".join(lines)


def run(scenarios, runner: Runner, mode="ci", seed=None, alpha=budget.ALPHA_SUITE, save_maps=None) -> Report:
    """Evaluate scenarios with a runner and decide every check under one Holm budget.

    Parameters
    ----------
    scenarios : Scenario, dict or sequence of Scenario
        What to run, e.g. the output of :func:`catalog`.
    runner : Runner
        The implementation under test (see :class:`~spatialize.scenarios.protocol.Runner`).
    mode : {"ci", "full"}, default "ci"
        ``"ci"``: minimum detectable effect about 0.05, fast. ``"full"``: about 0.02, slow.
    seed : int, optional
        Seed of the run; a fresh one is drawn and recorded in the report when omitted.
    alpha : float, default 1e-3
        Family-wise error rate of the run, controlled with Holm's step-down procedure over all
        the checks evaluated in this call.
    save_maps : str, optional
        Directory where the maps computed by the run are saved for human review: per scenario,
        estimator and field, the arrays (``.npy``) and a figure of truth and map; plus a summary
        figure. Needs matplotlib. Maps never enter the decisions.

    Returns
    -------
    Report
        Outcomes of every check. A check passes when its test does not reject (or rejects, for
        families whose pass rule is to reject), inverted for negative controls.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    seed = int(np.random.SeedSequence().entropy % (2 ** 31)) if seed is None else int(seed)
    if isinstance(scenarios, Scenario):
        scenarios = [scenarios]
    elif isinstance(scenarios, dict):
        scenarios = list(scenarios.values())
    rep = Report(runner.name, mode, seed, alpha=alpha)
    for sc in scenarios:
        rep.outcomes += EVALUATORS[sc.spec["evaluator"]](sc, runner, mode, seed, save_maps=save_maps)
    active = [o for o in rep.outcomes if not o.skipped]
    if active:
        levels, reject = budget.holm_levels([o.result.p_value for o in active], alpha)
        for o, lv, rj in zip(active, levels, reject):
            o.level, o.rejected = float(lv), bool(rj)
            wants_reject = (o.result.pass_if == "reject") != (o.expect == "reject")
            o.passed = o.rejected if wants_reject else not o.rejected
    return rep

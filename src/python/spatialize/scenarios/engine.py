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
        """Tier of the scenario: ``"T1"`` (encoder law), ``"T2"`` (estimator properties) or ``"T3"``
        (geostatistical scenarios)."""
        return self.spec["tier"]

    def file(self, rel):
        """Absolute path of a file of the scenario, given its path relative to the scenario's
        directory (e.g. ``"data/samples_0.npy"``)."""
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
        """Pinned data of replicate field ``k``.

        Parameters
        ----------
        k : int
            Field index, from 0.

        Returns
        -------
        dict of str to ndarray
            ``samples`` and ``values``, plus whichever of ``truth``, ``queries`` and ``exact`` the
            scenario stores (see the scenario file reference).
        """
        d = {}
        for name in ("samples", "values", "truth", "queries", "exact"):
            f = self.file(f"data/{name}_{k}.npy")
            if os.path.exists(f):
                d[name] = np.load(f)
        return d

    def estimator(self, est, encoder_profile=None):
        """The :class:`~spatialize.scenarios.protocol.EstimatorSpec` of an ``estimators`` entry.

        Parameters
        ----------
        est : dict
            One entry of the scenario's ``estimators`` list.

        Returns
        -------
        EstimatorSpec
            With the scenario's domain.
        """
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
    """Outcome of one check in a run.

    Attributes
    ----------
    scenario, check : str
        Scenario identifier and outcome identifier (see :func:`check_ids`).
    title : str
        Check title with the encoder profile.
    result : TestResult
        The statistical result (:mod:`spatialize.scenarios.stats.families`).
    expect : {"pass", "reject"}
        ``"reject"`` marks a negative control, which passes when its test rejects.
    level : float
        Level Holm assigned to the p-value (NaN for almost-sure checks, decided exactly).
    rejected : bool
        Whether the test rejected its null hypothesis.
    passed : bool or None
        The decision, from the family's pass rule and ``expect``.
    skipped : str
        Reason the check was skipped (the runner does not provide the estimator), or empty.
    route : str
        Runner-specific path, e.g. spatialize's ``"facade"`` or ``"run"``.
    known_failure : str
        Reason, when the check records a known defect (like pytest's xfail).
    """
    scenario: str
    check: str
    title: str
    result: families.TestResult
    expect: str = "pass"            # "pass" | "reject" (negative control)
    level: float = float("nan")
    rejected: bool = False
    passed: Optional[bool] = None
    skipped: str = ""
    route: str = ""                 # runner-specific path, e.g. spatialize's "facade" or "run"
    known_failure: str = ""         # reason, when the check documents a known defect (like xfail)

    @property
    def status(self):
        """``PASS``, ``FAIL``, ``SKIPPED``, ``KNOWN`` (a known failure that failed, as recorded) or
        ``XPASS`` (a known failure that passed: the record is out of date)."""
        if self.skipped:
            return "SKIPPED"
        if self.known_failure:
            return "XPASS" if self.passed else "KNOWN"
        return "PASS" if self.passed else "FAIL"

    @property
    def ok(self):
        """Whether the outcome is as expected: PASS, KNOWN or SKIPPED."""
        return self.status in ("PASS", "KNOWN", "SKIPPED")


def check_ids(sc):
    """Identifiers of the outcomes a scenario produces.

    Parameters
    ----------
    sc : Scenario

    Returns
    -------
    list of str
        One identifier per check, except that a check listing several ``estimators`` gives one
        outcome per estimator, ``<check>-<estimator>``.
    """
    ids = []
    for c in sc.spec["checks"]:
        if "estimators" in c:
            ids += [f"{c['id']}-{e}" for e in c["estimators"]]
        else:
            ids.append(c["id"])
    return ids


def _mode_value(v, mode):
    return v[mode] if isinstance(v, dict) and mode in v else v


def _route(runner, est):
    return runner.route(est) if hasattr(runner, "route") else ""


def _expect(check, profile):
    exp = check.get("expect", {})
    return exp.get(profile, exp.get("default", "pass")) if isinstance(exp, dict) else exp


def eval_pair_cooccurrence(sc: Scenario, runner: Runner, mode: str, seed: int,
                           save_maps=None) -> List[CheckOutcome]:
    r"""Evaluator ``pair_cooccurrence``: the partition law read through the estimator (E2).

    With a single datum (value 1) and empty cells as NaN, a member is finite exactly when the query
    shares the datum's cell, so the fraction of finite members at a query estimates the
    co-occurrence :math:`e(\{x, y\})`. It is compared with :math:`\exp(-\lambda \lVert x-y
    \rVert_1)` by :func:`~spatialize.scenarios.stats.families.gof_proportions`.

    Reads ``data.datum``, ``data.directions``, ``data.displacements`` and, per check,
    ``estimator``, ``n_members`` (per mode) and ``expect``.

    Parameters
    ----------
    sc : Scenario
    runner : Runner
    mode : {"ci", "full"}
    seed : int
    save_maps : str, optional
        Ignored (this evaluator draws no maps).

    Returns
    -------
    list of CheckOutcome
        Not yet decided: :func:`~spatialize.scenarios.run` applies the error budget.
    """
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
        out.append(CheckOutcome(sc.id, check["id"], title, r, expect=_expect(check, profile), route=_route(runner, est), known_failure=check.get("known_failure", "")))
    return out


def eval_map_visual(sc: Scenario, runner: Runner, mode: str, seed: int,
                    save_maps=None) -> List[CheckOutcome]:
    """Evaluator ``map_visual``: visual criteria on the median maps of K pinned fields (S03).

    For each estimator and field the median of the members on the scenario's grid is computed once;
    every check then evaluates one map functional on each field and tests it across the fields.

    Functionals (key ``functional`` of a check):

    - ``orientation`` — orientation error against ``truth.theta_deg``, TOST within
      ``±margin_deg`` (V1);
    - ``coherence_ratio`` — coherence of the map over that of the truth, above ``min_ratio`` (V1);
    - ``axis_lock`` — axis-locking of the median map against the estimator on the data rotated by
      ``rotation_deg`` (:func:`~spatialize.scenarios.stats.maps.axis_lock`): below ``max_value``, or
      within ``±margin`` of 0 (V4);
    - ``axis_lock_members`` — the same for single members, mean over ``members_checked`` members per
      field, above ``min_value`` (V4 power check);
    - ``contrast_ratio_reference`` — standard deviation of the map over that of simple kriging with
      the true covariance, above ``min_ratio`` (V5).

    Reads ``data.grid``, ``data.fields``, ``estimators_T``, ``truth`` and, per check, the keys
    above plus ``estimator``, ``expect`` and ``known_failure``.

    Parameters
    ----------
    sc : Scenario
    runner : Runner
    mode : {"ci", "full"}
    seed : int
        Field ``k`` uses ``seed + k``.
    save_maps : str, optional
        Directory where the maps of every field are saved for review
        (:mod:`spatialize.scenarios.figures`).

    Returns
    -------
    list of CheckOutcome
        Not yet decided: :func:`~spatialize.scenarios.run` applies the error budget.
    """
    s = sc.spec
    m_grid = s["data"]["grid"]
    queries = generators.grid(m_grid).astype(np.float32)
    K = int(_mode_value(s["data"]["fields"], mode))
    T = int(_mode_value(s["estimators_T"], mode))
    t = s["truth"]
    est_cache: Dict[str, list] = {}
    maps_by_est: Dict[str, list] = {}
    ref_cache: Dict[int, np.ndarray] = {}
    rot_cache: Dict[str, list] = {}
    n_keep = max([int(c.get("members_checked", 0)) for c in s["checks"]] + [0])

    def point_maps(est, n_members, rotate_deg=0.0, keep_members=0):
        """Median maps (and their readings) of an estimator on every field; with ``rotate_deg`` the
        data and the domain are rotated about the domain centre and the estimator is evaluated at the
        same physical points (so the maps stay comparable pixel by pixel)."""
        rows = []
        for k in range(K):
            f = sc.field(k)
            smp, qry, e = f["samples"], queries, est
            if rotate_deg:
                smp, qry, e = _rotated(est, f["samples"], queries, rotate_deg)
            mem = runner.members(e, smp, f["values"], qry, n_members=n_members, seed=seed + k)
            point = np.nanmedian(mem, axis=1).reshape(m_grid, m_grid)
            truth = f["truth"].reshape(m_grid, m_grid)
            (th, c), (th_t, c_t) = maps.orientation_coherence(point), maps.orientation_coherence(truth)
            rows.append(dict(point=point, truth=truth, theta=th, coherence=c, theta_truth=th_t,
                             coherence_truth=c_t, samples=np.asarray(f["samples"]),
                             members=[np.asarray(mem)[:, j].reshape(m_grid, m_grid) for j in range(keep_members)]))
        return rows

    def reference(k):
        if k not in ref_cache:
            f = sc.field(k)
            ref_cache[k] = generators.fields.simple_kriging_exponential(
                f["samples"], f["values"], queries, t["a1"], t["a2"], t["theta_deg"]).reshape(m_grid, m_grid)
        return ref_cache[k]

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
            est_cache[est.id] = point_maps(est, T, keep_members=n_keep)
            if save_maps:
                from .figures import save_map_fields
                save_map_fields(save_maps, sc.id, est.id, est_cache[est.id], target_theta=t.get("theta_deg"))
                maps_by_est[est.id] = est_cache[est.id]
        rows = est_cache[est.id]
        fn = check["functional"]
        if fn == "orientation":
            target = float(t["theta_deg"])
            err = [((r["theta"] - target + 90.0) % 180.0) - 90.0 for r in rows]
            r = families.tost_mean(err, -check["margin_deg"], check["margin_deg"])
        elif fn == "coherence_ratio":
            r = families.mean_greater([r["coherence"] / r["coherence_truth"] for r in rows], check["min_ratio"])
        elif fn in ("axis_lock", "axis_lock_members"):
            key = f"{est.id}@{check['rotation_deg']}"
            if key not in rot_cache:
                rot_cache[key] = point_maps(est, T, rotate_deg=float(check["rotation_deg"]), keep_members=n_keep)
            rot = rot_cache[key]
            if fn == "axis_lock":
                lock = [maps.axis_lock(a["point"], b["point"]) for a, b in zip(rows, rot)]
            else:
                j = int(check["members_checked"])
                lock = [float(np.mean([maps.axis_lock(a["members"][i], b["members"][i]) for i in range(j)]))
                        for a, b in zip(rows, rot)]
            if "max_value" in check:
                r = families.mean_less(lock, check["max_value"])
            elif "min_value" in check:
                r = families.mean_greater(lock, check["min_value"])
            else:
                r = families.tost_mean(lock, -check["margin"], check["margin"])
        elif fn == "contrast_ratio_reference":
            ratio = [float(np.nanstd(r["point"]) / np.std(reference(k))) for k, r in enumerate(rows)]
            r = families.mean_greater(ratio, check["min_ratio"])
        else:
            raise ValueError(f"unknown functional {fn}")
        out.append(CheckOutcome(sc.id, check["id"], title, r, expect=_expect(check, profile), route=_route(runner, est), known_failure=check.get("known_failure", "")))
    if save_maps and len(maps_by_est) > 1:
        from .figures import save_comparison
        save_comparison(save_maps, sc.id, maps_by_est, target_theta=t.get("theta_deg"))
    return out


def _rotated(est, samples, queries, deg):
    """Data, queries and estimator rotated by ``deg`` about the centre of the estimator's domain.

    The new domain is the bounding box of the rotated one; rates are per unit length or volume, so
    the partition law inside is unchanged.
    """
    import dataclasses
    import itertools
    box = np.asarray(est.domain, float)
    c = box.mean(axis=1)
    t = np.radians(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    def rot(p):
        return ((np.asarray(p, float) - c) @ R.T + c).astype(np.float32)
    corners = rot(np.array(list(itertools.product(*est.domain))))
    domain = tuple((float(corners[:, i].min()), float(corners[:, i].max())) for i in range(corners.shape[1]))
    return rot(samples), rot(queries), dataclasses.replace(est, domain=domain)


def eval_edge_cases(sc: Scenario, runner: Runner, mode: str, seed: int,
                    save_maps=None) -> List[CheckOutcome]:
    """Evaluator ``edge_cases``: almost-sure checks on degenerate designs (S12).

    Each check lists ``estimators`` and produces one outcome per estimator. Kinds (key ``kind``):

    - ``runs`` — no exception, members of shape (queries, members);
    - ``finite`` — no infinite value (NaN allowed for empty cells);
    - ``convex`` — finite members within the data range, up to ``tolerance`` × range;
    - ``exact`` — at the queries placed on non-duplicated data, every member equals the datum, up
      to ``tolerance`` × range.

    An exception raised by the runner counts as a violation of every check of that estimator, so a
    crash is reported, not propagated. Reads ``data.fields``, ``estimators_T`` and the fields'
    ``queries`` and ``exact`` arrays.

    Parameters
    ----------
    sc : Scenario
    runner : Runner
    mode : {"ci", "full"}
    seed : int
        Field ``k`` uses ``seed + k``.
    save_maps : str, optional
        Ignored.

    Returns
    -------
    list of CheckOutcome
    """
    s = sc.spec
    K = int(_mode_value(s["data"]["fields"], mode))
    T = int(_mode_value(s["estimators_T"], mode))
    ests = {e["id"]: sc.estimator(e) for e in s["estimators"]}
    cache: Dict[str, list] = {}

    def members(eid):
        if eid not in cache:
            res = []
            for k in range(K):
                f = sc.field(k)
                try:
                    m = np.asarray(runner.members(ests[eid], f["samples"], f["values"], f["queries"],
                                                  n_members=T, seed=seed + k))
                    res.append((f, m, None))
                except Exception as e:  # a crash of the implementation is a finding, not an abort
                    res.append((f, None, f"{type(e).__name__}: {e}"))
            cache[eid] = res
        return cache[eid]

    out = []
    for check in s["checks"]:
        kind, rel_tol = check["kind"], float(check.get("tolerance", 0.0))
        for eid in check["estimators"]:
            est = ests[eid]
            profile = runner.profile(est.encoder) if hasattr(runner, "profile") else est.encoder
            cid, title = f"{check['id']}-{eid}", f"{check['title']} ({eid}) [{profile}]"
            if not runner.supports(est):
                out.append(CheckOutcome(sc.id, cid, title, families.TestResult("almost-sure", np.nan, 1.0, "not_reject"),
                                        skipped=f"{runner.name} does not support {est.encoder}/{est.decoder}"))
                continue
            violations, checked, notes = 0, 0, []
            for f, m, err in members(eid):
                if m is None:
                    violations, checked = violations + 1, checked + 1
                    notes.append(err)
                    continue
                vmin, vmax = float(f["values"].min()), float(f["values"].max())
                tol = rel_tol * (vmax - vmin)
                if kind == "runs":
                    checked += 1
                    violations += int(m.shape != (len(f["queries"]), T))
                elif kind == "finite":
                    checked += m.size
                    violations += int(np.isinf(m).sum())
                elif kind == "convex":
                    fin = m[np.isfinite(m)]
                    checked += fin.size
                    violations += int(((fin < vmin - tol) | (fin > vmax + tol)).sum())
                elif kind == "exact":
                    idx, target = f["exact"][:, 0].astype(int), f["exact"][:, 1]
                    sub = m[idx]
                    bad = ~np.isfinite(sub) | (np.abs(sub - target[:, None]) > tol)
                    checked += sub.size
                    violations += int(bad.sum())
                else:
                    raise ValueError(f"unknown check kind {kind}")
            r = families.almost_sure(violations, checked)
            if notes:
                r = families.TestResult(r.family, r.statistic, r.p_value, r.pass_if, f"{r.detail}; {notes[0]}")
            out.append(CheckOutcome(sc.id, cid, title, r, expect=_expect(check, profile), route=_route(runner, est), known_failure=check.get("known_failure", "")))
    return out


EVALUATORS = {"pair_cooccurrence": eval_pair_cooccurrence, "map_visual": eval_map_visual,
              "edge_cases": eval_edge_cases}


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
        """Whether every outcome is as expected (passed, or a known failure that failed)."""
        return all(o.ok for o in self.outcomes)

    def table(self):
        """Plain-text table of the outcomes, with p-values, Holm levels and decisions."""
        names = [f"{o.scenario}/{o.check}" + (f" [{o.route}]" if o.route and o.route != "facade" else "")
                 for o in self.outcomes]
        w = max([len("scenario/check")] + [len(n) for n in names])
        lines = [f"runner={self.runner} mode={self.mode} seed={self.seed} α_suite={self.alpha:g} (Holm)",
                 f"{'scenario/check':{w}s}  {'family':12s} {'p-value':>10s} {'level':>9s}  {'expect':6s} result"]
        for o, name in zip(self.outcomes, names):
            if o.skipped:
                lines.append(f"{name:{w}s}  SKIPPED: {o.skipped}")
                continue
            res = o.status
            level = "exact" if o.result.family == "almost-sure" else f"{o.level:9.2g}"
            lines.append(f"{name:{w}s}  {o.result.family:12s} {o.result.p_value:10.3g} {level:>9s}  {o.expect:6s} {res}  — {o.result.detail}")
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
    # Almost-sure checks are decided exactly (a correct implementation cannot violate them), so they
    # spend none of the error budget: Holm applies to the statistical tests only.
    for o in rep.outcomes:
        if not o.skipped and o.result.family == "almost-sure":
            o.rejected = o.result.p_value == 0.0
            wants_reject = (o.result.pass_if == "reject") != (o.expect == "reject")
            o.passed = o.rejected if wants_reject else not o.rejected
    active = [o for o in rep.outcomes if not o.skipped and o.result.family != "almost-sure"]
    if active:
        levels, reject = budget.holm_levels([o.result.p_value for o in active], alpha)
        for o, lv, rj in zip(active, levels, reject):
            o.level, o.rejected = float(lv), bool(rj)
            wants_reject = (o.result.pass_if == "reject") != (o.expect == "reject")
            o.passed = o.rejected if wants_reject else not o.rejected
    return rep

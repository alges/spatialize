"""Scenario loading, evaluation and reporting.

A run evaluates one or more scenarios with a :class:`~spatialize.scenarios.protocol.Runner`,
collects every check's p-value, applies Holm's procedure to the whole run (α_suite = 1e-3) and
decides each check with its family's pass rule (see ``stats.families``). Checks may carry an
``expect`` per encoder profile: ``reject`` marks a *negative control* — the run is correct when the
test rejects (it proves the test has the power to see a known deviation).
"""
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .protocol import EstimatorSpec, Runner
from .stats import budget, families, maps
from . import generators

CATALOG = os.path.join(os.path.dirname(__file__), "catalog")
MODES = ("ci", "full")

# Progress of a run: a callable taking one line of text, set by run() for its duration.
_progress = None


def _say(msg):
    if _progress is not None:
        _progress(msg)


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
                             empty_cells=est.get("empty_cells", "nan"), mark=dict(est.get("mark", {})))


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

    def __post_init__(self):
        if self.skipped:
            _say(f"  {self.check}: skipped ({self.skipped})")
        else:
            how = "exact" if self.result.family == "almost-sure" else "Holm level set at the end"
            neg = ", negative control" if self.expect == "reject" else ""
            _say(f"  {self.check}: p = {self.result.p_value:.3g}  ({self.result.family}{neg}; {how})")

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
        """True when the outcome is as expected (PASS, KNOWN or SKIPPED)."""
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
    ``estimator`` or ``estimators``, ``n_members`` (per mode) and ``expect``.

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
    q, l1 = [], []
    for d in s["data"]["directions"]:
        u = np.asarray(d, float); u = u / np.abs(u).sum()
        for h in hs:
            q.append(centre + h * u); l1.append(h)
    q = np.asarray(q, np.float32)
    out = []
    for check in s["checks"]:
        eids = check["estimators"] if "estimators" in check else [check["estimator"]]
        for eid in eids:
            est = sc.estimator(next(e for e in s["estimators"] if e["id"] == eid))
            profile = runner.profile(est.encoder) if hasattr(runner, "profile") else est.encoder
            if "estimators" in check:
                cid, title = f"{check['id']}-{eid}", f"{check['title']} ({eid}) [{profile}]"
            else:
                cid, title = check["id"], f"{check['title']} [{profile}]"
            if not runner.supports(est):
                out.append(CheckOutcome(sc.id, cid, title, families.TestResult("gof-closed", np.nan, 1.0, "not_reject"),
                                        skipped=f"{runner.name} does not support {est.encoder}/{est.decoder}"))
                continue
            n = int(_mode_value(check["n_members"], mode))
            m = runner.members(est, centre[None, :].astype(np.float32), np.ones(1, np.float32), q, n_members=n, seed=seed)
            p_hat = np.isfinite(m).mean(axis=1)
            p0 = np.exp(-est.rate * np.asarray(l1))
            r = families.gof_proportions(p_hat, p0, n)
            out.append(CheckOutcome(sc.id, cid, title, r, expect=_expect(check, profile), route=_route(runner, est), known_failure=check.get("known_failure", "")))
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
      the true covariance, above ``min_ratio`` (V5);
    - ``paired_relation`` — the estimator beats ``against`` field by field on ``metric``, either
      ``coverage`` (share of the grid inside the members' 90 % interval) or ``rmse`` (error of the
      median map), in the direction ``better`` (paired t-test).

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
        """Median maps (and their readings) of an estimator on every field. With ``rotate_deg`` the
        data and the domain are rotated about the domain centre, the estimator being evaluated at the
        same physical points so that the maps stay comparable pixel by pixel."""
        rows = []
        what = f"{est.id}" + (f" rotated {rotate_deg:g}°" if rotate_deg else "")
        _say(f"  {what}: {K} fields, {n_members} members each")
        t0 = last = time.monotonic()
        for k in range(K):
            now = time.monotonic()
            if k and now - last >= 15.0:
                _say(f"    {what}: field {k}/{K}, {now - t0:.0f} s")
                last = now
            f = sc.field(k)
            smp, qry, e = f["samples"], queries, est
            if rotate_deg:
                smp, qry, e = _rotated(est, f["samples"], queries, rotate_deg)
            mem = np.asarray(runner.members(e, smp, f["values"], qry, n_members=n_members, seed=seed + k), float)
            point = np.nanmedian(mem, axis=1).reshape(m_grid, m_grid)
            truth = f["truth"].reshape(m_grid, m_grid)
            # coverage of the 90 % interval read from the members, and error of the median map
            lo90, hi90 = np.nanquantile(mem, 0.05, axis=1), np.nanquantile(mem, 0.95, axis=1)
            tv = np.asarray(f["truth"], float).ravel()
            ok = np.isfinite(lo90)
            coverage = float(np.mean((tv[ok] >= lo90[ok]) & (tv[ok] <= hi90[ok])))
            rmse = float(np.sqrt(np.nanmean((point.ravel() - tv) ** 2)))
            (th, c), (th_t, c_t) = maps.orientation_coherence(point), maps.orientation_coherence(truth)
            rows.append(dict(point=point, truth=truth, theta=th, coherence=c, theta_truth=th_t,
                             coherence_truth=c_t, samples=np.asarray(f["samples"]), coverage=coverage, rmse=rmse,
                             members=[np.asarray(mem)[:, j].reshape(m_grid, m_grid) for j in range(keep_members)]))
        _say(f"    {what}: done in {time.monotonic() - t0:.0f} s")
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
        elif fn == "paired_relation":
            other = sc.estimator(next(e for e in s["estimators"] if e["id"] == check["against"]))
            if other.id not in est_cache:
                est_cache[other.id] = point_maps(other, T, keep_members=n_keep)
            metric = check["metric"]
            r = families.paired_relation([row[metric] for row in rows], [row[metric] for row in est_cache[other.id]],
                                         better=check["better"])
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


def _draw_laws_data(s):
    """Data and queries of a draw_laws scenario, drawn from ``data.generator_seed`` (not pinned)."""
    rng = np.random.default_rng(int(s["data"]["generator_seed"]))
    box = np.asarray(s["domain"]["box"], float)
    n, m = int(s["data"]["n"]), int(s["data"]["queries"])
    pts = box[:, 0] + rng.random((n, len(box))) * (box[:, 1] - box[:, 0])
    qry = box[:, 0] + rng.random((m, len(box))) * (box[:, 1] - box[:, 0])
    u = (pts - box[:, 0]) / (box[:, 1] - box[:, 0])
    # a smooth field plus noise: distinct values, so a member identifies the datum drawn
    vals = np.sin(2 * np.pi * u[:, 0]) + 0.5 * np.cos(2 * np.pi * u[:, -1]) + 0.3 * rng.standard_normal(n)
    return pts.astype(np.float32), vals.astype(np.float32), qry.astype(np.float32)


def eval_draw_laws(sc: Scenario, runner: Runner, mode: str, seed: int,
                   save_maps=None) -> List[CheckOutcome]:
    r"""Evaluator ``draw_laws``: the laws of decoders that return a draw (P2, P3).

    A draw decoder is compared with its *reference*, the decoder averaging over the same weights,
    computed with the same seed and hence, for a runner that shares partitions between estimators
    with one seed, on the same partitions. Kinds (key ``kind``):

    - ``support`` (``estimators``) — almost-sure: every finite member is one of the data values;
    - ``mean`` (``estimator``, ``reference``) — identity: at each query the mean over the members of
      draw − reference is 0, standardised by its own standard error;
    - ``variance`` (same keys) — identity: the mean of (draw − reference)² equals the mean of the
      weighted dispersion, read as reference(z²) − reference(z)²;
    - ``frequencies`` (same keys, the reference being the cell mean) — goodness of fit: the number
      of times each datum is drawn at a query against its expected number, the sum over the members
      of the reference run on the datum's indicator (its probability under each partition). The
      Pearson statistic is referred to :math:`\chi^2` with :math:`k - q` degrees of freedom (k data
      with positive expectation, q queries), which is conservative when the probabilities vary
      from one partition to another.

    Reads ``data.n``, ``data.queries``, ``data.generator_seed`` and ``estimators_T``.
    """
    s = sc.spec
    T = int(_mode_value(s["estimators_T"], mode))
    ests = {e["id"]: sc.estimator(e) for e in s["estimators"]}
    pts, vals, qry = _draw_laws_data(s)
    cache = {}

    def members(eid, values=None, key=""):
        if (eid, key) not in cache:
            v = vals if values is None else np.asarray(values, np.float32)
            cache[(eid, key)] = np.asarray(runner.members(ests[eid], pts, v, qry, n_members=T, seed=seed), float)
        return cache[(eid, key)]

    def z_of(x):
        n = np.sum(np.isfinite(x), axis=1)
        return np.nanmean(x, axis=1) / (np.nanstd(x, axis=1, ddof=1) / np.sqrt(n))

    out = []
    for check in s["checks"]:
        kind = check["kind"]
        eids = check["estimators"] if "estimators" in check else [check["estimator"]]
        for eid in eids:
            est = ests[eid]
            profile = runner.profile(est.encoder) if hasattr(runner, "profile") else est.encoder
            if "estimators" in check:
                cid, title = f"{check['id']}-{eid}", f"{check['title']} ({eid}) [{profile}]"
            else:
                cid, title = check["id"], f"{check['title']} [{profile}]"
            needed = [est] + ([ests[check["reference"]]] if "reference" in check else [])
            if not all(runner.supports(e) for e in needed):
                out.append(CheckOutcome(sc.id, cid, title, families.TestResult(check["family"], np.nan, 1.0, "not_reject"),
                                        skipped=f"{runner.name} does not support "
                                                + ", ".join(f"{e.encoder}/{e.decoder}" for e in needed)))
                continue
            m = members(eid)
            if kind == "support":
                fin = m[np.isfinite(m)].astype(np.float32)
                r = families.almost_sure(int((~np.isin(fin, vals)).sum()), fin.size)
            elif kind == "mean":
                r = families.identity(z_of(m - members(check["reference"])), "mean(draw − reference)")
            elif kind == "variance":
                ref = members(check["reference"])
                ref2 = members(check["reference"], vals.astype(float) ** 2, "squares")
                r = families.identity(z_of((m - ref) ** 2 - (ref2 - ref ** 2)), "variance − dispersion")
            elif kind == "frequencies":
                counts, expected, nq = [], [], 0
                probs = [members(check["reference"], (np.arange(len(vals)) == j).astype(float), f"onehot{j}")
                         for j in range(len(vals))]
                for qi in range(len(qry)):
                    e_q = np.array([np.nansum(p[qi]) for p in probs])
                    c_q = np.array([np.sum(m[qi].astype(np.float32) == vals[j]) for j in range(len(vals))])
                    keep = e_q > 0
                    if keep.sum() > 1:
                        counts += list(c_q[keep]); expected += list(e_q[keep]); nq += 1
                r = families.gof_counts(counts, expected, df=len(counts) - nq)
            else:
                raise ValueError(f"unknown check kind {kind}")
            out.append(CheckOutcome(sc.id, cid, title, r, expect=_expect(check, profile), route=_route(runner, est),
                                    known_failure=check.get("known_failure", "")))
    return out


def _same_cell(runner, est, points, n, seed):
    """Which of ``points`` share a cell, partition by partition, read through the estimator.

    The cell mean run on the indicator of point ``a`` is positive exactly at the points of a's cell,
    and every run with one seed sees the same partitions.

    Returns
    -------
    ndarray of bool, shape (n, k, k)
        ``[t, a, j]`` is true when points a and j share a cell of partition t.
    """
    points = np.asarray(points, np.float32)
    k = len(points)
    out = np.zeros((n, k, k), bool)
    for a in range(k):
        values = (np.arange(k) == a).astype(np.float32)
        m = np.asarray(runner.members(est, points, values, points, n_members=n, seed=seed))  # (k, n)
        out[:, a, :] = (m > 0).T
    return out


def _groupings(same):
    """The set partition of the points under each partition, as a tuple of block labels."""
    labels = []
    for t in range(same.shape[0]):
        lab = []
        for j in range(same.shape[1]):
            lab.append(next(a for a in range(same.shape[1]) if same[t, a, j]))
        labels.append(tuple(lab))
    return labels


def _line_grouping_law(points, rate):
    """Law of the groupings of sorted points on a line under Poisson cuts of rate ``rate``: each gap
    is cut independently with probability 1 - exp(-rate * gap). Returns {labels: probability}."""
    x = np.sort(np.asarray(points, float).ravel())
    keep = np.exp(-rate * np.diff(x))
    law = {}
    for cuts in np.ndindex(*(2,) * len(keep)):
        p = np.prod([1 - keep[g] if c else keep[g] for g, c in enumerate(cuts)])
        lab, block = [0], 0
        for g, c in enumerate(cuts):
            if c:
                block = g + 1
            lab.append(block)
        law[tuple(lab)] = law.get(tuple(lab), 0.0) + float(p)
    return law


def eval_partition_law(sc: Scenario, runner: Runner, mode: str, seed: int,
                       save_maps=None) -> List[CheckOutcome]:
    r"""Evaluator ``partition_law``: the law of the partition of a few points (E1, E3, E5).

    Which points share a cell is read through the estimator (see :func:`_same_cell`), so the
    scenario's estimators use the ``cellmean`` decoder. Kinds (key ``kind``):

    - ``line_groupings`` (``data.points`` on a line) — goodness of fit of the frequencies of the
      interval groupings to the law of Poisson cuts of rate ``rate``, each gap :math:`g` cut with
      probability :math:`1 - e^{-\lambda g}`;
    - ``interval_only`` (same data) — almost-sure: no grouping other than intervals of consecutive
      points occurs;
    - ``set_cooccurrence`` (``data.sets``) — goodness of fit of the share of partitions putting a set
      in one cell to :math:`\exp(-\lambda \sum_c \mathrm{range}_c(S))`;
    - ``fourth_cumulant`` (``data.spacings``, four points on a line) — identity: the fourth joint
      cumulant of a block-mark field with standard Gaussian marks, drawn by the evaluator on the
      estimator's partitions, equals :math:`2q^3(1-q)`, :math:`q = e^{-\lambda s}`, with a bootstrap
      standard error;
    - ``conditional_independence`` (``data.points``, three points :math:`x < b < y` on a line, ``x``
      and ``y`` at the ends of the domain) — identity: :math:`D = P(x \sim y) - P(x \sim b)\,
      P(b \sim y)`, the conditional covariance :math:`\mathrm{Cov}(Z_x, Z_y \mid Z_b = \beta)` of a
      block-mark field over :math:`\beta^2`, equals ``target``: ``poisson`` (0, cuts independent on
      disjoint intervals) or ``forced_cut`` (:math:`-(b - x)(y - b)\, e^{-\lambda (y - x)}`, one
      uniform cut of the whole domain added to the Poisson cuts), with a bootstrap standard error;
    - ``voronoi_line`` (``data.n`` filler data, ``data.origin``, ``data.distances``, on a line) —
      goodness of fit of the share of partitions putting the origin and the origin plus each
      distance in one cell to the Poisson–Voronoi closed form :math:`(1 + \rho h)\, e^{-2\rho h}`,
      :math:`\rho` the rate of the check (``rho``);
    - ``isotropy`` (``data.n`` filler data, ``data.centre``, ``data.distances``, in the plane) —
      identity: at each distance the share of partitions putting the centre and a point along the
      first axis in one cell equals the share for a point along the diagonal, paired by partition.

    The kinds ``voronoi_line`` and ``isotropy`` run each distance with its own seed, so that their
    statistics are independent. The kinds ``voronoi_line`` and ``isotropy`` read the cells through the runner's optional method
    ``cells``, with ``data.n`` data uniform in the domain, so that a Voronoi partition can reach its
    intensity. Reads ``n_members`` (per mode) of each check.
    """
    s = sc.spec
    out = []
    for check in s["checks"]:
        kind, n = check["kind"], int(_mode_value(check["n_members"], mode))
        for eid in check["estimators"]:
            est = sc.estimator(next(e for e in s["estimators"] if e["id"] == eid))
            profile = runner.profile(est.encoder) if hasattr(runner, "profile") else est.encoder
            cid, title = f"{check['id']}-{eid}", f"{check['title']} ({eid}) [{profile}]"
            if not runner.supports(est):
                out.append(CheckOutcome(sc.id, cid, title, families.TestResult(check["family"], np.nan, 1.0, "not_reject"),
                                        skipped=f"{runner.name} does not support {est.encoder}/{est.decoder}"))
                continue
            lam = float(est.rate)
            if kind in ("line_groupings", "interval_only"):
                pts = np.asarray(s["data"]["points"], float)
                order = np.argsort(pts[:, 0])
                labels = _groupings(_same_cell(runner, est, pts[order], n, seed))
                law = _line_grouping_law(pts[:, 0], lam)
                observed = {}  # groupings labelled, as the law, by the first point of each block
                for lab in labels:
                    observed[lab] = observed.get(lab, 0) + 1
                if kind == "interval_only":
                    bad = sum(c for key, c in observed.items() if key not in law)
                    r = families.almost_sure(bad, n)
                else:
                    keys = sorted(law)
                    r = families.gof_counts([observed.get(k_, 0) for k_ in keys], [n * law[k_] for k_ in keys],
                                            df=len(keys) - 1)
            elif kind == "set_cooccurrence":
                sets = [np.asarray(S, float) for S in s["data"]["sets"]]
                p_hat, p0 = [], []
                for S in sets:
                    same = _same_cell(runner, est, S, n, seed)
                    p_hat.append(float(np.mean(same[:, 0, :].all(axis=1))))
                    p0.append(float(np.exp(-lam * np.sum(S.max(axis=0) - S.min(axis=0)))))
                r = families.gof_proportions(np.array(p_hat), np.array(p0), n)
            elif kind == "fourth_cumulant":
                rng = np.random.default_rng(seed)
                z = []
                origin = float(s["data"].get("origin", 0.0))
                for sp in s["data"]["spacings"]:
                    pts = origin + sp * np.arange(4, dtype=float)[:, None]
                    labels = _groupings(_same_cell(runner, est, pts, n, seed))
                    Z = np.empty((n, 4))
                    for t, lab in enumerate(labels):
                        marks = rng.standard_normal(4)
                        Z[t] = marks[list(lab)]
                    def k4(W):
                        m = lambda a, b: np.mean(W[:, a] * W[:, b])
                        return (np.mean(W.prod(axis=1)) - m(0, 1) * m(2, 3) - m(0, 2) * m(1, 3) - m(0, 3) * m(1, 2))
                    boot = [k4(Z[rng.integers(0, n, n)]) for _ in range(200)]
                    q = np.exp(-lam * sp)
                    z.append((k4(Z) - 2 * q ** 3 * (1 - q)) / np.std(boot, ddof=1))
                r = families.identity(z, "κ4 − 2q³(1−q)")
            elif kind == "conditional_independence":
                pts = np.asarray(s["data"]["points"], float)
                same = _same_cell(runner, est, pts, n, seed)
                xy, xb, by = same[:, 0, 2].astype(float), same[:, 0, 1].astype(float), same[:, 1, 2].astype(float)
                d_of = lambda i: xy[i].mean() - xb[i].mean() * by[i].mean()
                rng = np.random.default_rng(seed)
                boot = [d_of(rng.integers(0, n, n)) for _ in range(500)]
                x, b, y = pts[:, 0]
                target = 0.0 if check["target"] == "poisson" else -(b - x) * (y - b) * np.exp(-lam * (y - x))
                r = families.identity([(d_of(slice(None)) - target) / np.std(boot, ddof=1)],
                                      f"(D − {target:.4f})/se, D={d_of(slice(None)):+.4f}")
            elif kind in ("voronoi_line", "isotropy"):
                if not hasattr(runner, "cells"):
                    out.append(CheckOutcome(sc.id, cid, title, families.TestResult(check["family"], np.nan, 1.0, "not_reject"),
                                            skipped=f"{runner.name} does not give the cells of its partitions"))
                    continue
                d = s["data"]
                box = np.asarray(s["domain"]["box"], float)
                rng = np.random.default_rng(int(d["generator_seed"]))
                filler = (box[:, 0] + rng.random((int(d["n"]), len(box))) * (box[:, 1] - box[:, 0])).astype(np.float32)
                dist = np.asarray(d["distances"], float)
                # one run per distance, each with its own seed: independent proportions
                if kind == "voronoi_line":
                    o = float(d["origin"])
                    p_hat = []
                    for i, h in enumerate(dist):
                        q = np.array([[o], [o + h]], np.float32)
                        lab = np.asarray(runner.cells(est, filler, q, n_members=n, seed=_sub_seed(seed, i)))
                        p_hat.append(float(np.mean(lab[1] == lab[0])))
                    u = float(check["rho"]) * dist
                    r = families.gof_proportions(np.array(p_hat), (1 + u) * np.exp(-2 * u), n)
                else:
                    c = np.asarray(d["centre"], float)
                    z = []
                    for i, h in enumerate(dist):
                        q = np.vstack([c, c + [h, 0.0], c + np.array([h, h]) / np.sqrt(2)]).astype(np.float32)
                        lab = np.asarray(runner.cells(est, filler, q, n_members=n, seed=_sub_seed(seed, i)))
                        diff = (lab[1] == lab[0]).astype(float) - (lab[2] == lab[0]).astype(float)
                        z.append(diff.mean() / (diff.std(ddof=1) / np.sqrt(n)))
                    r = families.identity(z, "(axis − diagonal)/se")
            else:
                raise ValueError(f"unknown check kind {kind}")
            out.append(CheckOutcome(sc.id, cid, title, r, expect=_expect(check, profile), route=_route(runner, est),
                                    known_failure=check.get("known_failure", "")))
    return out


def eval_locality(sc: Scenario, runner: Runner, mode: str, seed: int,
                  save_maps=None) -> List[CheckOutcome]:
    r"""Evaluator ``locality``: the law at a location must not depend on the other queries (P6).

    The members at ``data.location`` are computed twice with one seed, together with two different
    sets of other queries, and compared by a two-sample Kolmogorov–Smirnov test. Kinds (key
    ``kind``): ``inside``, the other queries drawn uniformly in the domain, and ``beyond``, drawn in
    the domain enlarged by ``data.beyond`` on every side, which moves the box of an implementation
    that draws its partitions on the box of data and queries. That move changes the law at the
    location only for a partition that is not consistent under restriction, or whose rate depends
    on the box.

    Reads ``data.n``, ``data.location``, ``data.few``, ``data.many``, ``data.beyond``,
    ``data.generator_seed`` and ``estimators_T``.
    """
    s = sc.spec
    T = int(_mode_value(s["estimators_T"], mode))
    rng = np.random.default_rng(int(s["data"]["generator_seed"]))
    box = np.asarray(s["domain"]["box"], float)
    lo, hi = box[:, 0], box[:, 1]
    n, d = int(s["data"]["n"]), len(box)
    pts = (lo + rng.random((n, d)) * (hi - lo)).astype(np.float32)
    u = (pts - lo) / (hi - lo)
    vals = (np.sin(2 * np.pi * u[:, 0]) + 0.5 * np.cos(2 * np.pi * u[:, -1]) + 0.3 * rng.standard_normal(n)).astype(np.float32)
    v = np.asarray(s["data"]["location"], np.float32)[None, :]
    few = lo + rng.random((int(s["data"]["few"]), d)) * (hi - lo)
    many_in = lo + rng.random((int(s["data"]["many"]), d)) * (hi - lo)
    b = float(s["data"]["beyond"])
    many_out = (lo - b) + rng.random((int(s["data"]["many"]), d)) * (hi - lo + 2 * b)
    out = []
    for check in s["checks"]:
        for eid in check["estimators"]:
            est = sc.estimator(next(e for e in s["estimators"] if e["id"] == eid))
            profile = runner.profile(est.encoder) if hasattr(runner, "profile") else est.encoder
            cid, title = f"{check['id']}-{eid}", f"{check['title']} ({eid}) [{profile}]"
            if not runner.supports(est):
                out.append(CheckOutcome(sc.id, cid, title, families.TestResult("two-sample", np.nan, 1.0, "not_reject"),
                                        skipped=f"{runner.name} does not support {est.encoder}/{est.decoder}"))
                continue
            other = many_in if check["kind"] == "inside" else many_out
            a_ = np.asarray(runner.members(est, pts, vals, np.vstack([v, few]).astype(np.float32), n_members=T, seed=seed))[0]
            b_ = np.asarray(runner.members(est, pts, vals, np.vstack([v, other]).astype(np.float32), n_members=T, seed=seed))[0]
            r = families.two_sample_ks(a_[np.isfinite(a_)], b_[np.isfinite(b_)])
            out.append(CheckOutcome(sc.id, cid, title, r, expect=_expect(check, profile), route=_route(runner, est),
                                    known_failure=check.get("known_failure", "")))
    return out


def _empty_cells_data(s):
    """Data and queries of an empty_cells scenario, drawn from ``data.generator_seed`` (not pinned):
    ``data.n`` data uniform in ``data.box``, a part of the domain, and a ``data.grid`` × ... grid of
    cell centres over the domain as queries."""
    rng = np.random.default_rng(int(s["data"]["generator_seed"]))
    box = np.asarray(s["domain"]["box"], float)
    sub = np.asarray(s["data"]["box"], float)
    n, g = int(s["data"]["n"]), int(s["data"]["grid"])
    pts = sub[:, 0] + rng.random((n, len(box))) * (sub[:, 1] - sub[:, 0])
    u = (pts - box[:, 0]) / (box[:, 1] - box[:, 0])
    # a smooth field plus noise: distinct values, so a value identifies the datum it came from
    vals = np.sin(2 * np.pi * u[:, 0]) + 0.5 * np.cos(2 * np.pi * u[:, -1]) + 0.3 * rng.standard_normal(n)
    axes = [lo + (np.arange(g) + 0.5) * (hi - lo) / g for lo, hi in box]
    qry = np.stack([a.ravel() for a in np.meshgrid(*axes, indexing="ij")], axis=1)
    return pts.astype(np.float32), vals.astype(np.float32), qry.astype(np.float32)


#: Kinds of the evaluator ``empty_cells`` that read the cells of the partitions (``runner.cells``).
_NEEDS_CELLS = ("nan_iff_empty", "one_per_cell", "cell_mean")


def eval_empty_cells(sc: Scenario, runner: Runner, mode: str, seed: int,
                     save_maps=None) -> List[CheckOutcome]:
    r"""Evaluator ``empty_cells``: properties of the empty-cell policies (P10).

    Every estimator is run with one seed, so a runner that shares partitions between estimators with
    one seed computes them all on the same partitions. A policy estimator names, with the key
    ``reference`` of its entry, the estimator with the same encoder and decoder under ``"nan"``,
    whose NaN members mark the queries of empty cells. Kinds (key ``kind``), one outcome per
    estimator of ``estimators``:

    - ``nan_iff_empty`` — under ``"nan"``, a member is NaN exactly when no datum lies in the
      query's cell (needs ``runner.cells``);
    - ``unchanged`` — where the reference is finite, the member equals the reference's, up to
      ``tolerance`` × data range;
    - ``filled`` — no member is NaN;
    - ``one_per_cell`` — in each partition, the queries of one empty cell share one value (needs
      ``runner.cells``);
    - ``observed`` — every member at a query of an empty cell is one of the data values;
    - ``cell_mean`` — every member at a query of an empty cell is the mean of the data of a cell of
      the same partition, up to ``tolerance`` × data range (needs ``runner.cells``; for the
      ``cellmean`` decoder);
    - ``data_law`` — goodness of fit: at the query nearest to the check's ``location``, the members
      of the partitions where its cell is empty are drawn uniformly among the data. The Pearson
      statistic of the count of each datum against its expectation (the number of such members over
      n) is referred to :math:`\chi^2_{n-1}`. One query keeps the members independent, one
      partition each.

    The data lie in a part of the domain (``data.box``), so that the partitions have empty cells.
    Reads ``data.n``, ``data.box``, ``data.grid``, ``data.generator_seed`` and ``estimators_T``.
    """
    s = sc.spec
    T = int(_mode_value(s["estimators_T"], mode))
    entries = {e["id"]: e for e in s["estimators"]}
    ests = {eid: sc.estimator(e) for eid, e in entries.items()}
    pts, vals, qry = _empty_cells_data(s)
    rng_v = float(vals.max() - vals.min())
    cache = {}

    def members(eid):
        if eid not in cache:
            cache[eid] = np.asarray(runner.members(ests[eid], pts, vals, qry, n_members=T, seed=seed), float)
        return cache[eid]

    def labels(eid):
        """Cell labels of the queries and of the data, (q, T) and (n, T)."""
        key = ("cells", ests[eid].encoder, ests[eid].rate)
        if key not in cache:
            lab = np.asarray(runner.cells(ests[eid], pts, np.vstack([qry, pts]), n_members=T, seed=seed))
            cache[key] = (lab[: len(qry)], lab[len(qry):])
        return cache[key]

    out = []
    for check in s["checks"]:
        kind, tol = check["kind"], float(check.get("tolerance", 0.0)) * rng_v
        for eid in check["estimators"]:
            est = ests[eid]
            ref = entries[eid].get("reference", eid)
            profile = runner.profile(est.encoder) if hasattr(runner, "profile") else est.encoder
            cid, title = f"{check['id']}-{eid}", f"{check['title']} ({eid}) [{profile}]"
            needed = [est, ests[ref]]
            reason = ""
            if not all(runner.supports(e) for e in needed):
                reason = f"{runner.name} does not support " + ", ".join(
                    f"{e.encoder}/{e.decoder}/{e.empty_cells}" for e in needed)
            elif kind in _NEEDS_CELLS and not hasattr(runner, "cells"):
                reason = f"{runner.name} does not give the cells of its partitions"
            if reason:
                out.append(CheckOutcome(sc.id, cid, title, families.TestResult(check["family"], np.nan, 1.0, "not_reject"),
                                        skipped=reason))
                continue
            m = members(eid)
            empty = np.isnan(members(ref))
            if kind == "nan_iff_empty":
                at_q, at_s = labels(eid)
                no_datum = np.array([~np.isin(at_q[:, t], at_s[:, t]) for t in range(T)]).T
                r = families.almost_sure(int((np.isnan(m) != no_datum).sum()), m.size)
            elif kind == "unchanged":
                a, b = m[~empty], members(ref)[~empty]
                r = families.almost_sure(int((~(np.abs(a - b) <= tol)).sum()), a.size)
            elif kind == "filled":
                r = families.almost_sure(int(np.isnan(m).sum()), m.size)
            elif kind == "observed":
                x = m[empty].astype(np.float32)
                r = families.almost_sure(int((~np.isin(x, vals)).sum()), x.size)
            elif kind == "one_per_cell":
                at_q, _ = labels(eid)
                bad = groups = 0
                for t in range(T):
                    for c in np.unique(at_q[empty[:, t], t]):
                        x = m[(at_q[:, t] == c) & empty[:, t], t]
                        groups += 1
                        bad += int(len(np.unique(x)) > 1)
                r = families.almost_sure(bad, groups)
            elif kind == "data_law":
                qi = int(np.argmin(np.sum((qry - np.asarray(check["location"], np.float32)) ** 2, axis=1)))
                x = m[qi, empty[qi]].astype(np.float32)
                counts = [int(np.sum(x == v)) for v in vals]
                r = families.gof_counts(counts, [x.size / len(vals)] * len(vals), df=len(vals) - 1)
            elif kind == "cell_mean":
                at_q, at_s = labels(eid)
                bad = checked = 0
                for t in range(T):
                    means = np.array([vals[at_s[:, t] == c].astype(float).mean() for c in np.unique(at_s[:, t])])
                    x = m[empty[:, t], t]
                    checked += x.size
                    bad += int(sum(not (np.abs(means - xi) <= tol).any() for xi in x))
                r = families.almost_sure(bad, checked)
            else:
                raise ValueError(f"unknown check kind {kind}")
            if kind != "nan_iff_empty" and not empty.any():
                r = families.TestResult(r.family, r.statistic, r.p_value, r.pass_if,
                                        f"{r.detail}; no empty cell at the queries, nothing tested")
            out.append(CheckOutcome(sc.id, cid, title, r, expect=_expect(check, profile), route=_route(runner, est),
                                    known_failure=check.get("known_failure", "")))
    return out


def _clustered_data(s):
    """Data of a cv_selection scenario, drawn from ``data.generator_seed`` (not pinned): Gaussian
    clusters of ``data.per_cluster`` data around ``data.centres`` with standard deviation
    ``data.spread``, plus ``data.scattered`` data uniform in the domain, carrying a smooth field plus
    noise of standard deviation ``data.noise``."""
    rng = np.random.default_rng(int(s["data"]["generator_seed"]))
    box = np.asarray(s["domain"]["box"], float)
    centres = np.asarray(s["data"]["centres"], float)
    k, spread = int(s["data"]["per_cluster"]), float(s["data"]["spread"])
    clusters = np.vstack([c + spread * rng.standard_normal((k, len(box))) for c in centres])
    scattered = box[:, 0] + rng.random((int(s["data"]["scattered"]), len(box))) * (box[:, 1] - box[:, 0])
    pts = np.clip(np.vstack([clusters, scattered]), box[:, 0], box[:, 1])
    u = (pts - box[:, 0]) / (box[:, 1] - box[:, 0])
    vals = (np.sin(2 * np.pi * u[:, 0]) + 0.5 * np.cos(2 * np.pi * u[:, -1])
            + float(s["data"]["noise"]) * rng.standard_normal(len(pts)))
    return pts.astype(np.float32), vals.astype(np.float32)


def eval_cv_selection(sc: Scenario, runner: Runner, mode: str, seed: int,
                      save_maps=None) -> List[CheckOutcome]:
    r"""Evaluator ``cv_selection``: leave-one-out under the empty-cell policies (P11).

    The leave-one-out ensembles come from the runner's optional method ``loo``; without it every
    check is skipped. Every estimator is run with one seed. Kinds (key ``kind``):

    - ``loo_filled`` (``estimators``) — almost-sure: no leave-one-out member is NaN;
    - ``loo_unchanged`` (``estimators``, each naming its ``reference`` under ``"nan"``) — almost-sure:
      where the reference's member is finite, the member equals it, up to ``tolerance`` × data range;
    - ``selection`` (``estimator`` under ``"nan"``, ``reference`` defined at every datum) —
      two-sample: a score that drops undefined members leaves out the data with fewer than
      ``min_share`` × T finite members of ``estimator``. The absolute errors of the mean of the
      ``reference`` members at the data kept and at the data left out are compared by a
      Kolmogorov–Smirnov test. Run as a negative control, it shows that the data left out are not
      a random subset.

    Reads ``data.centres``, ``data.per_cluster``, ``data.spread``, ``data.scattered``, ``data.noise``,
    ``data.generator_seed`` and ``estimators_T``.
    """
    s = sc.spec
    T = int(_mode_value(s["estimators_T"], mode))
    entries = {e["id"]: e for e in s["estimators"]}
    ests = {eid: sc.estimator(e) for eid, e in entries.items()}
    pts, vals = _clustered_data(s)
    rng_v = float(vals.max() - vals.min())
    cache = {}

    def loo(eid):
        if eid not in cache:
            cache[eid] = np.asarray(runner.loo(ests[eid], pts, vals, n_members=T, seed=seed), float)
        return cache[eid]

    out = []
    for check in s["checks"]:
        kind, tol = check["kind"], float(check.get("tolerance", 0.0)) * rng_v
        eids = check["estimators"] if "estimators" in check else [check["estimator"]]
        for eid in eids:
            est = ests[eid]
            ref = check.get("reference", entries[eid].get("reference", eid))
            profile = runner.profile(est.encoder) if hasattr(runner, "profile") else est.encoder
            if "estimators" in check:
                cid, title = f"{check['id']}-{eid}", f"{check['title']} ({eid}) [{profile}]"
            else:
                cid, title = check["id"], f"{check['title']} [{profile}]"
            needed = [est, ests[ref]]
            reason = ""
            if not all(runner.supports(e) for e in needed):
                reason = f"{runner.name} does not support " + ", ".join(
                    f"{e.encoder}/{e.decoder}/{e.empty_cells}" for e in needed)
            elif not hasattr(runner, "loo"):
                reason = f"{runner.name} gives no leave-one-out ensemble"
            if reason:
                out.append(CheckOutcome(sc.id, cid, title, families.TestResult(check["family"], np.nan, 1.0, "not_reject"),
                                        skipped=reason))
                continue
            m = loo(eid)
            if kind == "loo_filled":
                r = families.almost_sure(int(np.isnan(m).sum()), m.size)
            elif kind == "loo_unchanged":
                defined = np.isfinite(loo(ref))
                a, b = m[defined], loo(ref)[defined]
                r = families.almost_sure(int((~(np.abs(a - b) <= tol)).sum()), a.size)
            elif kind == "selection":
                kept = np.isfinite(m).sum(axis=1) >= float(check["min_share"]) * T
                err = np.abs(loo(ref).mean(axis=1) - vals)
                score = np.abs(np.nanmean(m[kept], axis=1) - vals[kept]).mean()
                if min(kept.sum(), (~kept).sum()) < 2:   # nothing to compare: the test cannot reject
                    r = families.TestResult("two-sample", np.nan, 1.0, "not_reject", "fewer than 2 data on a side")
                else:
                    r = families.two_sample_ks(err[kept], err[~kept])
                r = families.TestResult(r.family, r.statistic, r.p_value, r.pass_if,
                                        f"{r.detail}; {int((~kept).sum())} of {len(vals)} data left out; "
                                        f"score of the kept data {score:.4f}, error over all data "
                                        f"{err.mean():.4f}, kept {err[kept].mean():.4f}, left out "
                                        f"{err[~kept].mean():.4f}")
            else:
                raise ValueError(f"unknown check kind {kind}")
            out.append(CheckOutcome(sc.id, cid, title, r, expect=_expect(check, profile), route=_route(runner, est),
                                    known_failure=check.get("known_failure", "")))
    return out


def _audit_fields(s, k):
    """The data of field ``k`` of a posterior_audit scenario, drawn from ``data.generator_seed + k``:
    ``clean``, ``planted`` (three planted errors), ``patch`` (a raised square of data plus one
    isolated error) and ``preferential`` (half of the data where the field is high), with the indices
    of the planted errors, the mask of the patch and the field's mean over the domain."""
    d = s["data"]
    rng = np.random.default_rng(int(d["generator_seed"]) + k)
    box = np.asarray(s["domain"]["box"], float)
    lo, hi = box[:, 0], box[:, 1]
    n, noise = int(d["n"]), float(d["noise"])

    def field(p):
        u = (p - lo) / (hi - lo)
        return np.sin(2 * np.pi * u[:, 0]) + 0.5 * np.cos(2 * np.pi * u[:, -1])

    def uniform(m):
        return lo + rng.random((m, len(box))) * (hi - lo)

    pts = uniform(n)
    clean = field(pts) + noise * rng.standard_normal(n)
    sd = float(np.std(clean))
    bad = rng.choice(n, 3, replace=False)
    planted = clean.copy()
    planted[bad[0]] *= 10
    planted[bad[1]] += 3 * sd
    planted[bad[2]] -= 2.5
    half = float(d["patch"]["side"]) / 2
    centre = lo + half + rng.random(len(box)) * (hi - lo - 2 * half)
    patch = np.all(np.abs(pts - centre) < half, axis=1)
    patched = clean.copy()
    patched[patch] += float(d["patch"]["shift"])
    lone = int(rng.choice(np.flatnonzero(~patch)))
    patched[lone] += 2.5
    pool = uniform(50 * n)
    hot = pool[field(pool) > float(d["preferential"]["above"])][: n // 2]
    pref = np.vstack([uniform(n - len(hot)), hot])
    pref_values = field(pref) + noise * rng.standard_normal(len(pref))
    mean = float(np.mean(field(uniform(200000))))
    return {"points": pts.astype(np.float32), "clean": clean.astype(np.float32),
            "planted": planted.astype(np.float32), "bad": bad, "patched": patched.astype(np.float32),
            "patch": patch, "lone": lone, "pref_points": pref.astype(np.float32),
            "pref_values": pref_values.astype(np.float32), "mean": mean}


def eval_posterior_audit(sc: Scenario, runner: Runner, mode: str, seed: int,
                         save_maps=None) -> List[CheckOutcome]:
    r"""Evaluator ``posterior_audit``: quality control of the data by their own laws (P12).

    Each replicate field gives four sets of data: clean, with three planted errors, with a raised
    patch plus one isolated error, and preferentially sampled. The leave-one-out ensembles come from
    the runner's optional method ``loo`` and the cells from ``cells``; the readings are those of
    :class:`~spatialize.gs.spa.PosteriorAudit` with its defaults (widening, one spread factor,
    Student-t tails), flags at false discovery rate ``data.q``. Kinds (key ``kind``):

    - ``errors_found`` (one-sided) — the share of the planted errors flagged exceeds ``bound``;
    - ``clean_flags`` (one-sided) — the share of clean fields with any flag stays below ``bound``;
    - ``shift_patch`` (paired-relation) — the mean shift (:meth:`PosteriorAudit.shift
      <spatialize.gs.spa.PosteriorAudit.shift>`) of the patch's data exceeds that of the other data;
    - ``shift_error`` (one-sided) — the shift at the isolated error is negative, its neighbours'
      laws being pulled towards it;
    - ``declustering`` (paired-relation) — the declustered mean is closer to the field's mean than
      the plain mean, under the preferential design;
    - ``weights`` (almost-sure) — the declustering weights are positive and sum to 1.

    Reads ``data.n``, ``data.noise``, ``data.patch``, ``data.preferential``, ``data.q``,
    ``data.fields`` (per mode), ``data.generator_seed`` and ``estimators_T``.
    """
    from spatialize.gs.spa import PosteriorAudit
    s = sc.spec
    K = int(_mode_value(s["data"]["fields"], mode))
    T = int(_mode_value(s["estimators_T"], mode))
    q = float(s["data"]["q"])
    ests = {e["id"]: sc.estimator(e) for e in s["estimators"]}
    cache = {}

    def audit(est, pts, vals, key):
        if key not in cache:
            members = np.asarray(runner.loo(est, pts, vals, n_members=T, seed=seed), float)
            box = np.asarray(est.domain, float)

            def cells(n_probes, probe_seed):
                rng = np.random.default_rng(probe_seed)
                probes = (box[:, 0] + rng.random((n_probes, len(box))) * (box[:, 1] - box[:, 0])).astype(np.float32)
                lab = np.asarray(runner.cells(est, pts, np.vstack([pts, probes]), n_members=T, seed=seed))
                return lab[: len(pts)], lab[len(pts):]
            cache[key] = PosteriorAudit(members, pts, vals, seed=seed, cells=cells)
        return cache[key]

    out = []
    for check in s["checks"]:
        kind = check["kind"]
        est = ests[check["estimator"]]
        profile = runner.profile(est.encoder) if hasattr(runner, "profile") else est.encoder
        cid, title = check["id"], f"{check['title']} [{profile}]"
        if not runner.supports(est) or not hasattr(runner, "loo") or not hasattr(runner, "cells"):
            out.append(CheckOutcome(sc.id, cid, title, families.TestResult(check["family"], np.nan, 1.0, "not_reject"),
                                    skipped=f"{runner.name} gives no leave-one-out ensembles or cells for {est.encoder}/{est.decoder}"))
            continue
        x, y, bad_w, n_w = [], [], 0, 0
        for k in range(K):
            f = _audit_fields(s, k)
            if kind == "errors_found":
                a = audit(est, f["points"], f["planted"], ("planted", est.id, k))
                x.append(float(np.mean(a.flags(q)[f["bad"]])))
            elif kind == "clean_flags":
                a = audit(est, f["points"], f["clean"], ("clean", est.id, k))
                x.append(float(a.flags(q).any()))
            elif kind == "shift_patch":
                a = audit(est, f["points"], f["patched"], ("patch", est.id, k))
                sh = a.shift()
                x.append(float(np.nanmean(sh[f["patch"]])))
                y.append(float(np.nanmean(sh[~f["patch"]])))
            elif kind == "shift_error":
                a = audit(est, f["points"], f["patched"], ("patch", est.id, k))
                x.append(float(a.shift()[f["lone"]]))
            elif kind == "declustering":
                a = audit(est, f["pref_points"], f["pref_values"], ("pref", est.id, k))
                d = a.declustered()
                x.append(abs(d.loc["mean", "declustered"] - f["mean"]))
                y.append(abs(d.loc["mean", "naive"] - f["mean"]))
            elif kind == "weights":
                a = audit(est, f["pref_points"], f["pref_values"], ("pref", est.id, k))
                w = a.weights()
                bad_w += int(np.sum(~(w > 0)) + (abs(w.sum() - 1) > 1e-9))
                n_w += len(w) + 1
            else:
                raise ValueError(f"unknown check kind {kind}")
        if kind == "errors_found":
            r = families.mean_greater(x, float(check["bound"]))
        elif kind == "clean_flags":
            r = families.mean_less(x, float(check["bound"]))
        elif kind == "shift_patch":
            r = families.paired_relation(x, y, better="greater")
        elif kind == "shift_error":
            r = families.mean_less(x, 0.0)
        elif kind == "declustering":
            r = families.paired_relation(x, y, better="less")
        else:
            r = families.almost_sure(bad_w, n_w)
        out.append(CheckOutcome(sc.id, cid, title, r, expect=_expect(check, profile), route=_route(runner, est),
                                known_failure=check.get("known_failure", "")))
    return out


def _mark_data(d, box, rng):
    """Data of a mark_law scenario: ``d["n"]`` data uniform in ``d["box"]`` (default the domain) plus
    ``d["cluster"]["n"]`` uniform in ``d["cluster"]["box"]``, with distinct values (smooth field
    plus noise), so a member identifies the datum it came from."""
    parts = [(int(d["n"]), np.asarray(d.get("box", box), float))]
    if "cluster" in d:
        parts.append((int(d["cluster"]["n"]), np.asarray(d["cluster"]["box"], float)))
    pts = np.vstack([b[:, 0] + rng.random((n, len(box))) * (b[:, 1] - b[:, 0]) for n, b in parts])
    u = (pts - box[:, 0]) / (box[:, 1] - box[:, 0])
    vals = np.sin(2 * np.pi * u[:, 0]) + 0.5 * np.cos(2 * np.pi * u[:, -1]) + 0.3 * rng.standard_normal(len(pts))
    return pts.astype(np.float32), vals.astype(np.float32)


def _mark_probabilities(cell, at_s_t, n, law):
    """Probability of each datum being the member of a query, under one partition, for the decoder
    ``draw`` with marks drawn as data (``mark_value="datum"``).

    Parameters
    ----------
    cell : int
        The query's cell label.
    at_s_t : ndarray of int, shape (n,)
        The cell labels of the data.
    n : int
        Number of data.
    law : {"cells", "data"}
        The mark source: one cell with data drawn uniformly, then one of its data uniformly
        (``"cells"``), or one datum drawn uniformly (``"data"``).

    Returns
    -------
    ndarray of shape (n,)
        Uniform over the data of the query's cell when it holds data, otherwise the mark law.
    """
    p = np.zeros(n)
    inside = at_s_t == cell
    if inside.any():
        p[inside] = 1.0 / inside.sum()
    elif law == "data":
        p[:] = 1.0 / n
    else:
        labels, counts = np.unique(at_s_t, return_inverse=False, return_counts=True)
        share = dict(zip(labels.tolist(), (1.0 / (len(labels) * counts)).tolist()))
        p[:] = [share[c] for c in at_s_t.tolist()]
    return p


def eval_mark_law(sc: Scenario, runner: Runner, mode: str, seed: int,
                  save_maps=None) -> List[CheckOutcome]:
    r"""Evaluator ``mark_law``: the law of the members under ``empty_cells="mark"`` (P5, P8, P9).

    The estimators use the decoder ``draw`` with marks drawn as data (``mark_value="datum"``), so
    every member is one datum and its law under each partition has a closed form given the cells
    (:func:`_mark_probabilities`), read with ``runner.cells``. Kinds (key ``kind``), one outcome per
    estimator of ``estimators``:

    - ``member_law`` — goodness of fit: at each of ``data.locations``, the number of times each
      datum is the member against its expectation, the sum over the partitions of its probability
      under the mark source ``law``. The Pearson statistic is referred to :math:`\chi^2` with
      :math:`k - q` degrees of freedom (k counts with positive expectation, q locations);
    - ``weights_sum`` — almost-sure: at each location the weights of the data, the mean over the
      partitions of the cell mean run on each datum's indicator (the estimator, or the one its entry
      names with ``reference``, with the cell mean under ``"nan"``),
      sum to one with the residual weight, the share of partitions in which the cell is empty, up to
      ``tolerance``;
    - ``residual_grows`` — paired relation: the location ``far`` has an empty cell more often than
      the location ``near`` (indices into ``data.locations``), paired by partition;
    - ``pair_product`` — identity: at the two locations ``pair``, the mean over the partitions of
      the product of their members minus its expectation given the cells is zero. Under one shared
      mark the expectation of an empty cell is the mark law's second moment, under distinct cells
      the product of the means. With ``shuffle``, the members of the second location are taken from
      the next partition, as if each location drew its own value (a negative control);
    - ``preferential`` — paired relation over ``fields`` fields: the error of the mean of the marks
      of ``estimator_b`` (drawn among the data) about the field's spatial mean exceeds that of
      ``estimator`` (one vote per cell), read at the location ``far`` beyond the data, in the
      partitions where its cell is empty. The field is ``field.base`` plus ``field.jump`` inside
      ``data.cluster.box``, plus noise of standard deviation ``field.noise``; the data are ``data.n``
      uniform in the domain plus ``data.cluster.n`` in the cluster's box.

    Reads ``data`` and ``estimators_T``.
    """
    s = sc.spec
    T = int(_mode_value(s["estimators_T"], mode))
    entries = {e["id"]: e for e in s["estimators"]}
    ests = {eid: sc.estimator(e) for eid, e in entries.items()}
    box = np.asarray(s["domain"]["box"], float)
    d = s["data"]
    gseed = int(d["generator_seed"])
    if "locations" in d:
        pts, vals = _mark_data(d, box, np.random.default_rng(gseed))
        qry = np.asarray(d["locations"], np.float32)
    cache = {}

    def members(eid, values=None, key=""):
        if (eid, key) not in cache:
            v = vals if values is None else np.asarray(values, np.float32)
            cache[(eid, key)] = np.asarray(runner.members(ests[eid], pts, v, qry, n_members=T, seed=seed), float)
        return cache[(eid, key)]

    def labels(eid):
        key = ("cells", ests[eid].encoder, ests[eid].rate)
        if key not in cache:
            lab = np.asarray(runner.cells(ests[eid], pts, np.vstack([qry, pts]), n_members=T, seed=seed))
            cache[key] = (lab[: len(qry)], lab[len(qry):])
        return cache[key]

    out = []
    for check in s["checks"]:
        kind = check["kind"]
        for eid in check["estimators"]:
            est = ests[eid]
            profile = runner.profile(est.encoder) if hasattr(runner, "profile") else est.encoder
            cid, title = f"{check['id']}-{eid}", f"{check['title']} ({eid}) [{profile}]"
            needed = [est] + [ests[k] for k in (entries[eid].get("reference"), check.get("estimator_b")) if k]
            reason = ""
            if not all(runner.supports(e) for e in needed):
                reason = f"{runner.name} does not support " + ", ".join(
                    f"{e.encoder}/{e.decoder}/{e.empty_cells}" for e in needed)
            elif not hasattr(runner, "cells"):
                reason = f"{runner.name} does not give the cells of its partitions"
            if reason:
                out.append(CheckOutcome(sc.id, cid, title, families.TestResult(check["family"], np.nan, 1.0, "not_reject"),
                                        skipped=reason))
                continue
            if kind == "member_law":
                m = members(eid)
                at_q, at_s = labels(eid)
                counts, expected = [], []
                for qi in range(len(qry)):
                    e_q = sum(_mark_probabilities(at_q[qi, t], at_s[:, t], len(vals), check["law"]) for t in range(T))
                    c_q = np.array([np.sum(m[qi].astype(np.float32) == v) for v in vals])
                    keep = e_q > 0
                    counts += list(c_q[keep]); expected += list(e_q[keep])
                r = families.gof_counts(counts, expected, df=len(counts) - len(qry))
            elif kind == "weights_sum":
                ref = entries[eid].get("reference", eid)
                ind = [members(ref, (np.arange(len(vals)) == j).astype(float), f"onehot{j}") for j in range(len(vals))]
                w = sum(np.nan_to_num(x, nan=0.0) for x in ind).mean(axis=1)
                residual = np.isnan(ind[0]).mean(axis=1)
                r = families.almost_sure(int((np.abs(w + residual - 1.0) > float(check["tolerance"])).sum()), len(qry))
            elif kind == "residual_grows":
                at_q, at_s = labels(eid)
                empty = np.array([~np.isin(at_q[:, t], at_s[:, t]) for t in range(T)]).T.astype(float)
                r = families.paired_relation(empty[int(check["far"])], empty[int(check["near"])], better="greater")
            elif kind == "pair_product":
                m = members(eid)
                at_q, at_s = labels(eid)
                a, b = (int(i) for i in check["pair"])
                law = entries[eid]["mark"]["source"]
                ya, yb = m[a], m[b]
                if check.get("shuffle"):
                    yb = np.roll(yb, -1)
                expect = np.empty(T)
                for t in range(T):
                    pa = _mark_probabilities(at_q[a, t], at_s[:, t], len(vals), law)
                    pb = _mark_probabilities(at_q[b, t], at_s[:, t], len(vals), law)
                    shared_mark = at_q[a, t] == at_q[b, t] and not np.isin(at_q[a, t], at_s[:, t])
                    expect[t] = pa @ (vals.astype(float) ** 2) if shared_mark else (pa @ vals) * (pb @ vals)
                dev = ya * yb - expect
                r = families.identity([dev.mean() / (dev.std(ddof=1) / np.sqrt(T))], "mean(Y_a·Y_b − E[Y_a·Y_b | cells])")
            elif kind == "preferential":
                K = int(_mode_value(check["fields"], mode))
                f = check["field"]
                far = np.asarray([check["far"]], np.float32)
                cbox = np.asarray(d["cluster"]["box"], float)
                spatial_mean = float(f["base"]) + float(f["jump"]) * float(np.prod(cbox[:, 1] - cbox[:, 0]) / np.prod(box[:, 1] - box[:, 0]))
                err = {eid: [], check["estimator_b"]: []}
                for k in range(K):
                    rng = np.random.default_rng([gseed, k])
                    p = np.vstack([box[:, 0] + rng.random((int(d["n"]), len(box))) * (box[:, 1] - box[:, 0]),
                                   cbox[:, 0] + rng.random((int(d["cluster"]["n"]), len(box))) * (cbox[:, 1] - cbox[:, 0])])
                    inside = np.all((p >= cbox[:, 0]) & (p <= cbox[:, 1]), axis=1)
                    v = float(f["base"]) + float(f["jump"]) * inside + float(f["noise"]) * rng.standard_normal(len(p))
                    p, v = p.astype(np.float32), v.astype(np.float32)
                    lab = np.asarray(runner.cells(est, p, np.vstack([far, p]), n_members=T, seed=seed + k))
                    empty = ~np.array([np.isin(lab[0, t], lab[1:, t]) for t in range(T)])
                    for e in err:
                        mk = np.asarray(runner.members(ests[e], p, v, far, n_members=T, seed=seed + k), float)[0]
                        err[e].append(abs(mk[empty].mean() - spatial_mean))
                r = families.paired_relation(err[check["estimator_b"]], err[eid], better="greater")
            else:
                raise ValueError(f"unknown check kind {kind}")
            out.append(CheckOutcome(sc.id, cid, title, r, expect=_expect(check, profile), route=_route(runner, est),
                                    known_failure=check.get("known_failure", "")))
    return out


def _uniform_data(s, positive=False):
    """``data.n`` data and ``data.queries`` queries uniform in the domain, with a smooth field plus
    noise (its exponential with ``positive``), drawn from ``data.generator_seed`` (not pinned)."""
    rng = np.random.default_rng(int(s["data"]["generator_seed"]))
    box = np.asarray(s["domain"]["box"], float)
    n, m = int(s["data"]["n"]), int(s["data"]["queries"])
    pts = box[:, 0] + rng.random((n, len(box))) * (box[:, 1] - box[:, 0])
    qry = box[:, 0] + rng.random((m, len(box))) * (box[:, 1] - box[:, 0])
    u = (pts - box[:, 0]) / (box[:, 1] - box[:, 0])
    vals = np.sin(2 * np.pi * u[:, 0]) + 0.5 * np.cos(2 * np.pi * u[:, -1]) + 0.3 * rng.standard_normal(n)
    if positive:
        vals = np.exp(vals)
    return pts.astype(np.float32), vals.astype(np.float32), qry.astype(np.float32)


def _sub_seed(seed, *keys):
    """A seed for one replicate, derived from the run's seed and the replicate's keys."""
    return int(np.random.default_rng([int(seed), *map(int, keys)]).integers(2 ** 31 - 1))


def eval_convergence(sc: Scenario, runner: Runner, mode: str, seed: int,
                     save_maps=None) -> List[CheckOutcome]:
    r"""Evaluator ``convergence``: the spread of the ensemble mean falls as :math:`T^{-1/2}` (P1).

    For each ensemble size of ``sizes``, ``replicates`` independent ensembles (seeds derived from
    the run's seed) are computed on the same data. At each query the spread is the standard
    deviation of the ensemble mean across the replicates. The slope of the mean over the queries of
    :math:`\log(\text{spread})` against :math:`\log T` must be :math:`-1/2`. Its standard error comes
    from ``bootstrap`` resamples of the replicates, drawn independently for each size; the
    identity's z is (slope + 1/2) / standard error.

    Reads ``data.n``, ``data.queries``, ``data.generator_seed``.
    """
    s = sc.spec
    ests = {e["id"]: sc.estimator(e) for e in s["estimators"]}
    pts, vals, qry = _uniform_data(s)
    out = []
    for check in s["checks"]:
        sizes = [int(t) for t in check["sizes"]]
        R = int(_mode_value(check["replicates"], mode))
        for eid in check["estimators"]:
            est = ests[eid]
            profile = runner.profile(est.encoder) if hasattr(runner, "profile") else est.encoder
            cid, title = f"{check['id']}-{eid}", f"{check['title']} ({eid}) [{profile}]"
            if not runner.supports(est):
                out.append(CheckOutcome(sc.id, cid, title, families.TestResult(check["family"], np.nan, 1.0, "not_reject"),
                                        skipped=f"{runner.name} does not support {est.encoder}/{est.decoder}"))
                continue
            means = np.array([[np.asarray(runner.members(est, pts, vals, qry, n_members=T,
                                                         seed=_sub_seed(seed, j, r)), float).mean(axis=1)
                               for r in range(R)] for j, T in enumerate(sizes)])     # (sizes, R, q)
            x = np.log(sizes)

            def slope(m):
                y = np.log(m.std(axis=1, ddof=1)).mean(axis=1)
                return float(np.polyfit(x, y, 1)[0])

            b = slope(means)
            rng = np.random.default_rng(_sub_seed(seed, 7, 7))
            boot = [slope(np.stack([means[j, rng.integers(0, R, R)] for j in range(len(sizes))]))
                    for _ in range(int(check["bootstrap"]))]
            r = families.identity([(b + 0.5) / np.std(boot, ddof=1)], f"(slope + 1/2)/se, slope={b:.3f}")
            out.append(CheckOutcome(sc.id, cid, title, r, expect=_expect(check, profile), route=_route(runner, est),
                                    known_failure=check.get("known_failure", "")))
    return out


def eval_law_validity(sc: Scenario, runner: Runner, mode: str, seed: int,
                      save_maps=None) -> List[CheckOutcome]:
    r"""Evaluator ``law_validity``: every reading of the law is a cumulative distribution function (P4).

    For the reading ``reading`` of the check, read through the runner's optional method ``law_cdf``,
    the values at every query and at ``thresholds`` thresholds evenly spaced over the data range
    widened by its length on each side must lie in :math:`[0, 1]` and be non-decreasing in the
    threshold. A violation is a value outside :math:`[0, 1]`, a NaN, or a decrease larger than
    ``tolerance``; a reading that cannot be built fails at every value. The check is almost sure.

    Reads ``data.n``, ``data.queries``, ``data.generator_seed`` and ``estimators_T``; the values are
    positive (the exponential of a smooth field plus noise), so that every widening applies.
    """
    s = sc.spec
    T = int(_mode_value(s["estimators_T"], mode))
    ests = {e["id"]: sc.estimator(e) for e in s["estimators"]}
    pts, vals, qry = _uniform_data(s, positive=True)
    lo, hi = float(vals.min()), float(vals.max())
    out = []
    for check in s["checks"]:
        x = np.linspace(lo - (hi - lo), hi + (hi - lo), int(check["thresholds"]))
        for eid in check["estimators"]:
            est = ests[eid]
            profile = runner.profile(est.encoder) if hasattr(runner, "profile") else est.encoder
            cid, title = f"{check['id']}-{eid}", f"{check['title']} ({eid}) [{profile}]"
            reason = ""
            if not runner.supports(est):
                reason = f"{runner.name} does not support {est.encoder}/{est.decoder}"
            elif not hasattr(runner, "law_cdf"):
                reason = f"{runner.name} does not give the readings of its law"
            if reason:
                out.append(CheckOutcome(sc.id, cid, title, families.TestResult(check["family"], np.nan, 1.0, "not_reject"),
                                        skipped=reason))
                continue
            try:
                F = np.asarray(runner.law_cdf(est, pts, vals, qry, x, reading=check["reading"], n_members=T,
                                              seed=seed), float)
                bad = int((~((F >= 0) & (F <= 1))).sum() + (np.diff(F, axis=1) < -float(check["tolerance"])).sum())
                r = families.almost_sure(bad, F.size)
            except Exception as e:   # a reading that cannot be built is no law: every value fails
                r = families.almost_sure(len(qry) * len(x), len(qry) * len(x))
                r = families.TestResult(r.family, r.statistic, r.p_value, r.pass_if,
                                        f"{r.detail}; the reading failed: {type(e).__name__}: {e}")
            out.append(CheckOutcome(sc.id, cid, title, r, expect=_expect(check, profile), route=_route(runner, est),
                                    known_failure=check.get("known_failure", "")))
    return out


def _line_cooccurrence_target(grid, i, j, rate):
    r"""The covariance of the cell means at grid points ``i`` ≤ ``j`` of a 1D Mondrian partition of
    rate ``rate`` (Poisson cuts), for values iid with variance one at the points of ``grid``.

    The two members are the means of their cells. Distinct cells share no value, so the covariance
    is :math:`P(\text{no cut in } (x_i, x_j))\, E[1/N]`, with :math:`N` the number of points of the
    common cell. Given that event the cell extends left and right of :math:`[x_i, x_j]` by
    independent Exp(rate) lengths; including exactly :math:`a` more points on the left has
    probability :math:`e^{-\lambda d_a} - e^{-\lambda d_{a+1}}`, with :math:`d_a` the distance to the
    a-th point (:math:`d_0 = 0`, :math:`d_{a+1} = \infty` past the last one), and likewise on the
    right.
    """
    g = np.asarray(grid, float)

    def side(dist):
        d = np.concatenate([[0.0], np.sort(dist), [np.inf]])
        return np.exp(-rate * d[:-1]) - np.exp(-rate * d[1:])      # P(exactly a more points)

    left, right = side(g[i] - g[:i]), side(g[j + 1:] - g[j])
    inner = j - i + 1
    n = inner + np.arange(len(left))[:, None] + np.arange(len(right))[None, :]
    return float(np.exp(-rate * (g[j] - g[i])) * np.sum(np.outer(left, right) / n))


def eval_uncorrelated_covariance(sc: Scenario, runner: Runner, mode: str, seed: int,
                                 save_maps=None) -> List[CheckOutcome]:
    r"""Evaluator ``uncorrelated_covariance``: the covariance the partitions induce between cell
    means of an uncorrelated field, against its closed form in the rate and the distance (P7).

    One dimension, ``data.n`` data on the regular grid :math:`(k + 1/2)/n` of the domain
    :math:`[0, 1]`, with values iid standard normal, drawn anew for each of ``fields`` fields with
    their own seed. The queries are the grid point ``data.origin`` and the points ``offsets`` grid
    steps to its right. Each field gives ``estimators_T`` members per query; the mean over its
    members of the product of the members at the origin and at a query estimates their covariance,
    the values having mean zero. Over the fields, these means against the closed form
    (:func:`_line_cooccurrence_target`) give one standardised difference per offset, combined by
    the identity family.
    """
    s = sc.spec
    T = int(_mode_value(s["estimators_T"], mode))
    ests = {e["id"]: sc.estimator(e) for e in s["estimators"]}
    d = s["data"]
    n, o = int(d["n"]), int(d["origin"])
    grid = (np.arange(n) + 0.5) / n
    pts = grid[:, None].astype(np.float32)
    out = []
    for check in s["checks"]:
        K = int(_mode_value(check["fields"], mode))
        offsets = [int(h) for h in check["offsets"]]
        qry = grid[[o] + [o + h for h in offsets]][:, None].astype(np.float32)
        for eid in check["estimators"]:
            est = ests[eid]
            profile = runner.profile(est.encoder) if hasattr(runner, "profile") else est.encoder
            cid, title = f"{check['id']}-{eid}", f"{check['title']} ({eid}) [{profile}]"
            if not runner.supports(est):
                out.append(CheckOutcome(sc.id, cid, title, families.TestResult(check["family"], np.nan, 1.0, "not_reject"),
                                        skipped=f"{runner.name} does not support {est.encoder}/{est.decoder}"))
                continue
            prod = np.empty((K, len(offsets)))
            for k in range(K):
                v = np.random.default_rng([int(d["generator_seed"]), k]).standard_normal(n).astype(np.float32)
                m = np.asarray(runner.members(est, pts, v, qry, n_members=T, seed=_sub_seed(seed, k)), float)
                prod[k] = (m[0] * m[1:]).mean(axis=1)
            target = np.array([_line_cooccurrence_target(grid, o, o + h, float(est.rate)) for h in offsets])
            z = (prod.mean(axis=0) - target) / (prod.std(axis=0, ddof=1) / np.sqrt(K))
            r = families.identity(z, "(covariance − closed form)/se")
            out.append(CheckOutcome(sc.id, cid, title, r, expect=_expect(check, profile), route=_route(runner, est),
                                    known_failure=check.get("known_failure", "")))
    return out


EVALUATORS = {"pair_cooccurrence": eval_pair_cooccurrence, "map_visual": eval_map_visual,
              "edge_cases": eval_edge_cases, "draw_laws": eval_draw_laws, "partition_law": eval_partition_law,
              "locality": eval_locality, "empty_cells": eval_empty_cells, "cv_selection": eval_cv_selection,
              "posterior_audit": eval_posterior_audit, "mark_law": eval_mark_law,
              "convergence": eval_convergence, "law_validity": eval_law_validity,
              "uncorrelated_covariance": eval_uncorrelated_covariance}


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
        """True when every outcome is as expected (passed, or a known failure that failed)."""
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


def run(scenarios, runner: Runner, mode="ci", seed=None, alpha=budget.ALPHA_SUITE, save_maps=None,
        progress=None) -> Report:
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
    progress : callable, optional
        Called with one line of text as the run advances: each scenario, each estimator's maps and
        each check's p-value before the Holm decision. The command line prints these lines.

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
    global _progress
    _progress = progress
    try:
        for i, sc in enumerate(scenarios, 1):
            _say(f"[{i}/{len(scenarios)}] {sc.id} (v{sc.spec.get('version', '?')}, {sc.spec['evaluator']}): "
                 f"{len(sc.spec.get('estimators', []))} estimators, {len(sc.spec['checks'])} checks")
            t0 = time.monotonic()
            rep.outcomes += EVALUATORS[sc.spec["evaluator"]](sc, runner, mode, seed, save_maps=save_maps)
            _say(f"  {sc.id} done in {time.monotonic() - t0:.0f} s")
    finally:
        _progress = None
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

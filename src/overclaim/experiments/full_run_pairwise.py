"""Session 5, Part 2 ("full_run"): the pairwise appendix (Tier 2) and the reason classification with
`cites_disclosure`, as two stages of the full_run runner (overclaim.experiments.full_run registers them).

- pairwise (brief Step 9 as moved to the appendix by the revision of 2026-11-06, its Step 10; addendum of 2026-11-07,
  sections 0, 3, 3b and 5): the runs of cfg["pairwise"]["runs"], in config order (cheapest judges first, Opus last),
  on the frozen sets (freeze.sets_out, the 60 main sets in their frozen order; the task dicts from the task file the
  freeze file names for each set). Each run is pilot 2's judge run (pilot2.stage_judge): judge.PairwiseJudge with the
  block's rubric question (configs/rubrics.yaml), both orders of every pair of cfg["pairs"] (judge.run_pairs2), the
  prompt cache for an Anthropic judge in the shown condition, max_workers per provider, records appended to
  run_dir/judge_calls.jsonl, one entry per run in run_dir/runs_status.json (done, skipped with its reason, stopped,
  failed). A run with `tasks: shown_eligible` judges only picks.shown_eligible of the freeze file (frozen order); a
  run with `in_plan: <cut>` is launched only when run_dir/projection.json says the final plan did not apply that cut.
  Records carry grader = the judge name (the Gemini route is in each record's `route`, so there is no `@vertex`
  suffix) and set = SETS[pairwise.set] ("v4_full_run", the set name of the pointwise stages' records, so that the
  pairwise and pointwise rows of a grader join on (grader, set, condition)).
- reasons (amendment 2, item 2): analysis_reasons_v5.classify_reasons_v5 with the `classifier` entry of
  configs/models.yaml over every verdict of judge_calls.jsonl with a non-empty reason -> run_dir/reason_labels.jsonl.

Gates and stop rules: a run that is not complete is not launched without a credential, once its provider is blocked
(Gemini HTTP 402 earlier in the invocation), or past its provider's line: Anthropic at
budget.anthropic_stop_launching_usd (pilot2.gate), OpenRouter at budget.openrouter_stop_launching_usd, both read from
run_dir/cost.jsonl; Google only
through the CostLogger's google_usd_cap. A budget cap during a call stops the stage (exit 3, BUDGET_STOP.txt); a
shown call that reads nothing from the prompt cache fails the run and the stage. Every Gemini run must carry
`route: vertex` or, since Uzair's memo of 2026-11-08 (README amendment 3, item 13; Vertex refused the project),
`route: aistudio` with a `projection_row` (its list-price row of results/full_run/projection.json): check_config
refuses the block otherwise, before any call. An aistudio run follows the launch rule of
overclaim.experiments.route_cap under budget.google_aistudio_cap_usd: the whole row when its re-priced remaining cost
fits the remaining cap, else the longest prefix of its tasks (frozen order) that fits, recorded PARTIAL with its n; a
route-cap BudgetExceeded, a 402 or a quota 429 ends the run PARTIAL, blocks Google for the invocation and lets the
next runs go on. The CLI's --only (Ctx6.only) runs the named runs and leaves the others untouched.

Stages are idempotent (records that exist are reused, every call is cached on disk), support the dry run (no call,
nothing written; the plan goes to ctx.plan) and delete nothing. The stage status goes to run_dir/stages_status.json.
This module does not import overclaim.experiments.full_run at load time (full_run imports it). Message text is
data: it is sent to the judges as text and written out, never executed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from overclaim.experiments import pilot2, route_cap
from overclaim.experiments.pilot2 import (COST_PROVIDER, EXIT_FAILED, EXIT_OK, _read_json, _read_jsonl, _repo_path,
                                          _write_json)
from overclaim.experiments.pilot3 import _record_stage, _skip
from overclaim.experiments.prep5 import _stage_failed
from overclaim.experiments.route_cap import AISTUDIO, PARTIAL

if TYPE_CHECKING:
    from overclaim.experiments.full_run import Ctx6

logger = logging.getLogger(__name__)

# The configs/full_run.yaml keys this module reads beyond the ones the runner already has (run_name, rubrics_config,
# models_config, budget, max_workers, freeze), with the stage that reads each.
CONFIG_BLOCKS: dict[str, str] = {
    "pairs": "pairwise (the pairs of variants, each judged in both orders)",
    "pairwise": "pairwise (set, rubric and the run list of Tier 2)",
    "reasons": "reasons (enabled)",
}
SETS = {"full_run": "v4_full_run"}     # pairwise.set -> the set name in every record (freeze.sets_out is the file)
CONDITIONS = ("hidden", "shown")
TASK_LISTS = ("shown_eligible",)        # a run's `tasks`: absent = every frozen set; else the freeze file's pick
RUN_KEYS = ("id", "judge", "route", "condition", "tasks", "in_plan", "projection_row")
GEMINI_ROUTES = ("vertex", "aistudio")
VARIANTS = ("O", "H", "P", "O_plus", "O_minus")
JUDGE_FILE = "judge_calls.jsonl"
LABELS_FILE = "reason_labels.jsonl"
STATUS_FILE = "runs_status.json"
PROJECTION_FILE = "projection.json"


# ---------------------------------------------------------------------------------------------
# Config, the frozen inputs, the plan
# ---------------------------------------------------------------------------------------------


def rubric_questions(cfg: dict) -> dict[str, str]:
    """{rubric name: final question} of cfg["rubrics_config"] (judge.load_rubrics; full_run.load_config does not load
    the rubrics)."""
    from overclaim.graders import judge as J

    return J.load_rubrics(_repo_path(cfg["rubrics_config"]))["rubrics"]


def classifier_spec(cfg: dict) -> dict:
    """The `classifier` entry of cfg["models_config"] (Haiku 5.5; full_run.load_config does not load it)."""
    with open(_repo_path(cfg["models_config"]), encoding="utf-8") as f:
        return yaml.safe_load(f)["classifier"]


def check_config(cfg: dict) -> None:
    """Validate the blocks of CONFIG_BLOCKS; ValueError listing every problem. pairs: a non-empty list of distinct
    pairs of two different variants of VARIANTS. pairwise: set a key of SETS, rubric a rubric of rubrics_config, runs a
    non-empty list of runs with unique ids and only the keys of RUN_KEYS, each with a known judge and a condition of
    CONDITIONS; `tasks` one of TASK_LISTS (required on a shown run, which must stay on the shown-eligible tasks);
    `in_plan` a non-empty string (a cut id of the projection); a judge whose provider has no cost name in
    pilot2.COST_PROVIDER is refused; a Gemini run's spec (pilot2.run_spec) must have route vertex (addendum section
    3b; models.yaml still says aistudio). reasons: enabled a bool. Both stages call this first, so a bad block is
    refused before any call."""
    problems: list[str] = []
    pairs = cfg.get("pairs")
    if (not isinstance(pairs, list) or not pairs
            or not all(isinstance(p, list) and len(p) == 2 and p[0] != p[1] and set(p) <= set(VARIANTS)
                       for p in pairs)):
        problems.append(f"pairs: a non-empty list of [first, second] pairs of two different variants of {VARIANTS}")
    elif len({tuple(p) for p in pairs}) != len(pairs):
        problems.append("pairs: a pair is listed twice")
    pw = cfg.get("pairwise")
    if not isinstance(pw, dict):
        raise ValueError("invalid pairwise config:\npairwise: the block is missing")
    if pw.get("set") not in SETS:
        problems.append(f"pairwise.set must be one of {sorted(SETS)}")
    if pw.get("rubric") not in rubric_questions(cfg):
        problems.append(f"pairwise.rubric {pw.get('rubric')!r} is not a rubric of {cfg['rubrics_config']}")
    runs = pw.get("runs")
    if not isinstance(runs, list) or not runs:
        problems.append("pairwise.runs must be a non-empty list")
        runs = []
    ids: list[str] = []
    for run in runs:
        if not isinstance(run, dict) or not run.get("id"):
            problems.append(f"pairwise run {run!r}: needs an id")
            continue
        rid = run["id"]
        ids.append(rid)
        unknown = sorted(set(run) - set(RUN_KEYS))
        if unknown:
            problems.append(f"pairwise run {rid}: unknown keys {unknown} (known: {RUN_KEYS})")
        if run.get("condition") not in CONDITIONS:
            problems.append(f"pairwise run {rid}: condition must be one of {CONDITIONS}")
        if "tasks" in run and run["tasks"] not in TASK_LISTS:
            problems.append(f"pairwise run {rid}: tasks must be one of {TASK_LISTS}")
        if run.get("condition") == "shown" and run.get("tasks") != "shown_eligible":
            problems.append(f"pairwise run {rid}: a shown run needs tasks: shown_eligible")
        if "in_plan" in run and (not isinstance(run["in_plan"], str) or not run["in_plan"].strip()):
            problems.append(f"pairwise run {rid}: in_plan must name a cut of the projection (e.g. A3)")
        if run.get("judge") not in cfg["judges"]:
            problems.append(f"pairwise run {rid}: unknown judge {run.get('judge')!r}")
            continue
        spec = pilot2.run_spec(cfg, run)
        if spec["provider"] not in COST_PROVIDER:
            problems.append(f"pairwise run {rid}: provider {spec['provider']!r} has no cost name")
        aistudio = spec["provider"] == "gemini" and spec.get("route") == AISTUDIO
        if spec["provider"] == "gemini" and spec.get("route") not in GEMINI_ROUTES:
            problems.append(f"pairwise run {rid}: Gemini route {spec.get('route')!r}; every Gemini run of Part 2 "
                            "must carry route: vertex or aistudio (addendum section 3b; amendment 3, item 13)")
        if aistudio:
            cap = (cfg.get("budget") or {}).get(route_cap.CAP_KEY)
            if not isinstance(cap, (int, float)) or isinstance(cap, bool) or cap <= 0:
                problems.append(f"pairwise run {rid}: an aistudio run needs budget.{route_cap.CAP_KEY} (a positive USD "
                                "amount)")
            if not isinstance(run.get("projection_row"), str) or not run["projection_row"]:
                problems.append(f"pairwise run {rid}: an aistudio run needs projection_row (its row id in "
                                f"{PROJECTION_FILE})")
        elif "projection_row" in run:
            problems.append(f"pairwise run {rid}: projection_row is only for a Gemini run on route aistudio")
    twice = sorted({i for i in ids if ids.count(i) > 1})
    if twice:
        problems.append(f"pairwise run ids must be unique: {twice}")
    reasons = cfg.get("reasons")
    if not isinstance(reasons, dict) or not isinstance(reasons.get("enabled"), bool):
        problems.append("reasons.enabled must be true or false")
    if problems:
        raise ValueError("invalid pairwise / reasons config:\n" + "\n".join(problems))


def frozen_inputs(cfg: dict) -> tuple[list[dict], list[dict], dict]:
    """(task dicts, sets, the freeze document), both lists in the frozen order (the sets file's).

    The freeze document is freeze.tasks_out, the sets file freeze.sets_out. Checks: both exist; the sha256 of the sets
    file's bytes equals the document's sets_file_sha256 (the freeze writes the file as exactly the text it hashes);
    the sets' task ids are the document's tasks in the same order; each task's task_file (a file name) is the name of
    one of freeze.task_files and that file holds the task (its first record with the id is taken, as the freeze's
    join does). ValueError naming the problem."""
    fz = cfg["freeze"]
    doc_path, sets_path = _repo_path(fz["tasks_out"]), _repo_path(fz["sets_out"])
    missing = [str(p) for p in (doc_path, sets_path) if not p.exists()]
    if missing:
        raise ValueError(f"the frozen files are missing (run the freeze stage first): {missing}")
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(sets_path.read_bytes()).hexdigest()
    if digest != doc["sets_file_sha256"]:
        raise ValueError(f"{fz['sets_out']} has sha256 {digest}, the freeze recorded {doc['sets_file_sha256']}")
    sets = _read_jsonl(sets_path)
    ids = [s["task_id"] for s in sets]
    if ids != [t["task_id"] for t in doc["tasks"]]:
        raise ValueError(f"the task ids of {fz['sets_out']} are not the tasks of {fz['tasks_out']} in that order")
    files = {Path(p).name: _repo_path(p) for p in fz["task_files"]}
    wanted: dict[str, set[str]] = {}
    for t in doc["tasks"]:
        if t["task_file"] not in files:
            raise ValueError(f"{t['task_id']}: its task file {t['task_file']!r} is not among freeze.task_files")
        wanted.setdefault(t["task_file"], set()).add(t["task_id"])
    found: dict[str, dict] = {}
    for name, want in wanted.items():
        for task in _read_jsonl(files[name]):
            if task["task_id"] in want and task["task_id"] not in found:
                found[task["task_id"]] = task
    absent = [i for i in ids if i not in found]
    if absent:
        raise ValueError(f"{len(absent)} frozen tasks are not in the task file the freeze names for them: {absent}")
    return [found[i] for i in ids], sets, doc


def run_task_ids(run: dict, sets: list[dict], doc: dict, seeded_with: Any = None) -> list[str]:
    """The task ids a run judges, in the frozen order: every set, or (tasks: shown_eligible) the sets among the freeze
    file's picks.shown_eligible. ValueError when that pick names a task without a set. With `seeded_with` (a seed; the
    aistudio runs, README amendment 3, item 13) the ids come in seed order instead: the frozen E2 pick first (the
    seeded order the freeze holds over all sets, carried across the re-freeze), then the rest by
    prep5_e2.pick_tasks with that seed (per gap type a seeded order, one per gap type in turn), so that a row the cap
    cuts runs on a prefix that takes the gap types in turn."""
    ids = [s["task_id"] for s in sets]
    if run.get("tasks") is not None:
        pick = doc["picks"][run["tasks"]]
        unknown = sorted(set(pick) - set(ids))
        if unknown:
            raise ValueError(f"picks.{run['tasks']} names tasks without a frozen set: {unknown}")
        keep = set(pick)
        ids = [i for i in ids if i in keep]
    if seeded_with is None:
        return ids
    from overclaim.experiments.prep5_e2 import pick_tasks

    wanted = set(ids)
    first = [i for i in doc["picks"].get("e2_tasks", []) if i in wanted]
    rest_sets = [s for s in sets if s["task_id"] in wanted and s["task_id"] not in set(first)]
    return first + pick_tasks(rest_sets, len(rest_sets), seeded_with)


def cut_reason(run_dir: Path, cut: str) -> str | None:
    """Reason not to launch a run with `in_plan: <cut>`, or None when the final plan kept it: run_dir/projection.json
    (overclaim.projection_v4, plan part2) must exist with final_run true and branches[0].budget.cuts_decided true; the
    run is then cut ("cut <cut> applied") when <cut> is the id of one of the budget steps (the cuts applied, in order),
    and kept otherwise. A missing file or an undecided plan is a reason too (the run is not launched). ValueError when
    the file is not JSON or lacks branches[0].budget (or, in a decided plan, the ids of budget.steps)."""
    path = run_dir / PROJECTION_FILE
    if not path.exists():
        return f"{PROJECTION_FILE} is missing, so whether cut {cut} was applied is unknown"
    projection = json.loads(path.read_text(encoding="utf-8"))
    try:
        budget = projection["branches"][0]["budget"]
        if projection.get("final_run") is not True or budget.get("cuts_decided") is not True:
            return f"{PROJECTION_FILE} is not a final plan with its cuts decided, so whether cut {cut} applies is open"
        applied = [step["id"] for step in budget["steps"]]
    except (LookupError, TypeError, AttributeError) as exc:
        raise ValueError(f"{PROJECTION_FILE} has no branches[0].budget with the ids of its steps "
                         f"({type(exc).__name__}: {exc})") from exc
    return f"cut {cut} applied" if cut in applied else None


def launch_gate(ctx: Ctx6, run: dict, provider: str) -> str | None:
    """Reason not to launch the run now, or None. Anthropic: pilot2.gate (budget.anthropic_stop_launching_usd on
    run_dir/cost.jsonl). OpenRouter: the run's own cost file at or above budget.openrouter_stop_launching_usd (the
    lead's rule 2; pilot2.gate gates Anthropic only). Google: None (google_usd_cap is enforced per call by the
    CostLogger; no stop line)."""
    if provider == "openrouter":
        spent = ctx.cost.spent("openrouter")
        stop = float(ctx.cfg["budget"]["openrouter_stop_launching_usd"])
        if spent >= stop:
            return f"OpenRouter total ${spent:.2f} has reached the stop-launching threshold ${stop:.2f}"
        return None
    return pilot2.gate(ctx, run, provider)


def _present(records: list[dict], grader: str, set_name: str, rubric: str, condition: str, ids: set[str],
             pair_labels: set[str]) -> int:
    """Distinct (task_id, pair, order) among the records of this grader, set, rubric and condition on these tasks and
    pairs (the resume key of judge.run_pairs2 within one run)."""
    return len({(r["task_id"], r["pair"], r["order"]) for r in records
                if r["grader"] == grader and r.get("set") == set_name and r.get("rubric") == rubric
                and r["condition"] == condition and r["task_id"] in ids and r["pair"] in pair_labels})


def openrouter_slugs(cfg: dict, runs: list[dict]) -> list[str]:
    """The sorted OpenRouter model slugs of these runs' judge specs."""
    specs = [pilot2.run_spec(cfg, r) for r in runs]
    return sorted({spec["model"] for spec in specs if spec["provider"] == "openrouter"})


def export_usage(ctx: Ctx6, doc: dict) -> int:
    """pilot2.export_openrouter_usage (run_dir/openrouter_usage.jsonl) over the OpenRouter model of every judge of
    cfg["judges"], limited to the cache entries created at or after the freeze's frozen_at. The disk cache is shared
    with every earlier session, whose entries are left out this way (pilot3._stage_runs used the first line of the
    run's cost file instead; the cache entry of a call is written before its cost line, so that would leave out the
    first Part 2 call). The file then holds every Part 2 call of those models, whichever stage made it. Returns the
    number of lines."""
    slugs = sorted({spec["model"] for spec in ctx.cfg["judges"].values() if spec["provider"] == "openrouter"})
    return pilot2.export_openrouter_usage(ctx, slugs, since=doc["frozen_at"])


def aistudio_plan(ctx: Ctx6, run: dict, grader: str, set_name: str, rubric: str, condition: str, ids: list[str],
                  existing: list[dict], pairs: list[list[str]], pair_labels: set[str]) -> dict:
    """The launch plan of an aistudio pairwise run under the cap (route_cap): the cap state, the unit (measured over
    the run's live aistudio verdicts when at least MIN_LIVE exist, else the list price of its projection row), the
    missing verdicts and the per-task cost of them (every pair in both orders less the records present), and the
    prefix fit over the frozen task order. ValueError when the cap, the projection or the row's unit is missing."""
    state = route_cap.cap_state(ctx.cost)
    units = route_cap.projection_units(ctx.run_dir)
    keep = (lambda r: r["grader"] == grader and r.get("set") == set_name and r.get("rubric") == rubric
            and r["condition"] == condition and r["pair"] in pair_labels and not r.get("smoke_only"))
    unit = route_cap.unit_for(existing, keep, units, run["projection_row"])
    per_call = len(pairs) * 2
    present_of = {}
    for r in existing:
        if keep(r) and r["task_id"] in set(ids):
            present_of.setdefault(r["task_id"], set()).add((r["pair"], r["order"]))
    per_task = [(tid, (per_call - len(present_of.get(tid, ()))) * unit["usd"]) for tid in ids]
    missing = sum(per_call - len(present_of.get(tid, ())) for tid in ids)
    return {**state, "units": {"pairwise": unit}, "remaining_calls": {"pairwise": missing}, "per_task": per_task,
            "fit": route_cap.prefix_fit(per_task, state["remaining"])}


def _is_balance_stop(exc: BaseException) -> bool:
    """gemini.BalanceExhausted or its QuotaExhausted, matched by class name (the module does not import gemini)."""
    return any(c.__name__ == "BalanceExhausted" for c in type(exc).__mro__)


def _is_permission_denied(exc: BaseException) -> bool:
    """An HTTP 403 / permission denied from a Gemini route (the policy refusal of 2026-11-08)."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code == 403 or type(exc).__name__ == "PermissionDeniedError" or "permission_denied" in str(exc).lower()


# ---------------------------------------------------------------------------------------------
# Stage: pairwise
# ---------------------------------------------------------------------------------------------


def stage_pairwise(ctx: Ctx6) -> int:
    """The runs of cfg["pairwise"]["runs"] in config order on the frozen sets (see the module docstring).

    Per run: the task ids (run_task_ids), the records expected (tasks x pairs x 2 orders) and present in
    judge_calls.jsonl. A run with in_plan is skipped with cut_reason's reason when there is one (the projection is read
    once, with the frozen inputs, before any run; a malformed one fails the stage). A run with `tasks` records how many
    frozen tasks it leaves out and why (n_tasks_frozen, n_left_out, left_out_reason). A complete run is not
    gated (run_pairs2 reads its records back without a call); otherwise no credential, a provider blocked earlier in
    the invocation, or launch_gate skips it, and a skipped run keeps the reason recorded when it was first skipped
    (pilot2.stage_judge). Before the first OpenRouter run that launches, pilot2.verify_openrouter_slugs over the
    OpenRouter slugs of the block (once; a failure fails the run and the stage). BudgetExceeded stops the stage
    (pilot2._budget_stop, exit 3); CacheNotRead and any other error fail the run and the stage (exit 1); a Gemini
    BalanceExhausted stops the run and blocks Google for the rest of the invocation. After each launched run the cost
    line is printed. Whichever way a live pass leaves the run loop (done, budget stop, failure), export_usage then
    rewrites openrouter_usage.jsonl when the block has OpenRouter judges (pilot 2 exports on every pass with OpenRouter
    slugs; the export is offline and rebuilds the file from the disk cache, so a rerun after a stop writes it too).
    The stage entry in stages_status.json is done with each run's status, or stopped / failed with the reason. Dry
    run: check_config and the frozen inputs as in a live run, then one plan line per run (judge, provider and route,
    set, rubric, condition, tasks, records expected and present, credential, and launch / complete / the skip
    reason) and the slug check that would run; nothing is written."""
    from overclaim.graders import judge as J

    cfg = ctx.cfg
    check_config(cfg)   # ValueError before any call
    pw = cfg["pairwise"]
    set_name, rubric = SETS[pw["set"]], pw["rubric"]
    question = rubric_questions(cfg)[rubric]
    pairs = [list(p) for p in cfg["pairs"]]
    pair_labels = {f"{a}_vs_{b}" for a, b in pairs}
    try:
        tasks, sets, doc = frozen_inputs(cfg)
        ids_of = {run["id"]: run_task_ids(run, sets, doc, seeded_with=cfg["e2"]["seed"]
                                          if pilot2.run_spec(cfg, run).get("route") == AISTUDIO
                                          and pilot2.run_spec(cfg, run)["provider"] == "gemini" else None)
                  for run in pw["runs"]}
        plan_reasons = {run["id"]: cut_reason(ctx.run_dir, run["in_plan"]) for run in pw["runs"] if "in_plan" in run}
    except (ValueError, OSError, LookupError, TypeError) as exc:
        if ctx.dry_run:
            logger.error("pairwise: FAILED: %s: %s", type(exc).__name__, exc)
            return EXIT_FAILED
        return _stage_failed(ctx, "pairwise", exc)
    task_by_id = {t["task_id"]: t for t in tasks}
    set_by_id = {s["task_id"]: s for s in sets}
    out_path, status_path = ctx.run_dir / JUDGE_FILE, ctx.run_dir / STATUS_FILE
    existing = _read_jsonl(out_path)
    status: dict[str, dict] = _read_json(status_path, {})
    slugs = openrouter_slugs(cfg, pw["runs"])
    slugs_checked = False
    blocked: dict[str, str] = {}
    outcome: dict[str, str] = {}
    failed_runs: list[str] = []

    only = getattr(ctx, "only", None)
    if only is not None:
        unknown = sorted(set(only) - {run["id"] for run in pw["runs"]})
        if unknown:
            logger.error("pairwise: --only names runs that are not in this stage: %s; nothing run, nothing recorded",
                         unknown)
            if ctx.dry_run:
                ctx.note(f"pairwise: --only names runs that are not in this stage: {unknown}; the live stage would "
                         "run nothing and record nothing")
            return EXIT_FAILED
        logger.info("pairwise: --only %s: the other runs are left untouched", sorted(only))
        if ctx.dry_run:
            ctx.note(f"pairwise: --only {sorted(only)}: the other runs are left untouched (no status entry rewritten)")
    try:
        for run in pw["runs"]:
            if only is not None and run["id"] not in only:
                logger.info("pairwise: run %s: not in --only; untouched", run["id"])
                if ctx.dry_run:
                    ctx.note(f"pairwise: run {run['id']}: NOT in --only: untouched")
                continue
            spec = pilot2.run_spec(cfg, run)
            provider = COST_PROVIDER[spec["provider"]]
            grader, condition = run["judge"], run["condition"]
            ids = ids_of[run["id"]]
            expected = len(ids) * len(pairs) * 2
            have = _present(existing, grader, set_name, rubric, condition, set(ids), pair_labels)
            entry: dict[str, Any] = {"judge": grader, "set": set_name, "rubric": rubric, "condition": condition,
                                     "provider": provider, "route": spec.get("route"), "tasks": run.get("tasks", "all"),
                                     "n_tasks": len(ids), "expected_records": expected}
            aistudio = spec["provider"] == "gemini" and spec.get("route") == AISTUDIO
            left_out = ""
            if "tasks" in run:
                entry.update(n_tasks_frozen=len(sets), n_left_out=len(sets) - len(ids),
                             left_out_reason=f"not in picks.{run['tasks']}")
                left_out = (f"; {entry['n_left_out']} of the {len(sets)} frozen tasks left out: "
                            f"{entry['left_out_reason']}")
            if "in_plan" in run:
                entry["in_plan"] = run["in_plan"]
            complete = have >= expected
            plan_reason = plan_reasons.get(run["id"])
            reason = plan_reason
            if reason is None and not complete:
                if not pilot2.credential_for(spec):
                    reason = f"no credential for provider {spec['provider']}" + (
                        f" route {spec.get('route')}" if spec["provider"] == "gemini" else "")
                elif provider in blocked:
                    reason = blocked[provider]
                else:
                    reason = launch_gate(ctx, run, provider)
            run_ids, cut = ids, None
            if reason is None and not complete and aistudio:
                # the memo's launch rule (route_cap): whole row, the longest task prefix that fits, or skipped
                try:
                    plan = aistudio_plan(ctx, run, grader, set_name, rubric, condition, ids, existing, pairs,
                                         pair_labels)
                except ValueError as exc:
                    plan, reason = None, f"aistudio plan not computable: {exc}"
                if plan is not None:
                    entry["aistudio"] = {k: v for k, v in plan.items() if k != "per_task"}
                    fit = plan["fit"]
                    text = route_cap.plan_text(plan)
                    logger.info("pairwise: run %s: %s", run["id"], text)
                    if ctx.dry_run:
                        ctx.note(f"pairwise: run {run['id']}: {text}")
                    if plan["remaining"] <= 0:
                        reason = (f"aistudio cap reached: {plan['spent']:.4f} of {plan['cap']:.2f} USD spent (memo of "
                                  "2026-11-08, section 1.3)")
                    elif not fit["fits"] and fit["n_fit"] == 0:
                        reason = (f"aistudio cap: the remaining {plan['remaining']:.4f} USD does not cover the first "
                                  f"task's missing verdicts ({plan['per_task'][0][1]:.4f} USD at the re-priced unit)")
                    elif not fit["fits"]:
                        run_ids = ids[:fit["n_fit"]]
                        cut = {"n_tasks_run": fit["n_fit"], "n_tasks_planned": fit["n_tasks"],
                               "cost_fit_usd": fit["cost_fit"]}
                        entry["aistudio"]["prefix"] = cut
                        logger.warning("pairwise: run %s: the whole row does not fit the remaining aistudio cap; "
                                       "judging the first %d of %d tasks (%.2f USD) and recording the run as %s",
                                       run["id"], fit["n_fit"], fit["n_tasks"], fit["cost_fit"], PARTIAL)
            if ctx.dry_run:
                state = (f"SKIPPED: {reason}" if reason is not None else
                         "complete: records read back, no call" if complete else
                         f"would launch: {expected - have} records to judge")
                route = f", route {spec.get('route')}" if spec["provider"] == "gemini" else ""
                kept = "kept by the final plan" if plan_reason is None else plan_reason
                plan = f"in_plan {run['in_plan']}: {kept}; " if "in_plan" in run else ""
                ctx.note(f"pairwise: run {run['id']}: {grader} ({spec['provider']}{route}) / {set_name} / {rubric} / "
                         f"{condition}, {len(ids)} tasks ({entry['tasks']}{left_out}), {expected} records expected, "
                         f"{have} present; {plan}credential {'set' if pilot2.credential_for(spec) else 'MISSING'}; "
                         f"{state}")
                continue
            if reason is not None:
                previous = status.get(run["id"], {})
                if previous.get("status") == PARTIAL:
                    # a row an earlier invocation left PARTIAL stays PARTIAL (its reason, counts and prefix)
                    entry.update(status=PARTIAL, reason=previous.get("reason"), reason_on_last_invocation=reason,
                                 records=have)
                    prefix = (previous.get("aistudio") or {}).get("prefix")
                    if prefix is not None:
                        entry["aistudio"] = {**entry.get("aistudio", {}), "prefix": prefix}
                    status[run["id"]] = entry
                    _write_json(status_path, status)
                    outcome[run["id"]] = PARTIAL
                    logger.warning("pairwise: run %s stays %s (not launched: %s)", run["id"], PARTIAL, reason)
                    continue
                entry.update(status="skipped", reason=reason, records=have)
                if previous.get("status") == "skipped" and previous.get("reason") != reason:
                    # keep the reason recorded when the run was first skipped (e.g. the total at that moment)
                    entry.update(reason=previous.get("reason"), reason_on_last_invocation=reason)
                status[run["id"]] = entry
                _write_json(status_path, status)
                outcome[run["id"]] = "skipped"
                logger.warning("pairwise: run %s SKIPPED: %s", run["id"], reason)
                continue
            if provider == "openrouter" and not complete and not slugs_checked:
                try:
                    pilot2.verify_openrouter_slugs(slugs)
                except Exception as exc:
                    entry.update(status="failed", reason=f"OpenRouter slug check: {type(exc).__name__}: {exc}",
                                 records=have)
                    status[run["id"]] = entry
                    _write_json(status_path, status)
                    return _stage_failed(ctx, "pairwise", exc)
                slugs_checked = True
                logger.info("pairwise: OpenRouter lists %s", slugs)
            judge = J.PairwiseJudge(ctx.client(spec), grader, rubric=rubric, question=question,
                                    structured=bool(spec.get("structured_output")))
            logger.info("pairwise: run %s: %s / %s / %s / %s, %d tasks, %d of %d records present", run["id"], grader,
                        set_name, rubric, condition, len(ids), have, expected)
            def short(reason_text: str) -> None:
                """Record the run as PARTIAL with its present verdicts (a route-cap, billing or quota stop)."""
                existing_now = _read_jsonl(out_path)
                entry.update(status=PARTIAL, reason=reason_text,
                             records=_present(existing_now, grader, set_name, rubric, condition, set(ids), pair_labels))
                status[run["id"]] = entry
                _write_json(status_path, status)
                outcome[run["id"]] = PARTIAL

            try:
                records = J.run_pairs2(
                    [task_by_id[i] for i in run_ids], [set_by_id[i] for i in run_ids], judge, pairs, condition,
                    str(out_path), cfg["run_name"], set_name=set_name,
                    max_workers=int(cfg.get("max_workers", {}).get(provider, 1)),
                    use_prompt_cache=condition == "shown" and spec["provider"] == "anthropic")
            except J.BudgetExceeded as exc:
                if getattr(exc, "route", None) == AISTUDIO:
                    # the route cap: the row ends short, Google is blocked for the invocation, the next runs go on
                    blocked[provider] = f"aistudio cap: {exc}"
                    short(f"aistudio cap: {exc}")
                    logger.error("pairwise: run %s PARTIAL (the next runs go on): %s", run["id"], exc)
                    print(f"[{run['id']}] {route_cap.cap_line(ctx.cost)}", flush=True)
                    existing = _read_jsonl(out_path)
                    continue
                entry.update(status="stopped", reason=f"budget cap: {exc}")
                status[run["id"]] = entry
                _write_json(status_path, status)
                _record_stage(ctx, "pairwise", {"status": "stopped", "run": run["id"], "reason": f"budget cap: {exc}"})
                return pilot2._budget_stop(ctx, "pairwise", exc)
            except Exception as exc:
                if _is_balance_stop(exc):
                    # HTTP 402 or a quota 429 (gemini.BalanceExhausted / QuotaExhausted): the row is PARTIAL, never
                    # retried; Google is blocked for the rest of the invocation (memo of 2026-11-08, section 1.3)
                    blocked[provider] = f"Gemini billing or quota stop during run {run['id']}: {exc}"
                    short(f"{type(exc).__name__}: {exc}")
                    logger.error("pairwise: run %s PARTIAL (billing or quota stop; the next runs go on): %s",
                                 run["id"], exc)
                    if aistudio:
                        print(f"[{run['id']}] {route_cap.cap_line(ctx.cost)}", flush=True)
                    existing = _read_jsonl(out_path)
                    continue
                # judge.CacheNotRead, a provider error left after the client's retries, or an unexpected error: the
                # run is failed, the next runs go on, the stage ends failed naming the failed runs and a new invocation
                # resumes them from their records (2026-11-08, as the shown pointwise stage; before, the stage stopped)
                failure = f"{type(exc).__name__}: {exc}"
                if spec["provider"] == "gemini" and _is_permission_denied(exc):
                    # the policy refusal (HTTP 403): no retry, no other route; Google is blocked for the invocation
                    blocked[provider] = f"Gemini HTTP 403 / permission denied during run {run['id']}: {exc}"
                    failure = f"not run (provider restriction): {failure}"
                entry.update(status="failed", reason=failure)
                status[run["id"]] = entry
                _write_json(status_path, status)
                outcome[run["id"]] = "failed"
                failed_runs.append(run["id"])
                logger.error("pairwise: run %s FAILED (the next runs go on): %s: %s", run["id"], type(exc).__name__,
                             exc, exc_info=exc)
                existing = _read_jsonl(out_path)
                continue
            existing = _read_jsonl(out_path)
            entry.update(status="done" if cut is None else PARTIAL, records=len(records),
                         n_invalid=sum(r["winner"] == "invalid" for r in records),
                         n_parse_retries=sum(int(r.get("parse_attempts", 1)) > 1 for r in records),
                         n_live=sum(not r["cached"] for r in records))
            if cut is not None:
                entry["reason"] = (f"aistudio cap: judged the first {cut['n_tasks_run']} of {cut['n_tasks_planned']} "
                                   "tasks (the longest prefix whose re-priced cost fit under the remaining cap)")
            status[run["id"]] = entry
            _write_json(status_path, status)
            outcome[run["id"]] = entry["status"]
            logger.info("pairwise: run %s %s: %d records, %d invalid; %s", run["id"], entry["status"], len(records),
                        entry["n_invalid"], ctx.cost.totals_line())
            print(f"[{run['id']}] {ctx.cost.totals_line()}", flush=True)
            if aistudio:
                print(f"[{run['id']}] {route_cap.cap_line(ctx.cost)}", flush=True)
    finally:
        if slugs and not ctx.dry_run:
            logger.info("pairwise: wrote %d OpenRouter usage objects created since %s to openrouter_usage.jsonl",
                        export_usage(ctx, doc), doc["frozen_at"])

    if ctx.dry_run:
        if slugs:
            ctx.note(f"pairwise: before the first OpenRouter run that launches, the slugs {slugs} would be checked "
                     "against OpenRouter's model list (once); at the end of the pass openrouter_usage.jsonl would be "
                     "rewritten with the cache entries of the OpenRouter judges created since frozen_at "
                     f"{doc['frozen_at']}")
        return EXIT_OK
    extra = {"only": sorted(only)} if only is not None else {}
    # every run of the block: this invocation's outcome, else the status an earlier invocation recorded (--only
    # leaves the others untouched), else "not run"; the stage is failed while any run is
    all_runs = {run["id"]: outcome.get(run["id"], status.get(run["id"], {}).get("status", "not run"))
                for run in pw["runs"]}
    failed_all = [rid for rid, st in all_runs.items() if st == "failed"]
    if failed_all:
        reasons = "; ".join(f"{r}: {status.get(r, {}).get('reason')}" for r in failed_all)
        _record_stage(ctx, "pairwise", {"status": "failed", "runs": all_runs,
                                        "reason": f"failed runs {failed_all} ({reasons}); this invocation: "
                                                  f"{failed_runs}; the other runs went on; a new invocation resumes "
                                                  "the failed ones from their records", **extra})
        return EXIT_FAILED if failed_runs else EXIT_OK
    _record_stage(ctx, "pairwise", {"status": "done", "runs": all_runs, **extra})
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage: reasons
# ---------------------------------------------------------------------------------------------


def stage_reasons(ctx: Ctx6) -> int:
    """analysis_reasons_v5.classify_reasons_v5 with the classifier (classifier_spec) over every non-smoke verdict of
    run_dir/judge_calls.jsonl -> run_dir/reason_labels.jsonl (smoke-only records, verdicts with an empty reason and
    verdicts with no valid winner are left out and counted; a verdict already labelled is not classified again).

    check_config first. With reasons.enabled false the stage is recorded as skipped (pilot3._skip). When there is
    something new to classify, launch_gate for the classifier's provider and its credential decide whether it is
    launched; otherwise the stage is skipped with the reason (a stage recorded as done stays done). BudgetExceeded
    stops the stage (exit 3). Done: stages_status.json gets the counts. Dry run: the counts (smoke-only, verdicts,
    eligible, already labelled, to classify) and the gate's reason, if any; nothing is written."""
    from overclaim.analysis_reasons import LABEL_OK, reason_key
    from overclaim.analysis_reasons_v5 import classify_reasons_v5, eligible_records
    from overclaim.graders.judge import BudgetExceeded

    cfg = ctx.cfg
    check_config(cfg)
    spec = classifier_spec(cfg)
    if not cfg["reasons"]["enabled"]:
        if ctx.dry_run:
            ctx.note("reasons: reasons.enabled is false; nothing would be classified")
            return EXIT_OK
        return _skip(ctx, "reasons", "reasons.enabled is false")
    labels_path = ctx.run_dir / LABELS_FILE
    all_records = _read_jsonl(ctx.run_dir / JUDGE_FILE)
    records = [r for r in all_records if not r.get("smoke_only")]
    eligible, left_out = eligible_records(records)
    labelled = {reason_key(lab) for lab in _read_jsonl(labels_path)}
    new = sum(k not in labelled for k in eligible)
    provider = COST_PROVIDER[spec["provider"]]
    reason = launch_gate(ctx, {}, provider) if new else None
    counts = {"smoke_only": len(all_records) - len(records), "verdicts": len(records), "eligible": len(eligible),
              "no_reason": left_out["no_reason"],
              "invalid_winner": left_out["invalid_winner"], "duplicate_keys": left_out["duplicate"],
              "already_labelled": len(eligible) - new, "to_classify": new}
    if ctx.dry_run:
        ctx.note(f"reasons: {counts['verdicts']} verdicts in {JUDGE_FILE} ({counts['smoke_only']} smoke-only records "
                 f"left out), {counts['eligible']} with a reason and a "
                 f"valid winner ({counts['no_reason']} without a reason, {counts['invalid_winner']} with an invalid "
                 f"winner, {counts['duplicate_keys']} duplicate keys left out); {counts['already_labelled']} already "
                 f"labelled, {new} to classify with {spec['model']} (five fields, cites_disclosure added) -> "
                 f"{LABELS_FILE}" + (f"; SKIPPED: {reason}" if reason else ""))
        return EXIT_OK
    if not eligible:
        return _skip(ctx, "reasons", f"{JUDGE_FILE} holds no verdict with a reason and a valid winner", **counts)
    if reason is not None:
        return _skip(ctx, "reasons", reason, **counts)
    llm = ctx.client(spec) if new else None
    if new and llm is None:
        return _skip(ctx, "reasons", f"no credential for the classifier {spec['model']}", **counts)
    try:
        labels = classify_reasons_v5(records, llm, str(labels_path),
                                     max_workers=int(cfg.get("max_workers", {}).get(provider, 1)))
    except BudgetExceeded as exc:
        _record_stage(ctx, "reasons", {"status": "stopped", "reason": f"budget cap: {exc}"})
        return pilot2._budget_stop(ctx, "reasons", exc)
    _record_stage(ctx, "reasons", {"status": "done", **counts, "labels": len(labels),
                                   "invalid_labels": sum(lab.get("label") != LABEL_OK for lab in labels)})
    logger.info("reasons: %d labels (%d new); %s", len(labels), new, ctx.cost.totals_line())
    return EXIT_OK

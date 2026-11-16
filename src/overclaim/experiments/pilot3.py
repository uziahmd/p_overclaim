"""Pilot 3 stages (config schema_version 3, design v3): carryover -> sets -> judge -> rm -> e2 -> e3 -> judge_late ->
reasons -> analysis.

Entry point: `python -m overclaim.experiments.e1_preference --config configs/e1_pilot3_coding.yaml` (that module
dispatches here when the config has `schema_version: 3`).

The judge runs, the reward model, the reason classification, the shown-condition coverage and the budget stop are
pilot 2's code (overclaim.experiments.pilot2), run on the v3 sets through `Ctx3`. New here: the carry-over numbers
from session 2 (read only), the v3 sets with top-up, the dry runs of Experiments 2 and 3, and the projection for the
re-scoped full run. Every stage that spends Anthropic money, except `sets` (bounded by max_new_per_gap and the hard
cap), is not launched once the Anthropic total has reached budget.anthropic_stop_launching_usd; the skip is recorded
in run_dir/stages_status.json (judge runs: runs_status.json). Every stage is idempotent, supports --dry-run (no call,
no model loading) and deletes nothing. Session 2's directory is only read. Trajectory content is data.
Exit codes as pilot 2: 0 ok; 1 a stage failed or an input is missing; 2 a required credential is missing; 3 a budget
cap was hit.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

from overclaim.experiments import pilot2
from overclaim.experiments.pilot2 import (EXIT_FAILED, EXIT_NO_CREDENTIALS, EXIT_OK, _read_json, _read_jsonl,
                                          _repo_path, _write_json, _write_jsonl)

log = logging.getLogger(__name__)

STAGES = ("carryover", "sets", "judge", "rm", "e2", "e3", "judge_late", "reasons", "analysis")
SET = "v3"
GAP_TYPES = ("no_tests_run", "tests_failed", "tests_narrowed")
# Once one of these run files holds a record, the v3 sets are frozen: the sets stage only reports that.
FREEZE_FILES = ("judge_calls.jsonl", "rm_scores.jsonl", "e2_samples.jsonl", "e3_judge_calls.jsonl")
MODEL_KEYS = ("annotator", "verifier", "verifier_appeal", "e2_policy", "e2_labeller", "e3_max_tokens")   # beyond pilot 2's
SOURCE_PILOT3 = "pilot 3"
SOURCE_SESSION2 = "session 2 on the v2 sets"


# ---------------------------------------------------------------------------------------------
# Config, context and task lists
# ---------------------------------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    """pilot2.load_config (judges, editor, classifier, reward models, prices, rubrics; runs validated) plus the
    pilot-3 entries of configs/models.yaml (MODEL_KEYS). Also validates the e2 selectors, the e3 judge and rubric,
    `requires` flags and `task_sample`."""
    cfg = pilot2.load_config(path)
    with open(_repo_path(cfg["models_config"]), encoding="utf-8") as f:
        models = yaml.safe_load(f)
    for key in MODEL_KEYS:
        cfg[key] = models[key]
    for name in [*cfg["e2"]["selectors"], cfg["e3"]["judge"]]:
        if name not in cfg["judges"]:
            raise ValueError(f"unknown judge {name!r} in e2.selectors or e3.judge")
    if cfg["e3"]["rubric"] not in cfg["rubrics"]["rubrics"] or cfg["e3"]["set"] not in cfg["sets"]:
        raise ValueError("e3: unknown rubric or set")
    for run in cfg["runs"]:
        if "requires" in run and run["requires"] not in cfg:
            raise ValueError(f"run {run['id']}: requires unknown config flag {run['requires']!r}")
        if "task_sample" in run and run["condition"] != "shown":
            raise ValueError(f"run {run['id']}: task_sample is defined on the shown task order only")
    return cfg


def seeded_order(task_ids: list[str], seed: Any) -> list[str]:
    """sorted(task_ids) shuffled with random.Random(seed)."""
    order = sorted(task_ids)
    random.Random(seed).shuffle(order)
    return order


class Ctx3(pilot2.Ctx):
    """pilot2.Ctx with pilot 3's task lists."""

    def pilot_tasks(self) -> list[dict]:
        """The tasks that have a v3 set: pilot task_files (file order) then added_tasks_file, filtered to the task_ids
        of the v3 sets file. Before that file exists: every task of the pilot task_files."""
        pilot = self.cfg["pilot"]
        tasks = [t for name in pilot["task_files"] for t in _read_jsonl(self.processed / name)]
        if not self.set_path(SET).exists():
            return tasks
        ids = {ms["task_id"] for ms in _read_jsonl(self.set_path(SET))}
        tasks += _read_jsonl(self.processed / pilot["added_tasks_file"])
        return [t for t in tasks if t["task_id"] in ids]

    def shown_tasks(self) -> list[dict]:
        """pilot_tasks() whose rendered transcript is at or under shown.max_transcript_tokens."""
        cap = int(self.cfg["shown"]["max_transcript_tokens"])
        return [t for t in self.pilot_tasks() if t["token_counts"]["transcript"] <= cap]

    def sample_ids(self, n: int) -> list[str]:
        """The first n of seeded_order(shown task ids, cfg seed): the Opus shown run (task_sample) and Experiments 2
        and 3 all cut the same order, so a smaller sample is contained in a larger one."""
        return seeded_order([t["task_id"] for t in self.shown_tasks()], self.cfg["seed"])[:n]


def _record_stage(ctx: Ctx3, stage: str, entry: dict) -> None:
    path = ctx.run_dir / "stages_status.json"
    status = _read_json(path, {})
    status[stage] = entry
    _write_json(path, status)


def _skip(ctx: Ctx3, stage: str, reason: str, **extra: Any) -> int:
    """Record a stage as skipped (stages_status.json) and let the run continue. A stage that is already recorded as
    done stays done (a later invocation above the stop-launching line launches nothing and must not hide a finished
    stage); the reason is then kept under "not_relaunched"."""
    log.warning("%s: SKIPPED: %s", stage, reason)
    previous = _read_json(ctx.run_dir / "stages_status.json", {}).get(stage) or {}
    if previous.get("status") == "done":
        _record_stage(ctx, stage, {**previous, "not_relaunched": reason})
    else:
        _record_stage(ctx, stage, {"status": "skipped", "reason": reason, **extra})
    return EXIT_OK


def _has_record(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as f:
        return any(line.strip() for line in f)


def _tasks_in_order(tasks: list[dict], ids: list[str]) -> list[dict]:
    by_id = {t["task_id"]: t for t in tasks}
    return [by_id[i] for i in ids]


# ---------------------------------------------------------------------------------------------
# Stage: carryover (session 2's records, read only; no call)
# ---------------------------------------------------------------------------------------------


def vertex_smoke(cost_rows: list[dict]) -> dict:
    """The cost.jsonl rows whose route starts with "vertex": counts, token sums, the summed cost_usd (computed by the
    client from the list prices of configs/models.yaml) and one entry per call."""
    rows = [r for r in cost_rows if str(r.get("route") or "").startswith("vertex")]
    stamps = [r["ts"] for r in rows]
    return {"n_calls": len(rows), "route": ", ".join(sorted({r["route"] for r in rows})) or None,
            "model": ", ".join(sorted({str(r.get("model")) for r in rows})) or None,
            "first_ts": min(stamps) if stamps else None, "last_ts": max(stamps) if stamps else None,
            **{k: sum(int(r.get(k) or 0) for r in rows) for k in ("input_tokens", "output_tokens", "thought_tokens")},
            "list_price_usd": sum(float(r["cost_usd"]) for r in rows if r.get("cost_usd") is not None),
            "calls": [{k: r.get(k) for k in ("ts", "input_tokens", "output_tokens", "thought_tokens", "cost_usd")}
                      for r in rows]}


def stage_carryover(ctx: Ctx3) -> int:
    """Session 2's tie rates (judge_calls.jsonl, smoke records left out) -> run_dir/session2_tie_rates.json, and its
    vertex smoke-test calls (cost.jsonl) -> run_dir/vertex_smoke_session2.json. Nothing is re-run."""
    s2 = _repo_path(ctx.cfg["session2_dir"])
    if ctx.dry_run:
        ctx.note(f"carryover: would write session2_tie_rates.json and vertex_smoke_session2.json from {s2} (read only)")
        return EXIT_OK
    from overclaim import analysis

    missing = [str(s2 / n) for n in ("judge_calls.jsonl", "cost.jsonl") if not (s2 / n).exists()]
    if missing:
        log.error("carryover: missing session 2 files %s", missing)
        return EXIT_FAILED
    ties = analysis.tie_rates(_read_jsonl(s2 / "judge_calls.jsonl"))
    _write_json(ctx.run_dir / "session2_tie_rates.json", ties)
    smoke = vertex_smoke(_read_jsonl(s2 / "cost.jsonl"))
    _write_json(ctx.run_dir / "vertex_smoke_session2.json", smoke)
    log.info("carryover: %d tie-rate rows; %d vertex calls in session 2", len(ties), smoke["n_calls"])
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage: sets (the v3 matched sets, with top-up from the session-2 pool)
# ---------------------------------------------------------------------------------------------


def _candidate_tasks(tasks_path: Path, order: list[dict]) -> Iterator[dict]:
    """The candidate tasks in candidate order, each read from the pool file only when it is consumed."""
    from overclaim.data import coding

    for c in order:
        yield coding.read_task_at(tasks_path, c["offset"], c["length"])


def verify_with_appeal(task: dict, set_dict: dict, verifier: Any, appeal: Any) -> dict:
    """The post-check of one set (verify_variants.verify_set with the verifier client). A set that does not pass
    (a failed check or an unusable reply) gets one second look by the appeal client (purpose
    "v3_verification_appeal", appeal=True), whose result decides; the first result is kept under "first"."""
    from overclaim import verify_variants

    first = verify_variants.verify_set(task, set_dict, verifier)
    if first["ok"] or appeal is None:
        return first
    second = verify_variants.verify_set(task, set_dict, appeal, purpose="v3_verification_appeal", appeal=True)
    second["first"] = {k: first.get(k) for k in ("invalid", "ok", "checks", "failed", "answer", "model", "attempts")}
    return second


def appeal_counts(infos: list[dict]) -> dict:
    """How many sets went to the appeal, and how it ended: {"asked", "passed", "failed", "invalid", and per first
    failed check of the FIRST post-check ("first_failed_check": {name: {"asked", "passed"}})}."""
    out: dict[str, Any] = {"asked": 0, "passed": 0, "failed": 0, "invalid": 0, "first_failed_check": {}}
    for info in infos:
        post = (info.get("verifier") or {}).get("post") or {}
        first = post.get("first")
        if first is None:
            continue
        out["asked"] += 1
        out["invalid" if post["invalid"] else "passed" if post["ok"] else "failed"] += 1
        name = "invalid" if first["invalid"] else first["failed"][0]
        entry = out["first_failed_check"].setdefault(name, {"asked": 0, "passed": 0})
        entry["asked"] += 1
        entry["passed"] += int(bool(post["ok"]))
    return out


def stage_sets(ctx: Ctx3) -> int:
    """Build the v3 sets (variants_v3.build_all_v3): the pilot tasks first, then candidates per gap type. The verifier
    client runs the pre-check of the original message (verify_variants.verify_original, before any rewrite) and the
    post-check of the built set (verify_variants.verify_set).

    Every model call is cached on disk, so the stage rebuilds from the cache each time it runs (this is how
    max_new_per_gap is raised), unless a run file in FREEZE_FILES already holds a record: then the sets are frozen.
    Writes the sets, added-tasks and excluded files, run_dir/v3_build.jsonl, funnel.json["v3"] (other keys kept) and
    the human-check CSV (never over a file in which a human filled in a judgement).
    """
    cfg, pilot = ctx.cfg, ctx.cfg["pilot"]
    frozen = [name for name in FREEZE_FILES if _has_record(ctx.run_dir / name)]
    if frozen:
        message = f"sets: frozen ({', '.join(frozen)} already hold records); the v3 sets are not rebuilt"
        log.info(message)
        ctx.note(message)
        return EXIT_OK
    tasks_path = ctx.processed / cfg["data"]["tasks_file"]
    if ctx.dry_run:
        from overclaim.data import coding

        raw = _repo_path(coding.DEFAULT_RAW_DIR)   # where Module B's default patch_fn reads each task's patch
        ctx.note(f"sets: would build the v3 sets: pilot tasks of {pilot['task_files']} first, then candidates from "
                 f"{tasks_path.name} ({'present' if tasks_path.exists() else 'MISSING'}) until {pilot['n_per_gap']} "
                 f"sets per gap type or {pilot['max_new_per_gap']} candidates; detection {cfg['editor']['model']} "
                 f"(disk cache), annotation with missed claims {cfg['annotator']['model']}, pre-check of the original "
                 f"message (checks a and b, before any rewrite) {cfg['verifier']['model']}, rewrites of content spans "
                 f"{cfg['editor']['model']} (with each task's patch from {raw}: "
                 f"{'present' if raw.is_dir() else 'MISSING'}), post-check of the built set (checks c, d and f) "
                 f"{cfg['verifier']['model']}; "
                 f"ANTHROPIC_API_KEY {'set' if pilot2.credential_for({'provider': 'anthropic'}) else 'MISSING'}")
        return EXIT_OK
    from overclaim import variants_v3, verify_variants
    from overclaim.data import coding
    from overclaim.graders.judge import BudgetExceeded

    detector, annotator, verifier = (ctx.client(cfg[key]) for key in ("editor", "annotator", "verifier"))
    appeal = ctx.client(cfg["verifier_appeal"]) if cfg.get("verifier_appeal") else None
    if detector is None or annotator is None or verifier is None:
        log.error("sets: ANTHROPIC_API_KEY is not set; the v3 sets cannot be built")
        return EXIT_NO_CREDENTIALS
    if not tasks_path.exists():
        log.error("sets: the candidate pool %s is missing", tasks_path)
        return EXIT_FAILED
    vcfg, bank, lint_patterns = variants_v3.load_v3_config(_repo_path(cfg["variants_config"]))
    pilot_tasks = [t for name in pilot["task_files"] for t in _read_jsonl(ctx.processed / name)]
    exclude = {t["instance_id"] for t in pilot_tasks}
    # The tasks the prompts were checked on before the build (pilot.held_out_task_ids) never become candidates.
    held_out = set(pilot.get("held_out_task_ids") or [])
    orders = {gap: coding.candidate_order(tasks_path, gap, cfg["seed"], pilot["prefer_resolved"],
                                          exclude_instance_ids=exclude) for gap in GAP_TYPES}
    candidates = {gap: _candidate_tasks(tasks_path, [c for c in order if c["task_id"] not in held_out])
                  for gap, order in orders.items()}
    funnel = _read_json(ctx.funnel_path, {})
    funnel["v3_candidate_pool"] = {gap: {"candidates": len(order),
                                         "held_out_removed": sum(c["task_id"] in held_out for c in order)}
                                   for gap, order in orders.items()}
    try:
        sets, added, excluded, infos = variants_v3.build_all_v3(
            pilot_tasks, candidates, vcfg, bank, lint_patterns, detector, annotator,
            lambda task, set_dict: verify_with_appeal(task, set_dict, verifier, appeal), funnel,
            n_per_gap=int(pilot["n_per_gap"]), max_new_per_gap=pilot.get("max_new_per_gap"),
            precheck_fn=lambda task, message, spans: verify_variants.verify_original(task, message, spans, verifier))
    except BudgetExceeded as exc:
        _write_json(ctx.funnel_path, funnel)
        return pilot2._budget_stop(ctx, "sets", exc)
    _write_jsonl(ctx.set_path(SET), sets)
    _write_jsonl(ctx.processed / pilot["added_tasks_file"], added)
    _write_jsonl(ctx.processed / pilot["excluded_file"], excluded)
    _write_jsonl(ctx.run_dir / "v3_build.jsonl", infos)
    funnel["v3"]["appeal"] = appeal_counts(infos)
    _write_json(ctx.funnel_path, funnel)
    check = ctx.processed / cfg["sets"][SET]["human_check"]
    try:
        log.info("sets: wrote %d rows to %s", variants_v3.write_human_check_v3(check, pilot_tasks + added, sets), check)
    except FileExistsError as exc:  # a human has filled in judgements: the file is kept as it is
        log.warning("sets: %s", exc)
    log.info("sets: %d v3 sets (%d added tasks, %d excluded)", len(sets), len(added), len(excluded))
    print(f"[sets] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stages: judge and judge_late (pilot2.stage_judge on the selected runs)
# ---------------------------------------------------------------------------------------------


def _stage_runs(ctx: Ctx3, late: bool, only: list[str] | None) -> int:
    """The runs with (late) or without `late: true`, restricted to `only`, through pilot2.stage_judge.

    A run whose `requires` flag is false is recorded as skipped in runs_status.json and not launched; `task_sample: n`
    becomes run["task_ids"] = the first n of the seeded shown order. When an OpenRouter run was launched,
    openrouter_usage.jsonl is rewritten afterwards with the cache entries of every OpenRouter model of the config
    created since the first line of run_dir/cost.jsonl (session 2's entries are left out).
    """
    cfg = ctx.cfg
    if only is not None:
        unknown = sorted(set(only) - {r["id"] for r in cfg["runs"]})
        if unknown:
            log.error("judge: unknown run ids %s", unknown)
            return EXIT_FAILED
    runs = [r for r in cfg["runs"] if bool(r.get("late", False)) == late and (only is None or r["id"] in only)]
    other = sorted({r["id"] for r in cfg["runs"] if bool(r.get("late", False)) != late and r["id"] in (only or [])})
    if other:  # e.g. `--stage judge --runs o46-v3-neutral`: say where those runs are launched instead of doing nothing
        message = (f"{'judge_late' if late else 'judge'}: runs {other} belong to the stage "
                   f"{'judge' if late else 'judge_late'} and are not launched here")
        log.warning(message)
        ctx.note(message)
    blocked = [r for r in runs if "requires" in r and not cfg[r["requires"]]]
    for run in runs:
        if "task_sample" in run:
            run["task_ids"] = ctx.sample_ids(int(run["task_sample"]))
            ctx.note(f"judge: run {run['id']}: task_sample {run['task_sample']} -> {run['task_ids']}")
    status_path = ctx.run_dir / "runs_status.json"
    status = _read_json(status_path, {})
    for run in blocked:
        reason = f"config {run['requires']} is false; set it to true to launch this run"
        ctx.note(f"judge: run {run['id']}: SKIPPED: {reason}")
        log.warning("judge: run %s SKIPPED: %s", run["id"], reason)
        status[run["id"]] = {"judge": run.get("grader_label", run["judge"]), "set": run["set"], "rubric": run["rubric"],
                             "condition": run["condition"], "status": "skipped", "reason": reason}
    if blocked and not ctx.dry_run:
        _write_json(status_path, status)
    launch = [r for r in runs if r not in blocked]
    code = pilot2.stage_judge(ctx, [r["id"] for r in launch])
    if code != EXIT_OK or ctx.dry_run:
        return code
    slugs = sorted({pilot2.run_spec(cfg, r)["model"] for r in cfg["runs"]
                    if pilot2.run_spec(cfg, r)["provider"] == "openrouter"})
    if any(pilot2.run_spec(cfg, r)["provider"] == "openrouter" for r in launch):
        cost_rows = _read_jsonl(ctx.run_dir / "cost.jsonl")
        since = cost_rows[0]["ts"] if cost_rows else datetime.now(timezone.utc).isoformat()
        n = pilot2.export_openrouter_usage(ctx, slugs, since=since)
        log.info("judge: wrote %d OpenRouter usage objects created since %s to openrouter_usage.jsonl", n, since)
    return EXIT_OK


def stage_judge(ctx: Ctx3, only: list[str] | None = None) -> int:
    """The judge runs without `late: true`, in config order."""
    return _stage_runs(ctx, False, only)


def stage_judge_late(ctx: Ctx3, only: list[str] | None = None) -> int:
    """The judge runs with `late: true` (the first cut of the full-run order), in config order."""
    return _stage_runs(ctx, True, only)


# ---------------------------------------------------------------------------------------------
# Stages: rm, e2, e3, reasons
# ---------------------------------------------------------------------------------------------


def stage_rm(ctx: Ctx3) -> int:
    """pilot2.stage_rm on the v3 sets (rm.sets, hidden). No Anthropic money: not gated."""
    return pilot2.stage_rm(ctx)


def _sets_of(ctx: Ctx3, set_name: str, ids: list[str]) -> list[dict]:
    wanted = set(ids)
    return [ms for ms in _read_jsonl(ctx.set_path(set_name)) if ms["task_id"] in wanted]


def stage_e2(ctx: Ctx3) -> int:
    """Experiment 2 dry run (e2_selection.run_e2) on the first e2.n_tasks of the seeded shown order; the labeller is
    validated first on EVERY v3 set (every task that has one). Writes Module D's files into run_dir,
    run_dir/e2_summary.json (run_e2's return value, NaN as null) and stages_status.json["e2"]."""
    cfg, e2 = ctx.cfg, ctx.cfg["e2"]
    ids = ctx.sample_ids(int(e2["n_tasks"]))
    if ctx.dry_run:
        ctx.note(f"e2: would validate the labeller {cfg['e2_labeller']['model']} on every v3 set, then run "
                 f"Experiment 2 on {len(ids)} tasks {ids} x {e2['n_samples']} samples: policy "
                 f"{cfg['e2_policy']['model']}, labeller {cfg['e2_labeller']['model']}, selectors {e2['selectors']}")
        return EXIT_OK
    if not _has_record(ctx.set_path(SET)):
        log.error("e2: %s is missing or empty; run the sets stage first", ctx.set_path(SET))
        return EXIT_FAILED
    reason = pilot2.gate(ctx, {}, "anthropic")
    if reason is not None:
        return _skip(ctx, "e2", reason)
    from overclaim import analysis
    from overclaim.experiments import e2_selection
    from overclaim.graders.judge import BudgetExceeded, CacheNotRead

    if tuple(e2["ns"]) != tuple(e2_selection.NS):
        log.error("e2: e2.ns %s differs from the N values e2_selection computes %s", e2["ns"], e2_selection.NS)
        return EXIT_FAILED
    clients = {"policy": ctx.client(cfg["e2_policy"]), "labeller": ctx.client(cfg["e2_labeller"]),
               **{name: ctx.client(cfg["judges"][name]) for name in e2["selectors"]}}
    missing = [k for k, c in clients.items() if c is None]
    if missing:
        return _skip(ctx, "e2", f"no credential for the clients {missing}")
    with_set = ctx.pilot_tasks()   # every task that has a v3 set
    tasks = _tasks_in_order(with_set, ids)
    try:
        result = e2_selection.run_e2(tasks, _read_jsonl(ctx.set_path(SET)), clients, ctx.run_dir,
                                     validation_tasks=with_set, n_samples=int(e2["n_samples"]),
                                     selectors=list(e2["selectors"]), max_workers=int(cfg["max_workers"]["anthropic"]),
                                     stop_below=float(e2["stop_below"]))
    except BudgetExceeded as exc:
        _record_stage(ctx, "e2", {"status": "stopped", "reason": f"budget cap: {exc}"})
        return pilot2._budget_stop(ctx, "e2", exc)
    except CacheNotRead as exc:  # a policy call of a task read nothing from the prompt cache (as e3 and pilot 2)
        _record_stage(ctx, "e2", {"status": "failed", "reason": str(exc)})
        log.error("e2: FAILED: %s", exc)
        return EXIT_FAILED
    _write_json(ctx.run_dir / "e2_summary.json", analysis._nan_to_none(result))
    status = "stopped_labeller_validation" if result["status"] == "stopped_labeller_validation" else "done"
    _record_stage(ctx, "e2", {"status": status, "n_tasks": len(tasks), "task_ids": ids})
    log.info("e2: %s", status)
    print(f"[e2] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK


def stage_e3(ctx: Ctx3) -> int:
    """Experiment 3 dry run: the verification-first judge (e3_verification.VerificationFirstJudge, shown, prompt
    cache) on the first e3.n_tasks of the seeded shown order -> run_dir/e3_judge_calls.jsonl, then e3_table against
    the plain judge's shown records of the same tasks, set and rubric -> run_dir/e3_summary.json (NaN as null)."""
    cfg, e3 = ctx.cfg, ctx.cfg["e3"]
    ids = ctx.sample_ids(int(e3["n_tasks"]))
    grader = f"{e3['judge']}+verify_first"
    if ctx.dry_run:
        ctx.note(f"e3: would run {grader} (max_tokens {cfg['e3_max_tokens']}), set {e3['set']}, rubric "
                 f"{e3['rubric']}, shown, on {len(ids)} tasks {ids}")
        return EXIT_OK
    if not _has_record(ctx.set_path(e3["set"])):
        log.error("e3: %s is missing or empty; run the sets stage first", ctx.set_path(e3["set"]))
        return EXIT_FAILED
    reason = pilot2.gate(ctx, {}, "anthropic")
    if reason is not None:
        return _skip(ctx, "e3", reason)
    from overclaim import analysis
    from overclaim.experiments import e3_verification
    from overclaim.graders import judge as J

    client = ctx.client({**cfg["judges"][e3["judge"]], "max_tokens": int(cfg["e3_max_tokens"])})
    if client is None:
        return _skip(ctx, "e3", f"no credential for {e3['judge']}")
    vf = e3_verification.VerificationFirstJudge(client, grader, rubric=e3["rubric"],
                                                question=cfg["rubrics"]["rubrics"][e3["rubric"]], structured=True)
    tasks = _tasks_in_order(ctx.pilot_tasks(), ids)
    try:
        records = J.run_pairs2(tasks, _sets_of(ctx, e3["set"], ids), vf, cfg["pairs"], "shown",
                               str(ctx.run_dir / "e3_judge_calls.jsonl"), cfg["run_name"], set_name=e3["set"],
                               max_workers=int(cfg["max_workers"]["anthropic"]), use_prompt_cache=True,
                               extra_fields=("claims_A", "claims_B"))
    except J.BudgetExceeded as exc:
        _record_stage(ctx, "e3", {"status": "stopped", "reason": f"budget cap: {exc}"})
        return pilot2._budget_stop(ctx, "e3", exc)
    except J.CacheNotRead as exc:
        _record_stage(ctx, "e3", {"status": "failed", "reason": str(exc)})
        log.error("e3: FAILED: %s", exc)
        return EXIT_FAILED
    wanted = set(ids)
    plain = [r for r in _read_jsonl(ctx.run_dir / "judge_calls.jsonl")
             if r["grader"] == e3["judge"] and r.get("set") == e3["set"] and r.get("rubric") == e3["rubric"]
             and r["condition"] == "shown" and r["task_id"] in wanted]
    data, markdown = e3_verification.e3_table(records, plain, cfg["pairs"], n_boot=int(cfg["analysis"]["n_bootstrap"]),
                                              seed=int(cfg["analysis"]["seed"]))
    _write_json(ctx.run_dir / "e3_summary.json", analysis._nan_to_none(
                {"status": "done", "judge": e3["judge"], "grader": grader, "set": e3["set"], "rubric": e3["rubric"],
                 "n_tasks": len(tasks), "task_ids": ids, "n_records": len(records), "n_plain_records": len(plain),
                 "data": data, "markdown": markdown}))
    _record_stage(ctx, "e3", {"status": "done", "n_tasks": len(tasks), "task_ids": ids})
    print(f"[e3] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK


def stage_reasons(ctx: Ctx3) -> int:
    """pilot2.stage_reasons (every verdict of run_dir/judge_calls.jsonl), behind the stop-launching gate."""
    if not ctx.dry_run and ctx.cfg.get("reasons", {}).get("enabled", True):
        reason = pilot2.gate(ctx, {}, "anthropic")
        if reason is not None:  # say how many verdicts already carry a label from an earlier invocation
            verdicts = sum(1 for r in _read_jsonl(ctx.run_dir / "judge_calls.jsonl") if not r.get("smoke_only"))
            labels = len(_read_jsonl(ctx.run_dir / "reason_labels.jsonl"))
            return _skip(ctx, "reasons", reason, verdicts=verdicts, labels_present=labels)
    return pilot2.stage_reasons(ctx)


# ---------------------------------------------------------------------------------------------
# Stage: analysis (summary + the re-scoped projection)
# ---------------------------------------------------------------------------------------------


def _pseudo_records(path: Path, condition: str, name: Any) -> list[dict]:
    """unit_stats-shaped records from an Experiment 2 file: grader = name(record), the record's cost and usage."""
    return [{"grader": name(r), "condition": condition, "cost_usd": r.get("cost_usd"), "cached": r.get("cached"),
             "usage": r.get("usage")} for r in _read_jsonl(path)]


def rescoped_stats(ctx: Ctx3) -> dict:
    """{(name, condition): {"n", "usd", "tokens", "source"}} for project_rescoped: session 2's judge records on the v2
    sets, overlaid by pilot 3's (judge_calls.jsonl and e3_judge_calls.jsonl on the v3 sets) and by the Experiment 2
    pseudo-records (e2_policy shown, e2_selector:<name> hidden, e2_label hidden). Live calls only (unit_stats)."""
    from overclaim import projection

    thought = tuple(ctx.cfg["projection"].get("thought_separate") or ())
    s2 = _read_jsonl(_repo_path(ctx.cfg["session2_dir"]) / "judge_calls.jsonl")
    stats = {k: {**v, "source": SOURCE_SESSION2}
             for k, v in projection.unit_stats(s2, sets=("v2",), thought_separate=thought).items()}
    run = ctx.run_dir
    p3 = _read_jsonl(run / "judge_calls.jsonl") + _read_jsonl(run / "e3_judge_calls.jsonl")
    pseudo = (_pseudo_records(run / "e2_samples.jsonl", "shown", lambda r: "e2_policy")
              + _pseudo_records(run / "e2_scores.jsonl", "hidden", lambda r: f"e2_selector:{r['selector']}")
              + _pseudo_records(run / "e2_labels.jsonl", "hidden", lambda r: "e2_label"))
    for k, v in [*projection.unit_stats(p3, sets=(SET,), thought_separate=thought).items(),
                 *projection.unit_stats(pseudo).items()]:
        stats[k] = {**v, "source": SOURCE_PILOT3}
    return stats


def projection_section(ctx: Ctx3) -> dict:
    """The re-scoped full-run projection (projection.project_rescoped) as a summary section."""
    from overclaim import projection

    result = projection.project_rescoped(ctx.cfg, rescoped_stats(ctx), ctx.cost.spent("anthropic"))
    lines = ["## Cost projection for the re-scoped full run", "",
             "Arithmetic on measured cost per live call; nothing was run for it. 150 tasks = 90 coding + 60 research "
             "reports; the research-report domain is not built, so its per-call cost is ASSUMED equal to coding. "
             "Cuts 1 to 3 of the cut order (session-3 brief) are already applied in the items: the 4.6 pair runs hidden "
             "only, the mitigated best-of-16 is optional (listed, not added to any total), and the shown runs use "
             "120 tasks. Unit costs: pilot 3's measurement on the v3 sets where there is one, else session 2's on "
             "the v2 sets, else an estimate from the reference judge's token profile (basis column). Experiment 2 "
             "and 3 unit costs come from this pilot's dry runs. Costs of building the matched sets and of reason "
             "classification are not in these tables.", "", projection.rescoped_markdown(result).rstrip("\n")]
    return {"key": "projection_rescoped", "markdown": "\n".join(lines) + "\n", "data": result}


def stage_analysis(ctx: Ctx3) -> int:
    """analysis.summarize_pilot3 -> run_dir/summary.md and summary.json, printed; the shown-condition coverage and
    the projection are passed as extra sections."""
    if ctx.dry_run:
        ctx.note("analysis: would write summary.md and summary.json")
        return EXIT_OK
    from overclaim import analysis

    extra = [pilot2.shown_section(ctx.cfg, ctx.pilot_tasks(), _read_json(ctx.run_dir / "runs_status.json", {}))]
    if ctx.cfg.get("projection"):
        extra.append(projection_section(ctx))
    print(analysis.summarize_pilot3(str(ctx.run_dir), ctx.cfg, extra_sections=extra))
    return EXIT_OK


_STAGE_FUNCS = {"carryover": stage_carryover, "sets": stage_sets, "rm": stage_rm, "e2": stage_e2, "e3": stage_e3,
                "reasons": stage_reasons, "analysis": stage_analysis}
_RUN_STAGES = {"judge": stage_judge, "judge_late": stage_judge_late}


def run(cfg: dict, cfg_path: Path, run_dir: Path, stage: str, dry_run: bool, only_runs: list[str] | None = None) -> int:
    """Run one stage or all of them (STAGES order); returns the exit code."""
    if stage != "all" and stage not in STAGES:
        log.error("unknown pilot-3 stage %r; expected one of %s", stage, STAGES)
        return EXIT_FAILED
    ctx = Ctx3(cfg, cfg_path, run_dir, dry_run)
    resolved = dict(cfg)
    resolved["source_config"] = str(cfg_path)
    resolved["resolved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
                                         encoding="utf-8")
    for name in STAGES if stage == "all" else (stage,):
        log.info("--- stage %s%s ---", name, " (dry run)" if dry_run else "")
        code = _RUN_STAGES[name](ctx, only_runs) if name in _RUN_STAGES else _STAGE_FUNCS[name](ctx)
        if code != EXIT_OK:
            log.error("run stopped at stage %r with exit code %d", name, code)
            return code
    if dry_run:
        print("=== pilot 3 dry-run plan (no API calls, no model loading) ===")
        for line in ctx.plan:
            print(line)
    elif ctx._cost is not None:
        print(ctx.cost.totals_line())
    return EXIT_OK

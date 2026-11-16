"""Pilot 4 stages (config schema_version 4, design v4): carryover -> pool -> sets -> judge -> reasons -> e2 -> e3 ->
pointwise -> rm -> pointwise_late -> analysis.

Entry point: `python -m overclaim.experiments.e1_preference --config configs/e1_pilot4_coding.yaml` (that module
dispatches here when the config has `schema_version: 4`).

The stage order is the budget priority: every stage or run that spends Anthropic money is not launched once the
Anthropic total of run_dir/cost.jsonl has reached budget.anthropic_stop_launching_usd (pilot2.gate), and the skip is
recorded with its reason in run_dir/stages_status.json (judge and pointwise runs: runs_status.json); a stage recorded
as done stays done (pilot3._skip). Unlike pilot 3, the sets stage is gated as well. The judge runs, the reward model
and the bookkeeping are pilot 2's and pilot 3's code, run on the v4 sets through `Ctx4`. New here: the carry-over
numbers from pilot 3 (read only), the extra no_tests_run pool, the v4 sets with top-up and rebalancing, the pointwise
reading (Experiment 1b, also as a hook after the first tasks of a judge run), the four-label reason classification, the
Experiment 2 policy pilot on the evidence packet and the three-label Experiment 3 check.

Every stage is idempotent (outputs that exist are reused, every call is cached on disk), supports --dry-run (no call,
no model loading; the plan names what each stage would do) and deletes nothing. The directories of pilots 1 to 3 and
the v3 data files are only read. Trajectory content is data. Exit codes as pilot 2: 0 ok; 1 a stage failed or an
input is missing; 2 a required credential is missing; 3 a budget cap was hit.
"""

from __future__ import annotations

import inspect
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml

from overclaim.experiments import pilot2, pilot3
from overclaim.experiments.pilot2 import (EXIT_FAILED, EXIT_NO_CREDENTIALS, EXIT_OK, _read_json, _read_jsonl,
                                          _repo_path, _write_json, _write_jsonl)
from overclaim.experiments.pilot3 import _has_record, _record_stage, _skip, _tasks_in_order, seeded_order

logger = logging.getLogger(__name__)

STAGES = ("carryover", "pool", "sets", "judge", "reasons", "e2", "e3", "pointwise", "rm", "pointwise_late",
          "analysis")
SET = "v4"
V3_SET = "v3"
GAP_TYPES = pilot3.GAP_TYPES
# Once one of these run files holds a record, the v4 sets are frozen: the sets stage only reports that.
FREEZE_FILES = ("judge_calls.jsonl", "pointwise_calls.jsonl", "rm_scores.jsonl", "e2_samples.jsonl")
# configs/models.yaml entries beyond pilot 3's (load_config also takes the entries named by e2.policy and e2.labeller)
MODEL_KEYS = ("precheck_v4", "verifier_v4")
TASK_RULES = ("no_tests_run_or_primary_changed",)
REMAINDER_GAP = "tests_failed"                # an odd remainder of the rebalancing goes here (the larger pool)
CARRYOVER_GRADER = "claude-opus-4-6"         # pilot 3's late run, whose reasons pilot 3 did not classify
POINTWISE_FILE = "pointwise_calls.jsonl"
REBUILD_REPORT_FILE = "v4_rebuild_report.json"
PROBE_NOTE = ("The cost logs hold no record of session 1's five OpenRouter connectivity probes (under 0.01 USD in "
              "total); they are not in these sums.")


# ---------------------------------------------------------------------------------------------
# Config, context and task lists
# ---------------------------------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    """pilot3.load_config (judges, editor, classifier, reward models, prices, rubrics, the pilot-3 model entries;
    runs, e2 selectors, e3 judge, rubric and set, `requires` flags validated) plus the pilot-4 entries of
    configs/models.yaml: MODEL_KEYS and the entries named by e2.policy and e2.labeller, each under its own name.
    Also validates every `task_rule`, every `pointwise_after_tasks` (a positive integer, on a run with
    pointwise_after), the pointwise block (set, condition, judges) and that run ids are unique over the judge runs and
    the pointwise runs together (they share runs_status.json)."""
    cfg = pilot3.load_config(path)
    with open(_repo_path(cfg["models_config"]), encoding="utf-8") as f:
        models = yaml.safe_load(f)
    for key in (*MODEL_KEYS, cfg["e2"]["policy"], cfg["e2"]["labeller"]):
        if key not in models:
            raise ValueError(f"{key!r} is not an entry of {cfg['models_config']}")
        cfg[key] = models[key]
    for run in cfg["runs"]:
        if "task_rule" in run and run["task_rule"] not in TASK_RULES:
            raise ValueError(f"run {run['id']}: unknown task_rule {run['task_rule']!r} (known: {TASK_RULES})")
        limit = run.get("pointwise_after_tasks")
        if limit is not None and (not run.get("pointwise_after") or isinstance(limit, bool)
                                  or not isinstance(limit, int) or limit < 1):
            raise ValueError(f"run {run['id']}: pointwise_after_tasks must be a positive integer on a run with "
                             "pointwise_after: true")
    pw = cfg["pointwise"]
    if pw["set"] not in cfg["sets"] or pw["condition"] not in ("hidden", "shown"):
        raise ValueError(f"pointwise: unknown set {pw['set']!r} or condition {pw['condition']!r}")
    for run in pw["runs"]:
        if run["judge"] not in cfg["judges"]:
            raise ValueError(f"pointwise run {run['id']}: unknown judge {run['judge']!r}")
    ids = [r["id"] for r in cfg["runs"]] + [r["id"] for r in pw["runs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("run ids must be unique over runs and pointwise.runs")
    return cfg


class Ctx4(pilot3.Ctx3):
    """pilot3.Ctx3 with pilot 4's task lists and the pointwise hook of the judge stage."""

    def first_tasks(self) -> list[dict]:
        """The tasks of pilot.first_task_files whose task_id is in pilot.first_sets_file (the v3 sets), in that file's
        order: the tasks that have a v3 set, rebuilt first. A set without a task raises ValueError."""
        pilot = self.cfg["pilot"]
        tasks = [t for name in pilot["first_task_files"] for t in _read_jsonl(self.processed / name)]
        ids = [ms["task_id"] for ms in _read_jsonl(self.processed / pilot["first_sets_file"])]
        missing = sorted(set(ids) - {t["task_id"] for t in tasks})
        if missing:
            raise ValueError(f"{len(missing)} sets of {pilot['first_sets_file']} have no task in "
                             f"{pilot['first_task_files']}, e.g. {missing[:3]}")
        return _tasks_in_order(tasks, ids)

    def pilot_tasks(self) -> list[dict]:
        """The tasks that have a v4 set: first_tasks() (first-sets order) then pilot.added_tasks_file (file order),
        filtered to the task_ids of the v4 sets file. Before that file exists: first_tasks()."""
        first = self.first_tasks()
        if not self.set_path(SET).exists():
            return first
        ids = {ms["task_id"] for ms in _read_jsonl(self.set_path(SET))}
        tasks = first + _read_jsonl(self.processed / self.cfg["pilot"]["added_tasks_file"])
        return [t for t in tasks if t["task_id"] in ids]

    def after_task_for(self, run: dict) -> Callable[[dict, dict, int], None] | None:
        """The after_task hook that pilot2.stage_judge passes to judge.run_pairs2 for `run`: None unless the run has
        `pointwise_after: true`. The hook scores O, H and P of the task whose pairwise calls just ended with
        e1b_pointwise.score_task: a client from the run's judge spec, grader = the run's label, the run's condition and
        set, structured = the spec's structured_output, use_prompt_cache as pilot2.stage_judge uses it for the run
        (shown condition and an Anthropic judge), records appended to run_dir/pointwise_calls.jsonl.

        live_before is 0 whatever run_pairs2 passes as the task's live pairwise calls: the pairwise calls send the
        verdict schema and the pointwise calls the score schema, and a structured-output request with another schema
        may not share their prompt-cache entry, so the first pointwise call of a task may write its own entry and
        score_task requires a cache read from the second call on. With `pointwise_after_tasks: N` the hook scores only
        the first N tasks of the run's task list (pilot2.run_tasks over pilot_tasks() and the run's sets file, the list
        pilot2.stage_judge passes to run_pairs2, in its order) and does nothing for the other tasks; without the key
        it scores every task."""
        if not run.get("pointwise_after"):
            return None
        from overclaim.experiments import e1b_pointwise

        spec = pilot2.run_spec(self.cfg, run)
        client = self.client(spec)
        label = run.get("grader_label", run["judge"])
        use_cache = run["condition"] == "shown" and spec["provider"] == "anthropic"
        out_path = str(self.run_dir / POINTWISE_FILE)
        limit = run.get("pointwise_after_tasks")
        allowed = None
        if limit is not None:
            todo = pilot2.run_tasks(self, run, self.pilot_tasks(), _read_jsonl(self.set_path(run["set"])))
            allowed = {t["task_id"] for t in todo[: int(limit)]}

        def hook(task: dict, matched_set: dict, n_live_calls: int) -> None:
            if allowed is not None and task["task_id"] not in allowed:
                return
            e1b_pointwise.score_task(task, matched_set, client, label, run["condition"], out_path, self.cfg["run_name"],
                                     set_name=run["set"], structured=bool(spec.get("structured_output")),
                                     use_prompt_cache=use_cache, live_before=0)

        return hook


# ---------------------------------------------------------------------------------------------
# Stage: carryover (pilot 3's records and the earlier cost logs, read only)
# ---------------------------------------------------------------------------------------------


def prior_spend(cost_logs: list[str]) -> dict:
    """Spend before this session from the given cost.jsonl files (read only): per file whether it exists and, per
    provider, costs._spend of its lines (calls, token buckets, usd = sum of non-null cost_usd, n_unpriced); the same
    per provider over all files; and PROBE_NOTE."""
    from overclaim import costs

    files: dict[str, dict] = {}
    pooled: dict[str, list[dict]] = {}
    for name in cost_logs:
        path = _repo_path(name)
        if not path.exists():
            logger.warning("carryover: prior cost log %s is missing; it adds nothing to the sums", path)
        groups: dict[str, list[dict]] = {}
        for row in costs._read_jsonl(path):
            groups.setdefault(str(row.get("provider")), []).append(row)
        files[name] = {"present": path.exists(),
                       "by_provider": {p: costs._spend(rows) for p, rows in sorted(groups.items())}}
        for p, rows in groups.items():
            pooled.setdefault(p, []).extend(rows)
    return {"files": files, "by_provider": {p: costs._spend(rows) for p, rows in sorted(pooled.items())},
            "note": PROBE_NOTE}


def stage_carryover(ctx: Ctx4) -> int:
    """From pilot 3's run directory (read only): (c) the spend before this session, per provider and per file of
    cfg["prior_cost_logs"] -> run_dir/prior_spend.json, one printed line per provider; (b) analysis_v4.judge_table
    over pilot 3's judge records -> run_dir/pilot3_decided_rates.json and .md; then, behind the stop-launching gate,
    (a) the reasons of pilot 3's CARRYOVER_GRADER verdicts classified with the pilot-3 prompt
    (analysis_reasons.classify_reasons, classifier client) -> run_dir/pilot3_opus46_reason_labels.jsonl, and
    analysis_reasons.reason_table over pilot 3's judge records with pilot 3's labels plus the new ones ->
    run_dir/pilot3_reason_table.json and .md. (b) and (c) make no call and are written before the gate."""
    cfg = ctx.cfg
    s3 = _repo_path(cfg["session3_dir"])
    if ctx.dry_run:
        ctx.note(f"carryover: would write prior_spend.json from {cfg['prior_cost_logs']}, pilot3_decided_rates.json "
                 f"from {s3 / 'judge_calls.jsonl'} (read only), then classify the reasons of pilot 3's "
                 f"{CARRYOVER_GRADER} verdicts with {cfg['classifier']['model']} (pilot-3 prompt) and write "
                 "pilot3_reason_table.json with that row added")
        return EXIT_OK
    missing = [str(s3 / n) for n in ("judge_calls.jsonl", "reason_labels.jsonl") if not (s3 / n).exists()]
    if missing:
        logger.error("carryover: missing pilot 3 files %s", missing)
        return EXIT_FAILED
    from overclaim import analysis, analysis_reasons, analysis_v4
    from overclaim.graders.judge import BudgetExceeded

    spend = prior_spend(list(cfg["prior_cost_logs"]))
    _write_json(ctx.run_dir / "prior_spend.json", spend)
    for provider, s in spend["by_provider"].items():
        print(f"[carryover] spend before this session, {provider}: ${s['usd']:.4f} over {s['calls']} logged calls"
              + (f" ({s['n_unpriced']} unpriced)" if s["n_unpriced"] else ""), flush=True)
    judge_records = _read_jsonl(s3 / "judge_calls.jsonl")
    n_boot, seed = int(cfg["analysis"]["n_bootstrap"]), int(cfg["analysis"]["seed"])
    rows, md_lines = analysis_v4.judge_table(judge_records, [list(p) for p in cfg["pairs"]], n_boot, seed)
    source = str(s3 / "judge_calls.jsonl")
    _write_json(ctx.run_dir / "pilot3_decided_rates.json",
                analysis._nan_to_none({"source": source, "n_records": len(judge_records), "rows": rows}))
    (ctx.run_dir / "pilot3_decided_rates.md").write_text(
        "\n".join(["# Pilot 3 judge records: tie-inclusive and decided-only rates", "", f"Source: {source}.", "",
                   *md_lines]) + "\n", encoding="utf-8")

    reason = pilot2.gate(ctx, {}, "anthropic")
    if reason is not None:
        return _skip(ctx, "carryover", reason, written=["prior_spend.json", "pilot3_decided_rates.json"])
    llm = ctx.client(cfg["classifier"])
    if llm is None:
        return _skip(ctx, "carryover", f"no credential for the classifier {cfg['classifier']['model']}",
                     written=["prior_spend.json", "pilot3_decided_rates.json"])
    late = [r for r in judge_records if r["grader"] == CARRYOVER_GRADER and not r.get("smoke_only")]
    logger.info("carryover: %d %s verdicts of pilot 3 to classify", len(late), CARRYOVER_GRADER)
    try:
        added = analysis_reasons.classify_reasons(late, llm, str(ctx.run_dir / "pilot3_opus46_reason_labels.jsonl"),
                                                  max_workers=int(cfg["max_workers"]["anthropic"]))
    except BudgetExceeded as exc:
        _record_stage(ctx, "carryover", {"status": "stopped", "reason": f"budget cap: {exc}"})
        return pilot2._budget_stop(ctx, "carryover", exc)
    p3_labels = _read_jsonl(s3 / "reason_labels.jsonl")
    reason_rows, reason_md = analysis_reasons.reason_table(judge_records, p3_labels + added)
    _write_json(ctx.run_dir / "pilot3_reason_table.json",
                {"source": source, "n_labels_pilot3": len(p3_labels), "n_labels_added": len(added),
                 "rows": reason_rows})
    (ctx.run_dir / "pilot3_reason_table.md").write_text(reason_md, encoding="utf-8")
    _record_stage(ctx, "carryover", {"status": "done", "n_verdicts_classified": len(late),
                                     "n_labels_added": len(added)})
    print(f"[carryover] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage: pool (the extra no_tests_run pool from row groups 8 to 16; CPU only)
# ---------------------------------------------------------------------------------------------


def extra_pool_data(cfg: dict) -> dict:
    """The data block of data.extra_pool.filters_from with its row_groups and gap_types replaced by the extra pool's."""
    extra = cfg["data"]["extra_pool"]
    with open(_repo_path(extra["filters_from"]), encoding="utf-8") as f:
        source = yaml.safe_load(f)
    return {**source["data"], "row_groups": list(extra["row_groups"]), "gap_types": list(extra["gap_types"])}


def stage_pool(ctx: Ctx4) -> int:
    """When data.extra_pool.file is missing: coding.build_tasks with extra_pool_data(cfg) -> that file, and
    funnel.json["extra_pool"] = the counts build_tasks records (other funnel keys kept). Never downloads."""
    extra = ctx.cfg["data"]["extra_pool"]
    path = ctx.processed / extra["file"]
    if path.exists():
        ctx.note(f"pool: {path.name} is present; not rebuilt")
        return EXIT_OK
    from overclaim.data import coding

    data = extra_pool_data(ctx.cfg)
    raw_dir = data.get("raw_dir", "data/raw/coding")
    missing = [str(p) for p in (coding.rg_path(raw_dir, rg) for rg in data["row_groups"]) if not p.exists()]
    if ctx.dry_run:
        ctx.note(f"pool: would build {path.name} from row groups {data['row_groups']} ({', '.join(data['gap_types'])} "
                 f"only) with the "
                 f"data filters of {extra['filters_from']}; raw files missing: {missing or 'none'}")
        return EXIT_OK
    if missing:
        logger.error("pool: raw row-group files are missing (this stage never downloads): %s", missing)
        return EXIT_FAILED
    counts: dict = {}
    n = coding.build_tasks({"data": data}, str(path), counts)
    funnel = _read_json(ctx.funnel_path, {})
    funnel["extra_pool"] = {"file": extra["file"], "row_groups": data["row_groups"], "gap_types": data["gap_types"],
                            "filters_from": extra["filters_from"], "n_written": n, "counts": counts}
    _write_json(ctx.funnel_path, funnel)
    logger.info("pool: wrote %d tasks to %s", n, path)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage: sets (the v4 matched sets: the 30 pilot-3 tasks rebuilt, then the top-up)
# ---------------------------------------------------------------------------------------------


def candidate_orders(ctx: Ctx4, first: list[dict]) -> tuple[dict[str, list[tuple[Path, dict]]], dict]:
    """Per gap type, the candidates in order as (pool file, coding.candidate_order entry): the candidate order of
    data.tasks_file without the first tasks' instances, then, for each gap type of data.extra_pool, the candidate
    order of the extra pool file without the instances of the first order and of the first tasks; held-out task ids
    removed from both. Also returns funnel["v4_candidate_pool"]: per gap type the sizes of both orders, the held-out
    ids removed and the candidates left."""
    from overclaim.data import coding

    cfg, pilot, data = ctx.cfg, ctx.cfg["pilot"], ctx.cfg["data"]
    tasks_path = ctx.processed / data["tasks_file"]
    extra = data.get("extra_pool")
    first_instances = {t["instance_id"] for t in first}
    held_out = set(pilot.get("held_out_task_ids") or [])
    orders: dict[str, list[tuple[Path, dict]]] = {}
    sizes: dict[str, dict] = {}
    for gap in GAP_TYPES:
        main = coding.candidate_order(tasks_path, gap, cfg["seed"], pilot["prefer_resolved"],
                                      exclude_instance_ids=first_instances)
        entries = [(tasks_path, c) for c in main if c["task_id"] not in held_out]
        size: dict[str, Any] = {"pool_file": len(main), "extra_pool_file": None,
                                "held_out_removed": len(main) - len(entries)}
        if extra and gap in extra["gap_types"]:
            extra_path = ctx.processed / extra["file"]
            more = coding.candidate_order(extra_path, gap, cfg["seed"], pilot["prefer_resolved"],
                                          exclude_instance_ids=first_instances | {c["instance_id"] for c in main})
            kept = [c for c in more if c["task_id"] not in held_out]
            entries += [(extra_path, c) for c in kept]
            size["extra_pool_file"] = len(more)
            size["held_out_removed"] += len(more) - len(kept)
        size["candidates"] = len(entries)
        orders[gap], sizes[gap] = entries, size
    return orders, sizes


def _lazy_tasks(entries: list[tuple[Path, dict]]) -> Iterator[dict]:
    """The candidate tasks in order, each read from its pool file only when it is consumed."""
    from overclaim.data import coding

    for path, c in entries:
        yield coding.read_task_at(path, c["offset"], c["length"])


def sets_purposes() -> frozenset[str]:
    """The cost.jsonl purposes of the sets stage's calls, read from the `purpose` defaults of the functions that make
    them: span detection (variants_inplace.detect_spans), annotation (variants_v3.annotate_spans), the v4 pre-check
    (verify_v4.precheck_original), the v4 rewrites (variants_v4.rewrite_spans_v4) and the v4 post-check
    (verify_v4.verify_set_v4)."""
    from overclaim import variants_inplace, variants_v3, variants_v4, verify_v4

    fns = (variants_inplace.detect_spans, variants_v3.annotate_spans, verify_v4.precheck_original,
           variants_v4.rewrite_spans_v4, verify_v4.verify_set_v4)
    return frozenset(inspect.signature(fn).parameters["purpose"].default for fn in fns)


def sets_spend(cost_path: Path, purposes: frozenset[str] | set[str]) -> float:
    """The Anthropic USD logged in cost_path by calls whose purpose is in `purposes` (null cost_usd counts 0). Only
    complete lines are read: a line that another thread is still appending has no newline yet and is counted on the
    next call."""
    if not cost_path.exists():
        return 0.0
    total = 0.0
    for line in cost_path.read_text(encoding="utf-8").split("\n")[:-1]:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("provider") == "anthropic" and row.get("purpose") in purposes and row.get("cost_usd") is not None:
            total += float(row["cost_usd"])
    return total


def rebalanced_targets(targets: dict[str, int], realised: dict[str, int], by_gap: dict[str, dict],
                       total: int) -> dict[str, int] | None:
    """The raised targets of the next build pass, or None when there is nothing to rebalance.

    Rebalancing applies when a gap type ended below its target with its pool exhausted (by_gap[gap]["pool_exhausted"],
    funnel["v4"]["by_gap"]) and the realised sets fall short of `total`. The shortfall (total - realised sets) is split
    equally over the gap types that can still grow: not pool_exhausted, not cap_reached and not stopped; a remainder
    goes one set at a time to REMAINDER_GAP first, then to the others in `targets` order. Each of them gets the target
    max(target, realised) + its share. None when the total is reached, when no gap type can grow, or when any gap type
    was stopped by the budget guard (should_stop would then end every gap type at once on the next pass).
    """
    def flag(gap: str, key: str) -> bool:
        return bool((by_gap.get(gap) or {}).get(key))

    if any(flag(g, "stopped") for g in targets):
        return None
    if not any(realised.get(g, 0) < targets[g] and flag(g, "pool_exhausted") for g in targets):
        return None
    shortfall = total - sum(realised.get(g, 0) for g in targets)
    grow = [g for g in targets if not (flag(g, "pool_exhausted") or flag(g, "cap_reached"))]
    if shortfall <= 0 or not grow:
        return None
    share, rest = divmod(shortfall, len(grow))
    new = dict(targets)
    for i, gap in enumerate(sorted(grow, key=lambda g: g != REMAINDER_GAP)):
        new[gap] = max(targets[gap], realised.get(gap, 0)) + share + (1 if i < rest else 0)
    return new


def _primary_of(matched_set: dict) -> dict:
    """{index, text, type} of a set's primary span (spans are 1-based in `primary`, 0-based as list positions)."""
    index = matched_set["primary"]["index"]
    span = matched_set["spans"][index - 1]
    return {"index": index, "text": span["text"], "type": span["type"]}


def rebuild_report(first: list[dict], v3_sets: list[dict], v4_sets: list[dict], infos: list[dict],
                   pilot: dict) -> list[dict]:
    """One entry per first task (first-task order): task_id, gap_type, audited and second_look (pilot.audited_task_ids
    and pilot.second_look_task_ids), v3_primary {index, text, type}, v4_outcome and v4_detail (the build info),
    v4_primary {index, text, type, rule} or None (not built), primary_changed (the primary span's text differs from
    v3's; None when not built), n_kept_true_spans (the set's, else the info's n_kept), named_failure_pre_existing (the
    set's, else the info's facts, else None) and relaxed_primary_exists (the info's)."""
    v3 = {s["task_id"]: s for s in v3_sets}
    v4 = {s["task_id"]: s for s in v4_sets}
    info_of = {i["task_id"]: i for i in infos}
    audited, second = set(pilot.get("audited_task_ids") or []), set(pilot.get("second_look_task_ids") or [])
    out = []
    for task in first:
        tid = task["task_id"]
        info, built = info_of.get(tid, {}), v4.get(tid)
        v3_primary = _primary_of(v3[tid])
        v4_primary = None if built is None else {**_primary_of(built), "rule": built["primary"].get("rule")}
        out.append({
            "task_id": tid, "gap_type": task["gap_type"], "audited": tid in audited, "second_look": tid in second,
            "v3_primary": v3_primary, "v4_outcome": info.get("outcome"), "v4_detail": info.get("detail"),
            "v4_primary": v4_primary,
            "primary_changed": None if v4_primary is None else v4_primary["text"] != v3_primary["text"],
            "n_kept_true_spans": built["n_kept_true_spans"] if built is not None else info.get("n_kept"),
            "named_failure_pre_existing": (built["named_failure_pre_existing"] if built is not None
                                           else (info.get("facts") or {}).get("named_failure_pre_existing")),
            "relaxed_primary_exists": info.get("relaxed_primary_exists"),
        })
    return out


def stage_sets(ctx: Ctx4) -> int:
    """Build the v4 sets (variants_v4.build_all_v4): the first tasks (pilot 3's 30 v3 tasks) rebuilt, then candidates
    per gap type (candidate_orders) up to pilot.targets, with rebalancing (rebalanced_targets) when pilot.rebalance is
    true: build_all_v4 then runs again with the raised targets and fresh candidate iterators (every earlier call is a
    disk-cache hit). should_stop is True once sets_spend of this stage's purposes has reached pilot.sets_stop_usd.

    Frozen (not rebuilt; a note says so) once a run file in FREEZE_FILES holds a record. Behind the stop-launching
    gate. Writes the sets file (each set with shown_eligible), pilot.added_tasks_file, pilot.excluded_file,
    run_dir/v4_build.jsonl, funnel.json ("v4" with "rebalance", "v4_candidate_pool"; other keys kept),
    run_dir/v4_rebuild_report.json and the human-check CSV (never over filled-in judgements).
    """
    cfg, pilot = ctx.cfg, ctx.cfg["pilot"]
    frozen = [name for name in FREEZE_FILES if _has_record(ctx.run_dir / name)]
    if frozen:
        message = f"sets: frozen ({', '.join(frozen)} already hold records); the v4 sets are not rebuilt"
        logger.info(message)
        ctx.note(message)
        return EXIT_OK
    tasks_path = ctx.processed / cfg["data"]["tasks_file"]
    extra = cfg["data"].get("extra_pool")
    pools = [tasks_path] + ([ctx.processed / extra["file"]] if extra else [])
    if ctx.dry_run:
        pool_state = ", ".join(f"{p.name} ({'present' if p.exists() else 'MISSING'})" for p in pools)
        ctx.note(f"sets: would rebuild the tasks of {pilot['first_sets_file']} (from {pilot['first_task_files']}) with "
                 f"design v4, then take candidates from {pool_state} until the targets {pilot['targets']} (total "
                 f"{pilot['total']}, rebalance {pilot.get('rebalance')}), "
                 f"max_new_per_gap {pilot.get('max_new_per_gap')}, no new candidate once this stage's Anthropic spend "
                 f"reaches ${float(pilot['sets_stop_usd']):.2f}; span detection and rewrites {cfg['editor']['model']}, "
                 f"annotation {cfg['annotator']['model']}, v4 pre-check {cfg['precheck_v4']['model']} (effort "
                 f"{cfg['precheck_v4'].get('effort')}), v4 post-check {cfg['verifier_v4']['model']} (effort "
                 f"{cfg['verifier_v4'].get('effort')}); ANTHROPIC_API_KEY "
                 f"{'set' if pilot2.credential_for({'provider': 'anthropic'}) else 'MISSING'}")
        return EXIT_OK
    reason = pilot2.gate(ctx, {}, "anthropic")
    if reason is not None:
        return _skip(ctx, "sets", reason)
    from overclaim import variants_v4, verify_v4
    from overclaim.graders.judge import BudgetExceeded

    detector, annotator, prechecker, verifier = (ctx.client(cfg[key])
                                                 for key in ("editor", "annotator", "precheck_v4", "verifier_v4"))
    if detector is None or annotator is None or prechecker is None or verifier is None:
        logger.error("sets: ANTHROPIC_API_KEY is not set; the v4 sets cannot be built")
        return EXIT_NO_CREDENTIALS
    missing = [str(p) for p in pools if not p.exists()]
    if missing:
        logger.error("sets: candidate pools are missing (run the pool stage for the extra pool): %s", missing)
        return EXIT_FAILED
    vcfg, bank, lint_patterns = variants_v4.load_v4_config(_repo_path(cfg["variants_config"]))
    first = ctx.first_tasks()
    orders, sizes = candidate_orders(ctx, first)
    funnel = _read_json(ctx.funnel_path, {})
    funnel["v4_candidate_pool"] = sizes
    purposes = sets_purposes()
    stop_usd = float(pilot["sets_stop_usd"])
    cost_path = ctx.run_dir / "cost.jsonl"

    def should_stop() -> bool:
        return sets_spend(cost_path, purposes) >= stop_usd

    def precheck_fn(task: dict, message: str, spans: list[dict], evidence_text: str) -> dict:
        return verify_v4.precheck_original(task, message, spans, evidence_text, prechecker)

    def verify_fn(task: dict, set_dict: dict) -> dict:
        return verify_v4.verify_set_v4(task, set_dict, verifier)

    targets = {gap: int(n) for gap, n in pilot["targets"].items()}
    passes, applied = 0, False
    try:
        while True:
            passes += 1
            candidates = {gap: _lazy_tasks(entries) for gap, entries in orders.items()}
            sets, added, excluded, infos = variants_v4.build_all_v4(
                first, candidates, targets, vcfg, bank, lint_patterns, detector, annotator, precheck_fn, verify_fn,
                funnel, max_new_per_gap=pilot.get("max_new_per_gap"), rewriter=detector, should_stop=should_stop)
            realised = {gap: sum(s["gap_type"] == gap for s in sets) for gap in targets}
            raised = (rebalanced_targets(targets, realised, funnel["v4"]["by_gap"], int(pilot["total"]))
                      if pilot.get("rebalance") else None)
            if raised is None:
                break
            if passes > len(targets):   # each rebalancing pass reaches the total or ends a gap type's growth
                raise RuntimeError(f"sets: rebalancing did not settle after {passes} passes: targets {targets}, "
                                   f"realised {realised}")
            logger.info("sets: pass %d realised %s; rebalancing: targets %s -> %s", passes, realised, targets, raised)
            targets, applied = raised, True
    except BudgetExceeded as exc:
        _write_json(ctx.funnel_path, funnel)
        _record_stage(ctx, "sets", {"status": "stopped", "reason": f"budget cap: {exc}"})
        return pilot2._budget_stop(ctx, "sets", exc)
    funnel["v4"]["rebalance"] = {"applied": applied, "passes": passes, "targets_final": targets, "realised": realised}
    task_of = {t["task_id"]: t for t in first + added}
    cap = int(cfg["shown"]["max_transcript_tokens"])
    for s in sets:
        s["shown_eligible"] = task_of[s["task_id"]]["token_counts"]["transcript"] <= cap
    _write_jsonl(ctx.set_path(SET), sets)
    _write_jsonl(ctx.processed / pilot["added_tasks_file"], added)
    _write_jsonl(ctx.processed / pilot["excluded_file"], excluded)
    _write_jsonl(ctx.run_dir / "v4_build.jsonl", infos)
    _write_json(ctx.funnel_path, funnel)
    report = rebuild_report(first, _read_jsonl(ctx.set_path(V3_SET)), sets, infos, pilot)
    _write_json(ctx.run_dir / REBUILD_REPORT_FILE, report)
    check = ctx.processed / cfg["sets"][SET]["human_check"]
    try:
        logger.info("sets: wrote %d rows to %s", variants_v4.write_human_check_v4(check, first + added, sets), check)
    except FileExistsError as exc:  # a human has filled in judgements: the file is kept as it is
        logger.warning("sets: %s", exc)
    _record_stage(ctx, "sets", {"status": "done", "n_sets": len(sets), "realised": realised, "passes": passes})
    logger.info("sets: %d v4 sets (%d added tasks, %d excluded); realised %s", len(sets), len(added), len(excluded),
                realised)
    print(f"[sets] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage: judge (pilot3._stage_runs on cfg["runs"]; the shown task rule; the pointwise hook)
# ---------------------------------------------------------------------------------------------


def shown_rule_ids(v4_sets: list[dict], report: list[dict]) -> list[str]:
    """task_rule no_tests_run_or_primary_changed: the v4 sets (file order) with shown_eligible whose gap type is
    no_tests_run or whose rebuild-report entry has primary_changed True."""
    changed = {e["task_id"] for e in report if e.get("primary_changed") is True}
    return [s["task_id"] for s in v4_sets
            if s["shown_eligible"] and (s["gap_type"] == "no_tests_run" or s["task_id"] in changed)]


def stage_judge(ctx: Ctx4, only: list[str] | None = None) -> int:
    """The runs of cfg["runs"] (or only the ids in `only`) through pilot3._stage_runs (pilot2.stage_judge: `requires`
    flags, statuses in runs_status.json, the gate per run). A run with a task_rule first gets run["task_ids"] =
    shown_rule_ids(v4 sets, run_dir/v4_rebuild_report.json), logged; a run with pointwise_after gets its hook from
    Ctx4.after_task_for through pilot2.stage_judge."""
    cfg = ctx.cfg
    chosen = [r for r in cfg["runs"] if only is None or r["id"] in only]
    if any(r.get("task_rule") for r in chosen):
        v4_sets = _read_jsonl(ctx.set_path(SET))
        ids = shown_rule_ids(v4_sets, _read_json(ctx.run_dir / REBUILD_REPORT_FILE, []))
        for run in chosen:
            if run.get("task_rule"):
                run["task_ids"] = ids
                ctx.note(f"judge: run {run['id']}: task_rule {run['task_rule']} -> {len(ids)} tasks {ids}"
                         + ("" if v4_sets else f" ({ctx.set_path(SET).name} is missing: the list is made from the "
                                               "sets stage's output)"))
    for run in chosen:
        if run.get("pointwise_after"):
            limit = run.get("pointwise_after_tasks")
            scope = "each task's" if limit is None else f"the first {int(limit)} tasks' (of the run's task list)"
            ctx.note(f"judge: run {run['id']}: after {scope} pairwise calls, O, H and P are scored pointwise "
                     f"(e1b_pointwise.score_task) -> {POINTWISE_FILE}")
    return pilot3._stage_runs(ctx, False, only)


# ---------------------------------------------------------------------------------------------
# Stages: reasons, e2, e3
# ---------------------------------------------------------------------------------------------


def stage_reasons(ctx: Ctx4) -> int:
    """analysis_reasons_v4.classify_reasons_v4 (classifier client) over every non-smoke verdict of
    run_dir/judge_calls.jsonl -> run_dir/reason_labels.jsonl, behind the stop-launching gate."""
    cfg = ctx.cfg
    if not cfg.get("reasons", {}).get("enabled", True):
        return EXIT_OK
    if ctx.dry_run:
        ctx.note(f"reasons: would classify every verdict's reason with {cfg['classifier']['model']} (four labels, "
                 "says_disclosure_false added) -> reason_labels.jsonl")
        return EXIT_OK
    records = [r for r in _read_jsonl(ctx.run_dir / "judge_calls.jsonl") if not r.get("smoke_only")]
    reason = pilot2.gate(ctx, {}, "anthropic")
    if reason is not None:
        labels = len(_read_jsonl(ctx.run_dir / "reason_labels.jsonl"))
        return _skip(ctx, "reasons", reason, verdicts=len(records), labels_present=labels)
    from overclaim import analysis_reasons_v4
    from overclaim.graders.judge import BudgetExceeded

    llm = ctx.client(cfg["classifier"])
    if llm is None:
        return _skip(ctx, "reasons", f"no credential for the classifier {cfg['classifier']['model']}")
    try:
        labels = analysis_reasons_v4.classify_reasons_v4(records, llm, str(ctx.run_dir / "reason_labels.jsonl"),
                                                         max_workers=int(cfg["max_workers"]["anthropic"]))
    except BudgetExceeded as exc:
        _record_stage(ctx, "reasons", {"status": "stopped", "reason": f"budget cap: {exc}"})
        return pilot2._budget_stop(ctx, "reasons", exc)
    _record_stage(ctx, "reasons", {"status": "done", "verdicts": len(records), "labels": len(labels)})
    logger.info("reasons: %d labels; %s", len(labels), ctx.cost.totals_line())
    return EXIT_OK


def stage_e2(ctx: Ctx4) -> int:
    """Experiment 2 policy pilot (e2_selection.run_e2_packet) on the first e2.n_tasks of seeded_order(v4 set ids,
    seed), each with its set's evidence_v4; the pilot-3 labeller is validated first on every task that has a v3 set,
    with the v3 sets, and the full-evidence labeller on every v4 set (validation_sets_v4 = the v4 sets,
    validation_tasks_v4 = pilot_tasks(), validation_evidence_v4 = each set's evidence_v4); n_tie_seeds =
    e2.n_tie_seeds. Clients: policy cfg[e2.policy], labeller cfg[e2.labeller] (both labellers), one per selector
    (cfg["judges"]), structured per selector from its structured_output. Writes the Experiment 2 files into run_dir,
    run_dir/e2_summary.json (run_e2_packet's return value, NaN as null) and stages_status.json["e2"]."""
    cfg, e2 = ctx.cfg, ctx.cfg["e2"]
    v4_sets = _read_jsonl(ctx.set_path(SET))
    ids = seeded_order([s["task_id"] for s in v4_sets], cfg["seed"])[: int(e2["n_tasks"])]
    if ctx.dry_run:
        ctx.note(f"e2: would validate the labeller {cfg[e2['labeller']]['model']} on every task with a v3 set (the v3 "
                 f"sets) and its full-evidence prompt on every v4 set ({len(v4_sets)} now), then draw "
                 f"{e2['n_samples']} samples per task from the policy {cfg[e2['policy']]['model']} on "
                 f"the evidence packet of the first {e2['n_tasks']} v4 sets of the seeded order "
                 f"({ids if ids else 'made from the sets stage output'}), score them with {e2['selectors']} and label "
                 f"them with both prompts; stop below labeller accuracy {e2['stop_below']} (pilot-3 prompt only), "
                 f"clean-overclaim threshold {e2['clean_rate_threshold']}, {e2['n_tie_seeds']} tie-break seeds")
        return EXIT_OK
    if not v4_sets:
        logger.error("e2: %s is missing or empty; run the sets stage first", ctx.set_path(SET))
        return EXIT_FAILED
    reason = pilot2.gate(ctx, {}, "anthropic")
    if reason is not None:
        return _skip(ctx, "e2", reason)
    from overclaim import analysis
    from overclaim.experiments import e2_selection
    from overclaim.graders.judge import BudgetExceeded

    if tuple(e2["ns"]) != tuple(e2_selection.NS):
        logger.error("e2: e2.ns %s differs from the N values e2_selection computes %s", e2["ns"], e2_selection.NS)
        return EXIT_FAILED
    clients = {"policy": ctx.client(cfg[e2["policy"]]), "labeller": ctx.client(cfg[e2["labeller"]]),
               **{name: ctx.client(cfg["judges"][name]) for name in e2["selectors"]}}
    missing = [k for k, c in clients.items() if c is None]
    if missing:
        return _skip(ctx, "e2", f"no credential for the clients {missing}")
    set_of = {s["task_id"]: s for s in v4_sets}
    tasks = _tasks_in_order(ctx.pilot_tasks(), ids)
    try:
        result = e2_selection.run_e2_packet(
            tasks, {tid: set_of[tid]["evidence_v4"] for tid in ids}, ctx.first_tasks(),
            _read_jsonl(ctx.set_path(V3_SET)), clients, ctx.run_dir, n_samples=int(e2["n_samples"]),
            selectors=list(e2["selectors"]),
            structured={name: bool(cfg["judges"][name].get("structured_output")) for name in e2["selectors"]},
            max_workers=int(cfg["max_workers"]["anthropic"]), stop_below=float(e2["stop_below"]),
            clean_rate_threshold=float(e2["clean_rate_threshold"]), validation_sets_v4=v4_sets,
            validation_tasks_v4=ctx.pilot_tasks(),
            validation_evidence_v4={s["task_id"]: s["evidence_v4"] for s in v4_sets},
            n_tie_seeds=int(e2["n_tie_seeds"]))
    except BudgetExceeded as exc:
        _record_stage(ctx, "e2", {"status": "stopped", "reason": f"budget cap: {exc}"})
        return pilot2._budget_stop(ctx, "e2", exc)
    _write_json(ctx.run_dir / "e2_summary.json", analysis._nan_to_none(result))
    status = "stopped_labeller_validation" if result["status"] == "stopped_labeller_validation" else "done"
    _record_stage(ctx, "e2", {"status": status, "n_tasks": len(tasks), "task_ids": ids})
    logger.info("e2: %s", status)
    print(f"[e2] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK


def e3_task_ids(ctx: Ctx4) -> list[str]:
    """The first e3.n_tasks ids of session3_dir/stages_status.json["e3"]["task_ids"] ([] when that is missing)."""
    status = _read_json(_repo_path(ctx.cfg["session3_dir"]) / "stages_status.json", {})
    return list((status.get("e3") or {}).get("task_ids") or [])[: int(ctx.cfg["e3"]["n_tasks"])]


def stage_e3(ctx: Ctx4) -> int:
    """The three-label check: e3_verification.VerificationFirstJudge3 (the e3 judge's spec with max_tokens =
    e3_max_tokens, grader `<judge>+verify_first_3`) through judge.run_pairs2 on the v3 sets of e3_task_ids, shown,
    prompt cache, claims kept -> run_dir/e3_judge3_calls.jsonl; then e3_verification.e3_compare against pilot 3's
    two-label records (e3_judge_calls.jsonl, grader `<judge>+verify_first`) and plain shown records (judge_calls.jsonl:
    that judge, set and rubric, shown) on the same ids, and claims_summary3 -> run_dir/e3_summary.json (NaN as null)."""
    cfg, e3 = ctx.cfg, ctx.cfg["e3"]
    s3 = _repo_path(cfg["session3_dir"])
    ids = e3_task_ids(ctx)
    grader = f"{e3['judge']}+verify_first_3"
    if ctx.dry_run:
        ctx.note(f"e3: would run {grader} (max_tokens {cfg['e3_max_tokens']}), set {e3['set']}, rubric {e3['rubric']}, "
                 f"shown, on {len(ids)} tasks of pilot 3's e3 list {ids}, then compare with pilot 3's two-label and "
                 "plain shown records")
        return EXIT_OK
    if len(ids) < int(e3["n_tasks"]):
        logger.error("e3: %s holds %d e3 task ids, %d wanted", s3 / "stages_status.json", len(ids), int(e3["n_tasks"]))
        return EXIT_FAILED
    reason = pilot2.gate(ctx, {}, "anthropic")
    if reason is not None:
        return _skip(ctx, "e3", reason)
    from overclaim import analysis
    from overclaim.experiments import e3_verification
    from overclaim.graders import judge as J

    spec = {**cfg["judges"][e3["judge"]], "max_tokens": int(cfg["e3_max_tokens"])}
    client = ctx.client(spec)
    if client is None:
        return _skip(ctx, "e3", f"no credential for {e3['judge']}")
    vf3 = e3_verification.VerificationFirstJudge3(client, grader, rubric=e3["rubric"],
                                                  question=cfg["rubrics"]["rubrics"][e3["rubric"]],
                                                  structured=bool(spec.get("structured_output")))
    wanted = set(ids)
    tasks = _tasks_in_order(ctx.first_tasks(), ids)
    v3_sets = [s for s in _read_jsonl(ctx.set_path(e3["set"])) if s["task_id"] in wanted]
    try:
        records = J.run_pairs2(tasks, v3_sets, vf3, cfg["pairs"], "shown", str(ctx.run_dir / "e3_judge3_calls.jsonl"),
                               cfg["run_name"], set_name=e3["set"], max_workers=int(cfg["max_workers"]["anthropic"]),
                               use_prompt_cache=True, extra_fields=("claims_A", "claims_B"))
    except J.BudgetExceeded as exc:
        _record_stage(ctx, "e3", {"status": "stopped", "reason": f"budget cap: {exc}"})
        return pilot2._budget_stop(ctx, "e3", exc)
    except J.CacheNotRead as exc:
        _record_stage(ctx, "e3", {"status": "failed", "reason": str(exc)})
        logger.error("e3: FAILED: %s", exc)
        return EXIT_FAILED

    def same(r: dict, name: str) -> bool:
        return (r["grader"] == name and r.get("set") == e3["set"] and r.get("rubric") == e3["rubric"]
                and r["condition"] == "shown" and r["task_id"] in wanted)

    vf2 = [r for r in _read_jsonl(s3 / "e3_judge_calls.jsonl") if same(r, f"{e3['judge']}+verify_first")]
    plain = [r for r in _read_jsonl(s3 / "judge_calls.jsonl") if same(r, e3["judge"])]
    rows, markdown = e3_verification.e3_compare(records, vf2, plain, cfg["pairs"],
                                                n_boot=int(cfg["analysis"]["n_bootstrap"]),
                                                seed=int(cfg["analysis"]["seed"]))
    _write_json(ctx.run_dir / "e3_summary.json", analysis._nan_to_none(
                {"status": "done", "judge": e3["judge"], "grader": grader, "set": e3["set"], "rubric": e3["rubric"],
                 "n_tasks": len(tasks), "task_ids": ids, "n_records": len(records), "n_two_label_records": len(vf2),
                 "n_plain_records": len(plain), "rows": rows, "markdown": markdown,
                 "claims": e3_verification.claims_summary3(records)}))
    _record_stage(ctx, "e3", {"status": "done", "n_tasks": len(tasks), "task_ids": ids})
    print(f"[e3] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stages: pointwise and pointwise_late (Experiment 1b), rm
# ---------------------------------------------------------------------------------------------


def pointwise_task_ids(first_ids: list[str], v4_sets: list[dict]) -> list[str]:
    """The pointwise order: the first tasks that have a v4 set (first-task order), then the other v4 sets (file
    order)."""
    have = {s["task_id"] for s in v4_sets}
    first = set(first_ids)
    return [t for t in first_ids if t in have] + [s["task_id"] for s in v4_sets if s["task_id"] not in first]


def _stage_pointwise(ctx: Ctx4, late: bool) -> int:
    """The runs of pointwise.runs without (late False) or with `late: true`, in order, each through
    e1b_pointwise.run_pointwise on the first n_tasks (the run's own, else pointwise.n_tasks) of the first
    pointwise.n_tasks of pointwise_task_ids -> run_dir/pointwise_calls.jsonl. Per run, as pilot2.stage_judge: a run
    whose records are complete is not gated; otherwise no credential, a provider blocked earlier in this invocation
    (Gemini balance exhausted) or pilot2.gate skips it. runs_status.json gets one entry per run (done, skipped with the
    reason, stopped, failed)."""
    cfg, pw = ctx.cfg, ctx.cfg["pointwise"]
    stage = "pointwise_late" if late else "pointwise"
    runs = [r for r in pw["runs"] if bool(r.get("late", False)) == late]
    v4_sets = _read_jsonl(ctx.set_path(pw["set"]))
    order = pointwise_task_ids([t["task_id"] for t in ctx.first_tasks()], v4_sets)[: int(pw["n_tasks"])]
    if ctx.dry_run:
        for run in runs:
            spec = pilot2.run_spec(cfg, run)
            n = int(run.get("n_tasks", pw["n_tasks"]))
            ctx.note(f"{stage}: run {run['id']}: {run.get('grader_label', run['judge'])} ({spec['provider']}), set "
                     f"{pw['set']}, {pw['condition']}, replicate {int(run.get('replicate', 0))}, three calls per task "
                     f"on the first {n} tasks of the pointwise list ({len(order[:n])} now) -> {POINTWISE_FILE}; "
                     f"credential {'set' if pilot2.credential_for(spec) else 'MISSING -> skipped'}")
        return EXIT_OK
    if not v4_sets:
        logger.error("%s: %s is missing or empty; run the sets stage first", stage, ctx.set_path(pw["set"]))
        return EXIT_FAILED
    from overclaim.experiments import e1b_pointwise
    from overclaim.graders import judge as J

    task_of = {t["task_id"]: t for t in ctx.pilot_tasks()}
    set_of = {s["task_id"]: s for s in v4_sets}
    out_path = ctx.run_dir / POINTWISE_FILE
    status_path = ctx.run_dir / "runs_status.json"
    status: dict[str, dict] = _read_json(status_path, {})
    existing = _read_jsonl(out_path)
    blocked_providers: dict[str, str] = {}
    for run in runs:
        spec = pilot2.run_spec(cfg, run)
        provider = pilot2.COST_PROVIDER[spec["provider"]]
        label = run.get("grader_label", run["judge"])
        replicate = int(run.get("replicate", 0))
        ids = order[: int(run.get("n_tasks", pw["n_tasks"]))]
        wanted = set(ids)
        expected = 3 * len(ids)
        have = sum(1 for r in existing if r["grader"] == label and r.get("set") == pw["set"]
                   and r["condition"] == pw["condition"] and int(r.get("replicate", 0)) == replicate
                   and r["task_id"] in wanted)
        entry: dict[str, Any] = {"stage": stage, "judge": label, "set": pw["set"], "condition": pw["condition"],
                                 "replicate": replicate, "n_tasks": len(ids), "expected_records": expected}
        reason = None
        if have < expected:
            if not pilot2.credential_for(spec):
                reason = f"no credential for provider {spec['provider']}" + (
                    f" route {spec.get('route')}" if spec["provider"] == "gemini" else "")
            elif provider in blocked_providers:
                reason = blocked_providers[provider]
            else:
                reason = pilot2.gate(ctx, run, provider)
        if reason is not None:
            entry.update(status="skipped", reason=reason, records=have)
            previous = status.get(run["id"], {})
            if previous.get("status") == "skipped" and previous.get("reason") != reason:
                entry.update(reason=previous.get("reason"), reason_on_last_invocation=reason)
            status[run["id"]] = entry
            _write_json(status_path, status)
            logger.warning("%s: run %s SKIPPED: %s", stage, run["id"], reason)
            continue
        client = ctx.client(spec)
        logger.info("%s: run %s: %s, %d tasks, %d of %d records present", stage, run["id"], label, len(ids), have,
                    expected)
        try:
            records = e1b_pointwise.run_pointwise(
                [task_of[i] for i in ids], [set_of[i] for i in ids], client, label, pw["condition"], str(out_path),
                cfg["run_name"], set_name=pw["set"], structured=bool(spec.get("structured_output")),
                max_workers=int(cfg.get("max_workers", {}).get(provider, 1)),
                use_prompt_cache=pw["condition"] == "shown" and spec["provider"] == "anthropic", replicate=replicate)
        except J.BudgetExceeded as exc:
            entry.update(status="stopped", reason=f"budget cap: {exc}")
            status[run["id"]] = entry
            _write_json(status_path, status)
            return pilot2._budget_stop(ctx, stage, exc)
        except J.CacheNotRead as exc:
            entry.update(status="failed", reason=str(exc))
            status[run["id"]] = entry
            _write_json(status_path, status)
            logger.error("%s: run %s FAILED: %s", stage, run["id"], exc)
            return EXIT_FAILED
        except Exception as exc:
            if type(exc).__name__ == "BalanceExhausted":
                blocked_providers[provider] = f"Gemini balance exhausted (HTTP 402) during run {run['id']}"
                entry.update(status="stopped", reason=str(exc))
                status[run["id"]] = entry
                _write_json(status_path, status)
                logger.error("%s: run %s STOPPED: %s", stage, run["id"], exc)
                existing = _read_jsonl(out_path)
                continue
            raise
        existing = _read_jsonl(out_path)
        entry.update(status="done", records=len(records), n_invalid=sum(r["score"] is None for r in records),
                     n_live=sum(not r["cached"] for r in records))
        status[run["id"]] = entry
        _write_json(status_path, status)
        print(f"[{run['id']}] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK


def stage_pointwise(ctx: Ctx4) -> int:
    """The pointwise runs without `late: true`, in config order."""
    return _stage_pointwise(ctx, False)


def stage_pointwise_late(ctx: Ctx4) -> int:
    """The pointwise runs with `late: true`, in config order."""
    return _stage_pointwise(ctx, True)


def stage_rm(ctx: Ctx4) -> int:
    """pilot2.stage_rm on the v4 sets (rm.sets, hidden). Local GPU, no Anthropic money: not gated."""
    return pilot2.stage_rm(ctx)


# ---------------------------------------------------------------------------------------------
# Stage: analysis, and the stage loop
# ---------------------------------------------------------------------------------------------


def stage_analysis(ctx: Ctx4) -> int:
    """summary_v4.summarize_pilot4 -> run_dir/summary.md and summary.json, printed."""
    if ctx.dry_run:
        ctx.note("analysis: would write summary.md and summary.json (summary_v4.summarize_pilot4)")
        return EXIT_OK
    from overclaim import summary_v4

    print(summary_v4.summarize_pilot4(str(ctx.run_dir), ctx.cfg, extra_sections=projection_sections(ctx.run_dir)))
    return EXIT_OK


PROJECTION_FILES = ("projection_session5.md", "projection_session5.json")


def projection_sections(run_dir: Path) -> list[dict] | None:
    """The session-5 cost projection as an extra summary section, when overclaim.projection_v4 has written its two
    files into run_dir (python -m overclaim.projection_v4 --config configs/projection_session5.yaml): the markdown
    with its headings moved one level down under "## Projection for the full run (session 5)", and the JSON under
    summary.json["projection_session5"]. None when either file is missing (the summary then has no such section)."""
    md, js = (run_dir / name for name in PROJECTION_FILES)
    if not (md.exists() and js.exists()):
        return None
    lines = md.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    body = "\n".join("#" + line if line.startswith("#") else line for line in lines).strip("\n")
    return [{"key": "projection_session5", "data": json.loads(js.read_text(encoding="utf-8")),
             "markdown": "## Projection for the full run (session 5)\n\n" + body}]


_STAGE_FUNCS = {"carryover": stage_carryover, "pool": stage_pool, "sets": stage_sets, "reasons": stage_reasons,
                "e2": stage_e2, "e3": stage_e3, "pointwise": stage_pointwise, "rm": stage_rm,
                "pointwise_late": stage_pointwise_late, "analysis": stage_analysis}
_RUN_STAGES = {"judge": stage_judge}


def run(cfg: dict, cfg_path: Path, run_dir: Path, stage: str, dry_run: bool, only_runs: list[str] | None = None) -> int:
    """Run one stage or all of them (STAGES order); returns the exit code. `only_runs` applies to the judge stage."""
    if stage != "all" and stage not in STAGES:
        logger.error("unknown pilot-4 stage %r; expected one of %s", stage, STAGES)
        return EXIT_FAILED
    ctx = Ctx4(cfg, cfg_path, run_dir, dry_run)
    resolved = dict(cfg)
    resolved["source_config"] = str(cfg_path)
    resolved["resolved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
                                         encoding="utf-8")
    for name in STAGES if stage == "all" else (stage,):
        logger.info("--- stage %s%s ---", name, " (dry run)" if dry_run else "")
        code = _RUN_STAGES[name](ctx, only_runs) if name in _RUN_STAGES else _STAGE_FUNCS[name](ctx)
        if code != EXIT_OK:
            logger.error("run stopped at stage %r with exit code %d", name, code)
            return code
    if dry_run:
        print("=== pilot 4 dry-run plan (no API calls, no model loading) ===")
        for line in ctx.plan:
            print(line)
    elif ctx._cost is not None:
        print(ctx.cost.totals_line())
    return EXIT_OK

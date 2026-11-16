"""Research-report pilot (session 4, contract D2): data -> harness -> gaps -> sets -> judge -> rm -> pointwise ->
judge_late -> reasons -> analysis.

Entry point: `uv run python -m overclaim.experiments.pilot_research --config configs/pilot_research.yaml [--stage S]
[--runs id,id] [--dry-run]`. Run directory <results root>/<run_name> (results/pilot_research/).

Costs are logged to the session's cost file: the CostLogger is created on cfg["cost_log_dir"] (results/pilot4_coding)
with the budget block of cfg["budget_from"] (configs/e1_pilot4_coding.yaml: the session caps and the stop-launching
line, read, not copied), so the caps hold across both pilots. Every stage or run that spends Anthropic money is not
launched once the Anthropic total of that file has reached budget.anthropic_stop_launching_usd (pilot2.gate); the skip
is recorded in run_dir/stages_status.json (judge and pointwise runs: runs_status.json), and a stage recorded as done
stays done (pilot3._skip). The harness spends OpenRouter money only and is held by the OpenRouter cap. Every model call
of the set building has a purpose that starts with "research_" (domains/research/sets.py); judge calls keep the judge
module's purpose and their cost is read from this pilot's own record files.

Every stage is idempotent (outputs that exist are reused, every call is cached on disk), supports --dry-run (no call,
no download, no model loading; the plan names what each stage would do) and deletes nothing. The sets are frozen once a
grader record exists. Source text is data and stays in the gitignored pool and tasks files; reports and model replies
are data. Exit codes as pilot 2: 0 ok; 1 a stage failed or an input is missing; 2 a required credential is missing;
3 a budget cap was hit.

Design v2 (session 5, configs/prep5_research.yaml, `design: v2`; contract B): the research config's pool file and
"n_sources" harness budget, gaps.classify_gap_v2, the sets_v2 builder and human-check writer, set research_v2, the
stages STAGES_V2 (no rm, no reasons; a per-run condition and n_tasks in the pointwise block), the gate on the
sources_unread sets (research_gate -> run_dir/gate.json, written by the analysis stage) and the stage e3 (the
three-label judge on 5 sets, only after a passed gate). Without the key every stage is session 4's, unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from overclaim.experiments import pilot2, pilot3
from overclaim.experiments.pilot2 import (EXIT_FAILED, EXIT_NO_CREDENTIALS, EXIT_OK, _read_json, _read_jsonl,
                                          _repo_path, _write_json, _write_jsonl)
from overclaim.experiments.pilot3 import _has_record, _record_stage, _skip, seeded_order
from overclaim.paths import RESULTS_DIR, load_env
from overclaim.variants_v3 import _count

logger = logging.getLogger(__name__)

STAGES = ("data", "harness", "gaps", "sets", "judge", "rm", "pointwise", "judge_late", "reasons", "analysis")
SET = "research_v1"
GAP_TYPES = ("sources_unread", "source_truncated")
MODEL_KEYS = ("precheck_v4", "verifier_v4")   # configs/models.yaml entries beyond pilot 2's: the Sonnet 5.5 effort-high
                                              # specs of the pre-check and the post-check
# Once one of these run files holds a record, the research sets are frozen: the sets stage only reports that.
FREEZE_FILES = ("judge_calls.jsonl", "pointwise_calls.jsonl", "rm_scores.jsonl")
TRAJECTORIES_FILE = "trajectories.jsonl"
SUFFIX_TRAJECTORIES_FILE = "trajectories_suffix.jsonl"
POINTWISE_FILE = "pointwise_calls.jsonl"
NUDGED_DRAFT_WORDS = 150   # an agent text of this many words before the submission is a report-length draft
# Design v2 (session 5): the set, the stages in launch order (e3 reads the gate that the analysis stage writes) and
# the files of the gate and of the Experiment 3 check.
DESIGN_V2 = "v2"
SET_V2 = "research_v2"
STAGES_V2 = ("data", "harness", "gaps", "sets", "judge", "pointwise", "judge_late", "analysis", "e3")
GATE_FILE = "gate.json"
E3_CALLS_FILE = "e3_judge3_calls.jsonl"
E3_SUMMARY_FILE = "e3_summary.json"


def is_v2(cfg: dict) -> bool:
    """Whether the config selects design v2 (`design: v2`)."""
    return cfg.get("design") == DESIGN_V2


def set_name(cfg: dict) -> str:
    """The research set of the design: SET_V2 for design v2, else SET."""
    return SET_V2 if is_v2(cfg) else SET


def stages(cfg: dict) -> tuple[str, ...]:
    """The stages of the design, in launch order: STAGES_V2 for design v2, else STAGES."""
    return STAGES_V2 if is_v2(cfg) else STAGES


# ---------------------------------------------------------------------------------------------
# Config and context
# ---------------------------------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    """pilot2.load_config (judges, editor, classifier, reward models, prices, rubrics; runs validated) plus:
    MODEL_KEYS and cfg["agent_spec"] (the configs/models.yaml entry named by cfg["agent"]); cfg["research"] (the yaml of
    cfg["research_config"]) and cfg["data"]["processed_dir"] taken from its data block; cfg["budget"] = the budget block
    of cfg["budget_from"]. ValueError when the targets are not exactly GAP_TYPES, a pointwise judge is unknown, or run
    ids are not unique over the judge runs and the pointwise runs together (they share runs_status.json).

    Design v2 (`design: v2`; any other value of `design` is a ValueError) also loads cfg["e3_max_tokens"] from
    configs/models.yaml and checks: min_gap_tasks names gap types of GAP_TYPES; the set SET_V2 is in sets and every run
    and the pointwise block use it; a pointwise run's condition (when given) is hidden or shown and its n_tasks (when
    given) a positive integer; the gate's and e3's judges and rubrics are known."""
    cfg = pilot2.load_config(path)
    if cfg.get("design") not in (None, DESIGN_V2):
        raise ValueError(f"unknown design {cfg['design']!r} (known: {DESIGN_V2!r}, or no key for session 4's)")
    with open(_repo_path(cfg["models_config"]), encoding="utf-8") as f:
        models = yaml.safe_load(f)
    for key in (*MODEL_KEYS, cfg["agent"]):
        if key not in models:
            raise ValueError(f"{key!r} is not an entry of {cfg['models_config']}")
    for key in MODEL_KEYS:
        cfg[key] = models[key]
    cfg["agent_spec"] = models[cfg["agent"]]
    with open(_repo_path(cfg["research_config"]), encoding="utf-8") as f:
        cfg["research"] = yaml.safe_load(f)
    cfg["data"] = {**(cfg.get("data") or {}), "processed_dir": cfg["research"]["data"]["processed_dir"]}
    with open(_repo_path(cfg["budget_from"]), encoding="utf-8") as f:
        cfg["budget"] = yaml.safe_load(f)["budget"]
    if set(cfg["targets"]) != set(GAP_TYPES):
        raise ValueError(f"targets must name exactly {list(GAP_TYPES)}")
    for run in cfg["pointwise"]["runs"]:
        if run["judge"] not in cfg["judges"]:
            raise ValueError(f"pointwise run {run['id']}: unknown judge {run['judge']!r}")
    ids = [r["id"] for r in cfg["runs"]] + [r["id"] for r in cfg["pointwise"]["runs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("run ids must be unique over runs and pointwise.runs")
    if is_v2(cfg):
        _check_v2(cfg, models)
    return cfg


def _check_v2(cfg: dict, models: dict) -> None:
    """The design-v2 part of load_config (see there); sets cfg["e3_max_tokens"]."""
    cfg["e3_max_tokens"] = models["e3_max_tokens"]
    unknown = set(cfg.get("min_gap_tasks") or {}) - set(GAP_TYPES)
    if unknown:
        raise ValueError(f"min_gap_tasks: unknown gap types {sorted(unknown)}")
    pw = cfg["pointwise"]
    if SET_V2 not in cfg["sets"] or pw["set"] != SET_V2 or any(r["set"] != SET_V2 for r in cfg["runs"]):
        raise ValueError(f"design v2: the runs, the pointwise block and sets must use the set {SET_V2!r}")
    for run in pw["runs"]:
        if run.get("condition", pw["condition"]) not in ("hidden", "shown"):
            raise ValueError(f"pointwise run {run['id']}: unknown condition {run.get('condition')!r}")
        n = run.get("n_tasks")
        if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 1):
            raise ValueError(f"pointwise run {run['id']}: n_tasks must be a positive integer")
    for block in ("gate", "e3"):
        if not isinstance(cfg.get(block), dict):
            raise ValueError(f"design v2 needs the {block} block")
        judges = cfg[block]["judges"] if block == "gate" else [cfg[block]["judge"]]
        if any(j not in cfg["judges"] for j in judges) or cfg[block]["rubric"] not in cfg["rubrics"]["rubrics"]:
            raise ValueError(f"{block}: unknown judge or rubric")
        if cfg[block]["gap_type"] not in GAP_TYPES:
            raise ValueError(f"{block}: unknown gap type {cfg[block]['gap_type']!r}")


class ResearchCtx(pilot3.Ctx3):
    """pilot3.Ctx3 with the session's cost file and the research task lists."""

    @property
    def cost(self) -> Any:
        """The CostLogger on cfg["cost_log_dir"] (the session's cost.jsonl) with cfg["budget"] as caps."""
        if self._cost is None:
            from overclaim.graders.judge import CostLogger

            path = _repo_path(self.cfg["cost_log_dir"])
            path.mkdir(parents=True, exist_ok=True)
            self._cost = CostLogger(str(path), prices=self.cfg["prices_usd_per_mtok"], caps=self.cfg["budget"])
        return self._cost

    def tasks(self) -> list[dict]:
        """Every task of data.tasks_file (the gap trajectories' tasks), in file order."""
        return _read_jsonl(self.processed / self.cfg["data"]["tasks_file"])

    def pilot_tasks(self) -> list[dict]:
        """The tasks that have a research set (set_name of the design), in the order of the sets file; before that
        file exists, tasks()."""
        tasks = self.tasks()
        path = self.set_path(set_name(self.cfg))
        if not path.exists():
            return tasks
        by_id = {t["task_id"]: t for t in tasks}
        return [by_id[s["task_id"]] for s in _read_jsonl(path)]


def _pool(ctx: ResearchCtx) -> list[dict]:
    from overclaim.data import research

    return research.load_pool(ctx.processed / ctx.cfg["research"]["data"]["pool_file"])


def batches(ctx: ResearchCtx, pool: list[dict]) -> dict[str, list[str]]:
    """The cluster ids of the two harness batches: "main" = the first research harness.n_trajectories of
    seeded_order(pool cluster ids, cfg seed); "suffix" = the next harness_suffix.n_trajectories of that order (at most
    the clusters left), so no cluster is in both."""
    order = seeded_order([c["cluster_id"] for c in pool], ctx.cfg["seed"])
    n = int(ctx.cfg["research"]["harness"]["n_trajectories"])
    return {"main": order[:n], "suffix": order[n:n + int(ctx.cfg["harness_suffix"]["n_trajectories"])]}


# ---------------------------------------------------------------------------------------------
# Stages: data, harness, gaps
# ---------------------------------------------------------------------------------------------


def stage_data(ctx: ResearchCtx) -> int:
    """research.build_pool (which calls research.download when the raw file is missing) -> the pool file and
    funnel.json["pool"] (the eligibility counts and the pool size; other funnel keys kept). Skipped when the pool file
    and funnel["pool"] both exist (build_pool is seeded, so a rebuild writes the same pool)."""
    from overclaim.data import research

    rcfg = ctx.cfg["research"]
    pool_path = ctx.processed / rcfg["data"]["pool_file"]
    funnel = _read_json(ctx.funnel_path, {})
    if pool_path.exists() and "pool" in funnel:
        ctx.note(f"data: {pool_path.name} and funnel.json[pool] are present; not rebuilt")
        return EXIT_OK
    if ctx.dry_run:
        raw = _repo_path(rcfg["data"]["raw_dir"]) / research.SOURCE_FILE
        ctx.note(f"data: would build the pool of {rcfg['data']['pool_size']} clusters (seed {rcfg['seed']}) from "
                 f"{raw} ({'present' if raw.exists() else 'MISSING: downloaded from ' + research.REPO_ID}) into "
                 f"{pool_path} and record the eligibility counts in funnel.json[pool]")
        return EXIT_OK
    counts: dict = {}
    try:
        pool = research.build_pool(rcfg, counts)
    except ValueError as exc:   # fewer eligible clusters than pool_size; the counts are filled
        funnel["pool"] = counts
        _write_json(ctx.funnel_path, funnel)
        logger.error("data: %s", exc)
        return EXIT_FAILED
    funnel["pool"] = {**counts, "pool_size": len(pool)}
    _write_json(ctx.funnel_path, funnel)
    logger.info("data: %d clusters in %s", len(pool), pool_path)
    return EXIT_OK


def nudged_draft(traj: dict) -> bool:
    """Whether the agent wrote a text of NUDGED_DRAFT_WORDS words or more in a turn before the one that delivered the
    report (such a text is part of the shown transcript; gaps.render_tool_log renders it)."""
    from overclaim.data.research import word_count
    from overclaim.domains.research.gaps import _report_turn

    report_turn = _report_turn(traj)
    return any(word_count(text) >= NUDGED_DRAFT_WORDS
               for turn, text in enumerate(traj["assistant_texts"], start=1) if turn != report_turn)


def harness_funnel(records: list[dict]) -> dict:
    """The harness funnel of one batch: runs, agent models, statuses, submitted_via ("none" for no report), tool calls
    (counted calls: total and per-run mean), refused calls (total and runs with any), report words (n, min, median,
    max over the runs with a report) and nudged_drafts (runs with nudged_draft).

    When every record has reading_detail (design v2), also: budget_mode ({mode: runs}), tool_budget ({B: runs}),
    sources_opened and sources_unopened per run ({"mean", "distribution": {n: runs}}) and runs_with_unopened."""
    words = [r["report_words"] for r in records if r["report_words"] is not None]
    calls = [r["n_tool_calls"] for r in records]
    out = {
        "runs": len(records),
        "agent_models": _count(r["agent_model"] for r in records),
        "statuses": _count(r["status"] for r in records),
        "submitted_via": _count(str(r["submitted_via"] or "none") for r in records),
        "tool_calls": {"total": sum(calls), "mean": statistics.mean(calls) if calls else None},
        "refused_calls": {"total": sum(r["n_refused"] for r in records),
                          "runs_with_any": sum(1 for r in records if r["n_refused"])},
        "report_words": {"n": len(words), "min": min(words, default=None),
                         "median": statistics.median(words) if words else None, "max": max(words, default=None)},
        "nudged_drafts": sum(1 for r in records if nudged_draft(r)),
    }
    if records and all("reading_detail" in r for r in records):
        opened = [sum(d["opened"] for d in r["reading_detail"]) for r in records]
        unopened = [len(r["reading_detail"]) - k for r, k in zip(records, opened)]
        out.update(
            budget_mode=_count(str(r.get("budget_mode")) for r in records),
            tool_budget=_count(str(r["tool_budget"]) for r in records),
            sources_opened={"mean": statistics.mean(opened), "distribution": _count(map(str, opened))},
            sources_unopened={"mean": statistics.mean(unopened), "distribution": _count(map(str, unopened))},
            runs_with_unopened=sum(1 for k in unopened if k))
    return out


def stage_harness(ctx: ResearchCtx) -> int:
    """The trajectories: a probe, then the batch, then (only when harness_suffix.enabled) the suffix batch.

    Probe: one harness.run_trajectory on the first cluster of the main batch (batches), checked and not written; when
    it ends without a report (status not "ok") or made no counted tool call, the stage stops with EXIT_FAILED and the
    probe's numbers in stages_status.json["harness"] (the lead then switches cfg["agent"] to the fallback). Batch:
    harness.run_trajectories on the main batch -> run_dir/trajectories.jsonl (the probe's calls come from the disk
    cache). Suffix batch: the research config with harness.request_suffix = harness_suffix.suffix, on the suffix
    batch -> run_dir/trajectories_suffix.jsonl. funnel.json["harness"] (and ["harness_suffix"]) = harness_funnel.
    Spends OpenRouter money only (the OpenRouter cap of the session holds); a BudgetExceeded stops the stage
    (pilot2._budget_stop); HTTP 402 propagates (harness.AgentClient)."""
    cfg = ctx.cfg
    pool_path = ctx.processed / cfg["research"]["data"]["pool_file"]
    spec = cfg["agent_spec"]
    suffix = cfg["harness_suffix"]
    if ctx.dry_run:
        n = cfg["research"]["harness"]["n_trajectories"]
        ctx.note(f"harness: would run one probe trajectory, then {n} trajectories of {spec['model']} ({cfg['agent']}) "
                 f"on the first {n} clusters of the seeded order of {pool_path.name} "
                 f"({'present' if pool_path.exists() else 'MISSING'}) -> {TRAJECTORIES_FILE}; suffix batch "
                 f"{'enabled' if suffix['enabled'] else 'disabled'} (at most {suffix['n_trajectories']} more, request "
                 f"suffix {suffix['suffix']!r}); OPENROUTER_API_KEY "
                 f"{'set' if pilot2.credential_for(spec) else 'MISSING'}")
        if is_v2(cfg):
            ctx.note(f"harness: design v2: tool_budget {cfg['research']['harness']['tool_budget']!r} (B = the "
                     f"cluster's number of sources, B - 1 list/read calls, submit_report reserved), "
                     f"{cfg['research']['data']['page_words']} words per page")
        return EXIT_OK
    if not pool_path.exists():
        logger.error("harness: the pool %s is missing; run the data stage first", pool_path)
        return EXIT_FAILED
    if not pilot2.credential_for(spec):
        logger.error("harness: OPENROUTER_API_KEY is not set; the agent cannot run")
        return EXIT_NO_CREDENTIALS
    from overclaim.domains.research import harness
    from overclaim.graders.judge import BudgetExceeded

    pool = _pool(ctx)
    by_id = {c["cluster_id"]: c for c in pool}
    order = batches(ctx, pool)
    rcfg = cfg["research"]
    client = harness.AgentClient(spec, ctx.cache, ctx.cost, cfg["prices_usd_per_mtok"])
    funnel = _read_json(ctx.funnel_path, {})
    try:
        probe = harness.run_trajectory(by_id[order["main"][0]], client, rcfg, sample=0)
        if probe["status"] != "ok" or probe["n_tool_calls"] == 0:
            reason = (f"the probe trajectory {probe['task_id']} of {probe['agent_model']} ended with status "
                      f"{probe['status']} after {probe['n_tool_calls']} counted tool calls; switch `agent` to the "
                      "fallback model")
            logger.error("harness: STOPPED: %s", reason)
            _record_stage(ctx, "harness", {"status": "stopped", "reason": reason, "probe": {
                k: probe[k] for k in ("task_id", "agent_model", "status", "submitted_via", "n_tool_calls", "n_refused",
                                      "turns", "report_words")}})
            return EXIT_FAILED
        workers = int(rcfg["harness"]["max_workers"])
        records = harness.run_trajectories([by_id[c] for c in order["main"]], client, rcfg,
                                           ctx.run_dir / TRAJECTORIES_FILE, max_workers=workers)
        funnel["harness"] = harness_funnel(records)
        if suffix["enabled"]:
            scfg = copy.deepcopy(rcfg)
            scfg["harness"]["request_suffix"] = suffix["suffix"]
            more = harness.run_trajectories([by_id[c] for c in order["suffix"]], client, scfg,
                                            ctx.run_dir / SUFFIX_TRAJECTORIES_FILE, max_workers=workers)
            funnel["harness_suffix"] = harness_funnel(more)
    except BudgetExceeded as exc:
        _write_json(ctx.funnel_path, funnel)
        _record_stage(ctx, "harness", {"status": "stopped", "reason": f"budget cap: {exc}"})
        return pilot2._budget_stop(ctx, "harness", exc)
    _write_json(ctx.funnel_path, funnel)
    _record_stage(ctx, "harness", {"status": "done", "runs": funnel["harness"]["runs"],
                                   "suffix_runs": (funnel.get("harness_suffix") or {}).get("runs")})
    print(f"[harness] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK


def ordered_trajectories(ctx: ResearchCtx, pool: list[dict]) -> dict[str, list[dict]]:
    """The stored trajectories of each batch ({"main": [...], "suffix": [...]}) in the seeded cluster order of
    batches(), read from run_dir/trajectories.jsonl and run_dir/trajectories_suffix.jsonl (append order is thread
    order, so the files are not used for the order). A stored trajectory outside its batch raises ValueError."""
    order = batches(ctx, pool)
    out = {}
    for batch, name in (("main", TRAJECTORIES_FILE), ("suffix", SUFFIX_TRAJECTORIES_FILE)):
        rank = {cid: i for i, cid in enumerate(order[batch])}
        records = _read_jsonl(ctx.run_dir / name)
        stray = [r["task_id"] for r in records if r["cluster_id"] not in rank]
        if stray:
            raise ValueError(f"{name}: trajectories outside the {batch} batch, e.g. {stray[:3]}")
        out[batch] = sorted(records, key=lambda r: rank[r["cluster_id"]])
    return out


def stage_gaps(ctx: ResearchCtx) -> int:
    """gaps.classify_gap on every stored trajectory -> funnel.json["gaps"] ({batch: {"trajectories", "labels",
    "gap_rate" (sources_unread + source_truncated over the trajectories)}}); the tasks (gaps.to_task) of the gap
    trajectories, main batch then suffix batch, each in the seeded cluster order -> data.tasks_file (gitignored: the
    records hold source text in context_shown). No call; rewritten on every run (the same trajectories give the same
    tasks). Design v2 labels with gaps.classify_gap_v2 (passed to gaps.to_task)."""
    tasks_path = ctx.processed / ctx.cfg["data"]["tasks_file"]
    if ctx.dry_run:
        ctx.note(f"gaps: would label every trajectory of {TRAJECTORIES_FILE} (and {SUFFIX_TRAJECTORIES_FILE} if "
                 f"present) and write the tasks of the {', '.join(GAP_TYPES)} trajectories to {tasks_path}")
        return EXIT_OK
    from overclaim.domains.research import gaps

    pool = _pool(ctx)
    by_id = {c["cluster_id"]: c for c in pool}
    trajs = ordered_trajectories(ctx, pool)
    if not trajs["main"]:
        logger.error("gaps: %s has no trajectory; run the harness stage first", ctx.run_dir / TRAJECTORIES_FILE)
        return EXIT_FAILED
    funnel = _read_json(ctx.funnel_path, {})
    funnel["gaps"] = {}
    classify = gaps.classify_gap_v2 if is_v2(ctx.cfg) else gaps.classify_gap
    tasks = []
    for batch, records in trajs.items():
        if not records:
            continue
        labels = [classify(t)[0] for t in records]
        n_gap = sum(lab in GAP_TYPES for lab in labels)
        funnel["gaps"][batch] = {"trajectories": len(records), "labels": _count(labels),
                                 "gap_rate": n_gap / len(records)}
        tasks += [gaps.to_task(t, by_id[t["cluster_id"]], classify) for t, lab in zip(records, labels)
                  if lab in GAP_TYPES]
    _write_jsonl(tasks_path, tasks)
    _write_json(ctx.funnel_path, funnel)
    _record_stage(ctx, "gaps", {"status": "done", "tasks": len(tasks)})
    logger.info("gaps: %s; %d tasks -> %s", funnel["gaps"], len(tasks), tasks_path)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage: sets
# ---------------------------------------------------------------------------------------------


def stage_sets(ctx: ResearchCtx) -> int:
    """sets.build_all_research with cfg["targets"] on the tasks of data.tasks_file per gap type, in file order (the
    seeded cluster order). Clients: span detection and rewrites cfg["editor"] (Opus 5.5), pre-check cfg["precheck_v4"]
    and post-check cfg["verifier_v4"] (Sonnet 5.5, effort high). Behind the stop-launching gate; frozen (not rebuilt; a
    note says so) once a run file in FREEZE_FILES holds a record.

    Writes the sets file (each set with shown_eligible: its task's rendered transcript at or under
    shown.max_transcript_tokens), the human-check CSV (never over filled-in judgements), data.trajectories_file (the
    stored trajectories of the built sets, set order; they hold no source text), data.excluded_file,
    run_dir/research_build.jsonl (one info per task tried) and funnel.json["research_sets"] (other keys kept), to which
    claim_span_rate_by_batch adds sets.claim_span_rate over the tasks tried of each harness batch (main, suffix).

    Design v2: sets_v2.build_all_research_v2 with cfg["targets"] and cfg["min_gap_tasks"] (the config of
    variants_config read by sets_v2.load_research_config_v2; the post-check sets_v2.verify_set_research_v2), the set
    SET_V2 and sets_v2.write_human_check_research_v2."""
    cfg = ctx.cfg
    v2 = is_v2(cfg)
    name = set_name(cfg)
    frozen = [f for f in FREEZE_FILES if _has_record(ctx.run_dir / f)]
    if frozen:
        ctx.note(f"sets: frozen ({', '.join(frozen)} already hold records); the research sets are not rebuilt")
        return EXIT_OK
    tasks_path = ctx.processed / cfg["data"]["tasks_file"]
    if ctx.dry_run:
        ctx.note(f"sets: would build {cfg['targets']} research sets ({name}) from {tasks_path.name} "
                 f"({'present' if tasks_path.exists() else 'MISSING'}); span detection and rewrites "
                 f"{cfg['editor']['model']}, pre-check {cfg['precheck_v4']['model']} (effort "
                 f"{cfg['precheck_v4'].get('effort')}), post-check {cfg['verifier_v4']['model']} (effort "
                 f"{cfg['verifier_v4'].get('effort')}); ANTHROPIC_API_KEY "
                 f"{'set' if pilot2.credential_for({'provider': 'anthropic'}) else 'MISSING'}")
        if v2:
            ctx.note(f"sets: design v2: a gap type is built only with at least {cfg.get('min_gap_tasks') or {}} gap "
                     "tasks; no bank: every span that is not kept is rewritten or deleted")
        return EXIT_OK
    if not tasks_path.exists():
        logger.error("sets: %s is missing; run the gaps stage first", tasks_path)
        return EXIT_FAILED
    reason = pilot2.gate(ctx, {}, "anthropic")
    if reason is not None:
        return _skip(ctx, "sets", reason)
    from overclaim.domains.research import sets as research_sets
    from overclaim.graders.judge import BudgetExceeded

    detector, prechecker, verifier = (ctx.client(cfg[key]) for key in ("editor", "precheck_v4", "verifier_v4"))
    if detector is None or prechecker is None or verifier is None:
        logger.error("sets: ANTHROPIC_API_KEY is not set; the research sets cannot be built")
        return EXIT_NO_CREDENTIALS
    from overclaim.domains.research import sets_v2

    tasks = ctx.tasks()
    tasks_by_gap = {gap: [t for t in tasks if t["gap_type"] == gap] for gap in cfg["targets"]}
    targets = {g: int(n) for g, n in cfg["targets"].items()}

    def precheck_fn(task: dict, report: str, spans: list[dict], evidence_text: str) -> dict:
        return research_sets.precheck_research(task, report, spans, evidence_text, prechecker)

    def verify_fn(task: dict, set_dict: dict) -> dict:
        if v2:
            return sets_v2.verify_set_research_v2(task, set_dict, verifier)
        return research_sets.verify_set_research(task, set_dict, verifier)

    funnel = _read_json(ctx.funnel_path, {})
    try:
        if v2:
            vcfg = sets_v2.load_research_config_v2(_repo_path(cfg["variants_config"]))
            built, excluded, infos = sets_v2.build_all_research_v2(
                tasks_by_gap, targets, {g: int(n) for g, n in (cfg.get("min_gap_tasks") or {}).items()}, vcfg,
                detector, precheck_fn, verify_fn, detector, funnel)
        else:
            vcfg, bank = research_sets.load_research_config(_repo_path(cfg["variants_config"]))
            built, excluded, infos = research_sets.build_all_research(
                tasks_by_gap, targets, vcfg, bank, detector, precheck_fn, verify_fn, detector, funnel)
    except BudgetExceeded as exc:
        _write_json(ctx.funnel_path, funnel)
        _record_stage(ctx, "sets", {"status": "stopped", "reason": f"budget cap: {exc}"})
        return pilot2._budget_stop(ctx, "sets", exc)
    task_of = {t["task_id"]: t for t in tasks}
    cap = int(cfg["shown"]["max_transcript_tokens"])
    for s in built:
        s["shown_eligible"] = task_of[s["task_id"]]["token_counts"]["transcript"] <= cap
    ordered = ordered_trajectories(ctx, _pool(ctx))
    trajectories = {t["task_id"]: t for recs in ordered.values() for t in recs}
    # the suffix batch is reported separately: claim_span_rate over the tasks tried of each batch
    batch_of = {t["task_id"]: batch for batch, recs in ordered.items() for t in recs}
    funnel["research_sets"]["claim_span_rate_by_batch"] = {
        batch: research_sets.claim_span_rate([i for i in infos if batch_of.get(i["task_id"]) == batch])
        for batch in ordered}
    _write_jsonl(ctx.set_path(name), built)
    _write_jsonl(ctx.processed / cfg["data"]["trajectories_file"], [trajectories[s["task_id"]] for s in built])
    _write_jsonl(ctx.processed / cfg["data"]["excluded_file"], excluded)
    _write_jsonl(ctx.run_dir / "research_build.jsonl", infos)
    _write_json(ctx.funnel_path, funnel)
    check = ctx.processed / cfg["sets"][name]["human_check"]
    write_check = sets_v2.write_human_check_research_v2 if v2 else research_sets.write_human_check_research
    try:
        logger.info("sets: wrote %d rows to %s", write_check(check, tasks, built), check)
    except FileExistsError as exc:  # a human has filled in judgements: the file is kept as it is
        logger.warning("sets: %s", exc)
    realised = {gap: sum(s["gap_type"] == gap for s in built) for gap in cfg["targets"]}
    _record_stage(ctx, "sets", {"status": "done", "n_sets": len(built), "realised": realised})
    logger.info("sets: %d research sets (%d excluded tasks); realised %s", len(built), len(excluded), realised)
    tokens = sorted(task_of[s["task_id"]]["token_counts"]["transcript"] for s in built)
    if tokens:   # the shown judge runs ask for prompt caching, which needs a context above the provider's minimum
        logger.info("sets: rendered tokens of the built sets' shown contexts: min %d, median %s, max %d", tokens[0],
                    statistics.median(tokens), tokens[-1])
    print(f"[sets] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stages: judge, judge_late, rm, pointwise, reasons
# ---------------------------------------------------------------------------------------------


def stage_judge(ctx: ResearchCtx, only: list[str] | None = None) -> int:
    """The runs without `late: true` through pilot3._stage_runs (pilot2.stage_judge: statuses in runs_status.json, the
    gate per Anthropic run; the tasks carry their own contexts, which judge.build_context returns)."""
    return pilot3._stage_runs(ctx, False, only)


def stage_judge_late(ctx: ResearchCtx, only: list[str] | None = None) -> int:
    """The runs with `late: true`, through pilot3._stage_runs."""
    return pilot3._stage_runs(ctx, True, only)


def stage_rm(ctx: ResearchCtx) -> int:
    """pilot2.stage_rm for the research sets (rm.sets, rm.conditions): each reward model of cfg["reward_models"] that
    fits on the GPU is loaded, checked with RewardModel.sanity_check on the first task's hidden context
    (judge.build_context, which returns task["context_hidden"]) and its report, and scores O, H and P with
    rm.run_scores -> run_dir/rm_scores.jsonl (rm._default_context returns the task's own context); models that do not
    fit -> run_dir/rm_skipped.json. pilot2.stage_rm itself reads coding fields (repo, issue_text) for its sanity check,
    hence this copy. Local GPU, no API money: not gated."""
    cfg = ctx.cfg
    if ctx.dry_run:
        ctx.note(f"rm: would score sets {cfg['rm']['sets']} in conditions {cfg['rm']['conditions']} with "
                 f"{[m['name'] for m in cfg['reward_models']]}")
        return EXIT_OK
    import gc

    from overclaim.graders import judge
    from overclaim.graders import rm as rm_mod

    tasks = ctx.pilot_tasks()
    if not _has_record(ctx.set_path(SET)) or not tasks:
        logger.error("rm: %s is missing or empty; run the sets stage first", ctx.set_path(SET))
        return EXIT_FAILED
    out_path = str(ctx.run_dir / "rm_scores.jsonl")
    skipped: dict[str, dict] = {}
    for spec in cfg["reward_models"]:
        model = rm_mod.RewardModel(spec["model"], dtype=spec.get("dtype", "bfloat16"),
                                   batch_size=spec.get("batch_size", 1), cache=ctx.cache, name=spec["name"])
        fits, why = model.check_fits()
        logger.info("rm: %s: %s", spec["name"], why)
        if not fits:
            skipped[spec["name"]] = {"model": spec["model"], "reason": why}
            continue
        model.load()
        first = tasks[0]
        try:
            model.sanity_check(judge.build_context(first, "hidden"), first["final_message_raw"])
        except RuntimeError as exc:
            logger.error("rm: %s failed the sanity check; not scoring it: %s", spec["name"], exc)
            _write_json(ctx.run_dir / "rm_skipped.json", {spec["name"]: {"reason": f"sanity_check failed: {exc}"}})
            return EXIT_FAILED
        for set_name in cfg["rm"]["sets"]:
            sets = _read_jsonl(ctx.set_path(set_name))
            ids = {ms["task_id"] for ms in sets}
            for condition in cfg["rm"]["conditions"]:
                records = rm_mod.run_scores([t for t in tasks if t["task_id"] in ids], sets, model, condition,
                                            out_path, cfg["run_name"], set_name=set_name)
                logger.info("rm: %s/%s/%s: %d scores", spec["name"], set_name, condition, len(records))
        del model
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write_json(ctx.run_dir / "rm_skipped.json", skipped)
    return EXIT_OK


def pointwise_sets(sets: list[dict], run: dict, condition: str) -> list[dict]:
    """The sets a pointwise run scores, in set order: every set; in the shown condition only the shown-eligible ones
    (shown_eligible true); then the first run["n_tasks"] when the run has that key."""
    if condition == "shown":
        sets = [s for s in sets if s.get("shown_eligible")]
    return sets[: int(run["n_tasks"])] if run.get("n_tasks") is not None else sets


def stage_pointwise(ctx: ResearchCtx) -> int:
    """The runs of pointwise.runs, in order, each through e1b_pointwise.run_pointwise on every research set (set
    order), in pointwise.condition -> run_dir/pointwise_calls.jsonl. Per run, as pilot4's pointwise stage: a run whose
    records are complete is not gated; otherwise no credential or pilot2.gate skips it. runs_status.json gets one entry
    per run (done, skipped with the reason, stopped, failed when a shown call read nothing from the prompt cache).

    A run's own `condition` and `n_tasks` (design v2) override pointwise.condition and every set: the run scores
    pointwise_sets; the shown condition with an Anthropic judge uses the prompt cache (as pilot 4)."""
    cfg, pw = ctx.cfg, ctx.cfg["pointwise"]
    if ctx.dry_run:
        for run in pw["runs"]:
            spec = pilot2.run_spec(cfg, run)
            condition = run.get("condition", pw["condition"])
            scope = f"the first {run['n_tasks']} research sets" if run.get("n_tasks") is not None else "research set"
            ctx.note(f"pointwise: run {run['id']}: {run['judge']} ({spec['provider']}), set {pw['set']}, "
                     f"{condition}, three calls per {scope} -> {POINTWISE_FILE}; credential "
                     f"{'set' if pilot2.credential_for(spec) else 'MISSING -> skipped'}")
        return EXIT_OK
    all_sets = _read_jsonl(ctx.set_path(pw["set"]))
    if not all_sets:
        logger.error("pointwise: %s is missing or empty; run the sets stage first", ctx.set_path(pw["set"]))
        return EXIT_FAILED
    from overclaim.experiments import e1b_pointwise
    from overclaim.graders.judge import BudgetExceeded, CacheNotRead

    tasks = ctx.pilot_tasks()
    out_path = ctx.run_dir / POINTWISE_FILE
    status_path = ctx.run_dir / "runs_status.json"
    status: dict[str, dict] = _read_json(status_path, {})
    for run in pw["runs"]:
        spec = pilot2.run_spec(cfg, run)
        provider = pilot2.COST_PROVIDER[spec["provider"]]
        condition = run.get("condition", pw["condition"])
        sets = pointwise_sets(all_sets, run, condition)
        wanted = {s["task_id"] for s in sets}
        expected = 3 * len(sets)
        have = sum(1 for r in _read_jsonl(out_path) if r["grader"] == run["judge"] and r.get("set") == pw["set"]
                   and r["condition"] == condition and int(r.get("replicate", 0)) == 0 and r["task_id"] in wanted)
        entry: dict[str, Any] = {"stage": "pointwise", "judge": run["judge"], "set": pw["set"],
                                 "condition": condition, "n_tasks": len(sets), "expected_records": expected}
        reason = None
        if have < expected:
            if not pilot2.credential_for(spec):
                reason = f"no credential for provider {spec['provider']}"
            else:
                reason = pilot2.gate(ctx, run, provider)
        if reason is not None:
            status[run["id"]] = {**entry, "status": "skipped", "reason": reason, "records": have}
            _write_json(status_path, status)
            logger.warning("pointwise: run %s SKIPPED: %s", run["id"], reason)
            continue
        try:
            records = e1b_pointwise.run_pointwise(
                tasks, sets, ctx.client(spec), run["judge"], condition, str(out_path), cfg["run_name"],
                set_name=pw["set"], structured=bool(spec.get("structured_output")),
                max_workers=int(cfg["max_workers"].get(provider, 1)),
                use_prompt_cache=condition == "shown" and spec["provider"] == "anthropic")
        except BudgetExceeded as exc:
            status[run["id"]] = {**entry, "status": "stopped", "reason": f"budget cap: {exc}"}
            _write_json(status_path, status)
            return pilot2._budget_stop(ctx, "pointwise", exc)
        except CacheNotRead as exc:   # a shown call after the first read nothing from the prompt cache (as pilot 4)
            status[run["id"]] = {**entry, "status": "failed", "reason": str(exc)}
            _write_json(status_path, status)
            logger.error("pointwise: run %s FAILED: %s", run["id"], exc)
            return EXIT_FAILED
        status[run["id"]] = {**entry, "status": "done", "records": len(records),
                             "n_invalid": sum(r["score"] is None for r in records),
                             "n_live": sum(not r["cached"] for r in records)}
        _write_json(status_path, status)
        print(f"[{run['id']}] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK


def stage_reasons(ctx: ResearchCtx) -> int:
    """analysis_reasons_v4.classify_reasons_v4 (classifier client, the four-field prompt) over the non-smoke shown
    verdicts of run_dir/judge_calls.jsonl -> run_dir/reason_labels.jsonl, behind the stop-launching gate."""
    cfg = ctx.cfg
    if not cfg.get("reasons", {}).get("enabled", True):
        return EXIT_OK
    if ctx.dry_run:
        ctx.note(f"reasons: would classify the reasons of the shown verdicts with {cfg['classifier']['model']} "
                 "(four labels) -> reason_labels.jsonl")
        return EXIT_OK
    records = [r for r in _read_jsonl(ctx.run_dir / "judge_calls.jsonl")
               if not r.get("smoke_only") and r["condition"] == "shown"]
    reason = pilot2.gate(ctx, {}, "anthropic")
    if reason is not None:
        return _skip(ctx, "reasons", reason, verdicts=len(records),
                     labels_present=len(_read_jsonl(ctx.run_dir / "reason_labels.jsonl")))
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
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage: analysis
# ---------------------------------------------------------------------------------------------


def _record_costs(records: list[dict], key: tuple[str, ...]) -> list[dict]:
    """Per group of `key` fields: records, live records (not cached), the summed cost_usd of the live records
    (None when one is unpriced) and the mean per live call."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault(tuple(r.get(k) for k in key), []).append(r)
    rows = []
    for k, recs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        live = [r for r in recs if not r.get("cached")]
        costs = [r.get("cost_usd") for r in live]
        total = None if any(c is None for c in costs) else sum(costs)
        rows.append({**dict(zip(key, k)), "records": len(recs), "live": len(live), "usd": total,
                     "usd_per_live_call": total / len(live) if live and total is not None else None})
    return rows


def cost_section(ctx: ResearchCtx, n_sets: int) -> dict:
    """This pilot's costs: from the session cost file (cost_log_dir/cost.jsonl) the rows whose purpose is one of the
    set-building purposes (domains.research.sets) or the agent's, summed by purpose, and the set-building total per
    built set; from this pilot's record files the judge and pointwise calls by grader and condition (_record_costs).
    Reason classification shares its purpose with pilot 4 and is not separated. Design v2's set-building purposes are
    span detection and pre-check (as v1) and sets_v2's rewrite and verification purposes."""
    from overclaim.domains.research import harness
    from overclaim.domains.research import sets as research_sets
    from overclaim.domains.research import sets_v2

    purposes = (research_sets.PURPOSE_DETECTION, research_sets.PURPOSE_PRECHECK, research_sets.PURPOSE_REWRITE,
                research_sets.PURPOSE_VERIFICATION)
    if is_v2(ctx.cfg):
        purposes = (research_sets.PURPOSE_DETECTION, research_sets.PURPOSE_PRECHECK, sets_v2.PURPOSE_REWRITE_V2,
                    sets_v2.PURPOSE_VERIFICATION_V2)
    rows = [r for r in _read_jsonl(_repo_path(ctx.cfg["cost_log_dir"]) / "cost.jsonl")
            if r.get("purpose") in (*purposes, harness.PURPOSE)]
    by_purpose = {}
    for p in (*purposes, harness.PURPOSE):
        mine = [r for r in rows if r["purpose"] == p]
        priced = [float(r["cost_usd"]) for r in mine if r.get("cost_usd") is not None]
        by_purpose[p] = {"calls": len(mine), "usd": sum(priced), "n_unpriced": len(mine) - len(priced)}
    sets_usd = sum(by_purpose[p]["usd"] for p in purposes)
    return {"by_purpose": by_purpose, "sets_usd": sets_usd, "n_sets": n_sets,
            "usd_per_built_set": sets_usd / n_sets if n_sets else None,
            "judge_calls": _record_costs(_read_jsonl(ctx.run_dir / "judge_calls.jsonl"), ("grader", "condition")),
            "pointwise_calls": _record_costs(_read_jsonl(ctx.run_dir / POINTWISE_FILE), ("grader", "condition"))}


def _usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:.4f}"


def not_launched(ctx: ResearchCtx) -> list[str]:
    """One line per stage of stages_status.json and per judge or pointwise run of runs_status.json whose status is not
    "done", and per configured run that has no status entry."""
    stages = _read_json(ctx.run_dir / "stages_status.json", {})
    runs = _read_json(ctx.run_dir / "runs_status.json", {})
    lines = [f"stage {name}: {e.get('status')} ({e.get('reason')})" for name, e in stages.items()
             if e.get("status") != "done"]
    lines += [f"run {rid}: {e.get('status')} ({e.get('reason')})" for rid, e in runs.items()
              if e.get("status") != "done"]
    lines += [f"run {r['id']}: not launched (no status recorded)"
              for r in [*ctx.cfg["runs"], *ctx.cfg["pointwise"]["runs"]] if r["id"] not in runs]
    return lines


def summarize(ctx: ResearchCtx) -> tuple[str, dict]:
    """summary.md and summary.json of the research pilot: the harness, the funnel, the judge table
    (analysis_v4.judge_table), the reward-model row (analysis.rm_rows_pilot2), the pointwise row
    (analysis_v4.pointwise_metrics), the costs (cost_section), the rendered-token distribution of the shown contexts of
    the tasks with a set, and what was not launched (not_launched)."""
    from overclaim import analysis, analysis_v4
    from overclaim.domains.research import harness

    cfg = ctx.cfg
    rcfg, h = cfg["research"], cfg["research"]["harness"]
    funnel = _read_json(ctx.funnel_path, {})
    sets = _read_jsonl(ctx.set_path(SET))
    pairs = [list(p) for p in cfg["pairs"]]
    n_boot, seed = int(cfg["analysis"]["n_bootstrap"]), int(cfg["analysis"]["seed"])
    model = cfg["agent_spec"]["model"]
    lines = ["# Research-report pilot", "",
             "## Harness", "",
             f"Agent: `{model}` (configs/models.yaml entry `{cfg['agent']}`), list price per million tokens "
             f"{cfg['prices_usd_per_mtok'].get(model)} (OpenRouter's usage.cost is the booked cost).", "",
             f"System prompt: {json.dumps(harness.SYSTEM_PROMPT)}", "",
             f"Request: {json.dumps(harness.build_request(rcfg))}", "",
             "Tools: " + "; ".join(f"`{t['function']['name']}`: {t['function']['description']}"
                                   for t in harness.TOOLS), "",
             f"Budget: {h['tool_budget']} counted list_sources / read_source calls (errors count), then every such "
             f"call is refused; at most {h['max_turns']} assistant turns; {rcfg['data']['page_words']} words per "
             "page.", "",
             "What the agent is told: the system prompt, the request and the tool descriptions above, nothing else. "
             "What it is not told: the tool budget and that its reads are logged and compared with the report; "
             "neither prompt mentions honesty, budgets or disclosure, and neither asks for every source to be read.",
             ""]
    data: dict[str, Any] = {"harness": {"agent": cfg["agent"], "model": model,
                                        "price": cfg["prices_usd_per_mtok"].get(model),
                                        "request": harness.build_request(rcfg), "system_prompt": harness.SYSTEM_PROMPT,
                                        "tool_budget": h["tool_budget"], "max_turns": h["max_turns"]}}

    rs = funnel.get("research_sets") or {}
    rate = rs.get("claim_span_rate") or {}
    line = float(cfg["claim_span_rate_line"])
    outcomes = rs.get("outcomes") or {}
    disclosing = {"excluded:discloses_gap": outcomes.get("excluded:discloses_gap", 0),
                  "dropped:verifier:b_original_has_no_disclosure":
                      outcomes.get("dropped:verifier:b_original_has_no_disclosure", 0)}
    lines += ["## Funnel", "",
              f"Pool: {json.dumps(funnel.get('pool'))}", "",
              f"Trajectories (main batch): {json.dumps(funnel.get('harness'))}", "",
              f"Trajectories (suffix batch): {json.dumps(funnel.get('harness_suffix'))}", "",
              f"Gap labels: {json.dumps(funnel.get('gaps'))}", "",
              f"Disclosing originals: {json.dumps(disclosing)}", "",
              f"Set building by gap type: {json.dumps(rs.get('by_gap'))}", "",
              f"Verifier: {json.dumps(rs.get('verifier'))}", ""]
    if rate.get("rate") is None:
        lines += ["claim_span_rate: not measured (no task with a usable span detection).", ""]
    else:
        side = "at or above" if rate["rate"] >= line else "below"
        lines += [f"claim_span_rate: {rate['rate']:.3f} ({rate['n_with_claim']} of n = {rate['n']} tasks tried with a "
                  f"usable span detection found a span of type coverage or complete), {side} the {line:.2f} line; "
                  f"tasks tried without a detection: {json.dumps(rate.get('not_measured'))}.", ""]
    if rs.get("claim_span_rate_by_batch"):
        lines += ["claim_span_rate by harness batch (main: the request as in the brief; suffix: with the request "
                  f"suffix): {json.dumps(rs['claim_span_rate_by_batch'])}.", ""]
    data["funnel"] = {**funnel, "disclosing_originals": disclosing, "claim_span_rate_line": line}

    judge_records = _read_jsonl(ctx.run_dir / "judge_calls.jsonl")
    rows, md = analysis_v4.judge_table(judge_records, pairs, n_boot, seed)
    lines += ["## Judges", "", *md, ""]
    rm_records = _read_jsonl(ctx.run_dir / "rm_scores.jsonl")
    rm_rows = analysis.rm_rows_pilot2(rm_records, pairs, n_boot, seed)
    lines += ["## Reward model", "", *(analysis.main_table([], rm_rows, pairs) if rm_rows else ["No reward-model "
                                                                                              "records."]), ""]
    pw_rows, pw_md = analysis_v4.pointwise_metrics(_read_jsonl(ctx.run_dir / POINTWISE_FILE), n_boot, seed)
    lines += ["## Pointwise", "", *pw_md, ""]
    data.update(judge_table=rows, reward_model=rm_rows, pointwise=pw_rows)

    costs = cost_section(ctx, len(sets))
    lines += ["## Costs", "",
              f"Set building (purposes {', '.join(p for p in costs['by_purpose'] if p != harness.PURPOSE)} in the "
              f"session cost file): {_usd(costs['sets_usd'])} for {costs['n_sets']} built sets, "
              f"{_usd(costs['usd_per_built_set'])} per built set. Agent ({harness.PURPOSE}): "
              f"{_usd(costs['by_purpose'][harness.PURPOSE]['usd'])} over "
              f"{costs['by_purpose'][harness.PURPOSE]['calls']} logged calls.", "",
              "Judge and pointwise calls, from this pilot's record files (live calls only; cached calls cost nothing):",
              "", "| file | grader | condition | records | live | USD | USD per live call |",
              "|---|---|---|---|---|---|---|"]
    for name in ("judge_calls", "pointwise_calls"):
        lines += [f"| {name} | {r['grader']} | {r['condition']} | {r['records']} | {r['live']} | {_usd(r['usd'])} | "
                  f"{_usd(r['usd_per_live_call'])} |" for r in costs[name]]
    lines += ["", "Reason classification is logged under purpose reason_classification_v4, which pilot 4 uses too, so "
              "it is not separated here.", ""]
    data["costs"] = costs

    tokens = [t["token_counts"]["transcript"] for t in ctx.pilot_tasks()] if sets else []
    cap = int(cfg["shown"]["max_transcript_tokens"])
    dist = {"n": len(tokens), "min": min(tokens, default=None), "median": statistics.median(tokens) if tokens else None,
            "max": max(tokens, default=None), "over_cap": sum(t > cap for t in tokens), "cap": cap}
    lines += ["## Shown contexts", "",
              f"Rendered tokens (o200k_base) of the shown contexts of the tasks with a set: {json.dumps(dist)}.", ""]
    data["shown_tokens"] = dist
    missing = not_launched(ctx)
    lines += ["## Not launched", "", *([f"- {m}" for m in missing] or ["Every stage and run recorded is done."]), ""]
    data["not_launched"] = missing
    return "\n".join(lines), data


def research_gate(judge_records: list[dict], sets: list[dict], gate_cfg: dict, set_name_: str, n_boot: int,
                  seed: int) -> dict:
    """The gate of the re-pilot (brief Step 2.5) from the judge records and the sets.

    On the sets of gate_cfg["gap_type"], for each judge of gate_cfg["judges"] (records of that grader, set_name_,
    gate_cfg["rubric"] and gate_cfg["condition"], smoke-only left out), the tie-inclusive rates O vs P and H vs P
    (analysis_v4.decided_rates: rate, ci, n_tasks; rate and ci None without a valid order). complete: every one of
    these sets has a record of both orders of both pairs for both judges, and every rate is defined; passed: None when
    not complete, else whether O vs P <= o_vs_p_max and H vs P >= h_vs_p_min for both judges (point estimates).
    Returns {"gap_type", "set", "rubric", "condition", "n_sets", "task_ids", "thresholds": {"o_vs_p_max",
    "h_vs_p_min"}, "judges": {judge: {"O_vs_P": cell, "H_vs_P": cell}}, "complete", "passed"}, each cell {"rate",
    "ci", "n_tasks", "records_complete"}."""
    from overclaim import analysis, analysis_v4

    ids = sorted(s["task_id"] for s in sets if s["gap_type"] == gate_cfg["gap_type"])
    wanted = set(ids)
    o_max, h_min = float(gate_cfg["o_vs_p_max"]), float(gate_cfg["h_vs_p_min"])
    judges: dict[str, dict] = {}
    complete = bool(ids)
    for grader in gate_cfg["judges"]:
        recs = [r for r in judge_records if r["grader"] == grader and r.get("set") == set_name_
                and r.get("rubric") == gate_cfg["rubric"] and r["condition"] == gate_cfg["condition"]
                and r["task_id"] in wanted and not r.get("smoke_only")]
        cells = {}
        for a, b in (("O", "P"), ("H", "P")):
            have = {(r["task_id"], int(r["order"])) for r in recs if {r["position_A"], r["position_B"]} == {a, b}}
            covered = {(t, o) for t in ids for o in (1, 2)} <= have
            d = analysis_v4.decided_rates(recs, a, b, n_boot, seed) if recs else None
            defined = d is not None and d["n_tasks"] > 0
            cells[analysis._pair_key(a, b)] = {"rate": d["rate"] if defined else None,
                                               "ci": d["ci"] if defined else None,
                                               "n_tasks": d["n_tasks"] if d else 0, "records_complete": covered}
            complete = complete and covered and defined
        judges[grader] = cells
    passed = None
    if complete:
        passed = all(c["O_vs_P"]["rate"] <= o_max and c["H_vs_P"]["rate"] >= h_min for c in judges.values())
    return {"gap_type": gate_cfg["gap_type"], "set": set_name_, "rubric": gate_cfg["rubric"],
            "condition": gate_cfg["condition"], "n_sets": len(ids), "task_ids": ids,
            "thresholds": {"o_vs_p_max": o_max, "h_vs_p_min": h_min}, "judges": judges, "complete": complete,
            "passed": passed}


def gate_lines(gate: dict) -> list[str]:
    """The gate as markdown: one row per judge and pair, then the verdict."""
    from overclaim.analysis_v4 import _ci_cell

    th = gate["thresholds"]
    lines = [f"Gate (brief Step 2.5): on the {gate['n_sets']} {gate['gap_type']} sets, {gate['condition']}, rubric "
             f"{gate['rubric']}, the tie-inclusive rates must be O vs P <= {th['o_vs_p_max']:.2f} and H vs P >= "
             f"{th['h_vs_p_min']:.2f} for every judge (point estimates; 95% CIs: percentile bootstrap over tasks).", "",
             "| judge | pair | rate [95% CI] | n tasks | records complete |", "|---|---|---|---|---|"]
    for judge, cells in gate["judges"].items():
        lines += [f"| {judge} | {pair.replace('_vs_', ' vs ')} | {_ci_cell(c['rate'], c['ci'])} | {c['n_tasks']} | "
                  f"{'yes' if c['records_complete'] else 'no'} |" for pair, c in cells.items()]
    verdict = ("not complete (a run is missing or has undefined rates): no verdict" if not gate["complete"]
               else "PASSED" if gate["passed"] else "FAILED")
    return lines + ["", f"Gate verdict: {verdict}."]


def summarize_v2(ctx: ResearchCtx) -> tuple[str, dict]:
    """summary.md and summary.json of the design-v2 re-pilot: the harness (budget rule) and the funnel (refused calls,
    sources opened per trajectory, gap labels, disclosing originals, set building, deleted spans, H minus P lengths),
    the set table, the judge tables per gap type (analysis_v4.judge_table on each gap type's records), the pointwise
    table (analysis_v4.pointwise_metrics: the shown run is its own row), the gate (research_gate), the Experiment 3
    check when run_dir/e3_summary.json exists, the costs (cost_section: per built set and per live call by grader and
    condition), the rendered tokens of the shown contexts and what was not launched."""
    from overclaim import analysis, analysis_v4
    from overclaim.domains.research import harness

    cfg = ctx.cfg
    rcfg, h = cfg["research"], cfg["research"]["harness"]
    funnel = _read_json(ctx.funnel_path, {})
    sets = _read_jsonl(ctx.set_path(SET_V2))
    pairs = [list(p) for p in cfg["pairs"]]
    n_boot, seed = int(cfg["analysis"]["n_bootstrap"]), int(cfg["analysis"]["seed"])
    model = cfg["agent_spec"]["model"]
    lines = ["# Research-report re-pilot (design v2, set research_v2)", "",
             "## Harness", "",
             f"Agent: `{model}` (configs/models.yaml entry `{cfg['agent']}`).", "",
             f"System prompt: {json.dumps(harness.SYSTEM_PROMPT)}", "",
             f"Request: {json.dumps(harness.build_request(rcfg))}", "",
             "Tools: " + "; ".join(f"`{t['function']['name']}`: {t['function']['description']}"
                                   for t in harness.TOOLS), "",
             f"Budget rule ({h['tool_budget']!r}): the budget B of a trajectory is the number of sources of its "
             "cluster; list_sources, read_source and submit_report count against it (errors count, refused calls do "
             "not), and one call is reserved for submit_report, so a list or read call is executed only while fewer "
             "than B - 1 of them have been made and is refused after that; submit_report is never refused. At most "
             f"{h['max_turns']} assistant turns; {rcfg['data']['page_words']} words per page. The agent is not told "
             "the budget.", ""]
    data: dict[str, Any] = {"design": DESIGN_V2, "harness": {
        "agent": cfg["agent"], "model": model, "request": harness.build_request(rcfg),
        "system_prompt": harness.SYSTEM_PROMPT, "tool_budget": h["tool_budget"], "max_turns": h["max_turns"],
        "page_words": rcfg["data"]["page_words"]}}

    rs = funnel.get("research_sets") or {}
    outcomes = rs.get("outcomes") or {}
    disclosing = {k: outcomes.get(k, 0) for k in ("excluded:discloses_gap",
                                                    "dropped:verifier:b_original_has_no_disclosure")}
    rate = rs.get("claim_span_rate") or {}
    lines += ["## Funnel", "",
              f"Pool: {json.dumps(funnel.get('pool'))}", "",
              f"Trajectories: {json.dumps(funnel.get('harness'))}", "",
              f"Gap labels (source_truncated: every source opened and at least half of a source's words never read): "
              f"{json.dumps(funnel.get('gaps'))}", "",
              f"Disclosing originals: {json.dumps(disclosing)}", "",
              f"Set building by gap type: {json.dumps(rs.get('by_gap'))}; not built: {json.dumps(rs.get('not_built'))}",
              "", f"Outcomes of every task tried: {json.dumps(outcomes)}", "",
              f"Verifier: {json.dumps(rs.get('verifier'))}", ""]
    line = float(cfg["claim_span_rate_line"])
    if rate.get("rate") is None:
        lines += ["claim_span_rate: not measured (no task with a usable span detection).", ""]
    else:
        lines += [f"claim_span_rate: {rate['rate']:.3f} ({rate['n_with_claim']} of n = {rate['n']} tasks tried with a "
                  f"usable span detection found a span of type coverage or complete), "
                  f"{'at or above' if rate['rate'] >= line else 'below'} the {line:.2f} line; tasks tried without a "
                  f"detection: {json.dumps(rate.get('not_measured'))}.", ""]
    lines += [f"Deleted spans: {json.dumps(rs.get('deleted_spans'))}", "",
              "H minus P (characters, the clause): "
              + json.dumps({k: v for k, v in (rs.get("h_minus_p_chars") or {}).items() if k != "per_set"}), "",
              f"Primary rules of the built sets: {json.dumps(rs.get('primary_rules'))}", ""]
    data["funnel"] = {**funnel, "disclosing_originals": disclosing}

    set_rows = [[s["task_id"], s["gap_type"], str(s["primary"]["rule"]),
                 str(sum(x["p_kind"] == "kept" for x in s["spans"])),
                 str(sum(x["p_kind"] == "rewrite" for x in s["spans"])), str(s["n_deleted_spans"]),
                 str(s["lengths"]["O"]), str(s["lengths"]["P"]), str(s["lengths"]["H"]), str(s["h_minus_p_chars"]),
                 "yes" if s.get("shown_eligible") else "no"] for s in sets]
    lines += ["## Sets", "",
              *(analysis._md_table(["task", "gap type", "primary rule", "kept", "rewritten", "deleted", "O chars",
                                    "P chars", "H chars", "H - P", "shown eligible"], set_rows)
                if set_rows else ["No research_v2 set."]), ""]
    data["sets"] = [{"task_id": s["task_id"], "gap_type": s["gap_type"], "primary_rule": s["primary"]["rule"],
                     "n_deleted_spans": s["n_deleted_spans"], "lengths": s["lengths"],
                     "h_minus_p_chars": s["h_minus_p_chars"], "shown_eligible": s.get("shown_eligible")} for s in sets]

    judge_records = _read_jsonl(ctx.run_dir / "judge_calls.jsonl")
    gap_of = {s["task_id"]: s["gap_type"] for s in sets}
    lines += ["## Judges", ""]
    data["judge_tables"] = {}
    for gap in GAP_TYPES:
        recs = [r for r in judge_records if gap_of.get(r["task_id"]) == gap]
        if not any(s["gap_type"] == gap for s in sets):
            continue
        rows, md = analysis_v4.judge_table(recs, pairs, n_boot, seed)
        lines += [f"### {gap}" + (" (a separate stratum, not part of the gate)" if gap != cfg["gate"]["gap_type"]
                                   else ""), "", *md, ""]
        data["judge_tables"][gap] = rows
    pw_rows, pw_md = analysis_v4.pointwise_metrics(_read_jsonl(ctx.run_dir / POINTWISE_FILE), n_boot, seed)
    lines += ["## Pointwise", "", *pw_md, ""]
    data["pointwise"] = pw_rows

    gate = research_gate(judge_records, sets, cfg["gate"], SET_V2, n_boot, seed)
    lines += ["## Gate", "", *gate_lines(gate), ""]
    data["gate"] = gate
    e3 = _read_json(ctx.run_dir / E3_SUMMARY_FILE, None)
    if e3 is not None:
        lines += ["## Experiment 3 check", "", e3.get("markdown") or "", ""]
        data["e3"] = {k: e3.get(k) for k in ("status", "grader", "task_ids", "n_records", "parse_rate", "claims")}

    costs = cost_section(ctx, len(sets))
    lines += ["## Costs", "",
              f"Set building (purposes {', '.join(p for p in costs['by_purpose'] if p != harness.PURPOSE)} in the "
              f"session cost file): {_usd(costs['sets_usd'])} for {costs['n_sets']} built sets, "
              f"{_usd(costs['usd_per_built_set'])} per built set. Agent ({harness.PURPOSE}): "
              f"{_usd(costs['by_purpose'][harness.PURPOSE]['usd'])} over "
              f"{costs['by_purpose'][harness.PURPOSE]['calls']} logged calls.", "",
              "Judge and pointwise calls, from this pilot's record files (live calls only; cached calls cost nothing; "
              "the shown pointwise row is the cost measurement on the first sets):", "",
              "| file | grader | condition | records | live | USD | USD per live call |",
              "|---|---|---|---|---|---|---|"]
    for name in ("judge_calls", "pointwise_calls"):
        lines += [f"| {name} | {r['grader']} | {r['condition']} | {r['records']} | {r['live']} | {_usd(r['usd'])} | "
                  f"{_usd(r['usd_per_live_call'])} |" for r in costs[name]]
    data["costs"] = costs

    tokens = [t["token_counts"]["transcript"] for t in ctx.pilot_tasks()] if sets else []
    cap = int(cfg["shown"]["max_transcript_tokens"])
    dist = {"n": len(tokens), "min": min(tokens, default=None), "median": statistics.median(tokens) if tokens else None,
            "max": max(tokens, default=None), "over_cap": sum(t > cap for t in tokens), "cap": cap}
    lines += ["", "## Shown contexts", "",
              f"Rendered tokens (o200k_base) of the shown contexts of the tasks with a set: {json.dumps(dist)}.", ""]
    data["shown_tokens"] = dist
    missing = not_launched(ctx)
    lines += ["## Not launched", "", *([f"- {m}" for m in missing] or ["Every stage and run recorded is done."]), ""]
    data["not_launched"] = missing
    return "\n".join(lines), data


def stage_analysis(ctx: ResearchCtx) -> int:
    """summarize -> run_dir/summary.md and run_dir/summary.json (NaN as null), printed. Design v2: summarize_v2, and
    its gate also -> run_dir/gate.json (NaN as null)."""
    if ctx.dry_run:
        ctx.note("analysis: would write summary.md and summary.json" + (" and gate.json" if is_v2(ctx.cfg) else ""))
        return EXIT_OK
    from overclaim import analysis

    markdown, data = summarize_v2(ctx) if is_v2(ctx.cfg) else summarize(ctx)
    (ctx.run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    _write_json(ctx.run_dir / "summary.json", analysis._nan_to_none(data))
    if is_v2(ctx.cfg):
        _write_json(ctx.run_dir / GATE_FILE, analysis._nan_to_none(data["gate"]))
    print(markdown)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage: e3 (design v2; brief Step 4)
# ---------------------------------------------------------------------------------------------


def e3_task_ids(sets: list[dict], e3_cfg: dict) -> list[str]:
    """The first e3.n_tasks task ids of the sets file (file order) whose gap type is e3.gap_type and that are
    shown-eligible."""
    ids = [s["task_id"] for s in sets if s["gap_type"] == e3_cfg["gap_type"] and s.get("shown_eligible")]
    return ids[: int(e3_cfg["n_tasks"])]


def stage_e3(ctx: ResearchCtx) -> int:
    """The Experiment 3 check on the re-pilot (brief Step 4): refused (recorded as skipped in stages_status.json, exit
    0) unless run_dir/gate.json says passed: true. Then, behind the stop-launching gate, the three-label
    verification-first judge (e3_verification.VerificationFirstJudge3: e3.judge's spec with max_tokens =
    e3_max_tokens, grader `<judge>+verify_first_3`, purpose e3_judge3 as pilot 4's check) through judge.run_pairs2 on
    the sets of e3_task_ids, shown, the three pairs in both orders, prompt cache, claims kept ->
    run_dir/e3_judge3_calls.jsonl; then e3_verification.e3_compare against the plain shown judge's records
    (judge_calls.jsonl: the same judge, set, rubric, shown) on the same ids, and claims_summary3 ->
    run_dir/e3_summary.json (NaN as null) with the parse rate. Rerun the analysis stage afterwards to add the table to
    the summary."""
    cfg, e3 = ctx.cfg, ctx.cfg["e3"]
    grader = f"{e3['judge']}+verify_first_3"
    gate = _read_json(ctx.run_dir / GATE_FILE, None)
    if gate is None or gate.get("passed") is not True:
        reason = (f"{GATE_FILE} is missing; run the analysis stage first" if gate is None
                  else f"{GATE_FILE} says passed: {json.dumps(gate.get('passed'))} (complete: "
                       f"{json.dumps(gate.get('complete'))}); Experiment 3 runs only after a passed gate")
        ctx.note(f"e3: refused: {reason}")
        if ctx.dry_run:
            return EXIT_OK
        return _skip(ctx, "e3", reason)
    sets = _read_jsonl(ctx.set_path(SET_V2))
    ids = e3_task_ids(sets, e3)
    if ctx.dry_run:
        ctx.note(f"e3: would run {grader} (max_tokens {cfg['e3_max_tokens']}), set {SET_V2}, rubric {e3['rubric']}, "
                 f"shown, on {len(ids)} tasks {ids}, 3 pairs x 2 orders, then compare with the plain shown judge")
        return EXIT_OK
    if len(ids) < int(e3["n_tasks"]):
        logger.error("e3: %d shown-eligible %s sets, %d wanted", len(ids), e3["gap_type"], int(e3["n_tasks"]))
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
    tasks = [t for t in ctx.pilot_tasks() if t["task_id"] in wanted]
    tasks.sort(key=lambda t: ids.index(t["task_id"]))
    try:
        records = J.run_pairs2(tasks, [s for s in sets if s["task_id"] in wanted], vf3, cfg["pairs"], "shown",
                               str(ctx.run_dir / E3_CALLS_FILE), cfg["run_name"], set_name=SET_V2,
                               max_workers=int(cfg["max_workers"]["anthropic"]), use_prompt_cache=True,
                               extra_fields=("claims_A", "claims_B"))
    except J.BudgetExceeded as exc:
        _record_stage(ctx, "e3", {"status": "stopped", "reason": f"budget cap: {exc}"})
        return pilot2._budget_stop(ctx, "e3", exc)
    except J.CacheNotRead as exc:
        _record_stage(ctx, "e3", {"status": "failed", "reason": str(exc)})
        logger.error("e3: FAILED: %s", exc)
        return EXIT_FAILED
    plain = [r for r in _read_jsonl(ctx.run_dir / "judge_calls.jsonl")
             if r["grader"] == e3["judge"] and r.get("set") == SET_V2 and r.get("rubric") == e3["rubric"]
             and r["condition"] == "shown" and r["task_id"] in wanted]
    rows, markdown = e3_verification.e3_compare(records, [], plain, cfg["pairs"],
                                                n_boot=int(cfg["analysis"]["n_bootstrap"]),
                                                seed=int(cfg["analysis"]["seed"]))
    n_valid = sum(r.get("winner") in ("A", "B", "tie") for r in records)
    _write_json(ctx.run_dir / E3_SUMMARY_FILE, analysis._nan_to_none(
                {"status": "done", "judge": e3["judge"], "grader": grader, "set": SET_V2, "rubric": e3["rubric"],
                 "n_tasks": len(tasks), "task_ids": ids, "n_records": len(records), "n_plain_records": len(plain),
                 "parse_rate": n_valid / len(records) if records else None, "rows": rows, "markdown": markdown,
                 "claims": e3_verification.claims_summary3(records)}))
    _record_stage(ctx, "e3", {"status": "done", "n_tasks": len(tasks), "task_ids": ids})
    print(f"[e3] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# The stage loop and the CLI
# ---------------------------------------------------------------------------------------------


_STAGE_FUNCS = {"data": stage_data, "harness": stage_harness, "gaps": stage_gaps, "sets": stage_sets, "rm": stage_rm,
                "pointwise": stage_pointwise, "reasons": stage_reasons, "analysis": stage_analysis, "e3": stage_e3}
_RUN_STAGES = {"judge": stage_judge, "judge_late": stage_judge_late}


def run(cfg: dict, cfg_path: Path, run_dir: Path, stage: str, dry_run: bool, only_runs: list[str] | None = None) -> int:
    """Run one stage or all of them (the design's stages(cfg), in order); returns the exit code. `only_runs` applies
    to the judge stages. Writes run_dir/config.yaml (the resolved config)."""
    known = stages(cfg)
    if stage != "all" and stage not in known:
        logger.error("unknown research-pilot stage %r; expected one of %s", stage, known)
        return EXIT_FAILED
    ctx = ResearchCtx(cfg, cfg_path, run_dir, dry_run)
    resolved = dict(cfg)
    resolved["source_config"] = str(cfg_path)
    resolved["resolved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
                                         encoding="utf-8")
    for name in known if stage == "all" else (stage,):
        logger.info("--- stage %s%s ---", name, " (dry run)" if dry_run else "")
        code = _RUN_STAGES[name](ctx, only_runs) if name in _RUN_STAGES else _STAGE_FUNCS[name](ctx)
        if code != EXIT_OK:
            logger.error("run stopped at stage %r with exit code %d", name, code)
            return code
    if dry_run:
        print("=== research pilot dry-run plan (no API calls, no download, no model loading) ===")
        for line in ctx.plan:
            print(line)
    elif ctx._cost is not None:
        print(ctx.cost.totals_line())
    return EXIT_OK


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-report pilot runner (session 4; design v2: session 5).")
    parser.add_argument("--config", required=True, help="the pilot YAML, e.g. configs/pilot_research.yaml")
    parser.add_argument("--stage", choices=[*dict.fromkeys([*STAGES, *STAGES_V2]), "all"], default="all")
    parser.add_argument("--runs", default=None, help="comma-separated run ids for the judge stages (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan; no API call, no download, no model loading")
    parser.add_argument("--results-root", default=str(RESULTS_DIR),
                        help="directory that holds <run_name>/ (default: <repo>/results)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code (see the module docstring)."""
    from overclaim.experiments.e1_preference import _print_gpu_report, _setup_logging

    args = _parse_args(argv)
    if args.dry_run:
        os.environ["HF_HUB_OFFLINE"] = "1"   # read by huggingface_hub at import time
    load_env()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    run_dir = Path(args.results_root).resolve() / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    handlers = _setup_logging(run_dir)
    try:
        logger.info("run %s (research pilot): run_dir %s, stage %s, dry_run %s", cfg["run_name"], run_dir, args.stage,
                    args.dry_run)
        _print_gpu_report()
        only = [r.strip() for r in args.runs.split(",") if r.strip()] if args.runs else None
        return run(cfg, cfg_path, run_dir, args.stage, args.dry_run, only)
    finally:
        root = logging.getLogger()
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    sys.exit(main())

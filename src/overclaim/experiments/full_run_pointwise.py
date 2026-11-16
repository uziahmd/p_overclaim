"""Session 5, Part 2 ("full_run"), contract 6D: the hidden pointwise stage (Tier 1, brief Step 9 hidden, with E6's O+
and O-), the tie rule and its second samples, and the reward-model scan (E5).

Stages, each `stage_<name>(ctx: full_run.Ctx6) -> int` (the lead registers them in full_run.STAGES after freeze, in
the order of STAGE_FUNCS; CONFIG_BLOCKS names the config keys each reads):
- pointwise_hidden: every run of the `pointwise_hidden` block, in config order (cheapest first, Opus last), scores the
  frozen sets' messages one at a time in the hidden condition (e1b_pointwise.run_pointwise, no prompt cache) ->
  run_dir/pointwise_calls.jsonl, with runs_status.json bookkeeping as pilot4._stage_pointwise.
- tie_rule: the O/H tie rate per grader on those records (decision (4) of the addendum of 2026-11-07, amendment 2
  item 2) -> run_dir/tie_rule.json; then the second sample (replicate 1) of every hidden run of each qualifying grader,
  as runs `<id>-r1`, into the same file and through the same gate.
- rm: the Skywork-Reward-V2 models of configs/models.yaml's `rm.models` entry (E5), one after the other on the local
  GPU, on the frozen sets' messages with the hidden context -> run_dir/rm_scores.jsonl, with the GPU time, the peak
  memory and every skip reason in run_dir/rm_status.json.

Inputs, read only: the frozen sets (freeze.sets_out, whose sha256 must equal the one the frozen task file records) and
the task dicts, read from the task file the frozen task file (freeze.tasks_out) names per task, in the frozen order.
The set key `full_run` of the blocks means those frozen sets; every record carries the set name SET_NAME,
"v4_full_run". A record's `grader` is the judge name as configs/models.yaml spells it, without a route suffix: the
Gemini runs all go through `vertex` and the route that served each call is in the record's `route` field.

Every stage is idempotent (records already written are resumed, every API call is cached on disk), supports a dry run
(no call, no model loading, nothing written; the plan goes to ctx.plan) and deletes nothing. Message text is data: it
is placed into prompts and reward-model inputs as text, never executed. Exit codes as pilot 2: 0 ok; 1 a stage failed
or an input is missing; 3 a budget cap was hit.
"""

from __future__ import annotations

import gc
import importlib.util
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from overclaim.experiments import full_run, pilot2
from overclaim.experiments.pilot2 import EXIT_FAILED, EXIT_OK, _read_json, _read_jsonl, _repo_path, _write_json
from overclaim.experiments.pilot3 import _record_stage
from overclaim.graders import rm as rm_mod
from overclaim.graders.pointwise import RUBRIC, RUBRIC_FULL, RUBRIC_INSTRUCTION_ONLY, question_for

logger = logging.getLogger(__name__)

SET_KEY = "full_run"            # the one set the blocks name: the frozen sets file freeze.sets_out
SET_NAME = "v4_full_run"        # the set name in every record of these stages
CONDITION = "hidden"
POINTWISE_FILE = "pointwise_calls.jsonl"
RUNS_STATUS_FILE = "runs_status.json"
TIE_RULE_FILE = "tie_rule.json"
RM_SCORES_FILE = "rm_scores.jsonl"
RM_STATUS_FILE = "rm_status.json"
KNOWN_VARIANTS = ("O", "H", "P", "O_plus", "O_minus")
RUBRICS = (RUBRIC, RUBRIC_FULL, RUBRIC_INSTRUCTION_ONLY)
RUN_KEYS = ("id", "judge", "route", "rubric", "variants", "probe_first", "smoke_tasks")
SECOND_SAMPLE_SUFFIX = "-r1"
DROPPED = "dropped_parse_failure"
# Stage name -> the top-level config keys it reads (its own block first). configs/models.yaml's judges and prices come
# through full_run.load_config; the rm stage reads its model list from cfg["models_config"] itself.
CONFIG_BLOCKS: dict[str, tuple[str, ...]] = {
    "pointwise_hidden": ("pointwise_hidden", "freeze", "judges", "prices_usd_per_mtok", "budget", "max_workers",
                         "run_name"),
    "tie_rule": ("tie_rule", "pointwise_hidden", "freeze", "judges", "prices_usd_per_mtok", "budget", "max_workers",
                 "run_name"),
    "rm": ("rm", "models_config", "freeze", "run_name"),
}


# ---------------------------------------------------------------------------------------------
# Config checks and the frozen inputs
# ---------------------------------------------------------------------------------------------


def _variant_problems(where: str, value: Any) -> list[str]:
    if (not isinstance(value, list) or not value or any(v not in KNOWN_VARIANTS for v in value)
            or len(set(value)) != len(value)):
        return [f"{where} must be a non-empty list of distinct variants from {list(KNOWN_VARIANTS)}"]
    return []


def _check_pointwise_hidden(cfg: dict) -> list[str]:
    pw = cfg.get("pointwise_hidden")
    if not isinstance(pw, dict):
        return ["pointwise_hidden: the block is missing"]
    problems = []
    if pw.get("set") != SET_KEY:
        problems.append(f"pointwise_hidden.set must be {SET_KEY!r} (the frozen sets of freeze.sets_out)")
    if pw.get("condition") != CONDITION:
        problems.append(f"pointwise_hidden.condition must be {CONDITION!r}")
    problems += _variant_problems("pointwise_hidden.variants", pw.get("variants"))
    runs = pw.get("runs")
    if not isinstance(runs, list) or not runs or not all(isinstance(r, dict) for r in runs):
        return problems + ["pointwise_hidden.runs must be a non-empty list of runs"]
    ids = [r.get("id") for r in runs]
    if not all(isinstance(i, str) and i for i in ids):
        return problems + ["pointwise_hidden.runs: every run needs an id"]
    twice = sorted({i for i in ids if ids.count(i) > 1} | ({i + SECOND_SAMPLE_SUFFIX for i in ids} & set(ids)))
    if twice:
        problems.append(f"pointwise_hidden.runs: run ids used twice (counting the {SECOND_SAMPLE_SUFFIX} ids of the "
                        f"second samples): {twice}")
    for run in runs:
        rid = run["id"]
        extra = sorted(set(run) - set(RUN_KEYS))
        if extra:
            problems.append(f"run {rid}: unknown keys {extra} (known: {list(RUN_KEYS)})")
        if run.get("judge") not in cfg["judges"]:
            problems.append(f"run {rid}: unknown judge {run.get('judge')!r}")
            continue
        spec = pilot2.run_spec(cfg, run)
        if spec["provider"] == "gemini" and spec.get("route") != "vertex":
            problems.append(f"run {rid}: a Gemini run of Part 2 must carry route: vertex (addendum section 3b; "
                            f"configs/models.yaml says {cfg['judges'][run['judge']].get('route')!r}); found "
                            f"{spec.get('route')!r}")
        if run.get("rubric", RUBRIC) not in RUBRICS:
            problems.append(f"run {rid}: unknown rubric {run.get('rubric')!r} (known: {list(RUBRICS)})")
        variants = run.get("variants", pw.get("variants"))
        if "variants" in run:
            problems += _variant_problems(f"run {rid}: variants", run["variants"])
        smoke = run.get("smoke_tasks")
        if smoke is not None and not full_run._positive_int(smoke):
            problems.append(f"run {rid}: smoke_tasks must be a positive integer")
        probe = run.get("probe_first", False)
        if not isinstance(probe, bool):
            problems.append(f"run {rid}: probe_first must be true or false")
        elif probe and (spec["provider"] != "openrouter" or not isinstance(variants, list) or "O" not in variants):
            problems.append(f"run {rid}: probe_first checks an OpenRouter slug and scores the first task's O: it needs "
                            "an OpenRouter judge and O among the run's variants")
    return problems


def _check_tie_rule(cfg: dict) -> list[str]:
    block = cfg.get("tie_rule")
    threshold = block.get("threshold") if isinstance(block, dict) else None
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        return ["tie_rule.threshold must be a number from 0 to 1"]
    return []


def _check_rm(cfg: dict) -> list[str]:
    block = cfg.get("rm")
    if not isinstance(block, dict):
        return ["rm: the block is missing"]
    problems = []
    if not isinstance(block.get("models"), str) or not block["models"]:
        problems.append("rm.models must name an entry of configs/models.yaml")
    if block.get("sets") != [SET_KEY]:
        problems.append(f"rm.sets must be [{SET_KEY!r}] (the frozen sets of freeze.sets_out)")
    conditions = block.get("conditions")
    if (not isinstance(conditions, list) or not conditions or any(c not in rm_mod.CONDITIONS for c in conditions)
            or len(set(conditions)) != len(conditions)):
        problems.append(f"rm.conditions must be a non-empty list of distinct conditions from {list(rm_mod.CONDITIONS)}")
    return problems + _variant_problems("rm.variants", block.get("variants"))


def _raise(problems: list[str]) -> None:
    if problems:
        raise ValueError("invalid full_run config (contract 6D blocks):\n" + "\n".join(problems))


def validate_config(cfg: dict) -> None:
    """The checks of the three blocks (ValueError listing every failed one), for full_run.load_config to call: the
    pointwise_hidden block (set full_run, condition hidden, valid variants, unique run ids that also leave room for
    the -r1 ids, known keys, judges and rubrics, smoke_tasks a positive integer, probe_first only on an OpenRouter
    judge with O among its variants, and every Gemini run with route vertex in its run entry: lead's rule 3); the
    tie_rule threshold a number from 0 to 1; the rm block (models a models.yaml entry name, sets [full_run], known
    conditions and variants). Each stage runs the checks of the blocks it reads before anything else."""
    _raise(_check_pointwise_hidden(cfg) + _check_tie_rule(cfg) + _check_rm(cfg))


def frozen_inputs(cfg: dict) -> tuple[list[dict], list[dict]]:
    """(tasks, sets) of the freeze in the frozen order (the order of the sets file freeze.sets_out). Each task dict is
    read from the task file that the frozen task file (freeze.tasks_out) names for it, found among freeze.task_files
    by file name (the first record of that id in the file). ValueError when a frozen file is missing, when the sets
    file's sha256 differs from the frozen task file's sets_file_sha256 (the sets changed after the freeze), when the
    task file's task order differs from the sets file's, when it names a task file the config does not list, or when
    a task is not in its file."""
    fz = cfg["freeze"]
    sets_path, doc_path = _repo_path(fz["sets_out"]), _repo_path(fz["tasks_out"])
    missing = [str(p) for p in (sets_path, doc_path) if not p.exists()]
    if missing:
        raise ValueError(f"the frozen files are missing: {missing}; run the freeze stage first")
    doc = _read_json(doc_path, {})
    if full_run.file_sha256(sets_path) != doc.get("sets_file_sha256"):
        raise ValueError(f"{fz['sets_out']} does not match the sha256 recorded in {fz['tasks_out']}: the frozen sets "
                         "changed after the freeze")
    sets = _read_jsonl(sets_path)
    order = [s["task_id"] for s in sets]
    if [t["task_id"] for t in doc.get("tasks", [])] != order:
        raise ValueError(f"the task order of {fz['tasks_out']} differs from the order of {fz['sets_out']}")
    files = {Path(p).name: _repo_path(p) for p in fz["task_files"]}
    file_of = {t["task_id"]: t["task_file"] for t in doc["tasks"]}
    unknown = sorted(set(file_of.values()) - set(files))
    if unknown:
        raise ValueError(f"{fz['tasks_out']} names task files that freeze.task_files does not list: {unknown}")
    found: dict[str, dict] = {}
    for name in sorted(set(file_of.values())):
        for task in _read_jsonl(files[name]):
            if file_of.get(task["task_id"]) == name and task["task_id"] not in found:
                found[task["task_id"]] = task
    lost = [i for i in order if i not in found]
    if lost:
        raise ValueError(f"{len(lost)} frozen tasks are not in the task file {fz['tasks_out']} names for them: {lost}")
    return [found[i] for i in order], sets


def _inputs_or_fail(ctx: full_run.Ctx6, stage: str) -> tuple[list[dict], list[dict]] | None:
    """frozen_inputs, or None after logging the reason and (not in a dry run) recording the stage as failed."""
    try:
        return frozen_inputs(ctx.cfg)
    except ValueError as exc:
        logger.error("%s: FAILED: %s", stage, exc)
        if not ctx.dry_run:
            _record_stage(ctx, stage, {"status": "failed", "reason": f"ValueError: {exc}"})
        return None


# ---------------------------------------------------------------------------------------------
# The hidden runs: resolution, gate and the run loop shared with the tie rule's second samples
# ---------------------------------------------------------------------------------------------


def hidden_runs(cfg: dict) -> list[dict]:
    """The runs of the pointwise_hidden block in config order, resolved: id, judge, spec (pilot2.run_spec: the
    models.yaml entry with the run's route), provider (the cost-log provider name), rubric (default
    pointwise-neutral) with its question and instruction (pointwise.question_for), variants (tuple; default the
    block's), replicate 0, probe_first (bool), smoke_tasks (int or None) and first_pass None."""
    pw = cfg["pointwise_hidden"]
    out = []
    for run in pw["runs"]:
        spec = pilot2.run_spec(cfg, run)
        rubric = run.get("rubric", RUBRIC)
        question, instruction = question_for(rubric)
        out.append({"id": run["id"], "judge": run["judge"], "spec": spec,
                    "provider": pilot2.COST_PROVIDER[spec["provider"]], "rubric": rubric, "question": question,
                    "instruction": instruction, "variants": tuple(run.get("variants", pw["variants"])),
                    "replicate": 0, "probe_first": bool(run.get("probe_first", False)),
                    "smoke_tasks": run.get("smoke_tasks"), "first_pass": None})
    return out


def second_sample_runs(cfg: dict, qualifying: list[str]) -> list[dict]:
    """The tie rule's second samples: for every hidden run (hidden_runs) whose judge is in `qualifying`, a copy with
    id `<id>-r1`, replicate 1 and first_pass = the run's id, in config order. probe_first and smoke_tasks are dropped
    from the copies: the first pass already checked the slug and the parse, and a -r1 run launches only when its
    first pass is done (a dropped or failed first pass has no second sample)."""
    return [{**run, "id": run["id"] + SECOND_SAMPLE_SUFFIX, "replicate": 1, "probe_first": False, "smoke_tasks": None,
             "first_pass": run["id"]} for run in hidden_runs(cfg) if run["judge"] in qualifying]


def launch_gate(ctx: full_run.Ctx6, run: dict, provider: str) -> str | None:
    """Reason not to launch a run now, or None. Anthropic: pilot2.gate (the Anthropic total of the run's cost file at
    or above budget.anthropic_stop_launching_usd). OpenRouter: the OpenRouter total of the same file at or above
    budget.openrouter_stop_launching_usd (lead's rule 2). Google: no stop line (the CostLogger's google_usd_cap stops
    a call that would exceed it)."""
    if provider == "openrouter":
        spent = ctx.cost.spent("openrouter")
        stop = float(ctx.cfg["budget"]["openrouter_stop_launching_usd"])
        if spent >= stop:
            return f"OpenRouter total ${spent:.2f} has reached the stop-launching threshold ${stop:.2f}"
        return None
    return pilot2.gate(ctx, run, provider)


def _run_records(records: list[dict], run: dict, task_ids: set[str]) -> dict[tuple[str, str], dict]:
    """{(task_id, variant): record} of a run's records in a pointwise file: grader, set, condition, replicate and
    rubric as the run's, task among task_ids, variant among the run's variants (the first record of a key wins)."""
    out: dict[tuple[str, str], dict] = {}
    for r in records:
        if (r.get("grader") == run["judge"] and r.get("set") == SET_NAME and r.get("condition") == CONDITION
                and int(r.get("replicate") or 0) == run["replicate"] and r.get("rubric") == run["rubric"]
                and r.get("task_id") in task_ids and r.get("variant") in run["variants"]):
            out.setdefault((r["task_id"], r["variant"]), r)
    return out


def _plan_line(stage: str, run: dict, n_tasks: int, present: int, max_workers: int) -> str:
    spec = run["spec"]
    route = f", route {spec.get('route')}" if spec["provider"] == "gemini" else ""
    line = (f"{stage}: run {run['id']}: {run['judge']} ({spec['provider']}{route}), rubric {run['rubric']}, variants "
            f"{list(run['variants'])}, replicate {run['replicate']}, {n_tasks} tasks, "
            f"{n_tasks * len(run['variants'])} records expected ({present} present), {max_workers} workers")
    if run["probe_first"]:
        line += ("; probe first: the slug on OpenRouter's model list, then the first task's O alone (its first "
                 "record); no valid score after two attempts -> failed with the raw reply, not iterated")
    if run["smoke_tasks"]:
        n = int(run["smoke_tasks"])
        line += (f"; smoke test: the first {n} tasks ({n * len(run['variants'])} calls), any invalid score -> "
                 f"{DROPPED} (records kept, not iterated), else the rest")
    if run["first_pass"]:
        line += f"; launches only when its first pass {run['first_pass']} is done"
    return line + f"; credential {'set' if pilot2.credential_for(spec) else 'MISSING -> skipped'}"


def _finish(ctx: full_run.Ctx6, status: dict, run_id: str, entry: dict) -> None:
    """Store a run's entry in runs_status.json, log it and print its status with the running cost totals."""
    status[run_id] = entry
    _write_json(ctx.run_dir / RUNS_STATUS_FILE, status)
    why = f" ({entry['reason']})" if entry.get("reason") else ""
    logger.info("%s: run %s %s%s; %s records", entry["stage"], run_id, entry["status"], why, entry.get("records"))
    print(f"[{run_id}] {entry['status']}{why}; {ctx.cost.totals_line()}", flush=True)


def _run_hidden(ctx: full_run.Ctx6, runs: list[dict], stage: str, tasks: list[dict], sets: list[dict]
                ) -> tuple[int, dict[str, str]]:
    """Execute resolved hidden runs in order on every frozen task; returns (exit code, {run id: status}).

    Per run, as pilot4._stage_pointwise: a run whose records are all present is not gated; otherwise its first pass
    not done (a -r1 run), no credential, a provider blocked earlier in this invocation (Gemini balance exhausted) or
    launch_gate skips it, recorded with the reason (a reason recorded at an earlier skip is kept, the new one goes to
    reason_on_last_invocation). A run recorded as dropped_parse_failure or as a failed probe is not relaunched (the
    row is dropped, not iterated). Then, with the run's rubric, question, instruction, variants and replicate, no
    prompt cache, the set name SET_NAME and the provider's max_workers: probe_first (the slug check when records are
    missing, then e1b_pointwise.score_task on the first task's O alone, its first record; a slug that fails the check
    -> failed; no valid score -> failed with probe_failed and the raw reply); smoke_tasks n (run_pointwise on the
    first n tasks; any invalid score -> dropped_parse_failure, the records stay); then run_pointwise on every task,
    which resumes what exists. BudgetExceeded stops the stage (pilot2._budget_stop, EXIT_BUDGET); a Gemini
    BalanceExhausted stops the run and blocks the provider for the rest of the invocation. A done entry has the
    record count, the invalid and live counts and providers_seen (the upstream providers of the run's records).
    Every entry goes to runs_status.json and is printed with ctx.cost.totals_line()."""
    from overclaim.experiments import e1b_pointwise
    from overclaim.graders import judge as J

    cfg = ctx.cfg
    out_path = ctx.run_dir / POINTWISE_FILE
    status: dict[str, dict] = _read_json(ctx.run_dir / RUNS_STATUS_FILE, {})
    ids = {t["task_id"] for t in tasks}
    blocked: dict[str, str] = {}
    outcome: dict[str, str] = {}
    for run in runs:
        spec, provider, label = run["spec"], run["provider"], run["judge"]
        expected = len(tasks) * len(run["variants"])
        have = len(_run_records(_read_jsonl(out_path), run, ids))
        entry: dict[str, Any] = {"stage": stage, "judge": label, "provider": spec["provider"],
                                 "route": spec.get("route"), "set": SET_NAME, "condition": CONDITION,
                                 "rubric": run["rubric"], "variants": list(run["variants"]),
                                 "replicate": run["replicate"], "n_tasks": len(tasks), "expected_records": expected}
        previous = status.get(run["id"], {})
        if previous.get("status") == DROPPED or previous.get("probe_failed"):
            logger.warning("%s: run %s is recorded as %s; not relaunched", stage, run["id"], previous["status"])
            outcome[run["id"]] = previous["status"]
            print(f"[{run['id']}] {previous['status']} earlier; not relaunched; {ctx.cost.totals_line()}", flush=True)
            continue
        reason = None
        if have < expected:
            first = status.get(run["first_pass"], {}).get("status") if run["first_pass"] else "done"
            if first != "done":
                reason = f"its first pass {run['first_pass']} is {first or 'not run'}, not done"
            elif not pilot2.credential_for(spec):
                reason = f"no credential for provider {spec['provider']}" + (
                    f" route {spec.get('route')}" if spec["provider"] == "gemini" else "")
            elif provider in blocked:
                reason = blocked[provider]
            else:
                reason = launch_gate(ctx, run, provider)
        if reason is not None:
            entry.update(status="skipped", reason=reason, records=have)
            if previous.get("status") == "skipped" and previous.get("reason") != reason:
                entry.update(reason=previous.get("reason"), reason_on_last_invocation=reason)
            logger.warning("%s: run %s SKIPPED: %s", stage, run["id"], reason)
            outcome[run["id"]] = "skipped"
            _finish(ctx, status, run["id"], entry)
            continue
        route = f", route {spec.get('route')}" if spec["provider"] == "gemini" else ""
        logger.info("%s: run %s: %s (%s%s), rubric %s, variants %s, replicate %d, %d tasks, %d of %d records present",
                    stage, run["id"], label, spec["provider"], route, run["rubric"], list(run["variants"]),
                    run["replicate"], len(tasks), have, expected)
        client = ctx.client(spec)
        kwargs: dict[str, Any] = {"set_name": SET_NAME, "structured": bool(spec.get("structured_output")),
                                  "replicate": run["replicate"], "variants": run["variants"], "rubric": run["rubric"],
                                  "question": run["question"], "instruction": run["instruction"]}
        workers = int(cfg.get("max_workers", {}).get(provider, 1))
        try:
            if run["probe_first"]:
                if have < expected:
                    try:
                        pilot2.verify_openrouter_slugs([spec["model"]])
                    except Exception as exc:   # an unlisted or :free slug, or the model list could not be read
                        entry.update(status="failed", reason=f"slug check: {type(exc).__name__}: {exc}", records=have)
                        outcome[run["id"]] = "failed"
                        _finish(ctx, status, run["id"], entry)
                        continue
                probe = e1b_pointwise.score_task(tasks[0], sets[0], client, label, CONDITION, str(out_path),
                                                 cfg["run_name"], use_prompt_cache=False,
                                                 **{**kwargs, "variants": ("O",)})[0]
                if probe["score"] is None:
                    entry.update(status="failed", probe_failed=True, raw_text=probe["raw_text"],
                                 records=len(_run_records(_read_jsonl(out_path), run, ids)),
                                 providers_seen=sorted({probe["upstream_provider"]} - {None}),
                                 reason=f"probe: the O of {tasks[0]['task_id']} got no valid score after "
                                        f"{probe['parse_attempts']} attempts; the judge is reported, the prompt is "
                                        "not iterated (addendum section 3)")
                    outcome[run["id"]] = "failed"
                    _finish(ctx, status, run["id"], entry)
                    continue
            if run["smoke_tasks"]:
                n = int(run["smoke_tasks"])
                smoke = e1b_pointwise.run_pointwise(tasks[:n], sets[:n], client, label, CONDITION, str(out_path),
                                                    cfg["run_name"], max_workers=workers, use_prompt_cache=False,
                                                    **kwargs)
                bad = [f"{r['task_id']}/{r['variant']}" for r in smoke if r["score"] is None]
                if bad:
                    entry.update(status=DROPPED, records=len(smoke), n_invalid=len(bad), invalid=bad,
                                 reason=f"smoke test: {len(bad)} of {len(smoke)} scores of the first {n} tasks are "
                                        "invalid; the row is dropped, not iterated (decision (5)), and its records "
                                        "stay")
                    outcome[run["id"]] = DROPPED
                    _finish(ctx, status, run["id"], entry)
                    continue
            records = e1b_pointwise.run_pointwise(tasks, sets, client, label, CONDITION, str(out_path),
                                                  cfg["run_name"], max_workers=workers, use_prompt_cache=False,
                                                  **kwargs)
        except J.BudgetExceeded as exc:
            entry.update(status="stopped", reason=f"budget cap: {exc}")
            outcome[run["id"]] = "stopped"
            _finish(ctx, status, run["id"], entry)
            return pilot2._budget_stop(ctx, stage, exc), outcome
        except Exception as exc:
            if type(exc).__name__ != "BalanceExhausted":
                raise
            blocked[provider] = f"Gemini balance exhausted (HTTP 402) during run {run['id']}"
            entry.update(status="stopped", reason=str(exc))
            logger.error("%s: run %s STOPPED: %s", stage, run["id"], exc)
            outcome[run["id"]] = "stopped"
            _finish(ctx, status, run["id"], entry)
            continue
        entry.update(status="done", records=len(records), n_invalid=sum(r["score"] is None for r in records),
                     n_live=sum(not r["cached"] for r in records),
                     providers_seen=sorted({r.get("upstream_provider") for r in records} - {None}))
        outcome[run["id"]] = "done"
        _finish(ctx, status, run["id"], entry)
    return EXIT_OK, outcome


# ---------------------------------------------------------------------------------------------
# Stage: pointwise_hidden
# ---------------------------------------------------------------------------------------------


def stage_pointwise_hidden(ctx: full_run.Ctx6) -> int:
    """Brief Step 9, hidden (Tier 1), with E6: the runs of the pointwise_hidden block in config order (cheapest first,
    Opus rows last), each on every frozen task in the frozen order, through _run_hidden ->
    run_dir/pointwise_calls.jsonl and runs_status.json. Each run scores its `variants` (default the block's: O, H, P,
    O_plus, O_minus; one call per message, no other variant in the prompt; the lead edits a Claude run's variants to
    match the printed plan when cut C1 removes its O-), with its rubric (default pointwise-neutral; the full-rubric
    run has smoke_tasks: 3) and structured output as the judge spec says. The record's grader is the judge name, with
    no route suffix (the route is in each record). The block's checks run first (ValueError before any call or plan:
    among them a Gemini run without route vertex). Missing or changed frozen inputs fail the stage (EXIT_FAILED,
    recorded). Dry run: per run the judge, provider, route, rubric, variants, tasks, expected and present records,
    workers, the probe and smoke rules and whether the credential is set; nothing is written. Records the stage in
    stages_status.json: done with each run's status, or stopped at a budget cap."""
    _raise(_check_pointwise_hidden(ctx.cfg))
    runs = hidden_runs(ctx.cfg)
    inputs = _inputs_or_fail(ctx, "pointwise_hidden")
    if inputs is None:
        return EXIT_FAILED
    tasks, sets = inputs
    if ctx.dry_run:
        out_path = ctx.run_dir / POINTWISE_FILE
        existing = _read_jsonl(out_path)
        ids = {t["task_id"] for t in tasks}
        ctx.note(f"pointwise_hidden: {len(tasks)} frozen tasks ({ctx.cfg['freeze']['sets_out']}, sha256 as recorded "
                 f"in {ctx.cfg['freeze']['tasks_out']}), set name {SET_NAME}, condition hidden, no prompt cache, "
                 f"records -> {out_path}; run order: {[r['id'] for r in runs]}; spend so far: "
                 f"{ctx.cost.totals_line()}")
        for run in runs:
            ctx.note(_plan_line("pointwise_hidden", run, len(tasks), len(_run_records(existing, run, ids)),
                                int(ctx.cfg.get("max_workers", {}).get(run["provider"], 1))))
        return EXIT_OK
    code, outcome = _run_hidden(ctx, runs, "pointwise_hidden", tasks, sets)
    _record_stage(ctx, "pointwise_hidden", {"status": "done" if code == EXIT_OK else "stopped", "runs": outcome})
    return code


# ---------------------------------------------------------------------------------------------
# Stage: tie_rule
# ---------------------------------------------------------------------------------------------


def tie_table(records: list[dict], task_ids: list[str], threshold: float, graders: set[str] | None = None) -> dict:
    """The tie rule's table (decision (4); the rate of projection_v4.tie_rates) from pointwise records: set
    SET_NAME, hidden, replicate 0, rubric pointwise-neutral, variants O and H, tasks among task_ids.

    A grader is decided once every task of task_ids has its O and its H record (valid or not), i.e. after its first
    hidden pass (amendment 3 item 6); the contract's rate has no completeness condition, but a rate on a partial pass
    would be frozen by tie_refusal at the first invocation. Its row: tasks, valid (tasks whose O and H scores are both
    integers), excluded (tasks - valid), ties (valid tasks with O == H), rate (ties / valid; None when no task is
    valid) and second_sample (rate > threshold). A grader with fewer is listed under not_decided with its count, and
    so is every name of `graders` (the judges the stage expects) that has no record at all, with 0. Returns {"set",
    "condition", "replicate", "rubric", "variants", "threshold", "n_tasks", "rows" (decided graders, sorted),
    "qualifying" (graders with second_sample), "not_decided" (sorted)}.
    """
    wanted = set(task_ids)
    scores: dict[str, dict[str, dict[str, Any]]] = {g: {} for g in graders or ()}
    for r in records:
        if (r.get("set") == SET_NAME and r.get("condition") == CONDITION and int(r.get("replicate") or 0) == 0
                and r.get("rubric") == RUBRIC and r.get("variant") in ("O", "H") and r.get("task_id") in wanted):
            scores.setdefault(r["grader"], {}).setdefault(r["task_id"], {}).setdefault(r["variant"], r.get("score"))
    rows, not_decided = [], []
    for grader in sorted(scores):
        by_task = {t: v for t, v in scores[grader].items() if "O" in v and "H" in v}
        if len(by_task) < len(wanted):
            not_decided.append({"grader": grader, "tasks_with_o_and_h": len(by_task), "tasks": len(wanted)})
            continue
        valid = [v for v in by_task.values() if isinstance(v["O"], int) and isinstance(v["H"], int)]
        ties = sum(v["O"] == v["H"] for v in valid)
        rate = ties / len(valid) if valid else None
        rows.append({"grader": grader, "tasks": len(by_task), "valid": len(valid),
                     "excluded": len(by_task) - len(valid), "ties": ties, "rate": rate,
                     "second_sample": rate is not None and rate > threshold})
    return {"set": SET_NAME, "condition": CONDITION, "replicate": 0, "rubric": RUBRIC, "variants": ["O", "H"],
            "threshold": threshold, "n_tasks": len(wanted), "rows": rows,
            "qualifying": [r["grader"] for r in rows if r["second_sample"]], "not_decided": not_decided}


def tie_lines(table: dict) -> list[str]:
    """The table as text lines: the rule, one line per decided grader, one per undecided grader, the qualifying list."""
    lines = [f"O/H tie rate: set {table['set']}, hidden, replicate 0, rubric {table['rubric']}, {table['n_tasks']} "
             f"frozen tasks; a second sample when the rate is above {table['threshold']}"]
    for r in table["rows"]:
        rate = "n/a" if r["rate"] is None else f"{r['rate']:.3f}"
        lines.append(f"{r['grader']}: {r['ties']} ties of {r['valid']} tasks with valid O and H scores "
                     f"({r['excluded']} excluded) = {rate} -> second sample {'yes' if r['second_sample'] else 'no'}")
    for r in table["not_decided"]:
        lines.append(f"{r['grader']}: not decided ({r['tasks_with_o_and_h']} of {r['tasks']} tasks have O and H "
                     "records)")
    lines.append(f"qualifying: {table['qualifying'] or 'none'}")
    return lines


def tie_refusal(old: dict | None, new: dict) -> str | None:
    """None when there is no table on disk (old None) or when every row on disk has the same computed fields in `new`
    (every key of the new row: grader, tasks, valid, excluded, ties, rate, second_sample) and the rule's fields (set,
    condition, replicate, rubric, variants, threshold, n_tasks) are the same; else the first difference
    (full_run.first_difference). A decided row is never changed or removed (the qualifying list is fixed before the
    shown runs, decision (4)); rows of graders decided since are added. Keys a row on disk has beyond the computed
    ones (an annotation such as second_sample_conditions, contract 6H addition 16) are not compared: stage_tie_rule
    keeps them when it rewrites the file."""
    if old is None:
        return None
    rule = ("set", "condition", "replicate", "rubric", "variants", "threshold", "n_tasks")
    diff = full_run.first_difference({k: old.get(k) for k in rule}, {k: new[k] for k in rule})
    if diff:
        return diff
    now = {r["grader"]: r for r in new["rows"]}
    for row in old.get("rows", []):
        if row["grader"] not in now:
            return f"rows: {row['grader']} is decided on disk and not decided now"
        computed = now[row["grader"]]
        diff = full_run.first_difference({k: row[k] for k in computed if k in row}, computed,
                                         f"rows[{row['grader']}]")
        if diff:
            return diff
    return None


def keep_annotations(old: dict | None, new: dict) -> dict:
    """`new` with the keys of the table on disk that `new` does not compute carried over: per row (by grader) the
    row's extra keys, e.g. second_sample_conditions (contract 6H addition 16; amendment 3, "Tie rule, budget": the
    choice is recorded in the file), and the table's extra top-level keys. Call after tie_refusal has passed, so the
    computed values agree."""
    if not old:
        return new
    old_rows = {r["grader"]: r for r in old.get("rows", [])}
    rows = [{**r, **{k: v for k, v in old_rows.get(r["grader"], {}).items() if k not in r}} for r in new["rows"]]
    return {**new, "rows": rows, **{k: v for k, v in old.items() if k not in new}}


def stage_tie_rule(ctx: full_run.Ctx6) -> int:
    """Decision (4) of the addendum of 2026-11-07 (amendment 2 item 2): tie_table over run_dir/pointwise_calls.jsonl
    and the frozen task ids with tie_rule.threshold, every judge of a pointwise-neutral run of the pointwise_hidden
    block listed (not_decided with 0 when it has no record yet) -> run_dir/tie_rule.json, printed, with the
    annotations of the file on disk kept (keep_annotations); then second_sample_runs of the qualifying graders
    (replicate 1 of every hidden run of each, all its variants and rubrics, ids <id>-r1) through _run_hidden into the
    same pointwise file, behind the same gate and bookkeeping. With no qualifying grader the file is written and the
    stage says so. A grader decided later (its hidden pass completed after a skip or a stop) is added by the next
    invocation, which also launches its -r1 runs. The checks of the pointwise_hidden and tie_rule blocks run first
    (ValueError).

    Fails (EXIT_FAILED, recorded, nothing written) when the frozen inputs are missing or changed, when no grader is
    decided yet (run pointwise_hidden first: an empty table would let the shown stage go ahead without second
    samples), or when a tie_rule.json on disk disagrees with the table now (tie_refusal: a decided row changed or
    vanished, or a field of the rule changed; an annotation key on a row is not a change; a previous done entry
    stays, with "refused"). Contract 6E reads the file for the shown runs. Dry run: the table (with not_decided),
    whether the file would be written or refused, and the -r1 runs that would launch; nothing is written."""
    cfg = ctx.cfg
    _raise(_check_pointwise_hidden(cfg) + _check_tie_rule(cfg))
    inputs = _inputs_or_fail(ctx, "tie_rule")
    if inputs is None:
        return EXIT_FAILED
    tasks, sets = inputs
    expected = {r["judge"] for r in hidden_runs(cfg) if r["rubric"] == RUBRIC}
    table = tie_table(_read_jsonl(ctx.run_dir / POINTWISE_FILE), [s["task_id"] for s in sets],
                      float(cfg["tie_rule"]["threshold"]), expected)
    path = ctx.run_dir / TIE_RULE_FILE
    on_disk = _read_json(path, None)
    refusal = tie_refusal(on_disk, table)
    r1 = second_sample_runs(cfg, table["qualifying"])
    if ctx.dry_run:
        for line in tie_lines(table):
            ctx.note("tie_rule: " + line)
        if not table["rows"]:
            ctx.note("tie_rule: no grader is decided yet: a live run fails until pointwise_hidden has written every "
                     "hidden O and H record of a grader")
        else:
            ctx.note(f"tie_rule: would write {path}" if refusal is None
                     else f"tie_rule: would REFUSE: {path} differs: {refusal}")
        existing = _read_jsonl(ctx.run_dir / POINTWISE_FILE)
        ids = {t["task_id"] for t in tasks}
        for run in r1:
            ctx.note(_plan_line("tie_rule", run, len(tasks), len(_run_records(existing, run, ids)),
                                int(cfg.get("max_workers", {}).get(run["provider"], 1))))
        if not r1:
            ctx.note("tie_rule: no second-sample run would launch")
        return EXIT_OK
    if not table["rows"]:
        reason = (f"no grader has every hidden O and H record (replicate 0, rubric {RUBRIC}) over the "
                  f"{len(tasks)} frozen tasks yet; run pointwise_hidden first (undecided: "
                  f"{[r['grader'] for r in table['not_decided']]})")
        logger.error("tie_rule: FAILED: %s", reason)
        _record_stage(ctx, "tie_rule", {"status": "failed", "reason": reason})
        return EXIT_FAILED
    if refusal is not None:
        reason = f"{path.name} on disk differs from the table computed now ({refusal}); nothing is written"
        logger.error("tie_rule: REFUSED: %s", reason)
        previous = _read_json(ctx.run_dir / "stages_status.json", {}).get("tie_rule") or {}
        _record_stage(ctx, "tie_rule", {**previous, "refused": reason} if previous.get("status") == "done"
                      else {"status": "failed", "reason": reason})
        return EXIT_FAILED
    added = sorted({r["grader"] for r in table["rows"]} - {r["grader"] for r in (on_disk or {}).get("rows", [])})
    table = keep_annotations(on_disk, table)
    _write_json(path, table)
    logger.info("tie_rule: %s written; graders decided now: %s; newly decided: %s", path, len(table["rows"]), added)
    for line in tie_lines(table):
        print(f"[tie_rule] {line}", flush=True)
    if not table["qualifying"]:
        print(f"[tie_rule] no grader's O/H tie rate is above {table['threshold']}: no second sample is launched; "
              f"{path} written", flush=True)
    code, outcome = _run_hidden(ctx, r1, "tie_rule", tasks, sets)
    _record_stage(ctx, "tie_rule", {"status": "done" if code == EXIT_OK else "stopped",
                                    "qualifying": table["qualifying"], "decided": len(table["rows"]),
                                    "not_decided": [r["grader"] for r in table["not_decided"]], "runs": outcome})
    return code


# ---------------------------------------------------------------------------------------------
# Stage: rm (E5)
# ---------------------------------------------------------------------------------------------


def reward_model_specs(cfg: dict) -> list[dict]:
    """The reward-model entries of configs/models.yaml (cfg["models_config"]) under the name cfg["rm"]["models"]
    (reward_models_part2), in file order. ValueError when the entry is missing or not a non-empty list of dicts with a
    name and a model, when names repeat, or when a quantization is not one rm.RewardModel knows."""
    with open(_repo_path(cfg["models_config"]), encoding="utf-8") as f:
        models = yaml.safe_load(f)
    key = cfg["rm"]["models"]
    specs = models.get(key)
    if not isinstance(specs, list) or not specs or not all(isinstance(s, dict) and s.get("name") and s.get("model")
                                                           for s in specs):
        raise ValueError(f"{cfg['models_config']}: {key!r} must be a non-empty list of entries with a name and a model")
    names = [s["name"] for s in specs]
    bad = [s["name"] for s in specs if s.get("quantization") not in (None, *rm_mod.QUANTIZATIONS)]
    if len(set(names)) != len(names) or bad:
        raise ValueError(f"{cfg['models_config']}: {key!r} needs distinct names and known quantizations "
                         f"(None or {list(rm_mod.QUANTIZATIONS)}); unknown quantization in {bad}")
    return specs


def _rm_have(records: list[dict], name: str, conditions: list[str], task_ids: set[str],
             variants: tuple[str, ...]) -> int:
    """Distinct (condition, task, variant) records of one reward model on SET_NAME."""
    return len({(r["condition"], r["task_id"], r["variant"]) for r in records
                if r.get("grader") == name and r.get("set") == SET_NAME and r.get("condition") in conditions
                and r.get("task_id") in task_ids and r.get("variant") in variants})


def _free_gpu() -> None:
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stage_rm(ctx: full_run.Ctx6) -> int:
    """E5 (a): reward_model_specs in config order (the 3B first, the headline; the 8B in 4-bit last), each as
    pilot2.stage_rm on the frozen sets: rm.RewardModel(model, dtype, batch_size, the disk cache, name, quantization),
    check_fits (a model that does not fit, or whose 4-bit packages are missing, is skipped with that reason), the
    weights' size logged before loading (in a live run the Hub download is the load; it is not a paid call), load
    (an ImportError is a skip reason), the sanity check as today on the first frozen task (score of its own final
    message above the empty reply, hidden context; a failure records the model as failed, scores nothing for it and
    the stage returns EXIT_FAILED after the other models), then rm.run_scores per condition of rm.conditions with the
    hidden context render.hidden_context(repo, issue_text) (run_scores' default), set name SET_NAME and rm.variants ->
    run_dir/rm_scores.jsonl (resume key (grader, condition, set, task_id, variant)). Per model run_dir/rm_status.json
    gets the status, the fit line, weights_bytes, load_seconds, score_seconds (wall clock on the GPU), peak_memory_gib
    (torch.cuda.max_memory_allocated since before the load) and the record counts, written after each model. A model
    with every record present and a done entry is not loaded again. Local GPU, no API money: not gated. The rm block's
    checks run first (ValueError). Dry run: the models, their records present and, for nf4, whether its packages are
    installed; no model is loaded, nothing is downloaded and check_fits is not called (it reads the Hub)."""
    cfg = ctx.cfg
    _raise(_check_rm(cfg))
    specs = reward_model_specs(cfg)
    inputs = _inputs_or_fail(ctx, "rm")
    if inputs is None:
        return EXIT_FAILED
    tasks, sets = inputs
    block = cfg["rm"]
    conditions, variants = list(block["conditions"]), tuple(block["variants"])
    ids = {t["task_id"] for t in tasks}
    expected = len(tasks) * len(conditions) * len(variants)
    out_path = ctx.run_dir / RM_SCORES_FILE
    first = tasks[0]
    if ctx.dry_run:
        existing = _read_jsonl(out_path)
        ctx.note(f"rm: {len(specs)} reward models ({cfg['models_config']}: {block['models']}), in this order, each on "
                 f"{len(tasks)} frozen tasks x conditions {conditions} x variants {list(variants)} = {expected} "
                 f"records -> {out_path} (set {SET_NAME}); context render.hidden_context(repo, issue_text) and the "
                 f"candidate message; sanity check on {first['task_id']}; GPU time and peak memory -> "
                 f"{ctx.run_dir / RM_STATUS_FILE}. A live run decides each fit with check_fits (CUDA free memory "
                 "against the weights' size from the Hub) and skips a model that does not fit, with the reason")
        for spec in specs:
            q = spec.get("quantization")
            note = ""
            if q == "nf4":
                missing = [p for p in rm_mod.NF4_PACKAGES if importlib.util.find_spec(p) is None]
                note = (f"; packages {missing} not installed -> skipped with that reason" if missing
                        else f"; packages {list(rm_mod.NF4_PACKAGES)} installed")
            ctx.note(f"rm: {spec['name']}: {spec['model']}, {spec.get('dtype', 'bfloat16')}"
                     f"{', ' + q if q else ''}, batch {spec.get('batch_size', 1)}; "
                     f"{_rm_have(existing, spec['name'], conditions, ids, variants)} of {expected} records present"
                     f"{note}")
        return EXIT_OK
    from overclaim import render

    status_path = ctx.run_dir / RM_STATUS_FILE
    status: dict[str, dict] = _read_json(status_path, {})
    failed: list[str] = []
    for spec in specs:
        name = spec["name"]
        have = _rm_have(_read_jsonl(out_path), name, conditions, ids, variants)
        if have >= expected and status.get(name, {}).get("status") == "done":
            logger.info("rm: %s: all %d records present; not loaded again", name, expected)
            continue
        entry: dict[str, Any] = {"model": spec["model"], "dtype": spec.get("dtype", "bfloat16"),
                                 "quantization": spec.get("quantization"), "set": SET_NAME, "conditions": conditions,
                                 "variants": list(variants), "expected_records": expected}
        model = rm_mod.RewardModel(spec["model"], dtype=spec.get("dtype", "bfloat16"),
                                   batch_size=spec.get("batch_size", 1), cache=ctx.cache, name=name,
                                   quantization=spec.get("quantization"))
        fits, why = model.check_fits()
        entry["fit"] = why
        logger.info("rm: %s: %s", name, why)
        if not fits:
            status[name] = {**entry, "status": "skipped", "reason": why, "records": have}
            _write_json(status_path, status)
            print(f"[rm] {name}: skipped: {why}", flush=True)
            continue
        entry["weights_bytes"] = round(model.weights_gib * rm_mod.GIB)
        disk_fits, disk_why = model.check_disk_fits()
        entry["disk"] = disk_why
        logger.info("rm: %s: %s", name, disk_why)
        if not disk_fits:                      # 2026-11-08: the 8B's download filled the disk; never start one that
            status[name] = {**entry, "status": "skipped", "reason": disk_why, "records": have}   # does not fit
            _write_json(status_path, status)
            print(f"[rm] {name}: skipped: {disk_why}", flush=True)
            del model
            continue
        logger.info("rm: %s: loading %s, %d bytes of safetensors weights (downloaded from the Hub unless cached)",
                    name, spec["model"], entry["weights_bytes"])
        print(f"[rm] {name}: loading {spec['model']} ({entry['weights_bytes']:,} bytes of weights)", flush=True)
        rm_mod.reset_peak_memory()
        start = time.monotonic()
        try:
            model.load()
        except ImportError as exc:
            status[name] = {**entry, "status": "skipped", "reason": f"ImportError while loading: {exc}",
                            "records": have}
            _write_json(status_path, status)
            print(f"[rm] {name}: skipped: ImportError while loading: {exc}", flush=True)
            del model
            _free_gpu()
            continue
        except Exception as exc:               # a download or loading error (2026-11-08: the Hub client's disk-full
            reason = f"load failed: {type(exc).__name__}: {exc}"   # RuntimeError crashed the stage before this)
            logger.error("rm: %s %s; not scoring it", name, reason)
            status[name] = {**entry, "status": "failed", "reason": reason, "records": have}
            _write_json(status_path, status)
            failed.append(name)
            del model
            _free_gpu()
            continue
        entry["load_seconds"] = round(time.monotonic() - start, 1)
        try:
            model.sanity_check(render.hidden_context(first["repo"], first["issue_text"]), first["final_message_raw"])
        except RuntimeError as exc:
            logger.error("rm: %s failed the sanity check; not scoring it: %s", name, exc)
            status[name] = {**entry, "status": "failed", "reason": f"sanity_check on {first['task_id']} failed: {exc}",
                            "records": have}
            _write_json(status_path, status)
            failed.append(name)
            del model
            _free_gpu()
            continue
        start = time.monotonic()
        records: list[dict] = []
        for condition in conditions:
            records += rm_mod.run_scores(tasks, sets, model, condition, str(out_path), cfg["run_name"],
                                         set_name=SET_NAME, variants=variants)
        status[name] = {**entry, "status": "done", "score_seconds": round(time.monotonic() - start, 1),
                        "peak_memory_gib": rm_mod.peak_memory_gib(), "records": len(records),
                        "n_cache_hits": sum(bool(r["cached"]) for r in records)}
        _write_json(status_path, status)
        print(f"[rm] {name}: {len(records)} records, load {entry['load_seconds']} s, scoring "
              f"{status[name]['score_seconds']} s, peak {status[name]['peak_memory_gib']} GiB", flush=True)
        del model
        _free_gpu()
    _record_stage(ctx, "rm", {"status": "failed" if failed else "done",
                              "models": {s["name"]: status.get(s["name"], {}).get("status") for s in specs},
                              **({"reason": f"load or sanity check failed: {failed}"} if failed else {})})
    return EXIT_FAILED if failed else EXIT_OK


# In run order, for full_run.STAGES (after "freeze") and full_run._STAGE_FUNCS.
STAGE_FUNCS = {"pointwise_hidden": stage_pointwise_hidden, "tie_rule": stage_tie_rule, "rm": stage_rm}

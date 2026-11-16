"""Session 5, Part 2, Step 11: Experiment 2 in full (stage e2 of overclaim.experiments.full_run; contract 6F).

Tasks: picks.e2_tasks of the frozen task file (data/processed/full_run_tasks.json; 40 tasks, pick order) and its
picks.e2_shown_eligible (26); each task dict from the task file the frozen task file names for it, each task's evidence
the frozen set's evidence_v4. The pool is the pilot's (prep5_e2): the Qwen3-Coder packet policy writes
n_per_instruction samples under instruction A and as many under B per task (prep5_e2.instruction_plan("B", n), so the
pilot's ten tasks get their samples, labels and Sonnet scores from the disk cache), and both session-4 labellers label
every sample (prep5_e2.run_pool -> e2_samples.jsonl, e2_labels.jsonl, e2_labels_full.jsonl). Three kinds of selector
score the samples, one record per (selector, task, sample) in e2_scores.jsonl:
- plain (e2.selectors): the hidden pointwise prompt (e2_selection.score_samples_pointwise), as the pilot, every task;
- mitigated (e2.mitigated_selectors; H4's second condition, addendum section 1): the verification-first scorer of the
  revised Step 4 (e3_verification_pointwise.make_vfp_scorer) on the shown context, sent as the prompt-cached prefix,
  at configs/models.yaml's e3_max_tokens, on the shown-eligible tasks only, a task's samples back to back; an optional
  one runs only when results/full_run/projection.json counts its row in the final plan;
- reward models (E5(b); the list of configs/models.yaml that e2.reward_model_selectors names): each model that
  RewardModel.check_fits passes scores every sample with the hidden context (render.hidden_context), a float; a model
  that fails to load or to score is left out and named, and the stage goes on; each model's load and scoring times
  (wall clock, summed over the runs that wrote its records) go to e2_rm_status.json.
Best of N (e2_selection.NS) per labeller and outcome (prep5_e2.selection_rows: clean_overclaim, overclaim,
claims_tests_passed; blocks over the interleaved instructions, ties averaged over n_tie_seeds seeds, task bootstrap) on
two task sets: the 40 tasks (plain and reward-model selectors) and the 26 shown-eligible tasks (every selector). A
reward model's floats enter best of N recoded to integers that keep the top set of every block (ordinal_scores):
e2_selection reads only integer scores from 1 to 10. Paired comparisons at N = 16: per task, the clean-overclaim share
of the selected sample under the mitigated selector minus under its judge's plain selector (tie-averaged; H4's second
condition for the reference judge), and each reward model minus the reference plain selector, with the task-bootstrap
CI. E4 (addendum section 2): the resolved value of the selected sample is the task's whatever N or selector picks, so
it is one number per task set, the resolved share of the tasks. -> e2_table.json and e2_table.md.

Gates: before each run that still has calls to make, the Anthropic stop-launching line (pilot2.gate) for an Anthropic
run and the OpenRouter stop line (budget.openrouter_stop_launching_usd against results/full_run/cost.jsonl) for an
OpenRouter run; a gated pool skips the stage, a gated selector is left out of the tables and named. The table names
each selector that did not run with a reason that does not change from run to run (GATE_REASONS, "does not fit", the
projection's verdict, the exception's type); the full text (dollar totals, free GPU memory, the exception's message) is
in stages_status.json and e2_rm_status.json. The exploratory parts (EXPLORATORY_NOTE: the reward-model selectors, the
OpenRouter selector and its mitigated selector, E4) are flagged as such in e2_table.json and e2_table.md. Idempotent:
every run resumes from its file and every call is disk-cached; an e2_table.json that exists and differs from the one
computed now is not overwritten (the stage fails naming the first difference). Agent messages are data: they are placed
into prompts as text and never executed.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import yaml

from overclaim import analysis, render
from overclaim.experiments import e2_selection, pilot2, prep5, prep5_e2
from overclaim.experiments.pilot2 import EXIT_FAILED, EXIT_OK, _read_json, _read_jsonl, _repo_path, _write_json
from overclaim.experiments.pilot3 import _record_stage, _skip

if TYPE_CHECKING:   # full_run registers this module's stage; importing it at run time here would be circular
    from overclaim.experiments.full_run import Ctx6

logger = logging.getLogger(__name__)

STAGE = "e2"
FILES = {"samples": "e2_samples.jsonl", "labels": "e2_labels.jsonl", "labels_full": "e2_labels_full.jsonl",
         "scores": "e2_scores.jsonl"}
TABLE_JSON, TABLE_MD, RM_STATUS = "e2_table.json", "e2_table.md", "e2_rm_status.json"
TASK_SETS = {"e2_tasks": "the Experiment 2 tasks", "e2_shown_eligible": "the shown-eligible Experiment 2 tasks"}
KINDS = ("plain", "mitigated", "reward_model")
COMPARISON_N = e2_selection.NS[-1]        # the paired comparisons read best of 16
COMPARISON_OUTCOME = "clean_overclaim"    # the pre-registered outcome
PRIMARY_LABELLER = "session3"             # the rule is read on the session-3 labeller (README, amendment 2, item 4)
PROMPT_CACHE_PROVIDERS = ("anthropic",)   # judges whose client sends cached_prefix as a prompt-cache block
# The config keys the stage reads (the lead's rule 1 for contracts D to H).
CONFIG_BLOCKS = {
    "e2": "this stage's block (it replaces contract 6A's e2: {n_tasks, seed}, which the freeze reads for its pick)",
    "freeze": "tasks_out (the picks and resolved), sets_out (evidence_v4) and task_files (the task dicts)",
    "budget": "anthropic_stop_launching_usd (pilot2.gate) and openrouter_stop_launching_usd",
    "max_workers": "threads per provider (anthropic, openrouter, google)",
    "models_config": "e2.policy, e2.labeller, e3_max_tokens and the list e2.reward_model_selectors names",
    "judges": "the selectors' specs (full_run.load_config loads them from models_config)",
    "domain": "the verification-first claim phrase",
    "seed": "the bootstrap seed",
}
E4_SENTENCE = ("Every sample of a task is a final message for the same trajectory and patch, so the resolved value of "
               "the selected sample is the task's whatever N or selector picks: the resolved rate of the selected "
               "sample at N = 1, 4 and 16 under every selector is one number, the resolved share of the tasks.")
# The reason the table gives for a selector held at a gate (the gate's own text, with the running total, goes to
# stages_status.json): the same from run to run, so a rerun with no new data computes the same table.
GATE_REASONS = {"anthropic": "the Anthropic stop-launching line was reached (pilot2.gate)",
                "openrouter": "the OpenRouter stop line was reached (budget.openrouter_stop_launching_usd)"}
LOAD_ERRORS = (ImportError, OSError, RuntimeError)   # from RewardModel.load(): the model is left out, the stage goes on
SCORE_ERRORS = (RuntimeError,)                       # from its scoring (torch.cuda.OutOfMemoryError is a RuntimeError)
EXPLORATORY_NOTE = ("Exploratory, added on 2026-11-07 after the pilots: the reward-model selectors (E5(b)) and E4 "
                    "(README amendment 3, item 2), the OpenRouter selector and its mitigated selector (item 3); marked "
                    "exploratory in every table.")


# ---------------------------------------------------------------------------------------------
# Config: the selectors and the frozen inputs
# ---------------------------------------------------------------------------------------------


def models_config(cfg: dict) -> dict:
    """configs/models.yaml (cfg["models_config"]) as loaded: full_run.load_config keeps only its judges and prices, and
    this stage also needs the policy and labeller entries, e3_max_tokens and the reward-model list."""
    with open(_repo_path(cfg["models_config"]), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def e2_config(cfg: dict, models: dict) -> dict:
    """The stage's plan from cfg["e2"] and the models registry, after these checks (ValueError listing every failed
    one): n_tasks, n_per_instruction, n_tie_seeds and n_bootstrap are positive integers; ns equals e2_selection.NS and
    the pool (2 x n_per_instruction samples) is its largest N; policy and labeller name registry entries with a model;
    the registry's e3_max_tokens is a positive integer; selectors is a non-empty list of judges and reference_selector
    is one of them; each mitigated_selectors entry has a name and a judge that is one of the selectors (its plain
    counterpart), and an optional one a projection_row (then e2.projection names the projection file);
    reward_model_selectors is a string; the selector names are distinct (plain, mitigated and the reward models'); a
    Gemini judge whose spec does not route through vertex is refused (the lead's rule 3 for Part 2).

    Returns {"e2": the block, "policy", "labeller" (registry entries), "e3_max_tokens", "selectors": one entry per
    selector in run order (plain, mitigated, reward models): {name, kind (KINDS), provider, model, tasks (a TASK_SETS
    key)} plus, for a judge, judge, spec (a mitigated judge's spec has max_tokens e3_max_tokens) and structured, for a
    mitigated one optional and projection_row, for a reward model its registry entry; "reward_models_missing": the
    reason no reward model is planned when the registry has no list under that name, else None}."""
    e2 = cfg.get("e2")
    if not isinstance(e2, dict):
        raise ValueError("the config has no e2 block")
    judges = cfg.get("judges") or {}
    bad = []
    for key in ("n_tasks", "n_per_instruction", "n_tie_seeds", "n_bootstrap"):
        if not _positive_int(e2.get(key)):
            bad.append(f"{key} must be a positive integer")
    if tuple(e2.get("ns") or ()) != tuple(e2_selection.NS):
        bad.append(f"ns {e2.get('ns')} differs from the N values e2_selection computes {list(e2_selection.NS)}")
    if _positive_int(e2.get("n_per_instruction")) and 2 * e2["n_per_instruction"] != e2_selection.NS[-1]:
        bad.append(f"the pool of 2 x {e2['n_per_instruction']} samples must be the largest N, {e2_selection.NS[-1]}")
    for key in ("policy", "labeller"):
        if not isinstance(models.get(e2.get(key)), dict) or "model" not in models[e2[key]]:
            bad.append(f"{key} {e2.get(key)!r} is not an entry of {cfg.get('models_config')} with a model")
    e3_max_tokens = models.get("e3_max_tokens")
    if not _positive_int(e3_max_tokens):
        bad.append(f"e3_max_tokens of {cfg.get('models_config')} must be a positive integer")
    plain = e2.get("selectors")
    if not isinstance(plain, list) or not plain or any(s not in judges for s in plain):
        bad.append(f"selectors {plain} must be a non-empty list of judges")
        plain = []
    if e2.get("reference_selector") not in plain:
        bad.append(f"reference_selector {e2.get('reference_selector')!r} must be one of the selectors")
    selectors: list[dict] = []

    def judge_entry(name: str, kind: str, judge: str, spec: dict, tasks: str) -> dict:
        if spec["provider"] == "gemini" and spec.get("route") != "vertex":
            bad.append(f"selector {name}: a Gemini judge runs through vertex in Part 2, its spec routes "
                       f"{spec.get('route')!r}")
        return {"name": name, "kind": kind, "judge": judge, "spec": spec, "provider": spec["provider"],
                "model": spec["model"], "structured": bool(spec.get("structured_output")), "tasks": tasks}

    for judge in plain:
        selectors.append(judge_entry(judge, "plain", judge, dict(judges[judge]), "e2_tasks"))
    mitigated = e2.get("mitigated_selectors") or []
    if not isinstance(mitigated, list):
        bad.append("mitigated_selectors must be a list")
        mitigated = []
    for m in mitigated:
        if not isinstance(m, dict) or not m.get("name") or m.get("judge") not in plain:
            bad.append(f"mitigated selector {m}: needs a name and a judge that is one of the selectors")
            continue
        if m.get("optional") and not (m.get("projection_row") and e2.get("projection")):
            bad.append(f"mitigated selector {m['name']}: an optional selector needs a projection_row and e2.projection")
        spec = {**judges[m["judge"]], "max_tokens": e3_max_tokens}
        selectors.append({**judge_entry(m["name"], "mitigated", m["judge"], spec, "e2_shown_eligible"),
                          "optional": bool(m.get("optional")), "projection_row": m.get("projection_row")})
    rm_key = e2.get("reward_model_selectors")
    missing = None
    if not isinstance(rm_key, str):
        bad.append("reward_model_selectors must name a list of the models registry")
    elif not isinstance(models.get(rm_key), list):
        missing = f"{cfg.get('models_config')} has no list {rm_key!r}"
    else:
        for entry in models[rm_key]:
            if not isinstance(entry, dict) or not entry.get("name") or not entry.get("model"):
                bad.append(f"{rm_key}: entry {entry} needs a name and a model")
                continue
            selectors.append({"name": entry["name"], "kind": "reward_model", "provider": "local",
                              "model": entry["model"], "tasks": "e2_tasks", "entry": dict(entry)})
    names = [s["name"] for s in selectors]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        bad.append(f"selector names given twice (they share {FILES['scores']}): {twice}")
    if bad:
        raise ValueError("e2: " + "; ".join(bad))
    return {"e2": e2, "policy": models[e2["policy"]], "labeller": models[e2["labeller"]],
            "e3_max_tokens": int(e3_max_tokens), "selectors": selectors, "reward_models_missing": missing}


def projection_status(path: Path, row_id: str) -> dict:
    """Whether the projection's final plan counts a row: {"counted", "final_plan"}, read from the first branch of
    projection.json (the branch of the gates that ran). final_plan is the text of the projection markdown's "final plan"
    column for the row (projection_v4._row_status; projection.json has no field of that name). counted: for an optional
    OpenRouter row, the branch's optional_openrouter_fits is True; for another row, its final "counted". ValueError
    when the file or the row is missing."""
    from overclaim import projection_v4

    if not path.exists():
        raise ValueError(f"the projection {path} does not exist")
    projection = _read_json(path, {})
    rows = {r["id"]: r for r in projection["rows"]}
    if row_id not in rows:
        raise ValueError(f"the projection {path} has no row {row_id!r}")
    budget = projection["branches"][0]["budget"]
    row = rows[row_id]
    counted = (budget.get("optional_openrouter_fits") is True if row.get("optional_openrouter")
               else bool(budget["final"]["rows"][row_id]["counted"]))
    return {"counted": counted, "final_plan": projection_v4._row_status(row, budget)}


def planned_skips(plan: dict) -> dict[str, dict[str, str]]:
    """{selector name: {"reason", "detail"}} for the selectors that do not run whatever the gates say: an optional
    mitigated selector whose projection row is not counted (projection_status on e2.projection). Reason and detail are
    both the projection's verdict, a text that changes only with the projection."""
    out = {}
    for sel in plan["selectors"]:
        if sel.get("optional"):
            path = plan["e2"]["projection"]
            status = projection_status(_repo_path(path), sel["projection_row"])
            if not status["counted"]:
                why = f"projection row {sel['projection_row']} ({path}): {status['final_plan']}"
                out[sel["name"]] = {"reason": why, "detail": why}
    return out


def is_exploratory(selector: dict) -> bool:
    """Whether a selector of the plan is one of the exploratory additions of 2026-11-07 (EXPLORATORY_NOTE): a reward
    model (E5(b), README amendment 3, item 2) or a judge through OpenRouter (the open-weight selector and its mitigated
    selector, item 3)."""
    return selector["kind"] == "reward_model" or selector["provider"] == "openrouter"


def frozen_task_dicts(cfg: dict, doc: dict, ids: list[str]) -> list[dict]:
    """The task dicts of ids, in that order: each the first record of its task id in the task file that the frozen
    task file names for it (tasks[].task_file, one of freeze.task_files by file name), as the freeze's join read them.
    ValueError naming a task file that is not among freeze.task_files or an id found in no named file."""
    by_name = {Path(p).name: _repo_path(p) for p in cfg["freeze"]["task_files"]}
    file_of = {t["task_id"]: t["task_file"] for t in doc["tasks"]}
    unknown = sorted({file_of.get(i) for i in ids} - set(by_name))
    if unknown:
        raise ValueError(f"task files {unknown} of the frozen task file are not among freeze.task_files")
    wanted = set(ids)
    found: dict[str, dict] = {}
    for name in dict.fromkeys(file_of[i] for i in ids):
        for task in _read_jsonl(by_name[name]):
            if task["task_id"] in wanted and file_of[task["task_id"]] == name:
                found.setdefault(task["task_id"], task)
    missing = [i for i in ids if i not in found]
    if missing:
        raise ValueError(f"{len(missing)} Experiment 2 tasks are not in the task file named for them: {missing[:3]}")
    return [found[i] for i in ids]


def e2_inputs(cfg: dict, n_tasks: int) -> dict:
    """The frozen inputs: {"ids": picks.e2_tasks (pick order), "shown": picks.e2_shown_eligible, "gap": {task_id:
    gap_type}, "resolved": {task_id: resolved}, "tasks": frozen_task_dicts of ids, "evidence": {task_id: the frozen
    set's evidence_v4}} from freeze.tasks_out and freeze.sets_out. ValueError when a frozen file is missing, the pick
    does not hold n_tasks ids, the shown-eligible ids are not the picked ids whose task is shown_eligible (pick order),
    or a picked task has no frozen set with evidence_v4."""
    fz = cfg["freeze"]
    tasks_path, sets_path = _repo_path(fz["tasks_out"]), _repo_path(fz["sets_out"])
    missing = [str(p) for p in (tasks_path, sets_path) if not p.exists()]
    if missing:
        raise ValueError(f"the frozen files {missing} are missing; run the freeze stage first")
    doc = _read_json(tasks_path, {})
    ids, shown = list(doc["picks"]["e2_tasks"]), list(doc["picks"]["e2_shown_eligible"])
    by_id = {t["task_id"]: t for t in doc["tasks"]}
    problems = []
    if len(ids) != n_tasks or len(set(ids)) != len(ids):
        problems.append(f"picks.e2_tasks holds {len(ids)} ids ({len(set(ids))} distinct), not e2.n_tasks {n_tasks}")
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        problems.append(f"picked tasks not in the frozen task list: {unknown[:3]}")
    elif shown != [i for i in ids if by_id[i]["shown_eligible"]]:
        problems.append("picks.e2_shown_eligible is not the picked tasks that are shown_eligible, in pick order")
    sets = {s["task_id"]: s for s in _read_jsonl(sets_path)}
    no_evidence = [i for i in ids if "evidence_v4" not in sets.get(i, {})]
    if no_evidence:
        problems.append(f"picked tasks without a frozen set with evidence_v4: {no_evidence[:3]}")
    if problems:
        raise ValueError("e2 inputs: " + "; ".join(problems))
    return {"ids": ids, "shown": shown, "gap": {i: by_id[i]["gap_type"] for i in ids},
            "resolved": {i: int(by_id[i]["resolved"]) for i in ids}, "tasks": frozen_task_dicts(cfg, doc, ids),
            "evidence": {i: sets[i]["evidence_v4"] for i in ids}}


# ---------------------------------------------------------------------------------------------
# The run plan, gates and scorers
# ---------------------------------------------------------------------------------------------


def run_plan(run_dir: Path, plan: dict, inputs: dict) -> list[dict]:
    """One entry per run in run order, the policy, the two labellers, then every selector of the plan: {run, kind
    ("policy", "labels" or a selector kind), provider, model, n_tasks, calls (first attempts: tasks x the pool size),
    on_disk (of those keys, the records already in the run's file under run_dir), todo (calls - on_disk)}."""
    n_pool = 2 * int(plan["e2"]["n_per_instruction"])
    ids = {"e2_tasks": inputs["ids"], "e2_shown_eligible": inputs["shown"]}

    def keys(task_set: str) -> set[tuple]:
        return {(t, i) for t in ids[task_set] for i in range(n_pool)}

    def on_disk(path: Path, wanted: set[tuple], selector: str | None = None) -> int:
        return len({(r["task_id"], r["sample"]) for r in _read_jsonl(path)
                    if selector is None or r.get("selector") == selector} & wanted)

    def entry(run: str, kind: str, provider: str, model: str, task_set: str, path: Path,
              selector: str | None = None) -> dict:
        wanted = keys(task_set)
        done = on_disk(path, wanted, selector)
        return {"run": run, "kind": kind, "provider": provider, "model": model, "n_tasks": len(ids[task_set]),
                "calls": len(wanted), "on_disk": done, "todo": len(wanted) - done}

    files = {k: run_dir / v for k, v in FILES.items()}
    policy, labeller = plan["policy"], plan["labeller"]
    runs = [entry("policy", "policy", policy["provider"], policy["model"], "e2_tasks", files["samples"]),
            entry("labels session3", "labels", labeller["provider"], labeller["model"], "e2_tasks", files["labels"]),
            entry("labels full_evidence", "labels", labeller["provider"], labeller["model"], "e2_tasks",
                  files["labels_full"])]
    return runs + [entry(s["name"], s["kind"], s["provider"], s["model"], s["tasks"], files["scores"], s["name"])
                   for s in plan["selectors"]]


def provider_gate(ctx: Ctx6, provider: str) -> str | None:
    """Reason not to launch a run of `provider` now, or None: pilot2.gate for Anthropic; for OpenRouter the stop line
    budget.openrouter_stop_launching_usd against the run's own cost file (the lead's rule 2); none for the local GPU or
    Google (whose cap the CostLogger enforces per call)."""
    if provider == "anthropic":
        return pilot2.gate(ctx, {}, "anthropic")
    if provider == "openrouter":
        spent, stop = ctx.cost.spent("openrouter"), float(ctx.cfg["budget"]["openrouter_stop_launching_usd"])
        if spent >= stop:
            return f"OpenRouter total ${spent:.2f} has reached the stop-launching line ${stop:.2f}"
    return None


def vfp_scorer(domain: str, selector: dict) -> Callable[[dict, str, Any], dict]:
    """e3_verification_pointwise.make_vfp_scorer(domain, the selector's structured flag); for a judge without a prompt
    cache (provider not in PROMPT_CACHE_PROVIDERS) the result's live_cache_reads is dropped, as score_samples_pointwise
    drops it, so score_samples' cache-read check applies to the Anthropic judge only."""
    from overclaim.experiments.e3_verification_pointwise import make_vfp_scorer

    scorer = make_vfp_scorer(domain, selector["structured"])
    if selector["provider"] in PROMPT_CACHE_PROVIDERS:
        return scorer

    def without_cache_reads(task: dict, message: str, llm: Any) -> dict:
        result = scorer(task, message, llm)
        result.pop("live_cache_reads", None)
        return result

    return without_cache_reads


def rm_score(task: dict, message: str, model: Any) -> dict:
    """The reward-model scorer for e2_selection.score_samples: model.score_detail(render.hidden_context(repo,
    issue_text), message), the input of the E1 reward-model scores (the issue text plus the candidate message) ->
    {"score": float, "reason": "", "model", "input_tokens", "cached"}."""
    detail = model.score_detail(render.hidden_context(task["repo"], task["issue_text"]), message)
    return {"score": detail["score"], "reason": "", "model": model.model_name, "input_tokens": detail["input_tokens"],
            "cached": detail["cached"]}


def _not_called(task: dict, message: str, model: Any) -> dict:
    raise RuntimeError("every score of this reward model was expected on disk")


def _release(model: Any) -> None:
    """Drop the model's weights and free the GPU memory they held, as pilot2.stage_rm does after each model."""
    model.model = model.tokenizer = None
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _seconds(before: Any, now: float | None) -> float | None:
    """A stored total (or None) plus this run's time (or None), rounded to 0.1 s; None when neither is a number."""
    if before is None and now is None:
        return None
    return round((before or 0.0) + (now or 0.0), 1)


def run_reward_model(ctx: Ctx6, selector: dict, samples: list[dict], tasks: list[dict], out_path: Path, todo: int,
                     previous: dict) -> tuple[list[dict] | None, dict, dict[str, str] | None]:
    """One reward-model selector: (its score records, or None when it did not run; its e2_rm_status.json entry; None,
    or {"reason", "detail"} why it did not run, reason the same from run to run).

    previous is the model's entry in e2_rm_status.json ({} when there is none). With nothing left to score (todo 0)
    the records come from out_path and no model is built; a previous entry with status done is then returned as it is
    (its times were measured by the runs that wrote the records). Otherwise rm.RewardModel from the registry entry
    (dtype, batch_size, quantization; the stage's disk cache); a model that check_fits refuses is skipped (reason "does
    not fit on the GPU (RewardModel.check_fits)", the check's text as detail), and so is one whose checkpoint download
    does not fit on the disk (check_disk_fits, 2026-11-08); else load() and then score_samples(...,
    scorer=rm_score) in one thread, each timed on the wall clock (load_seconds includes a first download of the weights
    from the Hub; score_seconds is the scoring alone), and the model released afterwards. An exception of LOAD_ERRORS
    from load() or of SCORE_ERRORS from the scoring is not raised: the model is left out with status failed (reason
    "failed while loading (<exception type>)" or "... scoring ...", the exception's text as detail), and the records
    it wrote stay in out_path for a later run to resume; any other exception (a stale score file among them) fails the
    stage. Entry: model, quantization, status (done, skipped or failed), reason (check_fits' text, or "while <phase>:
    <exception>"), load_seconds and score_seconds (this run's time plus, when records of earlier runs were on disk,
    the previous entry's totals; None when no time was measured), then for done n_records and n_cache_hits (scores
    from the reward model's disk cache), for failed n_records_on_disk."""
    entry, name = selector["entry"], selector["name"]
    status: dict[str, Any] = {"model": entry["model"], "quantization": entry.get("quantization")}
    base = previous if len(samples) - todo > 0 else {}     # records of earlier runs on disk: their time counts too

    def times(load_s: float | None, score_s: float | None) -> dict:
        return {"load_seconds": _seconds(base.get("load_seconds"), load_s),
                "score_seconds": _seconds(base.get("score_seconds"), score_s)}

    if not todo:
        records = e2_selection.score_samples(samples, tasks, None, name, out_path, scorer=_not_called)
        if previous.get("status") == "done":
            return records, previous, None
        return records, {**status, "status": "done", "reason": "every score was on disk; this run loaded no model",
                         **times(None, None), "n_records": len(records),
                         "n_cache_hits": sum(bool(r.get("cached")) for r in records)}, None
    from overclaim.graders import rm

    model = rm.RewardModel(entry["model"], dtype=entry.get("dtype", "bfloat16"),
                           batch_size=int(entry.get("batch_size", 1)), cache=ctx.cache, name=name,
                           quantization=entry.get("quantization"))
    fits, why = model.check_fits()
    logger.info("e2: %s: %s", name, why)
    if not fits:
        return None, {**status, "status": "skipped", "reason": why, **times(None, None)}, \
            {"reason": "does not fit on the GPU (RewardModel.check_fits)", "detail": why}
    disk_fits, disk_why = model.check_disk_fits()      # 2026-11-08: never start a download that fills the disk
    logger.info("e2: %s: %s", name, disk_why)
    if not disk_fits:
        return None, {**status, "status": "skipped", "reason": disk_why, **times(None, None)}, \
            {"reason": "its download does not fit on the disk (RewardModel.check_disk_fits)", "detail": disk_why}

    def failed(phase: str, exc: Exception, load_s: float | None, score_s: float | None) -> tuple[None, dict, dict]:
        detail = f"{type(exc).__name__}: {exc}"
        logger.error("e2: reward model %s failed while %s and is left out: %s", name, phase, detail)
        on_disk = sum(r.get("selector") == name for r in _read_jsonl(out_path))
        return None, {**status, "status": "failed", "reason": f"while {phase}: {detail}", **times(load_s, score_s),
                      "n_records_on_disk": on_disk}, {"reason": f"failed while {phase} ({type(exc).__name__})",
                                                      "detail": detail}

    start = time.monotonic()
    try:
        try:
            model.load()
        except LOAD_ERRORS as exc:
            return failed("loading", exc, time.monotonic() - start, None)
        load_s = time.monotonic() - start
        start = time.monotonic()
        try:
            records = e2_selection.score_samples(samples, tasks, model, name, out_path, scorer=rm_score)
        except SCORE_ERRORS as exc:
            return failed("scoring", exc, load_s, time.monotonic() - start)
        score_s = time.monotonic() - start
    finally:
        _release(model)
    return records, {**status, "status": "done", "reason": why, **times(load_s, score_s), "n_records": len(records),
                     "n_cache_hits": sum(bool(r.get("cached")) for r in records)}, None


def run_selectors(ctx: Ctx6, plan: dict, inputs: dict, samples: list[dict], skipped: dict[str, dict[str, str]],
                  todo: dict[str, int]) -> tuple[list[dict], list[str], list[str]]:
    """Every selector of the plan in run order that is not in `skipped`, on its task set's samples: plain ones with
    score_samples_pointwise (hidden), mitigated ones with score_samples(scorer=vfp_scorer(cfg["domain"], ...)),
    reward models with run_reward_model (given the model's entry of run_dir/e2_rm_status.json, which the entry it
    returns replaces unless it is that same entry). A selector with calls left (todo) is launched only when
    provider_gate allows it (else {"reason": GATE_REASONS[provider], "detail": the gate's text}) and its client has a
    credential (else "no credential for <provider>"); those and the reward models that did not run go into `skipped`
    (in place). After a reward model the number of its scores that are not finite numbers is logged (score_samples'
    own count of invalid scores reads integers 1 to 10 only). Returns (the score records of the selectors that ran, in
    plan order; their names; the reward models that failed while loading or scoring, also in `skipped`)."""
    cfg = ctx.cfg
    out_path = ctx.run_dir / FILES["scores"]
    status_path = ctx.run_dir / RM_STATUS
    shown = set(inputs["shown"])
    by_set = {"e2_tasks": samples, "e2_shown_eligible": [s for s in samples if s["task_id"] in shown]}
    records: list[dict] = []
    ran: list[str] = []
    failed: list[str] = []
    for sel in plan["selectors"]:
        name, provider = sel["name"], sel["provider"]
        if name in skipped:
            continue
        reason = provider_gate(ctx, provider) if todo[name] else None
        if reason is not None:
            logger.warning("e2: selector %s not launched: %s", name, reason)
            skipped[name] = {"reason": GATE_REASONS[provider], "detail": reason}
            continue
        subset = by_set[sel["tasks"]]
        if sel["kind"] == "reward_model":
            previous = _read_json(status_path, {}).get(name) or {}
            got, status, skip = run_reward_model(ctx, sel, subset, inputs["tasks"], out_path, todo[name], previous)
            if status is not previous:
                _write_json(status_path, {**_read_json(status_path, {}), name: status})
            if got is None:
                skipped[name] = skip
                if status["status"] == "failed":
                    failed.append(name)
                continue
            logger.info("e2: %s: %d records, %d scores that are not finite numbers (score_samples' count of invalid "
                        "scores reads integers 1 to 10 only; best of N reads these floats through ordinal_scores)",
                        name, len(got), sum(not _finite(r.get("score")) for r in got))
        else:
            client = ctx.client(sel["spec"]) if todo[name] else None
            if todo[name] and client is None:
                why = f"no credential for {provider}"
                skipped[name] = {"reason": why, "detail": why}
                continue
            workers = int(cfg["max_workers"].get(provider, 1))
            if sel["kind"] == "plain":
                got = e2_selection.score_samples_pointwise(subset, inputs["tasks"], client, name, out_path,
                                                           structured=sel["structured"], max_workers=workers)
            else:
                got = e2_selection.score_samples(subset, inputs["tasks"], client, name, out_path, max_workers=workers,
                                                 scorer=vfp_scorer(cfg["domain"], sel))
        records += got
        ran.append(name)
    return records, ran, failed


# ---------------------------------------------------------------------------------------------
# Analysis: float scores, the paired comparisons, E4
# ---------------------------------------------------------------------------------------------


def _finite(score: Any) -> bool:
    return isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)


def ordinal_scores(samples: list[dict], scores: list[dict], selectors: set[str],
                   ns: tuple[int, ...] = e2_selection.NS) -> list[dict]:
    """Copies of `scores` in which each score of `selectors` (a reward model's float) is replaced by an integer that
    e2_selection reads as a valid score and that leaves every best-of-N pick as a float-reading best_of_n_detail would
    make it; the floats stay in e2_scores.jsonl. The blocks are best_of_n_detail's as prep5_e2.selection_rows feeds
    them: each task's samples in block_positions order (the instructions interleaved) cut into consecutive blocks of N,
    for each N of ns. A valid score (a finite int or float, not a bool) becomes 1 plus the number of N > 1 in ns for
    which it equals the top valid score of its block of N. The blocks nest (each N divides the next), so the top of a
    block is a top of every smaller block inside it: within every block, the samples with the top integer are exactly
    those with the top float, ties stay ties in the same sample order and the seeded tie-break picks the same sample.
    Any other score of `selectors` becomes None (invalid, as before); other records are returned unchanged.
    ValueError when ns do not nest, a task's sample count is not a multiple of the largest N, or a score names a
    sample that is not in `samples`."""
    sizes = sorted(set(ns))
    if sizes[0] < 1 or any(b % a for a, b in zip(sizes, sizes[1:])):
        raise ValueError(f"ns {list(ns)} do not nest")
    moved, _, _ = prep5_e2.block_positions(samples, [], [])
    by_task: dict[str, list[tuple[int, int]]] = {}
    for s, m in zip(samples, moved):
        by_task.setdefault(s["task_id"], []).append((m["sample"], s["sample"]))
    rank: dict[tuple[str, int], int] = {}
    for task_id, pairs in by_task.items():
        if len(pairs) % sizes[-1]:
            raise ValueError(f"task {task_id} has {len(pairs)} samples, not a multiple of N={sizes[-1]}")
        for r, (_, sample) in enumerate(sorted(pairs)):
            rank[(task_id, sample)] = r
    value: dict[tuple[str, str, int], float] = {}
    for r in scores:
        if r["selector"] in selectors:
            if (r["task_id"], r["sample"]) not in rank:
                raise ValueError(f"a score of {r['selector']} names no sample: {(r['task_id'], r['sample'])}")
            if _finite(r.get("score")):
                value[(r["selector"], r["task_id"], r["sample"])] = float(r["score"])
    top: dict[tuple, float] = {}
    for (sel, task_id, sample), v in value.items():
        for n in sizes:
            block = (sel, task_id, n, rank[(task_id, sample)] // n)
            top[block] = max(top.get(block, v), v)
    out = []
    for r in scores:
        key = (r["selector"], r["task_id"], r["sample"])
        if r["selector"] not in selectors:
            out.append(r)
        elif key not in value:
            out.append({**r, "score": None})
        else:
            level = 1 + sum(value[key] == top[(*key[:2], n, rank[key[1:]] // n)] for n in sizes if n > 1)
            out.append({**r, "score": level})
    return out


def per_task_values(samples: list[dict], scores: list[dict], labels: list[dict], *, n: int, n_tie_seeds: int,
                    outcome: str = COMPARISON_OUTCOME) -> dict[str, dict[str, float]]:
    """{selector: {task_id: the share of the task's blocks of n whose selected sample has `outcome`, a tie for the top
    score averaged over n_tie_seeds seeds}}: e2_selection.best_of_n_averaged on prep5_e2.outcome_labels(labels,
    outcome), over the block positions of block_positions, as prep5_e2.selection_rows computes each task's value. A
    task without a value at n (no valid score, or no selected sample with a valid label) is absent."""
    s, sc, lab = prep5_e2.block_positions(samples, scores, prep5_e2.outcome_labels(labels, outcome))
    per_task, _ = e2_selection.best_of_n_averaged(s, sc, lab, ns=(n,), n_tie_seeds=n_tie_seeds)
    return {sel: {t: c[n]["averaged"] for t, c in cells.items() if n in c} for sel, cells in per_task.items()}


def paired_difference(values: dict[str, dict[str, float]], treated: str, control: str, *, tasks: list[str],
                      n_boot: int, seed: int) -> dict:
    """treated minus control, paired by task over the tasks of `tasks` (the comparison's task set) where both have a
    value: n_tasks, treated_mean, control_mean, mean (of the per-task differences), ci (95% percentile bootstrap over
    tasks, analysis.bootstrap_ci on the differences in task-id order), below_zero (the CI's upper end is below zero;
    None without a task), by_task {task_id: difference} and left_out {task_id: "no value for <selector>" (both names
    when neither has one)} for every other task of `tasks` (per_task_values gives no value for a task without a valid
    score or without a selected sample with a valid label). NaN for an undefined mean or CI."""
    a, b = values.get(treated, {}), values.get(control, {})
    wanted = sorted(set(tasks))
    common = [t for t in wanted if t in a and t in b]
    left_out = {t: "no value for " + " and ".join(name for name, v in ((treated, a), (control, b)) if t not in v)
                for t in wanted if t not in a or t not in b}
    by_task = {t: a[t] - b[t] for t in common}
    diffs = analysis._sorted_values(by_task)
    ci = analysis.bootstrap_ci(diffs, n_boot, seed)
    return {"treated": treated, "control": control, "n_tasks": len(common),
            "treated_mean": analysis._mean([a[t] for t in common]),
            "control_mean": analysis._mean([b[t] for t in common]), "mean": analysis._mean(diffs), "ci": list(ci),
            "below_zero": bool(ci[1] < 0) if diffs else None, "by_task": by_task, "left_out": left_out}


def comparisons(selectors: list[dict], ran: list[str], reference: str) -> list[dict]:
    """The paired comparisons at N = 16, in plan order: each mitigated selector that ran against its judge's plain
    selector (h4 True for the one that is not optional and whose judge is the reference selector: H4's second
    condition), then each reward model that ran against the reference plain selector. A comparison whose control did
    not run is left out (the control's skip reason is in the table). Each: {treated, control, kind, task_set, h4,
    exploratory (either selector is_exploratory)}."""
    by_name = {s["name"]: s for s in selectors}
    out = []
    for sel in selectors:
        if sel["name"] not in ran:
            continue
        if sel["kind"] == "mitigated" and sel["judge"] in ran:
            control = sel["judge"]
            entry = {"kind": "mitigated", "task_set": "e2_shown_eligible",
                     "h4": not sel["optional"] and sel["judge"] == reference}
        elif sel["kind"] == "reward_model" and reference in ran:
            control = reference
            entry = {"kind": "reward_model", "task_set": "e2_tasks", "h4": False}
        else:
            continue
        out.append({"treated": sel["name"], "control": control, **entry,
                    "exploratory": is_exploratory(sel) or is_exploratory(by_name[control])})
    return out


def e4_note(resolved: dict[str, int], ids: list[str], shown: list[str]) -> dict:
    """E4's one number per task set: {"e2_tasks" and "e2_shown_eligible": {resolved (tasks with resolved 1), n_tasks,
    share}, "sentence": E4_SENTENCE with both numbers}."""
    out: dict[str, Any] = {}
    for name, group in (("e2_tasks", ids), ("e2_shown_eligible", shown)):
        k = sum(int(resolved[t]) for t in group)
        out[name] = {"resolved": k, "n_tasks": len(group), "share": k / len(group) if group else None}
    a, b = out["e2_tasks"], out["e2_shown_eligible"]

    def share(x: dict) -> str:
        return analysis.NA if x["share"] is None else f"{x['share']:.3f}"

    out["sentence"] = (f"{E4_SENTENCE} Resolved: {a['resolved']} of {a['n_tasks']} Experiment 2 tasks ({share(a)}) and "
                       f"{b['resolved']} of {b['n_tasks']} shown-eligible Experiment 2 tasks ({share(b)}), whatever "
                       "the selector and N.")
    return out


def _restrict(records: list[dict], ids: set[str]) -> list[dict]:
    return [r for r in records if r["task_id"] in ids]


def e2_table(plan: dict, inputs: dict, pool: dict, scores: list[dict], ran: list[str],
             skipped: dict[str, dict[str, str]], *, n_tie_seeds: int, n_boot: int, seed: int) -> dict:
    """The table (e2_table.json before NaN becomes None) from the pool (run_pool's samples and labels), the score
    records of the selectors that ran and the skip entries: tasks, n_samples, selectors (each planned one: name, kind,
    provider, model, tasks, ran, exploratory), selectors_skipped {name: the skip entry's reason, a text that is the
    same from run to run}, reward_models_missing (e2_config's reason, or None), exploratory_note, block_note and
    score_note; selection {task set: {labeller: prep5_e2.selection_rows}} on the 40 tasks (plain and reward-model
    selectors) and on the shown-eligible tasks (every selector), the reward models' scores recoded by ordinal_scores;
    comparisons {labeller: paired_difference of each comparison at N = 16 on clean_overclaim over its task set, with
    the tasks left out}; h4_second_condition (the h4 comparison per labeller, read on PRIMARY_LABELLER; None when it
    did not run); base_rates (sample flags and prep5_e2.base_rates_mixed per labeller, per instruction and pooled),
    top_ties, selected_at_pool_size (the instruction of the sample selected at N = 16, a descriptive), e4 (e4_note,
    flagged exploratory); n_tie_seeds, n_bootstrap and seed."""
    samples, labels = pool["samples"], pool["labels"]
    kind_of = {s["name"]: s["kind"] for s in plan["selectors"]}
    rms = {n for n in ran if kind_of[n] == "reward_model"}
    recoded = ordinal_scores(samples, scores, rms)
    shown = set(inputs["shown"])
    task_ids = {"e2_tasks": inputs["ids"], "e2_shown_eligible": inputs["shown"]}
    on_all = {n for n in ran if kind_of[n] != "mitigated"}
    subsets = {"e2_tasks": (samples, [r for r in recoded if r["selector"] in on_all], labels),
               "e2_shown_eligible": (_restrict(samples, shown), _restrict(recoded, shown),
                                     {lab: _restrict(labels[lab], shown) for lab in labels})}
    selection = {name: {lab: prep5_e2.selection_rows(s, sc, lab_recs[lab], n_tie_seeds=n_tie_seeds, n_boot=n_boot,
                                                     seed=seed) for lab in prep5_e2.LABELLERS}
                 for name, (s, sc, lab_recs) in subsets.items()}
    planned = comparisons(plan["selectors"], ran, plan["e2"]["reference_selector"])
    compared: dict[str, list[dict]] = {}
    for lab in prep5_e2.LABELLERS:
        values = per_task_values(samples, recoded, labels[lab], n=COMPARISON_N, n_tie_seeds=n_tie_seeds)
        compared[lab] = [{**c, **paired_difference(values, c["treated"], c["control"], tasks=task_ids[c["task_set"]],
                                                   n_boot=n_boot, seed=seed)}
                         for c in planned]
    h4 = next((c for c in planned if c["h4"]), None)
    h4_out = None
    if h4 is not None:
        by_lab = {lab: next(c for c in compared[lab] if c["treated"] == h4["treated"]) for lab in prep5_e2.LABELLERS}
        by_lab = {lab: {k: c[k] for k in ("n_tasks", "treated_mean", "control_mean", "mean", "ci", "below_zero",
                                          "left_out")}
                  for lab, c in by_lab.items()}
        h4_out = {"treated": h4["treated"], "control": h4["control"], "n": COMPARISON_N,
                  "outcome": COMPARISON_OUTCOME, "read_on": PRIMARY_LABELLER, "by_labeller": by_lab,
                  "holds": by_lab[PRIMARY_LABELLER]["below_zero"],
                  "labellers_agree": len({c["below_zero"] for c in by_lab.values()}) == 1}
    n_per = int(plan["e2"]["n_per_instruction"])
    return {
        "part": "session 5, Part 2, Step 11 (Experiment 2 in full)",
        "tasks": {"e2_tasks": inputs["ids"], "e2_shown_eligible": inputs["shown"]},
        "n_samples": len(samples),
        "selectors": [{"name": s["name"], "kind": s["kind"], "provider": s["provider"], "model": s["model"],
                       "tasks": s["tasks"], "ran": s["name"] in ran, "exploratory": is_exploratory(s)}
                      for s in plan["selectors"]],
        "selectors_skipped": {name: entry["reason"] for name, entry in skipped.items()},
        "reward_models_missing": plan["reward_models_missing"],
        "exploratory_note": EXPLORATORY_NOTE,
        "block_note": (f"Best of N cuts each task's {2 * n_per} samples into consecutive blocks of N in an interleaved "
                       f"order, A0, B{n_per}, A1, B{n_per + 1}, ... (prep5_e2.block_positions; samples 0 to "
                       f"{n_per - 1} are A and {n_per} to {2 * n_per - 1} are B), so each N = 4 block holds 2 A and "
                       f"2 B samples and N = {2 * n_per} selects from the whole pool. N = 1 is the mean over the "
                       "task's samples."),
        "score_note": ("A reward model's float scores enter best of N recoded to integers that keep the top set of "
                       "every block of every N (full_run_e2.ordinal_scores; e2_selection reads only integer scores "
                       "1 to 10), so its picks and ties are the floats'; the floats are in e2_scores.jsonl."),
        "selection": selection,
        "comparisons": compared,
        "h4_second_condition": h4_out,
        "base_rates": {"samples": prep5_e2.sample_flags(samples),
                       **{lab: prep5_e2.base_rates_mixed(samples, labels[lab]) for lab in prep5_e2.LABELLERS}},
        "top_ties": prep5_e2.top_ties(samples, recoded),
        "selected_at_pool_size": prep5_e2.selected_split(samples, recoded, n_tie_seeds=n_tie_seeds),
        "e4": {**e4_note(inputs["resolved"], inputs["ids"], inputs["shown"]), "exploratory": True},
        "n_tie_seeds": n_tie_seeds, "n_bootstrap": n_boot, "seed": seed,
    }


def _num(x: float | None, digits: int = 3) -> str:
    return analysis.NA if x is None else f"{x:.{digits}f}"


def table_markdown(t: dict) -> str:
    """e2_table.md from the table as written to e2_table.json (NaN as None, the N keys as strings): the tasks and
    notes (the exploratory parts among them), H4's second condition per labeller, the paired comparisons at N = 16 with
    the tasks each leaves out, E4's sentence, the best-of-N curve of every selector per labeller (both task sets in one
    table), the base rates per instruction, the sample flags, the top-score ties, the instruction of the sample
    selected at N = 16 and the selectors that did not run, with their reasons. An exploratory selector's name carries
    "(exploratory)" in every table."""
    ns = e2_selection.NS
    kind_of = {s["name"]: s["kind"] for s in t["selectors"]}
    exploratory = {s["name"] for s in t["selectors"] if s["exploratory"]}
    lab_name = prep5_e2.LABELLER_NAMES

    def label(name: str) -> str:
        return f"{name} (exploratory)" if name in exploratory else name

    n_ids, n_shown = len(t["tasks"]["e2_tasks"]), len(t["tasks"]["e2_shown_eligible"])
    lines = ["# Experiment 2 in full (session 5, Part 2, Step 11)", "",
             f"{n_ids} tasks (picks.e2_tasks), {n_shown} of them shown-eligible (picks.e2_shown_eligible), "
             f"{t['n_samples']} samples. {t['block_note']} {t['score_note']} Each best-of-N cell: the share of the "
             f"selected samples with the outcome, a tie for the top score averaged over tie-break seeds 0 to "
             f"{t['n_tie_seeds'] - 1}; 95% percentile bootstrap over tasks ({t['n_bootstrap']} resamples, seed "
             f"{t['seed']}). The mitigated selectors score the shown-eligible tasks only; every selector is tabled on "
             f"those, the others on all tasks as well. {t['exploratory_note']}", "",
             f"## H4, second condition (N = {COMPARISON_N})", ""]
    h4 = t["h4_second_condition"]
    if h4 is None:
        lines.append("Not computed: the mitigated selector of the reference judge or its plain selector did not run "
                     "(see the selectors that did not run, below).")
    else:
        verdict = {True: "holds", False: "does not hold", None: "undefined"}[h4["holds"]]
        lines += [f"Clean-overclaim rate of the sample selected at N = {h4['n']}, {h4['treated']} minus "
                  f"{h4['control']}, paired by task. H4's second condition (the CI of the difference below zero), "
                  f"read on the {lab_name[h4['read_on']]}: **{verdict}**. The two labellers "
                  f"{'agree' if h4['labellers_agree'] else 'DISAGREE'}.", "",
                  *analysis._md_table(["labeller", "tasks", "left out", label(h4["treated"]), label(h4["control"]),
                                       "difference [95% CI]", "CI below zero"],
                                      [[lab_name[lab], str(c["n_tasks"]), str(len(c["left_out"])),
                                        _num(c["treated_mean"]), _num(c["control_mean"]), e2_selection._cell(c),
                                        str(c["below_zero"])]
                                       for lab, c in h4["by_labeller"].items()])]
    rows = [[lab_name[lab], label(c["treated"]), label(c["control"]), c["task_set"], str(c["n_tasks"]),
             str(len(c["left_out"])), _num(c["treated_mean"]), _num(c["control_mean"]), e2_selection._cell(c),
             str(c["below_zero"])]
            for lab, comps in t["comparisons"].items() for c in comps]
    left = [f"- {lab_name[lab]}, {c['treated']} against {c['control']}: "
            + "; ".join(f"{task} ({why})" for task, why in c["left_out"].items())
            for lab, comps in t["comparisons"].items() for c in comps if c["left_out"]]
    lines += ["", f"## Paired comparisons at N = {COMPARISON_N} ({COMPARISON_OUTCOME})", "",
              "Per task, the tie-averaged share of the selected sample with the outcome under the first selector minus "
              "under the second, over the tasks of the comparison's task set where both have a value; a task where "
              "one has none (no valid score, or no selected sample with a valid label) is left out and named below.",
              "",
              *(analysis._md_table(["labeller", "selector", "against", "task set", "tasks", "left out",
                                    "selector mean", "against mean", "difference [95% CI]", "CI below zero"], rows)
                if rows else ["No comparison ran."]),
              *(["", "Tasks left out:", "", *left] if left else []),
              "", "## E4: the resolved rate of the selected sample (exploratory)", "", t["e4"]["sentence"]]
    for lab in prep5_e2.LABELLERS:
        rows = []
        for task_set in TASK_SETS:
            for r in t["selection"][task_set][lab]:
                avg = r["tie_averaged"]
                rows.append([task_set, label(r["selector"]), kind_of[r["selector"]], r["outcome"],
                             *(e2_selection._cell(avg["rates"][str(n)]) for n in ns), e2_selection._cell(avg["diff"]),
                             str(avg["n_tasks"]), str(r["counts"]["invalid_scores"]),
                             str(r["counts"]["invalid_labels"])])
        lines += ["", f"## Best of N, {lab_name[lab]}", "",
                  *analysis._md_table(["task set", "selector", "kind", "outcome", *(f"N={n}" for n in ns),
                                       f"N={ns[-1]} minus N={ns[0]}", "tasks", "invalid scores", "invalid labels"],
                                      rows)]
    rows = [[lab_name[lab], name, f"{b['n_labelled']} of {b['n_samples']}",
             *(f"{b[f]['n']} ({_num(b[f]['rate'])})" for f in prep5_e2.RATE_FIELDS)]
            for lab in prep5_e2.LABELLERS for name, b in t["base_rates"][lab].items()]
    lines += ["", "## Base rates per instruction (over the samples, a descriptive)", "",
              *analysis._md_table(["labeller", "instruction", "labelled", *prep5_e2.RATE_FIELDS], rows), "",
              *analysis._md_table(["instruction", "samples", "truncated", "empty", "duplicate"],
                                  [[name, str(f["n_samples"]), str(f["truncated"]), str(f["empty"]),
                                    str(f["duplicate"])] for name, f in t["base_rates"]["samples"].items()])]
    lines += ["", "## Ties for the top score per selector", "",
              *analysis._md_table(["selector", "mean samples tied at the top", "mean share tied at the top",
                                   "tasks with a tie"],
                                  [[label(sel), _num(v["mean_tied_at_top"], 2), _num(v["mean_share_tied_at_top"]),
                                    f"{v['tasks_with_a_tie']} of {v['n_tasks']}"] for sel, v in t["top_ties"].items()])]
    names = sorted({k for v in t["selected_at_pool_size"].values() for k in v if k not in ("n_tasks", "by_task")})
    lines += ["", f"## Instruction of the sample selected at N = {ns[-1]} (a descriptive; ties averaged as above)", "",
              *analysis._md_table(["selector", *(f"share {n}" for n in names), "tasks"],
                                  [[label(sel), *(_num(v.get(n)) for n in names), str(v["n_tasks"])]
                                   for sel, v in t["selected_at_pool_size"].items()])]
    not_run = [f"- {label(name)}: {reason}" for name, reason in t["selectors_skipped"].items()]
    if t["reward_models_missing"]:
        not_run.append(f"- the reward-model selectors: {t['reward_models_missing']}")
    lines += ["", "## Selectors that did not run", "", *(not_run or ["None."])]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------------------------


def plan_lines(cfg: dict, plan: dict, inputs: dict, runs: list[dict],
               skipped: dict[str, dict[str, str]]) -> list[str]:
    """The dry-run plan: the tasks, the pool and labels, each selector with its calls (or why it does not run), the
    gates, the tables and E4's number. Reads no cost file and builds no client or model."""
    e2, ids, shown = plan["e2"], inputs["ids"], inputs["shown"]
    by_run = {r["run"]: r for r in runs}
    n_pool = 2 * int(e2["n_per_instruction"])
    pilot = _read_jsonl(_repo_path(cfg["freeze"]["e2_pilot_samples"]))
    pilot_tasks = sorted({r["task_id"] for r in pilot} & set(ids))
    gaps = {g: sum(inputs["gap"][i] == g for i in ids) for g in dict.fromkeys(inputs["gap"].values())}

    def calls(r: dict) -> str:
        return (f"{r['n_tasks']} tasks x {n_pool} = {r['calls']} first-attempt calls ({r['on_disk']} on disk, "
                f"{r['todo']} to make)")

    pol, lab3, labf = by_run["policy"], by_run["labels session3"], by_run["labels full_evidence"]
    lines = [f"e2: {len(ids)} tasks (picks.e2_tasks of {cfg['freeze']['tasks_out']}, pick order; per gap type "
             f"{gaps}): {ids}",
             f"e2: {len(shown)} shown-eligible tasks (picks.e2_shown_eligible), the mitigated selectors' tasks: "
             f"{shown}",
             f"e2: pool: {pol['model']} ({pol['provider']}, evidence packet) writes {e2['n_per_instruction']} samples "
             f"under A and {e2['n_per_instruction']} under B per task: {calls(pol)}; the E2 pilot's "
             f"{len(pilot_tasks)} tasks ({len(pilot_tasks) * n_pool} samples) are expected from the disk cache, with "
             f"their labels and the scores of the selectors the pilot ran",
             f"e2: labels: {lab3['model']} ({lab3['provider']}), session-3 labeller {calls(lab3)}; full-evidence "
             f"labeller {calls(labf)}"]
    what = {"plain": "plain: hidden pointwise prompt",
            "mitigated": (f"mitigated: verification-first scorer, shown context as the prompt-cached prefix, "
                          f"max_tokens {plan['e3_max_tokens']}"),
            "reward_model": "reward model: hidden context (issue text + message), local GPU if check_fits passes at "
                            "run time (not checked in a dry run)"}
    for sel in plan["selectors"]:
        r = by_run[sel["name"]]
        if sel["name"] in skipped:
            lines.append(f"e2: selector {sel['name']} ({sel['kind']}, {sel['provider']}): SKIPPED: "
                         f"{skipped[sel['name']]['detail']}")
        else:
            lines.append(f"e2: selector {sel['name']} ({what[sel['kind']]}; {sel['provider']}, {sel['model']}, "
                         f"{sel['tasks']}): {calls(r)}")
    if plan["reward_models_missing"]:
        lines.append(f"e2: no reward-model selector: {plan['reward_models_missing']} (recorded in the table and in "
                     "stages_status.json)")
    lines.append(f"e2: exploratory selectors: {[s['name'] for s in plan['selectors'] if is_exploratory(s)]} and E4. "
                 f"{EXPLORATORY_NOTE}")
    counted = [r for r in runs if r["run"] not in skipped]
    per_provider = {p: sum(r["todo"] for r in counted if r["provider"] == p)
                    for p in dict.fromkeys(r["provider"] for r in counted)}
    budget = cfg["budget"]
    h4 = next((s["name"] for s in plan["selectors"] if s["kind"] == "mitigated" and not s["optional"]
               and s["judge"] == e2["reference_selector"]), None)
    lines += [f"e2: calls to make per provider (first attempts; a retry adds one): {per_provider}",
              f"e2: gates before each run with calls to make: Anthropic stop-launching line "
              f"{budget['anthropic_stop_launching_usd']} (pilot2.gate), OpenRouter stop line "
              f"{budget['openrouter_stop_launching_usd']} (cost file of the run directory)",
              f"e2: tables: best of N {list(e2_selection.NS)} per labeller on clean_overclaim, overclaim and "
              f"claims_tests_passed with {e2['n_tie_seeds']} tie-break seeds and {e2['n_bootstrap']} bootstrap "
              f"resamples, on the {len(ids)} tasks (plain and reward-model selectors) and the {len(shown)} "
              f"shown-eligible tasks (every selector); H4 second condition: {h4} minus {e2['reference_selector']} at "
              f"N = {COMPARISON_N}; each reward model minus {e2['reference_selector']} -> {TABLE_JSON} and {TABLE_MD}",
              "e2: E4: " + e4_note(inputs["resolved"], ids, shown)["sentence"]]
    return lines


def stage_e2(ctx: Ctx6) -> int:
    """Step 11. e2_config (cfg["e2"], configs/models.yaml), e2_inputs (the frozen files), planned_skips (the
    projection) and run_plan. Dry run: plan_lines to ctx.plan, no client, no cost file, nothing written or recorded.
    Live: no reward-model list (e2_config's reward_models_missing) is logged as a warning and recorded; the pool's runs
    that still have calls to make pass provider_gate (else the stage is skipped, recorded) and need a credential (else
    skipped); prep5_e2.run_pool (no selector) writes the samples and both labels; run_selectors scores (a selector
    that did not run, a failed reward model among them, is named and the stage goes on); e2_table, then e2_table.json
    (refused, nothing written, when one exists that differs: the stage fails naming the first difference) and
    e2_table.md; stages_status.json["e2"] with selectors_run, selectors_skipped (each {reason, detail}),
    selectors_failed and reward_models_missing. EXIT_FAILED (nothing recorded) when the config fails e2_config;
    EXIT_FAILED, recorded, when an input fails; a BudgetExceeded stops the stage (pilot2._budget_stop); any other
    exception fails it (prep5._stage_failed)."""
    from overclaim.experiments import full_run

    cfg = ctx.cfg
    try:
        plan = e2_config(cfg, models_config(cfg))
    except ValueError as exc:
        logger.error("e2: %s", exc)
        return EXIT_FAILED
    try:
        inputs = e2_inputs(cfg, int(plan["e2"]["n_tasks"]))
        skipped = planned_skips(plan)
    except (ValueError, OSError, KeyError) as exc:
        logger.error("e2: FAILED: %s: %s", type(exc).__name__, exc)
        if not ctx.dry_run:
            _record_stage(ctx, STAGE, {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
        return EXIT_FAILED
    runs = run_plan(ctx.run_dir, plan, inputs)
    if ctx.dry_run:
        for line in plan_lines(cfg, plan, inputs, runs, skipped):
            ctx.note(line)
        return EXIT_OK
    if plan["reward_models_missing"]:
        logger.warning("e2: no reward-model selector: %s (recorded in the table and stages_status.json)",
                       plan["reward_models_missing"])
    for r in runs[:3]:
        reason = provider_gate(ctx, r["provider"]) if r["todo"] else None
        if reason is not None:
            return _skip(ctx, STAGE, f"{r['run']}: {reason}")
    need = {"policy": runs[0]["todo"] > 0, "labeller": runs[1]["todo"] + runs[2]["todo"] > 0}
    clients = {k: ctx.client(plan[k]) if need[k] else None for k in need}   # a run with nothing to make needs none
    missing = [k for k in need if need[k] and clients[k] is None]
    if missing:
        return _skip(ctx, STAGE, f"no credential for the clients {missing}")
    from overclaim.graders.judge import BudgetExceeded

    e2, n_per = plan["e2"], int(plan["e2"]["n_per_instruction"])
    files = {k: ctx.run_dir / v for k, v in FILES.items()}
    try:
        pool = prep5_e2.run_pool(inputs["tasks"], inputs["evidence"], clients, files,
                                 instructions=prep5_e2.instruction_plan("B", n_per), selectors=[], structured={},
                                 workers={k: int(cfg["max_workers"][k]) for k in ("openrouter", "anthropic")})
        scores, ran, failed = run_selectors(ctx, plan, inputs, pool["samples"], skipped,
                                            {r["run"]: r["todo"] for r in runs[3:]})
        table = analysis._nan_to_none(e2_table(plan, inputs, pool, scores, ran, skipped,
                                               n_tie_seeds=int(e2["n_tie_seeds"]), n_boot=int(e2["n_bootstrap"]),
                                               seed=int(cfg["seed"])))
    except BudgetExceeded as exc:
        _record_stage(ctx, STAGE, {"status": "stopped", "reason": f"budget cap: {exc}"})
        return pilot2._budget_stop(ctx, STAGE, exc)
    except Exception as exc:  # a stale run file, a prompt cache not read, a missing task: recorded, no table written
        return prep5._stage_failed(ctx, STAGE, exc)
    table_path = ctx.run_dir / TABLE_JSON
    table = json.loads(json.dumps(table))
    if table_path.exists():
        diff = full_run.first_difference(_read_json(table_path, {}), table)
        if diff:
            reason = (f"{table_path} exists and differs from the table computed now ({diff}); nothing is written. "
                      "To write the table again, move that file away first")
            logger.error("e2: REFUSED: %s", reason)
            previous = _read_json(ctx.run_dir / "stages_status.json", {}).get(STAGE) or {}
            _record_stage(ctx, STAGE, {**previous, "refused": reason} if previous.get("status") == "done"
                          else {"status": "failed", "reason": reason})
            return EXIT_FAILED
        logger.info("e2: %s is unchanged", table_path)
    full_run._write_text(table_path, json.dumps(table, indent=2, ensure_ascii=False) + "\n")
    full_run._write_text(ctx.run_dir / TABLE_MD, table_markdown(table))
    h4 = table["h4_second_condition"]
    _record_stage(ctx, STAGE, {"status": "done", "n_tasks": len(inputs["ids"]),
                               "n_shown_eligible": len(inputs["shown"]),
                               "selectors_run": ran, "selectors_skipped": skipped, "selectors_failed": failed,
                               "reward_models_missing": plan["reward_models_missing"],
                               "h4_second_condition": None if h4 is None else
                               {k: h4[k] for k in ("treated", "control", "holds", "labellers_agree")},
                               "table": str(table_path)})
    logger.info("e2: %d selectors ran, %d did not run (%d of them failed); H4 second condition: %s", len(ran),
                len(skipped), len(failed), None if h4 is None else h4["holds"])
    print(f"[e2] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK

"""Experiment 1b: the pointwise reading of Experiment 1. Every grader scores O, H and P one at a time from 1 to 10.

Each variant is scored alone with graders.pointwise.score_message: the context is judge.build_context(task,
condition) and the message is matched_set["variants"][variant]; no other variant is ever in the prompt. The calls of
one task run back to back (so with prompt caching the first call writes the cache and the next ones read it); tasks
run in parallel threads. One JSON line per (task, variant) is appended to out_path, and a rerun skips the records
already there (resume). `score_task` is the per-task step, also usable as judge.run_pairs2's `after_task` hook so
that a task's pointwise calls follow its six pairwise calls in the same thread.

Session 5, Part 2: `score_task` and `run_pointwise` take the rubric name, question and instruction of
graders.pointwise.score_message (defaults: today's neutral scorer, prompt and cache key), so the full-rubric and the
instruction-only scorers are the same code with pointwise.question_for(rubric); a record's `rubric` is part of its
resume key.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from overclaim.graders import judge
from overclaim.graders.pointwise import POINTWISE_QUESTION, RUBRIC, check_question, score_message

logger = logging.getLogger(__name__)

VARIANTS = ("O", "H", "P")


def _key(r: dict) -> tuple:
    """Resume key of a pointwise record: (task_id, variant, grader, set, condition, replicate)."""
    return r["task_id"], r["variant"], r["grader"], r["set"], r["condition"], r["replicate"]


def _rubric_key(r: dict) -> tuple:
    """Resume key of score_task: _key(r) + (r.get("rubric"),), i.e. (task_id, variant, grader, set, condition,
    replicate, rubric), so records of two rubrics in one file do not collide. Every record file written before session
    5, Part 2 holds one rubric (results/pilot4_coding, pilot_research and prep5: pointwise-neutral, or
    pointwise-verify-first in e3pw_calls.jsonl), so resume on them is unchanged. _key itself stays the 6-tuple, because
    e3_verification_pointwise.score_task_vf matches its own 6-tuple lookup key against _key."""
    return (*_key(r), r.get("rubric"))


def score_task(task: dict, matched_set: dict, client: Any, grader: str, condition: str, out_path: str | Path,
               run_name: str, *, set_name: str, structured: bool, use_prompt_cache: bool, replicate: int = 0,
               live_before: int = 0, variants: tuple[str, ...] = VARIANTS, rubric: str = RUBRIC,
               question: str = POINTWISE_QUESTION, instruction: str = "") -> list[dict]:
    """Score the variants of one task in order, one call sequence at a time, appending one record per new variant.

    A (task_id, variant, grader, set, condition, replicate, rubric) already in out_path (_rubric_key) is not scored
    again; its record is returned as written. Every record file written before session 5, Part 2 holds one rubric, so
    resume on those files is what it was with the 6-tuple key. The file is read under judge's append lock, so a record
    another thread is appending is never read half-written. The context (judge.build_context(task, condition)) is
    built only when a variant needs a call. Each message is scored by score_message with rubric, question and
    instruction (the defaults give session 4's prompt and cache key; pointwise.question_for(rubric) gives the pair of a
    rubric name). A known rubric name with another question or instruction raises ValueError
    (pointwise.check_question) before the file is read, so no record is resumed or written under the wrong name.
    With use_prompt_cache=True the context is sent as the cached prefix, and every live (not disk-cached) attempt
    counts: `live_before` is the number of live calls already made for this task's prefix (run_pairs2 passes its
    n_live_calls), and from the second live call on (live_before included) an attempt that read 0 tokens from the
    prompt cache raises judge.CacheNotRead before its record is written. With live_before 0 the first pointwise
    call writes the cache.
    Record: run, grader, set, rubric (the `rubric` argument), condition, replicate, task_id, gap_type, variant, score
    (None when invalid), reason, raw_text, parse_attempts, model, usage, cost_usd, cost_parts, cached, stop_reason,
    upstream_provider, route, ts. Returns the task's records in `variants` order.
    """
    if condition not in judge.CONDITIONS:
        raise ValueError(f"unknown condition {condition!r} (expected one of {judge.CONDITIONS})")
    check_question(rubric, question, instruction)
    with judge._append_lock:
        existing = {_rubric_key(r): r for r in judge._read_jsonl(out_path)}
    context: str | None = None
    live = live_before
    out: list[dict] = []
    for variant in variants:
        key = (task["task_id"], variant, grader, set_name, condition, replicate, rubric)
        if key in existing:
            out.append(existing[key])
            continue
        if context is None:
            context = judge.build_context(task, condition)
        v = score_message(client, context, matched_set["variants"][variant], structured=structured,
                          cache_prefix=use_prompt_cache, replicate=replicate, rubric=rubric, question=question,
                          instruction=instruction)
        for reads in v.pop("live_cache_reads"):
            live += 1
            if use_prompt_cache and live >= 2 and reads == 0:
                raise judge.CacheNotRead(
                    f"{grader}/{set_name}/{rubric}/{condition}/replicate {replicate}: call {live} of task "
                    f"{task['task_id']} (variant {variant}) read 0 tokens from the prompt cache"
                )
        rec = {
            "run": run_name,
            "grader": grader,
            "set": set_name,
            "rubric": rubric,
            "condition": condition,
            "replicate": replicate,
            "task_id": task["task_id"],
            "gap_type": task.get("gap_type"),
            "variant": variant,
            **v,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        judge._append_jsonl(out_path, rec)
        out.append(rec)
    return out


def run_pointwise(tasks: list[dict], matched_sets: list[dict], client: Any, grader: str, condition: str,
                  out_path: str | Path, run_name: str, *, set_name: str, structured: bool, max_workers: int = 1,
                  use_prompt_cache: bool = False, replicate: int = 0, variants: tuple[str, ...] = VARIANTS,
                  rubric: str = RUBRIC, question: str = POINTWISE_QUESTION, instruction: str = "") -> list[dict]:
    """Score every task that has a matched set (score_task per task, live_before 0, with this rubric, question and
    instruction) and return the records.

    A matched set given twice, or a matched set without a task, raises ValueError (as judge.run_pairs2); tasks
    without a matched set are not scored and their number is logged. Tasks run in `max_workers` threads; the calls
    of one task stay sequential. Returns the records of this run (previously written and new) in task order, then
    `variants` order.
    """
    if condition not in judge.CONDITIONS:
        raise ValueError(f"unknown condition {condition!r} (expected one of {judge.CONDITIONS})")
    sets: dict[str, dict] = {}
    for ms in matched_sets:
        if ms["task_id"] in sets:
            raise ValueError(f"duplicate matched set for task {ms['task_id']}")
        sets[ms["task_id"]] = ms
    orphans = sorted(set(sets) - {t["task_id"] for t in tasks})
    if orphans:
        raise ValueError(f"{len(orphans)} matched sets have no task, e.g. {orphans[:3]}")
    todo = [t for t in tasks if t["task_id"] in sets]
    if len(todo) < len(tasks):
        logger.info("run_pointwise: %d tasks have no matched set and were not scored", len(tasks) - len(todo))

    def one(task: dict) -> list[dict]:
        return score_task(task, sets[task["task_id"]], client, grader, condition, out_path, run_name,
                          set_name=set_name, structured=structured, use_prompt_cache=use_prompt_cache,
                          replicate=replicate, variants=variants, rubric=rubric, question=question,
                          instruction=instruction)

    if max_workers <= 1 or len(todo) <= 1:
        per_task = [one(t) for t in todo]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(one, t) for t in todo]
            try:
                per_task = [f.result() for f in futures]
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
    records = [r for recs in per_task for r in recs]
    logger.info("run_pointwise %s/%s/%s/%s/replicate %d: %d records, %d tasks, %d invalid scores", grader, set_name,
                rubric, condition, replicate, len(records), len(todo), sum(r["score"] is None for r in records))
    return records

"""Session 5, Part 1 ("prep5"): the final coding sets, the reserve and the human-check file (brief Step 1), and the
dispatch of the Experiment 2 mixed-pool pilot (stage e2, overclaim.experiments.prep5_e2).

Entry point: `uv run python -m overclaim.experiments.prep5 --config configs/prep5.yaml --stage S [--dry-run]`.
Stages, in order:
- recheck (Step 1.1): the v4 pre-check again, with the final reply parser and its own cache entries (tags
  {"recheck": 1}), on the session-4 sets named in cfg["recheck"]; a task fails and leaves the sets when the fresh
  answer is invalid, discloses a gap or supports the primary span; a kept span named by span_index whose fresh evidence
  is rejected becomes unsupported and the set is rebuilt without it.
- checkd (Step 1.2): the 26 session-4 tasks dropped by check d, rebuilt with the session-4 variants config and the
  amended post-check verify_v4.verify_set_v4b; recovered sets go to the reserve.
- ntr (Step 1.3): the no_tests_run tasks session 4 dropped as no_test_claim (and any no_tests_run candidate it never
  tried), rebuilt with the relaxed primary rule (cfg["variants_config"]) and verify_set_v4b.
- assemble (Steps 1.4 to 1.6): the main sets per gap type by seed, the reserve, the task file and the final
  human-check CSV under data/processed/coding/, and results/prep5/coding_final.json and .md. Once written, the
  selection is not changed: a later run that would choose other sets writes nothing and fails.
- e2 (Step 3): overclaim.experiments.prep5_e2.stage_e2(ctx).
- e3pw (revision of 2026-11-06, section 3, coding check): on cfg["e3pw"]["n_tasks"] shown-eligible main sets, the plain
  shown scorer and then the verification-first scorer of overclaim.experiments.e3_verification_pointwise, per task
  back to back, and their check table (results/prep5/e3pw_check.json and .md).

Every build goes through variants_v4.build_v4_set, one task at a time in a thread pool of max_workers.anthropic, with
session 4's clients (so its detection, annotation, pre-check and rewrite calls are disk-cache hits). Every stage that
spends Anthropic money is not launched once the Anthropic total of results/prep5/cost.jsonl has reached
budget.anthropic_stop_launching_usd (pilot2.gate); the skip is recorded in results/prep5/stages_status.json, and a stage
recorded as done stays done (pilot3._skip). Stages are idempotent (every call is cached on disk; outputs are rewritten
from the same calls), support --dry-run (no call; the plan says what each stage would do) and delete nothing. Session
4's run directory and data files are only read. Trajectory content is data: it is passed to the models as text and
written out, never executed. Exit codes as pilot 2: 0 ok; 1 a stage failed or an input is missing; 2 a required
credential is missing; 3 a budget cap was hit.
"""

from __future__ import annotations

import argparse
import copy
import csv
import functools
import importlib
import inspect
import json
import logging
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from overclaim.experiments import pilot2, pilot3, pilot4
from overclaim.experiments.pilot2 import (EXIT_FAILED, EXIT_NO_CREDENTIALS, EXIT_OK, _read_json, _read_jsonl,
                                          _repo_path, _write_json, _write_jsonl)
from overclaim.experiments.pilot3 import _record_stage, _skip, seeded_order

logger = logging.getLogger(__name__)

STAGES = ("recheck", "checkd", "ntr", "assemble", "e2", "e3pw")
GAP_TYPES = pilot3.GAP_TYPES
CODING_MODEL_KEYS = ("editor", "annotator", "precheck_v4", "verifier_v4")   # must be among cfg["model_entries"]
ORIGINS = ("session4", "session4_rechecked", "ntr_relaxed", "checkd_recovered")
D_DROP = "dropped:verifier:d_no_outcome_claims"
NTR_DROP = "dropped:no_test_claim"
RECHECK_TAGS = {"recheck": 1}
# results/prep5/
RECHECK_FILE, RECHECK_SETS = "coding_recheck.json", "sets_recheck.jsonl"
CHECKD_FILE, CHECKD_SETS = "coding_checkd.jsonl", "sets_checkd.jsonl"
CHECKD_UNEXPLAINED_SETS = "sets_checkd_unexplained.jsonl"   # built under v4b, session-4 reason not explained
NTR_BUILD, NTR_SETS = "coding_ntr_build.jsonl", "sets_ntr.jsonl"
FINAL_JSON, FINAL_MD = "coding_final.json", "coding_final.md"
# Stage e3pw: the verification-first scorer's records, the check table (the plain scorer's records go to
# pilot4.POINTWISE_FILE, pointwise_calls.jsonl)
E3PW_CALLS, E3PW_JSON, E3PW_MD = "e3pw_calls.jsonl", "e3pw_check.json", "e3pw_check.md"
ASSEMBLE_INPUTS = (RECHECK_FILE, RECHECK_SETS, CHECKD_FILE, CHECKD_SETS, NTR_BUILD, NTR_SETS)
# data/processed/coding/ (new files; session 4's files are never written)
MAIN_FILE, RESERVE_FILE = "matched_sets_v4_main.jsonl", "reserve_sets_v4.jsonl"
ADDED_TASKS_FILE, HUMAN_CHECK_FILE = "pilot_tasks_v4_final_added.jsonl", "human_check_v4_final.csv"


class ReconstructionError(RuntimeError):
    """The recheck's reconstructed pre-check input did not hit session 4's cache entry."""


# ---------------------------------------------------------------------------------------------
# Config, context and session 4's files
# ---------------------------------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    """configs/prep5.yaml with what prep5 needs from configs/models.yaml: every entry named in cfg["model_entries"]
    under its own name (CODING_MODEL_KEYS must be among them), the judges and the prices. cfg["data"] is
    {"processed_dir": the session-4 config's data.processed_dir} (pilot2.Ctx reads it; the final files are written
    there). Validates the session4 block (its config and variants config exist), shown, main (targets for exactly
    GAP_TYPES, summing to total), ntr.stop_usd and the recheck entries (task_id; span_index, a positive integer, only
    together with a non-empty rejected_evidence list). ValueError on a failed check."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(_repo_path(cfg["models_config"]), encoding="utf-8") as f:
        models = yaml.safe_load(f)
    entries = list(cfg.get("model_entries") or [])
    missing = [k for k in CODING_MODEL_KEYS if k not in entries]
    if missing:
        raise ValueError(f"model_entries must name {list(CODING_MODEL_KEYS)}; missing {missing}")
    for key in entries:
        if key not in models:
            raise ValueError(f"{key!r} is not an entry of {cfg['models_config']}")
        cfg[key] = models[key]
    cfg["judges"] = models["judges"]
    cfg["prices_usd_per_mtok"] = models["prices_usd_per_mtok"]
    s4 = cfg["session4"]
    for key in ("config", "run_dir", "variants_config"):
        if not _repo_path(s4[key]).exists():
            raise ValueError(f"session4.{key}: {s4[key]} does not exist")
    with open(_repo_path(s4["config"]), encoding="utf-8") as f:
        cfg["data"] = {"processed_dir": yaml.safe_load(f)["data"]["processed_dir"]}
    if not _repo_path(cfg["variants_config"]).exists():
        raise ValueError(f"variants_config: {cfg['variants_config']} does not exist")
    shown = cfg["shown"]
    if not 0 < int(shown["max_transcript_tokens"]) < int(shown["upper_band_tokens"]):
        raise ValueError("shown: need 0 < max_transcript_tokens < upper_band_tokens")
    main = cfg["main"]
    if set(main["targets"]) != set(GAP_TYPES) or sum(int(n) for n in main["targets"].values()) != int(main["total"]):
        raise ValueError(f"main.targets must give each of {list(GAP_TYPES)} a target, summing to main.total")
    if float(cfg["ntr"]["stop_usd"]) <= 0:
        raise ValueError("ntr.stop_usd must be positive")
    ids = [e["task_id"] for e in cfg["recheck"]]
    if len(ids) != len(set(ids)):
        raise ValueError("recheck: a task_id appears twice")
    for e in cfg["recheck"]:
        index, rejected = e.get("span_index"), e.get("rejected_evidence")
        if (index is None) != (rejected is None) or (index is not None and (
                isinstance(index, bool) or not isinstance(index, int) or index < 1
                or not isinstance(rejected, list) or not rejected
                or not all(isinstance(r, str) and r for r in rejected))):
            raise ValueError(f"recheck {e['task_id']}: span_index (a positive integer) and rejected_evidence (a "
                             "non-empty list of strings) go together")
    return cfg


class Ctx5(pilot4.Ctx4):
    """pilot4.Ctx4 on the prep5 config: run_dir results/prep5, cost file run_dir/cost.jsonl with the config's budget
    caps, clients from the prep5 config. `session4` is session 4's context, for reading only."""

    @functools.cached_property
    def session4(self) -> pilot4.Ctx4:
        """pilot4.Ctx4 on the raw session-4 config (cfg["session4"]["config"]) and its run directory, dry_run True. Used
        only for first_tasks(), set_path() and pilot4.candidate_orders(); its client and cost logger are never used
        (they would append to session 4's cost file)."""
        path = _repo_path(self.cfg["session4"]["config"])
        with open(path, encoding="utf-8") as f:
            return pilot4.Ctx4(yaml.safe_load(f), path, _repo_path(self.cfg["session4"]["run_dir"]), True)

    def s4_sets(self) -> list[dict]:
        """Session 4's v4 sets (matched_sets_v4.jsonl), in file order."""
        return _read_jsonl(self.session4.set_path("v4"))

    def s4_infos(self) -> list[dict]:
        """Session 4's build infos (v4_build.jsonl in its run directory): one per task tried, in build order."""
        return _read_jsonl(self.session4.run_dir / "v4_build.jsonl")

    def s4_task_files(self) -> list[Path]:
        """The four session-4 task files: pilot.first_task_files, then pilot.added_tasks_file."""
        pilot = self.session4.cfg["pilot"]
        return [self.session4.processed / name for name in [*pilot["first_task_files"], pilot["added_tasks_file"]]]

    def s4_known_tasks(self) -> dict[str, dict]:
        """{task_id: task} of session 4's first tasks (Ctx4.first_tasks) and of its added tasks file."""
        added = _read_jsonl(self.session4.processed / self.session4.cfg["pilot"]["added_tasks_file"])
        return {t["task_id"]: t for t in self.session4.first_tasks() + added}

    def s4_orders(self) -> dict[str, list[tuple[Path, dict]]]:
        """pilot4.candidate_orders of session 4 (its pools, seed, held-out ids and first tasks), per gap type."""
        s4 = self.session4
        return pilot4.candidate_orders(s4, s4.first_tasks())[0]


def coding_purposes() -> frozenset[str]:
    """The cost.jsonl purposes of the coding stages' calls: pilot4.sets_purposes() with the v4 post-check's purpose
    replaced by verify_v4.verify_set_v4b's (read from the `purpose` defaults)."""
    from overclaim import verify_v4

    def default(fn: Callable) -> str:
        return inspect.signature(fn).parameters["purpose"].default

    return (pilot4.sets_purposes() - {default(verify_v4.verify_set_v4)}) | {default(verify_v4.verify_set_v4b)}


def load_tasks(ctx: Ctx5, ids: list[str], orders: dict[str, list[tuple[Path, dict]]] | None = None
               ) -> dict[str, tuple[dict, dict | None]]:
    """{task_id: (task, pool)} for ids: a task of the session-4 first tasks or added tasks file has pool None; any
    other is read from its candidate pool by offset (coding.read_task_at; the pool file is never loaded whole), at the
    position given by `orders` (ctx.s4_orders() when None and needed), pool = {"file" (name in session 4's processed
    directory), "offset", "length"}. ValueError naming the ids found nowhere."""
    from overclaim.data import coding

    known = ctx.s4_known_tasks()
    out = {i: (known[i], None) for i in ids if i in known}
    rest = [i for i in ids if i not in out]
    if rest:
        orders = ctx.s4_orders() if orders is None else orders
        where = {c["task_id"]: (path, c) for entries in orders.values() for path, c in entries}
        missing = [i for i in rest if i not in where]
        if missing:
            raise ValueError(f"{len(missing)} tasks are in no session-4 task file and no candidate pool: {missing[:3]}")
        for i in rest:
            path, c = where[i]
            out[i] = (coding.read_task_at(path, c["offset"], c["length"]),
                      {"file": path.name, "offset": c["offset"], "length": c["length"]})
    return out


def read_pool_task(ctx: Ctx5, pool: dict) -> dict:
    """The task a build info's "pool" entry locates (session 4's processed directory)."""
    from overclaim.data import coding

    return coding.read_task_at(ctx.session4.processed / pool["file"], pool["offset"], pool["length"])


# ---------------------------------------------------------------------------------------------
# Building one task, and many in a thread pool
# ---------------------------------------------------------------------------------------------


def coding_clients(ctx: Ctx5) -> dict[str, Any] | None:
    """{key: ctx.client(cfg[key])} for CODING_MODEL_KEYS, or None when a credential is missing."""
    clients = {key: ctx.client(ctx.cfg[key]) for key in CODING_MODEL_KEYS}
    return None if any(c is None for c in clients.values()) else clients


def build_one(task: dict, source: str, vcfg: dict, bank: dict, lint_patterns: list[str], clients: dict[str, Any],
              precheck_fn: Callable[[dict, str, list[dict], str], dict], verify_fn: Callable[[dict, dict], dict],
              shown_cap: int, pool: dict | None = None) -> tuple[dict | None, str, dict]:
    """One task as variants_v4.build_all_v4 builds it: variants_inplace.detect_spans with the editor (skipped when
    variants_v4.step1_v4 drops the task), then variants_v4.build_v4_set (annotator, precheck_fn, verify_fn, rewriter
    the editor, precheck_without_test_claim for source "first" as build_all_v4's default). The info gets source,
    detection and pool. A built set gets verifier.prompt_version and verifier.excluded_quotes from the post-check
    result ("v4" and None when the result has none) and shown_eligible (token_counts.transcript <= shown_cap)."""
    from overclaim import variants_v4
    from overclaim.variants_inplace import detect_spans

    spans, meta = None, None
    if variants_v4.step1_v4(task, vcfg)[1] is not None:
        spans, meta = detect_spans(task["final_message_raw"], clients["editor"])
    matched_set, outcome, info = variants_v4.build_v4_set(
        task, spans, vcfg, bank, lint_patterns, clients["annotator"], precheck_fn, verify_fn, clients["editor"],
        precheck_without_test_claim=source == "first")
    info["source"], info["detection"], info["pool"] = source, meta, pool
    if matched_set is not None:
        post = info["verifier"]["post"]
        matched_set["verifier"]["prompt_version"] = post.get("prompt_version", "v4")
        matched_set["verifier"]["excluded_quotes"] = post.get("excluded_quotes")
        matched_set["shown_eligible"] = task["token_counts"]["transcript"] <= shown_cap
    return matched_set, outcome, info


def run_builds(items: list[Any], build: Callable[[Any], Any], max_workers: int,
               should_stop: Callable[[], bool] | None = None) -> tuple[list[Any], BaseException | None]:
    """build(item) for every item in a thread pool of max_workers; results in item order. An item is not started
    (its result is None) when, at its turn, should_stop() is True or an earlier build has raised. Returns (results,
    the first exception a build raised in item order, or None); the work of the other builds is kept."""
    halt = threading.Event()

    def one(item: Any) -> tuple[Any, BaseException | None]:
        if halt.is_set() or (should_stop is not None and should_stop()):
            return None, None
        try:
            return build(item), None
        except Exception as exc:  # returned to the caller, which decides
            halt.set()
            return None, exc

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        out = list(executor.map(one, items))
    errors = [e for _, e in out if e is not None]
    return [r for r, _ in out], errors[0] if errors else None


def uncached_steps(info: dict) -> list[str]:
    """The steps of a build whose model call ran and was not served from the disk cache, in build order: detection,
    annotation (the last call of each), precheck, rewrite (any attempt) and postcheck (every attempt)."""
    out = []
    for step, meta in (("detection", info.get("detection")), ("annotation", info.get("annotation"))):
        if meta is not None and meta.get("cached") is False:
            out.append(step)
    pre, post, rewrite = info["verifier"]["pre"], info["verifier"]["post"], info.get("rewrite")
    if pre is not None and not pre.get("cached"):
        out.append("precheck")
    if rewrite is not None and not all(rewrite.get("cached") or []):
        out.append("rewrite")
    if post is not None and not post.get("cached"):
        out.append("postcheck")
    return out


def _stage_inputs(ctx: Ctx5, stage: str, variants_config: str) -> tuple[dict, tuple, Path] | int:
    """The common start of a paid build stage: the gate (a skip returns pilot3._skip's code), the clients (missing:
    EXIT_NO_CREDENTIALS) and variants_v4.load_v4_config(variants_config). Returns (clients, (vcfg, bank,
    lint_patterns), cost path) or an exit code."""
    reason = pilot2.gate(ctx, {}, "anthropic")
    if reason is not None:
        return _skip(ctx, stage, reason)
    clients = coding_clients(ctx)
    if clients is None:
        logger.error("%s: ANTHROPIC_API_KEY is not set", stage)
        return EXIT_NO_CREDENTIALS
    from overclaim import variants_v4

    return clients, variants_v4.load_v4_config(_repo_path(variants_config)), ctx.run_dir / "cost.jsonl"


def _verify_v4b(verifier: Any) -> Callable[[dict, dict], dict]:
    from overclaim import verify_v4

    def verify_fn(task: dict, set_dict: dict) -> dict:
        return verify_v4.verify_set_v4b(task, set_dict, verifier)

    return verify_fn


def _precheck(prechecker: Any) -> Callable[[dict, str, list[dict], str], dict]:
    from overclaim import verify_v4

    def precheck_fn(task: dict, message: str, spans: list[dict], evidence_text: str) -> dict:
        return verify_v4.precheck_original(task, message, spans, evidence_text, prechecker)

    return precheck_fn


def _spend_total(ctx: Ctx5, stage: str, this_run: float) -> float:
    """The stage's Anthropic spend over every invocation: the spend_usd its entry of stages_status.json already holds
    plus this run's. A rerun from the disk cache adds 0, so it no longer overwrites what the first run paid."""
    previous = (_read_json(ctx.run_dir / "stages_status.json", {}).get(stage) or {}).get("spend_usd") or 0.0
    return round(float(previous) + this_run, 6)


def _stage_failed(ctx: Ctx5, stage: str, exc: BaseException) -> int:
    """Log exc with its traceback, record the stage as failed (reason: the exception's type and message) and return
    EXIT_FAILED. The calls made so far stay cached; the caller writes none of the stage's outputs."""
    logger.error("%s: FAILED: %s: %s", stage, type(exc).__name__, exc, exc_info=exc)
    _record_stage(ctx, stage, {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    return EXIT_FAILED


def _set_funnel(ctx: Ctx5, key: str, value: dict) -> None:
    funnel = _read_json(ctx.funnel_path, {})
    funnel[key] = value
    _write_json(ctx.funnel_path, funnel)


# ---------------------------------------------------------------------------------------------
# Stage: recheck (Step 1.1)
# ---------------------------------------------------------------------------------------------


def precheck_input(s4_set: dict) -> tuple[list[dict], dict[int, int]]:
    """The span list session 4's build sent to the pre-check, and {set span index: pre-check index}. The build listed
    its spans before the pre-check's own missed claims were merged in, so the list is the set's spans whose source is
    not "precheck", in set order (sorted by start, as before the merge), numbered 1..n, each {index, text, type}."""
    listed: list[dict] = []
    index_of: dict[int, int] = {}
    for span in s4_set["spans"]:
        if span["source"] == "precheck":
            continue
        listed.append({"index": len(listed) + 1, "text": span["text"], "type": span["type"]})
        index_of[span["index"]] = len(listed)
    return listed, index_of


def cached_reply(pre: dict) -> dict:
    """What session 4's final parser tolerated in the cached pre-check reply (a build info's verifier.pre): attempts,
    invalid, cached, the answer's unlisted_spans, the number of listed entries and of missed claims, and whether every
    quote_ok is true (None without an answer). A missing result (None) gives None and 0 throughout."""
    pre = pre or {}
    answer = pre.get("answer")
    entries = (answer["spans"] + answer["missed_claims"]) if answer else []
    return {"attempts": pre.get("attempts"), "invalid": pre.get("invalid"), "cached": pre.get("cached"),
            "unlisted_spans": answer.get("unlisted_spans") if answer else None,
            "n_listed_entries": len(answer["spans"]) if answer else 0,
            "n_missed_claims": len(answer["missed_claims"]) if answer else 0,
            "all_quote_ok": all(e.get("quote_ok") for e in entries) if answer else None}


def _kept(entry: dict | None) -> bool:
    return entry is not None and entry["supported_by"] not in (None, "none")


def span_differences(listed: list[dict], set_index: dict[int, int], cached: dict, fresh: dict) -> list[dict]:
    """The listed spans whose pre-check entry differs between the cached and the fresh answer ("supported" maps of
    precheck_original, keyed by pre-check index), in list order: {"precheck_index", "set_index", "text", "kind"
    ("kept_vs_unsupported" when one keeps the span and the other does not; else "supported_by" or "evidence"),
    "cached", "fresh"}. Spans neither answer judges (other_status) are skipped."""
    out = []
    for span in listed:
        i = span["index"]
        a, b = cached.get(i), fresh.get(i)
        if a is None and b is None:
            continue
        if _kept(a) != _kept(b):
            kind = "kept_vs_unsupported"
        elif (a or {}).get("supported_by") != (b or {}).get("supported_by"):
            kind = "supported_by"
        elif (a or {}).get("evidence") != (b or {}).get("evidence"):
            kind = "evidence"
        else:
            continue
        out.append({"precheck_index": i, "set_index": set_index[i], "text": span["text"], "kind": kind,
                    "cached": a, "fresh": b})
    return out


def changed_spans(old: dict, new: dict) -> list[dict]:
    """The spans whose p_kind, p_text or h_text differ between two builds of one message, matched by (start, end) in
    the order of the old set (then spans only in the new one): {"start", "end", "text", "index" [old, new], "p_kind"
    [old, new], "p_text" [old, new], "h_text" [old, new]}; a span missing from one build has None on that side."""
    old_by = {(s["start"], s["end"]): s for s in old["spans"]}
    new_by = {(s["start"], s["end"]): s for s in new["spans"]}
    keys = list(old_by) + [k for k in new_by if k not in old_by]
    out = []
    for key in keys:
        a, b = old_by.get(key) or {}, new_by.get(key) or {}
        if a and b and all(a[f] == b[f] for f in ("p_kind", "p_text", "h_text")):
            continue
        out.append({"start": key[0], "end": key[1], "text": (a or b)["text"],
                    **{f: [a.get(f), b.get(f)] for f in ("index", "p_kind", "p_text", "h_text")}})
    return out


def recheck_task(task: dict, s4_set: dict, s4_info: dict, entry: dict, prechecker: Any,
                 rebuild: Callable[[dict, Callable[..., dict]], tuple[dict | None, str, dict]]
                 ) -> tuple[dict, dict | None]:
    """Step 1.1 for one task: (report entry, rebuilt set or None).

    1. The pre-check input of session 4's build is reconstructed (precheck_input) and proved: precheck_original
       without tags must be a disk-cache hit (ReconstructionError otherwise, before the fresh call).
    2. One fresh pre-check, same arguments, tags RECHECK_TAGS.
    3. The task fails when the fresh result is invalid (recheck_invalid), discloses a gap (recheck_discloses_gap) or
       gives the set's primary span anything but supported_by "none" (recheck_primary_supported); else it passes.
    4. With entry["span_index"] (a set span index), on a pass: that span becomes unsupported when the fresh entry is
       "none" or its evidence contains one of entry["rejected_evidence"] (case-insensitive). Then rebuild(task,
       precheck_fn) builds the set again, precheck_fn returning the cached result with that span's entry replaced by
       {"supported_by": "none", "evidence": "", "reason": "recheck: " + the fresh reason} (in "supported" and in the
       answer's spans, and "recheck_forced" added; RuntimeError if the build sends other spans than the reconstructed
       list); built -> the set is returned, else the task fails with the build's outcome as reason.
    The entry: task_id, gap_type, mode ("task" or "span"), cached_reply (session 4's info), reconstruction (n listed
    spans, primary's pre-check index, replay cached), fresh (precheck_original's result), decision {"passed",
    "reason"}, span_differences, span_check (span mode) and rebuild {outcome, detail, primary, changed_spans, info}."""
    from overclaim import verify_v4

    tid = task["task_id"]
    listed, index_of = precheck_input(s4_set)
    set_index = {v: k for k, v in index_of.items()}
    primary = s4_set["primary"]["index"]
    if primary not in index_of:
        raise ValueError(f"{tid}: the primary span {primary} is a pre-check missed claim; the recheck lists only the "
                         "spans the pre-check judged")
    message, evidence_text = s4_set["final_message_raw"], s4_set["evidence_text"]
    replay = verify_v4.precheck_original(task, message, listed, evidence_text, prechecker)
    if not replay["cached"]:
        raise ReconstructionError(f"{tid}: the reconstructed pre-check input did not hit session 4's cache entry")
    fresh = verify_v4.precheck_original(task, message, listed, evidence_text, prechecker, tags=RECHECK_TAGS)
    p = index_of[primary]
    if fresh["invalid"]:
        reason = "recheck_invalid"
    elif fresh["discloses_gap"]:
        reason = "recheck_discloses_gap"
    elif fresh["supported"][p]["supported_by"] != "none":
        reason = "recheck_primary_supported"
    else:
        reason = None
    report: dict[str, Any] = {
        "task_id": tid, "gap_type": s4_set["gap_type"], "mode": "span" if "span_index" in entry else "task",
        "cached_reply": cached_reply(s4_info["verifier"]["pre"]),
        "reconstruction": {"n_listed_spans": len(listed), "primary_set_index": primary, "primary_precheck_index": p,
                           "replay_cached": replay["cached"]},
        "fresh": fresh, "decision": {"passed": reason is None, "reason": reason},
        "span_differences": [] if fresh["invalid"] else span_differences(listed, set_index, replay["supported"],
                                                                          fresh["supported"]),
        "span_check": None, "rebuild": None}
    if "span_index" not in entry or fresh["invalid"]:
        return report, None
    k = index_of.get(entry["span_index"])
    if k not in fresh["supported"]:
        raise ValueError(f"{tid}: span {entry['span_index']} is not a span the pre-check judged")
    found = fresh["supported"][k]
    rejected = [r for r in entry["rejected_evidence"] if r.lower() in found["evidence"].lower()]
    unsupported = found["supported_by"] == "none" or bool(rejected)
    report["span_check"] = {"set_index": entry["span_index"], "precheck_index": k, "text": listed[k - 1]["text"],
                            "cached": replay["supported"][k], "fresh": found, "rejected_evidence_found": rejected,
                            "unsupported": unsupported}
    if reason is not None or not unsupported:
        return report, None
    forced = copy.deepcopy(replay)
    forced["supported"][k] = {"supported_by": "none", "evidence": "", "reason": "recheck: " + found["reason"]}
    for span_entry in forced["answer"]["spans"]:   # the set's verifier.detail.pre then agrees with its spans
        if span_entry["index"] == k:
            span_entry.update(forced["supported"][k], quote_ok=True)
    forced["recheck_forced"] = {"precheck_index": k, "set_index": entry["span_index"]}

    def precheck_fn(task_: dict, message_: str, spans: list[dict], evidence_text_: str) -> dict:
        if spans != listed or message_ != message or evidence_text_ != evidence_text:
            raise RuntimeError(f"{tid}: the rebuild's pre-check input differs from session 4's")
        return forced

    new_set, outcome, info = rebuild(task, precheck_fn)
    report["rebuild"] = {"outcome": outcome, "detail": info["detail"],
                         "primary": None if new_set is None else new_set["primary"],
                         "changed_spans": None if new_set is None else changed_spans(s4_set, new_set), "info": info}
    if new_set is None:
        report["decision"] = {"passed": False, "reason": outcome}
    return report, new_set


def _rebuild(task: dict, precheck_fn: Callable[..., dict], *, source: str, vcfg: dict, bank: dict,
             lint_patterns: list[str], clients: dict[str, Any], shown_cap: int, pool: dict | None
             ) -> tuple[dict | None, str, dict]:
    """The recheck's rebuild: build_one with the given precheck_fn and verify_set_v4b (the verifier_v4 client)."""
    return build_one(task, source, vcfg, bank, lint_patterns, clients, precheck_fn, _verify_v4b(clients["verifier_v4"]),
                     shown_cap, pool)


def stage_recheck(ctx: Ctx5) -> int:
    """Step 1.1 (recheck_task) for every entry of cfg["recheck"], with session 4's set, build info and task, the
    precheck_v4 client and, for a rebuild, build_one with the session-4 variants config and verify_set_v4b. Behind the
    stop-launching gate. Writes run_dir/coding_recheck.json ({"tasks": [entry per task], "counts"}),
    run_dir/sets_recheck.jsonl (the rebuilt sets, possibly none), funnel.json["recheck"] and the stage status. A
    BudgetExceeded stops the stage as in pilot 4; a ReconstructionError or any other exception fails it (status
    "failed" with the reason); either way none of its outputs is written."""
    cfg = ctx.cfg
    entries = cfg["recheck"]
    if ctx.dry_run:
        sets = {s["task_id"]: s for s in ctx.s4_sets()}
        infos = {i["task_id"]: i for i in ctx.s4_infos()}
        for e in entries:
            s, info = sets.get(e["task_id"]), infos.get(e["task_id"])
            if s is None or info is None:
                ctx.note(f"recheck: {e['task_id']}: session-4 set {'present' if s else 'MISSING'}, build info "
                         f"{'present' if info else 'MISSING'}")
                continue
            listed, index_of = precheck_input(s)
            ctx.note(f"recheck: {e['task_id']} ({'span ' + str(e['span_index']) if 'span_index' in e else 'task'} "
                     f"mode): {len(listed)} spans to the pre-check, primary set span {s['primary']['index']} = "
                     f"pre-check span {index_of.get(s['primary']['index'])}; cached reply "
                     f"{json.dumps(cached_reply(info['verifier']['pre']))}; would replay the pre-check without tags "
                     f"(must be a cache hit) and call {cfg['precheck_v4']['model']} once with tags {RECHECK_TAGS}"
                     + (f"; a rebuild would use {cfg['session4']['variants_config']} and verify_set_v4b"
                        if "span_index" in e else ""))
        return EXIT_OK
    start = _stage_inputs(ctx, "recheck", cfg["session4"]["variants_config"])
    if isinstance(start, int):
        return start
    clients, (vcfg, bank, lint_patterns), cost_path = start
    from overclaim.graders.judge import BudgetExceeded

    sets = {s["task_id"]: s for s in ctx.s4_sets()}
    infos = {i["task_id"]: i for i in ctx.s4_infos()}
    missing = [e["task_id"] for e in entries if e["task_id"] not in sets or e["task_id"] not in infos]
    if missing:
        logger.error("recheck: no session-4 set or build info for %s", missing)
        return EXIT_FAILED
    tasks = load_tasks(ctx, [e["task_id"] for e in entries])
    cap = int(cfg["shown"]["max_transcript_tokens"])
    spend0 = pilot4.sets_spend(cost_path, coding_purposes())
    report, rebuilt = [], []
    try:
        for e in entries:
            task, pool = tasks[e["task_id"]]
            rebuild = functools.partial(_rebuild, source=infos[e["task_id"]]["source"], vcfg=vcfg, bank=bank,
                                        lint_patterns=lint_patterns, clients=clients, shown_cap=cap, pool=pool)
            entry, new_set = recheck_task(task, sets[e["task_id"]], infos[e["task_id"]], e, clients["precheck_v4"],
                                          rebuild)
            report.append(entry)
            if new_set is not None:
                rebuilt.append(new_set)
    except BudgetExceeded as exc:
        _record_stage(ctx, "recheck", {"status": "stopped", "reason": f"budget cap: {exc}"})
        return pilot2._budget_stop(ctx, "recheck", exc)
    except ReconstructionError as exc:
        logger.error("recheck: FAILED: %s", exc)
        _record_stage(ctx, "recheck", {"status": "failed", "reason": str(exc)})
        return EXIT_FAILED
    except Exception as exc:   # e.g. recheck_task's ValueError (span_index or primary) or the rebuild's RuntimeError
        return _stage_failed(ctx, "recheck", exc)
    counts = {"tasks": len(report), "passed": sum(r["decision"]["passed"] for r in report),
              "failed": dict(Counter(r["decision"]["reason"] for r in report if not r["decision"]["passed"])),
              "rebuilt": len(rebuilt), "span_unsupported": sum(bool((r["span_check"] or {}).get("unsupported"))
                                                               for r in report)}
    _write_json(ctx.run_dir / RECHECK_FILE, {"tasks": report, "counts": counts})
    _write_jsonl(ctx.run_dir / RECHECK_SETS, rebuilt)
    _set_funnel(ctx, "recheck", counts)
    spend = pilot4.sets_spend(cost_path, coding_purposes()) - spend0
    _record_stage(ctx, "recheck", {"status": "done", **counts, "spend_usd": _spend_total(ctx, "recheck", spend)})
    logger.info("recheck: %s", counts)
    print(f"[recheck] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage: checkd (Step 1.2)
# ---------------------------------------------------------------------------------------------


def unexplained_session4(s4_info: dict, info: dict) -> list[str]:
    """What dropped the task in session 4 and is not explained by one of the three exclusions of the v4b prompt
    (decision of the lead, 2026-11-07, on the brief's "the verifier must quote the sentence and classify it"): every
    outcome quote of session 4's post-check that was not inside a kept span and that the v4b reply does not quote
    again with an excluded kind (whitespace collapsed; equal, or one contains the other), then one entry
    "rewrite <index>" per rewrite that session 4's post-check judged as stating more than the patch shows (no
    exclusion concerns rewrites). A task with any such entry was not recovered by the exclusions but by a second
    draw of the verifier."""
    from overclaim.verify_v4 import _collapse

    post4 = s4_info["verifier"]["post"] or {}
    post = info["verifier"]["post"] or {}
    kept4 = {_collapse(q) for q in (post4.get("ignored_kept_quotes") or {}).get("outcome_sentences", [])}
    excluded = [_collapse(q["quote"]) for q in post.get("excluded_quotes", [])]
    out = []
    for quote in (post4.get("answer") or {}).get("outcome_sentences", []):
        q = _collapse(quote)
        if q in kept4 or any(q and e and (q == e or q in e or e in q) for e in excluded):
            continue
        out.append(quote)
    return out + [f"rewrite {r['index']}" for r in _false_rewrites(post4.get("answer"))]


def checkd_class(outcome: str, info: dict, unexplained: list[str] | tuple[str, ...] = ()) -> str:
    """The class of a checkd outcome: recovered (built, and every session-4 reason is explained by an exclusion:
    `unexplained` is empty); passed_unexplained (built, but unexplained_session4 is not empty: the set stays out of the
    reserve); d_outcome_claim (check d with a remaining outcome_claim quote) or d_rewrite (check d by a rewrite only);
    c; f; invalid (the post-check was unusable twice); else "other:<outcome>"."""
    post = info["verifier"]["post"] or {}
    if outcome == "built":
        return "passed_unexplained" if unexplained else "recovered"
    if outcome == D_DROP:
        return "d_outcome_claim" if post.get("outcome_quotes") else "d_rewrite"
    if outcome == "dropped:verifier:c_disclosure_matches_record":
        return "c"
    if outcome == "dropped:verifier:f_no_gap_mentions":
        return "f"
    if outcome == "dropped:verifier_invalid" and info["detail"] == "post-check":
        return "invalid"
    return "other:" + outcome


def _false_rewrites(answer: dict | None) -> list[dict]:
    return [r for r in (answer or {}).get("rewrites", []) if not r["only_what_patch_shows"]]


def checkd_record(s4_info: dict, outcome: str, info: dict) -> dict:
    """One line of coding_checkd.jsonl: task_id, gap_type, source, session4 (its outcome, detail, the quotes of its
    post-check answer, its false rewrites and ignored kept quotes), v4b (the quotes with kinds as returned, the false
    rewrites, outcome_quotes, excluded_quotes, ignored_kept_quotes, invalid, failed), outcome, detail, class
    (checkd_class), uncached_steps, pool and info (the build info in full, as build_one returns it)."""
    post4 = s4_info["verifier"]["post"] or {}
    post = info["verifier"]["post"] or {}
    answer = post.get("answer")
    return {
        "task_id": s4_info["task_id"], "gap_type": s4_info["gap_type"], "source": s4_info["source"],
        "session4": {"outcome": s4_info["outcome"], "detail": s4_info["detail"],
                     "quotes": (post4.get("answer") or {}).get("outcome_sentences", []),
                     "failed_rewrites": _false_rewrites(post4.get("answer")),
                     "ignored_kept_quotes": post4.get("ignored_kept_quotes")},
        "v4b": {"quotes": (answer or {}).get("outcome_sentences", []), "failed_rewrites": _false_rewrites(answer),
                "outcome_quotes": post.get("outcome_quotes", []), "excluded_quotes": post.get("excluded_quotes", []),
                "ignored_kept_quotes": post.get("ignored_kept_quotes"), "invalid": post.get("invalid"),
                "failed": post.get("failed")},
        "outcome": outcome, "detail": info["detail"],
        "unexplained_session4": unexplained_session4(s4_info, info) if outcome == "built" else [],
        "class": checkd_class(outcome, info, unexplained_session4(s4_info, info) if outcome == "built" else ()),
        "uncached_steps": uncached_steps(info), "pool": info.get("pool"), "info": info,
    }


def checkd_counts(records: list[dict]) -> dict:
    """funnel["checkd"]: tasks, recovered, still_dropped ({d_outcome_claim, d_rewrite, c, f, invalid, other: {class:
    n}}), by_gap ({gap: {tasks, recovered}}), quote_kinds (over the remaining quotes of the v4b answers, per kind of
    verify_v4.QUOTE_KINDS, plus kept_ignored: quotes removed as inside a kept span) and uncached ({step: tasks})."""
    from overclaim.verify_v4 import QUOTE_KINDS

    classes = Counter(r["class"] for r in records)
    kinds = Counter({k: 0 for k in QUOTE_KINDS})
    kinds["outcome_claim"] += sum(len(r["v4b"]["outcome_quotes"]) for r in records)
    kinds.update(q["kind"] for r in records for q in r["v4b"]["excluded_quotes"])
    kept = sum(len((r["v4b"]["ignored_kept_quotes"] or {}).get("outcome_sentences", [])) for r in records)
    return {
        "tasks": len(records), "recovered": classes["recovered"],
        "still_dropped": {**{k: classes[k] for k in ("passed_unexplained", "d_outcome_claim", "d_rewrite", "c", "f",
                                                       "invalid")},
                          "other": {k: n for k, n in classes.items() if k.startswith("other:")}},
        "by_gap": {g: {"tasks": sum(r["gap_type"] == g for r in records),
                       "recovered": sum(r["gap_type"] == g and r["class"] == "recovered" for r in records)}
                   for g in GAP_TYPES},
        "quote_kinds": {**{k: kinds[k] for k in QUOTE_KINDS}, "kept_ignored": kept},
        "uncached": dict(Counter(step for r in records for step in r["uncached_steps"])),
    }


def stage_checkd(ctx: Ctx5) -> int:
    """Step 1.2: the session-4 tasks whose outcome is D_DROP, in v4_build.jsonl order, each through build_one with the
    session-4 variants config and verify_set_v4b (no target: every task is tried). Detection, annotation, pre-check and
    rewrites should be disk-cache hits; the infos' cached flags are counted (funnel "uncached"). Behind the
    stop-launching gate. Writes run_dir/coding_checkd.jsonl (checkd_record per task), run_dir/sets_checkd.jsonl (the
    recovered sets, reserve only: built, and every session-4 reason explained by an exclusion),
    run_dir/sets_checkd_unexplained.jsonl (built without that; in no reserve), funnel.json["checkd"] (checkd_counts)
    and the stage status. An exception in a
    build other than BudgetExceeded fails the stage (_stage_failed) and writes none of its outputs."""
    cfg = ctx.cfg
    todo = [i for i in ctx.s4_infos() if i["outcome"] == D_DROP]
    if ctx.dry_run:
        by = Counter((i["gap_type"], i["source"]) for i in todo)
        ctx.note(f"checkd: would rebuild the {len(todo)} session-4 tasks dropped as {D_DROP} ({dict(by)}) with "
                 f"{cfg['session4']['variants_config']} (detection, annotation, pre-check and rewrites from the disk "
                 f"cache) and the v4b post-check of {cfg['verifier_v4']['model']} (effort "
                 f"{cfg['verifier_v4'].get('effort')}), {cfg['max_workers']['anthropic']} at a time")
        return EXIT_OK
    start = _stage_inputs(ctx, "checkd", cfg["session4"]["variants_config"])
    if isinstance(start, int):
        return start
    clients, (vcfg, bank, lint_patterns), cost_path = start
    from overclaim.graders.judge import BudgetExceeded

    tasks = load_tasks(ctx, [i["task_id"] for i in todo])
    cap = int(cfg["shown"]["max_transcript_tokens"])
    precheck_fn, verify_fn = _precheck(clients["precheck_v4"]), _verify_v4b(clients["verifier_v4"])
    spend0 = pilot4.sets_spend(cost_path, coding_purposes())

    def build(s4_info: dict) -> tuple[dict | None, str, dict]:
        task, pool = tasks[s4_info["task_id"]]
        return build_one(task, s4_info["source"], vcfg, bank, lint_patterns, clients, precheck_fn, verify_fn, cap, pool)

    results, error = run_builds(todo, build, int(cfg["max_workers"]["anthropic"]))
    if isinstance(error, BudgetExceeded):
        _record_stage(ctx, "checkd", {"status": "stopped", "reason": f"budget cap: {error}"})
        return pilot2._budget_stop(ctx, "checkd", error)
    if error is not None:
        return _stage_failed(ctx, "checkd", error)
    records = [checkd_record(s4_info, outcome, info) for s4_info, (_, outcome, info) in zip(todo, results)]
    recovered = {r["task_id"] for r in records if r["class"] == "recovered"}
    sets = [s for s, _, _ in results if s is not None and s["task_id"] in recovered]
    unexplained = [s for s, _, _ in results if s is not None and s["task_id"] not in recovered]
    _write_jsonl(ctx.run_dir / CHECKD_FILE, records)
    _write_jsonl(ctx.run_dir / CHECKD_SETS, sets)
    _write_jsonl(ctx.run_dir / CHECKD_UNEXPLAINED_SETS, unexplained)   # built, not recovered: kept, in no reserve
    counts = checkd_counts(records)
    _set_funnel(ctx, "checkd", counts)
    spend = pilot4.sets_spend(cost_path, coding_purposes()) - spend0
    _record_stage(ctx, "checkd", {"status": "done", "tasks": counts["tasks"], "recovered": counts["recovered"],
                                  "uncached": counts["uncached"],
                                  "spend_usd": _spend_total(ctx, "checkd", spend)})
    logger.info("checkd: %d of %d recovered; %s", counts["recovered"], counts["tasks"], counts["still_dropped"])
    print(f"[checkd] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage: ntr (Step 1.3)
# ---------------------------------------------------------------------------------------------


def ntr_todo(s4_infos: list[dict], ntr_order: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    """The tasks of the ntr stage as (task_id, source), and the untried candidate ids: session 4's no_tests_run tasks
    with outcome NTR_DROP, the "first" ones in build order, then the candidates in candidate order (ntr_order, the
    no_tests_run task ids of pilot4.candidate_orders); then the candidates of ntr_order that session 4 never tried
    (source "candidate"). ValueError when a dropped candidate is not in ntr_order."""
    dropped = [i for i in s4_infos if i["gap_type"] == "no_tests_run" and i["outcome"] == NTR_DROP]
    position = {tid: n for n, tid in enumerate(ntr_order)}
    candidates = [i["task_id"] for i in dropped if i["source"] != "first"]
    lost = [tid for tid in candidates if tid not in position]
    if lost:
        raise ValueError(f"{len(lost)} dropped no_tests_run candidates are not in the candidate order: {lost[:3]}")
    tried = {i["task_id"] for i in s4_infos}
    untried = [tid for tid in ntr_order if tid not in tried]
    todo = [(i["task_id"], "first") for i in dropped if i["source"] == "first"]
    todo += [(tid, "candidate") for tid in sorted(candidates, key=position.__getitem__)]
    return todo + [(tid, "candidate") for tid in untried], untried


def ntr_funnel(todo: list[tuple[str, str]], untried: list[str], results: list[Any], stop_usd: float,
               spend: float) -> dict:
    """funnel["ntr"]: the task counts (first, dropped candidates, untried candidates and their ids), tried, the tasks
    not started because the stage's own spend reached stop_usd (n and ids), outcomes ({outcome: n}), dropped and
    excluded ({reason: n}), built, uncached ({step: tasks}), built_sets ([{task_id, source, primary_index,
    primary_rule, primary_type, shown_eligible}]), primary_types and primary_rules ({value: n} over the built sets),
    stop_usd and spend_usd."""
    infos = [r[2] for r in results if r is not None]
    outcomes = Counter(i["outcome"] for i in infos)
    built = [(s, i) for s, _, i in (r for r in results if r is not None) if s is not None]
    not_tried = [tid for (tid, _), r in zip(todo, results) if r is None]
    return {
        "tasks": len(todo), "first": sum(src == "first" for _, src in todo),
        "dropped_candidates": sum(src == "candidate" for _, src in todo) - len(untried),
        "untried_candidates": len(untried), "untried_candidate_ids": untried, "tried": len(infos),
        "not_tried_stop_usd": {"n": len(not_tried), "task_ids": not_tried},
        "outcomes": dict(outcomes),
        "dropped": dict(Counter(o.split(":", 1)[1] for o in outcomes.elements() if o.startswith("dropped:"))),
        "excluded": dict(Counter(o.split(":", 1)[1] for o in outcomes.elements() if o.startswith("excluded:"))),
        "built": len(built),
        "uncached": dict(Counter(step for i in infos for step in uncached_steps(i))),
        "built_sets": [{"task_id": s["task_id"], "source": i["source"], "primary_index": s["primary"]["index"],
                        "primary_rule": s["primary"]["rule"], "primary_type": s["primary"]["type"],
                        "shown_eligible": s["shown_eligible"]} for s, i in built],
        "primary_types": dict(Counter(s["primary"]["type"] for s, _ in built)),
        "primary_rules": dict(Counter(s["primary"]["rule"] for s, _ in built)),
        "stop_usd": stop_usd, "spend_usd": round(spend, 6),
    }


def stage_ntr(ctx: Ctx5) -> int:
    """Step 1.3: the ntr_todo tasks, each through build_one with cfg["variants_config"] (the relaxed primary rule) and
    verify_set_v4b; no target, every task is tried, except that no task starts once the Anthropic spend of
    coding_purposes() logged since this invocation started has reached ntr.stop_usd (those tasks are counted). Behind
    the stop-launching gate. Writes run_dir/coding_ntr_build.jsonl (the build infos), run_dir/sets_ntr.jsonl (the built
    sets), funnel.json["ntr"] (ntr_funnel) and the stage status. An exception in a build other than BudgetExceeded
    fails the stage (_stage_failed) and writes none of its outputs."""
    cfg = ctx.cfg
    stop_usd = float(cfg["ntr"]["stop_usd"])
    orders = ctx.s4_orders()
    todo, untried = ntr_todo(ctx.s4_infos(), [c["task_id"] for _, c in orders["no_tests_run"]])
    if ctx.dry_run:
        ctx.note(f"ntr: would build {len(todo)} no_tests_run tasks ({sum(s == 'first' for _, s in todo)} first, "
                 f"{len(todo) - sum(s == 'first' for _, s in todo) - len(untried)} dropped candidates, {len(untried)} "
                 f"candidates session 4 never tried) with {cfg['variants_config']} and the v4b post-check; no task "
                 f"starts once this stage's Anthropic spend reaches ${stop_usd:.2f}; the pre-checks of the candidates "
                 "dropped before the pre-check in session 4 are new calls")
        return EXIT_OK
    start = _stage_inputs(ctx, "ntr", cfg["variants_config"])
    if isinstance(start, int):
        return start
    clients, (vcfg, bank, lint_patterns), cost_path = start
    from overclaim.graders.judge import BudgetExceeded

    tasks = load_tasks(ctx, [tid for tid, _ in todo], orders)
    cap = int(cfg["shown"]["max_transcript_tokens"])
    precheck_fn, verify_fn = _precheck(clients["precheck_v4"]), _verify_v4b(clients["verifier_v4"])
    purposes = coding_purposes()
    spend0 = pilot4.sets_spend(cost_path, purposes)

    def should_stop() -> bool:
        return pilot4.sets_spend(cost_path, purposes) - spend0 >= stop_usd

    def build(item: tuple[str, str]) -> tuple[dict | None, str, dict]:
        task, pool = tasks[item[0]]
        return build_one(task, item[1], vcfg, bank, lint_patterns, clients, precheck_fn, verify_fn, cap, pool)

    results, error = run_builds(todo, build, int(cfg["max_workers"]["anthropic"]), should_stop)
    if isinstance(error, BudgetExceeded):
        _record_stage(ctx, "ntr", {"status": "stopped", "reason": f"budget cap: {error}"})
        return pilot2._budget_stop(ctx, "ntr", error)
    if error is not None:
        return _stage_failed(ctx, "ntr", error)
    spend = pilot4.sets_spend(cost_path, purposes) - spend0
    funnel = ntr_funnel(todo, untried, results, stop_usd, _spend_total(ctx, "ntr", spend))
    _write_jsonl(ctx.run_dir / NTR_BUILD, [r[2] for r in results if r is not None])
    _write_jsonl(ctx.run_dir / NTR_SETS, [r[0] for r in results if r is not None and r[0] is not None])
    _set_funnel(ctx, "ntr", funnel)
    _record_stage(ctx, "ntr", {"status": "done", "tasks": funnel["tasks"], "tried": funnel["tried"],
                               "built": funnel["built"], "not_tried_stop_usd": funnel["not_tried_stop_usd"]["n"],
                               "spend_usd": funnel["spend_usd"]})
    logger.info("ntr: %d of %d tried, %d built; dropped %s", funnel["tried"], funnel["tasks"], funnel["built"],
                funnel["dropped"])
    print(f"[ntr] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage: assemble (Steps 1.4 to 1.6)
# ---------------------------------------------------------------------------------------------


def main_counts(available: dict[str, int], targets: dict[str, int]) -> dict[str, int]:
    """How many main sets each gap type takes: no_tests_run min(its target, available); its shortfall is split equally
    between tests_failed and tests_narrowed, an odd remainder to pilot4.REMAINDER_GAP (tests_failed); a gap type with
    too few sets for its share passes the rest to the other. Each count is at most what is available."""
    ntr = min(targets["no_tests_run"], available["no_tests_run"])
    short = targets["no_tests_run"] - ntr
    first = pilot4.REMAINDER_GAP
    other = next(g for g in GAP_TYPES if g not in ("no_tests_run", first))
    want = {first: targets[first] + short - short // 2, other: targets[other] + short // 2}
    take = {g: min(want[g], available[g]) for g in want}
    spare = {g: want[g] - take[g] for g in want}
    take[other] = min(available[other], take[other] + spare[first])
    take[first] = min(available[first], take[first] + spare[other])
    return {"no_tests_run": ntr, **take}


def assemble_sets(s4_sets: list[dict], recheck_entries: list[dict], rechecked: list[dict], ntr_sets: list[dict],
                  checkd_sets: list[dict], targets: dict[str, int], seed: Any) -> tuple[list[dict], list[dict], dict]:
    """(main, reserve, counts) by the rules of Step 1.4. Eligible for the main sets: the session-4 sets (a task whose
    recheck decision failed is removed; a rebuilt set replaces its session-4 set: origin session4_rechecked; the rest
    session4) and the ntr sets (ntr_relaxed); the checkd sets are reserve only (checkd_recovered). Per gap type the
    main sets are the first main_counts of seeded_order(eligible ids, seed); every other set is the reserve, per gap
    type in seeded_order(reserve ids, seed) with reserve_rank its 1-based position there. Both lists are in GAP_TYPES
    order, main in seeded order and reserve by rank. Each set is a copy with role, reserve_rank (None for main),
    origin, and verifier.prompt_version ("v4" when absent). counts: available, taken, reached (the total of
    main_counts against sum(targets)) and removed (the failed rechecks). ValueError when a task id has two sets."""
    failed = {e["task_id"] for e in recheck_entries if not e["decision"]["passed"]}
    replaced = {s["task_id"]: s for s in rechecked}
    pool = [(replaced.get(s["task_id"], s), "session4_rechecked" if s["task_id"] in replaced else "session4")
            for s in s4_sets if s["task_id"] not in failed]
    pool += [(s, "ntr_relaxed") for s in ntr_sets]
    reserve_only = [(s, "checkd_recovered") for s in checkd_sets]
    ids = [s["task_id"] for s, _ in pool + reserve_only]
    dupes = sorted(i for i, n in Counter(ids).items() if n > 1)
    if dupes:
        raise ValueError(f"tasks with two sets: {dupes}")

    def tagged(s: dict, origin: str, role: str, rank: int | None) -> dict:
        out = dict(s, role=role, reserve_rank=rank, origin=origin)
        out["verifier"] = {**(s.get("verifier") or {}), "prompt_version": (s.get("verifier") or {}).get(
            "prompt_version", "v4")}
        return out

    available = {g: sum(s["gap_type"] == g for s, _ in pool) for g in GAP_TYPES}
    take = main_counts(available, targets)
    main, reserve = [], []
    for gap in GAP_TYPES:
        of_gap = {s["task_id"]: (s, o) for s, o in pool if s["gap_type"] == gap}
        chosen = seeded_order(list(of_gap), seed)[: take[gap]]
        main += [tagged(*of_gap[tid], "main", None) for tid in chosen]
        rest = {tid: so for tid, so in of_gap.items() if tid not in chosen}
        rest.update({s["task_id"]: (s, o) for s, o in reserve_only if s["gap_type"] == gap})
        reserve += [tagged(*rest[tid], "reserve", n) for n, tid in enumerate(seeded_order(list(rest), seed), start=1)]
    counts = {"available": available, "taken": take, "total": sum(take.values()), "target_total": sum(targets.values()),
              "reached": sum(take.values()) == sum(targets.values()), "removed_by_recheck": sorted(failed)}
    return main, reserve, counts


def _band(tokens: int, cap: int, upper: int) -> str:
    return f"<= {cap}" if tokens <= cap else f"({cap}, {upper}]" if tokens <= upper else f"> {upper}"


def final_report(main: list[dict], reserve: list[dict], counts: dict, funnel: dict, status: dict,
                 spend_by_purpose: dict[str, float], cap: int, upper: int) -> dict:
    """coding_final.json: the funnel of the three stages (funnel.json's recheck, checkd and ntr), the main split per gap
    type and per origin, the reserve size per gap type and per origin, the transcript-token bands (<= cap, (cap,
    upper], > upper) per gap type for the main sets and for the reserve, the primary type per gap type of the main sets,
    the Anthropic spend of the coding purposes by purpose (cost.jsonl) and per stage (stages_status.json spend_usd),
    and main_counts' counts."""
    def by_gap(sets: list[dict], key: Callable[[dict], Any]) -> dict:
        return {g: dict(Counter(key(s) for s in sets if s["gap_type"] == g)) for g in GAP_TYPES}

    def bands(sets: list[dict]) -> dict:
        keys = [f"<= {cap}", f"({cap}, {upper}]", f"> {upper}"]
        return {g: {k: sum(s["gap_type"] == g and _band(s["transcript_tokens"], cap, upper) == k for s in sets)
                    for k in keys} for g in GAP_TYPES}

    return {
        "funnel": {k: funnel.get(k) for k in ("recheck", "checkd", "ntr")},
        "main": {"n": len(main), "per_gap": {g: sum(s["gap_type"] == g for s in main) for g in GAP_TYPES},
                 "per_origin": dict(Counter(s["origin"] for s in main)),
                 "per_gap_origin": by_gap(main, lambda s: s["origin"]), "transcript_bands": bands(main),
                 "shown_eligible": {g: sum(s["gap_type"] == g and s["shown_eligible"] for s in main)
                                    for g in GAP_TYPES},
                 "primary_types": by_gap(main, lambda s: s["primary"]["type"])},
        "reserve": {"n": len(reserve), "per_gap": {g: sum(s["gap_type"] == g for s in reserve) for g in GAP_TYPES},
                    "per_origin": dict(Counter(s["origin"] for s in reserve)),
                    "per_gap_origin": by_gap(reserve, lambda s: s["origin"]), "transcript_bands": bands(reserve)},
        "selection": counts,
        "spend_usd": {"by_purpose": {k: round(v, 6) for k, v in sorted(spend_by_purpose.items())},
                      "by_stage": {k: (status.get(k) or {}).get("spend_usd") for k in ("recheck", "checkd", "ntr")}},
    }


def final_markdown(report: dict, cap: int, upper: int) -> str:
    """The short coding_final.md: the main and reserve tables per gap type (sets, origins, token bands, shown-eligible,
    primary types), the selection counts, the funnel lines of the three stages and the spend."""
    main, reserve, sel = report["main"], report["reserve"], report["selection"]
    keys = [f"<= {cap}", f"({cap}, {upper}]", f"> {upper}"]
    lines = ["# Final coding sets (session 5, Step 1)", "",
             f"Main sets: {main['n']} of {sel['target_total']}" + ("" if sel["reached"] else
                                                                     " (the target is NOT reached)")
             + f"; reserve: {reserve['n']}. Eligible for the main sets per gap type: {sel['available']}; taken: "
               f"{sel['taken']}. Removed by the recheck: {sel['removed_by_recheck'] or 'none'}.", "",
             "| gap type | main | main by origin | main tokens " + " / ".join(keys) + " | main shown-eligible | "
             "main primary types | reserve | reserve by origin | reserve tokens " + " / ".join(keys) + " |",
             "|---|---|---|---|---|---|---|---|---|"]
    for g in GAP_TYPES:
        lines.append(
            f"| {g} | {main['per_gap'][g]} | {main['per_gap_origin'][g]} | "
            + " / ".join(str(main["transcript_bands"][g][k]) for k in keys)
            + f" | {main['shown_eligible'][g]} | {main['primary_types'][g]} | {reserve['per_gap'][g]} | "
            f"{reserve['per_gap_origin'][g]} | " + " / ".join(str(reserve["transcript_bands"][g][k]) for k in keys)
            + " |")
    funnel = report["funnel"]
    lines += ["", "## Funnel", ""]
    for stage in ("recheck", "checkd", "ntr"):
        data = funnel.get(stage) or {}
        if stage == "ntr" and data:
            data = {k: v for k, v in data.items() if k not in ("built_sets", "untried_candidate_ids")}
        lines.append(f"- {stage}: {json.dumps(data, ensure_ascii=False)}")
    spend = report["spend_usd"]
    lines += ["", "## Anthropic spend of the coding stages", "",
              "By purpose: " + (", ".join(f"{k} ${v:.4f}" for k, v in spend["by_purpose"].items()) or "none") + ".",
              "By stage: " + ", ".join(f"{k} " + ("n/a" if v is None else f"${v:.4f}")
                                       for k, v in spend["by_stage"].items()) + "."]
    return "\n".join(lines) + "\n"


def _first_difference(old: list, new: list, label: Callable[[Any], Any]) -> str:
    n = next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new)))

    def side(rows: list) -> Any:
        return label(rows[n]) if n < len(rows) else None

    return (f"{len(old)} rows on disk, {len(new)} now; first difference at row {n + 1}: on disk {side(old)}, "
            f"now {side(new)}")


def selection_conflicts(processed: Path, main: list[dict], reserve: list[dict]) -> list[str]:
    """The files an earlier assemble wrote under `processed` that disagree with the sets chosen now, one line each
    with its first difference: MAIN_FILE or RESERVE_FILE whose sets (compared as JSON, every field) differ from main
    or reserve, and HUMAN_CHECK_FILE whose (task_id, role) per row differs from that of main + reserve, whether or not
    it holds judgements (the annotators fill copies of it). A missing file agrees; an empty list means that nothing
    the human check reads would change."""
    out = []
    for name, sets in ((MAIN_FILE, main), (RESERVE_FILE, reserve)):
        path = processed / name
        if path.exists():
            old, new = _read_jsonl(path), json.loads(json.dumps(sets))
            if old != new:
                out.append(f"{name}: " + _first_difference(old, new, lambda s: (s.get("task_id"), s.get("role"),
                                                                                  s.get("reserve_rank"))))
    path = processed / HUMAN_CHECK_FILE
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            old = [(r.get("task_id"), r.get("role")) for r in csv.DictReader(f)]
        new = [(s["task_id"], s["role"]) for s in main + reserve]
        if old != new:
            out.append(f"{HUMAN_CHECK_FILE}: " + _first_difference(old, new, lambda row: row))
    return out


def _spend_by_purpose(cost_path: Path, purposes: frozenset[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in _read_jsonl(cost_path):
        if row.get("provider") == "anthropic" and row.get("purpose") in purposes and row.get("cost_usd") is not None:
            out[row["purpose"]] = out.get(row["purpose"], 0.0) + float(row["cost_usd"])
    return out


def stage_assemble(ctx: Ctx5) -> int:
    """Steps 1.4 to 1.6 from the outputs of recheck, checkd and ntr (refuses, EXIT_FAILED, when one of ASSEMBLE_INPUTS
    is missing) and session 4's sets: assemble_sets with cfg["main"]["targets"] and the config seed; each set gets
    transcript_tokens and shown_eligible from its task. No call. Writes, under the processed directory, MAIN_FILE,
    RESERVE_FILE, ADDED_TASKS_FILE (the task dicts of the main and reserve sets that are in none of the four session-4
    task files, in set order) and HUMAN_CHECK_FILE (variants_v4.write_human_check_v4_final: main rows, then reserve
    rows; a file with judgements is kept and a warning logged), and run_dir/coding_final.json and .md.

    The selection, once written, is not changed by a later run: when selection_conflicts finds a final file that
    disagrees with the sets chosen now (inputs changed after the first assemble, e.g. a rerun of ntr), the stage writes
    nothing, logs the differences and returns EXIT_FAILED (stages_status: a previous "done" entry stays, with
    "refused"; else "failed"). To choose the sets again, the final files are moved away first."""
    cfg = ctx.cfg
    missing = [name for name in ASSEMBLE_INPUTS if not (ctx.run_dir / name).exists()]
    if ctx.dry_run:
        present = [n for n in (MAIN_FILE, RESERVE_FILE, HUMAN_CHECK_FILE) if (ctx.processed / n).exists()]
        ctx.note(f"assemble: would choose the main {cfg['main']['total']} sets ({cfg['main']['targets']}) by seed "
                 f"{cfg['seed']} and the reserve, and write {MAIN_FILE}, {RESERVE_FILE}, {ADDED_TASKS_FILE} and "
                 f"{HUMAN_CHECK_FILE} under {ctx.processed}; inputs missing: {missing or 'none'}; final files of an "
                 f"earlier assemble: {present or 'none'} (the stage refuses when they disagree with the sets it "
                 "chooses)")
        return EXIT_OK
    if missing:
        logger.error("assemble: run the recheck, checkd and ntr stages first; missing %s", missing)
        return EXIT_FAILED
    from overclaim import variants_v4

    recheck = _read_json(ctx.run_dir / RECHECK_FILE, {})
    main, reserve, counts = assemble_sets(
        ctx.s4_sets(), recheck["tasks"], _read_jsonl(ctx.run_dir / RECHECK_SETS), _read_jsonl(ctx.run_dir / NTR_SETS),
        _read_jsonl(ctx.run_dir / CHECKD_SETS), {g: int(n) for g, n in cfg["main"]["targets"].items()}, cfg["seed"])
    if not counts["reached"]:
        logger.warning("assemble: only %d main sets of %d (%s); writing what exists", counts["total"],
                       counts["target_total"], counts["taken"])
    built = _read_jsonl(ctx.run_dir / CHECKD_FILE) + _read_jsonl(ctx.run_dir / NTR_BUILD)
    pools = {r["task_id"]: r["pool"] for r in built if r.get("pool")}
    known = ctx.s4_known_tasks()
    task_of: dict[str, dict] = {}
    for s in main + reserve:
        tid = s["task_id"]
        task_of[tid] = known[tid] if tid in known else read_pool_task(ctx, pools[tid])
    cap, upper = int(cfg["shown"]["max_transcript_tokens"]), int(cfg["shown"]["upper_band_tokens"])
    for s in main + reserve:
        s["transcript_tokens"] = task_of[s["task_id"]]["token_counts"]["transcript"]
        s["shown_eligible"] = s["transcript_tokens"] <= cap
    conflicts = selection_conflicts(ctx.processed, main, reserve)
    if conflicts:
        reason = ("the sets chosen now differ from the final files of an earlier assemble (" + "; ".join(conflicts)
                  + "); nothing is written. To choose the sets again, move those files away first")
        logger.error("assemble: REFUSED: %s", reason)
        previous = _read_json(ctx.run_dir / "stages_status.json", {}).get("assemble") or {}
        _record_stage(ctx, "assemble", {**previous, "refused": reason} if previous.get("status") == "done"
                      else {"status": "failed", "reason": reason})
        return EXIT_FAILED
    in_files ={t["task_id"] for path in ctx.s4_task_files() for t in _read_jsonl(path)}
    added = [task_of[s["task_id"]] for s in main + reserve if s["task_id"] not in in_files]
    _write_jsonl(ctx.processed / MAIN_FILE, main)
    _write_jsonl(ctx.processed / RESERVE_FILE, reserve)
    _write_jsonl(ctx.processed / ADDED_TASKS_FILE, added)
    check = ctx.processed / HUMAN_CHECK_FILE
    try:
        n = variants_v4.write_human_check_v4_final(check, list(task_of.values()), main + reserve)
        logger.info("assemble: wrote %d rows to %s", n, check)
    except FileExistsError as exc:  # a human has filled in judgements: the file is kept as it is
        logger.warning("assemble: %s", exc)
    status = _read_json(ctx.run_dir / "stages_status.json", {})
    report = final_report(main, reserve, counts, _read_json(ctx.funnel_path, {}), status,
                          _spend_by_purpose(ctx.run_dir / "cost.jsonl", coding_purposes()), cap, upper)
    _write_json(ctx.run_dir / FINAL_JSON, report)
    (ctx.run_dir / FINAL_MD).write_text(final_markdown(report, cap, upper), encoding="utf-8")
    _record_stage(ctx, "assemble", {"status": "done", "main": len(main), "reserve": len(reserve),
                                    "taken": counts["taken"], "reached": counts["reached"],
                                    "added_tasks": len(added)})
    logger.info("assemble: %d main sets %s, %d reserve sets, %d added tasks", len(main), counts["taken"], len(reserve),
                len(added))
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage: e2 (contract C), and the stage loop
# ---------------------------------------------------------------------------------------------


def stage_e2(ctx: Ctx5) -> int:
    """overclaim.experiments.prep5_e2.stage_e2(ctx), imported when the stage runs; EXIT_FAILED with a message while that
    module does not exist."""
    name = "overclaim.experiments.prep5_e2"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        logger.error("e2: the stage needs %s (the Experiment 2 mixed-pool pilot), which does not exist yet", name)
        return EXIT_FAILED
    return module.stage_e2(ctx)


# ---------------------------------------------------------------------------------------------
# Stage: e3pw (revision of 2026-11-06, section 3: Experiment 3 in pointwise form, coding check)
# ---------------------------------------------------------------------------------------------


def e3pw_task_ids(main_sets: list[dict], n_tasks: int, seed: Any) -> list[str]:
    """The tasks of the e3pw check, in pick order: prep5_e2.pick_tasks over the shown-eligible main sets (each gap
    type's ids in seeded_order(ids, seed), then one id per gap type in turn, no_tests_run, tests_failed,
    tests_narrowed, a gap type with none left passed over, until n_tasks are taken; 5 give 2, 2 and 1). ValueError
    for a task id given twice among all the main sets, shown-eligible or not."""
    from overclaim.experiments import e2_selection
    from overclaim.experiments.prep5_e2 import pick_tasks

    e2_selection._check_unique([s["task_id"] for s in main_sets], "main set task_id")
    return pick_tasks([s for s in main_sets if s["shown_eligible"]], n_tasks, seed)


def stage_e3pw(ctx: Ctx5) -> int:
    """The verification-first scorer's coding check on cfg["e3pw"]: n_tasks main sets (MAIN_FILE; EXIT_FAILED without
    it) chosen by e3pw_task_ids with the config seed, their task dicts from prep5_e2.main_task_dicts, the judge
    cfg["judges"][e3pw.judge] (its spec as it stands for the plain scorer; the verification-first scorer's client has
    max_tokens e3pw.max_tokens_verify_first, the value every verification-first judge of the project runs at, so that
    Part 2 reads these calls from the disk cache), condition
    e3pw.condition, prompt cache on for a shown Anthropic judge. Per task, back to back and one task at a time: the plain
    scorer on O, H and P (e1b_pointwise.score_task, live_before 0) -> run_dir/pointwise_calls.jsonl, then the
    verification-first scorer on O, H and P (e3_verification_pointwise.score_task_vf, cfg["domain"]'s claim phrase,
    live_before 0) -> run_dir/E3PW_CALLS; both with grader = the judge name and set = the set's own "set". Then
    vfp_check on the records of the picked tasks -> run_dir/E3PW_JSON and E3PW_MD, and the stage status with the spend
    of the two scorers' purposes. Behind the stop-launching gate (a skip is recorded); a missing credential skips the
    stage; a BudgetExceeded stops it (pilot2._budget_stop); any other exception (judge.CacheNotRead, a task id given
    twice in the main file, a task dict not found) fails it (_stage_failed) and writes no check. Idempotent: records
    already written are resumed and every call is cached on disk. Dry run: the plan names the tasks and resolves their
    task dicts; no client, no call."""
    cfg, block = ctx.cfg, ctx.cfg["e3pw"]
    judge_name, condition, n_tasks = block["judge"], block["condition"], int(block["n_tasks"])
    spec = cfg["judges"][judge_name]
    use_cache = condition == "shown" and spec["provider"] == "anthropic"
    main_path = ctx.processed / MAIN_FILE
    from overclaim.experiments.prep5_e2 import main_task_dicts

    if ctx.dry_run:
        if not main_path.exists():
            ctx.note(f"e3pw: {main_path} is MISSING: the stage refuses to run without it (run assemble first)")
            return EXIT_OK
        main_sets = _read_jsonl(main_path)
        set_of = {s["task_id"]: s for s in main_sets}
        ids = e3pw_task_ids(main_sets, n_tasks, cfg["seed"])
        try:
            found = f"all {len(main_task_dicts(ctx, ids))} found"
        except ValueError as exc:
            found = f"NOT all found: {exc}"
        ctx.note(f"e3pw: would score O, H and P of {len(ids)} shown-eligible main sets "
                 f"({dict(Counter(set_of[i]['gap_type'] for i in ids))}, seed {cfg['seed']}) with {judge_name} "
                 f"({condition}, prompt cache {'on' if use_cache else 'off'}, max_tokens {spec.get('max_tokens')}, "
                 f"verification-first scorer {int(cfg['e3pw']['max_tokens_verify_first'])}), "
                 f"per task back to back: the plain scorer -> {pilot4.POINTWISE_FILE}, then the verification-first "
                 f"scorer ({cfg['domain']} claim phrase) -> {E3PW_CALLS}; {6 * len(ids)} messages scored (one call "
                 f"each, a second for an unparseable reply; records on disk are resumed), then vfp_check -> "
                 f"{E3PW_JSON} and {E3PW_MD}; task dicts: {found}; credential "
                 f"{'set' if pilot2.credential_for(spec) else 'MISSING -> skipped'}")
        for i in ids:
            ctx.note(f"e3pw: task {i} ({set_of[i]['gap_type']}, {set_of[i]['transcript_tokens']} transcript tokens, "
                     f"origin {set_of[i].get('origin')})")
        return EXIT_OK
    if not main_path.exists():
        logger.error("e3pw: %s is missing; run the assemble stage first", main_path)
        return EXIT_FAILED
    reason = pilot2.gate(ctx, {}, pilot2.COST_PROVIDER[spec["provider"]])
    if reason is not None:
        return _skip(ctx, "e3pw", reason)
    client = ctx.client(spec)
    if client is None:
        return _skip(ctx, "e3pw", f"no credential for {judge_name}")
    vf_client = ctx.client({**spec, "max_tokens": int(cfg["e3pw"]["max_tokens_verify_first"])})
    from overclaim.experiments import e1b_pointwise
    from overclaim.experiments import e3_verification_pointwise as vfp
    from overclaim.graders.judge import BudgetExceeded

    main_sets = _read_jsonl(main_path)
    set_of = {s["task_id"]: s for s in main_sets}   # used only after e3pw_task_ids found no duplicate task id
    structured = bool(spec.get("structured_output"))
    cost_path = ctx.run_dir / "cost.jsonl"
    purposes = frozenset({"pointwise", vfp.VFP_PURPOSE})   # pointwise: graders.pointwise.score_message's default
    spend0 = pilot4.sets_spend(cost_path, purposes)
    plain: list[dict] = []
    vf: list[dict] = []
    try:
        ids = e3pw_task_ids(main_sets, n_tasks, cfg["seed"])
        for task in main_task_dicts(ctx, ids):
            ms = set_of[task["task_id"]]
            plain += e1b_pointwise.score_task(task, ms, client, judge_name, condition,
                                              ctx.run_dir / pilot4.POINTWISE_FILE, cfg["run_name"], set_name=ms["set"],
                                              structured=structured, use_prompt_cache=use_cache)
            vf += vfp.score_task_vf(task, ms, vf_client, judge_name, condition, ctx.run_dir / E3PW_CALLS,
                                    cfg["run_name"], set_name=ms["set"], domain=cfg["domain"], structured=structured,
                                    use_prompt_cache=use_cache)
    except BudgetExceeded as exc:
        _record_stage(ctx, "e3pw", {"status": "stopped", "reason": f"budget cap: {exc}"})
        return pilot2._budget_stop(ctx, "e3pw", exc)
    except Exception as exc:   # judge.CacheNotRead, a task dict not found: recorded, no check written
        return _stage_failed(ctx, "e3pw", exc)
    data, markdown = vfp.vfp_check(vf, plain, [set_of[i] for i in ids])
    _write_json(ctx.run_dir / E3PW_JSON, {"judge": judge_name, "condition": condition, **data})
    (ctx.run_dir / E3PW_MD).write_text(markdown, encoding="utf-8")
    spend = pilot4.sets_spend(cost_path, purposes) - spend0
    _record_stage(ctx, "e3pw", {"status": "done", "judge": judge_name, "n_tasks": len(ids), "task_ids": ids,
                                "parse_rate_verify_first": data["scorers"][vfp.SCORER]["parse_rate"],
                                "n_paired": data["paired"]["n_tasks"],
                                "premium_difference_mean": data["paired"]["premium_difference_mean"],
                                "spend_usd": _spend_total(ctx, "e3pw", spend)})
    logger.info("e3pw: %d tasks, %d paired; premium difference (verification-first - plain) %s", len(ids),
                data["paired"]["n_tasks"], data["paired"]["premium_difference_mean"])
    print(f"[e3pw] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK


_STAGE_FUNCS = {"recheck": stage_recheck, "checkd": stage_checkd, "ntr": stage_ntr, "assemble": stage_assemble,
                "e2": stage_e2, "e3pw": stage_e3pw}


def run(cfg: dict, cfg_path: Path, run_dir: Path, stage: str, dry_run: bool) -> int:
    """Run one stage or all of them (STAGES order) on a Ctx5; returns the exit code. Writes run_dir/config.yaml (the
    loaded config with source_config and resolved_at)."""
    if stage != "all" and stage not in STAGES:
        logger.error("unknown prep5 stage %r; expected one of %s", stage, STAGES)
        return EXIT_FAILED
    ctx = Ctx5(cfg, cfg_path, run_dir, dry_run)
    resolved = dict(cfg, source_config=str(cfg_path),
                    resolved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
                                         encoding="utf-8")
    for name in STAGES if stage == "all" else (stage,):
        logger.info("--- stage %s%s ---", name, " (dry run)" if dry_run else "")
        code = _STAGE_FUNCS[name](ctx)
        if code != EXIT_OK:
            logger.error("run stopped at stage %r with exit code %d", name, code)
            return code
    if dry_run:
        print("=== prep5 dry-run plan (no API calls) ===")
        print(f"ANTHROPIC_API_KEY {'set' if pilot2.credential_for({'provider': 'anthropic'}) else 'MISSING'}")
        for line in ctx.plan:
            print(line)
    elif ctx._cost is not None:
        print(ctx.cost.totals_line())
    return EXIT_OK


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Session 5, Part 1: final coding sets and the Experiment 2 pilot.")
    parser.add_argument("--config", required=True, help="configs/prep5.yaml")
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; no API calls")
    parser.add_argument("--results-root", default=None,
                        help="directory that holds <run_name>/ (default: <repo>/results)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    from overclaim.experiments.e1_preference import _setup_logging
    from overclaim.paths import RESULTS_DIR, load_env

    args = _parse_args(argv)
    if args.dry_run:
        os.environ["HF_HUB_OFFLINE"] = "1"   # read by huggingface_hub at import time
    load_env()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    run_dir = Path(args.results_root or RESULTS_DIR).resolve() / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    handlers = _setup_logging(run_dir)
    try:
        logger.info("run %s (prep5): run_dir %s, stage %s, dry_run %s", cfg["run_name"], run_dir, args.stage,
                    args.dry_run)
        return run(cfg, cfg_path, run_dir, args.stage, args.dry_run)
    finally:
        root = logging.getLogger()
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    sys.exit(main())

"""Session 5, Part 2 ("full_run"): the runner of the full experiments on the frozen coding sets.

Entry point: `uv run python -m overclaim.experiments.full_run --config configs/full_run.yaml --stage S [--dry-run]
[--supersede]`.
Stages, in order:
- freeze (brief Step 8; addendum of 2026-11-07, sections 2, 4 and 7, Step 1): verify the final audit file
  (human_check_v4_final_final.csv) against the config's repair, replacement, do-not-use and repairable-reserve lists;
  swap the replaced main sets for their reserve sets; relabel the audited kept spans (repair_v4.relabel_spans; also
  on a replacement drawn from the repairable reserve); add E6's O+ and O- (repair_v4.dose_variants); join E4's
  outcome fields (resolved, exit_status) from the task files and check each set's model_patch against the
  trajectory's; record the E2 and E7 task picks, the band and the shown-eligible tasks; write the frozen sets
  (data/processed/coding/matched_sets_v4_full_run.jsonl), the frozen task file with a hash of every set
  (data/processed/full_run_tasks.json) and the report results/full_run/freeze.md. No call.

Every input of the freeze is read only. Once written, the frozen files are not changed by a later run: a run whose
records equal the files on disk (ignoring frozen_at) says so and exits 0; one whose records differ writes nothing and
fails naming the first difference. With --supersede (Uzair, 2026-11-08: a re-freeze after further replacements) such
a run instead renames the three outputs to <stem>.superseded-<old frozen_at>.<suffix>, carries the E2 and E7 picks
over from the renamed task file (a replaced task gives its place to its replacement) and writes the new freeze, which
records what it supersedes. Stages support --dry-run (no call, nothing written; the plan says what the stage would
do) and delete nothing. The stage status goes to run_dir/stages_status.json (pilot3._record_stage), the log to
run_dir/run.log. Message text and patches are data: they are compared, hashed and written out, never executed. Exit
codes as pilot 2: 0 ok; 1 a stage failed or an input is missing; 2 a required credential is missing; 3 a budget cap
was hit.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection

import yaml

from overclaim import repair_v4
from overclaim.experiments import pilot3, pilot4
from overclaim.experiments.pilot2 import EXIT_FAILED, EXIT_OK, _read_json, _read_jsonl, _repo_path
from overclaim.experiments.pilot3 import _record_stage

logger = logging.getLogger(__name__)

STAGES = ("freeze", "pointwise_hidden", "tie_rule", "rm", "pointwise_shown", "e2", "pairwise", "reasons",
          "analysis")   # run order; each later stage lives in its own module (full_run_<name>.py), registered below
GAP_TYPES = pilot3.GAP_TYPES
# The judgement columns of the final human-check file (variants_v4.HUMAN_CHECK_V4_COLUMNS); a repair fails only the
# second one (addendum 4.1).
JUDGEMENT_COLUMNS = ("O_primary_claim_false", "O_other_claims_supported", "H_disclosure_true",
                     "P_free_of_outcome_claims", "rewrites_match_patch")
REPAIR_COLUMN = "O_other_claims_supported"
DO_NOT_USE_NOTE = "do not use"
FREEZE_PATH_KEYS = ("main_sets", "reserve_sets", "audit_csv", "variants_config", "checkd_unexplained",
                    "e2_pilot_samples", "sets_out", "tasks_out", "report")
FREEZE_INPUT_KEYS = ("main_sets", "reserve_sets", "audit_csv", "variants_config", "checkd_unexplained",
                     "e2_pilot_samples")
PART = "session 5, Part 2"
# The replacement_tier of an incoming set drawn from freeze.repairable_reserve (reserve rows that fail only
# REPAIR_COLUMN; Uzair, 2026-11-08). An incoming set from the sound reserve carries no replacement_tier, so the sets
# frozen before this tier existed stay byte for byte.
REPAIRABLE_TIER = "repairable"
# A re-freeze (--supersede): the pick lists carried over from the superseded task file, and the reasons of a
# substitution in them.
CARRIED_LISTS = ("e2_tasks", "e7_tasks")
SUB_REPLACEMENT = "replacement"
SUB_NEXT_IN_ORDER = "next in seeded order (replacement not shown-eligible)"


class FreezeError(RuntimeError):
    """A check of the freeze stage failed; the message names every problem found."""


# ---------------------------------------------------------------------------------------------
# Config and context
# ---------------------------------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    """configs/full_run.yaml with the judges and prices of configs/models.yaml (cfg["judges"],
    cfg["prices_usd_per_mtok"]). Checks (ValueError listing every failed one): shown has 0 < max_transcript_tokens <
    upper_band_tokens; e2.n_tasks is a positive integer; e7.n_tasks equals e7.per_gap times the number of gap types;
    the freeze block has every key of FREEZE_PATH_KEYS, a non-empty task_files list and repair_reason; each repairs
    entry is a non-empty list of distinct positive integers; each replacement has an out and an in; no task id is
    both repaired and replaced or replaced twice; print_sets, do_not_use and task_files are lists; the optional
    repairable_reserve (absent means none) maps task ids to non-empty lists of distinct positive integers, and none of
    its ids is in do_not_use or repairs too (that each is a reserve row that fails only REPAIR_COLUMN is verify_audit's
    check)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(_repo_path(cfg["models_config"]), encoding="utf-8") as f:
        models = yaml.safe_load(f)
    cfg["judges"] = models["judges"]
    cfg["prices_usd_per_mtok"] = models["prices_usd_per_mtok"]
    problems = []
    shown = cfg["shown"]
    if not 0 < int(shown["max_transcript_tokens"]) < int(shown["upper_band_tokens"]):
        problems.append("shown: need 0 < max_transcript_tokens < upper_band_tokens")
    if not _positive_int(cfg["e2"]["n_tasks"]):
        problems.append("e2.n_tasks must be a positive integer")
    e7 = cfg["e7"]
    if not (_positive_int(e7["n_tasks"]) and _positive_int(e7["per_gap"])
            and e7["n_tasks"] == e7["per_gap"] * len(GAP_TYPES)):
        problems.append(f"e7: n_tasks must be per_gap times {len(GAP_TYPES)} (both positive integers)")
    fz = cfg["freeze"]
    missing = [k for k in FREEZE_PATH_KEYS if not fz.get(k)]
    if missing:
        problems.append(f"freeze: missing {missing}")
    if not isinstance(fz.get("task_files"), list) or not fz["task_files"]:
        problems.append("freeze.task_files must be a non-empty list")
    if not isinstance(fz.get("repair_reason"), str) or not fz["repair_reason"].strip():
        problems.append("freeze.repair_reason must be a non-empty string")
    for key in ("print_sets", "do_not_use", "replacements"):
        if not isinstance(fz.get(key), list):
            problems.append(f"freeze.{key} must be a list")
    repairs = fz.get("repairs")
    if not isinstance(repairs, dict) or not repairs:
        problems.append("freeze.repairs must map task ids to span indices")
        repairs = {}
    for task_id, indices in repairs.items():
        if not _span_list(indices):
            problems.append(f"freeze.repairs[{task_id}]: a non-empty list of distinct positive integers")
    repairable = fz.get("repairable_reserve", {})
    if not isinstance(repairable, dict):
        problems.append("freeze.repairable_reserve must map reserve task ids to span indices")
        repairable = {}
    for task_id, indices in repairable.items():
        if not _span_list(indices):
            problems.append(f"freeze.repairable_reserve[{task_id}]: a non-empty list of distinct positive integers")
    do_not_use = fz.get("do_not_use") if isinstance(fz.get("do_not_use"), list) else []
    clash = sorted(set(repairable) & (set(do_not_use) | set(repairs)))
    if clash:
        problems.append(f"freeze.repairable_reserve: ids that are also in do_not_use or repairs: {clash}")
    replacements = fz.get("replacements") if isinstance(fz.get("replacements"), list) else []
    if not all(isinstance(r, dict) and r.get("out") and r.get("in") for r in replacements):
        problems.append("freeze.replacements: each entry needs an out and an in")
    else:
        outs, ins = [r["out"] for r in replacements], [r["in"] for r in replacements]
        twice = sorted({i for i in outs + ins if (outs + ins).count(i) > 1} | (set(outs) & set(repairs)))
        if twice:
            problems.append(f"freeze: task ids named twice among the repairs and replacements: {twice}")
    # The later stages' blocks, checked by their own modules when present (each stage checks again before it runs).
    from overclaim.experiments import full_run_analysis, full_run_pairwise, full_run_pointwise, full_run_shown

    for present, check in (("pointwise_hidden" in cfg and "tie_rule" in cfg and "rm" in cfg,
                            full_run_pointwise.validate_config),
                           ("pointwise_shown" in cfg, full_run_shown.check_block),
                           (all(k in cfg for k in full_run_pairwise.CONFIG_BLOCKS), full_run_pairwise.check_config),
                           ("analysis" in cfg, full_run_analysis.check_config)):
        if present:
            try:
                check(cfg)
            except ValueError as exc:
                problems.append(str(exc))
    if problems:
        raise ValueError("invalid full_run config:\n" + "\n".join(problems))
    return cfg


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _span_list(value: Any) -> bool:
    """True for a non-empty list of distinct positive integers."""
    return (isinstance(value, list) and bool(value) and all(_positive_int(i) for i in value)
            and len(set(value)) == len(value))


class Ctx6(pilot4.Ctx4):
    """pilot4.Ctx4 on the full_run config: run_dir results/full_run, cost file run_dir/cost.jsonl with the config's
    budget caps, the processed directory cfg["data"]["processed_dir"]. The freeze stage uses no client. supersede
    (the CLI's --supersede) lets the freeze stage replace frozen files that differ (stage_freeze)."""

    def __init__(self, cfg: dict, cfg_path: Path, run_dir: Path, dry_run: bool, supersede: bool = False,
                 only: Collection[str] | None = None) -> None:
        super().__init__(cfg, cfg_path, run_dir, dry_run)
        self.supersede = supersede
        # --only (Uzair's memo of 2026-11-08: the priority order of the AI Studio rows crosses the pointwise_shown and
        # pairwise stages): the run ids a stage runs this invocation; the other runs are left untouched. None = all.
        self.only: set[str] | None = None if only is None else set(only)


# ---------------------------------------------------------------------------------------------
# Small helpers: hashes, JSON comparison
# ---------------------------------------------------------------------------------------------


def text_sha256(text: str) -> str:
    """sha256 hex digest of the utf-8 bytes of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_sha256(record: dict) -> str:
    """sha256 hex digest of json.dumps(record, sort_keys=True, ensure_ascii=False) (the project's record hash)."""
    return text_sha256(json.dumps(record, sort_keys=True, ensure_ascii=False))


def file_sha256(path: Path) -> str:
    """sha256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl_text(records: list[dict]) -> str:
    """The bytes pilot2._write_jsonl writes, as text: one json.dumps(record, ensure_ascii=False) line per record."""
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def _write_text(path: Path, text: str) -> None:
    """Write text (utf-8) to a temporary file next to path, then move it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _short(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 120 else text[:117] + "..."


def first_difference(old: Any, new: Any, path: str = "") -> str | None:
    """None when old == new; else where they first differ, as "<path>: on disk <old>, now <new>" (dict keys in old's
    order, then new's extra keys; list items in order, then a length difference). Values are shown cut to 120
    characters."""
    where = path or "(top)"
    if isinstance(old, dict) and isinstance(new, dict):
        for key in list(old) + [k for k in new if k not in old]:
            if key not in new or key not in old:
                return (f"{path}.{key}: " + ("missing now" if key not in new else "missing on disk")).lstrip(".")
            found = first_difference(old[key], new[key], f"{path}.{key}".lstrip("."))
            if found:
                return found
        return None
    if isinstance(old, list) and isinstance(new, list):
        for i, (a, b) in enumerate(zip(old, new)):
            found = first_difference(a, b, f"{path}[{i}]")
            if found:
                return found
        return None if len(old) == len(new) else f"{where}: {len(old)} items on disk, {len(new)} now"
    return None if old == new and type(old) is type(new) else f"{where}: on disk {_short(old)}, now {_short(new)}"


# ---------------------------------------------------------------------------------------------
# Step 1: the audit file against the lists
# ---------------------------------------------------------------------------------------------


def read_audit(path: Path) -> list[dict]:
    """The rows of the final human-check CSV (utf-8 with or without BOM), in file order."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _verdicts(row: dict) -> dict[str, str]:
    return {c: (row.get(c) or "").strip().lower() for c in JUDGEMENT_COLUMNS}


def verify_audit(rows: list[dict], repairs: dict[str, list[int]], replacements: list[dict],
                 do_not_use: list[str], repairable: Collection[str] | None = None) -> dict:
    """Addendum 4.5, "verify that the file matches this list before applying it". Values are compared stripped and
    lower-cased. FreezeError listing every problem when: a task id has two rows; a repairs id is not a main row with
    "no" on REPAIR_COLUMN and "yes" on the four other JUDGEMENT_COLUMNS; a repairable id (freeze.repairable_reserve)
    is not a reserve row with that same pattern; a replacement's out is not a main row with a "no" in some column; its
    in is not a reserve row of the out's gap type that is the first candidate of that gap type, the candidates being
    the reserve rows with "yes" in all five that are neither in do_not_use nor the in of an earlier replacement, in
    file order, then (Uzair, 2026-11-08) the repairable rows that are not an earlier in, in file order; a do_not_use
    id is not a reserve row whose notes contain DO_NOT_USE_NOTE (case-insensitive); a main row with a "no" is neither
    a repair nor a replacement's out.

    Returns {"rows", "main_rows", "reserve_rows", "main_failing" (ids, file order), "repairs_verified",
    "replacements_verified", "do_not_use_verified", "reserve_failing_unlisted": the reserve rows with a "no" that are
    neither in do_not_use nor an in (file order; reported, not an error)}; when repairable is given (None: today's
    result), also "repairable_verified" (how many) and "reserve_failing_in_no_list": the rows of
    reserve_failing_unlisted that are not repairable either, i.e. in none of the three lists."""
    repairable_ids = set(repairable or ())
    problems: list[str] = []
    by_id: dict[str, dict] = {}
    for row in rows:
        if row["task_id"] in by_id:
            problems.append(f"{row['task_id']}: two rows")
        by_id[row["task_id"]] = row

    def role(task_id: str) -> str | None:
        return by_id[task_id]["role"] if task_id in by_id else None

    def has_no(row: dict) -> bool:
        return "no" in _verdicts(row).values()

    def sound(row: dict) -> bool:
        return all(v == "yes" for v in _verdicts(row).values())

    def repair_only(row: dict) -> bool:
        v = _verdicts(row)
        return v[REPAIR_COLUMN] == "no" and all(v[c] == "yes" for c in JUDGEMENT_COLUMNS if c != REPAIR_COLUMN)

    for task_id in repairs:
        if role(task_id) != "main":
            problems.append(f"repair {task_id}: not a main row of the audit file")
        elif not repair_only(by_id[task_id]):
            problems.append(f"repair {task_id}: expected no on {REPAIR_COLUMN} and yes elsewhere, found "
                            f"{_verdicts(by_id[task_id])}")
    for task_id in repairable or ():
        if role(task_id) != "reserve":
            problems.append(f"repairable {task_id}: not a reserve row of the audit file")
        elif not repair_only(by_id[task_id]):
            problems.append(f"repairable {task_id}: expected no on {REPAIR_COLUMN} and yes elsewhere, found "
                            f"{_verdicts(by_id[task_id])}")
    taken: set[str] = set()
    for r in replacements:
        out, inc = r["out"], r["in"]
        if role(out) != "main" or not has_no(by_id[out]):
            problems.append(f"replacement out {out}: not a main row with a no")
            continue
        gap = by_id[out]["gap_type"]
        reserve_of_gap = [row for row in rows if row["role"] == "reserve" and row["gap_type"] == gap]
        first_sound = next((row["task_id"] for row in reserve_of_gap if sound(row) and row["task_id"] not in do_not_use
                            and row["task_id"] not in taken), None)
        first_repairable = next((row["task_id"] for row in reserve_of_gap if row["task_id"] in repairable_ids
                                 and row["task_id"] not in taken), None)
        if (role(inc) != "reserve" or by_id[inc]["gap_type"] != gap
                or not (sound(by_id[inc]) or inc in repairable_ids)):
            problems.append(f"replacement in {inc}: not a reserve row of gap type {gap} with yes in all five columns"
                            + (" or on the repairable list" if repairable_ids else ""))
        elif (first_sound is not None or not repairable_ids) and inc != first_sound:
            problems.append(f"replacement in {inc}: the first sound {gap} reserve row (file order, do_not_use and "
                            f"earlier replacements skipped) is {first_sound}")
        elif first_sound is None and inc != first_repairable:
            problems.append(f"replacement in {inc}: no sound {gap} reserve row is left (file order, do_not_use and "
                            f"earlier replacements skipped); the first repairable one is {first_repairable}")
        taken.add(inc)
    for task_id in do_not_use:
        if role(task_id) != "reserve" or DO_NOT_USE_NOTE not in (by_id[task_id].get("notes") or "").lower():
            problems.append(f"do_not_use {task_id}: not a reserve row whose notes contain {DO_NOT_USE_NOTE!r}")
    outs = {r["out"] for r in replacements}
    main_failing = [row["task_id"] for row in rows if row["role"] == "main" and has_no(row)]
    unlisted = [i for i in main_failing if i not in repairs and i not in outs]
    if unlisted:
        problems.append(f"main rows with a no that are neither a repair nor a replacement: {unlisted}")
    if problems:
        raise FreezeError("the audit file does not match the lists:\n" + "\n".join(problems))
    ins = {r["in"] for r in replacements}
    result = {"rows": len(rows), "main_rows": sum(row["role"] == "main" for row in rows),
              "reserve_rows": sum(row["role"] == "reserve" for row in rows), "main_failing": main_failing,
              "repairs_verified": len(repairs), "replacements_verified": len(replacements),
              "do_not_use_verified": len(do_not_use),
              "reserve_failing_unlisted": [row["task_id"] for row in rows if row["role"] == "reserve" and has_no(row)
                                           and row["task_id"] not in do_not_use and row["task_id"] not in ins]}
    if repairable is not None:
        result["repairable_verified"] = len(repairable_ids)
        result["reserve_failing_in_no_list"] = [i for i in result["reserve_failing_unlisted"]
                                                if i not in repairable_ids]
    return result


# ---------------------------------------------------------------------------------------------
# Steps 2 to 6: replacements, repairs, E6, E4 join, picks
# ---------------------------------------------------------------------------------------------


def replace_sets(main: list[dict], reserve: list[dict], replacements: list[dict],
                 repairable: Collection[str] = ()) -> list[dict]:
    """Copies of the main sets in their order, each replacement's out swapped for its in (the reserve set) at the same
    position. The incoming set gets role "main", replaced_for (the out id), replacement_reserve_rank (its reserve_rank)
    and reserve_rank None, and, when its id is in repairable, replacement_tier REPAIRABLE_TIER (an incoming set from
    the sound reserve gets no replacement_tier); every other set gets replaced_for None. FreezeError when an out is not
    a main set, an in not a reserve set, or their gap types differ."""
    by_main = {s["task_id"]: s for s in main}
    by_reserve = {s["task_id"]: s for s in reserve}
    incoming: dict[str, dict] = {}
    for r in replacements:
        if r["out"] not in by_main or r["in"] not in by_reserve:
            raise FreezeError(f"replacement {r['out']} -> {r['in']}: the out must be a main set and the in a reserve "
                              "set")
        new = copy.deepcopy(by_reserve[r["in"]])
        if new["gap_type"] != by_main[r["out"]]["gap_type"]:
            raise FreezeError(f"replacement {r['out']} -> {r['in']}: gap types differ")
        new.update(role="main", replaced_for=r["out"], replacement_reserve_rank=new["reserve_rank"], reserve_rank=None)
        if r["in"] in repairable:
            new["replacement_tier"] = REPAIRABLE_TIER
        incoming[r["out"]] = new
    return [incoming[s["task_id"]] if s["task_id"] in incoming else {**copy.deepcopy(s), "replaced_for": None}
            for s in main]


def repair_sets(sets: list[dict], repairs: dict[str, list[int]], bank: dict, vcfg: dict, lint_patterns: list[str],
                reason: str, repairable: dict[str, list[int]] | None = None) -> list[dict]:
    """repair_v4.relabel_spans on each set named in repairs (its indices, `reason`) and on each incoming replacement
    (replaced_for set) whose id is in repairable (its configured indices, the same reason); every other set gets
    repaired_spans 0 and repair None. FreezeError naming the repairs ids that are not among the sets, and a set named in
    both; a RepairError of relabel_spans propagates."""
    missing = sorted(set(repairs) - {s["task_id"] for s in sets})
    if missing:
        raise FreezeError(f"repairs name sets that are not among the main sets: {missing}")
    spans_of, repairable = dict(repairs), repairable or {}
    for s in sets:
        if s.get("replaced_for") and s["task_id"] in repairable:
            if s["task_id"] in spans_of:
                raise FreezeError(f"{s['task_id']}: named in both repairs and repairable_reserve")
            spans_of[s["task_id"]] = repairable[s["task_id"]]
    return [repair_v4.relabel_spans(s, spans_of[s["task_id"]], bank, vcfg, lint_patterns, reason=reason)
            if s["task_id"] in spans_of else {**s, "repaired_spans": 0, "repair": None} for s in sets]


def add_dose_variants(sets: list[dict], vcfg: dict, lint_patterns: list[str]) -> None:
    """E6: repair_v4.dose_variants on every set, in place: variants["O_plus"], variants["O_minus"] and the top-level
    O_plus and O_minus (next to O, H and P)."""
    for s in sets:
        dose = repair_v4.dose_variants(s, vcfg, lint_patterns)
        s["variants"] = {**s["variants"], "O_plus": dose["O_plus"], "O_minus": dose["O_minus"]}
        s["O_plus"], s["O_minus"] = dose["O_plus"], dose["O_minus"]


def join_tasks(sets: list[dict], task_files: list[Path], patch_fn: Callable[[dict], str | None] | None = None
               ) -> dict[str, str]:
    """E4, in place: each set's task record from task_files (the first file that has the task id wins) gives
    resolved (as int) and exit_status. The task files carry no model_patch, so the check of the set's model_patch is
    against the trajectory's own, read from the raw row the task points at by (patch_fn or
    coding.model_patch_for_task, looked up at call time)(task), which also checks the trajectory id. Returns
    {task_id: the task file's name}. FreezeError naming the task ids found in no file, and the first set whose raw row
    cannot be read (patch_fn returns None; this fails even when the set's model_patch is None too) or whose
    model_patch differs."""
    from overclaim.data import coding

    wanted = {s["task_id"] for s in sets}
    found: dict[str, tuple[str, dict]] = {}
    for path in task_files:
        for task in _read_jsonl(path):
            if task["task_id"] in wanted and task["task_id"] not in found:
                found[task["task_id"]] = (path.name, task)
    missing = [s["task_id"] for s in sets if s["task_id"] not in found]
    if missing:
        raise FreezeError(f"{len(missing)} sets have no task record in {[p.name for p in task_files]}: {missing}")
    read_patch = patch_fn or coding.model_patch_for_task
    for s in sets:
        task = found[s["task_id"]][1]
        patch = read_patch(task)
        if patch is None:
            raise FreezeError(f"{s['task_id']}: the trajectory's raw row could not be read (file missing or its id "
                              "differs), so its model_patch cannot be checked")
        if patch != s["model_patch"]:
            raise FreezeError(f"{s['task_id']}: the set's model_patch differs from the trajectory's")
        s["resolved"], s["exit_status"] = int(task["resolved"]), task["exit_status"]
    return {task_id: name for task_id, (name, _) in found.items()}


def e7_pick(sets: list[dict], n_tasks: int, seed: Any) -> list[str]:
    """E7's tasks: prep5_e2.pick_tasks over the shown-eligible sets, i.e. per gap type seeded_order(ids, seed), then one
    per gap type in turn in prep5_e2.GAP_ORDER (a gap type with none left is passed over) until n_tasks are taken; a
    gap type with fewer eligible tasks gives its place to the others. Pick order."""
    from overclaim.experiments.prep5_e2 import pick_tasks

    return pick_tasks([s for s in sets if s["shown_eligible"]], n_tasks, seed)


def picks(sets: list[dict], cfg: dict) -> dict:
    """The recorded picks: E2 (prep5_e2.pick_tasks over all sets, cfg["e2"]) and its shown-eligible ids; E7 (e7_pick,
    cfg["e7"]["n_tasks"], cfg["seed"]) and its split per gap type; the band tasks (shown.max_transcript_tokens <
    transcript_tokens <= shown.upper_band_tokens) and the shown-eligible ids, in set order. FreezeError when a set's
    shown_eligible disagrees with transcript_tokens <= shown.max_transcript_tokens."""
    from overclaim.experiments.prep5_e2 import pick_tasks

    cap, upper = int(cfg["shown"]["max_transcript_tokens"]), int(cfg["shown"]["upper_band_tokens"])
    wrong = [s["task_id"] for s in sets if s["shown_eligible"] != (s["transcript_tokens"] <= cap)]
    if wrong:
        raise FreezeError(f"shown_eligible disagrees with transcript_tokens <= {cap} for {wrong}")
    eligible = {s["task_id"] for s in sets if s["shown_eligible"]}
    gap_of = {s["task_id"]: s["gap_type"] for s in sets}
    e2 = pick_tasks(sets, int(cfg["e2"]["n_tasks"]), cfg["e2"]["seed"])
    e7 = e7_pick(sets, int(cfg["e7"]["n_tasks"]), cfg["seed"])
    return {"e2_tasks": e2, "e2_shown_eligible": [i for i in e2 if i in eligible], "e7_tasks": e7,
            "e7_per_gap": {g: sum(gap_of[i] == g for i in e7) for g in GAP_TYPES},
            "band_tasks": [s["task_id"] for s in sets if cap < s["transcript_tokens"] <= upper],
            "shown_eligible": [s["task_id"] for s in sets if s["shown_eligible"]]}


def carry_picks(sets: list[dict], cfg: dict, old_picks: dict) -> tuple[dict, list[dict]]:
    """A re-freeze (Uzair, 2026-11-08): the E2 and E7 lists (CARRIED_LISTS) of old_picks (the superseded task file's
    picks) kept as frozen on the new sets, instead of picked afresh. In each list, in order, a task id that is still a
    set keeps its position; one that is not is taken by the set whose replaced_for is that id when it is not already in
    the list and, for E7, is shown-eligible (reason SUB_REPLACEMENT), else by the first set of the same gap type in
    the list's fresh order on the new sets (prep5_e2.pick_tasks with cfg["e2"]["seed"] for E2, e7_pick with
    cfg["seed"] for E7, over every set) that is not already in the list and, for E7, is shown-eligible (reason
    SUB_NEXT_IN_ORDER). "Already in the list" counts the kept ids and the earlier substitutions. e2_shown_eligible,
    e7_per_gap, band_tasks and shown_eligible are computed on the new sets as picks() does (with its check of
    shown_eligible). Returns (picks, substitutions: [{"list", "out", "in", "reason"}] in list then position order).
    FreezeError when a gone task id has no replacing set or no candidate is left."""
    from overclaim.experiments.prep5_e2 import pick_tasks

    fresh = picks(sets, cfg)
    by_id = {s["task_id"]: s for s in sets}
    replacing = {s["replaced_for"]: s["task_id"] for s in sets if s.get("replaced_for")}
    orders = {"e2_tasks": pick_tasks(sets, len(sets), cfg["e2"]["seed"]),
              "e7_tasks": e7_pick(sets, len(sets), cfg["seed"])}
    carried: dict[str, list[str]] = {}
    substitutions: list[dict] = []

    def usable(task_id: str, key: str, in_list: set[str]) -> bool:
        return task_id not in in_list and (key != "e7_tasks" or by_id[task_id]["shown_eligible"])

    for key in CARRIED_LISTS:
        ids = list(old_picks[key])
        in_list = {i for i in ids if i in by_id}
        for pos, old in enumerate(ids):
            if old in by_id:
                continue
            if old not in replacing:
                raise FreezeError(f"{key}: {old} is not among the frozen sets and no set replaces it")
            new, reason = replacing[old], SUB_REPLACEMENT
            if not usable(new, key, in_list):
                gap = by_id[new]["gap_type"]
                new = next((i for i in orders[key] if by_id[i]["gap_type"] == gap and usable(i, key, in_list)), None)
                if new is None:
                    raise FreezeError(f"{key}: no {gap} set is left to take the place of {old}")
                reason = SUB_NEXT_IN_ORDER
            ids[pos] = new
            in_list.add(new)
            substitutions.append({"list": key, "out": old, "in": new, "reason": reason})
        carried[key] = ids
    eligible = set(fresh["shown_eligible"])
    gap_of = {s["task_id"]: s["gap_type"] for s in sets}
    e2, e7 = carried["e2_tasks"], carried["e7_tasks"]
    return ({"e2_tasks": e2, "e2_shown_eligible": [i for i in e2 if i in eligible], "e7_tasks": e7,
             "e7_per_gap": {g: sum(gap_of[i] == g for i in e7) for g in GAP_TYPES},
             "band_tasks": fresh["band_tasks"], "shown_eligible": fresh["shown_eligible"]}, substitutions)


# ---------------------------------------------------------------------------------------------
# The freeze: compute (steps 1 to 6), the task file (step 7), the report (step 8)
# ---------------------------------------------------------------------------------------------


def compute_freeze(cfg: dict) -> dict:
    """Steps 1 to 6 of the freeze, reading only: {"audit": verify_audit's result, "originals": {task_id: the set as
    read (main or reserve)}, "sets": the frozen sets as they would be written (JSON round-tripped), "task_files":
    {task_id: file name}, "picks" (picked afresh; a re-freeze carries them, see stage_freeze), "checkd": the excluded
    check-d ids, "e2_pilot": the sorted task ids of the E2 pilot's samples}. freeze.repairable_reserve, when present,
    goes to verify_audit, replace_sets and repair_sets. FreezeError naming the inputs that do not exist; FreezeError,
    repair_v4.RepairError and ValueError from the steps propagate."""
    from overclaim.variants_v4 import load_v4_config

    fz = cfg["freeze"]
    paths = {k: _repo_path(fz[k]) for k in FREEZE_INPUT_KEYS}
    task_files = [_repo_path(p) for p in fz["task_files"]]
    missing = [str(p) for p in [*paths.values(), *task_files] if not p.exists()]
    if missing:
        raise FreezeError(f"inputs missing: {missing}")
    repairs = {k: list(v) for k, v in fz["repairs"].items()}
    repairable = ({k: list(v) for k, v in fz["repairable_reserve"].items()} if "repairable_reserve" in fz
                  else None)
    audit = verify_audit(read_audit(paths["audit_csv"]), repairs, fz["replacements"], fz["do_not_use"], repairable)
    main, reserve = _read_jsonl(paths["main_sets"]), _read_jsonl(paths["reserve_sets"])
    originals = {s["task_id"]: s for s in reserve + main}
    sets = replace_sets(main, reserve, fz["replacements"], repairable or {})
    vcfg, bank, lint_patterns = load_v4_config(paths["variants_config"])
    sets = repair_sets(sets, repairs, bank, vcfg, lint_patterns, fz["repair_reason"], repairable)
    add_dose_variants(sets, vcfg, lint_patterns)
    files_of = join_tasks(sets, task_files)
    unknown = [i for i in fz["print_sets"] if i not in {s["task_id"] for s in sets}]
    if unknown:
        raise FreezeError(f"print_sets names sets that are not among the frozen sets: {unknown}")
    return {"audit": audit, "originals": originals, "sets": json.loads(json.dumps(sets)), "task_files": files_of,
            "picks": picks(sets, cfg),
            "checkd": [s["task_id"] for s in _read_jsonl(paths["checkd_unexplained"])],
            "e2_pilot": sorted({r["task_id"] for r in _read_jsonl(paths["e2_pilot_samples"])})}


def tasks_document(cfg: dict, frozen: dict) -> dict:
    """The frozen task file (tasks_out) without frozen_at: part, the sets file and the sha256 of the bytes it is
    written with, the sha256 of each source file, the counts, the replacements, the do-not-use and unlisted failing
    reserve rows, the excluded check-d ids, the repairs, the picks and one entry per set (hashes of O, H, P, O_plus,
    O_minus and of the whole record, record_sha256 of the set as written). Only when present in frozen (a re-freeze,
    stage_freeze): "supersedes" after part, and "picks_carried_from" and "picks_substitutions" after picks; a
    replacement whose incoming set has a replacement_tier carries it too. Without these the document is today's."""
    fz = cfg["freeze"]
    sets, pk = frozen["sets"], frozen["picks"]
    in_e2, in_e7, in_band = set(pk["e2_tasks"]), set(pk["e7_tasks"]), set(pk["band_tasks"])
    gap_of = {s["task_id"]: s["gap_type"] for s in sets}
    tier_of = {s["task_id"]: s["replacement_tier"] for s in sets if s.get("replacement_tier")}
    repaired = [s for s in sets if s["repaired_spans"]]
    return {
        "part": PART,
        **({"supersedes": frozen["supersedes"]} if frozen.get("supersedes") else {}),
        "sets_file": fz["sets_out"],
        "sets_file_sha256": text_sha256(_jsonl_text(sets)),
        "sources": {k: {"path": fz[k], "sha256": file_sha256(_repo_path(fz[k]))}
                    for k in ("main_sets", "reserve_sets", "audit_csv", "variants_config")},
        "counts": {"sets": len(sets), "per_gap": {g: sum(s["gap_type"] == g for s in sets) for g in GAP_TYPES},
                   "shown_eligible": len(pk["shown_eligible"]), "band": len(pk["band_tasks"]),
                   "repaired_sets": len(repaired), "repaired_spans_total": sum(s["repaired_spans"] for s in sets),
                   "resolved": sum(s["resolved"] for s in sets),
                   "resolved_per_gap": {g: sum(s["resolved"] for s in sets if s["gap_type"] == g) for g in GAP_TYPES}},
        "replacements": [{"out": r["out"], "in": r["in"], "gap_type": gap_of[r["in"]],
                          "reserve_rank": frozen["originals"][r["in"]]["reserve_rank"],
                          **({"replacement_tier": tier_of[r["in"]]} if r["in"] in tier_of else {})}
                         for r in fz["replacements"]],
        "do_not_use_reserve": list(fz["do_not_use"]),
        "reserve_failing_unlisted": frozen["audit"]["reserve_failing_unlisted"],
        "excluded_checkd_unexplained": frozen["checkd"],
        "repairs": [{"task_id": s["task_id"], "gap_type": s["gap_type"], "spans": s["repair"]["spans"],
                     "filler_ids": s["repair"]["filler_ids"],
                     "kept_before": frozen["originals"][s["task_id"]]["n_kept_true_spans"],
                     "kept_after": s["n_kept_true_spans"], "lengths_before": s["repair"]["lengths_before"],
                     "lengths_after": s["lengths"]} for s in repaired],
        "picks": pk,
        **({"picks_carried_from": frozen["picks_carried_from"], "picks_substitutions": frozen["picks_substitutions"]}
           if "picks_carried_from" in frozen else {}),
        "tasks": [{"task_id": s["task_id"], "gap_type": s["gap_type"], "origin": s.get("origin"),
                   "replaced_for": s["replaced_for"], "task_file": frozen["task_files"][s["task_id"]],
                   "shown_eligible": s["shown_eligible"], "transcript_tokens": s["transcript_tokens"],
                   "repaired_spans": s["repaired_spans"],
                   "repaired_indices": s["repair"]["spans"] if s["repair"] else [],
                   "resolved": s["resolved"], "exit_status": s["exit_status"],
                   "model_patch_sha256": text_sha256(s["model_patch"]),
                   "sha256": {**{k: text_sha256(s[k]) for k in ("O", "H", "P", "O_plus", "O_minus")},
                              "record": record_sha256(s)},
                   "in_e2": s["task_id"] in in_e2, "in_e7": s["task_id"] in in_e7, "in_band": s["task_id"] in in_band}
                  for s in sets],
    }


def output_differences(cfg: dict, sets: list[dict], doc: dict) -> tuple[list[str], list[str]]:
    """(differences, existing): for each of sets_out and tasks_out that exists, whether it equals what would be
    written (sets compared as JSON records; the task file without frozen_at); differences holds one line per file
    that differs, with first_difference; existing names the files that exist."""
    fz = cfg["freeze"]
    out, existing = [], []
    sets_path, tasks_path = _repo_path(fz["sets_out"]), _repo_path(fz["tasks_out"])
    if sets_path.exists():
        existing.append(fz["sets_out"])
        diff = first_difference(_read_jsonl(sets_path), sets)
        if diff:
            out.append(f"{fz['sets_out']}: {diff}")
    if tasks_path.exists():
        existing.append(fz["tasks_out"])
        old = {k: v for k, v in _read_json(tasks_path, {}).items() if k != "frozen_at"}
        diff = first_difference(old, doc)
        if diff:
            out.append(f"{fz['tasks_out']}: {diff}")
    return out, existing


def superseded_name(path: str, frozen_at: str) -> str:
    """path (as the config writes it) with ".superseded-<stamp>" put before its suffix, stamp = frozen_at in UTC as
    YYYYMMDDTHHMMSSZ: ("x/full_run_tasks.json", "2026-11-07T17:26:55+00:00") gives
    "x/full_run_tasks.superseded-20261107T172655Z.json"."""
    stamp = datetime.fromisoformat(frozen_at).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = Path(path)
    return str(p.with_name(f"{p.stem}.superseded-{stamp}{p.suffix}"))


def supersede_plan(cfg: dict, old_doc: dict) -> dict:
    """What a re-freeze (--supersede) does with the outputs on disk, read only: {"renames": [(path, renamed path)] for
    sets_out, tasks_out and, when it exists, the report (paths as the config writes them, renamed with
    superseded_name and old_doc's frozen_at), "supersedes": {"frozen_at": old_doc's, "sets_file" and "tasks_file":
    {"path": the renamed path, "sha256": of the file's bytes}, "report": the renamed path or None}}. FreezeError when
    sets_out or tasks_out does not exist, old_doc has no frozen_at, or a renamed path exists already."""
    fz = cfg["freeze"]
    missing = [fz[k] for k in ("sets_out", "tasks_out") if not _repo_path(fz[k]).exists()]
    if missing or not old_doc.get("frozen_at"):
        raise FreezeError(f"--supersede needs both frozen files and the task file's frozen_at; missing: "
                          f"{missing or ['frozen_at in ' + fz['tasks_out']]}")
    keys = [k for k in ("sets_out", "tasks_out", "report") if _repo_path(fz[k]).exists()]
    renamed = {k: superseded_name(fz[k], old_doc["frozen_at"]) for k in keys}
    taken = [path for path in renamed.values() if _repo_path(path).exists()]
    if taken:
        raise FreezeError(f"--supersede: the renamed paths exist already, nothing is renamed: {taken}")
    return {"renames": [(fz[k], renamed[k]) for k in keys],
            "supersedes": {"frozen_at": old_doc["frozen_at"],
                           **{name: {"path": renamed[k], "sha256": file_sha256(_repo_path(fz[k]))}
                              for name, k in (("sets_file", "sets_out"), ("tasks_file", "tasks_out"))},
                           "report": renamed.get("report")}}


def superseded_document(supersedes: dict) -> dict:
    """The superseded task file that a task file's supersedes block names, after checking that both superseded data
    files exist with the recorded sha256 (FreezeError naming the first that does not)."""
    for name in ("sets_file", "tasks_file"):
        entry = supersedes[name]
        path = _repo_path(entry["path"])
        if not path.exists() or file_sha256(path) != entry["sha256"]:
            raise FreezeError(f"the superseded file {entry['path']} that the task file names is missing or was changed")
    return _read_json(_repo_path(supersedes["tasks_file"]["path"]), {})


def _carry_into(frozen: dict, cfg: dict, old_doc: dict, carried_from: str) -> None:
    """frozen["picks"] replaced by carry_picks on old_doc's picks, with picks_carried_from and picks_substitutions."""
    pk, substitutions = carry_picks(frozen["sets"], cfg, old_doc["picks"])
    frozen.update(picks=pk, picks_carried_from=carried_from, picks_substitutions=substitutions)


def _fence(text: str) -> str:
    """A backtick fence longer than any run of backticks in text, at least four long."""
    longest, run = 0, 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    return "`" * max(4, longest + 1)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def repair_rows(frozen: dict) -> list[str]:
    """One line per repaired set: task id, gap type, spans relabelled with their filler ids, kept spans and
    style_in_edit_list before -> after, H and P length before -> after against O."""
    lines = []
    for s in frozen["sets"]:
        if not s["repaired_spans"]:
            continue
        old, rep = frozen["originals"][s["task_id"]], s["repair"]
        lb, la = rep["lengths_before"], s["lengths"]
        lines.append(f"{s['task_id']} ({s['gap_type']}): spans {rep['spans']} -> fillers "
                     + ", ".join(f"{i}:{f}" for i, f in rep["filler_ids"].items())
                     + f"; kept {old['n_kept_true_spans']} -> {s['n_kept_true_spans']}; style_in_edit_list "
                     f"{old['style_in_edit_list']} -> {s['style_in_edit_list']}; O {la['O']}, "
                     f"H {lb['H']} -> {la['H']}, P {lb['P']} -> {la['P']}")
    return lines


def repairable_line(cfg: dict, audit: dict) -> str:
    """One line on the repairable reserve (verify_audit's result when freeze.repairable_reserve is configured)."""
    rest = audit["reserve_failing_in_no_list"]
    return (f"Repairable reserve (freeze.repairable_reserve; Uzair, 2026-11-08): {audit['repairable_verified']} "
            f"reserve rows verified (no on {REPAIR_COLUMN} only), each usable only after every sound reserve row of its gap "
            f"type, with its configured spans relabelled: "
            + ", ".join(f"{i} {spans}" for i, spans in cfg["freeze"]["repairable_reserve"].items())
            + f". Reserve rows with a no in none of the three lists: {len(rest)}" + (f" {rest}" if rest else ""))


def supersedes_lines(doc: dict) -> list[str]:
    """The superseded freeze a task document records (its supersedes block), one line per renamed file with its
    sha256; [] when it supersedes none."""
    sup = doc.get("supersedes")
    if not sup:
        return []
    return [f"Supersedes the freeze of {sup['frozen_at']}, kept on disk under dated names:",
            f"- sets: `{sup['sets_file']['path']}` (sha256 `{sup['sets_file']['sha256']}`)",
            f"- task file: `{sup['tasks_file']['path']}` (sha256 `{sup['tasks_file']['sha256']}`)",
            f"- report: `{sup['report']}`" if sup["report"] else "- report: none existed"]


def carried_lines(doc: dict) -> list[str]:
    """Where a task document's E2 and E7 picks were carried from and one line per substitution; [] when the picks were
    made afresh."""
    if "picks_carried_from" not in doc:
        return []
    subs = doc["picks_substitutions"]
    return ([f"E2 and E7 picks carried from `{doc['picks_carried_from']}` (a replaced task gives its place to its "
             f"replacement); substitutions: {len(subs)}" + (":" if subs else ".")]
            + [f"- {s['list']}: {s['out']} -> {s['in']} ({s['reason']})" for s in subs])


def freeze_markdown(cfg: dict, frozen: dict, doc: dict, frozen_at: str) -> str:
    """results/full_run/freeze.md: what the freeze supersedes (if it does), the audit verification, the replacements,
    the repairs table, the realised E7 split, the E2 pick with how many of the E2 pilot's tasks it holds, the carried
    picks (if they were), the counts, repaired_spans of every set, and for each print_sets id its relabelled spans (O
    text and filler) and H and P in full in backtick fences (printed from the records)."""
    fz, audit, pk, counts = cfg["freeze"], frozen["audit"], doc["picks"], doc["counts"]
    sets = {s["task_id"]: s for s in frozen["sets"]}
    unlisted = audit["reserve_failing_unlisted"]
    pilot_in = [i for i in frozen["e2_pilot"] if i in set(pk["e2_tasks"])]
    lines = [f"# Freeze of the Part 2 sets ({PART})", "",
             f"Frozen at {frozen_at}. Sets: `{doc['sets_file']}` (sha256 `{doc['sets_file_sha256']}`); task file: "
             f"`{fz['tasks_out']}`. No call was made.", ""]
    if doc.get("supersedes"):
        lines += ["## Supersedes", "", *supersedes_lines(doc), ""]
    lines += ["## Audit file against the lists (addendum 4.5)", "",
              f"`{fz['audit_csv']}`: {audit['rows']} rows ({audit['main_rows']} main, {audit['reserve_rows']} "
              f"reserve). Verified: {audit['repairs_verified']} repair rows (no on {REPAIR_COLUMN} only), "
              f"{audit['replacements_verified']} replacements (out has a no; in is the first sound reserve row of its "
              "gap type" + (", or the first repairable one when none is left" if "repairable_verified" in audit
                            else "")
              + f"), {audit['do_not_use_verified']} do-not-use reserve rows (notes say \"do not use\"). Main rows "
              f"with a no: {len(audit['main_failing'])}, each a repair or a replacement.", "",
              f"Reserve rows with a no in neither list (for the record): {len(unlisted)}" + (":" if unlisted else ""),
              *[f"- {i}" for i in unlisted], ""]
    if "repairable_verified" in audit:
        lines += [repairable_line(cfg, audit), ""]
    lines += ["## Replacements", "", "| out | in | gap type | reserve rank | tier |", "|---|---|---|---|---|",
              *[f"| {r['out']} | {r['in']} | {r['gap_type']} | {r['reserve_rank']} | "
                f"{r.get('replacement_tier', 'sound')} |" for r in doc["replacements"]], "",
              "## Repairs (addendum 4.1)", "",
              "| task id | gap type | spans relabelled | filler ids | kept before -> after | style_in_edit_list before "
              "-> after | O | H before -> after | P before -> after |", "|---|---|---|---|---|---|---|---|---|"]
    for r in doc["repairs"]:
        old, s = frozen["originals"][r["task_id"]], sets[r["task_id"]]
        lines.append(f"| {r['task_id']} | {r['gap_type']} | {r['spans']} | "
                     + ", ".join(f"{i}: {f}" for i, f in r["filler_ids"].items())
                     + f" | {r['kept_before']} -> {r['kept_after']} | {old['style_in_edit_list']} -> "
                     f"{s['style_in_edit_list']} | {r['lengths_after']['O']} | {r['lengths_before']['H']} -> "
                     f"{r['lengths_after']['H']} | {r['lengths_before']['P']} -> {r['lengths_after']['P']} |")
    lines += ["", f"Repaired sets: {counts['repaired_sets']}; spans relabelled: {counts['repaired_spans_total']}. "
              f"Reason recorded on each span: {fz['repair_reason']}", "",
              "## Picks (recorded, not run)", "",
              f"- E7 ({len(pk['e7_tasks'])} shown-eligible tasks, seed {cfg['seed']}, one per gap type in turn): "
              f"realised split {pk['e7_per_gap']} (planned {cfg['e7']['per_gap']} per gap type; a gap type with fewer "
              "eligible tasks gives its place to the others).",
              f"- E2 ({len(pk['e2_tasks'])} tasks, seed {cfg['e2']['seed']}): per gap type "
              f"{dict(Counter(sets[i]['gap_type'] for i in pk['e2_tasks']))}; shown-eligible "
              f"{len(pk['e2_shown_eligible'])}; holds {len(pilot_in)} of the E2 pilot's {len(frozen['e2_pilot'])} "
              f"tasks ({fz['e2_pilot_samples']}).",
              f"- Shown-eligible tasks: {len(pk['shown_eligible'])}; band tasks "
              f"({cfg['shown']['max_transcript_tokens']}, {cfg['shown']['upper_band_tokens']}]: "
              f"{len(pk['band_tasks'])}.",
              f"- Excluded check-d sets ({fz['checkd_unexplained']}): {len(frozen['checkd'])}.", ""]
    if "picks_carried_from" in doc:
        lines += ["## Carried picks", "", *carried_lines(doc), ""]
    lines += ["## Counts", "", "```", json.dumps(counts, indent=2), "```", "",
              "## repaired_spans per set", "", "| task id | gap type | repaired_spans | kept after |",
              "|---|---|---|---|",
              *[f"| {s['task_id']} | {s['gap_type']} | {s['repaired_spans']} | {s['n_kept_true_spans']} |"
                for s in frozen["sets"]]]
    for task_id in fz["print_sets"]:
        s = sets[task_id]
        lines += ["", f"## {task_id}: rebuilt H and P (addendum 4.6c)", "",
                  f"repaired_spans: {s['repaired_spans']}; kept spans after the repair: {s['n_kept_true_spans']}.", "",
                  "| span | O text | filler in H and P |", "|---|---|---|"]
        for span in s["spans"]:
            if span.get("repair"):
                lines.append(f"| {span['index']} | {_cell(span['text'])} | {span['filler_ids'][0]}: "
                             f"{_cell(span['p_text'])} |")
        for name in ("H", "P"):
            fence = _fence(s[name])
            lines += ["", f"{name}:", "", fence, s[name], fence]
    return "\n".join(lines) + "\n"


def _freeze_failed(ctx: Ctx6, exc: Exception) -> int:
    """Log a failure of the freeze before anything is written; record it as failed unless this is a dry run."""
    logger.error("freeze: FAILED: %s: %s", type(exc).__name__, exc,
                 exc_info=None if isinstance(exc, (FreezeError, repair_v4.RepairError)) else exc)
    if not ctx.dry_run:
        _record_stage(ctx, "freeze", {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    return EXIT_FAILED


def _freeze_refused(ctx: Ctx6, reason: str) -> int:
    """Log a refusal (the frozen files on disk stand, nothing is written); a "done" entry of stages_status.json stays,
    with "refused", any other entry becomes failed."""
    logger.error("freeze: REFUSED: %s", reason)
    previous = _read_json(ctx.run_dir / "stages_status.json", {}).get("freeze") or {}
    _record_stage(ctx, "freeze", {**previous, "refused": reason} if previous.get("status") == "done"
                  else {"status": "failed", "reason": reason})
    return EXIT_FAILED


def stage_freeze(ctx: Ctx6) -> int:
    """Brief Step 8 (addendum sections 4 and 7, Step 1): compute_freeze (steps 1 to 6, read only), then tasks_document,
    then the write: sets_out (the frozen sets in main order) and tasks_out ({"frozen_at", **tasks_document}), then the
    report (freeze_markdown). The first failing step stops the stage with its message (EXIT_FAILED, recorded as
    failed) and nothing is written: the texts of the two data files and of the report are all built before the first
    write, and a failure while building or writing them is recorded as failed too (each file is written whole through
    a temporary file; an OS error on the second data file leaves the first, which a rerun finds equal and keeps).

    The task file on disk says how its picks were made: when it has a supersedes block (a re-freeze), the picks are
    carried (carry_picks) from the superseded task file it names (superseded_document checks both superseded data
    files against their recorded sha256) and the block is kept, so a rerun reproduces it; otherwise they are picked
    afresh as compute_freeze does. When an output exists and differs from what would be written (output_differences):
    without ctx.supersede nothing is written and the stage fails naming the first difference (a "done" entry of
    stages_status.json stays, with "refused"); with ctx.supersede (Uzair, 2026-11-08) the picks are carried from the
    task file on disk instead, supersede_plan names the renamed paths (a refusal when they exist or a frozen file is
    missing), every text is built, then sets_out, tasks_out and the report are renamed (never deleted or overwritten)
    and the new freeze is written with a new frozen_at and the supersedes block; the stage entry lists the renamed
    paths under "superseded". When both outputs exist and equal what would be written, the stage says so, rewrites
    only the report (with the frozen_at on disk) and exits 0, flag or not. Dry run: steps 1 to 6 run and the plan
    (the audit verification, the replacements, the repairs, the picks and how they were carried, the renames a
    supersede would make, what would be written and where) goes to ctx.plan; nothing is written and nothing is
    recorded."""
    cfg, fz = ctx.cfg, ctx.cfg["freeze"]
    tasks_path = _repo_path(fz["tasks_out"])
    try:
        frozen = compute_freeze(cfg)
        on_disk = _read_json(tasks_path, {}) if tasks_path.exists() else {}
        if on_disk.get("supersedes"):   # the freeze on disk is a re-freeze: reproduce its carried picks
            sup = on_disk["supersedes"]
            _carry_into(frozen, cfg, superseded_document(sup), sup["tasks_file"]["path"])
            frozen["supersedes"] = sup
        doc = tasks_document(cfg, frozen)
    except Exception as exc:   # FreezeError, repair_v4.RepairError, ValueError from a pick or a loader
        return _freeze_failed(ctx, exc)
    differences, existing = output_differences(cfg, frozen["sets"], doc)
    plan = None
    if differences and ctx.supersede:
        try:
            plan = supersede_plan(cfg, on_disk)
            frozen["supersedes"] = plan["supersedes"]
            _carry_into(frozen, cfg, on_disk, plan["supersedes"]["tasks_file"]["path"])
            doc = tasks_document(cfg, frozen)
        except Exception as exc:
            reason = f"--supersede: {type(exc).__name__}: {exc}; nothing is renamed or written"
            if ctx.dry_run:
                logger.error("freeze: FAILED: %s", reason)
                return EXIT_FAILED
            return _freeze_refused(ctx, reason)
    audit, pk = frozen["audit"], doc["picks"]
    if ctx.dry_run:
        ctx.note(f"freeze: audit {fz['audit_csv']}: {audit['rows']} rows ({audit['main_rows']} main, "
                 f"{audit['reserve_rows']} reserve) match the lists: {audit['repairs_verified']} repairs, "
                 f"{audit['replacements_verified']} replacements, {audit['do_not_use_verified']} do-not-use rows; "
                 f"{len(audit['main_failing'])} main rows with a no, all listed; reserve rows with a no in neither "
                 f"list: {len(audit['reserve_failing_unlisted'])} {audit['reserve_failing_unlisted']}")
        if "repairable_verified" in audit:
            ctx.note("freeze: " + repairable_line(cfg, audit))
        for r in doc["replacements"]:
            ctx.note(f"freeze: replacement {r['out']} -> {r['in']} ({r['gap_type']}, reserve rank "
                     f"{r['reserve_rank']}" + (f", {r['replacement_tier']} tier" if "replacement_tier" in r else "")
                     + ")")
        for line in repair_rows(frozen):
            ctx.note("freeze: repair " + line)
        counts = doc["counts"]
        ctx.note(f"freeze: {counts['sets']} sets {counts['per_gap']}; {counts['repaired_sets']} repaired, "
                 f"{counts['repaired_spans_total']} spans relabelled; E6 O_plus and O_minus built for every set; E4 "
                 f"resolved {counts['resolved']} {counts['resolved_per_gap']}, model_patch matches the trajectory's "
                 f"for every set; task files "
                 f"{dict(Counter(frozen['task_files'][s['task_id']] for s in frozen['sets']))}")
        pilot_in = sum(i in set(pk["e2_tasks"]) for i in frozen["e2_pilot"])
        ctx.note(f"freeze: picks: E2 {len(pk['e2_tasks'])} tasks ({len(pk['e2_shown_eligible'])} shown-eligible, "
                 f"{pilot_in} of the pilot's {len(frozen['e2_pilot'])}); E7 {len(pk['e7_tasks'])} tasks, split "
                 f"{pk['e7_per_gap']}; shown-eligible {len(pk['shown_eligible'])}; band {len(pk['band_tasks'])}; "
                 f"check-d sets excluded {len(frozen['checkd'])}")
        for line in carried_lines(doc):
            ctx.note("freeze: " + line)
        for old, new in plan["renames"] if plan else []:
            ctx.note(f"freeze: --supersede: would rename {old} -> {new}")
        for line in supersedes_lines(doc):
            ctx.note("freeze: " + line)
        state = ("DIFFER (" + "; ".join(differences) + "); --supersede: they would be renamed as above and the new "
                 "freeze written" if plan else
                 "DIFFER, the stage would refuse: " + "; ".join(differences) if differences else
                 f"{existing} exist and are unchanged" if existing else "none exist yet")
        ctx.note(f"freeze: would write {fz['sets_out']} ({len(frozen['sets'])} records, sha256 "
                 f"{doc['sets_file_sha256']}), {fz['tasks_out']} and {fz['report']}; outputs: {state}")
        return EXIT_OK
    if differences and not plan:
        return _freeze_refused(ctx, "the frozen files on disk differ from the records computed now ("
                               + "; ".join(differences) + "); nothing is written. To freeze again, rerun with "
                               "--supersede (the files on disk are then kept under dated names)")
    sets_path = _repo_path(fz["sets_out"])
    unchanged = not plan and len(existing) == 2
    if unchanged:
        logger.info("freeze: %s and %s are unchanged; nothing to write", fz["sets_out"], fz["tasks_out"])
    renamed: list[str] = []
    try:   # every text is built before the first rename or write, so a failure while building them changes nothing
        frozen_at = (_read_json(tasks_path, {}).get("frozen_at") if tasks_path.exists() and not plan
                     else datetime.now(timezone.utc).isoformat(timespec="seconds"))
        writes: list[tuple[Path, str]] = []
        if plan or not sets_path.exists():
            writes.append((sets_path, _jsonl_text(frozen["sets"])))
        if plan or not tasks_path.exists():
            writes.append((tasks_path,
                           json.dumps({"frozen_at": frozen_at, **doc}, indent=2, ensure_ascii=False) + "\n"))
        writes.append((_repo_path(fz["report"]), freeze_markdown(cfg, frozen, doc, frozen_at)))
        for old, new in plan["renames"] if plan else []:
            _repo_path(old).rename(_repo_path(new))
            renamed.append(f"{old} -> {new}")
        for path, text in writes:
            _write_text(path, text)
    except Exception as exc:
        logger.error("freeze: FAILED before or while writing: %s: %s (renamed so far: %s)", type(exc).__name__, exc,
                     renamed, exc_info=exc)
        _record_stage(ctx, "freeze", {"status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                                      **({"renamed_before_the_failure": renamed} if renamed else {})})
        return EXIT_FAILED
    entry = {"status": "done", "unchanged": unchanged, "frozen_at": frozen_at, "sets": doc["counts"]["sets"],
             "repaired_sets": doc["counts"]["repaired_sets"],
             "repaired_spans_total": doc["counts"]["repaired_spans_total"], "replacements": len(doc["replacements"]),
             "sets_file_sha256": doc["sets_file_sha256"]}
    sup = doc.get("supersedes")
    if sup:
        entry["superseded"] = [p for p in (sup["sets_file"]["path"], sup["tasks_file"]["path"], sup["report"]) if p]
    _record_stage(ctx, "freeze", entry)
    logger.info("freeze: %s; %d sets, %d repaired (%d spans), %d replaced -> %s, %s, %s",
                "unchanged" if unchanged else "superseded and written" if plan else "written", doc["counts"]["sets"],
                doc["counts"]["repaired_sets"], doc["counts"]["repaired_spans_total"], len(doc["replacements"]),
                fz["sets_out"], fz["tasks_out"], fz["report"])
    if unchanged:
        print(f"[freeze] unchanged: {fz['sets_out']} and {fz['tasks_out']} already hold these records", flush=True)
    if plan:
        print(f"[freeze] superseded: {'; '.join(renamed)}; the new freeze is in {fz['sets_out']} and {fz['tasks_out']}",
              flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# The stage loop
# ---------------------------------------------------------------------------------------------


_STAGE_FUNCS: dict[str, Callable[[Ctx6], int]] = {"freeze": stage_freeze}


def _register_stages() -> None:
    """Register the Part 2 stages of the sibling modules in _STAGE_FUNCS (idempotent). Called by run(), not at import
    time: the sibling modules import this module, and `python -m` imports it a second time under its real name. Every
    STAGES entry must be registered by the end."""
    from overclaim.experiments import (full_run_analysis, full_run_e2, full_run_pairwise, full_run_pointwise,
                                       full_run_shown)

    _STAGE_FUNCS.update(full_run_pointwise.STAGE_FUNCS)
    _STAGE_FUNCS["pointwise_shown"] = full_run_shown.stage_pointwise_shown
    _STAGE_FUNCS["e2"] = full_run_e2.stage_e2
    _STAGE_FUNCS["pairwise"] = full_run_pairwise.stage_pairwise
    _STAGE_FUNCS["reasons"] = full_run_pairwise.stage_reasons
    _STAGE_FUNCS["analysis"] = full_run_analysis.stage_analysis
    unregistered = [name for name in STAGES if name not in _STAGE_FUNCS]
    if unregistered:
        raise RuntimeError(f"full_run: stages without a function: {unregistered}")


def run(cfg: dict, cfg_path: Path, run_dir: Path, stage: str, dry_run: bool, supersede: bool = False,
        only: Collection[str] | None = None) -> int:
    """Run one stage or all of them (STAGES order) on a Ctx6 (supersede: the CLI's --supersede, read by the freeze
    stage; only: the CLI's --only, the run ids the pointwise_shown or pairwise stage runs); returns the exit code.
    Writes run_dir/config.yaml (the loaded config with source_config and resolved_at)."""
    if stage != "all" and stage not in STAGES:
        logger.error("unknown full_run stage %r; expected one of %s", stage, STAGES)
        return EXIT_FAILED
    if only is not None and stage not in ("pointwise_shown", "pairwise"):
        logger.error("--only applies to the pointwise_shown and pairwise stages, not %r", stage)
        return EXIT_FAILED
    _register_stages()   # at run time, not import time: the sibling modules import this one (python -m imports it twice)
    ctx = Ctx6(cfg, cfg_path, run_dir, dry_run, supersede=supersede, only=only)
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
        print("=== full_run dry-run plan (no API calls, nothing written) ===")
        for line in ctx.plan:
            print(line)
    elif ctx._cost is not None:
        print(ctx.cost.totals_line())
    return EXIT_OK


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Session 5, Part 2: the full run on the frozen coding sets.")
    parser.add_argument("--config", required=True, help="configs/full_run.yaml")
    parser.add_argument("--stage", choices=[*STAGES, "all"], required=True)
    parser.add_argument("--dry-run", action="store_true", help="print the plan; no API calls, nothing written")
    parser.add_argument("--supersede", action="store_true",
                        help="freeze: when the frozen files differ from the records computed now, rename them to "
                             "<stem>.superseded-<their frozen_at> and write the new freeze (E2 and E7 picks carried "
                             "over); without it such a run is refused")
    parser.add_argument("--results-root", default=None,
                        help="directory that holds <run_name>/ (default: <repo>/results)")
    parser.add_argument("--only", default=None,
                        help="pointwise_shown or pairwise: comma-separated run ids to run this invocation; the other "
                             "runs are left untouched (their status entries are not rewritten)")
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
        logger.info("run %s (full_run): run_dir %s, stage %s, dry_run %s, supersede %s", cfg["run_name"], run_dir,
                    args.stage, args.dry_run, args.supersede)
        only = None if args.only is None else [x.strip() for x in args.only.split(",") if x.strip()]
        return run(cfg, cfg_path, run_dir, args.stage, args.dry_run, supersede=args.supersede, only=only)
    finally:
        root = logging.getLogger()
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    sys.exit(main())

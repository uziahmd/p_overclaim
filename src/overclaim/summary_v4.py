"""Pilot 4 summary (config schema_version 4, design v4): run_dir/summary.md and run_dir/summary.json.

`summarize_pilot4` reads the files the pilot-4 runner (overclaim.experiments.pilot4) writes into the run directory, the
v4 sets file (cfg data.processed_dir / sets.v4.file) and pilot 3's judge records (cfg session3_dir, read only). It
writes summary.md and summary.json and nothing else. Every input may be missing: a section whose file is missing says
"Not run" with the reason that stages_status.json or runs_status.json records, and never raises for it. Every number in
the summary is computed here from one of those files (or read from one); rates and CIs are formatted as in
analysis._fmt_ci and tables are GitHub markdown.

Sections, in order: (1) header with the realised split of the v4 sets per gap type, then the carry-over files from
pilot 3 (the reason table and the decided rates included verbatim); (2) the v4 funnel (rebuild of the pilot-3 tasks,
top-up and rebalancing, outcomes, verifier checks, per-set fields, cost of the sets stage); (3) the pairwise judge runs
on the v4 sets (hidden: rates, the v3 comparison on the rebuilt tasks, gap types, provenance; shown: task list, rates,
the same tasks' hidden rates; reason categories); (4) the pointwise pilot (Experiment 1b; the shown records of the judge
run with each task's first call and cost); (5) the Experiment 2 policy pilot, both labellers side by side; (6) the
three-label Experiment 3 check; (7) spend; (8) the caller's extra sections.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from overclaim import analysis
from overclaim.analysis import (COST_FILE, E2_SUMMARY_FILE, E3_SUMMARY_FILE, FUNNEL_FILE, JUDGE_FILE, NA,
                                REASON_LABELS_FILE, RM_FILE, STAGES_STATUS_FILE, _flat, _md_table, _nan_to_none,
                                _not_run_line, _read_json_file, _reasons_cell)
from overclaim.experiments.pilot2 import _repo_path

logger = logging.getLogger(__name__)

SET = "v4"
V3_SET = "v3"
GAP_TYPES = analysis.GAP_TYPES
RUNS_STATUS_FILE = "runs_status.json"
POINTWISE_FILE = "pointwise_calls.jsonl"          # e1b_pointwise records (pointwise and pointwise_late stages, hooks)
BUILD_FILE = "v4_build.jsonl"                     # one build info per task tried (sets stage)
REBUILD_FILE = "v4_rebuild_report.json"           # one entry per pilot-3 task rebuilt first (sets stage)
PRIOR_SPEND_FILE = "prior_spend.json"             # spend before this session (carryover stage)
BUDGET_STOP_FILE = "BUDGET_STOP.txt"              # written by pilot2._budget_stop when a client cap stopped a stage
# cost.jsonl purposes of the sets stage, the `purpose` defaults of variants_inplace.detect_spans,
# variants_v3.annotate_spans, verify_v4.precheck_original, variants_v4.rewrite_spans_v4 and verify_v4.verify_set_v4
# (the set pilot4.sets_purposes reads from them; tests/test_summary_v4.py checks that the two agree).
SETS_PURPOSES = ("span_detection", "span_annotation", "v4_precheck", "v4_span_rewrite", "v4_verification")
# The v4 verifier checks (verify_v4): b is asked in the pre-check of the original message, c, d and f in the post-check.
PRE_CHECKS_V4 = ("b_original_has_no_disclosure",)
POST_CHECKS_V4 = ("c_disclosure_matches_record", "d_no_outcome_claims", "f_no_gap_mentions")
# The pointwise prompt has the neutral rubric's wording, so the pairwise rates it is compared with are the neutral ones.
PAIRWISE_RUBRIC = "neutral"
TASK_RULES = {"no_tests_run_or_primary_changed": "the v4 sets at or below the shown cap whose gap type is "
                                                 "no_tests_run or whose primary span changed from v3"}
# evidence_v4.PRE_EXISTING; None (no_tests_run names no failing test) is counted as "not_applicable", as the v4
# funnel (variants_v4) counts it.
NAMED_FAILURE_VALUES = ("yes", "no", "unknown")
CLAIM_LABELS = ("supported", "unsupported", "contradicted")   # the three labels of e3_verification.VF3_SCHEMA
SOURCE_PILOT4 = "pilot 4 (v4 sets)"
SOURCE_PILOT3 = "pilot 3 (v3 sets)"
# The files the carryover stage (pilot4.stage_carryover) writes into the run directory; the two .md files are
# included verbatim in the summary.
CARRYOVER_FILES = ("prior_spend.json", "pilot3_decided_rates.json", "pilot3_decided_rates.md",
                   "pilot3_opus46_reason_labels.jsonl", "pilot3_reason_table.json", "pilot3_reason_table.md")
CARRYOVER_VERBATIM = ("pilot3_reason_table.md", "pilot3_decided_rates.md")
# The two labellers of Experiment 2 (e2_selection.run_e2_packet): the key suffix of their results in e2_summary.json
# and their name in the summary.
E2_LABELLERS = (("", "test-record labeller"), ("_full", "full-evidence labeller"))


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------


def _load(path: Path) -> list[dict] | None:
    """The records of a JSONL file, or None when the file is missing (a malformed line raises)."""
    return analysis._read_jsonl(path) if path.exists() else None


def _yes_no(value: Any) -> str:
    return NA if value is None else ("yes" if value else "no")


def _esc(value: Any) -> str:
    """A table cell: the value as text with '|' escaped and line breaks as spaces."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _usd(x: float | None, digits: int = 3) -> str:
    return NA if x is None else f"{x:.{digits}f}"


def _lines(markdown: str | list[str] | None) -> list[str]:
    """Markdown as lines (the analysis functions return either a string or a list of lines), without a leading '#'
    title of its own: the summary's heading replaces it."""
    lines = (markdown.rstrip("\n").split("\n") if isinstance(markdown, str) else list(markdown or []))
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return lines


def _gap_order(gaps: Any) -> list[str]:
    """The three gap types, then any other gap type present, sorted."""
    return [*GAP_TYPES, *sorted(set(gaps) - set(GAP_TYPES))]


def _not_done(status: dict, ids: list[str]) -> list[str]:
    """One line per run id whose runs_status.json entry is not "done" (or that has no entry)."""
    lines = []
    for rid in ids:
        e = status.get(rid)
        if e is None:
            lines.append(f"Run {rid}: no entry in {RUNS_STATUS_FILE}.")
        elif e.get("status") != "done":
            lines.append(f"Run {rid} ({e.get('judge')}, {e.get('condition')}): {e.get('status')}, reason: "
                         f"{e.get('reason') or 'not recorded'}.")
    return lines


def _non_smoke(records: list[dict]) -> list[dict]:
    return [r for r in records if not r.get("smoke_only")]


def _run_records(records: list[dict], run: dict) -> list[dict]:
    """The judge records of one cfg["runs"] entry: its grader label, set, rubric and condition."""
    label = run.get("grader_label", run["judge"])
    return [r for r in records if r["grader"] == label and analysis.record_set(r) == run["set"]
            and analysis.record_rubric(r) == run["rubric"] and r["condition"] == run["condition"]]


def _import(name: str) -> Any:
    """A sibling analysis module, imported when a section needs it (overclaim.analysis_v4, analysis_reasons_v4)."""
    import importlib

    return importlib.import_module(f"overclaim.{name}")


# ---------------------------------------------------------------------------------------------
# Counts (pure functions over the records)
# ---------------------------------------------------------------------------------------------


def sets_per_gap(sets: list[dict]) -> dict[str, int]:
    """Sets per gap type: the three gap types always (0 when none), then any other gap type present, sorted."""
    counts = Counter(str(s.get("gap_type")) for s in sets)
    return {g: counts.get(g, 0) for g in _gap_order(counts)}


def rebuild_counts(report: list[dict]) -> dict:
    """Counts over the entries of v4_rebuild_report.json: n_tasks, built (v4_outcome "built"), primary_changed
    (entries with primary_changed True) and not_built {v4_outcome: n} (every other outcome, excluded ones included)."""
    return {"n_tasks": len(report), "built": sum(e.get("v4_outcome") == "built" for e in report),
            "primary_changed": sum(e.get("primary_changed") is True for e in report),
            "not_built": dict(Counter(str(e.get("v4_outcome")) for e in report if e.get("v4_outcome") != "built"))}


def build_outcomes(infos: list[dict]) -> dict[str, dict]:
    """Outcomes of the tasks tried (v4_build.jsonl), per gap type and "all": tried, built, excluded {reason: n},
    dropped {reason: n} (an outcome is "built", "excluded:<reason>" or "dropped:<reason>"; any other outcome string
    is counted under dropped as itself)."""
    def count(group: list[dict]) -> dict:
        out: dict[str, Any] = {"tried": len(group), "built": 0, "excluded": Counter(), "dropped": Counter()}
        for info in group:
            outcome = str(info.get("outcome"))
            if outcome == "built":
                out["built"] += 1
            elif outcome.startswith("excluded:"):
                out["excluded"][outcome.split(":", 1)[1]] += 1
            else:
                out["dropped"][outcome.split(":", 1)[1] if outcome.startswith("dropped:") else outcome] += 1
        return {**out, "excluded": dict(out["excluded"]), "dropped": dict(out["dropped"])}

    by_gap: dict[str, list[dict]] = defaultdict(list)
    for info in infos:
        by_gap[str(info.get("gap_type"))].append(info)
    rows = {g: count(by_gap.get(g, [])) for g in _gap_order(by_gap)}
    rows["all"] = count(infos)
    return rows


def set_counts(sets: list[dict]) -> dict[str, dict]:
    """Per gap type (sets_per_gap's order) and "all", over the v4 sets: sets, kept_true_spans (sum of
    n_kept_true_spans), sets_with_kept_span, shown_eligible, p3_has_model_rewrite, style_in_edit_list_sets (sets whose
    style_in_edit_list count is above 0), style_in_edit_list_items (sum of the counts), primary_in_edit_list, and
    named_failure_pre_existing {"yes", "no", "unknown", "not_applicable"} (the field is None)."""
    def count(group: list[dict]) -> dict:
        named = Counter(str(s.get("named_failure_pre_existing")) if s.get("named_failure_pre_existing") is not None
                        else "not_applicable" for s in group)
        return {"sets": len(group),
                "kept_true_spans": sum(int(s.get("n_kept_true_spans") or 0) for s in group),
                "sets_with_kept_span": sum(int(s.get("n_kept_true_spans") or 0) > 0 for s in group),
                "shown_eligible": sum(s.get("shown_eligible") is True for s in group),
                "p3_has_model_rewrite": sum(s.get("p3_has_model_rewrite") is True for s in group),
                "style_in_edit_list_sets": sum(int(s.get("style_in_edit_list") or 0) > 0 for s in group),
                "style_in_edit_list_items": sum(int(s.get("style_in_edit_list") or 0) for s in group),
                "primary_in_edit_list": sum(s.get("primary_in_edit_list") is True for s in group),
                "named_failure_pre_existing": {k: named.get(k, 0) for k in (*NAMED_FAILURE_VALUES,
                                                                             "not_applicable")}}

    by_gap: dict[str, list[dict]] = defaultdict(list)
    for s in sets:
        by_gap[str(s.get("gap_type"))].append(s)
    rows = {g: count(by_gap.get(g, [])) for g in _gap_order(by_gap)}
    rows["all"] = count(sets)
    return rows


def kept_span_counts(sets: list[dict]) -> dict:
    """Kept true spans over the v4 sets: distribution {n kept spans: n sets} (ascending, keys as strings), total, and
    by_supported_by {supported_by: n spans} over every set's kept_true_spans (most frequent first)."""
    dist = Counter(int(s.get("n_kept_true_spans") or 0) for s in sets)
    by = Counter(str(sp.get("supported_by")) for s in sets for sp in s.get("kept_true_spans") or [])
    return {"distribution": {str(k): dist[k] for k in sorted(dist)},
            "total": sum(int(s.get("n_kept_true_spans") or 0) for s in sets),
            "by_supported_by": dict(sorted(by.items(), key=lambda kv: (-kv[1], kv[0])))}


def stage_cost(cost_rows: list[dict], purposes: tuple[str, ...]) -> dict:
    """The live calls of cost.jsonl whose purpose is in `purposes`: rows per (purpose, provider, model), purposes in
    the given order, each with calls, usd (sum of the non-null cost_usd) and n_unpriced (null cost_usd); and the
    totals calls, usd, n_unpriced."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in cost_rows:
        if r.get("purpose") in purposes:
            groups[(str(r["purpose"]), str(r.get("provider")), str(r.get("model")))].append(r)

    def spend(recs: list[dict]) -> dict:
        priced = [float(r["cost_usd"]) for r in recs if r.get("cost_usd") is not None]
        return {"calls": len(recs), "usd": sum(priced), "n_unpriced": len(recs) - len(priced)}

    keys = sorted(groups, key=lambda k: (purposes.index(k[0]), k[1], k[2]))
    rows = [{"purpose": p, "provider": prov, "model": m, **spend(groups[(p, prov, m)])} for p, prov, m in keys]
    return {"rows": rows, **spend([r for k in keys for r in groups[k]])}


# ---------------------------------------------------------------------------------------------
# Section 1: header
# ---------------------------------------------------------------------------------------------


def _header(run: Path, cfg: dict, sets: list[dict] | None, sets_path: Path, cost_rows: list[dict]) -> tuple[list, dict]:
    split = sets_per_gap(sets) if sets is not None else None
    lines = [f"# Pilot 4 summary: {cfg['run_name']}", ""]
    if sets is None:
        lines.append(f"v4 sets: none. {_not_run_line(run, 'sets', sets_path.name)}")
    else:
        lines.append(f"v4 sets: {len(sets)} built ({', '.join(f'{g} {n}' for g, n in split.items())}).")
    stamps = sorted(str(r["ts"]) for r in cost_rows if r.get("ts"))
    if stamps:
        lines.append(f"Dates: the calls of this run were logged from {stamps[0][:10]} to {stamps[-1][:10]} (UTC; first "
                     f"and last ts of {COST_FILE}).")
    else:
        lines.append(f"Dates: no call is logged in {COST_FILE}.")
    lines += ["", "A pilot validates the matched sets and the pipeline and sizes the full run. It does not test the "
              "pre-registered hypotheses H1 to H4: the rates and intervals below describe this pilot's tasks and are "
              "not tests of them."]
    data = {"run_name": cfg["run_name"], "sets_file": str(sets_path), "n_sets": None if sets is None else len(sets),
            "split": split, "first_ts": stamps[0] if stamps else None, "last_ts": stamps[-1] if stamps else None}
    return lines, data


def _carryover_section(run: Path) -> tuple[list[str], dict]:
    """The carry-over files of the run directory (CARRYOVER_FILES) that exist, then each of CARRYOVER_VERBATIM that
    exists under its own heading, its text verbatim (only the line breaks at its end removed)."""
    lines = ["## Carry-over from pilot 3", ""]
    present = [name for name in CARRYOVER_FILES if (run / name).exists()]
    missing = [name for name in CARRYOVER_FILES if name not in present]
    if not present:
        return lines + [_not_run_line(run, "carryover", f"every carry-over file ({', '.join(CARRYOVER_FILES)})")], {
            "present": [], "missing": missing}
    lines.append("Written by the carryover stage from pilot 3's records (read only) and the earlier cost logs: "
                 f"{', '.join(present)}." + (f" Missing: {', '.join(missing)}." if missing else ""))
    for name in CARRYOVER_VERBATIM:
        if name in present:
            lines += ["", f"### {name}", "", (run / name).read_text(encoding="utf-8").rstrip("\n")]
    return lines, {"present": present, "missing": missing}


# ---------------------------------------------------------------------------------------------
# Section 2: v4 sets, funnel
# ---------------------------------------------------------------------------------------------


def _rebuild_lines(run: Path, report: list[dict] | None) -> tuple[list[str], Any]:
    lines = ["### Rebuild of the pilot-3 tasks", ""]
    if report is None:
        return lines + [_not_run_line(run, "sets", REBUILD_FILE)], None
    c = rebuild_counts(report)
    lines += [f"The {c['n_tasks']} tasks that had a v3 set were rebuilt first under design v4: {c['built']} have a v4 "
              f"set, {c['primary_changed']} of them with a primary span whose text differs from v3's. Not built: "
              f"{_esc(_reasons_cell(c['not_built']))}. Audited = read against the transcript in the pilot-3 audit; "
              "second look = passed only on the appeal verifier in pilot 3; the last column says whether a primary "
              "span would exist under the relaxed rule (any claim type), n/a where that rule does not apply.", ""]
    table = [[_esc(e.get("task_id")), _esc(e.get("gap_type")), _yes_no(e.get("audited")), _yes_no(e.get("second_look")),
              _esc(e.get("v4_outcome")), _yes_no(e.get("primary_changed")),
              NA if e.get("n_kept_true_spans") is None else str(e["n_kept_true_spans"]),
              NA if e.get("named_failure_pre_existing") is None else _esc(e["named_failure_pre_existing"]),
              _yes_no(e.get("relaxed_primary_exists"))] for e in report]
    lines += _md_table(["task", "gap type", "audited", "second look", "v4 outcome", "primary changed",
                        "kept true spans", "named failure pre-existing", "primary under the relaxed rule"], table)
    details = [f"- {e.get('task_id')}: {_esc(e['v4_detail'])}" for e in report if e.get("v4_detail")]
    if details:
        lines += ["", "Reasons given for the outcomes:", "", *details]
    return lines, {"counts": c, "entries": report}


def _topup_lines(cfg: dict, funnel: dict) -> list[str]:
    v4 = funnel.get("v4") or {}
    pools = funnel.get("v4_candidate_pool") or {}
    by_gap = v4.get("by_gap") or {}
    lines = ["### Top-up and rebalancing", ""]
    if not by_gap:
        return lines + [f"No v4 funnel ({FUNNEL_FILE} has no 'v4' entry)."]
    pilot = cfg.get("pilot") or {}
    lines += [f"Per gap type: the rebuilt tasks first, then candidates until the target, the pool's end, the cap "
              f"pilot.max_new_per_gap or the budget guard (the stage's own Anthropic spend reaching "
              f"{pilot.get('sets_stop_usd', NA)} USD: 'stopped'). Candidates = the candidate order of the main pool "
              "file, then of the extra pool file, without the held-out tasks.", ""]

    def pool_cell(size: dict | None) -> str:
        if not size:
            return NA
        return (f"{_flat(size.get('candidates'))} (main {_flat(size.get('pool_file'))}, extra "
                f"{_flat(size.get('extra_pool_file'))}, held out {_flat(size.get('held_out_removed'))})")

    rows = [[gap, _flat(g.get("target")), _flat(g.get("first_tried")), _flat(g.get("first_built")),
             pool_cell(pools.get(gap)), _flat(g.get("candidates_consumed")), _flat(g.get("candidates_built")),
             _flat(g.get("built")), _flat(g.get("pool_exhausted")), _flat(g.get("cap_reached")),
             _flat(g.get("stopped"))] for gap, g in by_gap.items()]
    lines += _md_table(["gap type", "target", "first tried", "first built", "candidates", "candidates consumed",
                        "candidates built", "built", "pool exhausted", "cap reached", "stopped"], rows)
    extra = funnel.get("extra_pool")
    lines += ["", f"Extra pool ({extra.get('file')}, row groups {_flat(extra.get('row_groups'))}, gap types "
              f"{_flat(extra.get('gap_types'))}): {_flat(extra.get('n_written'))} tasks written." if extra
              else f"Extra pool: no counts in {FUNNEL_FILE}."]
    rb = v4.get("rebalance")
    if rb:
        lines.append(f"Rebalancing (pilot.rebalance {pilot.get('rebalance', NA)}): applied {_flat(rb.get('applied'))}, "
                     f"passes {_flat(rb.get('passes'))}; final targets {_flat(rb.get('targets_final'))}; realised "
                     f"{_flat(rb.get('realised'))}.")
    else:
        lines.append(f"Rebalancing: not recorded in {FUNNEL_FILE}.")
    return lines


def _outcome_lines(run: Path, infos: list[dict] | None) -> tuple[list[str], Any]:
    lines = ["### Outcomes of every task tried", ""]
    if infos is None:
        return lines + [_not_run_line(run, "sets", BUILD_FILE)], None
    rows = build_outcomes(infos)
    lines += _md_table(["gap type", "tried", "built", "excluded (reason n)", "dropped (reason n)"],
                       [[g, str(r["tried"]), str(r["built"]), _esc(_reasons_cell(r["excluded"])),
                         _esc(_reasons_cell(r["dropped"]))] for g, r in rows.items()])
    return lines, rows


def _verifier_lines(funnel: dict) -> list[str]:
    lines = ["### Verifier checks", ""]
    ver = (funnel.get("v4") or {}).get("verifier")
    if not ver:
        return lines + [f"No verifier counts ({FUNNEL_FILE} has no v4 verifier entry)."]
    pre, post = int(ver.get("reached_pre") or 0), int(ver.get("reached_post") or 0)
    failed = ver.get("failed_by_check") or {}
    lines += [f"{pre} tasks reached the pre-check of the original message and {post} sets the post-check of the built "
              f"set; {ver.get('invalid', 0)} replies were invalid (both together). Rate = failed / reached, over the "
              "valid replies.", ""]
    reached = {**dict.fromkeys(PRE_CHECKS_V4, pre), **dict.fromkeys(POST_CHECKS_V4, post)}
    checks = [*reached, *sorted(set(failed) - set(reached))]
    lines += _md_table(["check", "asked in", "failed", "reached", "rate"],
                       [[c, "pre-check" if c in PRE_CHECKS_V4 else "post-check" if c in POST_CHECKS_V4 else NA,
                         str(failed.get(c, 0)), str(reached[c]) if c in reached else NA,
                         analysis._rate_cell(analysis._rate(failed.get(c, 0), reached[c])) if c in reached else NA]
                        for c in checks])
    return lines


def _set_field_lines(run: Path, sets: list[dict] | None, funnel: dict, sets_path: Path) -> tuple[list[str], Any]:
    lines = ["### The built sets", ""]
    if sets is None:
        return lines + [_not_run_line(run, "sets", sets_path.name)], None
    rows = set_counts(sets)
    kept = kept_span_counts(sets)
    lines += ["Kept true spans: claims the pre-check found supported, left word for word in O, P and H. Shown "
              "eligible: rendered transcript at or below the shown cap. P has a model rewrite: p3_has_model_rewrite. "
              "Style remarks in an edit list: sets with a bank entry under a line that announces edits (items in "
              "brackets). "
              "Named failure pre-existing: whether the failing test the disclosure names already failed before the "
              "agent's first edit (not_applicable: the disclosure names no failing test).", ""]
    lines += _md_table(["gap type", "sets", "kept true spans", "sets with a kept span", "shown eligible",
                        "P has a model rewrite", "style remark in an edit list", "primary in an edit list",
                        "named failure pre-existing"],
                       [[g, str(r["sets"]), str(r["kept_true_spans"]), str(r["sets_with_kept_span"]),
                         str(r["shown_eligible"]), str(r["p3_has_model_rewrite"]),
                         f"{r['style_in_edit_list_sets']} ({r['style_in_edit_list_items']})",
                         str(r["primary_in_edit_list"]),
                         ", ".join(f"{k} {n}" for k, n in r["named_failure_pre_existing"].items())]
                        for g, r in rows.items()])
    dist = ", ".join(f"{k}: {n}" for k, n in kept["distribution"].items()) or "no sets"
    lines += ["", f"Kept true spans per set (n kept spans: n sets): {dist}; total {kept['total']}; by supported_by: "
              f"{_reasons_cell(kept['by_supported_by'])}."]
    tried = (funnel.get("v4") or {}).get("named_failure_pre_existing")
    if tried is not None:
        lines.append(f"named_failure_pre_existing as {FUNNEL_FILE} counts it: {_esc(_flat(tried))}.")
    return lines, {"by_gap": rows, "kept_true_spans": kept}


def _stage_cost_lines(cost_rows: list[dict], infos: list[dict] | None, sets: list[dict] | None) -> tuple[list, dict]:
    c = stage_cost(cost_rows, SETS_PURPOSES)
    n_tried = None if infos is None else len(infos)
    n_sets = None if sets is None else len(sets)
    per_task = c["usd"] / n_tried if n_tried else None
    per_set = c["usd"] / n_sets if n_sets else None
    lines = ["### Cost of the sets stage", "",
             f"Live calls in {COST_FILE} with the purposes {', '.join(SETS_PURPOSES)} (disk-cache hits cost nothing "
             "and are not logged).", ""]
    if c["rows"]:
        lines += _md_table(["purpose", "provider", "model", "calls", "USD", "unpriced calls"],
                           [[r["purpose"], r["provider"], r["model"], str(r["calls"]), _usd(r["usd"]),
                             str(r["n_unpriced"])] for r in c["rows"]])
        lines.append("")
    lines.append(f"Total {_usd(c['usd'])} USD over {c['calls']} calls ({c['n_unpriced']} unpriced); per task tried "
                 f"{_usd(per_task, 4)} USD ({NA if n_tried is None else n_tried} tasks in {BUILD_FILE}); per built set "
                 f"{_usd(per_set, 4)} USD ({NA if n_sets is None else n_sets} sets).")
    return lines, {**c, "n_tasks_tried": n_tried, "n_sets": n_sets, "usd_per_task_tried": per_task,
                   "usd_per_set": per_set}


def _sets_section(run: Path, cfg: dict, sets: list[dict] | None, sets_path: Path, report: list[dict] | None,
                  infos: list[dict] | None, funnel: dict, cost_rows: list[dict]) -> tuple[list[str], dict]:
    rebuild, rebuild_data = _rebuild_lines(run, report)
    outcomes, outcome_data = _outcome_lines(run, infos)
    fields, field_data = _set_field_lines(run, sets, funnel, sets_path)
    cost, cost_data = _stage_cost_lines(cost_rows, infos, sets)
    lines = ["## v4 sets: funnel", "", *rebuild, "", *_topup_lines(cfg, funnel), "", *outcomes, "",
             *_verifier_lines(funnel), "", *fields, "", *cost]
    data = {"rebuild": rebuild_data, "funnel_v4": funnel.get("v4"), "candidate_pool": funnel.get("v4_candidate_pool"),
            "extra_pool": funnel.get("extra_pool"), "outcomes": outcome_data, "sets": field_data, "cost": cost_data}
    return lines, data


# ---------------------------------------------------------------------------------------------
# Section 3: the pairwise judge runs on the v4 sets
# ---------------------------------------------------------------------------------------------


def _judge_table(records: list[dict], pairs: list[list[str]], n_boot: int, seed: int) -> tuple[list[dict], list[str]]:
    if not records:
        return [], ["No judge records."]
    av4 = _import("analysis_v4")
    rows, md = av4.judge_table(records, pairs, n_boot, seed)
    return rows, _lines(md)


def _hidden_run_lines(run_cfg: dict, records: list[dict], s3_records: list[dict] | None, s3_name: str,
                      report: list[dict] | None, sets: list[dict], pairs: list[list[str]], n_boot: int,
                      seed: int) -> tuple[list[str], dict]:
    label = run_cfg.get("grader_label", run_cfg["judge"])
    lines = ["", f"#### {run_cfg['id']}: the rebuilt pilot-3 tasks, v4 next to v3", ""]
    data: dict[str, Any] = {"run": run_cfg["id"]}
    if report is None:
        lines.append(f"No comparison: {REBUILD_FILE} is missing.")
    elif s3_records is None:
        lines.append(f"No comparison: pilot 3's {s3_name} is missing.")
    else:
        v4_ids = {s["task_id"] for s in sets}
        ids = sorted({e["task_id"] for e in report if e.get("v4_outcome") == "built"} & v4_ids)
        wanted = set(ids)
        v3 = [r for r in _non_smoke(s3_records) if r["grader"] == label and analysis.record_set(r) == V3_SET
              and analysis.record_rubric(r) == run_cfg["rubric"] and r["condition"] == run_cfg["condition"]
              and r["task_id"] in wanted]
        v4 = [r for r in records if r["task_id"] in wanted]
        rows, md = _judge_table(v3 + v4, pairs, n_boot, seed)
        lines += [f"The {len(ids)} rebuilt tasks that have a v4 set; pilot 3's records of {label} (set v3, rubric "
                  f"{run_cfg['rubric']}, {run_cfg['condition']}; {s3_name}, read only) on the same task ids: "
                  f"{len(v3)} v3 records and {len(v4)} v4 records.", "", *md]
        data["comparison"] = {"task_ids": ids, "n_v3_records": len(v3), "n_v4_records": len(v4), "rows": rows}
    combo = {"grader": label, "set": run_cfg["set"], "rubric": run_cfg["rubric"], "condition": run_cfg["condition"]}
    gap_rows, gap_md = analysis.gap_breakdown(records, combo, pairs, n_boot, seed)
    lines += ["", f"#### {run_cfg['id']} by gap type", "", *gap_md]
    av4 = _import("analysis_v4")
    prov_rows, prov_md = av4.provenance_split(records, {s["task_id"]: s for s in sets}, n_boot=n_boot, seed=seed)
    lines += ["", f"#### {run_cfg['id']}: provenance split", "", *_lines(prov_md)]
    data.update(gap_breakdown=gap_rows, provenance_split=prov_rows)
    return lines, data


def _shown_run_lines(run_cfg: dict, records: list[dict], hidden: list[dict], report: list[dict] | None,
                     pairs: list[list[str]], n_boot: int, seed: int) -> tuple[list[str], dict]:
    label = run_cfg.get("grader_label", run_cfg["judge"])
    ids = sorted({r["task_id"] for r in records})
    gap_of = {r["task_id"]: r.get("gap_type") for r in records}
    changed = {e["task_id"] for e in report or [] if e.get("primary_changed") is True}
    rule = run_cfg.get("task_rule")
    rule_text = f"task rule {rule}: {TASK_RULES.get(rule, 'not described here')}" if rule else "no task rule"
    lines = ["", f"#### {run_cfg['id']}: task list", "",
             f"{label}, set {run_cfg['set']}, rubric {run_cfg['rubric']}, shown; {rule_text}. {len(ids)} tasks have "
             f"records: {sum(gap_of[t] == 'no_tests_run' for t in ids)} no_tests_run, "
             f"{sum(t in changed for t in ids)} with a primary span changed from v3"
             f"{'' if report is not None else f' (unknown: {REBUILD_FILE} is missing)'}; a task can be both. Task "
             f"ids: {', '.join(ids)}."]
    combo = {"grader": label, "set": run_cfg["set"], "rubric": run_cfg["rubric"], "condition": "shown"}
    gap_rows, gap_md = analysis.gap_breakdown(records, combo, pairs, n_boot, seed)
    lines += ["", f"#### {run_cfg['id']} by gap type", "", *gap_md]
    wanted = set(ids)
    same = [r for r in hidden if r["grader"] == label and analysis.record_rubric(r) == run_cfg["rubric"]
            and r["task_id"] in wanted]
    hidden_rows, hidden_md = _judge_table(same, pairs, n_boot, seed)
    lines += ["", f"#### {run_cfg['id']}: the same tasks hidden", "",
              f"The hidden records of {label} (set {run_cfg['set']}, rubric {run_cfg['rubric']}) on these task ids: "
              f"{len(same)} records.", "", *hidden_md]
    return lines, {"run": run_cfg["id"], "task_ids": ids, "gap_breakdown": gap_rows, "hidden_same_tasks": hidden_rows}


def _judge_section(run: Path, cfg: dict, judge: list[dict] | None, s3_records: list[dict] | None, s3_name: str,
                   sets: list[dict] | None, report: list[dict] | None, labels: list[dict] | None, status: dict,
                   pairs: list[list[str]], n_boot: int, seed: int) -> tuple[list[str], dict]:
    runs = [r for r in cfg["runs"] if r["set"] == SET]
    lines = ["## Pairwise judges on the v4 sets", "",
             "The judge runs of the config on the v4 sets (tables: analysis_v4.judge_table). Values above 0.5 favour "
             "the first-named variant. Smoke-only records are left out."]
    data: dict[str, Any] = {"hidden": None, "shown": None, "reasons": None, "disclosure_false": None}
    if judge is None:
        return lines + ["", _not_run_line(run, "judge", JUDGE_FILE), *_not_done(status, [r["id"] for r in runs])], data
    v4 = [r for r in _non_smoke(judge) if analysis.record_set(r) == SET]
    hidden = [r for r in v4 if r["condition"] == "hidden"]
    for condition in ("hidden", "shown"):
        cond_runs = [r for r in runs if r["condition"] == condition]
        recs = [r for r in v4 if r["condition"] == condition]
        rows, md = _judge_table(recs, pairs, n_boot, seed)
        lines += ["", f"### {condition.capitalize()}", "", *md]
        not_done = _not_done(status, [r["id"] for r in cond_runs])
        if not_done:
            lines += ["", *not_done]
        per_run = []
        for run_cfg in cond_runs:
            run_recs = _run_records(recs, run_cfg)
            if not run_recs:
                continue
            if condition == "hidden":
                run_lines, run_data = _hidden_run_lines(run_cfg, run_recs, s3_records, s3_name, report, sets or [],
                                                        pairs, n_boot, seed)
            else:
                run_lines, run_data = _shown_run_lines(run_cfg, run_recs, hidden, report, pairs, n_boot, seed)
            lines += run_lines
            per_run.append(run_data)
        data[condition] = {"rows": rows, "runs": per_run}
    lines += ["", "### What the reasons cite", ""]
    if labels is None:
        return lines + [_not_run_line(run, "reasons", REASON_LABELS_FILE)], data
    rv4 = _import("analysis_reasons_v4")
    reason_rows, reason_md = rv4.reason_table_v4(v4, labels, by_gap=True)
    false_rows = [{k: r[k] for k in ("grader", "set", "rubric", "condition", "gap_type")}
                  | {k: r["O_vs_H_only"]["for_O"][k] for k in ("n", "n_labelled", "says_disclosure_false")}
                  for r in reason_rows]
    lines += [*_lines(reason_md), "",
              "O-over-H verdicts (O won an O-vs-H order) and how many of the labelled ones have a reason that says the "
              "disclosure itself is false (says_disclosure_false):", ""]
    lines += (_md_table(["grader", "set", "rubric", "condition", "gap type", "O over H verdicts", "labelled",
                         "says disclosure false"],
                        [[r["grader"], r["set"], r["rubric"], r["condition"], r["gap_type"], str(r["n"]),
                          str(r["n_labelled"]), str(r["says_disclosure_false"])] for r in false_rows])
              if false_rows else ["No judge records."])
    data.update(reasons=reason_rows, disclosure_false=false_rows)
    return lines, data


# ---------------------------------------------------------------------------------------------
# Section 4: the pointwise pilot (Experiment 1b)
# ---------------------------------------------------------------------------------------------


def _pairwise_rows(graders: list[str], judge_v4: list[dict], s3_records: list[dict] | None, condition: str,
                   pointwise_set: str, pointwise_ids: dict[str, set[str]], pairs: list[list[str]], n_boot: int,
                   seed: int) -> tuple[list[dict], dict[str, dict]]:
    """The pairwise rates of the pointwise graders, as analysis_v4.judge_table rows of the condition and rubric
    PAIRWISE_RUBRIC: a grader's v4 records (this run) when it has any, else its pilot-3 records on the v3 sets. A
    grader's v4 records are cut to its pointwise task ids (pointwise_ids[grader]) when they cover every one of them;
    otherwise, and for pilot-3 records (other sets), every record is kept. Each row gets "source" and "pairwise_set"
    (the set it was judged on); a pilot-3 row takes the pointwise set as "set", so that analysis_v4.sign_agreement
    joins it with the pointwise row on (grader, set, condition). Also returns, per grader with pairwise records,
    {source, cut (the records were cut to the pointwise tasks), n_pairwise_tasks (tasks of the records used),
    n_pointwise_tasks}."""
    wanted = set(graders)
    own = [r for r in judge_v4 if r["grader"] in wanted and r["condition"] == condition
           and analysis.record_rubric(r) == PAIRWISE_RUBRIC]
    have = {r["grader"] for r in own}
    old = [r for r in _non_smoke(s3_records or []) if r["grader"] in wanted - have and r["condition"] == condition
           and analysis.record_set(r) == V3_SET and analysis.record_rubric(r) == PAIRWISE_RUBRIC]
    scope: dict[str, dict] = {}
    kept: list[dict] = []
    for grader in sorted(have):
        recs = [r for r in own if r["grader"] == grader]
        ids = pointwise_ids.get(grader, set())
        cut = bool(ids) and ids <= {r["task_id"] for r in recs}
        if cut:
            recs = [r for r in recs if r["task_id"] in ids]
        kept += recs
        scope[grader] = {"source": SOURCE_PILOT4, "cut": cut, "n_pairwise_tasks": len({r["task_id"] for r in recs}),
                         "n_pointwise_tasks": len(ids)}
    for grader in sorted({r["grader"] for r in old}):
        scope[grader] = {"source": SOURCE_PILOT3, "cut": False,
                         "n_pairwise_tasks": len({r["task_id"] for r in old if r["grader"] == grader}),
                         "n_pointwise_tasks": len(pointwise_ids.get(grader, set()))}
    rows = [{**row, "source": SOURCE_PILOT4, "pairwise_set": row["set"]}
            for row in _judge_table(kept, pairs, n_boot, seed)[0]]
    rows += [{**row, "source": SOURCE_PILOT3, "pairwise_set": row["set"], "set": pointwise_set}
             for row in _judge_table(old, pairs, n_boot, seed)[0]]
    return rows, scope


def _scope_text(grader: str, scope: dict | None, n_pointwise: int, n_complete: int | None = None) -> str:
    """Which tasks each side of the sign-agreement row of `grader` used, in words. n_pointwise = the grader's
    replicate-0 pointwise tasks (the ids the pairwise records are cut to); n_complete = how many of them have a valid
    score for all of O, H and P (the premium's n_tasks in analysis_v4.pointwise_metrics), named when it differs."""
    pointwise = f"pointwise {n_pointwise} tasks"
    if n_complete is not None and n_complete != n_pointwise:
        pointwise += f" (the premium over the {n_complete} with a valid score for O, H and P)"
    if scope is None:
        return f"{grader}: no pairwise records; {pointwise}"
    if scope["cut"]:
        pairwise = f"{scope['source']} cut to the {scope['n_pairwise_tasks']} pointwise tasks"
    elif scope["source"] == SOURCE_PILOT4:
        pairwise = (f"{scope['source']}, all {scope['n_pairwise_tasks']} tasks of its run ("
                    + ("no replicate-0 pointwise task to cut to)" if not n_pointwise
                       else "they do not include every pointwise task)"))
    else:
        pairwise = f"{scope['source']}, the {scope['n_pairwise_tasks']} tasks of its pilot-3 run"
    return f"{grader}: pairwise {pairwise}; {pointwise}"


def shown_first_calls(records: list[dict]) -> list[dict]:
    """Per (grader, task_id) of the given (shown) pointwise records, in order of first appearance: the task's first
    record (e1b_pointwise appends a record as its call returns, O, H, P in turn, so the first record of a task in the
    file is its first call): its variant, cached (a disk-cache hit), cache_read and cache_write (usage
    cache_read_input_tokens and cache_creation_input_tokens, summed over that call's attempts; None without usage);
    and over the task's records: n_records and cost_usd (their sum; None when one has no cost_usd)."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        groups.setdefault((str(r["grader"]), str(r["task_id"])), []).append(r)
    rows = []
    for (grader, task_id), recs in groups.items():
        first = recs[0]
        usage = first.get("usage") or {}
        costs = [r.get("cost_usd") for r in recs]
        rows.append({"grader": grader, "task_id": task_id, "gap_type": first.get("gap_type"),
                     "first_variant": first.get("variant"), "first_cached": first.get("cached"),
                     "cache_read": usage.get("cache_read_input_tokens"),
                     "cache_write": usage.get("cache_creation_input_tokens"), "n_records": len(recs),
                     "cost_usd": None if any(c is None for c in costs) else sum(costs)})
    return rows


def _shown_call_lines(shown: list[dict]) -> tuple[list[str], dict]:
    """The table of shown_first_calls and one line per grader: first calls that read the prompt cache and the cost
    per task."""
    rows = shown_first_calls(shown)
    lines = ["", "Per task: the first pointwise call (the task's first record in pointwise_calls.jsonl) with its "
             "cache-read and cache-write tokens, and the cost of the task's pointwise calls (sum of their cost_usd; a "
             "disk-cache hit is logged with the tokens of the call it repeats and costs 0).", ""]
    lines += _md_table(["grader", "task", "gap type", "first call", "disk-cache hit", "cache-read tokens",
                        "cache-write tokens", "records", "cost USD"],
                       [[r["grader"], _esc(r["task_id"]), _esc(r["gap_type"]), _esc(r["first_variant"]),
                         _yes_no(r["first_cached"]), _flat(r["cache_read"]), _flat(r["cache_write"]),
                         str(r["n_records"]), _usd(r["cost_usd"], 4)] for r in rows])
    per_grader: dict[str, dict] = {}
    for grader in dict.fromkeys(r["grader"] for r in rows):
        mine = [r for r in rows if r["grader"] == grader]
        with_usage = [r for r in mine if r["cache_read"] is not None]
        priced = [r["cost_usd"] for r in mine if r["cost_usd"] is not None]
        per_grader[grader] = {"n_tasks": len(mine), "n_first_calls_with_usage": len(with_usage),
                              "n_first_calls_reading_cache": sum(r["cache_read"] > 0 for r in with_usage),
                              "n_tasks_priced": len(priced),
                              "mean_cost_per_task": sum(priced) / len(priced) if priced else None}
        g = per_grader[grader]
        lines += ["", f"{grader}: {g['n_tasks']} tasks; first calls that read the prompt cache (cache-read tokens "
                  f"above 0): {g['n_first_calls_reading_cache']} of the {g['n_first_calls_with_usage']} with usage; "
                  f"cost per task {_usd(g['mean_cost_per_task'], 4)} USD on average over the {g['n_tasks_priced']} "
                  "tasks whose calls are all priced."]
    return lines, {"first_calls": rows, "by_grader": per_grader}


def _pointwise_section(run: Path, cfg: dict, pw: list[dict] | None, judge: list[dict] | None,
                       s3_records: list[dict] | None, rm: list[dict] | None, status: dict, pairs: list[list[str]],
                       n_boot: int, seed: int) -> tuple[list[str], dict]:
    pcfg = cfg.get("pointwise") or {}
    condition, pw_set = pcfg.get("condition", "hidden"), pcfg.get("set", SET)
    run_ids = [r["id"] for r in pcfg.get("runs") or []]
    lines = ["## Pointwise pilot (Experiment 1b)", "",
             f"The runs of pointwise.runs: each judge scores O, H and P one at a time ({condition}, set {pw_set}; "
             "tables: analysis_v4.pointwise_metrics)."]
    data: dict[str, Any] = {"metrics": None, "sign_agreement": None, "pairwise_sources": None, "pairwise_tasks": None,
                            "reward_model": None, "self_consistency": None, "shown": None, "shown_calls": None}
    not_done = _not_done(status, run_ids)
    hidden = [r for r in pw or [] if r.get("condition") == condition]
    if pw is None:
        lines += ["", _not_run_line(run, "pointwise", POINTWISE_FILE)]
    elif not hidden:
        lines += ["", f"No pointwise records of the {condition} condition in {POINTWISE_FILE}."]
    else:
        av4 = _import("analysis_v4")
        pw_rows, pw_md = av4.pointwise_metrics(hidden, n_boot=n_boot, seed=seed)
        lines += ["", "### Per judge", "", *_lines(pw_md)]
        graders = sorted({r["grader"] for r in hidden})
        pointwise_ids = {g: {r["task_id"] for r in hidden if r["grader"] == g and int(r.get("replicate", 0)) == 0
                             and analysis.record_set(r) == pw_set} for g in graders}
        judge_v4 = [r for r in _non_smoke(judge or []) if analysis.record_set(r) == SET]
        pair_rows, scope = _pairwise_rows(graders, judge_v4, s3_records, condition, pw_set, pointwise_ids, pairs,
                                          n_boot, seed)
        sources = {r["grader"]: r["source"] for r in pair_rows}
        rm_rows = analysis.rm_rows_pilot2([r for r in rm or [] if analysis.record_set(r) == pw_set
                                           and r["condition"] == condition], pairs, n_boot, seed)
        sa_rows, sa_md = av4.sign_agreement([*pair_rows, *rm_rows], pw_rows)
        complete = {r["grader"]: r["n_tasks"] for r in pw_rows if r["set"] == pw_set}
        lines += ["", "### Sign agreement with the pairwise judges", "",
                  f"Pairwise O vs H rates of the same judges ({condition}, rubric {PAIRWISE_RUBRIC}). A judge's "
                  "pilot-4 pairwise records are cut to its pointwise tasks (replicate 0) when they cover all of them; "
                  "the tasks each side used: "
                  + ("; ".join(_scope_text(g, scope.get(g), len(pointwise_ids[g]), complete.get(g)) for g in graders)
                     or "no pointwise judge") + ".", "", *_lines(sa_md)]
        sc = av4.self_consistency(hidden)
        lines += ["", "### Self-consistency (replicates 0 and 1)", "", *_lines(sc["markdown"])]
        data.update(metrics=pw_rows, sign_agreement=sa_rows, pairwise_sources=sources, pairwise_tasks=scope,
                    self_consistency=sc["rows"])
    shown = [r for r in pw or [] if r.get("condition") == "shown"]
    if pw is not None:
        lines += ["", "### Shown pointwise records of the judge run", ""]
        if shown:
            shown_rows, shown_md = _import("analysis_v4").pointwise_metrics(shown, n_boot=n_boot, seed=seed)
            call_lines, call_data = _shown_call_lines(shown)
            lines += [*_lines(shown_md), *call_lines]
            data["shown"] = shown_rows
            data["shown_calls"] = call_data
        else:
            lines.append("No shown pointwise records (the shown judge run with pointwise_after has not run).")
    if not_done:
        lines += ["", *not_done]
    lines += ["", "### Reward model", ""]
    if rm is None:
        lines.append(_not_run_line(run, "rm", RM_FILE))
    else:
        rm_all = analysis.rm_rows_pilot2([r for r in rm if analysis.record_set(r) == pw_set], pairs, n_boot, seed)
        lines += [f"Cells: P(score X > score Y) within a task (equal scores = 0.5), then the mean standardised "
                  f"difference (population SD of all scores of that grader, set and condition), with 95% CIs over "
                  f"tasks; set {pw_set}.", "", *analysis.main_table([], rm_all, pairs), "",
                  *analysis.rm_std_table(rm_all, pairs)]
        data["reward_model"] = rm_all
    return lines, data


# ---------------------------------------------------------------------------------------------
# Sections 5 and 6: Experiment 2 policy pilot, Experiment 3 three-label check
# ---------------------------------------------------------------------------------------------


def _verdict_sentence(verdict: dict | None, labeller: str) -> str:
    """The verdict of one labeller against the clean-overclaim rule, in one sentence."""
    if not verdict:
        return f"Verdict of the {labeller}: not recorded in e2_summary.json."
    rate, threshold = verdict.get("clean_overclaim_rate_n1"), verdict.get("threshold")
    full = verdict.get("full_e2_runs")
    if rate is None:
        return (f"Verdict of the {labeller}: the clean-overclaim rate at N = 1 is undefined (no validly labelled "
                f"sample), so the rule (threshold {_flat(threshold)}) gives no answer.")
    return (f"Verdict of the {labeller}: the clean-overclaim rate at N = 1 is {rate:.3f}, "
            f"{'at or above' if full else 'below'} the threshold {_flat(threshold)}, so the full Experiment 2 "
            f"{'runs' if full else 'does not run'} in session 5.")


def _agreement_sentence(e2: dict) -> str:
    """One plain sentence: do the two labellers' verdicts agree (e2_summary.json's verdicts_agree)?"""
    verdict, verdict_full = e2.get("verdict") or {}, e2.get("verdict_full")
    if not verdict_full:
        return ("The full-evidence labeller did not run (e2_summary.json has no full-evidence results), so there is no "
                "second verdict to compare.")
    if verdict.get("clean_overclaim_rate_n1") is None or verdict_full.get("clean_overclaim_rate_n1") is None:
        return "At least one of the two clean-overclaim rates is undefined, so the two verdicts are not compared."

    def runs(v: dict) -> str:
        return "runs" if v.get("full_e2_runs") else "does not run"

    if e2.get("verdicts_agree"):
        return f"The two verdicts agree: under both labellers the full Experiment 2 {runs(verdict)} in session 5."
    return (f"The two verdicts disagree: under the test-record labeller the full Experiment 2 {runs(verdict)} in "
            f"session 5, under the full-evidence labeller it {runs(verdict_full)}.")


def _e2_rate_cell(rates: dict | None, key: str) -> str:
    """'rate (count)' of one base_rates_v4 entry, n/a when the labeller or the group is missing."""
    if rates is None:
        return NA
    return f"{_flat(rates.get(key))} ({_flat(rates.get('n_' + key))})"


def _e2_section(run: Path, cfg: dict) -> tuple[list[str], Any]:
    lines = ["## Experiment 2 policy pilot", ""]
    e2 = _read_json_file(run / E2_SUMMARY_FILE)
    if e2 is None:
        return lines + [_not_run_line(run, "e2", E2_SUMMARY_FILE)], None
    stopped = e2.get("status") == "stopped_labeller_validation"
    lines += [f"Status: {e2.get('status')}.", "",
              "Two labellers label the same samples: the test-record labeller (pilot 3's prompt; it reads the "
              "test-runner record only and is validated on the v3 sets; below its stop line the run stops) and the "
              "full-evidence labeller (it also reads the other commands that ran code, with their exit codes and last "
              "output lines, and is validated on the v4 sets; it never stops the run). Stop line: accuracy "
              f"{_flat((cfg.get('e2') or {}).get('stop_below'))}.", ""]
    validations = []
    for suffix, name in E2_LABELLERS:
        v = e2.get(f"validation{suffix}")
        below = (_yes_no(stopped) if not suffix else _yes_no(v.get("below_threshold"))) if v else NA
        validations.append([name, V3_SET if not suffix else SET] + ([NA] * 5 if v is None else [
            _flat(v.get("n")), f"{_flat(v.get('originals_correct'))} ({_flat(v.get('acc_originals'))})",
            f"{_flat(v.get('honest_correct'))} ({_flat(v.get('acc_honest'))})", _flat(v.get("invalid")), below]))
    lines += _md_table(["labeller", "validated on", "sets", "originals labelled overclaim (accuracy)",
                        "honest variants labelled honest (accuracy)", "invalid labels", "below the stop line"],
                       validations)
    if stopped:
        return lines + ["", "The labeller's accuracy was below the stop line, so no sample was generated."], e2
    words = e2.get("words") or {}
    lines += ["", f"Samples: cut off at the token limit {_flat(e2.get('n_truncated'))}; empty "
              f"{_flat(e2.get('n_empty'))}; duplicates of an earlier sample of the same task "
              f"{_flat(e2.get('n_duplicates'))}; distinct texts {_flat(e2.get('n_distinct_texts'))}; words min "
              f"{_flat(words.get('min'))}, median "
              f"{_flat(words.get('median'))}, max {_flat(words.get('max'))}."]
    base, base_full = e2.get("base_rates"), e2.get("base_rates_full")
    if base or base_full:
        gaps = sorted({*((base or {}).get("by_gap") or {}), *((base_full or {}).get("by_gap") or {})})
        rows = []
        for g in ["all", *gaps]:
            cells = []
            for rates in (base, base_full):
                r = rates if g == "all" or rates is None else (rates.get("by_gap") or {}).get(g)
                cells += [_e2_rate_cell(r, "overclaim"), _e2_rate_cell(r, "clean_overclaim"),
                          NA if r is None else f"{_flat(r.get('n_labelled'))} / {_flat(r.get('n_invalid'))}"]
            rows.append([g, *cells])
        head = [f"{what}, {name}" for _suffix, name in E2_LABELLERS
                for what in ("overclaim", "clean overclaim", "labelled / invalid")]
        lines += ["", "Rates at N = 1, pooled over the validly labelled samples, rate (count) (clean overclaim = an "
                  "overclaim that does not disclose the gap):", "", *_md_table(["samples", *head], rows)]
    for suffix, name in E2_LABELLERS:
        markdown = e2.get(f"markdown{suffix}")
        lines += ["", f"Best-of-N per selector, {name} (tie-averaged, then seed 0):", ""]
        lines += _lines(markdown) if markdown else ["Not in e2_summary.json."]
    lines += ["", *[f"- {_verdict_sentence(e2.get(f'verdict{suffix}'), name)}" for suffix, name in E2_LABELLERS
                    if not suffix or f"verdict{suffix}" in e2], "", _agreement_sentence(e2)]
    return lines, e2


def _e3_section(run: Path) -> tuple[list[str], Any]:
    lines = ["## Experiment 3 three-label check", ""]
    e3 = _read_json_file(run / E3_SUMMARY_FILE)
    if e3 is None:
        return lines + [_not_run_line(run, "e3", E3_SUMMARY_FILE)], None
    lines.append(f"Judge {e3.get('grader', e3.get('judge'))}, set {e3.get('set')}, rubric {e3.get('rubric')}, shown, "
                 f"{e3.get('n_tasks')} tasks: {', '.join(e3.get('task_ids') or []) or NA}.")
    if e3.get("markdown"):
        lines += ["", "Three judges on the same tasks (plain, two-label, three-label), with the parse rate of each:",
                  "", *_lines(e3["markdown"])]
    else:
        lines += ["", "Comparison: not recorded in e3_summary.json."]
    claims = e3.get("claims")
    lines += ["", "Claims listed by the three-label judge per variant, over its valid records: count of each label "
              "(share of the claims listed):", ""]
    if claims:
        lines += _md_table(["variant", "appearances", "claims listed", *CLAIM_LABELS],
                           [[v, _flat(e.get("appearances")), _flat(e.get("claims")),
                             *[f"{_flat(e.get(lab))} ({_flat(e.get(f'share_{lab}'))})" for lab in CLAIM_LABELS]]
                            for v, e in (claims.get("variants") or {}).items()])
        lines += ["", f"Records: {_flat(claims.get('n_records'))}, valid {_flat(claims.get('n_valid'))}, invalid "
                  f"{_flat(claims.get('n_invalid'))} (no claim lists)."]
    else:
        lines.append("Not recorded in e3_summary.json.")
    return lines, e3


# ---------------------------------------------------------------------------------------------
# Section 7: spend
# ---------------------------------------------------------------------------------------------


def _spend_section(run: Path, cfg: dict, stages: dict, status: dict) -> tuple[list[str], dict]:
    lines = ["## Spend", ""]
    data: dict[str, Any] = {"by_provider": None, "by_model_purpose": None, "prior": None, "not_launched": None}
    budget = cfg.get("budget") or {}
    if (run / COST_FILE).exists():
        from overclaim.costs import cost_tables

        costs, _ = cost_tables(run)
        providers = []
        for r in costs["by_provider"]:
            cap = budget.get(f"{r['provider']}_usd_cap")
            providers.append({**r, "cap": cap, "share_of_cap": r["usd"] / float(cap) if cap else None})
        lines += [f"This session: {COST_FILE} of this run directory (one line per live call; disk-cache hits cost "
                  "nothing and are not logged), against the caps of the config.", "",
                  *_md_table(["provider", "calls", "USD", "unpriced calls", "cap USD", "share of cap"],
                             [[r["provider"], str(r["calls"]), _usd(r["usd"]), str(r["n_unpriced"]),
                               _usd(None if r["cap"] is None else float(r["cap"]), 2),
                               analysis._rate_cell(r["share_of_cap"])] for r in providers])]
        stop = budget.get("anthropic_stop_launching_usd")
        if stop is not None:
            spent = next((r["usd"] for r in providers if r["provider"] == "anthropic"), 0.0)
            lines += ["", f"Anthropic stop-launching line {float(stop):.2f} USD: the Anthropic total {spent:.3f} USD "
                      f"has {'reached' if spent >= float(stop) else 'not reached'} it."]
        lines += ["", "By purpose:", "",
                  *_md_table(["provider", "model", "purpose", "calls", "USD", "unpriced calls"],
                             [[r["provider"], r["model"], r["purpose"], str(r["calls"]), _usd(r["usd"]),
                               str(r["n_unpriced"])] for r in costs["by_model_purpose"]])]
        data.update(by_provider=providers, by_model_purpose=costs["by_model_purpose"])
    else:
        lines.append(f"This session: no cost records ({COST_FILE} is missing).")
    prior = _read_json_file(run / PRIOR_SPEND_FILE)
    lines += ["", f"Before this session ({PRIOR_SPEND_FILE}: the cost logs of the earlier pilots, read only):", ""]
    if prior is None:
        lines.append(_not_run_line(run, "carryover", PRIOR_SPEND_FILE))
    else:
        head = ["calls", "USD", "unpriced calls"]

        def cells(spend: dict) -> list[str]:
            return [_flat(spend.get("calls")), _usd(spend.get("usd")), _flat(spend.get("n_unpriced"))]

        lines += _md_table(["provider", *head], [[p, *cells(sp)] for p, sp in (prior.get("by_provider") or {}).items()])
        lines += ["", "Per file:", "",
                  *_md_table(["file", "present", "provider", *head],
                             [[name, _yes_no(f.get("present")), p, *cells(sp)]
                              for name, f in (prior.get("files") or {}).items()
                              for p, sp in ((f.get("by_provider") or {}).items() or [(NA, {})])])]
        if prior.get("note"):
            lines += ["", str(prior["note"])]
        data["prior"] = prior
    not_launched = [f"Stage {name}: {e.get('status')}, reason: {e.get('reason') or 'not recorded'}."
                    for name, e in stages.items() if e.get("status") != "done"]
    not_launched += _not_done(status, list(status))
    stop_file = run / BUDGET_STOP_FILE
    if stop_file.exists():
        not_launched.append(f"{BUDGET_STOP_FILE}: {_esc(stop_file.read_text(encoding='utf-8').strip())}")
    lines += ["", "Stages and runs not launched or not complete:", "",
              *([f"- {line}" for line in not_launched] or ["None recorded."])]
    data["not_launched"] = not_launched
    return lines, data


# ---------------------------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------------------------


def summarize_pilot4(run_dir: str | Path, cfg: dict, extra_sections: list[dict] | None = None) -> str:
    """Pilot 4 summary from the files of run_dir, the v4 sets file and pilot 3's judge records (read only).

    Sections, in order (see the module docstring): header, v4 funnel, pairwise judges on the v4 sets, pointwise
    pilot, Experiment 2, Experiment 3, spend, then `extra_sections` ({"key", "markdown", "data"}; the markdown goes
    into summary.md and the data under summary.json[key]). A missing input gives a "Not run" line with the reason from
    stages_status.json / runs_status.json, never an exception. Bootstrap: cfg["analysis"]["n_bootstrap"] resamples,
    seed cfg["analysis"]["seed"]. Writes run_dir/summary.md and run_dir/summary.json (the same numbers as data, NaN
    as null); returns the markdown.
    """
    run = Path(run_dir)
    pairs = [list(p) for p in cfg["pairs"]]
    n_boot, seed = int(cfg["analysis"]["n_bootstrap"]), int(cfg["analysis"]["seed"])
    sets_path = _repo_path(cfg["data"]["processed_dir"]) / cfg["sets"][SET]["file"]
    s3_path = _repo_path(cfg["session3_dir"]) / JUDGE_FILE
    stages = _read_json_file(run / STAGES_STATUS_FILE) or {}
    status = _read_json_file(run / RUNS_STATUS_FILE) or {}
    funnel = _read_json_file(run / FUNNEL_FILE) or {}
    report = _read_json_file(run / REBUILD_FILE)
    loaded = {name: _load(run / name) for name in (JUDGE_FILE, POINTWISE_FILE, RM_FILE, REASON_LABELS_FILE,
                                                   BUILD_FILE, COST_FILE)}
    sets, s3_records = _load(sets_path), _load(s3_path)
    cost_rows = loaded[COST_FILE] or []
    inputs = {name: {"present": recs is not None, "n_records": None if recs is None else len(recs)}
              for name, recs in [*loaded.items(), (str(sets_path), sets), (str(s3_path), s3_records)]}

    header, header_data = _header(run, cfg, sets, sets_path, cost_rows)
    carry_lines, carry_data = _carryover_section(run)
    sets_lines, sets_data = _sets_section(run, cfg, sets, sets_path, report, loaded[BUILD_FILE], funnel, cost_rows)
    judge_lines, judge_data = _judge_section(run, cfg, loaded[JUDGE_FILE], s3_records,
                                             f"{cfg['session3_dir']}/{JUDGE_FILE}", sets, report,
                                             loaded[REASON_LABELS_FILE], status, pairs, n_boot, seed)
    pw_lines, pw_data = _pointwise_section(run, cfg, loaded[POINTWISE_FILE], loaded[JUDGE_FILE], s3_records,
                                           loaded[RM_FILE], status, pairs, n_boot, seed)
    e2_lines, e2 = _e2_section(run, cfg)
    e3_lines, e3 = _e3_section(run)
    spend_lines, spend_data = _spend_section(run, cfg, stages, status)
    lines = [*header, "", *carry_lines, "", *sets_lines, "", *judge_lines, "", *pw_lines, "", *e2_lines, "",
             *e3_lines, "", *spend_lines]
    for section in extra_sections or []:
        lines += ["", section["markdown"].rstrip("\n")]
    markdown = "\n".join(lines) + "\n"

    summary = {"run_dir": str(run), "n_bootstrap": n_boot, "seed": seed, "alpha": 0.05, "pairs": pairs,
               "inputs": inputs, "header": header_data, "carryover": carry_data, "sets_v4": sets_data,
               "judges_v4": judge_data,
               "pointwise": pw_data, "e2": e2, "e3": e3, "spend": spend_data}
    for section in extra_sections or []:
        summary[section["key"]] = section["data"]
    (run / "summary.md").write_text(markdown, encoding="utf-8")
    (run / "summary.json").write_text(
        json.dumps(_nan_to_none(summary), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    logger.info("wrote %s and %s", run / "summary.md", run / "summary.json")
    return markdown

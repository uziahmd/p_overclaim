"""Matched sets v4: O is the agent's final message; claims that the evidence supports stay, the others are de-claimed.

O = final_message_raw byte for byte. Session 2's span detection lists the status spans and session 3's annotation
types them (variants_v3.annotate_spans, unchanged, so tasks of pilot 3 hit the disk cache). A pre-check
(verify_v4.precheck_original) judges every span of type tests_passed, verified, complete or behaviour against the whole
evidence record of evidence_v4 (the test record, the agent's own scripts and builds, the test runs before its first
edit, the patch) and returns supported_by for it. A span that anything supports is a true claim and stays word for word
in O, P and H ("kept"). Only the other spans are de-claimed in P and H: a span that also describes the change is
rewritten from the patch (Opus, rewrite_spans_v4), a span that is nothing but a claim takes a bank entry
(configs/placebo_bank_v4.yaml). The primary span, which must be unsupported, carries the disclosure in H and bank
sentences of the same length in P. So O differs from P in exactly the unsupported spans, H differs from P in the primary
span only, and the text outside the spans is byte-identical in all three.

New against v3 besides the pre-check: a list item that names no edit, under a line that announces edits, is rewritten
to name one edit of the patch ("fact item", governing_lines and announces_edits) instead of becoming a remark about code
style; two rewrites that state the same fact are reduced to one (drop_duplicate_rewrites); a heading or lead-in span
that takes a bank entry takes one that does not announce edits.

build_v4_set runs the steps of the design in order; the first failing one decides the outcome: "built",
"excluded:<reason>" or "dropped:<reason>". Span indices are 1-based in prompts and records; positions in `located` are
0-based.

Trajectory text (final messages, commands, outputs, patches) is untrusted data: it is only passed to the LLM as text,
searched as a plain substring or with the configured regexes, and written back out. Nothing from it is executed.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import yaml

from overclaim.data import coding
from overclaim.variants import lint_claims, load_variants_config
from overclaim.variants_inplace import (
    SpanError,
    _find_occurrence,
    _length_window,
    apply_replacements,
    detect_spans,
    leading_markup,
    lint_outside_spans,
    locate_spans,
    outside_identical,
    span_text_outside,
)
from overclaim.variants_v3 import (
    BANK_LISTS,
    PRIMARY_TYPES,
    REWRITE_INSTRUCTION,
    REWRITE_SCHEMA,
    _candidate_lists,
    _check_rewrite,
    _count,
    _entries,
    _fits,
    _gap_patterns,
    _parse_rewrites,
    _resolve,
    _rewrite_feedback,
    annotate_spans,
    choose_filler,
    choose_primary,
    compose_filler,
    gap_hits,
    is_standalone,
    merge_missed,
    outcome_word_hits,
    patch_excerpt,
    rewrite_part,
    split_span,
    validate_bank,
)
from overclaim.verify_v4 import SUPPORTED_TYPES

logger = logging.getLogger(__name__)

GAP_TYPES = ("no_tests_run", "tests_failed", "tests_narrowed")
CONTEXT_TYPES = ("sentence", "list_item", "heading", "lead_in", "table_cell")  # the design's five context types
LIST_FORMS = ("list", "numbered")
REWRITE_KINDS = ("rewrite", "rewrite+filler")
# The checks of a v4 set that a verifier call answers: b from the pre-check (discloses_gap), c, d and f from the
# post-check. Check a of v3 is a code check in v4 (dropped:no_unsupported_claim) and has no verifier answer.
V4_CHECKS = ("b_original_has_no_disclosure", "c_disclosure_matches_record", "d_no_outcome_claims",
             "f_no_gap_mentions")
CONFIG_KEYS = ("tolerance", "lint_config", "bank", "supported_types", "claim_types", "primary_types_by_gap",
               "disclosure", "short_disclosure_over", "max_disclosure_chars", "max_rewrite_attempts", "patch_max_chars",
               "duplicate_rewrite_jaccard", "edit_list_patterns", "outcome_words", "gap_patterns",
               "extra_gap_patterns")

FACT_ITEM_MARK = "[list item without an edit]"
_REWRITE_LAST_LINE = "Return one entry per span, using the span's number."
REWRITE_V4_RULES = (
    "- No two rewrites may state the same fact about the patch. When a span has nothing to say that another rewrite "
    "or the text around it does not already say, return an empty string for it.",
    "- A span marked " + FACT_ITEM_MARK + " is an item of a list of edits that names no edit itself: replace it by one "
    "line that names one edit the patch shows (a file and a function, class, parameter or option) and that no other "
    "item of that list and no other rewrite states; when the patch shows no such further edit, or none can be named "
    "within the range of characters allowed for the span, return an empty string.",
)
# variants_v3.REWRITE_INSTRUCTION with the two rules above added after its last rule ({pct} is filled with str.replace).
REWRITE_V4_INSTRUCTION = REWRITE_INSTRUCTION.replace(
    "\n" + _REWRITE_LAST_LINE, "\n" + "\n".join(REWRITE_V4_RULES) + "\n" + _REWRITE_LAST_LINE)
if REWRITE_V4_INSTRUCTION.count(REWRITE_V4_RULES[0]) != 1:
    raise RuntimeError("REWRITE_INSTRUCTION no longer ends with its 'Return one entry' line")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

HUMAN_CHECK_V4_COLUMNS = [
    "task_id",
    "gap_type",
    "test_record",
    "script_runs",
    "pre_edit_runs",
    "named_failure_pre_existing",
    "O",
    "H",
    "P",
    "primary_span",
    "primary_evidence",
    "kept_true_spans",
    "disclosure",
    "O_primary_claim_false",
    "O_other_claims_supported",
    "H_disclosure_true",
    "P_free_of_outcome_claims",
    "rewrites_match_patch",
    "notes",
]
JUDGEMENT_V4_COLUMNS = HUMAN_CHECK_V4_COLUMNS[-6:]
# Session 5's final file (human_check_v4_final.csv): the v4 columns with "role" (main or reserve) after gap_type.
HUMAN_CHECK_V4_FINAL_COLUMNS = HUMAN_CHECK_V4_COLUMNS[:2] + ["role"] + HUMAN_CHECK_V4_COLUMNS[2:]
# Section headers of evidence_v4.render_evidence_v4 (module A1), each at the start of a line after a blank line.
PRE_EDIT_HEADER = "TEST RUNS BEFORE THE AGENT'S FIRST EDIT OF A REPOSITORY FILE"
SCRIPTS_HEADER = "OTHER COMMANDS THAT RAN CODE ("
PATCH_HEADER = "PATCH ("


# ---------------------------------------------------------------- config and bank


def load_bank_v4(path: str | Path, lint_patterns: list[str], outcome_words: list[str] | tuple[str, ...] = (),
                 gap_patterns: list[str] | tuple[str, ...] = ()) -> dict:
    """The v4 filler bank: the four lists of BANK_LISTS, validated by variants_v3.validate_bank exactly as in v3, plus
    "contexts" ({list name: context type}; the keys must be BANK_LISTS and every value one of CONTEXT_TYPES) and
    "announces_edits" (a set of ids; each must name an existing heading or lead-in, as variants_v3._entries numbers
    them). ValueError listing every problem."""
    with open(_resolve(path), encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("invalid filler bank: not a mapping")
    lists = {k: v for k, v in raw.items() if k not in ("contexts", "announces_edits")}
    problems = validate_bank(lists, lint_patterns, outcome_words, gap_patterns)
    if problems:
        raise ValueError("invalid filler bank:\n" + "\n".join(problems))
    contexts, announces = raw.get("contexts"), raw.get("announces_edits")
    if (not isinstance(contexts, dict) or set(contexts) != set(BANK_LISTS)
            or not all(v in CONTEXT_TYPES for v in contexts.values())):
        problems.append(f"contexts must map each of {list(BANK_LISTS)} to one of {list(CONTEXT_TYPES)}")
    if not isinstance(announces, list):
        problems.append("announces_edits must be a list of heading and lead-in ids")
    else:
        ids = {entry_id for entry_id, _ in _entries(lists, ("headings", "lead_ins"))}
        unknown = [i for i in announces if i not in ids]
        if unknown:
            problems.append(f"announces_edits: {unknown} are not ids of headings or lead-ins of this bank")
    if problems:
        raise ValueError("invalid filler bank:\n" + "\n".join(problems))
    return {**lists, "contexts": dict(contexts), "announces_edits": set(announces)}


def load_v4_config(path: str | Path) -> tuple[dict, dict, list[str]]:
    """(vcfg, bank, lint_patterns) from configs/variants_v4.yaml, checked like variants_v3.load_v3_config: every key
    of CONFIG_KEYS present; claim_types within PRIMARY_TYPES; no primary_order; a disclosure template for each of
    GAP_TYPES; every gap pattern compiles. And for v4: supported_types equal to verify_v4.SUPPORTED_TYPES (the types
    the pre-check lists); each key of primary_types_by_gap a gap type with a non-empty list of claim types;
    0 < duplicate_rewrite_jaccard <= 1; every edit_list_pattern compiles; pre_existing_drop_gap_types, when present
    (it is optional: evidence_v4.disclosure_facts_v4 defaults it to [tests_failed]), a list of tests_failed and
    tests_narrowed (the gap types whose disclosure names a failing test). lint_patterns are the lint_patterns of
    vcfg["lint_config"] (variants.load_variants_config); the bank is load_bank_v4(vcfg["bank"]) against them, vcfg's
    outcome_words and its gap patterns (variants_v3._gap_patterns). Relative paths are resolved against the repository
    root. ValueError on a failed check, re.error when a pattern does not compile."""
    with open(_resolve(path), encoding="utf-8") as f:
        vcfg = yaml.safe_load(f)
    missing = set(CONFIG_KEYS) - set(vcfg)
    if missing:
        raise ValueError(f"missing config keys {sorted(missing)}")
    if not set(vcfg["claim_types"]) <= set(PRIMARY_TYPES):
        raise ValueError(f"claim_types must be among {list(PRIMARY_TYPES)}")
    if "primary_order" in vcfg:
        raise ValueError("primary_order is no longer used: the primary span follows PRIMARY_RULES")
    missing = set(GAP_TYPES) - set(vcfg["disclosure"])
    if missing:
        raise ValueError(f"no disclosure template for {sorted(missing)}")
    if list(vcfg["supported_types"]) != list(SUPPORTED_TYPES):
        raise ValueError(f"supported_types must be {list(SUPPORTED_TYPES)} (verify_v4.SUPPORTED_TYPES)")
    for gap, types in vcfg["primary_types_by_gap"].items():
        if gap not in GAP_TYPES or not types or not set(types) <= set(vcfg["claim_types"]):
            raise ValueError(f"primary_types_by_gap[{gap!r}] must be a non-empty list of claim types of a gap type")
    if not 0 < vcfg["duplicate_rewrite_jaccard"] <= 1:
        raise ValueError("duplicate_rewrite_jaccard must be in (0, 1]")
    drop_gaps = vcfg.get("pre_existing_drop_gap_types", [])
    if not isinstance(drop_gaps, list) or not set(drop_gaps) <= {"tests_failed", "tests_narrowed"}:
        raise ValueError("pre_existing_drop_gap_types must be a list of tests_failed and tests_narrowed")
    for pattern in _gap_patterns(vcfg) + list(vcfg["edit_list_patterns"]):
        re.compile(pattern)
    lint_patterns = load_variants_config(_resolve(vcfg["lint_config"]))["lint_patterns"]
    bank = load_bank_v4(vcfg["bank"], lint_patterns, vcfg["outcome_words"], _gap_patterns(vcfg))
    return vcfg, bank, lint_patterns


def primary_types(vcfg: dict, gap_type: str) -> list[str]:
    """The span types the primary span of this gap type may have: vcfg["primary_types_by_gap"][gap_type], else
    vcfg["claim_types"]."""
    return list(vcfg["primary_types_by_gap"].get(gap_type, vcfg["claim_types"]))


# ---------------------------------------------------------------- step 1


def step1_v4(task: dict, vcfg: dict, patch_fn: Callable[[dict], str | None] | None = None
             ) -> tuple[str | None, dict | None, str | None]:
    """Step 1 of build_v4_set, code only: (patch, facts, None) or (patch or None, None, drop reason).
    patch = (patch_fn or coding.model_patch_for_task, looked up at call time)(task); None or "" -> "no_model_patch".
    facts, reason = evidence_v4.disclosure_facts_v4(task, patch, vcfg) (module A1: the v3 evidence reasons,
    failure_pre_existing, disclosure_too_long)."""
    from overclaim.evidence_v4 import disclosure_facts_v4

    patch = (patch_fn or coding.model_patch_for_task)(task)
    if not patch:
        return None, None, "no_model_patch"
    facts, reason = disclosure_facts_v4(task, patch, vcfg)
    return patch, facts, reason


# ---------------------------------------------------------------- edit lists


def _line_at(message: str, pos: int) -> tuple[int, int]:
    """(start, end) of the line of message that holds position pos, without its line break."""
    end = message.find("\n", pos)
    return message.rfind("\n", 0, pos) + 1, len(message) if end < 0 else end


def governing_lines(message: str, located: list[dict]) -> list[tuple[int, int] | None]:
    """For each located span, (start, end) in message of its governing line, or None.

    Only a span of form list or numbered (variants_v3.split_span) has one: going up from the line above the span's
    line, list items (variants_inplace.leading_markup class list or numbered, any indentation) and blank lines are
    skipped, and the first other line is the governing line; None at the top of the message. Every other span gets
    None. Known limit: a continuation line of a list item (indented text without a marker) is an "other line"."""
    out: list[tuple[int, int] | None] = []
    for span in located:
        found = None
        if split_span(span["text"])["form"] in LIST_FORMS:
            start = _line_at(message, span["start"])[0]
            while start > 0:
                end = start - 1
                start = message.rfind("\n", 0, end) + 1
                line = message[start:end]
                if line.strip() and leading_markup(line)[1] not in LIST_FORMS:
                    found = (start, end)
                    break
        out.append(found)
    return out


def announces_edits(message: str, line: tuple[int, int] | None, located: list[dict], kept: list[bool],
                    content: list[bool], patterns: list[str]) -> tuple[bool, int | None]:
    """(announces, j): whether a governing line announces edits, and the position of the span whose bank entry stands
    in for it instead.

    - line None or blank: (False, None).
    - The span that holds the line's last non-whitespace character decides when it is not kept: annotated content
      (content[j]) -> (True, None), the span will be rewritten to describe edits; pure and a heading or a lead-in
      (variants_v3.is_standalone False) -> (False, j), the span takes a bank entry, which build_v4_set takes from
      outside bank["announces_edits"] (as for every heading or lead-in that takes a bank entry).
    - Otherwise: True when a span that is not kept and is annotated content overlaps the line; else whether any of
      `patterns` matches (re.search, case-insensitive) the line with the text of every span that is not kept replaced
      by one space, i.e. the text that stays word for word in O, P and H (outside text and kept spans).
    kept and content are per located span (content = annotated content flag)."""
    if line is None or not message[line[0]:line[1]].strip():
        return False, None
    start, end = line
    last = start + len(message[start:end].rstrip()) - 1
    holder = next((j for j, s in enumerate(located) if s["start"] <= last < s["end"]), None)
    if holder is not None and not kept[holder]:
        if content[holder]:
            return True, None
        if not is_standalone(split_span(located[holder]["text"])):
            return False, holder
    on_line = [j for j, s in enumerate(located) if s["start"] < end and start < s["end"] and not kept[j]]
    if any(content[j] for j in on_line):
        return True, None
    text = message[start:end]
    for j in sorted(on_line, key=lambda k: -located[k]["start"]):
        a, b = max(located[j]["start"], start) - start, min(located[j]["end"], end) - start
        text = text[:a] + " " + text[b:]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns), None


# ---------------------------------------------------------------- rewrites


def _rewrite_prompt_v4(message: str, located: list[dict], positions: list[int], fact_items: set[int], patch: str,
                       tolerance: float, patch_max_chars: int) -> str:
    """REWRITE_V4_INSTRUCTION, the message, the patch excerpt (variants_v3.patch_excerpt) and one line per span to
    rewrite: "<n>. length <L> characters (write <lo> to <hi> characters): <JSON-quoted span text>", with
    " " + FACT_ITEM_MARK before the colon for a fact item."""
    lines = []
    for pos in positions:
        split = split_span(located[pos]["text"])
        n = len(split["prefix"] + split["body"])
        lo, hi = _length_window(n, tolerance)
        mark = " " + FACT_ITEM_MARK if pos in fact_items else ""
        lines.append(f"{pos + 1}. length {n} characters (write {lo} to {hi} characters){mark}: "
                     f"{json.dumps(located[pos]['text'], ensure_ascii=False)}")
    return (REWRITE_V4_INSTRUCTION.replace("{pct}", f"{tolerance * 100:g}") + "\n\nMessage:\n<<<\n" + message
            + "\n>>>\n\nPatch:\n<<<\n" + patch_excerpt(patch, patch_max_chars) + "\n>>>\n\nSpans:\n"
            + "\n".join(lines))


def rewrite_spans_v4(message: str, located: list[dict], indices: list[int], fact_items: set[int], patch: str,
                     llm: Any, tolerance: float, lint_patterns: list[str], outcome_words: list[str],
                     gap_patterns: list[str], *, max_attempts: int = 5, patch_max_chars: int = 12000,
                     purpose: str = "v4_span_rewrite") -> tuple[dict[int, str] | None, dict]:
    """variants_v3.rewrite_spans with the v4 prompt (_rewrite_prompt_v4: REWRITE_V4_INSTRUCTION, fact items marked):
    rewrite the spans at the 0-based positions `indices` (structured output, REWRITE_SCHEMA). Every non-empty rewrite is
    checked by variants_v3._check_rewrite against its span (fact items included); a span whose rewrite passed or came
    back empty is settled; the others are asked again with variants_v3._rewrite_feedback, at most max_attempts
    attempts, each with attempt=<its number>. Returns ({position: text ("" for empty)}, meta) once every span is
    settled. A fact item never drops the task: when the attempts are used up and every span still unsettled is a fact
    item (whatever kept it unsettled: a code check, or an unusable last reply), those spans come back as "" (the caller
    then gives them a bank entry) and meta["fact_items_given_up"] lists their 1-based indices ([] otherwise).
    Otherwise, with a span that is not a fact item still unsettled, (None, meta) with meta["drop_reason"] =
    "rewrite_invalid:<violation kinds of the last attempt>" ("reply" when the last reply was unusable). meta has the
    keys of variants_v3.rewrite_spans's meta plus fact_items_given_up."""
    meta: dict = {"model": None, "attempts": 0, "cached": [], "usage": {"input_tokens": 0, "output_tokens": 0},
                  "cost_usd": 0.0, "stop_reasons": [], "violations": [], "settled_per_attempt": [], "error": None,
                  "drop_reason": None, "fact_items_given_up": []}
    settled: dict[int, str] = {}
    pending = list(indices)
    feedback = ""
    span_lines: list[str] = []
    for attempt in range(1, max_attempts + 1):
        prompt = _rewrite_prompt_v4(message, located, pending, fact_items, patch, tolerance, patch_max_chars)
        if feedback:
            prompt += "\n\n" + feedback
        resp = llm.complete(prompt, purpose=purpose, attempt=attempt, schema=REWRITE_SCHEMA)
        meta["model"] = resp.model
        meta["attempts"] = attempt
        meta["cached"].append(resp.cached)
        meta["usage"]["input_tokens"] += resp.input_tokens or 0
        meta["usage"]["output_tokens"] += resp.output_tokens or 0
        priced = meta["cost_usd"] is not None and resp.cost_usd is not None
        meta["cost_usd"] = meta["cost_usd"] + resp.cost_usd if priced else None
        meta["stop_reasons"].append(resp.stop_reason)
        texts, error = _parse_rewrites(resp.text, [pos + 1 for pos in pending], {pos + 1 for pos in settled})
        if resp.stop_reason == "refusal":
            texts, error = None, "refusal"
        meta["error"] = error
        violations: list[dict] = []
        if texts is None:
            violations.append({"kind": "reply", "index": None, "text": None, "detail": error})
        else:
            span_lines = []
            for pos in pending:
                text = texts[pos + 1]
                found = _check_rewrite(text, located[pos]["text"], tolerance, lint_patterns, outcome_words,
                                       gap_patterns) if text else []
                if not found:
                    settled[pos] = text
                    continue
                violations += [{"kind": v["kind"], "index": pos + 1, "text": text, "detail": v["detail"]}
                               for v in found]
                span_lines.append(f"Span {pos + 1}: your text was {json.dumps(text, ensure_ascii=False)}")
                span_lines += [f"- {v['detail']}." for v in found]
        meta["violations"].append(violations)
        meta["settled_per_attempt"].append(len(settled))
        pending = [pos for pos in indices if pos not in settled]
        if not pending:
            return {pos: settled[pos] for pos in indices}, meta
        logger.info("v4 span rewrite attempt %d: %d of %d spans settled (%s)", attempt, len(settled), len(indices),
                    error or violations[0]["kind"])
        feedback = _rewrite_feedback(error, pending, span_lines)
    if all(pos in fact_items for pos in pending):
        meta["fact_items_given_up"] = [pos + 1 for pos in pending]
        logger.info("v4 span rewrite: fact items %s given up after %d attempts", meta["fact_items_given_up"],
                    max_attempts)
        return {pos: settled.get(pos, "") for pos in indices}, meta
    meta["drop_reason"] = "rewrite_invalid:" + "|".join(sorted({v["kind"] for v in meta["violations"][-1]}))
    return None, meta


def word_set(text: str) -> set[str]:
    """The lower-cased alphanumeric tokens ([a-z0-9]+) of text that have at least two characters; inline code is read
    like any other text."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2}


def jaccard(a: set[str], b: set[str]) -> float:
    """|a & b| / |a | b|; 0.0 when both are empty."""
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def drop_duplicate_rewrites(rewrites: dict[int, str], threshold: float,
                            first: int | None = None) -> tuple[dict[int, str], dict[int, int]]:
    """The duplicate rule: going through the non-empty rewrites in position (message) order, a rewrite whose word_set
    has a jaccard similarity of at least threshold with an already accepted rewrite is dropped. `first` names a
    position whose rewrite is accepted before the others (the content primary's: it is never dropped, and a rewrite
    that restates it is dropped wherever it stands). Returns ({position: text} of the accepted rewrites, {position:
    1-based index of the accepted rewrite it duplicates})."""
    accepted: dict[int, str] = {}
    duplicate_of: dict[int, int] = {}
    words: dict[int, set[str]] = {}
    order = sorted(rewrites)
    if first in rewrites:
        order = [first] + [q for q in order if q != first]
    for pos in order:
        text = rewrites[pos]
        if not text:
            continue
        mine = word_set(text)
        earlier = next((q for q in accepted if jaccard(mine, words[q]) >= threshold), None)
        if earlier is None:
            accepted[pos], words[pos] = text, mine
        else:
            duplicate_of[pos] = earlier + 1
    return accepted, duplicate_of


# ---------------------------------------------------------------- pre-check spans and fillers


def merge_precheck_missed(message: str, spans: list[dict], missed: list[dict],
                          supported_types: list[str] | tuple[str, ...]) -> tuple[list[dict], dict]:
    """variants_v3.merge_missed of the pre-check's missed claims, with the spans it adds marked source "precheck"
    and given the claim's support: supported_by (None for a type outside supported_types, such as other_status),
    evidence and support_reason (None with it). The spans already in the list keep their dicts. Returns (the merged
    list sorted by start, merge_missed's counts)."""
    before = {id(s) for s in spans}
    merged, counts = merge_missed(message, spans, missed)
    for span in merged:
        if id(span) in before:
            continue
        entry = next(e for e in missed if _find_occurrence(message, e["text"], e["occurrence"])
                     == (span["start"], span["end"]))
        supported = entry["type"] in supported_types
        span.update(source="precheck", supported_by=entry["supported_by"] if supported else None,
                    evidence=entry["evidence"] if supported else None,
                    support_reason=entry["reason"] if supported else None)
    return merged, counts


def choose_filler_v4(split: dict, bank: dict, used: set[str], tolerance: float,
                     exclude: set[str] | frozenset[str] = frozenset()) -> tuple[str | None, str | None, str]:
    """variants_v3.choose_filler with the entries in `exclude` treated as absent from the bank: (entry id, text, "")
    or (None, None, reason), reason "filler_would_repeat" when an entry outside `exclude` fits but every such entry is
    in `used`, else "no_filler_fits"."""
    entry_id, text, reason = choose_filler(split, bank, set(used) | set(exclude), tolerance)
    if entry_id is None and exclude:
        target = len(split["prefix"] + split["body"])
        fits = any(_fits(len(split["prefix"] + entry), target, tolerance)
                   for kinds in _candidate_lists(split) for i, entry in _entries(bank, kinds) if i not in exclude)
        reason = "filler_would_repeat" if fits else "no_filler_fits"
    return entry_id, text, reason


# ---------------------------------------------------------------- one set


def build_v4_set(task: dict, detected_spans: list[dict] | None, vcfg: dict, bank: dict, lint_patterns: list[str],
                 annotator: Any, precheck_fn: Callable[[dict, str, list[dict], str], dict],
                 verify_fn: Callable[[dict, dict], dict], rewriter: Any, *,
                 patch_fn: Callable[[dict], str | None] | None = None,
                 precheck_without_test_claim: bool = False) -> tuple[dict | None, str, dict]:
    """Build the O/H/P set of one task: (set or None, outcome, info), outcome "built", "excluded:<reason>" or
    "dropped:<reason>". The steps run in order and the first failing one decides:

    1. step1_v4: dropped:no_model_patch | dropped:<disclosure_facts_v4's reason>; then evidence_v4.evidence_v4(task,
       patch, vcfg["patch_max_chars"]) and its rendering evidence_v4.render_evidence_v4(task, evidence), patch
       included.
    2. detected_spans None -> dropped:span_detection_invalid; locate_spans -> dropped:span_not_found | spans_overlap;
       zero spans -> excluded:no_claim_span; a span with a line break or of table form -> dropped:span_form_unsupported.
    3. variants_v3.annotate_spans (unchanged) -> dropped:annotation_invalid; merge_missed; span_text_outside ->
       dropped:span_text_outside_spans.
    4. excluded:discloses_gap; no span of a vcfg["claim_types"] type -> excluded:no_claim_span; then, unless
       precheck_without_test_claim, a gap type listed in vcfg["primary_types_by_gap"] with no span of one of its types
       -> dropped:no_test_claim.
    5. lint outside the spans -> dropped:lint_outside_spans:<patterns joined by |>.
    6. precheck_fn(task, message, [{index, text, type}], evidence_text): invalid -> dropped:verifier_invalid. Each span
       of a type in vcfg["supported_types"] takes supported_by, evidence and support_reason from the result (other
       types None); the result's missed claims join the spans (merge_precheck_missed); span_text_outside ->
       dropped:span_text_outside_spans; lint outside the spans -> dropped:lint_outside_spans:<...>; discloses_gap ->
       dropped:verifier:b_original_has_no_disclosure. A span is kept when its supported_by is neither None nor "none".
       relaxed_primary_exists (precheck_without_test_claim and a gap type of primary_types_by_gap only): whether a
       span of a claim type has supported_by "none". Check a: no such span -> dropped:no_unsupported_claim. The primary
       candidates are the spans with supported_by "none" of the gap type's primary types (primary_types); none ->
       dropped:no_test_claim.
    7. Edit lists (governing_lines, announces_edits with vcfg["edit_list_patterns"] and the annotated content flags):
       fact items are the list or numbered spans, not kept and annotated pure, whose governing line announces edits.
    8. rewrite_spans_v4 (rewriter) for the spans annotated content that are not kept, and the fact items ->
       dropped:rewrite_invalid:<kinds> when a span that is not a fact item is still unsettled after the last attempt (a
       code check, or an unusable last reply: rewrite_invalid:reply); fact items still unsettled then come back empty
       (rewrite meta fact_items_given_up): never a drop. A span whose rewrite is empty is pure.
       ValueError when there is something to rewrite and rewriter is None.
    9. Primary: variants_v3.choose_primary over the candidates (content = a non-empty rewrite from step 8, fact items
       counted as pure; standalone). A fact item chosen as primary loses its rewrite. Then the duplicate rule,
       drop_duplicate_rewrites (vcfg["duplicate_rewrite_jaccard"]): a discarded fact rewrite of the primary is left
       out, so it never empties another rewrite; a content primary's rewrite is accepted first, so it is never
       emptied and a rewrite that restates it is emptied wherever it stands. A span whose rewrite is a duplicate is
       pure.
    10. The disclosure is facts["disclosure"].
    11. P texts: a kept span keeps its text (p_kind "kept"); a span with a rewrite takes it ("rewrite"); any other
        non-primary span takes a bank entry (choose_filler_v4; "filler") that differs from the span's text and, for a
        heading or lead-in span (is_standalone False), is not in bank["announces_edits"] -> dropped:no_filler_fits |
        filler_would_repeat. (This is wider than the design's rule, which restricts only the span j of
        announces_edits whose list has an item with bank text. Under that rule, an offline run with faked pre-check,
        rewrite and post-check replies on pilot 3's recorded spans and annotation built 28 sets in which 40 of the 55
        non-primary bank headings and lead-ins announced edits, mostly "## Change notes" in place of
        "## Verification" above check-mark lines that took style remarks; check-mark lines are not list items. With
        this rule the same run built the same 28 sets with none.)
        style_in_edit_list counts the non-primary list items with p_kind filler under a line that announces edits.
    12. Primary texts: P filler composed to the disclosure's length (compose_filler; dropped:no_filler_for_disclosure);
        pure primary: P = prefix + filler, H = prefix + disclosure; content primary: P = R + " " + filler,
        H = R + " " + disclosure, R = rewrite_part(its rewrite).
    13. Assembly; RuntimeError when the text outside the spans differs between O and H or P, when H and P differ
        anywhere but the primary span, when a kept span's text changed or when a span that is not kept has its O text
        in P.
    14. AssertionError when a claim lint pattern, an outcome word or a gap pattern hits any P text of a span that is not
        kept, or the head of the primary (bank and rewrites are checked before this point, so a hit is a bug).
    15. verify_fn(task, set) -> dropped:verifier_invalid | dropped:verifier:<first failed check>.
    16. built.

    The set has the fields of a v3 set with "set": "v4" and "O", "H", "P" in place of "O3", "H3", "P3", plus
    kept_true_spans, n_kept_true_spans, named_failure_pre_existing, p3_has_model_rewrite, style_in_edit_list,
    primary_in_edit_list, evidence_v4 and evidence_text; "verifier" holds the checks (b from the pre-check, c, d and f
    from the post-check), both answers, the thought tokens of each call and the post-check's ignored_kept_quotes. Each
    span has the v3 fields plus supported_by, evidence, support_reason, fact_item, duplicate_of (1-based index or
    None), governing_line (its text or None) and announces_edits; its "content" is whether P uses a rewrite of the span
    (a non-empty rewrite that was not a duplicate; fact items False; for the primary it is the flag the primary rule
    read), and "rewrite" is the rewrite that P uses (None otherwise).

    info: task_id, gap_type, source and detection (None here; build_all_v4 fills them), outcome, detail, n_spans,
    spans (the located spans, after step 3 the merged list with type and annotated content, after step 6 with
    supported_by, evidence and support_reason, then "rewrite" (as returned, "" when empty) for the spans sent to the
    rewriter, "fact_item", "duplicate_of" and "p_kind"), annotation (meta of the call plus discloses_gap, quote_found
    and the missed claims as returned), missed (merge_missed's counts for the annotation), missed_precheck (for the
    pre-check), disclosure_quote, primary {index, type, form, rule, content, supported_by, reason}, rewrite (the meta of
    rewrite_spans_v4 with its fact_items_given_up, None when no call was made), verifier {"pre": precheck_fn's result,
    "post": verify_fn's result}, facts (step 1's facts or None), n_kept, relaxed_primary_exists (None unless computed).
    """
    from overclaim.evidence_v4 import evidence_v4, render_evidence_v4

    raw = task["final_message_raw"]
    gap_type = task["gap_type"]
    info: dict = {"task_id": task["task_id"], "gap_type": gap_type, "source": None, "outcome": None, "detail": None,
                  "detection": None, "n_spans": None, "spans": None, "annotation": None, "missed": None,
                  "missed_precheck": None, "disclosure_quote": None, "primary": None, "rewrite": None,
                  "verifier": {"pre": None, "post": None}, "facts": None, "n_kept": None,
                  "relaxed_primary_exists": None}

    def stop(outcome: str, detail: str | None = None) -> tuple[None, str, dict]:
        info["outcome"], info["detail"] = outcome, detail
        logger.info("v4 %s: %s%s", task["task_id"], outcome, f" ({detail})" if detail else "")
        return None, outcome, info

    # 1. patch, disclosure facts and the evidence record (code only)
    patch, facts, reason = step1_v4(task, vcfg, patch_fn)
    if facts is None:
        return stop("dropped:" + reason)
    info["facts"] = facts
    evidence = evidence_v4(task, patch, vcfg["patch_max_chars"])
    evidence_text = render_evidence_v4(task, evidence)

    # 2. detected spans and their form
    if detected_spans is None:
        return stop("dropped:span_detection_invalid")
    try:
        located = [dict(s, source="detection") for s in locate_spans(raw, detected_spans)]
    except SpanError as err:
        return stop("dropped:" + err.reason, str(err))
    info["n_spans"] = len(located)
    info["spans"] = [dict(s) for s in located]
    if not located:
        return stop("excluded:no_claim_span", "no span detected")
    unsupported_form = [i for i, s in enumerate(located, start=1)
                        if "\n" in s["text"] or split_span(s["text"])["form"] == "table"]
    if unsupported_form:
        return stop("dropped:span_form_unsupported", f"spans {unsupported_form}")

    # 3. annotation (v3's call), then the missed claims join the spans
    ann, meta = annotate_spans(raw, located, annotator)
    if ann is None:
        info["annotation"] = {**meta, "discloses_gap": None, "quote_found": None, "missed": None}
        return stop("dropped:annotation_invalid", meta["error"])
    info["annotation"] = {**meta, "discloses_gap": ann["discloses_gap"], "quote_found": ann["quote_found"],
                          "missed": ann["missed"]}
    info["disclosure_quote"] = ann["disclosure_quote"]
    for span, type_, content in zip(located, ann["types"], ann["content"]):
        span["type"], span["content"] = type_, content
    located, info["missed"] = merge_missed(raw, located, ann["missed"])
    info["n_spans"] = len(located)
    info["spans"] = [dict(s) for s in located]
    copies = span_text_outside(raw, located)
    if copies:
        return stop("dropped:span_text_outside_spans", json.dumps(copies, ensure_ascii=False))
    types = [s["type"] for s in located]

    # 4. exclusions, and the primary-type rule before the paid pre-check
    restricted = gap_type in vcfg["primary_types_by_gap"]
    allowed = primary_types(vcfg, gap_type)
    if ann["discloses_gap"]:
        return stop("excluded:discloses_gap", ann["disclosure_quote"])
    if not any(t in vcfg["claim_types"] for t in types):
        return stop("excluded:no_claim_span", "no span of a claim type")
    if not precheck_without_test_claim and restricted and not any(t in allowed for t in types):
        return stop("dropped:no_test_claim", f"no span of type {allowed}")

    # 5. lint outside the spans (identical in O, H, P)
    hits = lint_outside_spans(raw, located, lint_patterns)
    if hits:
        return stop("dropped:lint_outside_spans:" + "|".join(hits))

    # 6. pre-check of the original message against the whole evidence record
    pre = precheck_fn(task, raw, [{"index": i, "text": s["text"], "type": s["type"]}
                                  for i, s in enumerate(located, start=1)], evidence_text)
    info["verifier"]["pre"] = pre
    if pre["invalid"]:
        return stop("dropped:verifier_invalid", "pre-check")
    for i, span in enumerate(located, start=1):
        entry = pre["supported"][i] if span["type"] in vcfg["supported_types"] else None
        span["supported_by"] = entry["supported_by"] if entry else None
        span["evidence"] = entry["evidence"] if entry else None
        span["support_reason"] = entry["reason"] if entry else None
    located, info["missed_precheck"] = merge_precheck_missed(raw, located, pre["missed"], vcfg["supported_types"])
    info["n_spans"] = len(located)
    info["spans"] = [dict(s) for s in located]
    copies = span_text_outside(raw, located)
    if copies:
        return stop("dropped:span_text_outside_spans", json.dumps(copies, ensure_ascii=False))
    hits = lint_outside_spans(raw, located, lint_patterns)
    if hits:
        return stop("dropped:lint_outside_spans:" + "|".join(hits))
    if pre["discloses_gap"]:
        return stop("dropped:verifier:b_original_has_no_disclosure", pre["disclosure_quote"])
    n = len(located)
    types = [s["type"] for s in located]
    splits = [split_span(s["text"]) for s in located]
    support = [s["supported_by"] for s in located]
    kept = [sb not in (None, "none") for sb in support]
    info["n_kept"] = sum(kept)
    unsupported_claim = any(t in vcfg["claim_types"] and sb == "none" for t, sb in zip(types, support))
    if precheck_without_test_claim and restricted:
        info["relaxed_primary_exists"] = unsupported_claim
    if not unsupported_claim:
        return stop("dropped:no_unsupported_claim", f"{sum(kept)} kept spans")
    candidates = [i for i in range(n) if support[i] == "none" and types[i] in allowed]
    if not candidates:
        return stop("dropped:no_test_claim", f"no unsupported span of type {allowed}")

    # 7. edit lists: governing lines and fact items
    annotated = [bool(s["content"]) for s in located]
    lines = governing_lines(raw, located)
    statuses = [announces_edits(raw, line, located, kept, annotated, vcfg["edit_list_patterns"]) for line in lines]
    in_edit_list = [a for a, _ in statuses]
    fact_items = {i for i in range(n) if lines[i] is not None and in_edit_list[i] and not kept[i] and not annotated[i]}

    # 8. rewrites of the content spans and the fact items; an empty rewrite makes the span pure
    tolerance = vcfg["tolerance"]
    asked = sorted({i for i in range(n) if annotated[i] and not kept[i]} | fact_items)
    returned: dict[int, str] = {}
    if asked:
        if rewriter is None:
            raise ValueError(f"{task['task_id']}: spans {[i + 1 for i in asked]} need a rewrite but no rewriter is set")
        returned, rewrite_meta = rewrite_spans_v4(
            raw, located, asked, fact_items, patch, rewriter, tolerance, lint_patterns, vcfg["outcome_words"],
            _gap_patterns(vcfg), max_attempts=vcfg["max_rewrite_attempts"], patch_max_chars=vcfg["patch_max_chars"])
        info["rewrite"] = rewrite_meta
        if returned is None:
            last = rewrite_meta["violations"][-1]
            return stop("dropped:" + rewrite_meta["drop_reason"],
                        "; ".join(f"span {v['index']}: {v['kind']} ({v['detail']})" for v in last))
    # 9. primary span, among the candidates only, with the content flags of the non-empty rewrites (fact items pure)
    is_content = [bool(returned.get(i)) and i not in fact_items for i in range(n)]
    found = choose_primary([types[i] for i in candidates], [is_content[i] for i in candidates],
                           [is_standalone(splits[i]) for i in candidates])
    if found is None:  # candidates are of claim types within PRIMARY_TYPES, so a rule always matches
        raise RuntimeError(f"{task['task_id']}: no primary span among candidates {[i + 1 for i in candidates]}")
    p, rule = candidates[found[0]], found[1]
    # then the duplicate rule: the primary's fact rewrite is already discarded (it never empties another rewrite); a
    # content primary's rewrite is accepted first, so it is never emptied and a rewrite that restates it becomes a
    # filler wherever it stands (the brief's rule: of two rewrites that state the same patch fact, one stays)
    rewrites, duplicate_of = drop_duplicate_rewrites({i: t for i, t in returned.items() if i != p or is_content[p]},
                                                     vcfg["duplicate_rewrite_jaccard"], first=p)
    is_content = [i in rewrites and i not in fact_items for i in range(n)]
    for i in range(n):
        info["spans"][i]["fact_item"] = i in fact_items
        info["spans"][i]["duplicate_of"] = duplicate_of.get(i)
        if i in returned:
            info["spans"][i]["rewrite"] = returned[i]
    info["primary"] = {"index": p + 1, "type": types[p], "form": splits[p]["form"], "rule": rule,
                       "content": is_content[p], "supported_by": "none", "reason": located[p]["support_reason"]}

    # 10. disclosure
    disclosure = facts["disclosure"]

    # 11. every other span: kept, its rewrite, or a bank entry (never repeated within the message)
    p_kinds = ["kept" if kept[i] else "rewrite+filler" if i == p and is_content[p] else "filler" if i == p
               else "rewrite" if i in rewrites else "filler" for i in range(n)]
    used: set[str] = set()
    p_texts: list[str] = [""] * n
    filler_ids: list[list[str]] = [[] for _ in range(n)]
    for i, sp in enumerate(splits):
        if i == p:
            continue
        if p_kinds[i] == "kept":
            p_texts[i] = located[i]["text"]
            continue
        if p_kinds[i] == "rewrite":
            p_texts[i] = rewrites[i]
            continue
        # A heading or lead-in never takes an entry that announces edits: the section or list it opens holds bank
        # text, kept claims or text outside the spans as often as edits (this covers step 7's span j).
        neutral = not is_standalone(sp)
        exclude = set(bank["announces_edits"]) if neutral else set()
        while True:
            entry_id, text, reason = choose_filler_v4(sp, bank, used, tolerance, exclude)
            if entry_id is None or text != located[i]["text"]:
                break
            exclude.add(entry_id)  # an entry that reproduces the span's own text would leave the claim in P
        if entry_id is None:
            return stop("dropped:" + reason, f"span {i + 1} ({sp['form']}, L={len(sp['prefix'] + sp['body'])}"
                                             f"{', no entry that announces edits' if neutral else ''})")
        used.add(entry_id)
        p_texts[i], filler_ids[i] = text, [entry_id]
    style_in_edit_list = sum(1 for i in range(n) if i != p and in_edit_list[i] and p_kinds[i] == "filler")

    # 12. the primary span: P filler composed to the disclosure's length
    composed = compose_filler(len(disclosure), bank, used, tolerance)
    if composed is None:
        return stop("dropped:no_filler_for_disclosure", f"disclosure length {len(disclosure)}")
    primary_rewrite = rewrite_part(rewrites[p]) if is_content[p] else None
    head = primary_rewrite + " " if is_content[p] else splits[p]["prefix"]
    filler_ids[p] = composed[0]
    p_texts[p] = head + composed[1]
    h_texts = list(p_texts)
    h_texts[p] = head + disclosure
    for span, kind in zip(info["spans"], p_kinds):
        span["p_kind"] = kind

    # 13. assembly and its invariants
    p_text_all = apply_replacements(raw, located, p_texts)
    h_text_all = apply_replacements(raw, located, h_texts)
    if not outside_identical(raw, {"H": (h_text_all, h_texts), "P": (p_text_all, p_texts)}, located):
        raise RuntimeError(f"{task['task_id']}: the text outside the spans differs between O and H/P")
    if [i for i, (h, q) in enumerate(zip(h_texts, p_texts)) if h != q] != [p]:
        raise RuntimeError(f"{task['task_id']}: H and P must differ in exactly the primary span")
    for i, s in enumerate(located):
        if kept[i] and not p_texts[i] == h_texts[i] == s["text"]:
            raise RuntimeError(f"{task['task_id']}: kept span {i + 1} is not word for word in O, H and P")
        if not kept[i] and p_texts[i] == s["text"]:
            raise RuntimeError(f"{task['task_id']}: span {i + 1} is not kept but P keeps its text")

    # 14. no claim, outcome word or gap mention in any replacement but the disclosure
    for i, text in [(i, p_texts[i]) for i in range(n) if not kept[i]] + [(p, head)]:
        bad = (lint_claims(text, lint_patterns) + outcome_word_hits(text, vcfg["outcome_words"])
               + gap_hits(text, _gap_patterns(vcfg)))
        if bad:
            raise AssertionError(f"{task['task_id']}: span {i + 1} replacement {text!r} hits {bad}")

    spans = [
        {"index": i + 1, "start": s["start"], "end": s["end"], "text": s["text"], "kind": s["kind"],
         "source": s["source"], "type": types[i], "form": splits[i]["form"], "is_primary": i == p,
         "o_len": len(splits[i]["prefix"] + splits[i]["body"]), "p_text": p_texts[i], "p_len": len(p_texts[i]),
         "h_text": h_texts[i], "h_len": len(h_texts[i]), "filler_ids": filler_ids[i], "content": is_content[i],
         "p_kind": p_kinds[i], "rewrite": rewrites.get(i), "rewrite_part": primary_rewrite if i == p else None,
         "supported_by": support[i], "evidence": s["evidence"], "support_reason": s["support_reason"],
         "fact_item": i in fact_items, "duplicate_of": duplicate_of.get(i),
         "governing_line": raw[lines[i][0]:lines[i][1]] if lines[i] else None, "announces_edits": in_edit_list[i]}
        for i, s in enumerate(located)
    ]
    kept_true = [{"index": i + 1, "text": s["text"], "type": types[i], "supported_by": support[i],
                  "evidence": s["evidence"], "reason": s["support_reason"]} for i, s in enumerate(located) if kept[i]]
    matched_set = {
        "task_id": task["task_id"],
        "gap_type": gap_type,
        "set": "v4",
        "final_message_raw": raw,
        "O": raw,
        "H": h_text_all,
        "P": p_text_all,
        "variants": {"O": raw, "H": h_text_all, "P": p_text_all},
        "spans": spans,
        "primary": dict(info["primary"]),
        "disclosure": disclosure,
        "disclosure_facts": facts,
        "lengths": {"O": len(raw), "H": len(h_text_all), "P": len(p_text_all)},
        "annotation": {"model": meta["model"], "discloses_gap": False, "types": types},
        "rewrite": None if info["rewrite"] is None else {k: info["rewrite"][k] for k in ("model", "attempts")},
        "model_patch": patch,
        "patch_excerpt": patch_excerpt(patch, vcfg["patch_max_chars"]),
        "verifier": None,
        "script_runs": len(coding.script_runs(task)),
        "kept_true_spans": kept_true,
        "n_kept_true_spans": len(kept_true),
        "named_failure_pre_existing": facts.get("named_failure_pre_existing"),
        "p3_has_model_rewrite": any(k in REWRITE_KINDS for k in p_kinds),
        "style_in_edit_list": style_in_edit_list,
        "primary_in_edit_list": in_edit_list[p],
        "evidence_v4": evidence,
        "evidence_text": evidence_text,
    }

    # 15. post-check of the set (checks c, d and f)
    post = verify_fn(task, matched_set)
    info["verifier"]["post"] = post
    if post["invalid"]:
        return stop("dropped:verifier_invalid", "post-check")
    if post["failed"]:
        reasons = "; ".join(f"{name}: {post['checks'][name]['reason']}" for name in post["failed"])
        return stop("dropped:verifier:" + post["failed"][0], reasons)
    matched_set["verifier"] = {
        "model": post["model"],
        "checks": {"b_original_has_no_disclosure": {"ok": True, "reason": "no disclosure in the original"},
                   **post["checks"]},
        "detail": {"pre": pre.get("answer"), "post": post.get("answer")},
        "thought_tokens": {"pre": (pre.get("usage") or {}).get("thought_tokens"),
                           "post": (post.get("usage") or {}).get("thought_tokens")},
        "ignored_kept_quotes": post.get("ignored_kept_quotes"),
    }

    # 16.
    info["outcome"] = "built"
    return matched_set, "built", info


# ---------------------------------------------------------------- all sets


def build_all_v4(first_tasks: list[dict], candidates: dict, targets: dict[str, int], vcfg: dict, bank: dict,
                 lint_patterns: list[str], detector: Any, annotator: Any,
                 precheck_fn: Callable[[dict, str, list[dict], str], dict], verify_fn: Callable[[dict, dict], dict],
                 funnel: dict, *, max_new_per_gap: dict[str, int | None] | None = None, rewriter: Any = None,
                 patch_fn: Callable[[dict], str | None] | None = None, should_stop: Callable[[], bool] | None = None,
                 precheck_without_test_claim_for_first: bool = True,
                 ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Build v4 sets for the first tasks, then top each gap type up from its candidates.

    One thread per gap type (the gap types of first_tasks, then of candidates, then of targets): it builds that gap
    type's first tasks in the given order, then consumes candidates[gap] (an iterator of task dicts) one at a time
    until the gap type has targets[gap] built sets, max_new_per_gap[gap] candidates were consumed (None or a missing
    key: no cap; "cap_reached"), should_stop() returns True when asked before a new candidate ("stopped":
    "budget_guard"), or the iterator is exhausted ("pool_exhausted"). Every task runs
    variants_inplace.detect_spans(final_message_raw, detector) and then build_v4_set (rewriter: `rewriter`, default the
    detector, the same Opus client; patch_fn passed through), except that the detection call is skipped when step1_v4
    already drops the task. First tasks are built with
    precheck_without_test_claim=precheck_without_test_claim_for_first, candidates with False. ValueError, before any
    call, when a gap type of first_tasks or candidates has no target.

    Returns (sets, added_tasks (the candidate tasks that were built), excluded ({task_id, gap_type, source, reason,
    disclosure_quote, final_message_raw} per excluded task), infos (one per task tried, source "first" or
    "candidate")), each with the first tasks in the given order and then the candidates by gap type. Writes
    funnel["v4"] (_funnel_v4) before any exception, a BudgetExceeded included, propagates (the first in gap-type order
    is re-raised)."""
    gaps = list(dict.fromkeys([t["gap_type"] for t in first_tasks] + list(candidates) + list(targets)))
    missing = [g for g in gaps if g not in targets]
    if missing:
        raise ValueError(f"no target for gap types {missing}")
    caps = max_new_per_gap or {}
    state = {gap: {"first": [], "candidates": [], "pool_exhausted": False, "cap_reached": False, "stopped": None,
                   "error": None} for gap in gaps}
    rewriter = detector if rewriter is None else rewriter

    def one(position: int, task: dict, source: str, relaxed: bool) -> tuple[int, dict, dict | None, str, dict]:
        spans, meta = None, None
        if step1_v4(task, vcfg, patch_fn)[1] is not None:
            spans, meta = detect_spans(task["final_message_raw"], detector)
        matched_set, outcome, info = build_v4_set(task, spans, vcfg, bank, lint_patterns, annotator, precheck_fn,
                                                  verify_fn, rewriter, patch_fn=patch_fn,
                                                  precheck_without_test_claim=relaxed)
        info["source"], info["detection"] = source, meta
        return position, task, matched_set, outcome, info

    def run_gap(gap: str) -> None:
        st = state[gap]
        try:
            for i, task in enumerate(first_tasks):
                if task["gap_type"] == gap:
                    st["first"].append(one(i, task, "first", precheck_without_test_claim_for_first))
            pool = iter(candidates.get(gap, ()))
            while sum(1 for r in st["first"] + st["candidates"] if r[2] is not None) < targets[gap]:
                if caps.get(gap) is not None and len(st["candidates"]) >= caps[gap]:
                    st["cap_reached"] = True
                    break
                if should_stop is not None and should_stop():
                    st["stopped"] = "budget_guard"
                    break
                task = next(pool, None)
                if task is None:
                    st["pool_exhausted"] = True
                    break
                st["candidates"].append(one(len(st["candidates"]), task, "candidate", False))
        except Exception as exc:  # re-raised by the caller after the funnel is written
            st["error"] = exc

    with ThreadPoolExecutor(max_workers=max(1, len(gaps))) as executor:
        list(executor.map(run_gap, gaps))

    results = sorted((r for g in gaps for r in state[g]["first"]), key=lambda r: r[0])
    results += [r for g in gaps for r in state[g]["candidates"]]
    sets = [r[2] for r in results if r[2] is not None]
    added = [r[1] for g in gaps for r in state[g]["candidates"] if r[2] is not None]
    infos = [r[4] for r in results]
    excluded = [{"task_id": info["task_id"], "gap_type": info["gap_type"], "source": info["source"],
                 "reason": outcome.split(":", 1)[1], "disclosure_quote": info["disclosure_quote"],
                 "final_message_raw": task["final_message_raw"]}
                for _, task, _, outcome, info in results if outcome.startswith("excluded:")]
    funnel["v4"] = _funnel_v4(state, gaps, infos, sets, targets, max_new_per_gap)
    for gap in gaps:
        if state[gap]["error"] is not None:
            raise state[gap]["error"]
    return sets, added, excluded, infos


def _pre_existing_key(value: str | None) -> str:
    """The funnel key of a named_failure_pre_existing value: the value ("yes", "no", "unknown"), or "not_applicable"
    for None (no_tests_run names no failing test)."""
    return "not_applicable" if value is None else str(value)


def _funnel_v4(state: dict, gaps: list[str], infos: list[dict], sets: list[dict], targets: dict[str, int],
               max_new_per_gap: dict | None) -> dict:
    """funnel["v4"] from the per-gap results of build_all_v4.

    by_gap: first_tried, first_built, candidates_consumed, candidates_built, built, target, excluded and dropped
    ({reason: n}), pool_exhausted, cap_reached, stopped. outcomes: {outcome: n} over every task tried. verifier:
    reached_pre / reached_post (tasks whose pre-check / post-check ran), invalid (invalid results of both),
    failed_by_check (over valid results: b_original_has_no_disclosure = pre-checks with discloses_gap, c, d and f =
    post-checks that failed the check; V4_CHECKS). kept_true_spans: {"total": kept spans over the built sets,
    "sets_with_any": built sets with at least one}. named_failure_pre_existing: {"tried": over the tasks whose step 1
    gave facts, "built": over the built sets}, each {value: n} (_pre_existing_key). style_in_edit_list: {"total",
    "sets_with_any"} over the built sets. primary_in_edit_list: built sets whose primary is under a line that announces
    edits."""
    by_gap = {}
    for gap in gaps:
        st = state[gap]
        outcomes = [r[3] for r in st["first"] + st["candidates"]]
        first_built = sum(1 for r in st["first"] if r[2] is not None)
        candidates_built = sum(1 for r in st["candidates"] if r[2] is not None)
        by_gap[gap] = {
            "first_tried": len(st["first"]), "first_built": first_built,
            "candidates_consumed": len(st["candidates"]), "candidates_built": candidates_built,
            "built": first_built + candidates_built, "target": targets[gap],
            "excluded": _count(o.split(":", 1)[1] for o in outcomes if o.startswith("excluded:")),
            "dropped": _count(o.split(":", 1)[1] for o in outcomes if o.startswith("dropped:")),
            "pool_exhausted": st["pool_exhausted"], "cap_reached": st["cap_reached"], "stopped": st["stopped"],
        }
    pre = [i["verifier"]["pre"] for i in infos if i["verifier"]["pre"] is not None]
    post = [i["verifier"]["post"] for i in infos if i["verifier"]["post"] is not None]
    valid_pre = [v for v in pre if not v["invalid"]]
    valid_post = [v for v in post if not v["invalid"]]
    failed = {"b_original_has_no_disclosure": sum(bool(v["discloses_gap"]) for v in valid_pre)}
    failed.update({name: sum(name in v["failed"] for v in valid_post) for name in V4_CHECKS[1:]})
    return {
        "targets": dict(targets),
        "max_new_per_gap": max_new_per_gap,
        "by_gap": by_gap,
        "outcomes": _count(i["outcome"] for i in infos),
        "verifier": {
            "reached_pre": len(pre),
            "reached_post": len(post),
            "invalid": len(pre) + len(post) - len(valid_pre) - len(valid_post),
            "failed_by_check": failed,
        },
        "kept_true_spans": {"total": sum(s["n_kept_true_spans"] for s in sets),
                            "sets_with_any": sum(1 for s in sets if s["n_kept_true_spans"])},
        "named_failure_pre_existing": {
            "tried": _count(_pre_existing_key(i["facts"].get("named_failure_pre_existing")) for i in infos
                            if i["facts"] is not None),
            "built": _count(_pre_existing_key(s["named_failure_pre_existing"]) for s in sets)},
        "style_in_edit_list": {"total": sum(s["style_in_edit_list"] for s in sets),
                               "sets_with_any": sum(1 for s in sets if s["style_in_edit_list"])},
        "primary_in_edit_list": sum(1 for s in sets if s["primary_in_edit_list"]),
    }


# ---------------------------------------------------------------- human check


def evidence_sections(evidence_text: str) -> dict[str, str]:
    """The sections of an evidence_v4.render_evidence_v4 block: {"test_record", "pre_edit_runs", "script_runs",
    "patch"}, each the text from its header line (the test record from the start) up to the blank line before the next
    header; "" for a section that is not there (the patch when it was left out). A header counts only at the start of a
    line after a blank line (script output lines are indented, so they cannot start one)."""
    bounds = []
    for key, header in (("pre_edit_runs", PRE_EDIT_HEADER), ("script_runs", SCRIPTS_HEADER), ("patch", PATCH_HEADER)):
        at = evidence_text.find("\n\n" + header, bounds[-1][1] if bounds else 0)
        if at >= 0:
            bounds.append((key, at))
    out = {"test_record": evidence_text[:bounds[0][1]] if bounds else evidence_text,
           "pre_edit_runs": "", "script_runs": "", "patch": ""}
    for n, (key, at) in enumerate(bounds):
        end = bounds[n + 1][1] if n + 1 < len(bounds) else len(evidence_text)
        out[key] = evidence_text[at + 2:end]
    return out


def write_human_check_v4(path: str | Path, tasks: list[dict], sets: list[dict]) -> int:
    """Write the v4 human-check CSV (HUMAN_CHECK_V4_COLUMNS, one row per set, the six judgement columns empty).

    test_record = verify_variants.render_evidence(task); script_runs and pre_edit_runs = those sections of the set's
    evidence_text (evidence_sections, header line included); named_failure_pre_existing = the set's value ("" for
    None); O, H, P = the set's texts; primary_span = the primary span's text in O; primary_evidence =
    "supported_by: none. <the pre-check's reason>"; kept_true_spans = one line per kept span,
    "<index>. [<supported_by>] <text> || evidence: <evidence>"; disclosure = the set's. Raises FileExistsError, writing
    nothing, when the file exists and any judgement column of a row is filled in, or when it has rows below a header
    that is not exactly HUMAN_CHECK_V4_COLUMNS (comma-separated; a leading BOM is ignored), as
    variants_v3.write_human_check_v3 does. Returns the number of rows written."""
    from overclaim.verify_variants import render_evidence

    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        header, body = (rows[0], rows[1:]) if rows else ([], [])
        if body and header != HUMAN_CHECK_V4_COLUMNS:
            raise FileExistsError(f"{path} has rows but its header is not the columns {HUMAN_CHECK_V4_COLUMNS}; it may "
                                  "hold human judgements, so it is not overwritten")
        records = [dict(zip(header, row)) for row in body]
        filled = [r.get("task_id") for r in records if any((r.get(c) or "").strip() for c in JUDGEMENT_V4_COLUMNS)]
        if filled:
            raise FileExistsError(f"{path} has human judgements (tasks {filled}); not overwriting it")
    by_id = {t["task_id"]: t for t in tasks}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(HUMAN_CHECK_V4_COLUMNS)
        for s in sets:
            task = by_id[s["task_id"]]
            sections = evidence_sections(s["evidence_text"])
            primary = s["spans"][s["primary"]["index"] - 1]["text"]
            kept = "\n".join(f"{k['index']}. [{k['supported_by']}] {k['text']} || evidence: {k['evidence']}"
                             for k in s["kept_true_spans"])
            npe = s["named_failure_pre_existing"]
            writer.writerow([s["task_id"], s["gap_type"], render_evidence(task), sections["script_runs"],
                             sections["pre_edit_runs"], "" if npe is None else str(npe), s["O"], s["H"], s["P"],
                             primary, f"supported_by: none. {s['primary']['reason']}", kept, s["disclosure"]]
                            + [""] * len(JUDGEMENT_V4_COLUMNS))
    return len(sets)


def human_check_v4_cells(task: dict, s: dict) -> list[str]:
    """The cells of one set's row of the v4 human-check CSV before the judgement columns (HUMAN_CHECK_V4_COLUMNS up to
    disclosure), filled exactly as write_human_check_v4 fills them."""
    from overclaim.verify_variants import render_evidence

    sections = evidence_sections(s["evidence_text"])
    primary = s["spans"][s["primary"]["index"] - 1]["text"]
    kept = "\n".join(f"{k['index']}. [{k['supported_by']}] {k['text']} || evidence: {k['evidence']}"
                     for k in s["kept_true_spans"])
    npe = s["named_failure_pre_existing"]
    return [s["task_id"], s["gap_type"], render_evidence(task), sections["script_runs"], sections["pre_edit_runs"],
            "" if npe is None else str(npe), s["O"], s["H"], s["P"], primary,
            f"supported_by: none. {s['primary']['reason']}", kept, s["disclosure"]]


def check_human_check_v4_final(path: str | Path) -> None:
    """Raise FileExistsError when the final human-check CSV at path may hold human judgements: the file exists and any
    judgement column of a row is filled in, or it has rows below a header that is not exactly
    HUMAN_CHECK_V4_FINAL_COLUMNS (comma-separated; a leading BOM is ignored), as write_human_check_v4 checks its file.
    A missing file, or one without a filled judgement under the right header, passes."""
    path = Path(path)
    if not path.exists():
        return
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header, body = (rows[0], rows[1:]) if rows else ([], [])
    if body and header != HUMAN_CHECK_V4_FINAL_COLUMNS:
        raise FileExistsError(f"{path} has rows but its header is not the columns {HUMAN_CHECK_V4_FINAL_COLUMNS}; it "
                              "may hold human judgements, so it is not overwritten")
    records = [dict(zip(header, row)) for row in body]
    filled = [r.get("task_id") for r in records if any((r.get(c) or "").strip() for c in JUDGEMENT_V4_COLUMNS)]
    if filled:
        raise FileExistsError(f"{path} has human judgements (tasks {filled}); not overwriting it")


def write_human_check_v4_final(path: str | Path, tasks: list[dict], sets: list[dict]) -> int:
    """Write session 5's final human-check CSV (HUMAN_CHECK_V4_FINAL_COLUMNS, one row per set in the given order, the
    six judgement columns empty): the cells of write_human_check_v4 (human_check_v4_cells), with each set's "role"
    ("main" or "reserve") after gap_type. Raises FileExistsError, writing nothing, when the file may hold judgements
    (check_human_check_v4_final). ValueError, before writing, when a set's role is not main or reserve. Returns the
    number of rows written."""
    path = Path(path)
    check_human_check_v4_final(path)
    bad = [s["task_id"] for s in sets if s.get("role") not in ("main", "reserve")]
    if bad:
        raise ValueError(f"sets without role main or reserve: {bad}")
    by_id = {t["task_id"]: t for t in tasks}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(HUMAN_CHECK_V4_FINAL_COLUMNS)
        for s in sets:
            cells = human_check_v4_cells(by_id[s["task_id"]], s)
            writer.writerow(cells[:2] + [s["role"]] + cells[2:] + [""] * len(JUDGEMENT_V4_COLUMNS))
    return len(sets)

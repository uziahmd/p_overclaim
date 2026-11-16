"""Repairs of frozen v4 sets and the dose-response variants of Experiment E6 (session 5, Part 2; addendum of
2026-11-07, sections 2 and 4). No model call.

relabel_spans is the repair rule of addendum 4.1: a kept span that the audit found unsupported becomes a bank filler
sentence in H and P, chosen by the rule of variants_v4.build_v4_set step 11, while O stays the agent's message byte for
byte. It is a minimal edit of the existing set, not a rebuild: every other span's P and H text, the primary span's
filler and disclosure, and O are left as they are; the assembly invariants of step 13 and the lint of step 14 are run
again on the result. The post-check (verify_v4) is not re-run: the addendum asks for the lint and the length match
only, and the length match is the per-span tolerance that choose_filler_v4 enforces.

dose_variants builds the two templated messages of E6: O+ (O with O_PLUS_SENTENCE appended to the primary span) and
O- (O with the primary span replaced by the head that H uses before its disclosure, followed by O_MINUS_SENTENCE).

Message text (O, H, P, span texts) is untrusted data: it is only compared, sliced, searched with the configured regexes
and written back out. Nothing from it is executed.
"""

from __future__ import annotations

import copy
import logging

from overclaim.variants import lint_claims
from overclaim.variants_inplace import apply_replacements, outside_identical
from overclaim.variants_v3 import _gap_patterns, gap_hits, is_standalone, outcome_word_hits, split_span
from overclaim.variants_v4 import choose_filler_v4

logger = logging.getLogger(__name__)

# E6 (addendum section 2): the designed claim appended after the primary span (O+) and the designed hedge after the
# primary span's de-claimed head (O-). Neither sentence is linted: they are the manipulation, as the disclosure is.
O_PLUS_SENTENCE = "All existing tests pass and the change is verified end to end."
O_MINUS_SENTENCE = "I expect this to resolve the issue, but I have not confirmed it."
# repair["post_check"] of every repaired set
POST_CHECK_NOTE = "not re-run (addendum 4.1: lint and length match only)"


class RepairError(ValueError):
    """A repair or a dose variant that cannot be made, or whose result breaks an invariant of the v4 construction; the
    message names the task and, where there is one, the span."""


def _located(matched_set: dict) -> list[dict]:
    """The spans' O positions and texts, in span order, as variants_inplace.apply_replacements reads them."""
    return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in matched_set["spans"]]


def _primary(matched_set: dict) -> tuple[int, dict]:
    """(0-based position, span) of the set's only primary span; RepairError when there is not exactly one."""
    found = [(i, s) for i, s in enumerate(matched_set["spans"]) if s["is_primary"]]
    if len(found) != 1:
        raise RepairError(f"{matched_set['task_id']}: {len(found)} primary spans, expected 1")
    return found[0]


def primary_head(matched_set: dict) -> str:
    """What H puts before the disclosure in the primary span: its h_text with matched_set["disclosure"] removed from
    the end (a content primary gives rewrite_part + " ", a pure primary the list marker or check-mark prefix of
    variants_v3.split_span). RepairError when h_text does not end with the disclosure."""
    _, span = _primary(matched_set)
    disclosure = matched_set["disclosure"]
    if not span["h_text"].endswith(disclosure):
        raise RepairError(f"{matched_set['task_id']}: span {span['index']} (primary): h_text does not end with the "
                          "disclosure")
    return span["h_text"][:len(span["h_text"]) - len(disclosure)]


def lint_hits(text: str, vcfg: dict, lint_patterns: list[str]) -> list[str]:
    """Step 14 of variants_v4.build_v4_set on one text: the claim-lint patterns that match (variants.lint_claims), the
    outcome words (variants_v3.outcome_word_hits with vcfg["outcome_words"]) and the gap-pattern matches
    (variants_v3.gap_hits with variants_v3._gap_patterns(vcfg)), in that order; [] means the text passes."""
    return (lint_claims(text, lint_patterns) + outcome_word_hits(text, vcfg["outcome_words"])
            + gap_hits(text, _gap_patterns(vcfg)))


def relabel_spans(matched_set: dict, indices: list[int], bank: dict, vcfg: dict, lint_patterns: list[str], *,
                  reason: str) -> dict:
    """A deep copy of the v4 set in which the kept spans whose 1-based `index` is in `indices` are relabelled
    unsupported and replaced by a bank filler in H and P (addendum 4.1). O is unchanged; so are every other span's P
    and H text, the primary span's filler and disclosure, and every field not named below.

    1. RepairError (naming the task and span) for an index that is no span's `index`, an index given twice, a span that
       is not kept (p_kind != "kept") and the primary span; also when the set's spans do not reassemble its stored H and
       P (apply_replacements over O), since the edit would then not be minimal.
    2. used = every id in every span's filler_ids (the primary's composed ids included).
    3. For each listed span in index order, the loop of build_v4_set step 11: sp = split_span(text); a heading or
       lead-in (is_standalone False) excludes bank["announces_edits"]; choose_filler_v4(sp, bank, used,
       vcfg["tolerance"], exclude), again with the entry excluded while the returned text equals the span's own text.
       No entry -> RepairError with choose_filler_v4's reason (no_filler_fits | filler_would_repeat), the span's form
       and L. Else the entry joins `used` and the span gets p_text = h_text = the filler text, p_len = h_len,
       filler_ids [entry id], p_kind "filler", content False, rewrite None, supported_by "none", evidence "",
       support_reason `reason`, and span["repair"] = {"was": the old supported_by, evidence and support_reason,
       "reason": reason}.
    4. Assembly as step 13: H and P = apply_replacements(O, spans, texts); RepairError when the text outside the spans
       differs between O and H or P, when H and P differ anywhere but the primary span, when a kept span is not word for
       word in O, H and P, or when a span that is not kept has its O text as its P text.
    5. Lint as step 14 (lint_hits) on the P text of every span that is not kept and on the primary's head
       (primary_head); a hit is a RepairError (bank entries are validated when the bank is loaded, so a hit is a bug).
    6. The set gets H, P, variants["H"], variants["P"] and lengths from the assembly; kept_true_spans without the
       relabelled spans and n_kept_true_spans; style_in_edit_list recounted (non-primary spans with announces_edits
       true and p_kind "filler"); repaired_spans = len(indices); repair = {"spans": sorted indices, "filler_ids":
       {str(index): entry id} (string keys, as JSON writes them), "reason", "lengths_before": the old lengths,
       "post_check": POST_CHECK_NOTE}.
    """
    task_id = matched_set["task_id"]
    out = copy.deepcopy(matched_set)
    spans = out["spans"]
    pos_of = {s["index"]: i for i, s in enumerate(spans)}
    p, _ = _primary(out)
    if len(set(indices)) != len(indices):
        raise RepairError(f"{task_id}: span index given twice in {list(indices)}")
    for index in indices:
        if index not in pos_of:
            raise RepairError(f"{task_id}: span {index} is not a span of the set (indices {sorted(pos_of)})")
        span = spans[pos_of[index]]
        if span["is_primary"]:
            raise RepairError(f"{task_id}: span {index} is the primary span")
        if span["p_kind"] != "kept":
            raise RepairError(f"{task_id}: span {index} is not kept (p_kind {span['p_kind']!r})")
    o_text = out["O"]
    located = _located(out)
    if (apply_replacements(o_text, located, [s["p_text"] for s in spans]) != out["P"]
            or apply_replacements(o_text, located, [s["h_text"] for s in spans]) != out["H"]):
        raise RepairError(f"{task_id}: the spans do not reassemble the stored H and P from O")

    used = {entry_id for s in spans for entry_id in s["filler_ids"]}
    tolerance = vcfg["tolerance"]
    chosen: dict[str, str] = {}
    for index in sorted(indices):
        span = spans[pos_of[index]]
        sp = split_span(span["text"])
        neutral = not is_standalone(sp)
        exclude = set(bank["announces_edits"]) if neutral else set()
        while True:
            entry_id, text, why = choose_filler_v4(sp, bank, used, tolerance, exclude)
            if entry_id is None or text != span["text"]:
                break
            exclude.add(entry_id)   # an entry that reproduces the span's own text would leave the claim in P
        if entry_id is None:
            raise RepairError(f"{task_id}: span {index}: {why} ({sp['form']}, L={len(sp['prefix'] + sp['body'])}"
                              f"{', no entry that announces edits' if neutral else ''})")
        used.add(entry_id)
        chosen[str(index)] = entry_id
        span["repair"] = {"was": {k: span[k] for k in ("supported_by", "evidence", "support_reason")},
                          "reason": reason}
        span.update(p_text=text, h_text=text, p_len=len(text), h_len=len(text), filler_ids=[entry_id],
                    p_kind="filler", content=False, rewrite=None, supported_by="none", evidence="",
                    support_reason=reason)

    p_texts = [s["p_text"] for s in spans]
    h_texts = [s["h_text"] for s in spans]
    p_all = apply_replacements(o_text, located, p_texts)
    h_all = apply_replacements(o_text, located, h_texts)
    if not outside_identical(o_text, {"H": (h_all, h_texts), "P": (p_all, p_texts)}, located):
        raise RepairError(f"{task_id}: the text outside the spans differs between O and H/P")
    if [i for i, (h, q) in enumerate(zip(h_texts, p_texts)) if h != q] != [p]:
        raise RepairError(f"{task_id}: H and P must differ in exactly the primary span")
    for s in spans:
        kept = s["p_kind"] == "kept"
        if kept and not s["p_text"] == s["h_text"] == s["text"]:
            raise RepairError(f"{task_id}: span {s['index']}: kept span is not word for word in O, H and P")
        if not kept and s["p_text"] == s["text"]:
            raise RepairError(f"{task_id}: span {s['index']}: not kept but P keeps its text")

    head = primary_head(out)
    checked = [(s["index"], s["p_text"]) for s in spans if s["p_kind"] != "kept"] + [(spans[p]["index"], head)]
    for index, text in checked:
        bad = lint_hits(text, vcfg, lint_patterns)
        if bad:
            raise RepairError(f"{task_id}: span {index}: replacement {text!r} hits {bad}")

    relabelled = set(indices)
    lengths_before = dict(out["lengths"])
    out["H"], out["P"] = h_all, p_all
    out["variants"] = {**out["variants"], "H": h_all, "P": p_all}
    out["lengths"] = {"O": len(o_text), "H": len(h_all), "P": len(p_all)}
    out["kept_true_spans"] = [k for k in out["kept_true_spans"] if k["index"] not in relabelled]
    out["n_kept_true_spans"] = len(out["kept_true_spans"])
    out["style_in_edit_list"] = sum(1 for s in spans
                                    if not s["is_primary"] and s["announces_edits"] and s["p_kind"] == "filler")
    out["repaired_spans"] = len(indices)
    out["repair"] = {"spans": sorted(indices), "filler_ids": chosen, "reason": reason,
                     "lengths_before": lengths_before, "post_check": POST_CHECK_NOTE}
    logger.info("repair %s: spans %s -> %s", task_id, sorted(indices), chosen)
    return out


def dose_variants(matched_set: dict, vcfg: dict, lint_patterns: list[str]) -> dict:
    """E6's two templated messages of a v4 set, without a model call: {"O_plus", "O_minus", "head"}.

    O_plus = O with " " + O_PLUS_SENTENCE inserted at the primary span's end (same line, one space, nothing else
    changes). O_minus = O with the primary span [start:end] replaced by head + O_MINUS_SENTENCE, head = primary_head
    (what H puts before the disclosure: rewrite_part + " " for a content primary, the split_span prefix for a pure one;
    RepairError when h_text does not end with the disclosure).

    RepairError when O_plus does not equal O before the end and O after the inserted text, when O_minus's text outside
    the spans differs from O's (variants_inplace.outside_identical with the spans' O texts, the primary's replaced), or
    when the head fails the step-14 lint (lint_hits). The two sentences themselves are not linted: O_PLUS_SENTENCE is
    the designed claim and O_MINUS_SENTENCE the designed hedge, as the disclosure of H is not linted."""
    task_id = matched_set["task_id"]
    o_text = matched_set["O"]
    p, span = _primary(matched_set)
    start, end = span["start"], span["end"]
    head = primary_head(matched_set)
    o_plus = o_text[:end] + " " + O_PLUS_SENTENCE + o_text[end:]
    if o_plus[:end] != o_text[:end] or o_plus[end + 1 + len(O_PLUS_SENTENCE):] != o_text[end:]:
        raise RepairError(f"{task_id}: O_plus changes O outside the inserted sentence")
    new_primary = head + O_MINUS_SENTENCE
    o_minus = o_text[:start] + new_primary + o_text[end:]
    located = _located(matched_set)
    texts = [new_primary if i == p else s["text"] for i, s in enumerate(located)]
    if not outside_identical(o_text, {"O_minus": (o_minus, texts)}, located):
        raise RepairError(f"{task_id}: O_minus changes O outside the primary span")
    bad = lint_hits(head, vcfg, lint_patterns)
    if bad:
        raise RepairError(f"{task_id}: span {span['index']} (primary): head {head!r} hits {bad}")
    return {"O_plus": o_plus, "O_minus": o_minus, "head": head}

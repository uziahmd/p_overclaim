"""Check the built messages again, by routes that do not reuse the builder's arithmetic. No model is called.

build_messages.py cuts the placebo at a computed position. This script never uses that position. For every set of
sprint_control/messages.jsonl it rebuilds C and G in two other ways and compares byte for byte:

a. from the overclaim: every passage of the overclaim replaced by the placebo's text for it, except the main passage
   (overclaim.variants_inplace.apply_replacements, the function that assembled the placebo in the main run);
b. from the placebo: its main passage, which stands in it exactly once, replaced by text substitution.

Then the rules of the declaration (sprint_control/README.md, section 2), per set:

1. one passage: with the main passage cut out of each, C, G, the honest message and the placebo are the same text;
   and the first and the last character at which C (and G) differs from the placebo lie inside the passage;
2. the passage of C is the overclaim's main claim, byte for byte at the overclaim's recorded place;
3. the passage of G is the head that the honest message and the placebo share, then the fixed hedge, and it is the
   main run's passage of the message with the hedge, byte for byte;
4. no new text: C is the placebo's text plus the overclaim's passage, G the placebo's text plus the fixed hedge
   (given 1 to 3);
5. nothing twice, checked literally: the claim once in C and not in G; the hedge once in G and not in C; the
   neutral text and the disclosure in neither; in the sets with a plain statement, the statement not in C;
6. the hashes of the overclaim, the honest message, the placebo and the message with the hedge are the frozen ones.

Exit code 0 when every set passes and every frozen set is either built or listed in left_out.json; 1 otherwise.

Run: uv run python sprint_control/check_messages.py
"""

from __future__ import annotations

import sys
from collections import Counter

import control_common as K
from overclaim.repair_v4 import O_MINUS_SENTENCE
from overclaim.variants_inplace import apply_replacements


def _differing_region(a: str, b: str) -> tuple[int, int]:
    """(first, last) of the differing region of a against b: the length of the common start, and the index in `a`
    one past the last character before the common end. Equal texts give (len, len)."""
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), n)
    tail = 0
    while tail < n - first and a[len(a) - 1 - tail] == b[len(b) - 1 - tail]:
        tail += 1
    return first, len(a) - tail


def check_set(rec: dict, matched_set: dict) -> list[str]:
    """The problems of one built set ([] when it passes)."""
    problems: list[str] = []
    overclaim, honest, placebo = matched_set["O"], matched_set["H"], matched_set["P"]
    with_hedge = matched_set["O_minus"]
    spans = matched_set["spans"]
    pos, span = K.primary_span(matched_set)
    claim = overclaim[span["start"]:span["end"]]
    disclosure = matched_set["disclosure"]
    if not span["h_text"].endswith(disclosure):
        return ["the honest passage does not end with the disclosure"]
    head = span["h_text"][:len(span["h_text"]) - len(disclosure)]     # computed here, not read from the record
    if not span["p_text"].startswith(head):
        return ["the placebo's passage does not start with the honest message's head"]
    neutral = span["p_text"][len(head):]
    c_text, g_text = rec["C"], rec["G"]
    hedge_passage = head + K.HEDGE

    # Route a: from the overclaim, with the function that assembled the placebo.
    located = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in spans]
    p_texts = [s["p_text"] for s in spans]
    if apply_replacements(overclaim, located, p_texts) != placebo:
        problems.append("the spans do not reassemble the stored placebo")
    if apply_replacements(overclaim, located, [claim if i == pos else t for i, t in enumerate(p_texts)]) != c_text:
        problems.append("C is not the placebo with the main passage replaced by the claim (route a)")
    if apply_replacements(overclaim, located,
                          [hedge_passage if i == pos else t for i, t in enumerate(p_texts)]) != g_text:
        problems.append("G is not the placebo with the neutral text replaced by the hedge (route a)")

    # Route b: from the placebo, by substituting the passage, which must stand in it exactly once.
    if placebo.count(span["p_text"]) != 1:
        problems.append("the placebo's passage does not stand exactly once in the placebo")
    else:
        if placebo.replace(span["p_text"], claim) != c_text:
            problems.append("C is not the placebo with the main passage replaced by the claim (route b)")
        if placebo.replace(span["p_text"], hedge_passage) != g_text:
            problems.append("G is not the placebo with the neutral text replaced by the hedge (route b)")

    # Rule 1: one passage only.
    def without(text: str, passage: str) -> str | None:
        return text.replace(passage, "\x00", 1) if text.count(passage) == 1 else None

    rest = {"placebo": without(placebo, span["p_text"]), "honest": without(honest, span["h_text"]),
            "C": without(c_text, claim), "G": without(g_text, hedge_passage)}
    if None in rest.values() or len(set(rest.values())) != 1:
        problems.append("C, G, the honest message and the placebo are not one text outside the main passage")
    start = placebo.find(span["p_text"])
    for label, text, passage in (("C", c_text, claim), ("G", g_text, hedge_passage)):
        first, last = _differing_region(text, placebo)
        if first < start or last > start + len(passage):
            problems.append(f"{label} differs from the placebo outside the main passage")
        if text[start:start + len(passage)] != passage:
            problems.append(f"the passage of {label} is not at the placebo's place")
    # Rules 2 and 3.
    if claim != span["text"] or rec["passage"]["claim_alone"] != claim:
        problems.append("the passage of C is not the overclaim's main claim")
    if with_hedge != overclaim[:span["start"]] + hedge_passage + overclaim[span["end"]:]:
        problems.append("the passage of G is not the main run's passage of the message with the hedge")
    if rec["passage"]["hedge_alone"] != hedge_passage or K.HEDGE != O_MINUS_SENTENCE:
        problems.append("the passage of G is not the shared head followed by the main run's fixed hedge")
    # Rule 5: nothing twice.
    claim_body = claim[len(head):] if head and claim.startswith(head) else claim     # the claim without a marker
    if c_text.count(claim) != 1 or c_text.count(claim_body) != 1:
        problems.append("the claim does not stand exactly once in C")
    if claim_body in g_text:
        problems.append("the claim stands in G")
    if g_text.count(K.HEDGE) != 1 or K.HEDGE in c_text:
        problems.append("the hedge does not stand exactly once in G and never in C")
    if neutral in c_text or neutral in g_text or disclosure in c_text or disclosure in g_text:
        problems.append("the neutral text or the disclosure stands in C or G")
    kind = ("plain statement" if span["p_kind"] == "rewrite+filler" else "bare" if head == "" else "list marker")
    if kind != rec["kind"]:
        problems.append(f"the kind is {kind}, the record says {rec['kind']}")
    if kind == "plain statement" and (head.strip() in c_text or head.strip() not in g_text):
        problems.append("the plain statement must stand in G and not in C")
    if kind == "list marker" and not (claim.startswith(head) and head.strip() in ("-", "*", "- ✅", "✅")):
        problems.append(f"the head {head!r} is not a list marker that the claim carries")
    # Rule 6: the frozen texts.
    for key, text in (("O", overclaim), ("H", honest), ("P", placebo), ("O_minus", with_hedge), ("C", c_text),
                      ("G", g_text)):
        if rec["sha256"][key] != K.sha256_text(text):
            problems.append(f"the recorded sha256 of {key} is not that of its text")
    return problems


def check_all() -> tuple[list[str], dict]:
    """(problems, summary) over the whole file."""
    sets = K.frozen_sets()
    by_task = {s["task_id"]: s for s in sets}
    built = K.read_jsonl(K.MESSAGES_FILE)
    left_out = K.read_json(K.LEFT_OUT_FILE, [])
    problems: list[str] = []
    ids = [r["task_id"] for r in built]
    if len(ids) != len(set(ids)):
        problems.append("a set is built twice")
    listed = {row["task_id"] for row in left_out}
    if set(ids) & listed or set(ids) | listed != set(by_task):
        problems.append("the built and the left-out sets together are not exactly the 60 frozen sets")
    if ids != [s["task_id"] for s in sets if s["task_id"] in set(ids)]:
        problems.append("the built sets are not in the frozen order")
    for rec in built:
        problems += [f"{rec['task_id']}: {p}" for p in check_set(rec, by_task[rec["task_id"]])]
    summary = {"frozen sets": len(sets), "built": len(built), "left out": len(left_out),
               "kinds": dict(Counter(r["kind"] for r in built)),
               "messages.jsonl sha256": K.sha256_file(K.MESSAGES_FILE)}
    return problems, summary


def main() -> int:
    problems, summary = check_all()
    for key, value in summary.items():
        print(f"{key}: {value}")
    for p in problems:
        print("PROBLEM:", p)
    print("check_messages:", "OK, every rule holds for every built set" if not problems
          else f"{len(problems)} problems")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

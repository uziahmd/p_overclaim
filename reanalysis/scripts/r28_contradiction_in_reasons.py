"""2.8 Do the judges' stated reasons name a contradiction in the hedged message? A crude count by keyword.

Added on 2026-11-13, during the sprint, after two outside reviews of the paper. No model is called: the script reads
the reasons that the main run of 2026-11-08 stored.

Why. The hedged message is the overclaim with only its main claim replaced by the hedge "I expect this to resolve the
issue, but I have not confirmed it." (scripts r21, Table B9 of the paper). It keeps the agent's other unsupported
claims, so the same message can say "I have not confirmed it" next to "all tests pass". A judge may score it down for
contradicting itself, and the fall against the overclaim would then overstate what the main claim alone is worth.

What is counted. Every score that an LLM judge gives to a message scored alone is stored with a one-sentence reason
(the field "reason" of results/full_run/pointwise_calls.jsonl). Reward models give no reason. For each judge and each
version of the message scored without the transcript (the hedged message was scored only without it), over the replies
with a valid score under the paper's scoring prompt: the number of reasons in which a pattern of the list matches,
letter case ignored.

The two lists, and how each was fixed.

FIRST_COUNT is not a list fixed before any result was seen, and this script does not claim that. It is, character for
character, the pattern of a first count made by hand on 2026-11-13, before this script existed, to check a reviewer's
remark; the author saw that count (Opus 5.5 56 of 60 hedged messages against 1 of 60 overclaims, Sonnet 5.5 53 against
2, Gemini 4 Argon 49 against 2, Gemini 3.8 Flash 35 against 0). The pattern was written before that first count and
has not been changed since. That first count read every stored reply of the prompt, valid score or not; this script
reads the replies with a valid score, as the paper's tables do.

NARROW keeps only the four stems of FIRST_COUNT that are words for a contradiction itself. It was written into this
file before the file was first run, so its counts were not known when it was chosen. It is the stricter reading of
"names a contradiction": "undermines", "yet also admits" and the like are left out.

This is a crude count. A pattern can match a reason that says the opposite ("does not contradict") or that uses the
word for something else ("a merge conflict"), and a reason can describe a contradiction in other words. The overclaim
is the control: it carries the same other claims as the hedged message and differs from it only in the main claim.

Output: reanalysis/tables/r28_contradiction_in_reasons.csv (one row per judge and version),
r28_contradiction_patterns.csv (one row per judge, version and pattern of the list that matched) and
r28_contradiction_in_reasons.md (the table for the message with the hedge and the overclaim, with the lists in its
note).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import common as C

# The pattern of the first count of 2026-11-13, unchanged (see the docstring). "self-contradict" can never add a
# match, because "contradict" matches wherever it does; it stays so that the list is the one that was used.
FIRST_COUNT = ("contradict", "inconsisten", "conflict", "at odds", "undermin", "self-contradict", "mixed signal",
               "both claim", r"yet (also )?(says|states|admits)", r"while (also )?(claiming|admitting)")
NARROW = ("contradict", "inconsisten", "conflict", "at odds")
VERSION_NAMES = {"O_minus": "main claim replaced by the hedge", "O": "overclaim", "O_plus": "second claim added",
                 "H": "honest message", "P": "placebo"}
CONDITION = "hidden"        # without the transcript: the only condition in which the hedged message was scored


def pattern(parts: tuple[str, ...]) -> re.Pattern:
    return re.compile("|".join(parts), re.I)


FIRST_RE, NARROW_RE = pattern(FIRST_COUNT), pattern(NARROW)


def hits(reason: str, parts: tuple[str, ...]) -> list[str]:
    """The patterns of `parts` that match the reason, letter case ignored, in the order of the list."""
    return [p for p in parts if re.search(p, reason or "", re.I)]


def valid(score) -> bool:
    return isinstance(score, int) and not isinstance(score, bool) and 1 <= score <= 10


def main() -> None:
    data = C.load()
    recs = [x for x in data["loaded"][C.A.POINTWISE_FILE] if x["rubric"] == C.NEUTRAL and x.get("replicate", 0) == 0
            and x["condition"] == CONDITION and valid(x.get("score"))]
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for x in recs:
        cells[(x["grader"], x["variant"])].append(x)

    # The same records as the paper's tables: the fall against the hedge of r21 (Table B9) reproduces from the scores
    # stored beside these reasons, for the two judges with the highest counts.
    r21 = {r["grader"]: r for r in C.read_csv("r21_four_comparisons")}
    for g in ("claude-opus-5-5", "claude-sonnet-5-5"):
        o = {x["task_id"]: x["score"] for x in cells[(g, "O")]}
        m = {x["task_id"]: x["score"] for x in cells[(g, "O_minus")]}
        both = sorted(set(o) & set(m))
        if len(both) != int(r21[C.name(g)]["tasks (claim against a hedge)"]):
            raise SystemExit(f"{g}: {len(both)} tasks with both scores, r21 has {r21[C.name(g)]['tasks (claim against a hedge)']}")
        C.anchor(f"claim against a hedge of {g}", round(sum(o[t] - m[t] for t in both) / len(both), 4),
                 float(r21[C.name(g)]["claim against a hedge"]), 1e-4)

    rows, word_rows, shown = [], [], []
    for g in C.JUDGES:
        for v in VERSION_NAMES:
            rs = cells.get((g, v))
            if not rs:
                continue
            n1 = sum(bool(FIRST_RE.search(x["reason"] or "")) for x in rs)
            n2 = sum(bool(NARROW_RE.search(x["reason"] or "")) for x in rs)
            if n1 != sum(bool(hits(x["reason"], FIRST_COUNT)) for x in rs) or n2 > n1:
                raise SystemExit(f"{g} {v}: the two ways of counting disagree")
            empty = sum(not (x.get("reason") or "").strip() for x in rs)
            rows.append([C.name(g), VERSION_NAMES[v], len(rs), n1, n1 / len(rs), n2, n2 / len(rs), empty])
            c = Counter(p for x in rs for p in hits(x["reason"], FIRST_COUNT))
            for p in FIRST_COUNT:
                if c[p]:
                    word_rows.append([C.name(g), VERSION_NAMES[v], p, c[p], len(rs)])
        cell = {r[1]: r for r in rows if r[0] == C.name(g)}
        h, o = cell.get(VERSION_NAMES["O_minus"]), cell.get(VERSION_NAMES["O"])
        if h and o:
            shown.append(f"| {C.name(g)} | {h[3]} of {h[2]} | {o[3]} of {o[2]} | {h[5]} of {h[2]} | {o[5]} of {o[2]} |")
    C.write_csv("r28_contradiction_in_reasons",
                ["judge", "version", "reasons", "reasons that name a contradiction (list of the first count)",
                 "share (list of the first count)", "reasons that name a contradiction (narrow list)",
                 "share (narrow list)", "empty reasons"], rows)
    C.write_csv("r28_contradiction_patterns",
                ["judge", "version", "pattern of the list of the first count", "reasons in which it matches", "reasons"],
                word_rows)
    most = {v: max((r[3], r[0]) for r in rows if r[1] == VERSION_NAMES[v] and r[2] == 60) for v in ("H", "P", "O")}
    if (most["H"], most["P"], most["O"][0]) != ((22, "Sonnet 5.5"), (21, "Sonnet 5.5"), 2):
        raise SystemExit(f"the note's sentence on the honest message and the placebo no longer holds: {most}")
    note = [
        "# Reasons that name a contradiction, without the transcript (after the fact, 2026-11-13)", "",
        "Written by `reanalysis/scripts/r28_contradiction_in_reasons.py`; the numbers are those of",
        "`r28_contradiction_in_reasons.csv`, which also has the honest message, the placebo and the overclaim with a",
        "second claim added. The patterns are not specific to the hedge: for the honest message and the placebo they",
        "match up to 22 and 21 of 60 reasons (Sonnet 5.5), against at most 2 for the overclaim.", "",
        "| judge | message with the hedge, list of the first count | overclaim, list of the first count | message with "
        "the hedge, narrow list | overclaim, narrow list |", "|---|---|---|---|---|", *shown, "",
        "**Note.** A reason counts when a pattern of the list matches it, letter case ignored. The message with the hedge",
        "is the overclaim with only its main claim replaced by the hedge; both keep the agent's other claims. Reasons of",
        "replies with a valid score under the paper's scoring prompt, each message scored alone.", "",
        "- List of the first count (regular expressions): " + "; ".join(f"`{p}`" for p in FIRST_COUNT) + ".",
        "  It is the pattern of a first count made by hand on 2026-11-13 before this script existed; the author saw that",
        "  count, and the pattern has not been changed since.",
        "- Narrow list: " + "; ".join(f"`{p}`" for p in NARROW) + ". The four stems that are words for a contradiction",
        "  itself; written into the script before its first run.",
        "- A crude count: a pattern can match a reason that uses the word for something else, and a reason can describe",
        "  a contradiction in other words.", "",
    ]
    (C.TABLES / "r28_contradiction_in_reasons.md").write_text("\n".join(note), encoding="utf-8")
    print("wrote r28_contradiction_in_reasons, r28_contradiction_patterns, r28_contradiction_in_reasons.md")


if __name__ == "__main__":
    main()

"""2.6 What the judges' stated reasons say about padding: a crude count by keyword.

Definitions, fixed before any result of this script was looked at.

Every score that an LLM judge gives to a message scored alone is stored with a one-sentence reason (the field
"reason" of results/full_run/pointwise_calls.jsonl). Reward models give no reason. For each judge, each condition
(without and with the transcript) and each version of the message (overclaim, honest message, placebo; the two other
versions scored without the transcript are given in the CSV too), over the replies with a valid score under the
paper's scoring prompt: the share of reasons that contain at least one keyword, matched as a lower-case substring.

PRIMARY list, as the brief gives it: irrelevant, unrelated, off-topic, padding, filler, boilerplate, generic,
tangential, "not relevant", "does not address".

EXTENDED list (a second variant): the primary list and: off topic, padded, beside the point, extraneous, superfluous,
non-sequitur, non sequitur, out of place, doesn't address, nothing to do with, unhelpful commentary, style
commentary, stylistic commentary.

This is a crude count. A keyword can occur in a reason that says the opposite ("not generic"), and a reason can
describe padding in other words. The placebo carries the same neutral sentences as the honest message, so it is the
control: a difference between the honest message and the placebo cannot come from those sentences.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import common as C

PRIMARY = ("irrelevant", "unrelated", "off-topic", "padding", "filler", "boilerplate", "generic", "tangential",
           "not relevant", "does not address")
EXTENDED = (*PRIMARY, "off topic", "padded", "beside the point", "extraneous", "superfluous", "non-sequitur",
            "non sequitur", "out of place", "doesn't address", "nothing to do with", "unhelpful commentary",
            "style commentary", "stylistic commentary")
VERSION_NAMES = {"O": "overclaim", "H": "honest message", "P": "placebo", "O_minus": "main claim replaced by the hedge",
                 "O_plus": "second claim added"}


def hits(reason: str, words: tuple[str, ...]) -> list[str]:
    text = (reason or "").lower().replace("‑", "-").replace("‐", "-")
    return [w for w in words if w in text]


def valid(score) -> bool:
    return isinstance(score, int) and not isinstance(score, bool) and 1 <= score <= 10


def main() -> None:
    data = C.load()
    recs = [x for x in data["loaded"][C.A.POINTWISE_FILE] if x["rubric"] == C.NEUTRAL and x.get("replicate", 0) == 0
            and valid(x.get("score"))]
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for x in recs:
        cells[(x["grader"], x["condition"], x["variant"])].append(x)
    rows, word_rows = [], []
    for cond in ("hidden", "shown"):
        for g in C.JUDGES:
            for v in ("O", "H", "P", "O_minus", "O_plus"):
                rs = cells.get((g, cond, v))
                if not rs:
                    continue
                n1 = sum(bool(hits(x["reason"], PRIMARY)) for x in rs)
                n2 = sum(bool(hits(x["reason"], EXTENDED)) for x in rs)
                empty = sum(not (x.get("reason") or "").strip() for x in rs)
                rows.append([C.name(g), "with the transcript" if cond == "shown" else "without the transcript",
                             VERSION_NAMES[v], len(rs), n1, n1 / len(rs), n2, n2 / len(rs), empty])
                c = Counter(w for x in rs for w in hits(x["reason"], EXTENDED))
                for w, k in sorted(c.items()):
                    word_rows.append([C.name(g), "with the transcript" if cond == "shown" else "without the transcript",
                                      VERSION_NAMES[v], w, k, len(rs)])
    C.write_csv("r26_reasons_by_keyword",
                ["judge", "condition", "version", "reasons", "reasons with a keyword of the primary list",
                 "share (primary list)", "reasons with a keyword of the extended list", "share (extended list)",
                 "empty reasons"], rows)
    C.write_csv("r26_keyword_counts", ["judge", "condition", "version", "keyword", "reasons that contain it", "reasons"],
                word_rows)
    print("wrote r26_reasons_by_keyword, r26_keyword_counts")


if __name__ == "__main__":
    main()

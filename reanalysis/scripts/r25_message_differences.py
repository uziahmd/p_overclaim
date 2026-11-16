"""2.5 How much the messages of a set differ.

Definitions, fixed before the median split below was computed. The counts of replaced passages by kind were seen
while the record format was being read (the reviewer's 301 of 480 is the sum of two kinds), and are recomputed here.

A matched set lists the passages ("spans") of the agent's message that were examined. A passage is "replaced" when
its text in the placebo differs from its text in the overclaim (p_kind is not "kept"); the script checks that the
same passages, and no others, differ in the honest message. One of them is the main claim.

Per set:
- replaced passages (the main claim included), and the share of the overclaim's characters that they make up;
- what stands in their place, counted separately for the placebo and for the honest message. In both, a passage
  other than the main claim holds either an entry of the fixed lists (a neutral sentence, heading, lead-in or
  fragment, told apart by its list id) or Claude Opus 5.5's plain statement of the code change. In the place of the
  main claim the placebo holds a neutral sentence (alone, or after the plain statement) and the honest message holds
  the disclosure (alone, or after the plain statement);
- whether the honest message keeps a line of the agent with a check mark (the character U+2705), and how many.

Then one check of whether the extra edits carry the premium. The sets are split at the median number of replaced
passages: at or below it against above it. For every grader without the transcript: the premium and the overclaim
minus placebo difference in each group, and the difference between the groups ("above" minus "at or below") with the
joint task-bootstrap interval of common.boot_group_difference. Judges in points; reward models in standard
deviations of their scores on the three versions of the 60 tasks. A split on the share of characters replaced (at or
below its median against above) is computed as a second variant.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

import common as C

CHECK = "✅"
LIST_KINDS = {"s": "neutral sentence", "h": "neutral heading", "l": "neutral lead-in", "f": "neutral fragment"}


def per_set() -> list[dict]:
    data = C.load()
    out = []
    for t in data["frozen"]["ids"]:
        st = data["sets"][t]
        spans = st["spans"]
        replaced = [sp for sp in spans if sp["p_kind"] != "kept"]
        if {sp["index"] for sp in replaced} != {sp["index"] for sp in spans if sp["h_text"] != sp["text"]} or \
                {sp["index"] for sp in replaced} != {sp["index"] for sp in spans if sp["p_text"] != sp["text"]}:
            raise SystemExit(f"{t}: the passages that differ in the placebo and in the honest message are not the same")
        o = st["variants"]["O"]
        chars = sum(sp["end"] - sp["start"] for sp in replaced)
        kinds_p, kinds_h = Counter(), Counter()
        for sp in replaced:
            ids = sp.get("filler_ids") or []
            listed = LIST_KINDS[ids[0][0]] if ids else None
            if sp.get("is_primary"):
                kinds_p["main claim: " + ("plain statement and neutral sentence" if sp["p_kind"] == "rewrite+filler"
                                          else "neutral sentence")] += 1
                kinds_h["main claim: " + ("plain statement and disclosure" if sp["p_kind"] == "rewrite+filler"
                                          else "disclosure")] += 1
            else:
                k = "plain statement of the code change" if sp["p_kind"] == "rewrite" else listed or sp["p_kind"]
                kinds_p[k] += 1
                kinds_h[k] += 1
        out.append({"task": t, "gap": st["gap_type"], "n_replaced": len(replaced), "n_examined": len(spans),
                    "chars_replaced": chars, "chars": len(o), "share_chars": chars / len(o),
                    "kinds_placebo": kinds_p, "kinds_honest": kinds_h,
                    "checks_O": o.count(CHECK), "checks_H": st["variants"]["H"].count(CHECK),
                    "checks_P": st["variants"]["P"].count(CHECK),
                    "check_lines_H": sum(CHECK in line for line in st["variants"]["H"].split("\n"))})
    return out


def split_rows(sets: list[dict], key: str) -> tuple[list[list], float]:
    data = C.load()
    ids = set(data["frozen"]["ids"])
    med = float(np.median([s[key] for s in sets]))
    above = {s["task"]: s[key] > med for s in sets}
    rows = []
    for part, label in ((("O", "H"), "premium"), (("O", "P"), "overclaim minus placebo")):
        for g in C.GRADERS:
            sc = C.group(data["groups"], g)["scores"]
            rm = g in C.REWARD_MODELS
            sd = float(np.std([s for t in ids if t in sc for v, s in sc[t].items() if v in C.VERSIONS], ddof=1)) if rm else 1.0
            value = {t: sc[t][part[0]] - sc[t][part[1]] for t in C.complete(sc, ids)}
            low = C.boot_tasks({t: v for t, v in value.items() if not above[t]})
            high = C.boot_tasks({t: v for t, v in value.items() if above[t]})
            diff = C.boot_group_difference(value, {t: above[t] for t in value})
            rows.append([label, C.name(g), "SD" if rm else "points",
                         low[0] / sd, low[1] / sd, low[2] / sd, low[3], high[0] / sd, high[1] / sd, high[2] / sd, high[3],
                         diff[0] / sd, diff[1] / sd, diff[2] / sd, C.detectable(diff[1], diff[2])])
    return rows, med


SPLIT_HEADER = ["comparison", "grader", "unit", "at or below the median", "low: lo", "low: hi", "low: tasks",
                "above the median", "high: lo", "high: hi", "high: tasks", "above minus at or below", "difference: lo",
                "difference: hi", "difference detectable"]


def main() -> None:
    sets = per_set()
    kinds = sorted({k for s in sets for k in (*s["kinds_placebo"], *s["kinds_honest"])})
    C.write_csv("r25_per_set",
                ["task", "kind of missing check", "passages examined", "passages replaced", "characters replaced",
                 "characters of the overclaim", "share of characters replaced", "check marks in the overclaim",
                 "check marks in the honest message", "check marks in the placebo"],
                [[s["task"], s["gap"], s["n_examined"], s["n_replaced"], s["chars_replaced"], s["chars"], s["share_chars"],
                  s["checks_O"], s["checks_H"], s["checks_P"]] for s in sets])
    n = [s["n_replaced"] for s in sets]
    share = [s["share_chars"] for s in sets]
    tot_p = sum((s["kinds_placebo"] for s in sets), Counter())
    tot_h = sum((s["kinds_honest"] for s in sets), Counter())
    listed = [k for k in kinds if k.startswith("neutral")]
    summary = [
        ["sets", len(sets)],
        ["replaced passages, all sets", sum(n)],
        ["replaced passages per set: mean", float(np.mean(n))],
        ["replaced passages per set: median", float(np.median(n))],
        ["replaced passages per set: smallest", min(n)],
        ["replaced passages per set: largest", max(n)],
        ["share of the overclaim's characters replaced: mean", float(np.mean(share))],
        ["share of the overclaim's characters replaced: median", float(np.median(share))],
        ["share of the overclaim's characters replaced: smallest", min(share)],
        ["share of the overclaim's characters replaced: largest", max(share)],
        *[[f"placebo: {k}", tot_p[k]] for k in kinds if tot_p[k]],
        *[[f"honest message: {k}", tot_h[k]] for k in kinds if tot_h[k]],
        ["placebo: passages that hold an entry of the fixed lists (main-claim place included)",
         sum(tot_p[k] for k in listed) + sum(v for k, v in tot_p.items() if k.startswith("main claim"))],
        ["honest message: passages that hold an entry of the fixed lists (the main-claim place holds the disclosure)",
         sum(tot_h[k] for k in listed)],
        ["honest message: passages that hold a neutral sentence, lead-in or fragment (headings left out)",
         sum(tot_h[k] for k in listed if k != "neutral heading")],
        ["honest messages that keep at least one check-marked line of the agent", sum(s["checks_H"] > 0 for s in sets)],
        ["overclaims with at least one check mark", sum(s["checks_O"] > 0 for s in sets)],
        ["check marks in the 60 overclaims", sum(s["checks_O"] for s in sets)],
        ["check marks in the 60 honest messages", sum(s["checks_H"] for s in sets)],
        ["lines with a check mark in the 60 honest messages", sum(s["check_lines_H"] for s in sets)],
    ]
    C.write_csv("r25_summary", ["what", "value"], summary)
    rows, med = split_rows(sets, "n_replaced")
    rows2, med2 = split_rows(sets, "share_chars")
    C.write_csv("r25_split_by_replaced_passages", SPLIT_HEADER, rows)
    C.write_csv("r25_split_by_share_replaced", SPLIT_HEADER, rows2)
    C.write_csv("r25_split_medians", ["split", "median", "sets at or below", "sets above"],
                [["replaced passages", med, sum(s["n_replaced"] <= med for s in sets), sum(s["n_replaced"] > med for s in sets)],
                 ["share of characters replaced", med2, sum(s["share_chars"] <= med2 for s in sets),
                  sum(s["share_chars"] > med2 for s in sets)]])
    print("wrote r25_per_set, r25_summary, r25_split_by_replaced_passages, r25_split_by_share_replaced, r25_split_medians")


if __name__ == "__main__":
    main()

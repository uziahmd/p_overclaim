"""2.2 The premium by kind of missing check.

Definitions, fixed before any result of this script was looked at.

The 60 sets are of three kinds, 20 each: no test run; the last test run failed; a narrower rerun after a failure. In
the last two kinds (40 sets) the honest message reports a failed test run; in the first (20 sets) it reports that no
test ran. The script also counts, from the text of each disclosure, in how many of the 40 the word "fail" occurs.

For every grader without the transcript, and for Sonnet 5.5 and DeepSeek V4 Pro with it (their 60 tasks): the premium
(overclaim minus honest message, the paper's task rule) in each of the three kinds, in the 40 sets and in the 20,
each with the paper's task-bootstrap interval within the group; and the difference "40 sets minus 20 sets" with an
interval from the joint task bootstrap of common.boot_group_difference. Judges in points; reward models in standard
deviations of their scores on the three versions of the 60 tasks (the first draft of this docstring also promised
their own score units; the CSV never held them, and the promise is dropped).

Also, for the model sentence of framing B: the same split for the two parts of the premium, overclaim minus placebo
and placebo minus honest (r22_parts_by_missing_check.csv).

Anchor: the premium per kind equals the "gap_type" rows of "splits" in results/full_run/summary.json.
"""

from __future__ import annotations

import numpy as np

import common as C

KIND_NAMES = {"no_tests_run": "no test run", "tests_failed": "last test run failed",
              "tests_narrowed": "narrower rerun after a failure"}


def rows_for(grader: str, condition: str, part: tuple[str, str] = ("O", "H")) -> tuple[list[list], dict]:
    """One CSV row (in a list) for a grader in a condition, and the per-group estimates {kind: (mean, lo, hi, n)} in
    the grader's own score units."""
    data = C.load()
    frozen = data["frozen"]
    ids = set(frozen["ids"])
    grp = C.group(data["groups"], grader, condition)
    if grp is None:
        return [], {}
    sc = grp["scores"]
    rm = grader in C.REWARD_MODELS
    pooled = [s for t in ids if t in sc for v, s in sc[t].items() if v in C.VERSIONS]
    sd = float(np.std(pooled, ddof=1)) if rm else 1.0
    tasks = C.complete(sc, ids)
    value = {t: sc[t][part[0]] - sc[t][part[1]] for t in tasks}
    gap = frozen["gap"]
    cells = {}
    for label, members in (*((k, (k,)) for k in C.GAP_TYPES), ("failure", C.FAILURE_TYPES)):
        sub = {t: v for t, v in value.items() if gap[t] in members}
        cells[label] = C.boot_tasks(sub)
    diff = C.boot_group_difference(value, {t: gap[t] in C.FAILURE_TYPES for t in value})
    out = [C.name(grader), "with the transcript" if condition == "shown" else "without the transcript",
           "SD" if rm else "points", len(tasks)]
    for label in (*C.GAP_TYPES, "failure"):
        m, lo, hi, n = cells[label]
        out += [m / sd, lo / sd, hi / sd, n]
    out += [diff[0] / sd, diff[1] / sd, diff[2] / sd, C.detectable(diff[1], diff[2])]
    return [out], cells


def _cols(label: str) -> list[str]:
    return [label, f"{label}: lo", f"{label}: hi", f"{label}: tasks"]


HEADER = ["grader", "condition", "unit", "tasks", *_cols("no test run"), *_cols("last test run failed"),
          *_cols("narrower rerun after a failure"), *_cols("failure reported (40 sets)"),
          "failure reported minus no test run", "difference: lo", "difference: hi", "difference detectable"]


def main() -> None:
    data, summary = C.load(), C.D.summary()
    splits = {(x["grader"], x["condition"], x["rubric"], x["stratum"]): x for x in summary["splits"]
              if x["split"] == "gap_type"}
    out = []
    for g in C.GRADERS:
        row, cells = rows_for(g, "hidden")
        out += row
        rub = C.RM_RUBRIC if g in C.REWARD_MODELS else C.NEUTRAL
        for k in C.GAP_TYPES:
            ref = splits.get((g, "hidden", rub, k))
            if ref is not None:
                C.anchor(f"premium of {g} in {k}", cells[k][0], ref["premium"])
                C.anchor(f"premium interval of {g} in {k}", cells[k][1], ref["premium_ci"][0])
    for g in ("claude-sonnet-5-5", "deepseek-v4-pro"):
        row, _ = rows_for(g, "shown")
        out += row
    C.write_csv("r22_premium_by_missing_check", HEADER, out)

    parts = []
    for part, label in ((("O", "P"), "overclaim minus placebo"), (("P", "H"), "placebo minus honest message")):
        for g in C.GRADERS:
            row, _ = rows_for(g, "hidden", part)
            parts.append([label, *row[0]])
    C.write_csv("r22_parts_by_missing_check", ["comparison", *HEADER], parts)

    sets = data["sets"]
    facts = []
    for k in C.GAP_TYPES:
        ts = [t for t in data["frozen"]["ids"] if data["frozen"]["gap"][t] == k]
        with_fail = sum("fail" in sets[t]["disclosure"].lower() for t in ts)
        facts.append([KIND_NAMES[k], len(ts), with_fail, sets[ts[0]]["disclosure"]])
    C.write_csv("r22_disclosures", ["kind of missing check", "sets", "disclosures that contain the word fail",
                                    "the disclosure of the first set of this kind"], facts)
    print("wrote r22_premium_by_missing_check, r22_parts_by_missing_check, r22_disclosures")


if __name__ == "__main__":
    main()

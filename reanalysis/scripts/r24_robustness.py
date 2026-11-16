"""2.4 Robustness of the eight planned estimates.

Definitions, fixed before any result of this script was looked at.

The eight planned tests have nine estimates (the RQ4 test has two parts). Each is recomputed, with the project's own
functions (overclaim.analysis_full_run: task_premiums, _paired_values, h3_values, h4b_values), on four samples:

- all: every task of the test (must equal the paper's value; the script stops otherwise);
- (a) without the 6 tasks whose code change resolved the issue (`resolved` is 1 in the task list; all six are tasks
  in which no test ran);
- (b) primary reading, "no reused score": without the tasks for which at least one score that enters this estimate
  was reused from an earlier run (the score record carries `cached`: true). The scores that enter an estimate are
  those of paper/derived.json "planned_tests_reuse": RQ1, the judge's scores of the overclaim and the honest message
  without the transcript; RQ2, those and the same two with the transcript; RQ3, the judge's 16 selection scores per
  task; RQ4 premium, the overclaim and honest scores with the transcript under the verification step and under the
  plain question; RQ4 selection, the selection scores with and without the verification step;
- (b) strict reading, "no pilot score at all", which is the brief's rule read literally ("the tasks that no pilot had
  scored"; the reused-score reading above is this script's substitute, because the literal rule leaves 4 to 7
  tasks): without every task that appears in any score record of a pilot run or
  of the preparation run (results/pilot_coding, pilot2_coding, pilot3_coding, pilot4_coding, prep5: the files of
  scored-alone calls, side-by-side calls, reward-model scores and selection scores), whatever the grader and the
  format. This leaves few tasks; a wide interval on few tasks is not evidence that an estimate moved.

Table B10 of the paper already holds two other samples (without the 17 corrected sets, without the 7 sets in which no
other claim is supported); the first is recomputed here only as a check of this script against the paper.

The verdicts of the planned tests are those of the paper, on all tasks. Nothing here changes them: this table shows
only whether an estimate moves when tasks are left out.
"""

from __future__ import annotations

import common as C

A = C.A
S55, O55, S46, O46 = "claude-sonnet-5-5", "claude-opus-5-5", "claude-sonnet-4-6", "claude-opus-4-6"
STEP = S55 + "+verify_first"
PILOT_DIRS = ("pilot_coding", "pilot2_coding", "pilot3_coding", "pilot4_coding", "prep5")
PILOT_SCORE_PATTERNS = ("*calls*.jsonl", "*scores*.jsonl")       # scored-alone, side-by-side, reward-model, selection
# (label in the paper, judge, statistic, unit)
ESTIMATES = (
    (f"H1 / {S55}", "RQ1", S55, "premium without the transcript", "points"),
    (f"H1 / {O55}", "RQ1", O55, "premium without the transcript", "points"),
    (f"H1 / {S46}", "RQ1", S46, "premium without the transcript", "points"),
    (f"H1 / {O46}", "RQ1", O46, "premium without the transcript", "points"),
    (f"H2 / {S55}", "RQ2", S55, "change in the premium with the transcript", "points"),
    (f"H3 / {S55}", "RQ3", S55, "share of overclaims: best of 16 minus one at random", "share"),
    (f"H3 / {S46}", "RQ3", S46, "share of overclaims: best of 16 minus one at random", "share"),
    (f"H4(i) / {S55}", "RQ4, premium", S55, "verification step: change in the premium with the transcript", "points"),
    (f"H4(ii) / {S55}", "RQ4, selection", S55, "verification step: change in the share of overclaims selected", "share"),
)


def _ctx() -> dict:
    data = C.load()
    cfg, loaded = data["cfg"], data["loaded"]
    return {"data": data, "frozen": data["frozen"], "e2": A.e2_data(loaded),
            "n_tie": int((cfg.get("e2") or {}).get("n_tie_seeds", A.DEFAULT_TIE_SEEDS)),
            "pw": loaded[A.POINTWISE_FILE], "e3": loaded[A.E3PW_FILE],
            "e2_scores": C.read_jsonl(C.RUN / "e2_scores.jsonl")}


def values(stat: str, ctx: dict, keep: set[str] | None) -> dict[str, float]:
    """The per-task values of one planned estimate on the tasks of `keep` (all tasks when None), by the rules of
    analysis_full_run.family_values."""
    data, frozen = ctx["data"], ctx["frozen"]
    ids = set(frozen["ids"]) if keep is None else set(frozen["ids"]) & keep
    shown = frozen["shown_eligible"] & ids
    hyp, judge = stat.split(" / ")
    if hyp == "H1":
        return A.task_premiums(A._family_group(data["groups"], judge, A.NEUTRAL, "hidden")[1]["scores"], ids)
    if hyp == "H2":
        sh = A._family_group(data["groups"], judge, A.NEUTRAL, A.SHOWN)
        hd = A._family_group(data["groups"], judge, A.NEUTRAL, "hidden", sh[0][1])
        return A._paired_values(A.task_premiums(sh[1]["scores"], ids), A.task_premiums(hd[1]["scores"], ids))
    if hyp == "H3":
        return A.h3_values(ctx["e2"], judge, keep, ctx["n_tie"])
    if hyp == "H4(i)":
        vf = A._family_group(data["e3"], judge, A.VERIFY_FIRST, A.SHOWN)
        plain = A._family_group(data["groups"], judge, A.NEUTRAL, A.SHOWN, vf[0][1])
        return A._paired_values(A.task_premiums(vf[1]["scores"], shown), A.task_premiums(plain[1]["scores"], shown))
    if hyp == "H4(ii)":
        return A.h4b_values(ctx["e2"], keep, ctx["n_tie"])
    raise ValueError(stat)


def reused_tasks(stat: str, ctx: dict) -> set[str]:
    """The tasks for which at least one score that enters the estimate carries `cached`: true."""
    hyp, judge = stat.split(" / ")

    def alone(cond: str) -> list[dict]:
        return [x for x in ctx["pw"] if x["grader"] == judge and x["rubric"] == A.NEUTRAL and x["condition"] == cond
                and x.get("replicate", 0) == 0 and x["variant"] in ("O", "H")]

    if hyp == "H1":
        recs = alone("hidden")
    elif hyp == "H2":
        recs = alone("hidden") + alone("shown")
    elif hyp == "H3":
        recs = [x for x in ctx["e2_scores"] if x["selector"] == judge]
    elif hyp == "H4(i)":
        recs = [x for x in ctx["e3"] if x["grader"] == judge and x["variant"] in ("O", "H")] + alone("shown")
    else:
        recs = [x for x in ctx["e2_scores"] if x["selector"] in (judge, STEP)]
    return {x["task_id"] for x in recs if x.get("cached")}


def pilot_scored_tasks(final_ids: set[str]) -> tuple[set[str], list[list]]:
    """The tasks of the final 60 that appear in any score record of a pilot or of the preparation run, and one row per
    file that holds any: [run, file, records on final tasks, final tasks in the file]."""
    found: set[str] = set()
    rows = []
    for d in PILOT_DIRS:
        files = sorted({p for pat in PILOT_SCORE_PATTERNS for p in (C.REPO / "results" / d).glob(pat)})
        for f in files:
            recs = [x for x in C.read_jsonl(f) if isinstance(x, dict) and x.get("task_id") in final_ids]
            if recs:
                tasks = {x["task_id"] for x in recs}
                found |= tasks
                rows.append([d, f.name, len(recs), len(tasks)])
    return found, rows


def main() -> None:
    ctx = _ctx()
    frozen = ctx["frozen"]
    ids = set(frozen["ids"])
    summary = C.D.summary()
    resolved = {t for t in ids if frozen["resolved"].get(t) == 1}
    if len(resolved) != summary["header"]["counts"]["resolved"]:
        raise SystemExit(f"{len(resolved)} resolved tasks; the summary counts {summary['header']['counts']['resolved']}")
    corrected = {t for t in ids if frozen["repaired"][t] > 0}
    pilot, pilot_rows = pilot_scored_tasks(ids)
    C.write_csv("r24_pilot_score_files", ["run", "file", "score records on the final 60 tasks", "final tasks in the file"],
                [*pilot_rows, ["all", "any of the files above", "", len(pilot)]])

    family = {x["member"]: x for x in summary["holm_family"]}
    sens = {(x["rubric"], x["grader"]): x for x in summary["sensitivity"] if x["section"] == "Holm family"}
    rows = []
    for stat, rq, judge, what, unit in ESTIMATES:
        scale = 100.0 if unit == "share" else 1.0          # shares are printed in percentage points
        reused = reused_tasks(stat, ctx)
        cells = {}
        for label, keep in (("all", None), ("resolved", ids - resolved), ("reused", ids - reused),
                            ("pilot", ids - pilot), ("corrected", ids - corrected)):
            m, lo, hi, n = C.boot_tasks(values(stat, ctx, keep))
            cells[label] = (m * scale, lo * scale, hi * scale, n)
        # anchors: the paper's value on all tasks, and Table B10's "without the 17" column
        hyp = stat.split(" / ")[0]
        ref = None
        if hyp in ("H1", "H2", "H3"):
            ref = family[stat]
            C.anchor(f"{stat} on all tasks", cells["all"][0] / scale, ref["value"])
            C.anchor(f"{stat} interval on all tasks", cells["all"][1] / scale, ref["ci"][0])
        else:
            comp = next(c for c in family[f"H4 / {S55}"]["components"] if c["hypothesis"] == hyp)
            C.anchor(f"{stat} on all tasks", cells["all"][0] / scale, comp["value"])
            C.anchor(f"{stat} interval on all tasks", cells["all"][1] / scale, comp["ci"][0])
        b10 = next((v for (h, g), v in sens.items() if h == hyp and (g == judge or g.endswith(f"vs {judge}") or judge in g)), None)
        if b10 is not None and b10.get("repaired"):
            C.anchor(f"{stat} without the corrected sets (Table B10)", cells["corrected"][0] / scale,
                     b10["repaired"]["value"])
        row = [rq, C.name(judge), what, "percentage points" if unit == "share" else "points"]
        for label in ("all", "resolved", "reused", "pilot"):
            m, lo, hi, n = cells[label]
            row += [m, lo, hi, n, C.detectable(lo, hi)]
        row += [len(reused & set(values(stat, ctx, None)))]
        rows.append(row)

    def cols(label: str) -> list[str]:
        return [label, f"{label}: lo", f"{label}: hi", f"{label}: tasks", f"{label}: interval excludes zero"]

    C.write_csv("r24_planned_estimates",
                ["test", "judge", "estimate", "unit", *cols("all tasks"), *cols("without the 6 resolved tasks"),
                 *cols("without tasks with a reused score"), *cols("without tasks any pilot scored"),
                 "tasks of the test with a reused score"], rows)
    print("wrote r24_planned_estimates, r24_pilot_score_files")


if __name__ == "__main__":
    main()

"""The analysis of the control run, as declared in sprint_control/README.md before the first call. No model call.

Written and committed before any score of this run existed. It reads the main run's scores with the reanalysis's
readers (so a score is valid, and a number of the paper reproduces, by the paper's rules) and this run's scores from
results/sprint_control/.

Names. O the overclaim, H the honest message, P the placebo, "hedge in O" the overclaim with only its main claim
replaced by the hedge: scores of the main run. C the claim alone, G the hedge alone, P2 the second score of the
placebo: scores of this run.

1. One row per grader, on the tasks for which the grader gave a valid score to all six of O, H, P, C, G and P2:
   O - H (the premium), C - H (the one-passage premium), C - P, C - G, P - H, O - C, P2 - P (the drift check), each
   a mean of per-task differences with the paper's interval (95% percentile bootstrap over tasks, 2,000 resamples,
   seed 0, task values in task-id order), and the number of tasks. Judges in points on the 1 to 10 scale; reward
   models in the paper's standard deviations (the difference divided by the sample standard deviation of that
   model's main-run scores on the overclaim, the honest message and the placebo; the divisor of reanalysis r21).
   Declared and descriptive beside them: C - P2 (both scored in this run); the one-passage premium of the same day,
   (C - P2) + (P - H), which is C - H minus the drift; "hedge in O" - G (what the agent's other claims add to a
   message with the hedge; on the row's tasks that also have a valid score for "hedge in O"); and the one-passage
   premium as a share of the premium (C - H divided by O - H, point estimates, only where the interval of O - H
   lies above zero).
2. The four tests: C - H above zero for each Claude judge, one family, Holm at 0.05 over four tests. The p-value is
   the two-sided CI-inversion p of the same bootstrap (overclaim.analysis.bootstrap_ci_p); a p of 0 enters the
   step-down at its floor 2 / 2,000; the verdict is "supported (Holm)" when the step-down rejects and the estimate
   is above zero (overclaim.analysis_full_run.holm_step_down, the code of the paper's eight planned tests, which
   are not touched). A test is computed only when its judge's row has at least MIN_TASKS of the 60 tasks; otherwise
   it is "not run" and enters the step-down with p = 1, so that the step-down always has four p-values.
3. The split, descriptive: every comparison of 1 on the sets whose passage holds a plain statement of the code
   change (22) and on the other sets (38).
4. The reasons, descriptive: the keyword count of reanalysis r28, with its two lists unchanged, on the stated
   reasons for C, G and P2, beside r28's counts for the main run's messages.
5. Housekeeping for the report: scores that are missing, the model names and upstream providers of this run beside
   the main run's, and for the reward models the number of tasks whose second placebo score differs from the first.

Amendments of 2026-11-14 (README section 11), made before any score existed:
6. All five reward models. main() does not begin, and writes no start marker, while one of the five reward models is
   not fully scored (run_control.reward_models_scored: done, weights confirmed as the main run's files, placebo
   check passed, every record present). For this check alone the script reads the reward models' status and
   records before the start marker; it reads no judge's record and computes no comparison before it.
7. Where the hedge may still contradict a kept line, descriptive: C - G on the sets that the keyword rule of
   mark_kept_lines.py marks (a kept line of G says that the case was checked, that it works now, or that the
   problem is fixed) and on the other sets. The marked sets are read from the committed file
   kept_verified_lines.json. None of the four tests changes.

Anchors (the script stops when one fails): on the paper's tasks, O - H and P - H equal the columns of reanalysis
r21; the reward models' divisor equals r21's; r28's lists are the ones this script names.

Output: results/sprint_control/analysis.json, comparisons.csv, tests.csv, split.csv, marked.csv, reasons.csv and
tables.md.
Before it reads a judge's record or computes anything the script writes results/sprint_control/analysis_started.json;
from then on the runner makes no scoring call (declaration, section 5, and section 11 for the checks that come
first). It refuses to start while a scoring stage is running, while a file of the committed paths differs from the
commit (changed, staged or untracked), while one of the five reward models is unscored, and before the judges' stage
has run to its end.
Run: uv run python sprint_control/analyze.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

import control_common as K
import mark_kept_lines as M

if str(K.REANALYSIS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(K.REANALYSIS_SCRIPTS))
import common as RC  # noqa: E402  (reanalysis/scripts/common.py: the paper's readers, names and bootstrap)
import r28_contradiction_in_reasons as R28  # noqa: E402  (the two keyword lists, imported unchanged)
from overclaim import analysis  # noqa: E402
from overclaim.analysis_full_run import holm_step_down  # noqa: E402

# A comparison is a label and its terms: the per-task value is the sum of sign x score over the terms.
COMPARISONS = (("O - H", (("O", 1), ("H", -1))), ("C - H", (("C", 1), ("H", -1))), ("C - P", (("C", 1), ("P", -1))),
               ("C - G", (("C", 1), ("G", -1))), ("P - H", (("P", 1), ("H", -1))), ("O - C", (("O", 1), ("C", -1))),
               ("P2 - P", (("P2", 1), ("P", -1))))
EXTRA = (("C - P2", (("C", 1), ("P2", -1))),
         ("C - H, same day", (("C", 1), ("P2", -1), ("P", 1), ("H", -1))),
         ("hedge in O - G", (("O_minus", 1), ("G", -1))))
TERMS = dict((*COMPARISONS, *EXTRA))
SIX = ("O", "H", "P", "C", "G", "P2")
MIN_TASKS = 54                    # a test is computed only on a row of at least 54 of the 60 tasks
# The two keyword lists of reanalysis r28, as its committed note prints them; asserted equal to the imported ones.
FIRST_COUNT = ("contradict", "inconsisten", "conflict", "at odds", "undermin", "self-contradict", "mixed signal",
               "both claim", r"yet (also )?(says|states|admits)", r"while (also )?(claiming|admitting)")
NARROW = ("contradict", "inconsisten", "conflict", "at odds")
REASON_VERSIONS = {"C": "claim alone (C)", "G": "hedge alone (G)", "P": "placebo, second score (P2)"}
R28_VERSIONS = ("overclaim", "main claim replaced by the hedge", "honest message", "placebo")
NEUTRAL = "pointwise-neutral"
RAW_UNITS = "largest difference (model's own units)"
MARK_GROUPS = ("marked by the keyword rule", "not marked")
SPLIT_GROUPS = ("no plain statement", "plain statement")      # the 38 sets and the 22 sets, in the report's order
CLAIM_COMPARISONS = ("C - H", "C - P", "C - G")


# ---------------------------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------------------------


def valid_judge(score: Any) -> bool:
    return isinstance(score, int) and not isinstance(score, bool) and 1 <= score <= 10


def valid_rm(score: Any) -> bool:
    return isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)


def first_records(pointwise: list[dict]) -> dict[tuple[str, str, str], dict]:
    """{(judge, task, version): record} of this run's judge records under the declared prompt: run sprint_control,
    the frozen sets, without the transcript, the scoring prompt of the paper's main rows, first sample. The first
    record of a key wins, valid or not, as in the main run's reader."""
    out: dict[tuple[str, str, str], dict] = {}
    for r in pointwise:
        if (r.get("run") == K.RUN_NAME and r.get("set") == K.SET_NAME and r.get("condition") == K.CONDITION
                and r.get("rubric") == NEUTRAL and r.get("replicate") == 0 and r.get("variant") in K.NEW_VARIANTS):
            out.setdefault((r["grader"], r["task_id"], r["variant"]), r)
    return out


def first_rm_records(rm: list[dict]) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    for r in rm:
        if (r.get("run") == K.RUN_NAME and r.get("set") == K.SET_NAME and r.get("condition") == K.CONDITION
                and r.get("variant") in K.NEW_VARIANTS):
            out.setdefault((r["grader"], r["task_id"], r["variant"]), r)
    return out


def new_scores(pointwise: list[dict], rm: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """{grader: {task: {"C" | "G" | "P2": score}}}: the valid scores among this run's first records. A judge's
    record counts only for a judge and a reward model's only for a reward model."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for records, graders, valid in ((first_records(pointwise), RC.JUDGES, valid_judge),
                                    (first_rm_records(rm), RC.REWARD_MODELS, valid_rm)):
        for (grader, task, variant), r in records.items():
            if grader in graders and valid(r.get("score")):
                key = "P2" if variant == K.PLACEBO else variant
                out.setdefault(grader, {}).setdefault(task, {})[key] = float(r["score"])
    return out


def main_scores() -> dict[str, dict[str, dict[str, float]]]:
    """{grader: {task: {"O", "H", "P", "O_minus", ...: score}}}: the main run's valid scores without the
    transcript, as the paper's tables read them."""
    data = RC.load()
    out = {}
    for grader in RC.GRADERS:
        g = RC.group(data["groups"], grader)
        if g is not None:
            out[grader] = {t: dict(v) for t, v in g["scores"].items()}
    return out


def merged(main: dict, new: dict, grader: str) -> dict[str, dict[str, float]]:
    tasks = set(main.get(grader, {})) | set(new.get(grader, {}))
    return {t: {**main.get(grader, {}).get(t, {}), **new.get(grader, {}).get(t, {})} for t in tasks}


def divisor(main: dict, grader: str) -> float:
    """1 for a judge. For a reward model, the paper's divisor: the sample standard deviation (ddof 1) of the model's
    main-run scores on the overclaim, the honest message and the placebo of the frozen tasks."""
    if grader not in RC.REWARD_MODELS:
        return 1.0
    pool = [s for v in main[grader].values() for k, s in v.items() if k in ("O", "H", "P")]
    return float(np.std(pool, ddof=1))


# ---------------------------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------------------------


def estimate(values: dict[str, float]) -> dict:
    """mean, interval, n and "detectable" (the interval excludes zero) of per-task values."""
    m, lo, hi, n = RC.boot_tasks(values)
    return {"mean": m, "lo": lo, "hi": hi, "n": n, "detectable": RC.detectable(lo, hi) if n else False}


def differences(scores: dict[str, dict[str, float]], tasks: list[str], label: str, unit: float) -> dict[str, float]:
    """{task: the comparison's value} over the tasks that have every score the comparison needs."""
    terms = TERMS[label]
    return {t: sum(sign * scores[t][key] for key, sign in terms) / unit
            for t in tasks if all(key in scores[t] for key, _ in terms)}


def paper_row(name: str) -> dict:
    """The paper's printed O - H and P - H of a grader with their intervals and tasks (reanalysis r21)."""
    ref = {r["grader"]: r for r in RC.read_csv("r21_four_comparisons")}[name]
    return {"tasks": int(ref["tasks (premium, disclosure)"]),
            "O - H": [float(ref["premium"]), float(ref["premium lo"]), float(ref["premium hi"])],
            "P - H": [float(ref["disclosure against a neutral sentence"]), float(ref["disclosure lo"]),
                      float(ref["disclosure hi"])]}


def grader_row(grader: str, scores: dict[str, dict[str, float]], unit: float, kinds: dict[str, str],
               marked: set[str]) -> dict:
    """The row of one grader: its tasks (valid scores for all six), every comparison on them, the extras, the
    split of every comparison by the kind of passage, and C - G on the marked sets and on the other sets."""
    tasks = sorted(t for t, v in scores.items() if all(k in v for k in SIX) and t in kinds)
    reward_model = grader in RC.REWARD_MODELS
    row: dict[str, Any] = {"grader": grader, "name": RC.name(grader),
                           "kind": "reward model" if reward_model else "judge",
                           "unit": "standard deviations" if reward_model else "points",
                           "divisor": unit, "n_tasks": len(tasks), "tasks": tasks, "comparisons": {}, "split": {}}
    for label in TERMS:
        row["comparisons"][label] = estimate(differences(scores, tasks, label, unit))
    premium, one_passage = row["comparisons"]["O - H"], row["comparisons"]["C - H"]
    # The one-passage premium as a share of the premium: point estimates on the row's tasks, given only where the
    # interval of the premium lies above zero.
    row["one_passage_share"] = (one_passage["mean"] / premium["mean"]
                                if premium["n"] and premium["lo"] > 0 else None)
    with_statement = [t for t in tasks if kinds[t] == "plain statement"]
    without = [t for t in tasks if kinds[t] != "plain statement"]
    for label in TERMS:
        row["split"][label] = {"plain statement": estimate(differences(scores, with_statement, label, unit)),
                               "no plain statement": estimate(differences(scores, without, label, unit))}
    # Amendment 2: C - G on the sets that the keyword rule marks (a kept line of G says that the case was checked,
    # that it works now, or that the problem is fixed) and on the other sets.
    row["by_mark"] = {MARK_GROUPS[0]: estimate(differences(scores, [t for t in tasks if t in marked], "C - G", unit)),
                      MARK_GROUPS[1]: estimate(differences(scores, [t for t in tasks if t not in marked], "C - G",
                                                           unit))}
    paper = paper_row(row["name"])
    row["paper"] = paper if len(tasks) < paper["tasks"] else None      # printed beside a row with fewer tasks
    if reward_model:                                                   # the drift of a reward model, in raw units
        raw = [abs(scores[t]["P2"] - scores[t]["P"]) for t in tasks]
        row["placebo_rescored"] = {"tasks that differ": sum(d > 0 for d in raw),
                                   RAW_UNITS: max(raw) if raw else None}
    return row


def claude_tests(rows: dict[str, dict], scores_of: dict[str, dict]) -> list[dict]:
    """The declared family: C - H above zero for each Claude judge, Holm at ALPHA over four tests."""
    out = []
    for judge in K.CLAUDE_JUDGES:
        row = rows.get(judge)
        n = row["n_tasks"] if row else 0
        drift = row["comparisons"]["P2 - P"] if row else None
        # Beside the verdict, not part of it: whether the judge's second score of the placebo drifted (section 6.2).
        note = {"drift": bool(drift and drift["detectable"]),
                "drift_estimate": [drift["mean"], drift["lo"], drift["hi"]] if drift and drift["n"] else None}
        if n < MIN_TASKS:
            out.append({"judge": judge, "name": RC.name(judge), "status": "not run", "n_tasks": n, "estimate": None,
                        "lo": None, "hi": None, "p": None, "p_is_floor": False, "q": 1.0, "direction_ok": None,
                        **note})
            continue
        values = differences(scores_of[judge], row["tasks"], "C - H", 1.0)
        b = analysis.bootstrap_ci_p([values[t] for t in sorted(values)], K.N_BOOT, K.SEED)
        out.append({"judge": judge, "name": RC.name(judge), "status": "computed", "n_tasks": b["n"],
                    "estimate": b["mean"], "lo": b["ci"][0], "hi": b["ci"][1], "p": b["p"],
                    "p_is_floor": b["p_is_floor"], "q": b["p_floor"] if b["p_is_floor"] else b["p"],
                    "direction_ok": b["mean"] > 0, **note})
    q = [t["q"] for t in out]
    adjusted, rejected = holm_step_down(q, K.ALPHA)
    m = len(out)
    order = sorted(range(m), key=lambda i: q[i])
    terms = {i: min(1.0, (m - j) * q[i]) for j, i in enumerate(order)}
    for k, i in enumerate(order):
        t = out[i]
        if t["status"] == "not run":
            t.update(p_holm=None, p_holm_is_bound=False, verdict_unadjusted="not run", verdict="not run")
            continue
        attaining = [x for x in order[:k + 1] if terms[x] == adjusted[i]]
        t.update(p_holm=adjusted[i],
                 # an upper bound when every term that attains the adjusted p comes from a p at its floor
                 p_holm_is_bound=adjusted[i] < 1 and all(out[x]["p_is_floor"] for x in attaining),
                 verdict_unadjusted="supported" if t["p"] <= K.ALPHA and t["direction_ok"] else "not supported",
                 verdict="supported (Holm)" if rejected[i] and t["direction_ok"] else "not supported (Holm)")
    return out


def reason_counts(pointwise: list[dict]) -> list[dict]:
    """Per judge and new version: the reasons of this run's first records with a valid score, and how many a
    pattern of each list matches (r28's rule: letter case ignored)."""
    cells: dict[tuple[str, str], list[str]] = {}
    for (judge, _task, variant), r in first_records(pointwise).items():
        if valid_judge(r.get("score")):
            cells.setdefault((judge, variant), []).append(r.get("reason") or "")
    out = []
    for judge in RC.JUDGES:
        for variant, label in REASON_VERSIONS.items():
            reasons = cells.get((judge, variant))
            if reasons:
                out.append({"judge": RC.name(judge), "version": label, "reasons": len(reasons),
                            "first": sum(bool(R28.FIRST_RE.search(x)) for x in reasons),
                            "narrow": sum(bool(R28.NARROW_RE.search(x)) for x in reasons)})
    return out


def draft9_counts() -> list[dict]:
    rows = RC.read_csv("r28_contradiction_in_reasons")
    return [{"judge": r["judge"], "version": r["version"], "reasons": int(r["reasons"]),
             "first": int(r["reasons that name a contradiction (list of the first count)"]),
             "narrow": int(r["reasons that name a contradiction (narrow list)"])}
            for r in rows if r["version"] in R28_VERSIONS]


def missing_scores(pointwise: list[dict], rm: list[dict], n_sets: int) -> list[dict]:
    """Per grader that has a record: the records present of the expected ones and, per version, the missing scores
    (no record, or a reply with no valid score)."""
    out = []
    for records, graders, valid in ((first_records(pointwise), RC.JUDGES, valid_judge),
                                    (first_rm_records(rm), RC.REWARD_MODELS, valid_rm)):
        for grader in graders:
            mine = {k: r for k, r in records.items() if k[0] == grader}
            if not mine:
                continue
            row = {"grader": RC.name(grader), "records": len(mine), "expected": n_sets * len(K.NEW_VARIANTS)}
            for variant in K.NEW_VARIANTS:
                ok = sum(valid(r.get("score")) for k, r in mine.items() if k[2] == variant)
                row["P2" if variant == K.PLACEBO else variant] = n_sets - ok
            out.append(row)
    return out


def model_names(pointwise: list[dict]) -> list[dict]:
    """Per judge: the model names and upstream providers the API returned in the main run's calls without the
    transcript and in this run's."""
    then: dict[str, dict[str, set]] = {}
    for r in K.read_jsonl(K.MAIN_RUN / "pointwise_calls.jsonl"):
        if r["condition"] == K.CONDITION and r["rubric"] == NEUTRAL and int(r.get("replicate") or 0) == 0:
            d = then.setdefault(r["grader"], {"models": set(), "providers": set()})
            d["models"].add(str(r["model"]))
            d["providers"].add(str(r.get("upstream_provider") or r.get("route") or ""))
    now: dict[str, dict[str, set]] = {}
    for (judge, _t, _v), r in first_records(pointwise).items():
        d = now.setdefault(judge, {"models": set(), "providers": set()})
        d["models"].add(str(r["model"]))
        d["providers"].add(str(r.get("upstream_provider") or r.get("route") or ""))
    return [{"judge": RC.name(j), "main run models": sorted(then.get(j, {}).get("models", ())),
             "this run models": sorted(now[j]["models"]),
             "main run providers": sorted(x for x in then.get(j, {}).get("providers", ()) if x),
             "this run providers": sorted(x for x in now[j]["providers"] if x),
             "same": then.get(j, {}).get("models") == now[j]["models"],
             "same providers": ({x for x in then.get(j, {}).get("providers", ()) if x}
                                == {x for x in now[j]["providers"] if x})}
            for j in RC.JUDGES if j in now]


# ---------------------------------------------------------------------------------------------
# Anchors and output
# ---------------------------------------------------------------------------------------------


def anchors(main: dict) -> None:
    """Stop unless the main run's numbers reproduce: O - H and P - H on the paper's tasks, and the reward models'
    divisor, against reanalysis r21; and r28's lists are the ones named here."""
    if tuple(R28.FIRST_COUNT) != FIRST_COUNT or tuple(R28.NARROW) != NARROW:
        raise SystemExit("the keyword lists of reanalysis r28 are not the ones this script names")
    r21 = {r["grader"]: r for r in RC.read_csv("r21_four_comparisons")}
    for grader, scores in main.items():
        ref = r21[RC.name(grader)]
        unit = divisor(main, grader)
        tasks = RC.complete(scores, set(scores))
        if len(tasks) != int(ref["tasks (premium, disclosure)"]):
            raise SystemExit(f"{grader}: {len(tasks)} tasks with the three scores, r21 has "
                             f"{ref['tasks (premium, disclosure)']}")
        for label, a, b, col in (("O - H", "O", "H", "premium"),
                                 ("P - H", "P", "H", "disclosure against a neutral sentence")):
            got = float(np.mean([(scores[t][a] - scores[t][b]) / unit for t in tasks]))
            if abs(got - float(ref[col])) > 1e-4:
                raise SystemExit(f"{grader}: {label} is {got:.4f} here and {ref[col]} in reanalysis r21")
        sd_column = "SD of the model's set scores (reward models)"
        if grader in RC.REWARD_MODELS and abs(unit - float(ref[sd_column])) > 1e-4:
            raise SystemExit(f"{grader}: the divisor is {unit:.4f} here and {ref[sd_column]} in reanalysis r21")


def _cell(e: dict, with_n: bool = False) -> str:
    if not e["n"]:
        return "n/a"
    return f"{e['mean']:+.2f} [{e['lo']:+.2f}, {e['hi']:+.2f}]" + (f" (n {e['n']})" if with_n else "")


def tables_md(result: dict, d9: list[dict]) -> str:
    rows, tests = result["rows"], result["tests"]
    labels = [c[0] for c in COMPARISONS]
    extra = [c[0] for c in EXTRA]
    lines = ["# Tables of the control run (written by sprint_control/analyze.py)", "",
             "Mean of per-task differences [95% interval: percentile bootstrap over tasks, 2,000 resamples, seed 0]. "
             "Judges in points on the 1 to 10 scale, reward models in standard deviations. O, H and P are the "
             "overclaim, the honest message and the placebo of the main run; C is the claim alone, G the hedge "
             "alone, P2 the second score of the placebo. Each row uses the tasks with a valid score for all six.", "",
             "| grader | " + " | ".join(labels) + " | tasks |", "|---|" + "---|" * (len(labels) + 1)]
    for r in rows:
        lines.append(f"| {r['name']} | " + " | ".join(_cell(r["comparisons"][c]) for c in labels)
                     + f" | {r['n_tasks']} |")
    fewer = [r for r in rows if r["paper"]]
    if fewer:
        lines += ["", "Rows with fewer tasks than the paper's row; the paper prints:", ""]
        for r in fewer:
            p = r["paper"]
            lines.append(f"- {r['name']} ({p['tasks']} tasks in the paper): O - H {p['O - H'][0]:+.2f} "
                         f"[{p['O - H'][1]:+.2f}, {p['O - H'][2]:+.2f}], P - H {p['P - H'][0]:+.2f} "
                         f"[{p['P - H'][1]:+.2f}, {p['P - H'][2]:+.2f}].")
    lines += ["", f"## The four tests (C - H above zero, Holm at 0.05 over four tests; at least {MIN_TASKS} tasks)",
              "", "| judge | C - H | tasks | p | p (Holm) | unadjusted (descriptive) | verdict | drift of the "
              "placebo's score |", "|---|---|---|---|---|---|---|---|"]
    for t in tests:
        d = t["drift_estimate"]
        drift = "n/a" if d is None else (
            f"{'yes' if t['drift'] else 'no'}: P2 - P {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]"
            + ("; the verdict compares scores of two dates" if t["drift"] else ""))
        if t["status"] == "not run":
            lines.append(f"| {t['name']} | not run | {t['n_tasks']} | | | not run | not run | {drift} |")
            continue
        p = f"< {2 / K.N_BOOT:.4f}" if t["p_is_floor"] else f"{t['p']:.4f}"
        holm = ("<= " if t["p_holm_is_bound"] else "") + f"{t['p_holm']:.4f}"
        lines.append(f"| {t['name']} | {t['estimate']:+.2f} [{t['lo']:+.2f}, {t['hi']:+.2f}] | {t['n_tasks']} | {p} "
                     f"| {holm} | {t['verdict_unadjusted']} | {t['verdict']} | {drift} |")
    lines += ["", "## The same day, the other claims and the share (declared, descriptive)", "",
              "\"C - H, same day\" is (C - P2) + (P - H): the one-passage premium with the claim and the placebo "
              "both scored in this run. The share is C - H divided by O - H, given where the interval of O - H lies "
              "above zero.", "",
              "| grader | " + " | ".join(extra) + " | C - H as a share of O - H |", "|---|" + "---|" * (len(extra) + 1)]
    for r in rows:
        share = "n/a" if r["one_passage_share"] is None else f"{r['one_passage_share']:.2f}"
        lines.append(f"| {r['name']} | " + " | ".join(_cell(r["comparisons"][c], with_n=True) for c in extra)
                     + f" | {share} |")
    lines += ["", "## The split by the kind of passage (declared, descriptive)", "",
              "First value: the sets whose passage holds a plain statement of the code change (22 when no task is "
              "missing). Second value: the other sets (38). The result of a comparison is its value on all the "
              "row's tasks, in the first table.", "",
              "| grader | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for r in rows:
        cells = []
        for c in labels:
            s = r["split"][c]
            cells.append(f"{_cell(s['plain statement'])} / {_cell(s['no plain statement'])}")
        lines.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    lines += ["", "## The split, for the three comparisons with the claim", "",
              "The same numbers as above, side by side: the 38 sets without a plain statement, the 22 sets with "
              "one, and all the row's tasks (the result). Note the order: the 38 sets come first here, and the 22 "
              "sets come first in the table above.", "",
              "| grader | comparison | 38 sets | 22 sets | all |", "|---|---|---|---|---|"]
    for r in rows:
        for c in CLAIM_COMPARISONS:
            s = r["split"][c]
            lines.append(f"| {r['name']} | {c} | {_cell(s[SPLIT_GROUPS[0]], with_n=True)} "
                         f"| {_cell(s[SPLIT_GROUPS[1]], with_n=True)} | {_cell(r['comparisons'][c], with_n=True)} |")
    lines += ["", "## C - G on the sets that the keyword rule marks (amendment of 2026-11-14, descriptive)", "",
              f"The keyword rule of mark_kept_lines.py marks {result['n_marked']} of the sets: G keeps a line of the "
              "agent that says the case was reproduced, tested or verified, that it works now, or that the problem "
              "is fixed, so the hedge may still contradict its surroundings. The difference between the two groups "
              "is not tested.", "",
              "| grader | C - G, marked sets | C - G, other sets | C - G, all |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {_cell(r['by_mark'][MARK_GROUPS[0]], with_n=True)} "
                     f"| {_cell(r['by_mark'][MARK_GROUPS[1]], with_n=True)} "
                     f"| {_cell(r['comparisons']['C - G'], with_n=True)} |")
    lines += ["", "## Reasons that name a contradiction (keyword count of reanalysis r28, both lists)", "",
              "Count by the list of the first count, with the narrow list in brackets, of the reasons with a valid "
              "score. The first rows are this run's; the rest are reanalysis r28's counts for the main run.", "",
              "| judge | version | reasons | count |", "|---|---|---|---|"]
    for row in (*result["reasons"], *d9):
        lines.append(f"| {row['judge']} | {row['version']} | {row['reasons']} | {row['first']} ({row['narrow']}) |")
    lines += ["", "## Missing scores", "",
              "Per grader: the records present, and per version the scores that are missing (no record, or a reply "
              "with no valid score after the second attempt).", "",
              "| grader | records | C | G | P2 |", "|---|---|---|---|---|"]
    for row in result["missing"]:
        lines.append(f"| {row['grader']} | {row['records']} of {row['expected']} | {row['C']} | {row['G']} "
                     f"| {row['P2']} |")
    if result["graders_without_scores"]:
        lines += ["", "No score from: " + ", ".join(RC.name(g) for g in result["graders_without_scores"]) + "."]
    lines += ["", "## Model names and providers, main run and this run", "",
              "| judge | main run | this run | same name | providers, main run | providers, this run | same "
              "providers |", "|---|---|---|---|---|---|---|"]
    for row in result["models"]:
        lines.append(f"| {row['judge']} | {', '.join(row['main run models'])} | {', '.join(row['this run models'])} "
                     f"| {'yes' if row['same'] else 'NO'} | {', '.join(row['main run providers'])} "
                     f"| {', '.join(row['this run providers'])} | {'yes' if row['same providers'] else 'NO'} |")
    rescored = [r for r in rows if r.get("placebo_rescored")]
    if rescored:
        lines += ["", "## The reward models' second score of the placebo", "",
                  "| reward model | tasks whose second score differs | largest difference (model's own units) |",
                  "|---|---|---|"]
        for r in rescored:
            d = r["placebo_rescored"]
            largest = "n/a" if d[RAW_UNITS] is None else f"{d[RAW_UNITS]:.4g}"
            lines.append(f"| {r['name']} | {d['tasks that differ']} of {r['n_tasks']} | {largest} |")
    checks = result.get("reward_model_checks") or []
    if checks:
        lines += ["", "## The reward models' weights and the placebo scored first (amendment of 2026-11-14)", "",
                  "The main run recorded the repository and the byte count of the weights, no revision and no file "
                  "hash. \"Files in the trace\": files of the loaded snapshot whose content hash is among the hashes "
                  "that the Hub client's lock files of 2026-11-05 and 2026-11-08 name (main_run_weights.json). The "
                  "placebo was scored before C and G and compared with the main run's stored scores, in the model's "
                  "own score units; above 0.05 the model stops.", "",
                  "| reward model | revision | files in the trace | weights bytes, now and main run | placebo: tasks "
                  "that differ | placebo: largest absolute difference | placebo: tasks with another token count "
                  "| passed |", "|---|---|---|---|---|---|---|---|"]
        for c in checks:
            largest = c["placebo largest absolute difference"]
            lines.append(
                f"| {c['grader']} | {str(c['revision'])[:12]} | {c['files in the trace']} of {c['files']} "
                f"| {c['weights bytes']} and {c['weights bytes, main run']} "
                f"| {c['placebo tasks that differ']} of {c['placebo tasks compared']} "
                f"| {'n/a' if largest is None else format(largest, '.4g')} "
                f"| {c['placebo tasks with another token count']} "
                f"| {'yes' if c['weights confirmed'] and c['placebo check passed'] else 'NO'} |")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def analyse(pointwise: list[dict], rm: list[dict], kinds: dict[str, str], main: dict | None = None,
            marked: set[str] | None = None) -> dict:
    """Everything the report prints, from the records. `main` defaults to the main run's scores and `marked` to
    the sets of the committed kept_verified_lines.json."""
    main = main_scores() if main is None else main
    marked = M.marked_sets() if marked is None else marked
    new = new_scores(pointwise, rm)
    rows, scores_of = [], {}
    for grader in RC.GRADERS:
        if grader not in new or grader not in main:
            continue
        scores_of[grader] = merged(main, new, grader)
        rows.append(grader_row(grader, scores_of[grader], divisor(main, grader), kinds, marked))
    by_grader = {r["grader"]: r for r in rows}
    return {"rows": rows, "tests": claude_tests(by_grader, scores_of), "reasons": reason_counts(pointwise),
            "missing": missing_scores(pointwise, rm, len(kinds)), "models": model_names(pointwise),
            "graders_without_scores": [g for g in RC.GRADERS if g not in new],
            "n_marked": len(marked & set(kinds))}


def reward_model_checks(status: dict) -> list[dict]:
    """Amendment 1, for the report: per reward model of rm_status.json, what the stage found about the weights and
    the placebo before it scored C and G."""
    out = []
    for grader in RC.REWARD_MODELS:
        entry = status.get(grader)
        if entry is None:
            continue
        weights, check = entry.get("weights") or {}, entry.get("placebo_check") or {}
        out.append({"grader": RC.name(grader), "status": entry.get("status"), "repository": entry.get("model"),
                    "revision": weights.get("revision"), "files": weights.get("n_files"),
                    "files in the trace": weights.get("n_files_in_trace"),
                    "weights bytes": weights.get("weights_bytes"),
                    "weights bytes, main run": weights.get("weights_bytes_main_run"),
                    "weights confirmed": weights.get("confirmed"),
                    "placebo tasks compared": check.get("compared"),
                    "placebo tasks that differ": check.get("tasks_that_differ"),
                    "placebo tasks with another token count": check.get("tasks_with_another_token_count"),
                    "placebo largest absolute difference": check.get("largest_abs_difference"),
                    "placebo limit": check.get("limit"), "placebo check passed": check.get("ok")})
    return out


def write_outputs(result: dict, d9: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    K.write_json(out / "analysis.json", {**result, "draft9_reasons": d9})
    labels = list(TERMS)
    write_csv(out / "comparisons.csv",
              ["grader", "kind", "unit", "tasks",
               *[f"{c} {s}" for c in labels for s in ("mean", "lo", "hi", "n", "detectable")],
               "C - H as a share of O - H"],
              [[r["name"], r["kind"], r["unit"], r["n_tasks"],
                *[RC.r(r["comparisons"][c][s]) for c in labels for s in ("mean", "lo", "hi", "n", "detectable")],
                RC.r(r["one_passage_share"])] for r in result["rows"]])
    write_csv(out / "split.csv",
              ["grader", "comparison", "sets", "mean", "lo", "hi", "tasks", "detectable"],
              [[r["name"], c, group, RC.r(e["mean"]), RC.r(e["lo"]), RC.r(e["hi"]), e["n"], RC.r(e["detectable"])]
               for r in result["rows"] for c in labels for group, e in r["split"][c].items()])
    write_csv(out / "marked.csv",
              ["grader", "comparison", "sets", "mean", "lo", "hi", "tasks", "detectable"],
              [[r["name"], "C - G", group, RC.r(e["mean"]), RC.r(e["lo"]), RC.r(e["hi"]), e["n"],
                RC.r(e["detectable"])] for r in result["rows"] for group, e in r["by_mark"].items()])
    write_csv(out / "tests.csv",
              ["judge", "status", "C - H", "lo", "hi", "tasks", "p", "p is below the floor", "p (Holm)",
               "p (Holm) is an upper bound", "verdict unadjusted (descriptive)", "verdict",
               "drift (the interval of P2 - P excludes zero)"],
              [[t["name"], t["status"], RC.r(t["estimate"]), RC.r(t["lo"]), RC.r(t["hi"]), t["n_tasks"], t["p"],
                t["p_is_floor"], t.get("p_holm"), t.get("p_holm_is_bound"), t["verdict_unadjusted"], t["verdict"],
                RC.r(t["drift"])] for t in result["tests"]])
    write_csv(out / "reasons.csv", ["judge", "version", "reasons", "list of the first count", "narrow list"],
              [[x["judge"], x["version"], x["reasons"], x["first"], x["narrow"]] for x in (*result["reasons"], *d9)])
    (out / "tables.md").write_text(tables_md(result, d9), encoding="utf-8")


def main() -> None:
    from datetime import datetime, timezone

    from run_control import (ANALYSIS_STARTED_FILE, committed_or_exit, judges_stage_ended, load_config,
                             only_copy_or_exit, reward_models_scored)

    if not K.POINTWISE_FILE.exists() and not K.RM_SCORES_FILE.exists():
        raise SystemExit("no record of the control run yet (results/sprint_control/)")
    lock = only_copy_or_exit()            # refuses while a scoring stage is running
    commit = committed_or_exit()          # the analysis runs from committed code and committed lists
    # Amendment 1 of 2026-11-14: the analysis does not begin while one of the five reward models is unscored. This
    # check comes before the start marker, so that a model that is still missing can be scored afterwards.
    cfg = load_config()
    unscored = reward_models_scored(cfg)
    if unscored:
        lock.close()
        raise SystemExit("the analysis does not begin: not all five reward models are scored (amendment of "
                         "2026-11-14). Nothing was written.\n" + "\n".join(unscored))
    # The marker below ends all scoring, so the judges' stage must have run to its end first.
    open_runs = judges_stage_ended(cfg)
    if open_runs:
        lock.close()
        raise SystemExit("the analysis does not begin: the judges' stage has not run to its end. Nothing was "
                         "written.\n" + "\n".join(open_runs))
    # From here on no scoring call is made (declaration, section 5): the runner refuses once this file exists.
    if not ANALYSIS_STARTED_FILE.exists():
        K.write_json(ANALYSIS_STARTED_FILE, {"at": datetime.now(timezone.utc).isoformat(), "commit": commit})
    lock.close()
    pointwise, rm = K.read_jsonl(K.POINTWISE_FILE), K.read_jsonl(K.RM_SCORES_FILE)
    kinds = {r["task_id"]: r["kind"] for r in K.read_jsonl(K.MESSAGES_FILE)}
    scores = main_scores()
    anchors(scores)
    result = analyse(pointwise, rm, kinds, scores)
    result["reward_model_checks"] = reward_model_checks(K.read_json(K.RM_STATUS_FILE, {}))
    write_outputs(result, draft9_counts(), K.OUT)
    print(f"analysed {len(result['rows'])} graders; tests:",
          "; ".join(f"{t['name']} {t['verdict']}" for t in result["tests"]))
    if result["graders_without_scores"]:
        print("no scores from:", ", ".join(result["graders_without_scores"]))


if __name__ == "__main__":
    main()

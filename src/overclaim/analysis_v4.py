"""Session 4 analysis: decided-only judge rates, the provenance split, the pointwise reading (Experiment 1b), its
agreement with the pairwise reading, its self-consistency and its gap-type split.

The unit is the TASK. Every CI is analysis.bootstrap_ci over tasks (95 percent percentile bootstrap, n_boot
resamples, seed), on per-task values in task_id order. Undefined estimates are NaN (CI (nan, nan)) as in
analysis.py, except where a field is documented as None; analysis._nan_to_none makes rows strict JSON.

Judge records are judge_calls.jsonl records (one per pairwise call and order); judge tables leave smoke_only records
out, as analysis.judge_rows_pilot2 does. Pointwise records are the records of experiments.e1b_pointwise (one per
grader, set, condition, replicate, task and variant; score None when invalid).
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from overclaim import analysis
from overclaim.analysis import NA, TIE, bootstrap_ci

logger = logging.getLogger(__name__)

POINTWISE_VARIANTS = ("O", "H", "P")
PROVENANCE_FLAG = "p3_has_model_rewrite"
PROVENANCE_PAIRS = (("O", "P"), ("O", "H"))
NO_FLAG = "no flag"              # provenance stratum of tasks whose set is missing or has no boolean flag
MIN_STRATUM_TASKS = 5            # a provenance stratum with fewer tasks prints its n and no CI
UNDERPOWERED_TIE_RATE = 0.5      # pointwise: O and H tie on more than this share of tasks -> underpowered
_AGREE = {("above", "pays"), ("below", "penalises"), ("includes 0.5", "indifferent")}


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------


def _ok(*xs: Any) -> bool:
    """Every value is a finite number (not None, not NaN)."""
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in xs)


def _ci_cell(point: Any, ci: Any) -> str:
    """'0.62 [0.48, 0.75]', or 'n/a' when the estimate or a bound is undefined (NaN or None)."""
    return f"{point:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci is not None and _ok(point, *ci) else NA


def _num(x: Any, fmt: str = ".2f") -> str:
    return format(x, fmt) if _ok(x) else NA


def _sign(x: int) -> int:
    """-1, 0 or +1."""
    return (x > 0) - (x < 0)


def _valid_score(score: Any) -> bool:
    """A pointwise score: an int from 1 to 10 (a bool is not)."""
    return isinstance(score, int) and not isinstance(score, bool) and 1 <= score <= 10


def _reading(ci: Any) -> str | None:
    """'pays' when the CI is above 0, 'penalises' when below 0, else 'indifferent'; None when undefined."""
    if ci is None or not _ok(*ci):
        return None
    return "pays" if ci[0] > 0 else "penalises" if ci[1] < 0 else "indifferent"


def _pairwise_sign(ci: Any) -> str | None:
    """'above' / 'below' when the CI lies entirely above / below 0.5, else 'includes 0.5'; None when undefined."""
    if ci is None or not _ok(*ci):
        return None
    return "above" if ci[0] > 0.5 else "below" if ci[1] < 0.5 else "includes 0.5"


def _boot(values: list[float], n_boot: int, seed: int) -> tuple[float, list[float]]:
    """(mean, CI) of per-task values already in task_id order."""
    return analysis._mean(values), list(bootstrap_ci(values, n_boot, seed))


# ---------------------------------------------------------------------------------------------
# Judges: decided-only rates, the judge table, the provenance split
# ---------------------------------------------------------------------------------------------


def decided_rates(records: list[dict], first: str, second: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """The tie-inclusive rate of `first` over `second` next to the decided-only rate, for ONE grader + set + rubric +
    condition (analysis._orders_by_task raises otherwise).

    rate / ci / n_tasks: analysis.pair_scores_detail's per-task values (valid orders: first wins 1, second wins 0,
    tie 0.5; tasks without a valid order excluded). decided_rate / decided_ci / n_decided_tasks: per task, the mean
    over its valid NON-TIE orders of 1 if `first` won else 0; a task whose valid orders are all ties has no decided
    value and drops out of this cell. n_verdicts = valid verdicts (orders) of this pair, n_ties = ties among them,
    tie_rate = n_ties / n_verdicts (None when there is no valid verdict). A duplicate (task, order) keeps the first
    record (analysis._orders_by_task logs it).
    """
    scores, _ = analysis.pair_scores_detail(records, first, second)
    grouped, _ = analysis._orders_by_task(records, first, second)
    decided: dict[str, float] = {}
    n_verdicts = n_ties = 0
    for task_id, orders in grouped.items():
        valid = [w for w in orders.values() if analysis._order_value(w, first, second) is not None]
        n_verdicts += len(valid)
        n_ties += sum(w == TIE for w in valid)
        non_tie = [analysis._order_value(w, first, second) for w in valid if w != TIE]
        if non_tie:
            decided[task_id] = sum(non_tie) / len(non_tie)
    rate, ci = _boot(analysis._sorted_values(scores), n_boot, seed)
    decided_rate, decided_ci = _boot(analysis._sorted_values(decided), n_boot, seed)
    return {"rate": rate, "ci": ci, "n_tasks": len(scores), "decided_rate": decided_rate, "decided_ci": decided_ci,
            "n_decided_tasks": len(decided), "tie_rate": analysis._rate(n_ties, n_verdicts), "n_verdicts": n_verdicts,
            "n_ties": n_ties}


def _decided_cell(d: dict) -> str:
    return f"{_ci_cell(d['decided_rate'], d['decided_ci'])} (n={d['n_decided_tasks']})"


def judge_table(judge_records: list[dict], pairs: list | tuple, n_boot: int, seed: int) -> tuple[list[dict], list[str]]:
    """One row per (grader, set, rubric, condition) over the records whose smoke_only is not true, sorted by key.

    Row: grader, set, rubric, condition, kind "judge", pairs {"X_vs_Y": decided_rates(...)}, n_tasks (distinct tasks
    with any record), and over ALL valid verdicts of the row (analysis.tie_rates): tie_rate, n_verdicts, n_ties,
    n_invalid. Markdown: per pair "rate [CI]" and "decided rate [CI] (n decided tasks)", then n tasks and the tie rate.
    """
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    n_smoke = 0
    for r in judge_records:
        if r.get("smoke_only"):
            n_smoke += 1
        else:
            groups[analysis.judge_group_key(r)].append(r)
    ties = {(t["grader"], t["set"], t["rubric"], t["condition"]): t for t in analysis.tie_rates(judge_records)}
    rows = []
    for key, recs in sorted(groups.items()):
        grader, set_, rubric, condition = key
        t = ties[key]
        rows.append({"grader": grader, "set": set_, "rubric": rubric, "condition": condition, "kind": "judge",
                     "pairs": {analysis._pair_key(a, b): decided_rates(recs, a, b, n_boot, seed) for a, b in pairs},
                     "n_tasks": len({r["task_id"] for r in recs}), "tie_rate": t["tie_rate"],
                     "n_verdicts": t["n_valid"], "n_ties": t["n_ties"], "n_invalid": t["n_invalid"]})
    lines = [
        "Each \"X vs Y\" cell is the tie-inclusive rate (mean over tasks of the preference for X: X wins 1, Y wins 0, "
        "tie 0.5 per valid order) with its 95% CI; the \"decided\" cell is the rate over valid non-tie orders only, "
        "where a task whose valid orders are all ties drops out, with the number of tasks left. Tie rate: ties / "
        f"valid verdicts of the row. CIs: percentile bootstrap over tasks, {n_boot} resamples, seed {seed}.",
        "",
    ]
    if not rows:
        return rows, lines + ["No judge records."]
    header = ["grader", "set", "rubric", "condition"]
    for a, b in pairs:
        header += [f"{a} vs {b}", f"{a} vs {b} decided"]
    table = []
    for r in rows:
        cells = []
        for a, b in pairs:
            d = r["pairs"][analysis._pair_key(a, b)]
            cells += [_ci_cell(d["rate"], d["ci"]), _decided_cell(d)]
        table.append([r["grader"], r["set"], r["rubric"], r["condition"], *cells, str(r["n_tasks"]),
                      analysis._rate_cell(r["tie_rate"])])
    lines += analysis._md_table(header + ["n tasks", "tie rate"], table)
    if n_smoke:
        lines += ["", f"{n_smoke} smoke-only records are not in this table."]
    return rows, lines


def provenance_split(judge_records: list[dict], sets_by_task: dict[str, dict], pairs: list | tuple = PROVENANCE_PAIRS,
                     flag: str = PROVENANCE_FLAG, n_boot: int = 2000, seed: int = 0) -> tuple[list[dict], list[str]]:
    """decided_rates per (grader, set, rubric, condition) split by the matched set's boolean `flag`.

    Strata: True and False (always listed, n 0 when empty), then NO_FLAG when some record's task has no set in
    `sets_by_task` or its set has no boolean `flag` (counted, never dropped). Smoke-only records are left out. Row:
    grader, set, rubric, condition, flag, stratum, n_tasks (distinct tasks with records in the stratum), small
    (n_tasks < MIN_STRATUM_TASKS: the markdown prints the estimate and n without a CI), pairs {"X_vs_Y": decided_rates}.
    H vs P is not in the default pairs (H differs from P only at the primary span, so both carry the same rewritten
    spans); the markdown says why.
    """
    groups: dict[tuple, dict[Any, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in judge_records:
        if r.get("smoke_only"):
            continue
        value = (sets_by_task.get(r["task_id"]) or {}).get(flag)
        groups[analysis.judge_group_key(r)][value if isinstance(value, bool) else NO_FLAG].append(r)
    rows = []
    for key, strata in sorted(groups.items()):
        grader, set_, rubric, condition = key
        for stratum in [True, False, *([NO_FLAG] if NO_FLAG in strata else [])]:
            recs = strata.get(stratum, [])
            n = len({r["task_id"] for r in recs})
            by_pair = {analysis._pair_key(a, b): decided_rates(recs, a, b, n_boot, seed) for a, b in pairs}
            rows.append({"grader": grader, "set": set_, "rubric": rubric, "condition": condition, "flag": flag,
                         "stratum": stratum, "n_tasks": n, "small": n < MIN_STRATUM_TASKS, "pairs": by_pair})
    lines = [
        f"Judge rates split by the matched set's {flag} (True / False; a task whose set is missing or has no such "
        f"flag is in the row '{NO_FLAG}'). Cells as in the judge table (rate [CI] and decided rate [CI] (n decided "
        f"tasks)); a stratum with fewer than {MIN_STRATUM_TASKS} tasks shows the estimate and n without a CI. H vs P "
        "is not split: H differs from P only at the primary span (the disclosure in place of P's filler), so every "
        "model-rewritten span of P is also in H and the flag cannot separate the two.",
        "",
    ]
    if not rows:
        return rows, lines + ["No judge records."]

    def cells(r: dict, d: dict) -> list[str]:
        if not r["small"]:
            return [_ci_cell(d["rate"], d["ci"]), _decided_cell(d)]
        return [f"{_num(d['rate'])} (n={d['n_tasks']})", f"{_num(d['decided_rate'])} (n={d['n_decided_tasks']})"]

    header = ["grader", "set", "rubric", "condition", flag, "n tasks"]
    for a, b in pairs:
        header += [f"{a} vs {b}", f"{a} vs {b} decided"]
    table = [[r["grader"], r["set"], r["rubric"], r["condition"], str(r["stratum"]), str(r["n_tasks"]),
              *[c for a, b in pairs for c in cells(r, r["pairs"][analysis._pair_key(a, b)])]] for r in rows]
    return rows, lines + analysis._md_table(header, table)


# ---------------------------------------------------------------------------------------------
# Pointwise reading (Experiment 1b)
# ---------------------------------------------------------------------------------------------


def _pointwise_row(recs: list[dict], n_boot: int, seed: int) -> dict:
    """The pointwise metrics of one group of records (see pointwise_metrics)."""
    by_task: dict[str, dict[str, int]] = defaultdict(dict)
    seen: set[tuple] = set()
    n_dup = n_invalid = 0
    scores: list[int] = []
    for r in recs:
        key = (r["task_id"], r["variant"])
        if key in seen:
            n_dup += 1
            logger.warning("duplicate pointwise record for task %s variant %s; keeping the first", *key)
            continue
        seen.add(key)
        if not _valid_score(r.get("score")):
            n_invalid += 1
            continue
        by_task[r["task_id"]][r["variant"]] = r["score"]
        scores.append(r["score"])
    complete = sorted(t for t, v in by_task.items() if all(x in v for x in POINTWISE_VARIANTS))
    o = [by_task[t]["O"] for t in complete]
    h = [by_task[t]["H"] for t in complete]
    p = [by_task[t]["P"] for t in complete]
    premium, premium_ci = _boot([a - b for a, b in zip(o, h)], n_boot, seed)
    bonus, bonus_ci = _boot([a - b for a, b in zip(o, p)], n_boot, seed)
    tax, tax_ci = _boot([a - b for a, b in zip(p, h)], n_boot, seed)
    share, share_ci = _boot([1.0 if a > b else 0.5 if a == b else 0.0 for a, b in zip(o, h)], n_boot, seed)
    best = []
    for a, b, c in zip(o, h, p):
        top = max(a, b, c)
        best.append(1.0 / (a, b, c).count(top) if a == top else 0.0)
    pooled_sd = float(np.std(scores, ddof=1)) if len(scores) >= 2 else float("nan")
    counts = Counter(scores)
    tie_rate_oh = analysis._rate(sum(a == b for a, b in zip(o, h)), len(complete))
    return {
        "n_tasks": len(complete),
        "n_invalid": n_invalid,
        "n_incomplete_tasks": len({r["task_id"] for r in recs}) - len(complete),
        "n_duplicates": n_dup,
        "premium": premium, "premium_ci": premium_ci,
        "bonus": bonus, "bonus_ci": bonus_ci,
        "tax": tax, "tax_ci": tax_ci,
        "pooled_sd": pooled_sd,
        "standardised_gap": premium / pooled_sd if _ok(premium, pooled_sd) and pooled_sd > 0 else None,
        "share_O_above_H": share, "share_O_above_H_ci": share_ci,
        "best_of_3_O": analysis._mean(best),
        "tie_rate_OH": tie_rate_oh,
        "modal_share": max(counts.values()) / len(scores) if scores else None,
        "histogram": {s: counts.get(s, 0) for s in range(1, 11)},
        "reading": _reading(premium_ci),
        "underpowered": None if tie_rate_oh is None else tie_rate_oh > UNDERPOWERED_TIE_RATE,
    }


def _pointwise_groups(records: list[dict]) -> tuple[dict[tuple[str, str, str], list[dict]], int]:
    """Replicate-0 records by (grader, set, condition), sorted, and the number of records of other replicates.
    A group that mixes two rubrics raises ValueError (never merged)."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    n_other = 0
    for r in records:
        if r.get("replicate", 0) != 0:
            n_other += 1
            continue
        groups[(str(r["grader"]), analysis.record_set(r), str(r["condition"]))].append(r)
    for key, recs in groups.items():
        rubrics = sorted({str(r.get("rubric")) for r in recs})
        if len(rubrics) > 1:
            raise ValueError(f"pointwise records of {key} mix the rubrics {rubrics}")
    return dict(sorted(groups.items())), n_other


def _pointwise_cells(r: dict) -> list[str]:
    return [_ci_cell(r["premium"], r["premium_ci"]), _ci_cell(r["bonus"], r["bonus_ci"]),
            _ci_cell(r["tax"], r["tax_ci"]), _num(r["standardised_gap"]),
            _ci_cell(r["share_O_above_H"], r["share_O_above_H_ci"])]


def _reading_cell(r: dict) -> str:
    return (r["reading"] or NA) + (" (underpowered)" if r["underpowered"] else "")


def pointwise_metrics(records: list[dict], n_boot: int = 2000, seed: int = 0) -> tuple[list[dict], list[str]]:
    """One row per (grader, set, condition) over the replicate-0 pointwise records, sorted by that key.

    Over the tasks with a valid score (int 1 to 10) for all of O, H and P (n_tasks): premium = mean(O - H), bonus =
    mean(O - P), tax = mean(P - H), each with its CI; share_O_above_H = mean of 1 / 0.5 / 0 (O above / equal to /
    below H) with its CI; best_of_3_O = mean over tasks of O's share of the top score (1 when O alone is top, 1/k when
    tied with k - 1 others, 0 otherwise); tie_rate_OH = share of tasks with score O == score H.
    Over ALL valid scores of the group (every task and variant, including tasks without all three): pooled_sd (sample
    SD, ddof 1; NaN with fewer than two scores), standardised_gap = premium / pooled_sd (None when the SD is 0 or
    undefined), modal_share (share of the scores at the most frequent value) and histogram {1..10: count}.
    reading: "pays" when the premium's CI is above 0, "penalises" when below 0, else "indifferent" (None without a
    task); underpowered: tie_rate_OH > 0.5. Counts: n_invalid (records whose score is not valid), n_incomplete_tasks
    (tasks with records but without a valid score for all three), n_duplicates (a repeated (task, variant) keeps the
    first record). Records of replicate > 0 are left out (self_consistency reads them) and counted in the markdown.
    """
    groups, n_other = _pointwise_groups(records)
    rows = [{"grader": g, "set": s, "condition": c, "replicate": 0, **_pointwise_row(recs, n_boot, seed)}
            for (g, s, c), recs in groups.items()]
    lines = [
        "Pointwise reading (replicate 0): each grader scored O, H and P alone from 1 to 10. Over the tasks with a "
        "valid score for all three: premium = mean(O - H), bonus = mean(O - P), tax = mean(P - H), in score points, "
        "with 95% CIs (percentile bootstrap over tasks, "
        f"{n_boot} resamples, seed {seed}); standardised gap = premium / sample SD (ddof 1) of all valid scores of "
        "the row; O above H = share of tasks with score O > H (ties 0.5); best of 3 (O) = mean share of the top "
        "score held by O (ties split); ties O = H = share of tasks with equal O and H scores; modal share = share of "
        "all valid scores at the most frequent value. Reading: pays when the premium's CI is above 0, penalises when "
        f"below 0, otherwise indifferent; underpowered when O and H tie on more than {UNDERPOWERED_TIE_RATE:g} of "
        "the tasks.",
        "",
    ]
    if not rows:
        return rows, lines + ["No pointwise records of replicate 0."]
    header = ["grader", "set", "condition", "premium O - H", "bonus O - P", "tax P - H", "standardised gap",
              "O above H", "best of 3 (O)", "ties O = H", "modal share", "reading", "n tasks", "invalid"]
    table = [[r["grader"], r["set"], r["condition"], *_pointwise_cells(r), _num(r["best_of_3_O"]),
              analysis._rate_cell(r["tie_rate_OH"]), analysis._rate_cell(r["modal_share"]), _reading_cell(r),
              str(r["n_tasks"]), str(r["n_invalid"])] for r in rows]
    lines += analysis._md_table(header, table)
    lines += ["", "Score histograms (all valid scores of the row):", "",
              *analysis._md_table(["grader", "set", "condition", *map(str, range(1, 11))],
                                  [[r["grader"], r["set"], r["condition"],
                                    *[str(r["histogram"][s]) for s in range(1, 11)]] for r in rows])]
    if n_other:
        lines += ["", f"{n_other} records of replicate > 0 are not in these tables (see self-consistency)."]
    return rows, lines


def sign_agreement(pairwise_rows: list[dict], pointwise_rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Does the pairwise O vs H reading point the same way as the pointwise premium? One row per (grader, set,
    condition) present in both, sorted by that key.

    Judge rows are judge_table rows (pairwise rate = pairs["O_vs_H"]["rate"] with its CI), joined on (grader, set,
    condition) with a pointwise_metrics row (premium and premium_ci); two judge rows with the same key (two rubrics)
    raise ValueError, so pass one rubric. A reward-model row (kind "reward_model", as analysis.rm_rows_pilot2 builds
    it) needs no pointwise row: its pairwise rate is P(score O > score H) and its premium the standardised difference
    of analysis.rm_pair_metrics (mean_std_diff, ci_std_diff).
    Row: grader, set, condition, kind, rubric (None for a reward model), pairwise_rate, pairwise_ci, pairwise_sign
    ("above" / "below" / "includes 0.5" by the CI), premium, premium_ci, premium_measure, reading ("pays" /
    "penalises" / "indifferent" by the CI), agree (same direction, or both inconclusive; None when either side is
    undefined). The markdown has the table, a second table with only the disagreements, and the keys that are in only
    one of the two inputs.
    """
    pointwise = {}
    for r in pointwise_rows:
        key = (r["grader"], r["set"], r["condition"])
        if key in pointwise:
            raise ValueError(f"two pointwise rows for {key}")
        pointwise[key] = r
    rows: list[dict] = []
    seen: dict[tuple, Any] = {}
    unjoined: list[str] = []
    for r in pairwise_rows:
        key = (r["grader"], r["set"], r["condition"])
        if key in seen:
            raise ValueError(f"two pairwise rows for {key} (rubrics {seen[key]!r} and {r.get('rubric')!r}); pass one "
                             "rubric")
        seen[key] = r.get("rubric")
        pair = (r.get("pairs") or {}).get("O_vs_H")
        if pair is None:
            raise ValueError(f"pairwise row {key} has no O_vs_H pair")
        if r.get("kind") == "reward_model":
            rate, ci, rubric = pair["p_first_gt_second"], pair["ci"], None
            premium, premium_ci = pair["mean_std_diff"], pair["ci_std_diff"]
            measure = "standardised difference (reward model)"
        else:
            if key not in pointwise:
                unjoined.append(f"pairwise {'/'.join(key)}")
                continue
            rate, ci, rubric = pair["rate"], pair["ci"], r.get("rubric")
            premium, premium_ci = pointwise[key]["premium"], pointwise[key]["premium_ci"]
            measure = "mean O - H (score points)"
        sign, reading = _pairwise_sign(ci), _reading(premium_ci)
        rows.append({"grader": key[0], "set": key[1], "condition": key[2], "kind": r.get("kind", "judge"),
                     "rubric": rubric, "pairwise_rate": rate, "pairwise_ci": ci, "pairwise_sign": sign,
                     "premium": premium, "premium_ci": premium_ci, "premium_measure": measure, "reading": reading,
                     "agree": None if sign is None or reading is None else (sign, reading) in _AGREE})
    unjoined += [f"pointwise {'/'.join(k)}" for k in pointwise if k not in seen]
    rows.sort(key=lambda x: (x["grader"], x["set"], x["condition"]))
    header = ["grader", "set", "condition", "kind", "pairwise O vs H", "sign", "premium", "reading", "agree"]

    def table(rs: list[dict]) -> list[str]:
        return analysis._md_table(header, [
            [r["grader"], r["set"], r["condition"], r["kind"], _ci_cell(r["pairwise_rate"], r["pairwise_ci"]),
             r["pairwise_sign"] or NA, _ci_cell(r["premium"], r["premium_ci"]), r["reading"] or NA,
             NA if r["agree"] is None else ("yes" if r["agree"] else "no")] for r in rs])

    disagree = [r for r in rows if r["agree"] is False]
    lines = [
        "Pairwise O vs H rate with its CI (sign: above / below / includes 0.5) against the pointwise premium with its "
        "CI (reading: pays / penalises / indifferent). Judges: premium = mean O - H score of the pointwise run; "
        "reward models: the standardised difference of their scores. Agree = same direction, or both inconclusive.",
        "",
        *(table(rows) if rows else ["No (grader, set, condition) is in both inputs."]),
        "",
        "Disagreements:",
        "",
        *(table(disagree) if disagree else ["None."]),
    ]
    if unjoined:
        lines += ["", f"In only one input (not compared): {', '.join(sorted(unjoined))}."]
    return rows, lines


def self_consistency(records: list[dict]) -> dict:
    """Replicate 0 against replicate 1 of the same grader, set and condition, on the same (task, variant).

    Per (grader, set, condition) that has replicate-1 records, sorted: n_pairs ((task, variant) with a valid score in
    both replicates), n_invalid_pairs (present in both, invalid in at least one), exact_agreement (share of pairs with
    equal scores), mean_abs_diff, n_tasks_sign (tasks with valid O and H in both replicates) and sign_agreement (share
    of those tasks whose sign of O - H, in {-1, 0, +1}, is the same in both replicates); rates None without data. A
    repeated (task, variant) within a replicate keeps the first record. Records of other replicates are not read.
    Returns {"rows": [...], "markdown": [lines]}.
    """
    groups: dict[tuple[str, str, str], dict[int, dict[tuple, Any]]] = defaultdict(lambda: {0: {}, 1: {}})
    for r in records:
        rep = r.get("replicate", 0)
        if rep in (0, 1):
            groups[(str(r["grader"]), analysis.record_set(r), str(r["condition"]))][rep].setdefault(
                (r["task_id"], r["variant"]), r.get("score"))
    rows = []
    for (grader, set_, condition), reps in sorted(groups.items()):
        if not reps[1]:
            continue
        common = sorted(set(reps[0]) & set(reps[1]))
        pairs = [(reps[0][k], reps[1][k]) for k in common if _valid_score(reps[0][k]) and _valid_score(reps[1][k])]
        signs = []
        for task_id in sorted({t for t, _ in common}):
            s = [reps[rep].get((task_id, v)) for rep in (0, 1) for v in ("O", "H")]
            if all(map(_valid_score, s)):
                signs.append(_sign(s[0] - s[1]) == _sign(s[2] - s[3]))
        rows.append({"grader": grader, "set": set_, "condition": condition, "n_pairs": len(pairs),
                     "n_invalid_pairs": len(common) - len(pairs),
                     "exact_agreement": analysis._rate(sum(a == b for a, b in pairs), len(pairs)),
                     "mean_abs_diff": sum(abs(a - b) for a, b in pairs) / len(pairs) if pairs else None,
                     "n_tasks_sign": len(signs), "sign_agreement": analysis._rate(sum(signs), len(signs))})
    lines = ["Self-consistency: replicate 0 against replicate 1 on the same (task, variant). Exact agreement = share "
             "of pairs with the same score; O - H sign agreement = share of tasks whose O - H sign is the same in both "
             "replicates.", ""]
    if not rows:
        lines.append("No grader has replicate-1 records.")
    else:
        header = ["grader", "set", "condition", "pairs", "exact agreement", "mean |difference|", "tasks",
                  "O - H sign agrees"]
        lines += analysis._md_table(
            header,
            [[r["grader"], r["set"], r["condition"], str(r["n_pairs"]), analysis._rate_cell(r["exact_agreement"]),
              analysis._rate_cell(r["mean_abs_diff"]), str(r["n_tasks_sign"]), analysis._rate_cell(r["sign_agreement"])]
             for r in rows])
    return {"rows": rows, "markdown": lines}


def gap_split(records: list[dict], n_boot: int = 2000, seed: int = 0) -> tuple[list[dict], list[str]]:
    """pointwise_metrics per gap type: for each replicate-0 (grader, set, condition), one row per gap type with the
    same metrics computed on that gap type's records alone (so the pooled SD of the standardised gap is the stratum's).

    Rows: no_tests_run, tests_failed, tests_narrowed always (n_tasks 0 when empty), then any other gap_type present,
    with records lacking a gap_type in the row 'unknown' (counted, never dropped). Row: grader, set, condition,
    gap_type, then _pointwise_row's keys (n_tasks is the stratum's n).
    """
    groups, n_other = _pointwise_groups(records)
    rows = []
    for (grader, set_, condition), recs in groups.items():
        by_gap: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            by_gap[str(r.get("gap_type") or analysis.UNKNOWN_GAP)].append(r)
        for gap in [*analysis.GAP_TYPES, *sorted(set(by_gap) - set(analysis.GAP_TYPES))]:
            rows.append({"grader": grader, "set": set_, "condition": condition, "replicate": 0, "gap_type": gap,
                         **_pointwise_row(by_gap.get(gap, []), n_boot, seed)})
    lines = ["The pointwise metrics by gap type (replicate 0); each row is computed on that gap type's tasks alone, "
             "including the pooled SD of the standardised gap; n tasks is the stratum's number of complete tasks.", ""]
    if not rows:
        return rows, lines + ["No pointwise records of replicate 0."]
    header = ["grader", "set", "condition", "gap type", "premium O - H", "bonus O - P", "tax P - H",
              "standardised gap", "O above H", "reading", "n tasks"]
    lines += analysis._md_table(header, [[r["grader"], r["set"], r["condition"], r["gap_type"], *_pointwise_cells(r),
                                          _reading_cell(r), str(r["n_tasks"])] for r in rows])
    if n_other:
        lines += ["", f"{n_other} records of replicate > 0 are not in this table."]
    return rows, lines

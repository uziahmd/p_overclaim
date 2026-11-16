"""Session 5, Step 3: the Experiment 2 mixed-pool pilot (stage e2 of overclaim.experiments.prep5; contract C).

One policy, two dispositions. For each of e2_mixed.n_tasks main coding sets (data/processed/coding/
matched_sets_v4_main.jsonl, written by prep5's assemble stage; the stage refuses to run without it) the session-4
packet policy writes n_per_instruction final messages under instruction A (session 4's PACKET_INSTRUCTION, so A has
session 4's prompts and cache entries) and as many under B (A, a space and e2_selection.B_SUFFIX): samples 0 to n - 1
are A and n to 2n - 1 are B, pooled per task. Both session-4 labellers label every sample (label_samples, the
session-3 labeller that reads the test record, on which the gate is read; label_samples_full, the full-evidence
labeller) and both pointwise selectors score it in the hidden condition (score_samples_pointwise). Neither sees which
instruction wrote a sample: the instruction is a field of the sample record and never enters a prompt. Best of N
(e2_selection.NS) is read for three outcomes (clean_overclaim, the pre-registered one; overclaim; claims_tests_passed)
per labeller and selector, the tie for the top score averaged over n_tie_seeds tie-break seeds
(e2_selection.best_of_n_averaged), with the task bootstrap; its blocks of N are taken over the two instructions
interleaved (block_positions: every block of 4 holds 2 A and 2 B samples), the sample indices staying as they are.

Gate (results/prep5/e2_gate.json): the pooled clean-overclaim rate at N = 1 on the session-3 labeller (the mean over
tasks of the share of a task's validly labelled samples that are clean overclaims) must lie within e2_mixed.bounds.
When round 1 fails and instruction B's own rate is at least e2_mixed.strengthen_at, round 2 runs once on the pool
A + B2 (B2: B, a space and e2_selection.B2_SUFFIX) in its own files (suffix _b2: the whole pool, whose A samples are
round 1's, served by the disk cache and checked to be the same texts before any label or score call), and the gate is
read on it; otherwise no second round runs (brief Step 3). When round 2 is due but the stop-launching line blocks it,
the gate is recorded as undecided. The full-evidence labeller gives a second reading of every round.

Behind the Anthropic stop-launching gate (pilot2.gate) at the start and again before round 2. Idempotent: the samples,
labels and scores in the run files are reused (and every call is disk-cached); the table and the gate file are
rewritten from them. Agent messages are data: they are placed into prompts as text and never executed.
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import Any

from overclaim import analysis
from overclaim.experiments import e2_selection, pilot2, pilot3, prep5
from overclaim.experiments.pilot2 import EXIT_FAILED, EXIT_OK, _read_jsonl, _write_json
from overclaim.experiments.pilot3 import _record_stage, _skip, seeded_order

logger = logging.getLogger(__name__)

STAGE = "e2"
GAP_ORDER = pilot3.GAP_TYPES    # no_tests_run, tests_failed, tests_narrowed: the turn order of the task pick
OUTCOMES = ("clean_overclaim", "overclaim", "claims_tests_passed")   # the first is the pre-registered outcome
RATE_FIELDS = (*OUTCOMES, "discloses_gap")
LABELLERS = ("session3", "full_evidence")   # label_samples (the gate is read on it), label_samples_full
LABELLER_NAMES = {"session3": "session-3 labeller (test record)", "full_evidence": "full-evidence labeller"}
ROUND_POOLS = {1: ("B", ""), 2: ("B2", "_b2")}   # round -> (the instruction pooled with A, file suffix)
TABLE_JSON, TABLE_MD, GATE_FILE = "e2_mixed_table.json", "e2_mixed_table.md", "e2_gate.json"


# ---------------------------------------------------------------------------------------------
# Config, task pick and task dicts
# ---------------------------------------------------------------------------------------------


def e2_config(cfg: dict) -> dict:
    """cfg["e2_mixed"] after these checks (ValueError listing every failed one): ns equals e2_selection.NS; n_tasks,
    n_per_instruction, n_tie_seeds and n_bootstrap are positive integers; the pool (2 * n_per_instruction samples) is
    the largest N (best of 16 from the pooled 16 samples); policy and labeller name model entries that
    prep5.load_config loaded into cfg (listed in model_entries); selectors is a non-empty list of cfg["judges"] names;
    bounds is [low, high] with 0 <= low < high <= 1; 0 <= strengthen_at <= 1."""
    e2 = cfg.get("e2_mixed")
    if not isinstance(e2, dict):
        raise ValueError("the config has no e2_mixed block")
    bad = []
    if tuple(e2.get("ns") or ()) != tuple(e2_selection.NS):
        bad.append(f"ns {e2.get('ns')} differs from the N values e2_selection computes {list(e2_selection.NS)}")
    for key in ("n_tasks", "n_per_instruction", "n_tie_seeds", "n_bootstrap"):
        value = e2.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            bad.append(f"{key} must be a positive integer")
    n = e2.get("n_per_instruction")
    if isinstance(n, int) and 2 * n != e2_selection.NS[-1]:
        bad.append(f"the pool of 2 x {n} samples must be the largest N, {e2_selection.NS[-1]}")
    for key in ("policy", "labeller"):
        if not isinstance(cfg.get(e2.get(key)), dict) or "model" not in cfg[e2[key]]:
            bad.append(f"{key} {e2.get(key)!r} is not a loaded model entry (list it in model_entries)")
    selectors = e2.get("selectors")
    if not isinstance(selectors, list) or not selectors or any(s not in cfg.get("judges", {}) for s in selectors):
        bad.append(f"selectors {selectors} must be a non-empty list of judges")
    bounds = e2.get("bounds")
    if not (isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(b, (int, float)) for b in bounds)
            and 0 <= bounds[0] < bounds[1] <= 1):
        bad.append(f"bounds {bounds} must be [low, high] with 0 <= low < high <= 1")
    at = e2.get("strengthen_at")
    if isinstance(at, bool) or not isinstance(at, (int, float)) or not 0 <= at <= 1:
        bad.append(f"strengthen_at {at} must be a number from 0 to 1")
    if bad:
        raise ValueError("e2_mixed: " + "; ".join(bad))
    return e2


def pick_tasks(main_sets: list[dict], n_tasks: int, seed: Any) -> list[str]:
    """n_tasks task ids of the main sets, stratified by gap type: each gap type's ids in seeded_order(ids, seed), then
    one id per gap type in turn in GAP_ORDER (a gap type with no id left is passed over) until n_tasks are taken or
    none is left. With 10 tasks and at least 4, 3 and 3 sets: 4 no_tests_run, 3 tests_failed, 3 tests_narrowed.
    Returned in pick order. ValueError for a set whose gap type is not in GAP_ORDER and for a task id given twice."""
    e2_selection._check_unique([s["task_id"] for s in main_sets], "main set task_id")
    unknown = sorted({str(s["gap_type"]) for s in main_sets} - set(GAP_ORDER))
    if unknown:
        raise ValueError(f"main sets with an unknown gap type: {unknown}")
    queues = {g: seeded_order([s["task_id"] for s in main_sets if s["gap_type"] == g], seed) for g in GAP_ORDER}
    picked: list[str] = []
    while len(picked) < n_tasks and any(queues.values()):
        for g in GAP_ORDER:
            if queues[g] and len(picked) < n_tasks:
                picked.append(queues[g].pop(0))
    return picked


def main_task_dicts(ctx: prep5.Ctx5, ids: list[str]) -> list[dict]:
    """The task dicts of `ids`, in that order, from session 4's four task files (ctx.s4_task_files()) and the assemble
    stage's prep5.ADDED_TASKS_FILE (the main and reserve tasks that are in none of those four); the first record of a
    task_id wins. ValueError naming the ids found in none of them."""
    wanted = set(ids)
    found: dict[str, dict] = {}
    for path in [*ctx.s4_task_files(), ctx.processed / prep5.ADDED_TASKS_FILE]:
        for task in _read_jsonl(path):
            if task["task_id"] in wanted:
                found.setdefault(task["task_id"], task)
    missing = [i for i in ids if i not in found]
    if missing:
        raise ValueError(f"{len(missing)} picked tasks are in no session-4 task file and not in "
                         f"{prep5.ADDED_TASKS_FILE}: {missing[:3]}")
    return [found[i] for i in ids]


# ---------------------------------------------------------------------------------------------
# One pool: samples, labels and scores
# ---------------------------------------------------------------------------------------------


def run_files(run_dir: Path, suffix: str) -> dict[str, Path]:
    """The pool's four run files: e2_mixed_{samples,labels,labels_full,scores}<suffix>.jsonl under run_dir."""
    return {kind: run_dir / f"e2_mixed_{kind}{suffix}.jsonl" for kind in ("samples", "labels", "labels_full", "scores")}


def instruction_plan(second: str, n_per_instruction: int) -> list[str]:
    """The instruction of each sample index: n_per_instruction times "A", then as many times `second`."""
    return ["A"] * n_per_instruction + [second] * n_per_instruction


def run_pool(tasks: list[dict], evidence_by_task: dict, clients: dict, files: dict[str, Path], *,
             instructions: list[str], selectors: list[str], structured: dict[str, bool],
             workers: dict[str, int], a_same_as: list[dict] | None = None) -> dict:
    """One pool, every step resume-safe: e2_selection.generate_samples_mixed (the policy, workers["openrouter"]
    threads), label_samples and label_samples_full (clients["labeller"]), then score_samples_pointwise per selector
    (workers["anthropic"] threads for the labels and scores). The labellers and selectors get the sample records, whose
    prompts are built from the task and the text alone. With a_same_as (round 1's samples, for round 2),
    check_same_a(a_same_as, samples) runs after the policy and before any label or score call, so that a pool whose A
    samples are not round 1's raises ValueError before an Anthropic call. Returns {samples, labels: {session3,
    full_evidence}, scores}."""
    samples = e2_selection.generate_samples_mixed(tasks, evidence_by_task, clients["policy"], files["samples"],
                                                  instructions=instructions, max_workers=workers["openrouter"])
    if a_same_as is not None:
        check_same_a(a_same_as, samples)
    labels = e2_selection.label_samples(samples, tasks, clients["labeller"], files["labels"],
                                        max_workers=workers["anthropic"])
    labels_full = e2_selection.label_samples_full(samples, tasks, evidence_by_task, clients["labeller"],
                                                  files["labels_full"], max_workers=workers["anthropic"])
    scores: list[dict] = []
    for name in selectors:
        scores += e2_selection.score_samples_pointwise(samples, tasks, clients[name], name, files["scores"],
                                                       structured=structured[name], max_workers=workers["anthropic"])
    return {"samples": samples, "labels": {"session3": labels, "full_evidence": labels_full}, "scores": scores}


def check_same_a(first: list[dict], second: list[dict]) -> None:
    """ValueError naming the A samples whose text differs between two pools (round 2's A samples must be round 1's,
    served by the disk cache), or that are in one pool only."""
    def a_texts(samples: list[dict]) -> dict:
        return {(s["task_id"], s["sample"]): s["text"] for s in samples if s["instruction"] == "A"}

    a, b = a_texts(first), a_texts(second)
    differ = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if differ:
        raise ValueError(f"{len(differ)} A samples of round 2 are not round 1's, e.g. {differ[:3]}")


# ---------------------------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------------------------


def gate_reading(samples: list[dict], labels: list[dict], bounds: list[float]) -> dict:
    """The gate's numbers for one labeller's labels of one pool: rate (the mean over tasks of the share of the task's
    validly labelled samples that are clean overclaims, e2_selection.outcomes; a task without a valid label is left
    out; None without any), rate_by_instruction (the same over each instruction's samples, instructions in sample
    order), rate_by_task, n_tasks (with a rate), n_invalid_labels (samples whose label is invalid or missing), bounds
    and passed (low <= rate <= high; False when the rate is None)."""
    outcome_of = e2_selection.outcomes(labels)

    def shares(subset: list[dict]) -> dict[str, float]:
        by_task: dict[str, list[bool]] = {}
        for s in subset:
            o = outcome_of.get((s["task_id"], s["sample"]))
            if o is not None:
                by_task.setdefault(s["task_id"], []).append(o["clean_overclaim"])
        return {t: sum(v) / len(v) for t, v in by_task.items()}

    def mean(values: dict[str, float]) -> float | None:
        return statistics.fmean(values.values()) if values else None

    by_task = shares(samples)
    rate = mean(by_task)
    names = list(dict.fromkeys(s["instruction"] for s in samples))
    low, high = bounds
    return {"rate": rate,
            "rate_by_instruction": {n: mean(shares([s for s in samples if s["instruction"] == n])) for n in names},
            "rate_by_task": by_task, "n_tasks": len(by_task),
            "n_invalid_labels": sum(outcome_of.get((s["task_id"], s["sample"])) is None for s in samples),
            "bounds": [low, high], "passed": rate is not None and low <= rate <= high}


def round2_decision(reading: dict, strengthen_at: float) -> dict:
    """{"due", "reason"} after round 1 on the session-3 labeller: round 2 is due when round 1 failed and instruction B's
    own clean-overclaim rate is at least strengthen_at (brief Step 3); otherwise no second round runs."""
    b = reading["rate_by_instruction"].get("B")
    if reading["passed"]:
        return {"due": False, "reason": "round 1 passed"}
    if b is None:
        return {"due": False, "reason": "round 1 failed and B's clean-overclaim rate is undefined (no valid label): "
                                        "no second round"}
    if b >= strengthen_at:
        return {"due": True, "reason": f"round 1 failed and B's clean-overclaim rate {b:.4f} is at least "
                                       f"{strengthen_at}: B is strengthened once (B2)"}
    return {"due": False, "reason": f"round 1 failed and B's clean-overclaim rate {b:.4f} is below {strengthen_at}: "
                                    "no second round (brief Step 3)"}


def gate_summary(pools: list[dict], bounds: list[float], strengthen_at: float, round2: dict) -> dict:
    """e2_gate.json: rounds (gate_reading on the session-3 labeller per round run, with round and pool), round2 (due,
    ran, reason), passed and pool (the last round's), decided, the same readings on the full-evidence labeller as a
    second reading (full_evidence: rounds, passed), labellers_agree (the two last-round verdicts are the same) and
    verdict. decided is False when round 2 is due but did not run (the stop-launching line): the gate rule is then not
    applied in full, passed and pool are round 1's and the verdict says "undecided", not that the experiment is dropped.
    """
    readings = {lab: [{"round": p["round"], "pool": p["pool"], **gate_reading(p["samples"], p["labels"][lab], bounds)}
                      for p in pools] for lab in LABELLERS}
    last, last_full = readings["session3"][-1], readings["full_evidence"][-1]
    decided = not (round2["due"] and not round2["ran"])
    if not decided:
        verdict = (f"undecided: round 1 failed on {last['pool']} and round 2 (A+B2) is due but was not launched "
                   "(round2.reason); Experiment 2 is neither run nor dropped until round 2 runs")
    elif last["passed"]:
        verdict = f"passed on {last['pool']}: the full Experiment 2 runs on the mixed pool"
    else:
        verdict = (f"failed on {last['pool']}: Experiment 2 is dropped from Part 2 and H3 is reported as untestable at "
                   "this policy's base rate (brief decision 4)")
    return {"labeller": LABELLER_NAMES["session3"] + "; the gate is read on it",
            "outcome": "clean_overclaim at N = 1: the mean over tasks of the share of a task's validly labelled "
                       "samples that are clean overclaims",
            "bounds": list(bounds), "strengthen_at": strengthen_at, "rounds": readings["session3"], "round2": round2,
            "passed": last["passed"], "pool": last["pool"], "decided": decided,
            "full_evidence": {"labeller": LABELLER_NAMES["full_evidence"] + "; a second reading",
                              "rounds": readings["full_evidence"], "passed": last_full["passed"]},
            "labellers_agree": last["passed"] == last_full["passed"], "verdict": verdict}


# ---------------------------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------------------------


def outcome_labels(labels: list[dict], outcome: str) -> list[dict]:
    """Label records that best_of_n_averaged's outcome "overclaim" reads as `outcome` (one of OUTCOMES): label
    "overclaim" when the sample has the outcome, "honest" when not, "invalid" when its label is invalid
    (clean_overclaim and overclaim as e2_selection.outcomes defines them; claims_tests_passed is the label's field)."""
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r} (expected one of {OUTCOMES})")
    out = []
    for r in labels:
        o = e2_selection._outcome(r)
        has = None if o is None else (r["claims_tests_passed"] is True if outcome == "claims_tests_passed"
                                      else o[outcome])
        out.append({"task_id": r["task_id"], "sample": r["sample"], "discloses_gap": False,
                    "label": e2_selection.INVALID if has is None else ("overclaim" if has else "honest")})
    return out


def block_positions(samples: list[dict], scores: list[dict], labels: list[dict]) -> tuple[list[dict], list[dict],
                                                                                          list[dict]]:
    """Copies of samples, scores and labels whose "sample" is the sample's block position, so that best_of_n_detail's
    consecutive blocks of N mix the instructions. Within each task the instructions are taken in order of first
    appearance (by sample index) and interleaved: the k-th sample of the j-th instruction goes to position
    k * (number of instructions) + j. With samples 0-7 A and 8-15 B the order is A0, B8, A1, B9, ..., A7, B15, so every
    block of 4 holds 2 A and 2 B samples and the block of 16 is the whole pool. Only these copies change; the records,
    their indices and their cache tags do not. ValueError when a task's instructions have different numbers of samples
    or a score or label record names a (task_id, sample) that is not in samples."""
    by_task: dict[str, dict[str, list[int]]] = {}
    for s in sorted(samples, key=lambda s: (s["task_id"], s["sample"])):
        by_task.setdefault(s["task_id"], {}).setdefault(s["instruction"], []).append(s["sample"])
    position: dict[tuple[str, int], int] = {}
    for task_id, groups in by_task.items():
        sizes = {name: len(idx) for name, idx in groups.items()}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"task {task_id}: the instructions have different numbers of samples {sizes}")
        for j, idx in enumerate(groups.values()):
            for k, i in enumerate(idx):
                position[(task_id, i)] = k * len(groups) + j

    def moved(records: list[dict], what: str) -> list[dict]:
        unknown = sorted({(r["task_id"], r["sample"]) for r in records} - set(position))
        if unknown:
            raise ValueError(f"{len(unknown)} {what} records name no sample of the pool, e.g. {unknown[:3]}")
        return [{**r, "sample": position[(r["task_id"], r["sample"])]} for r in records]

    return moved(samples, "sample"), moved(scores, "score"), moved(labels, "label")


def selection_rows(samples: list[dict], scores: list[dict], labels: list[dict], *, n_tie_seeds: int, n_boot: int,
                   seed: int) -> list[dict]:
    """One row per outcome (OUTCOMES order) and selector: e2_selection.selection_table_v4 on outcome_labels(labels,
    outcome), its "overclaim" row with outcome set to the outcome (selector, outcome, tie_averaged and seed0 {n_tasks,
    rates, diff, monotone}, counts). Blocks are best_of_n_detail's consecutive blocks of N taken over the block
    positions of block_positions (the instructions interleaved), not over the sample indices."""
    samples, scores, labels = block_positions(samples, scores, labels)
    rows = []
    for outcome in OUTCOMES:
        got, _ = e2_selection.selection_table_v4(samples, scores, outcome_labels(labels, outcome),
                                                 n_tie_seeds=n_tie_seeds, n_boot=n_boot, seed=seed)
        rows += [{**r, "outcome": outcome} for r in got if r["outcome"] == "overclaim"]
    return rows


def _groups(samples: list[dict]) -> dict[str, list[dict]]:
    """{instruction: its samples} in sample order of first appearance, then "pooled": every sample."""
    names = list(dict.fromkeys(s["instruction"] for s in samples))
    return {**{n: [s for s in samples if s["instruction"] == n] for n in names}, "pooled": samples}


def base_rates_mixed(samples: list[dict], labels: list[dict]) -> dict:
    """Per instruction and pooled (_groups), over the samples (not per task): n_samples, n_labelled, n_invalid (label
    invalid or missing) and, per field of RATE_FIELDS, {"n", "rate"} over the validly labelled samples (clean_overclaim
    and overclaim as e2_selection.outcomes defines them, claims_tests_passed and discloses_gap the label's fields;
    rate None without a valid label)."""
    label_of = {(r["task_id"], r["sample"]): r for r in labels}
    out = {}
    for name, group in _groups(samples).items():
        valid = []
        for s in group:
            r = label_of.get((s["task_id"], s["sample"]))
            o = None if r is None else e2_selection._outcome(r)
            if o is not None:
                valid.append({**o, "claims_tests_passed": r["claims_tests_passed"] is True,
                              "discloses_gap": r["discloses_gap"] is True})
        counts = {f: sum(v[f] for v in valid) for f in RATE_FIELDS}
        out[name] = {"n_samples": len(group), "n_labelled": len(valid), "n_invalid": len(group) - len(valid),
                     **{f: {"n": counts[f], "rate": counts[f] / len(valid) if valid else None} for f in RATE_FIELDS}}
    return out


def sample_flags(samples: list[dict]) -> dict:
    """Per instruction and pooled (_groups): n_samples and the samples flagged truncated, empty and duplicate
    (duplicate_of set: the same stripped text as an earlier sample of its task in the pool's file)."""
    return {name: {"n_samples": len(group), "truncated": sum(s["truncated"] for s in group),
                   "empty": sum(s["empty"] for s in group),
                   "duplicate": sum(s["duplicate_of"] is not None for s in group)}
            for name, group in _groups(samples).items()}


def top_ties(samples: list[dict], scores: list[dict]) -> dict:
    """Per selector (order of first appearance in scores): the tie for the top score over each task's pool:
    mean_tied_at_top (the mean over tasks of the number of samples with the task's top valid score),
    mean_share_tied_at_top (of the number with that score over the number with a valid score), tasks_with_a_tie (more
    than one at the top) and n_tasks (tasks with a valid score)."""
    by_task: dict[str, list[int]] = {}
    for s in samples:
        by_task.setdefault(s["task_id"], []).append(s["sample"])
    score_of: dict[str, dict] = {}
    for r in scores:
        score_of.setdefault(r["selector"], {})[(r["task_id"], r["sample"])] = r.get("score")
    out = {}
    for selector, sc in score_of.items():
        tied, share = [], []
        for task_id, idx in by_task.items():
            valid = [sc[(task_id, i)] for i in idx if e2_selection._valid_score(sc.get((task_id, i)))]
            if valid:
                n_top = valid.count(max(valid))
                tied.append(n_top)
                share.append(n_top / len(valid))
        out[selector] = {"mean_tied_at_top": statistics.fmean(tied) if tied else None,
                         "mean_share_tied_at_top": statistics.fmean(share) if share else None,
                         "tasks_with_a_tie": sum(n > 1 for n in tied), "n_tasks": len(tied)}
    return out


def selected_split(samples: list[dict], scores: list[dict], *, n_tie_seeds: int) -> dict:
    """The instruction of the sample selected at N = the pool size (one block per task), a descriptive: per selector,
    the share of A over the tasks (e2_selection.best_of_n_averaged with outcome "the sample is A", on the block
    positions of block_positions as in the table: ties averaged over the same tie-break seeds), the other
    instruction's share (1 minus it), n_tasks and by_task (the A share per task)."""
    n_pool = {len([s for s in samples if s["task_id"] == t]) for t in {s["task_id"] for s in samples}}
    if len(n_pool) != 1:
        raise ValueError(f"the tasks' pools differ in size: {sorted(n_pool)}")
    n = n_pool.pop()
    other = next(name for name in _groups(samples) if name not in ("A", "pooled"))
    is_a = [{"task_id": s["task_id"], "sample": s["sample"], "discloses_gap": False,
             "label": "overclaim" if s["instruction"] == "A" else "honest"} for s in samples]
    samples, scores, is_a = block_positions(samples, scores, is_a)
    per_task, _ = e2_selection.best_of_n_averaged(samples, scores, is_a, ns=(n,), n_tie_seeds=n_tie_seeds)
    out = {}
    for selector, cells in per_task.items():
        by_task = {t: c[n]["averaged"] for t, c in cells.items()}
        a = statistics.fmean(by_task.values()) if by_task else None
        out[selector] = {"A": a, other: None if a is None else 1 - a, "n_tasks": len(by_task), "by_task": by_task}
    return out


def pool_table(pool: dict, *, n_tie_seeds: int, n_boot: int, seed: int) -> dict:
    """The table of one pool (a run_pool result with round and pool): task_ids, n_samples, instructions, block_note,
    selection {labeller: selection_rows}, base_rates {"samples": sample_flags, labeller: base_rates_mixed}, top_ties
    and selected_at_pool_size (selected_split)."""
    samples, scores = pool["samples"], pool["scores"]
    names = [n for n in _groups(samples) if n != "pooled"]
    n_per = sum(s["instruction"] == "A" for s in samples) // max(1, len({s["task_id"] for s in samples}))
    note = (f"Best of N cuts each task's {2 * n_per} samples into consecutive blocks of N (e2_selection."
            f"best_of_n_detail) in an interleaved order, A0, {names[-1]}{n_per}, A1, {names[-1]}{n_per + 1}, ... "
            f"(block_positions; samples 0 to {n_per - 1} are A and {n_per} to {2 * n_per - 1} are {names[-1]}), so "
            f"each N = 4 block holds 2 A and 2 {names[-1]} samples and N = {2 * n_per} selects from the whole pool. "
            "N = 1 is the mean over the task's samples.")
    return {"round": pool["round"], "pool": pool["pool"],
            "task_ids": list(dict.fromkeys(s["task_id"] for s in samples)), "n_samples": len(samples),
            "instructions": names, "block_note": note,
            "selection": {lab: selection_rows(samples, scores, pool["labels"][lab], n_tie_seeds=n_tie_seeds,
                                              n_boot=n_boot, seed=seed) for lab in LABELLERS},
            "base_rates": {"samples": sample_flags(samples),
                           **{lab: base_rates_mixed(samples, pool["labels"][lab]) for lab in LABELLERS}},
            "top_ties": top_ties(samples, scores),
            "selected_at_pool_size": selected_split(samples, scores, n_tie_seeds=n_tie_seeds)}


def _num(x: float | None, digits: int = 3) -> str:
    return analysis.NA if x is None else f"{x:.{digits}f}"


def table_markdown(tables: list[dict], gate: dict, *, n_tie_seeds: int, n_boot: int, seed: int) -> str:
    """e2_mixed_table.md: the gate per round and labeller, then per pool the selection table per labeller (selector,
    outcome, tie-averaged rate at each N with its CI, the paired difference largest minus smallest N, tasks, invalid
    scores and labels), the base rates and sample flags per instruction, the top-score ties per selector and the
    instruction of the sample selected at the pool size."""
    ns = e2_selection.NS
    lines = ["# Experiment 2 mixed-pool pilot (session 5, Step 3)", "",
             f"Gate (session-3 labeller, bounds {gate['bounds']}): **{gate['verdict']}**. Round 2: "
             f"{gate['round2']['reason']}"
             + ("" if gate["round2"]["ran"] or not gate["round2"]["due"] else " (round 2 did not run)")
             + f". The two labellers {'agree' if gate['labellers_agree'] else 'DISAGREE'} on the verdict.", ""]
    rows = []
    for lab, readings in (("session3", gate["rounds"]), ("full_evidence", gate["full_evidence"]["rounds"])):
        for r in readings:
            rows.append([str(r["round"]), r["pool"], LABELLER_NAMES[lab], _num(r["rate"]),
                         ", ".join(f"{k} {_num(v)}" for k, v in r["rate_by_instruction"].items()),
                         str(r["n_invalid_labels"]), "yes" if r["passed"] else "no"])
    lines += analysis._md_table(["round", "pool", "labeller", "clean-overclaim rate at N = 1",
                                 "per instruction", "invalid labels", "within bounds"], rows)
    for t in tables:
        lines += ["", f"## Pool {t['pool']} (round {t['round']})", "",
                  f"{len(t['task_ids'])} tasks, {t['n_samples']} samples. {t['block_note']} Each cell: the share of "
                  f"the selected samples with the outcome, a tie for the top score averaged over tie-break seeds 0 to "
                  f"{n_tie_seeds - 1}; 95% percentile bootstrap over tasks ({n_boot} resamples, seed {seed}). The "
                  "seed-0 values are in the JSON file."]
        for lab in LABELLERS:
            rows = []
            for r in t["selection"][lab]:
                avg = r["tie_averaged"]
                rows.append([r["selector"], r["outcome"], *(e2_selection._cell(avg["rates"][n]) for n in ns),
                             e2_selection._cell(avg["diff"]), str(avg["n_tasks"]), str(r["counts"]["invalid_scores"]),
                             str(r["counts"]["invalid_labels"])])
            header = ["selector", "outcome", *(f"N={n}" for n in ns), f"N={ns[-1]} minus N={ns[0]}", "tasks",
                      "invalid scores", "invalid labels"]
            lines += ["", f"### Best of N, {LABELLER_NAMES[lab]}", "", *analysis._md_table(header, rows)]
        rows = []
        for lab in LABELLERS:
            for name, b in t["base_rates"][lab].items():
                rows.append([LABELLER_NAMES[lab], name, f"{b['n_labelled']} of {b['n_samples']}",
                             *(f"{b[f]['n']} ({_num(b[f]['rate'])})" for f in RATE_FIELDS)])
        lines += ["", "### Base rates per instruction (over the samples)", "",
                  *analysis._md_table(["labeller", "instruction", "labelled", *RATE_FIELDS], rows), ""]
        rows = [[name, str(f["n_samples"]), str(f["truncated"]), str(f["empty"]), str(f["duplicate"])]
                for name, f in t["base_rates"]["samples"].items()]
        lines += analysis._md_table(["instruction", "samples", "truncated", "empty", "duplicate"], rows)
        rows = [[sel, _num(v["mean_tied_at_top"], 2), _num(v["mean_share_tied_at_top"]),
                 f"{v['tasks_with_a_tie']} of {v['n_tasks']}"] for sel, v in t["top_ties"].items()]
        lines += ["", "### Ties for the top score per selector", "",
                  *analysis._md_table(["selector", "mean samples tied at the top", "mean share tied at the top",
                                       "tasks with a tie"], rows)]
        names = t["instructions"]
        rows = [[sel, *(_num(v[n]) for n in names), str(v["n_tasks"])]
                for sel, v in t["selected_at_pool_size"].items()]
        lines += ["", f"### Instruction of the sample selected at N = {t['n_samples'] // max(1, len(t['task_ids']))} "
                      "(descriptive; ties averaged as above)", "",
                  *analysis._md_table(["selector", *(f"share {n}" for n in names), "tasks"], rows)]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------------------------


def run_rounds(ctx: prep5.Ctx5, e2: dict, tasks: list[dict], evidence_by_task: dict, clients: dict) -> tuple[
        list[dict], dict]:
    """Round 1 (pool A + B, run_files suffix ""), the round-2 decision on the session-3 labeller (round2_decision)
    and, when due and pilot2.gate allows a launch, round 2 (pool A + B2, suffix "_b2"; its A samples are checked
    against round 1's by run_pool's a_same_as before any label or score call). Returns (the pools run, each a run_pool
    result with round and pool; round2 {due, ran, reason})."""
    cfg = ctx.cfg
    common = {"selectors": list(e2["selectors"]),
              "structured": {n: bool(cfg["judges"][n].get("structured_output")) for n in e2["selectors"]},
              "workers": {k: int(cfg["max_workers"][k]) for k in ("openrouter", "anthropic")}}

    def one_round(rnd: int, a_same_as: list[dict] | None = None) -> dict:
        second, suffix = ROUND_POOLS[rnd]
        logger.info("e2: round %d, pool A+%s", rnd, second)
        result = run_pool(tasks, evidence_by_task, clients, run_files(ctx.run_dir, suffix),
                          instructions=instruction_plan(second, int(e2["n_per_instruction"])), a_same_as=a_same_as,
                          **common)
        return {"round": rnd, "pool": f"A+{second}", **result}

    pools = [one_round(1)]
    decision = round2_decision(gate_reading(pools[0]["samples"], pools[0]["labels"]["session3"], e2["bounds"]),
                               float(e2["strengthen_at"]))
    round2 = {"due": decision["due"], "ran": False, "reason": decision["reason"]}
    if decision["due"]:
        stop = pilot2.gate(ctx, {}, "anthropic")
        if stop is not None:
            logger.warning("e2: round 2 is due but not launched: %s", stop)
            round2["reason"] += f"; not launched: {stop}"
        else:
            pools.append(one_round(2, pools[0]["samples"]))
            round2["ran"] = True
    return pools, round2


def stage_e2(ctx: prep5.Ctx5) -> int:
    """Step 3 on the main sets: pick_tasks (cfg["seed"]), each task's evidence the set's evidence_v4, the task dicts
    from main_task_dicts; clients policy cfg[e2_mixed.policy], labeller cfg[e2_mixed.labeller] (both labellers), one
    per selector (cfg["judges"]); run_rounds. Writes run_dir/e2_mixed_table.json and .md, run_dir/e2_gate.json and
    stages_status.json["e2"]. EXIT_FAILED (nothing recorded) when the e2_mixed block fails e2_config or the main sets
    file is missing. Behind the stop-launching gate (a skip is recorded); a missing credential skips the stage; a
    BudgetExceeded stops it (pilot2._budget_stop); any other exception fails it (prep5._stage_failed). Dry run: the plan
    only, no call."""
    cfg = ctx.cfg
    try:
        e2 = e2_config(cfg)
    except ValueError as exc:
        logger.error("e2: %s", exc)
        return EXIT_FAILED
    main_path = ctx.processed / prep5.MAIN_FILE
    if ctx.dry_run:
        where = (f"the tasks {pick_tasks(_read_jsonl(main_path), int(e2['n_tasks']), cfg['seed'])}"
                 if main_path.exists() else f"tasks of {main_path}, which is MISSING (the stage refuses to run "
                                            "without it; run assemble first)")
        ctx.note(f"e2: would draw {e2['n_per_instruction']} samples under A and {e2['n_per_instruction']} under B per "
                 f"task from {cfg[e2['policy']]['model']} (evidence packet) for {e2['n_tasks']} main sets stratified "
                 f"by gap type ({where}), label them with {cfg[e2['labeller']]['model']} (session-3 and full-evidence "
                 f"prompts), score them with {e2['selectors']} (pointwise, hidden), best of N {list(e2_selection.NS)} "
                 f"with {e2['n_tie_seeds']} tie-break seeds; gate {e2['bounds']} on the pooled clean-overclaim rate; "
                 f"round 2 (A + B2) only if round 1 fails and B is at or above {e2['strengthen_at']}")
        return EXIT_OK
    if not main_path.exists():
        logger.error("e2: %s is missing; run the assemble stage first", main_path)
        return EXIT_FAILED
    reason = pilot2.gate(ctx, {}, "anthropic")
    if reason is not None:
        return _skip(ctx, STAGE, reason)
    clients = {"policy": ctx.client(cfg[e2["policy"]]), "labeller": ctx.client(cfg[e2["labeller"]]),
               **{name: ctx.client(cfg["judges"][name]) for name in e2["selectors"]}}
    missing = [k for k, c in clients.items() if c is None]
    if missing:
        return _skip(ctx, STAGE, f"no credential for the clients {missing}")
    from overclaim.graders.judge import BudgetExceeded

    main_sets = _read_jsonl(main_path)
    set_of = {s["task_id"]: s for s in main_sets}
    try:
        ids = pick_tasks(main_sets, int(e2["n_tasks"]), cfg["seed"])
        no_evidence = [tid for tid in ids if "evidence_v4" not in set_of[tid]]
        if no_evidence:
            raise ValueError(f"main sets without evidence_v4: {no_evidence}")
        tasks = main_task_dicts(ctx, ids)
        pools, round2 = run_rounds(ctx, e2, tasks, {tid: set_of[tid]["evidence_v4"] for tid in ids}, clients)
    except BudgetExceeded as exc:
        _record_stage(ctx, STAGE, {"status": "stopped", "reason": f"budget cap: {exc}"})
        return pilot2._budget_stop(ctx, STAGE, exc)
    except Exception as exc:  # a stale run file, a changed A sample, a missing task: recorded, nothing written
        return prep5._stage_failed(ctx, STAGE, exc)
    n_tie, n_boot, seed = int(e2["n_tie_seeds"]), int(e2["n_bootstrap"]), int(cfg["seed"])
    gate = gate_summary(pools, e2["bounds"], float(e2["strengthen_at"]), round2)
    tables = [pool_table(p, n_tie_seeds=n_tie, n_boot=n_boot, seed=seed) for p in pools]
    _write_json(ctx.run_dir / GATE_FILE, gate)
    _write_json(ctx.run_dir / TABLE_JSON, analysis._nan_to_none({"gate_pool": gate["pool"], "pools": tables}))
    (ctx.run_dir / TABLE_MD).write_text(table_markdown(tables, gate, n_tie_seeds=n_tie, n_boot=n_boot, seed=seed),
                                        encoding="utf-8")
    _record_stage(ctx, STAGE, {"status": "done", "n_tasks": len(ids), "task_ids": ids,
                               "rounds": [p["round"] for p in pools], "pool": gate["pool"], "passed": gate["passed"],
                               "decided": gate["decided"], "labellers_agree": gate["labellers_agree"]})
    logger.info("e2: gate %s (session-3 labeller rate %s)", gate["verdict"], gate["rounds"][-1]["rate"])
    print(f"[e2] {ctx.cost.totals_line()}", flush=True)
    return EXIT_OK

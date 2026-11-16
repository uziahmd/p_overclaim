"""Shared readers and helpers of the reanalysis (2026-11-11).

Nothing here writes outside reanalysis/tables. Nothing here calls a model. The score records are read with the
project's own loaders (overclaim.analysis_full_run), so that a score is valid, and a task is counted, by the rules
that produced the paper's tables:

- a judge's score is valid when it is an integer from 1 to 10; a reward model's when it is a finite number;
- only the scoring prompt of the paper's main rows is read (rubric "pointwise-neutral"); Sonnet 5.5 has a second run
  without the transcript under another prompt ("pointwise-full"), which is never mixed in;
- a premium is computed on the tasks of a grader that have a valid score for the overclaim, the honest message and
  the placebo (the paper's rule);
- every interval is the paper's: a 95% percentile bootstrap over tasks, 2,000 resamples, seed 0, the task values in
  task-id order (overclaim.analysis.bootstrap_ci);
- a difference is called detectable when its interval excludes zero. No correction is applied: these analyses were
  suggested after the results were known.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paper_data as D  # noqa: E402  (read-only helpers of the paper scripts: names, orders, paths)
from overclaim import analysis  # noqa: E402
from overclaim import analysis_full_run as A  # noqa: E402

RUN = REPO / "results" / "full_run"
TABLES = REPO / "reanalysis" / "tables"
N_BOOT, SEED = 2000, 0
JUDGES, REWARD_MODELS = D.LLM_JUDGES, D.REWARD_MODELS
GRADERS = (*JUDGES, *REWARD_MODELS)
NEUTRAL, RM_RUBRIC = A.NEUTRAL, A.RM_RUBRIC
GAP_TYPES = ("no_tests_run", "tests_failed", "tests_narrowed")
FAILURE_TYPES = ("tests_failed", "tests_narrowed")      # the honest message reports a failed test run
VERSIONS = ("O", "H", "P")
# The selection experiment: samples 0 to 7 were written under the plain instruction ("A"), samples 8 to 15 under the
# added instruction to state exactly what was and was not verified ("B", "the honesty instruction").
PLAIN, HONESTY = "A", "B"
POOL_SELECTORS = ("claude-sonnet-5-5", "claude-sonnet-4-6", "deepseek-v4-pro", *REWARD_MODELS)
STEP_SELECTORS = ("claude-sonnet-5-5+verify_first", "deepseek-v4-pro+verify_first")   # with the transcript, 26 tasks


def name(grader: str) -> str:
    return D.NAMES.get(grader, grader)


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


_CACHE: dict[str, Any] = {}


def load() -> dict:
    """The main run's records as the analysis stage reads them: {"cfg", "frozen", "loaded" (records restricted to
    their design), "pw" / "rm" / "e3" / "e7" (score groups keyed (grader, set, rubric, condition)), "groups" (pw and
    rm), "sets" ({task_id: matched set})}. Read once per process."""
    if "main" not in _CACHE:
        cfg = D.load_resolved_config()
        frozen = A.load_frozen(cfg)
        loaded, _ = A.restrict_records(A.load_inputs(RUN), frozen)
        tie = A.tie_rule_conditions(loaded[A.TIE_RULE_FILE])
        pw = A.score_groups(loaded[A.POINTWISE_FILE] or [], tie)
        rm = A.score_groups(loaded[A.RM_FILE] or [], reward_model=True)
        e3 = A.score_groups(loaded[A.E3PW_FILE] or [], tie)
        e7 = A.score_groups(loaded[A.E7_FILE] or [], tie)
        _CACHE["main"] = {"cfg": cfg, "frozen": frozen, "loaded": loaded, "pw": pw, "rm": rm, "e3": e3, "e7": e7,
                          "groups": {**pw, **rm}, "sets": frozen["sets_by_task"]}
    return _CACHE["main"]


def group(groups: dict, grader: str, condition: str = "hidden", rubric: str | None = None) -> dict | None:
    """The one score group of a grader in a condition: rubric "pointwise-neutral" for a judge, the reward-model label
    for a reward model, unless `rubric` is given. None when there is none; ValueError when several sets match."""
    want = rubric or (RM_RUBRIC if grader in REWARD_MODELS else NEUTRAL)
    found = [g for k, g in groups.items() if k[0] == grader and k[2] == want and k[3] == condition]
    if len(found) > 1:
        raise ValueError(f"several groups for {grader} {want} {condition}")
    return found[0] if found else None


def complete(scores: dict, tasks: set[str] | list[str], versions: tuple[str, ...] = VERSIONS) -> list[str]:
    """The tasks of `tasks` with a score for every version of `versions`, in task-id order."""
    return sorted(t for t in tasks if t in scores and all(v in scores[t] for v in versions))


def boot(values: list[float]) -> tuple[float, float, float]:
    """(mean, lower, upper) of per-task values given in task-id order: the paper's task bootstrap."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    lo, hi = analysis.bootstrap_ci(list(values), N_BOOT, SEED)
    return float(np.mean(values)), lo, hi


def boot_tasks(per_task: dict[str, float]) -> tuple[float, float, float, int]:
    """boot() on {task_id: value}, the values in task-id order; also the number of tasks."""
    values = [per_task[t] for t in sorted(per_task)]
    m, lo, hi = boot(values)
    return m, lo, hi, len(values)


def boot_group_difference(per_task: dict[str, float], in_first: dict[str, bool]) -> tuple[float, float, float]:
    """The mean of the tasks of the first group minus the mean of the tasks of the second, with an interval from the
    same task bootstrap: the tasks (both groups together, task-id order) are resampled with the paper's draw
    (default_rng(0).integers(0, n, (2000, n))), and the difference of the two group means is recomputed in each
    resample. A resample in which one group is empty is left out (reported by the caller if it ever happens)."""
    tasks = sorted(per_task)
    v = np.array([per_task[t] for t in tasks], dtype=float)
    a = np.array([bool(in_first[t]) for t in tasks])
    if a.sum() == 0 or (~a).sum() == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(v[a].mean() - v[~a].mean())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(tasks), size=(N_BOOT, len(tasks)))
    va, aa = v[idx], a[idx]
    na, nb = aa.sum(axis=1), (~aa).sum(axis=1)
    ok = (na > 0) & (nb > 0)
    diffs = (va * aa).sum(axis=1)[ok] / na[ok] - (va * ~aa).sum(axis=1)[ok] / nb[ok]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def detectable(lo: float, hi: float) -> bool:
    """A difference is detectable when its 95% interval excludes zero."""
    return bool(not (math.isnan(lo) or math.isnan(hi)) and (lo > 0 or hi < 0))


def r(x: Any, digits: int = 4) -> Any:
    """A number rounded for a CSV cell ('' for None or NaN); other values unchanged."""
    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (float, np.floating)):
        return "" if math.isnan(float(x)) else round(float(x), digits)
    return x


def write_csv(stem: str, header: list[str], rows: list[list[Any]]) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{stem}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            w.writerow([r(c) for c in row])
    return path


def read_csv(stem: str) -> list[dict]:
    with (TABLES / f"{stem}.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def anchor(what: str, got: float, want: float, tol: float = 1e-9) -> None:
    """Stop when a number that the paper already prints does not reproduce from the records."""
    if not close(got, want, tol):
        raise SystemExit(f"anchor failed: {what}: recomputed {got!r}, the paper's records give {want!r}")


# ---------------------------------------------------------------- the selection experiment (the 16 messages per task)


def pool() -> dict:
    """The selection experiment's records: {"tasks" (the 40, in order), "instruction" {(task, sample): "A" | "B"},
    "labels" {labeller: {(task, sample): {"overclaim": bool, "clean": bool} | None}}, "scores" {selector: {(task,
    sample): number}} (valid scores only), "raw_scores" (the score records)}.

    Labels: labeller "main" is the paper's (results/full_run/e2_labels.jsonl, compared with the test runs) and
    "second" the second labelling (e2_labels_full.jsonl). "overclaim" = the label is "overclaim"; "clean" = the label
    is "overclaim" and the message states no missing check (discloses_gap is False). "clean" is the paper's outcome:
    the 68% of Section 4.3. A record whose label is neither "overclaim" nor "honest" has no entry.
    Scores: a judge's score is valid when it is an integer from 1 to 10 (bool excluded); a reward model's when it is a
    finite number. A (task, sample) scored twice by a selector keeps its last record, as the project's code does."""
    if "pool" in _CACHE:
        return _CACHE["pool"]
    data = load()
    e2_tasks = list(data["frozen"]["e2"])
    samples = [s for s in read_jsonl(RUN / "e2_samples.jsonl") if s["task_id"] in set(e2_tasks)]
    instruction = {(s["task_id"], s["sample"]): s["instruction"] for s in samples}
    labels = {}
    for lab, fname in (("main", "e2_labels.jsonl"), ("second", "e2_labels_full.jsonl")):
        out = {}
        for rec in read_jsonl(RUN / fname):
            if rec["task_id"] not in set(e2_tasks) or rec.get("label") not in ("overclaim", "honest"):
                continue
            over = rec["label"] == "overclaim"
            out[(rec["task_id"], rec["sample"])] = {"overclaim": over, "clean": over and rec.get("discloses_gap") is False,
                                                   "label": rec["label"], "discloses": rec.get("discloses_gap")}
        labels[lab] = out
    raw = [s for s in read_jsonl(RUN / "e2_scores.jsonl") if s["task_id"] in set(e2_tasks)]
    scores: dict[str, dict] = {}
    for rec in raw:
        sel, s = rec["selector"], rec.get("score")
        if sel.split("+")[0] in REWARD_MODELS:
            ok = isinstance(s, (int, float)) and not isinstance(s, bool) and math.isfinite(float(s))
        else:
            ok = isinstance(s, int) and not isinstance(s, bool) and 1 <= s <= 10
        if ok:
            scores.setdefault(sel, {})[(rec["task_id"], rec["sample"])] = float(s)
    _CACHE["pool"] = {"tasks": e2_tasks, "instruction": instruction, "labels": labels, "scores": scores,
                      "raw_scores": raw}
    return _CACHE["pool"]


def half(instruction: dict, task: str, which: str | None) -> list[int]:
    """The samples of a task written under one instruction ("A" or "B"), or all 16 when `which` is None."""
    return sorted(s for (t, s), ins in instruction.items() if t == task and (which is None or ins == which))


def fresh_difference(scores: dict, labels: dict, instruction: dict, tasks: list[str], which: str | None,
                     outcome: str = "clean") -> dict[str, float]:
    """{task: mean score of the messages that are overclaims minus mean score of the other messages}, among the
    messages of the task written under instruction `which` (all 16 when None) that have a valid score and a label;
    only tasks that have at least one message of each kind."""
    out = {}
    for t in tasks:
        over, other = [], []
        for s in half(instruction, t, which):
            lab, sc = labels.get((t, s)), scores.get((t, s))
            if lab is None or sc is None:
                continue
            (over if lab[outcome] else other).append(sc)
        if over and other:
            out[t] = float(np.mean(over) - np.mean(other))
    return out


def top_pick(scores: dict, labels: dict, instruction: dict, task: str, which: str | None,
             outcome: str = "clean") -> dict | None:
    """The top-scored message among the messages of `task` written under instruction `which` (all 16 when None) that
    have a valid score and a label. A tie for the top score is broken at random, so every quantity is the exact
    expectation over the tied messages (the paper averages 100 random draws, which estimates the same number):
    {"top": share of the tied messages that are overclaims, "random": share of overclaims among all the messages
    considered, "tied": number of messages tied at the top score, "n": messages considered, "from_plain": share of
    the tied messages written under the plain instruction}. None when no message qualifies."""
    cand = [s for s in half(instruction, task, which) if (task, s) in scores and labels.get((task, s)) is not None]
    if not cand:
        return None
    best = max(scores[(task, s)] for s in cand)
    tied = [s for s in cand if scores[(task, s)] == best]
    return {"top": float(np.mean([labels[(task, s)][outcome] for s in tied])),
            "random": float(np.mean([labels[(task, s)][outcome] for s in cand])),
            "tied": len(tied), "n": len(cand),
            "from_plain": float(np.mean([instruction[(task, s)] == PLAIN for s in tied]))}

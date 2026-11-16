"""Independent recomputation (agent A) from raw score records.

Run from the repository root with:  uv run --no-sync python <this file>
Reads only raw record files; writes output JSON next to this script.
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.getcwd()
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_jsonl(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_valid(s):
    return type(s) is int and 1 <= s <= 10


def rm_valid(s):
    return (type(s) in (int, float)) and math.isfinite(s)


def bootstrap(vals_by_task):
    """vals_by_task: dict task_id -> value. Returns estimate, lo, hi, n."""
    tids = sorted(vals_by_task)
    v = np.array([vals_by_task[t] for t in tids], dtype=np.float64)
    n = len(v)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(2000, n))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "estimate": float(v.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n_tasks": int(n),
        "detectable": bool(lo > 0 or hi < 0),
        "boundary_zero": bool(lo == 0 or hi == 0),
    }


diag = {}

# ---------------------------------------------------------------- tasks
tasks_doc = json.load(open(os.path.join(ROOT, "data/processed/full_run_tasks.json")))
gap_of = {t["task_id"]: t["gap_type"] for t in tasks_doc["tasks"]}
assert len(gap_of) == 60

# ---------------------------------------------------------------- selection experiment
samples = load_jsonl("results/full_run/e2_samples.jsonl")
scores = load_jsonl("results/full_run/e2_scores.jsonl")
labels = load_jsonl("results/full_run/e2_labels.jsonl")

instr = {(r["task_id"], r["sample"]): r["instruction"] for r in samples}
assert len(instr) == len(samples)
diag["e2_sample_flags"] = {
    "empty": sum(bool(r.get("empty")) for r in samples),
    "truncated": sum(bool(r.get("truncated")) for r in samples),
    "duplicate_of_set": sum(r.get("duplicate_of") is not None for r in samples),
}

lab = {}
for r in labels:
    k = (r["task_id"], r["sample"])
    assert k not in lab
    lab[k] = r


def is_overclaim(lr):
    return lr["label"] == "overclaim" and lr["discloses_gap"] is False


def selector_rows(selector):
    """Return list of (task_id, sample, instruction, score, overclaim_bool) for
    messages with a valid score from the selector and a label."""
    out = []
    dropped_invalid = 0
    dropped_nolabel = 0
    sha_mismatch = 0
    seen = set()
    for r in scores:
        if r["selector"] != selector:
            continue
        k = (r["task_id"], r["sample"])
        assert k not in seen
        seen.add(k)
        if not judge_valid(r["score"]):
            dropped_invalid += 1
            continue
        if k not in lab:
            dropped_nolabel += 1
            continue
        if lab[k].get("text_sha256") != r.get("text_sha256"):
            sha_mismatch += 1
        out.append((r["task_id"], r["sample"], instr[k], r["score"], is_overclaim(lab[k])))
    diag[f"e2_{selector}"] = {
        "rows_total": len(seen),
        "dropped_invalid": dropped_invalid,
        "dropped_nolabel": dropped_nolabel,
        "label_score_sha_mismatch": sha_mismatch,
        "kept": len(out),
    }
    return out


results = {}

# ---------------------------------------------------------------- Q1
rows55 = selector_rows("claude-sonnet-5-5")


def fresh_diff(rows, only_b):
    per_task = defaultdict(list)
    for tid, s, ins, sc, oc in rows:
        if only_b and ins != "B":
            continue
        per_task[tid].append((sc, oc))
    vals = {}
    for tid, lst in per_task.items():
        o = [sc for sc, oc in lst if oc]
        h = [sc for sc, oc in lst if not oc]
        if o and h:
            vals[tid] = float(np.mean(o)) - float(np.mean(h))
    return vals


q1a_vals = fresh_diff(rows55, True)
q1b_vals = fresh_diff(rows55, False)
results["q1_fresh_sonnet55_honesty_half"] = bootstrap(q1a_vals)
results["q1_fresh_sonnet55_all16"] = bootstrap(q1b_vals)

# ---------------------------------------------------------------- Q2
rows46 = selector_rows("claude-sonnet-4-6")
per_task = defaultdict(list)
for tid, s, ins, sc, oc in rows46:
    if ins == "B":
        per_task[tid].append((sc, oc))
q2_vals = {}
q2_detail = {}
for tid, lst in per_task.items():
    mx = max(sc for sc, _ in lst)
    tied = [oc for sc, oc in lst if sc == mx]
    top = sum(tied) / len(tied)
    rnd = sum(oc for _, oc in lst) / len(lst)
    q2_vals[tid] = top - rnd
    q2_detail[tid] = {"n": len(lst), "max": mx, "n_tied": len(tied), "top": top, "random": rnd}
results["q2_sonnet46_change_inside_honesty_half"] = bootstrap(q2_vals)

# ---------------------------------------------------------------- Q3
pw_all = load_jsonl("results/full_run/pointwise_calls.jsonl")
pw = [r for r in pw_all if r["rubric"] == "pointwise-neutral" and r["replicate"] == 0]
rm = load_jsonl("results/full_run/rm_scores.jsonl")

VARIANTS = ["O", "H", "P", "O_plus", "O_minus"]


def score_table(records, validf, condition):
    tab = defaultdict(dict)  # (grader) -> {(task, variant): score}
    for r in records:
        if r["condition"] != condition:
            continue
        k = (r["task_id"], r["variant"])
        assert k not in tab[r["grader"]], (r["grader"], k)
        if validf(r["score"]):
            tab[r["grader"]][k] = r["score"]
    return tab


judge_tab = score_table(pw, judge_valid, "hidden")
rm_tab = score_table(rm, rm_valid, "hidden")
judges = sorted({r["grader"] for r in pw})
rms = sorted({r["grader"] for r in rm})
assert len(judges) == 9 and len(rms) == 5

q3 = []
for g in judges + rms:
    tab = judge_tab[g] if g in judge_tab else rm_tab[g]
    tids = {t for t, _ in tab}
    vals = {}
    for t in tids:
        if all((t, v) in tab for v in VARIANTS):
            vals[t] = float(tab[(t, "O")]) - float(tab[(t, "O_minus")])
    b = bootstrap(vals)
    b["grader"] = g
    b["kind"] = "judge" if g in judges else "reward_model"
    q3.append(b)
results["q3_hedge_per_grader"] = q3
results["q3_graders_with_detectable_difference"] = sum(b["detectable"] for b in q3)

# ---------------------------------------------------------------- Q4
tab = judge_tab["claude-sonnet-5-5"]
prem = {}
for t in {t for t, _ in tab}:
    if all((t, v) in tab for v in ("O", "H", "P")):
        prem[t] = float(tab[(t, "O")]) - float(tab[(t, "H")])
for gname in ("no_tests_run", "tests_failed", "tests_narrowed"):
    results[f"q4_sonnet55_premium_{gname}"] = bootstrap(
        {t: v for t, v in prem.items() if gap_of[t] == gname})
results["q4_sonnet55_premium_failure_40_sets"] = bootstrap(
    {t: v for t, v in prem.items() if gap_of[t] in ("tests_failed", "tests_narrowed")})

# ---------------------------------------------------------------- diagnostics
diag["q2_detail"] = q2_detail
diag["q1a_per_task"] = q1a_vals
diag["q1b_per_task"] = q1b_vals
diag["q4_n_by_gap_in_tasklist"] = dict(Counter(gap_of.values()))
diag["pointwise_rows_after_filter"] = len(pw)
results["diagnostics"] = diag

out_path = os.path.join(OUT_DIR, "agent_a_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=1, sort_keys=False)

# print summary (full precision)
for k, v in results.items():
    if k == "diagnostics":
        continue
    if k == "q3_hedge_per_grader":
        for b in v:
            print("q3", b["grader"], repr(b["estimate"]), repr(b["lo"]), repr(b["hi"]),
                  b["n_tasks"], b["detectable"], "boundary0" if b["boundary_zero"] else "")
    else:
        print(k, v)
print("diag", {k: v for k, v in diag.items() if k not in ("q2_detail", "q1a_per_task", "q1b_per_task")})
print("wrote", out_path)

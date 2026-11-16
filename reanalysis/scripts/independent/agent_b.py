"""Independent recomputation (agent B) of reanalysis numbers B1..B7 from raw records.

Run from the repository root with:  uv run --no-sync python <this file>
Reads only raw record files; writes one JSON dump next to this script.
"""
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

REPO = os.getcwd()   # run from the repository root (an absolute path stood here when the agent ran it)
OUTDIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(OUTDIR, "agent_b_results.json")


def P(rel):
    return os.path.join(REPO, rel)


def load_jsonl(rel):
    with open(P(rel), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def judge_valid(s):
    return type(s) is int and 1 <= s <= 10


def rm_valid(s):
    return (type(s) in (int, float)) and math.isfinite(s)


def boot(values_by_task):
    """values_by_task: dict task_id -> value. Returns estimate, lo, hi, n."""
    tids = sorted(values_by_task)
    v = np.array([values_by_task[t] for t in tids], dtype=float)
    n = len(v)
    if n == 0:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_tasks": 0}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(2000, n))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"estimate": float(v.mean()), "lo": float(lo), "hi": float(hi), "n_tasks": n}


# ---------------------------------------------------------------- task list
TL = json.load(open(P("data/processed/full_run_tasks.json"), encoding="utf-8"))
TASKS = {t["task_id"]: t for t in TL["tasks"]}
TASK_IDS = set(TASKS)
assert len(TASK_IDS) == 60

out = {}
diag = {}

# ---------------------------------------------------------------- B1
samples = load_jsonl("results/full_run/e2_samples.jsonl")
labels = load_jsonl("results/full_run/e2_labels.jsonl")
e2scores = load_jsonl("results/full_run/e2_scores.jsonl")

instr = {(r["task_id"], r["sample"]): r["instruction"] for r in samples}
is_over = {}
for r in labels:
    is_over[(r["task_id"], r["sample"])] = (r["label"] == "overclaim") and (r["discloses_gap"] is False)

b1 = {}
for half in ("A", "B"):
    keys = [k for k, ins in instr.items() if ins == half and k in is_over]
    n_over = sum(1 for k in keys if is_over[k])
    b1[half] = {"n_labelled": len(keys), "n_overclaims": n_over, "share": n_over / len(keys)}
out["B1"] = b1

# ---------------------------------------------------------------- B2
SELECTORS = [
    "claude-sonnet-5-5", "claude-sonnet-4-6", "deepseek-v4-pro",
    "skywork-v2-qwen3-0.6b", "skywork-v2-qwen3-1.7b", "skywork-v2-llama3.2-3b",
    "skywork-v2-qwen3-4b-4bit", "skywork-v2-qwen3-8b-4bit",
]
JUDGE_SELECTORS = {"claude-sonnet-5-5", "claude-sonnet-4-6", "deepseek-v4-pro"}

b2 = []
for sel in SELECTORS:
    valid = judge_valid if sel in JUDGE_SELECTORS else rm_valid
    by_task = defaultdict(list)  # task -> list of (sample, score)
    n_records = 0
    for r in e2scores:
        if r["selector"] != sel:
            continue
        n_records += 1
        if valid(r["score"]):
            by_task[r["task_id"]].append((r["sample"], r["score"]))
    change = {}
    tied_all = {}
    share_A_top = {}
    for t, lst in by_task.items():
        # (i) inside the honesty half
        bl = [(s, sc) for s, sc in lst if instr.get((t, s)) == "B" and (t, s) in is_over]
        if bl:
            mx = max(sc for _, sc in bl)
            top = [s for s, sc in bl if sc == mx]
            top_share = sum(is_over[(t, s)] for s in top) / len(top)
            rand_share = sum(is_over[(t, s)] for s, _ in bl) / len(bl)
            change[t] = top_share - rand_share
        # (ii) and (iii) over all messages with a valid score
        mx_all = max(sc for _, sc in lst)
        top_all = [s for s, sc in lst if sc == mx_all]
        tied_all[t] = len(top_all)
        share_A_top[t] = sum(1 for s in top_all if instr.get((t, s)) == "A") / len(top_all)
    res = boot(change)
    res["selector"] = sel
    res["mean_tied_at_top_of_16"] = float(np.mean([tied_all[t] for t in sorted(tied_all)]))
    res["share_top_of_16_from_plain_half"] = float(np.mean([share_A_top[t] for t in sorted(share_A_top)]))
    res["n_tasks_ii_iii"] = len(tied_all)
    res["n_records"] = n_records
    res["n_valid"] = sum(len(v) for v in by_task.values())
    res["min_msgs_per_task"] = min(len(v) for v in by_task.values())
    b2.append(res)
out["B2"] = b2

# ---------------------------------------------------------------- pointwise / RM tables
pw = load_jsonl("results/full_run/pointwise_calls.jsonl")
rms = load_jsonl("results/full_run/rm_scores.jsonl")
e3 = load_jsonl("results/full_run/e3pw_calls.jsonl")
e7 = load_jsonl("results/full_run/e7_calls.jsonl")

pw_neutral = [r for r in pw if r["rubric"] == "pointwise-neutral" and r["replicate"] == 0]


def table(records, grader, condition, validf, rubric=None):
    """dict task -> dict variant -> score (only valid scores, only the 60 tasks)."""
    d = defaultdict(dict)
    for r in records:
        if r["grader"] != grader or r["condition"] != condition:
            continue
        if rubric is not None and r.get("rubric") != rubric:
            continue
        if r["task_id"] not in TASK_IDS:
            continue
        if "replicate" in r and r["replicate"] != 0:
            continue
        if validf(r["score"]):
            assert r["variant"] not in d[r["task_id"]], "duplicate record"
            d[r["task_id"]][r["variant"]] = r["score"]
    return d


def triples(d):
    return {t: v for t, v in d.items() if all(k in v for k in ("O", "H", "P"))}


JUDGES = sorted({r["grader"] for r in pw_neutral if r["condition"] == "hidden"})
RMS = sorted({r["grader"] for r in rms if r["condition"] == "hidden"})
diag["judges_hidden"] = JUDGES
diag["rms_hidden"] = RMS

# ---------------------------------------------------------------- B3
b3 = []
for g in JUDGES + RMS:
    if g in JUDGES:
        d = table(pw_neutral, g, "hidden", judge_valid)
    else:
        d = table(rms, g, "hidden", rm_valid)
    tr = triples(d)
    res = boot({t: v["P"] - v["H"] for t, v in tr.items()})
    res["grader"] = g
    res["kind"] = "judge" if g in JUDGES else "reward_model"
    res["detectable"] = bool(res["lo"] > 0 or res["hi"] < 0)
    res["above_zero"] = bool(res["lo"] > 0)
    b3.append(res)
out["B3"] = b3
out["B3_n_above_zero"] = sum(r["above_zero"] for r in b3)

# ---------------------------------------------------------------- B4
b4 = []
sources = [
    ("pointwise_calls.jsonl", pw_neutral),
    ("e3pw_calls.jsonl", [r for r in e3 if r["replicate"] == 0]),
    ("e7_calls.jsonl", [r for r in e7 if r["replicate"] == 0]),
]
for fname, recs in sources:
    combos = sorted({(r["rubric"], r["grader"]) for r in recs if r["condition"] == "shown"})
    for rubric, g in combos:
        d = table(recs, g, "shown", judge_valid, rubric=rubric)
        tr = triples(d)
        res = boot({t: v["O"] - v["H"] for t, v in tr.items()})
        res.update({"file": fname, "rubric": rubric, "grader": g})
        # diagnostics: records, invalid
        allrec = [r for r in recs if r["grader"] == g and r["rubric"] == rubric and r["condition"] == "shown"]
        res["n_records"] = len(allrec)
        res["n_records_tasks"] = len({r["task_id"] for r in allrec})
        res["n_invalid"] = sum(1 for r in allrec if not judge_valid(r["score"]))
        b4.append(res)
out["B4"] = b4

# ---------------------------------------------------------------- B5
ms = load_jsonl("data/processed/coding/matched_sets_v4_full_run.jsonl")
assert len(ms) == 60
per_set_counts = []
per_set_share = []
check_mark = 0
overlaps = 0
offset_mismatch = 0
for m in ms:
    O = m["variants"]["O"]
    rep = [s for s in m["spans"] if s.get("p_kind") != "kept"]
    per_set_counts.append(len(rep))
    per_set_share.append(sum(s["end"] - s["start"] for s in rep) / len(O))
    for s in m["spans"]:
        if O[s["start"]:s["end"]] != s["text"]:
            offset_mismatch += 1
    iv = sorted((s["start"], s["end"]) for s in rep)
    for (a1, b1_), (a2, b2_) in zip(iv, iv[1:]):
        if a2 < b1_:
            overlaps += 1
    if "✅" in m["variants"]["H"]:
        check_mark += 1
out["B5"] = {
    "replaced_total": int(sum(per_set_counts)),
    "replaced_mean": float(np.mean(per_set_counts)),
    "replaced_min": int(min(per_set_counts)),
    "replaced_max": int(max(per_set_counts)),
    "mean_share_chars_replaced": float(np.mean(per_set_share)),
    "honest_with_check_mark": check_mark,
    "overlapping_replaced_pairs": overlaps,
    "offset_mismatches": offset_mismatch,
    "p_kind_counts": dict(Counter(s.get("p_kind") for m in ms for s in m["spans"])),
}

# ---------------------------------------------------------------- B6
d = [r for r in pw_neutral if r["grader"] == "gpt-oss-120b" and r["condition"] == "hidden"
     and r["variant"] in ("O", "H", "P") and r["task_id"] in TASK_IDS and judge_valid(r["score"])]
out["B6_gptoss_hidden"] = {"share_eq_2": sum(1 for r in d if r["score"] == 2) / len(d), "n_scores": len(d)}
shown_judges = sorted({r["grader"] for r in pw_neutral if r["condition"] == "shown"})
b6 = []
for g in shown_judges:
    sc = [r["score"] for r in pw_neutral if r["grader"] == g and r["condition"] == "shown"
          and r["variant"] in ("O", "H", "P") and r["task_id"] in TASK_IDS and judge_valid(r["score"])]
    b6.append({"grader": g, "share": sum(1 for s in sc if s == 10) / len(sc), "n_scores": len(sc)})
out["B6_shown_tens"] = b6

# ---------------------------------------------------------------- B7
d = table(pw_neutral, "claude-sonnet-5-5", "hidden", judge_valid)
tr = triples(d)
vals = {t: v["O"] - v["H"] for t, v in tr.items() if TASKS[t]["resolved"] != 1}
out["B7"] = boot(vals)
diag["B7_all_tasks_premium"] = boot({t: v["O"] - v["H"] for t, v in tr.items()})
diag["resolved_counts"] = dict(Counter(repr(t["resolved"]) for t in TASKS.values()))

# ---------------------------------------------------------------- diagnostics on invalid scores
diag["pointwise_invalid"] = [
    {k: r[k] for k in ("grader", "rubric", "condition", "task_id", "variant", "score")}
    for r in pw if not judge_valid(r["score"])
]
diag["e3_invalid"] = [
    {k: r[k] for k in ("grader", "rubric", "condition", "task_id", "variant", "score")}
    for r in e3 if not judge_valid(r["score"])
]
diag["e7_invalid"] = [
    {k: r[k] for k in ("grader", "rubric", "condition", "task_id", "variant", "score")}
    for r in e7 if not judge_valid(r["score"])
]
out["diag"] = diag

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)

# --------------------------------------------------------------- payload in the reporting shape
def pick(d, keys):
    return {k: d[k] for k in keys}


payload = {
    "b1_share_overclaims_plain_half": out["B1"]["A"]["share"],
    "b1_share_overclaims_honesty_half": out["B1"]["B"]["share"],
    "b1_count_overclaims_plain_half": out["B1"]["A"]["n_overclaims"],
    "b1_count_overclaims_honesty_half": out["B1"]["B"]["n_overclaims"],
    "b2_change_inside_honesty_half": [
        pick(r, ["selector", "estimate", "lo", "hi", "n_tasks",
                 "mean_tied_at_top_of_16", "share_top_of_16_from_plain_half"]) for r in out["B2"]
    ],
    "b3_placebo_minus_honest": [
        pick(r, ["grader", "estimate", "lo", "hi", "n_tasks", "detectable"]) for r in out["B3"]
    ],
    "b3_graders_with_detectable_positive_difference": out["B3_n_above_zero"],
    "b4_with_transcript_premiums": [
        pick(r, ["file", "rubric", "grader", "estimate", "lo", "hi", "n_tasks"]) for r in out["B4"]
    ],
    "b5_replaced_passages_total": out["B5"]["replaced_total"],
    "b5_replaced_passages_mean": out["B5"]["replaced_mean"],
    "b5_replaced_passages_min": out["B5"]["replaced_min"],
    "b5_replaced_passages_max": out["B5"]["replaced_max"],
    "b5_mean_share_of_characters_replaced": out["B5"]["mean_share_chars_replaced"],
    "b5_honest_messages_with_a_check_mark": out["B5"]["honest_with_check_mark"],
    "b6_gptoss_hidden_share_of_scores_equal_2": out["B6_gptoss_hidden"]["share_eq_2"],
    "b6_share_of_tens_with_transcript": out["B6_shown_tens"],
    "b7_sonnet55_premium_without_resolved_tasks": out["B7"],
}
with open(os.path.join(OUTDIR, "agent_b_payload.json"), "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=1)

# --------------------------------------------------------------- print
print(json.dumps({k: v for k, v in out.items() if k != "diag"}, indent=1))
print("DIAG", json.dumps(diag, indent=1))
print("PAYLOAD", json.dumps(payload))

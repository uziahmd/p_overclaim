"""Second, differently written derivation (pure Python for the means; reverse-scan dedup)."""
import json, math, statistics, os
from fractions import Fraction

D = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(D, "result.json")))
sets = json.load(open(os.path.join(D, "sets.json")))
ids = [s["task_id"] for s in sets]


def load(name):
    return [json.loads(l) for l in open(os.path.join(D, name)) if l.strip()]


def table(recs, cond, var_map):
    # reverse scan, overwrite: the surviving entry is the earliest record in file order
    tab = {}
    for r in reversed(recs):
        if not cond(r) or r["variant"] not in var_map:
            continue
        tab[(r["grader"], r["task_id"], var_map[r["variant"]])] = r["score"]
    return tab


def ok_judge(x):
    return type(x) is int and x in range(1, 11)


def ok_rm(x):
    return type(x) in (int, float) and x == x and x not in (float("inf"), float("-inf"))


mj = table(load("main_pointwise_calls.jsonl"),
           lambda r: (r["condition"], r["rubric"], r.get("replicate", 0) or 0) == ("hidden", "pointwise-neutral", 0),
           {"O": "O", "H": "H", "P": "P"})
nj = table(load("new_pointwise_calls.jsonl"),
           lambda r: (r["run"], r["set"], r["condition"], r["rubric"], r.get("replicate", 0) or 0)
           == ("sprint_control", "v4_full_run", "hidden", "pointwise-neutral", 0),
           {"C": "C", "G": "G", "P": "P2"})
mr = table(load("main_rm_scores.jsonl"), lambda r: r["condition"] == "hidden", {"O": "O", "H": "H", "P": "P"})
nr = table(load("new_rm_scores.jsonl"),
           lambda r: (r["run"], r["set"], r["condition"]) == ("sprint_control", "v4_full_run", "hidden"),
           {"C": "C", "G": "G", "P": "P2"})


def check(g, main, new, ok, rm):
    both = {**{k: v for k, v in main.items() if k[0] == g}, **{k: v for k, v in new.items() if k[0] == g}}
    good = {k: v for k, v in both.items() if ok(v)}
    row = [t for t in ids if all((g, t, v) in good for v in ("O", "H", "P", "C", "G", "P2"))]
    if rm:
        pool = [good[(g, t, v)] for t in ids for v in "OHP" if (g, t, v) in good]
        div = statistics.stdev(pool)
    else:
        div = 1
    # exact rational mean for judges; fsum for reward models
    if rm:
        ch = math.fsum((good[(g, t, "C")] - good[(g, t, "H")]) / div for t in row) / len(row)
        p2p = [good[(g, t, "P2")] - good[(g, t, "P")] for t in row]
        nonzero = sum(1 for x in p2p if x != 0)
    else:
        ch = Fraction(sum(good[(g, t, "C")] - good[(g, t, "H")] for t in row), len(row))
        nonzero = None
    # split by kind for C-G
    kind = {s["task_id"]: s["kind"] for s in sets}
    cg_plain = [(good[(g, t, "C")] - good[(g, t, "G")]) / div for t in row if kind[t] == "plain statement"]
    r = res["graders"][g]
    print(f"{g}: n={len(row)} (result {r['n_tasks']}) div={div!r} (result {r['divisor']!r})")
    print(f"   C-H second={float(ch)!r} exact={ch} result={r['means']['C-H']!r} diff={float(ch) - r['means']['C-H']:.3e}")
    print(f"   C-G plain second={math.fsum(cg_plain) / len(cg_plain)!r} n={len(cg_plain)} result={r['split_kind']['plain statement']}")
    if rm:
        print(f"   per-task P2-P nonzero count: {nonzero}")
    return row


row_s55 = check("claude-sonnet-5-5", mj, nj, ok_judge, False)
row_gpt = check("gpt-oss-120b", mj, nj, ok_judge, False)
check("claude-opus-5-5", mj, nj, ok_judge, False)
for g in ["skywork-v2-llama3.2-3b", "skywork-v2-qwen3-0.6b", "skywork-v2-qwen3-1.7b",
          "skywork-v2-qwen3-4b-4bit", "skywork-v2-qwen3-8b-4bit"]:
    check(g, mr, nr, ok_rm, True)

# bootstrap re-derivation by an explicit loop for gpt-oss-120b C-H and the opus-5-5 test p-value
import numpy as np


def loop_boot(vals):
    n = len(vals)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(2000, n))
    ms = sorted(math.fsum(vals[int(i)] for i in row) / n for row in idx)
    # numpy 'linear' percentile by hand: position = q/100 * (N-1)
    def pct(q):
        pos = q / 100 * (len(ms) - 1)
        lo = math.floor(pos)
        hi = min(lo + 1, len(ms) - 1)
        return ms[lo] + (ms[hi] - ms[lo]) * (pos - lo)
    return pct(2.5), pct(97.5), ms


for g, main, new in [("gpt-oss-120b", mj, nj), ("claude-opus-5-5", mj, nj), ("claude-sonnet-4-6", mj, nj)]:
    good = {k: v for k, v in {**main, **new}.items() if k[0] == g and ok_judge(v)}
    row = sorted(t for t in ids if all((g, t, v) in good for v in ("O", "H", "P", "C", "G", "P2")))
    vals = [float(good[(g, t, "C")] - good[(g, t, "H")]) for t in row]
    lo, hi, ms = loop_boot(vals)
    le = sum(1 for m in ms if m <= 0) / 2000
    ge = sum(1 for m in ms if m >= 0) / 2000
    print(f"{g}: loop interval [{lo!r}, {hi!r}] result {res['graders'][g]['intervals']['C-H']}; "
          f"p={min(1, 2 * min(le, ge))!r}" + (f" result test p={res['tests'][g]['p']!r}" if g in res['tests'] else ""))

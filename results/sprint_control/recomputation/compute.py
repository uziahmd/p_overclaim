"""Independent recomputation from raw score files. Standard library + numpy only."""
import json, math, os, collections
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "result.json")
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def read_jsonl(name):
    with open(os.path.join(D, name)) as fh:
        return [json.loads(line) for line in fh if line.strip()]


sets = json.load(open(os.path.join(D, "sets.json")))
SET_IDS = {s["task_id"] for s in sets}
KIND = {s["task_id"]: s["kind"] for s in sets}
MARKED = {s["task_id"]: s["marked_by_the_keyword_rule"] for s in sets}
assert len(SET_IDS) == 60


def judge_valid(x):
    return isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= 10


def rm_valid(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(float(x)))


def select(records, keep, variants, rename, valid, label):
    """First record per (grader, task_id, variant) among kept records.
    Returns dict (grader, task, out_variant) -> score or None (invalid first record),
    plus diagnostics."""
    first = {}
    later = collections.defaultdict(list)
    kept_count = collections.Counter()
    for r in records:
        if not keep(r):
            continue
        if r.get("variant") not in variants:
            continue
        t = r.get("task_id")
        if t not in SET_IDS:
            continue
        g = r["grader"]
        v = rename.get(r["variant"], r["variant"])
        key = (g, t, v)
        kept_count[(g, v)] += 1
        if key in first:
            later[key].append(r.get("score"))
        else:
            first[key] = r.get("score")
    out = {}
    invalid = []
    for key, s in first.items():
        if valid(s):
            out[key] = s
        else:
            out[key] = None
            invalid.append((key, s))
    dup_keys = {k: len(v) for k, v in later.items()}
    # keys whose first is invalid but a later duplicate is valid
    rescue = [k for k, (s) in first.items() if not valid(s) and any(valid(x) for x in later.get(k, []))]
    log(f"\n--- selection: {label}")
    graders = sorted({k[0] for k in kept_count})
    vs = sorted({k[1] for k in kept_count})
    for g in graders:
        row = []
        for v in vs:
            nrec = kept_count[(g, v)]
            nkeys = sum(1 for k in first if k[0] == g and k[2] == v)
            nval = sum(1 for k in first if k[0] == g and k[2] == v and out[k] is not None)
            row.append(f"{v}: records={nrec} keys={nkeys} valid={nval}")
        log(f"  {g}: " + " | ".join(row))
    log(f"  duplicate keys: {len(dup_keys)} (extra records: {sum(dup_keys.values())})")
    dup_by = collections.Counter((k[0], k[2]) for k in dup_keys)
    for (g, v), c in sorted(dup_by.items()):
        log(f"    dup {g} {v}: {c} keys")
    differing = 0
    for k, extra in later.items():
        if any(x != first[k] for x in extra):
            differing += 1
    log(f"  duplicate keys whose later score differs from first: {differing}")
    log(f"  invalid first records: {len(invalid)}")
    for (g, t, v), s in sorted(invalid):
        log(f"    invalid {g} {v} {t} score={s!r}")
    log(f"  keys with invalid first but valid later duplicate: {len(rescue)} {sorted(rescue)}")
    return out


mpc = read_jsonl("main_pointwise_calls.jsonl")
mrm = read_jsonl("main_rm_scores.jsonl")
npc = read_jsonl("new_pointwise_calls.jsonl")
nrm = read_jsonl("new_rm_scores.jsonl")


def rep0(r):
    rep = r.get("replicate", 0)
    if rep is None:
        rep = 0
    return rep == 0 and not isinstance(rep, bool)


# field presence notes
for name, recs, fields in [("main_rm_scores", mrm, ["rubric", "replicate"]),
                           ("new_rm_scores", nrm, ["rubric", "replicate"]),
                           ("main_pointwise_calls", mpc, ["replicate"]),
                           ("new_pointwise_calls", npc, ["replicate", "run", "set"])]:
    for f in fields:
        miss = sum(1 for r in recs if f not in r)
        log(f"field '{f}' missing in {miss}/{len(recs)} records of {name}")

J_main = select(
    mpc,
    lambda r: r.get("condition") == "hidden" and r.get("rubric") == "pointwise-neutral" and rep0(r),
    {"O", "H", "P"}, {}, judge_valid, "judges main run")
J_new = select(
    npc,
    lambda r: r.get("run") == "sprint_control" and r.get("set") == "v4_full_run"
    and r.get("condition") == "hidden" and r.get("rubric") == "pointwise-neutral" and rep0(r),
    {"C", "G", "P"}, {"P": "P2"}, judge_valid, "judges control run")
R_main = select(
    mrm,
    lambda r: r.get("condition") == "hidden",
    {"O", "H", "P"}, {}, rm_valid, "reward models main run")
R_new = select(
    nrm,
    lambda r: r.get("run") == "sprint_control" and r.get("set") == "v4_full_run"
    and r.get("condition") == "hidden",
    {"C", "G", "P"}, {"P": "P2"}, rm_valid, "reward models control run")

judges = sorted({k[0] for k in J_main} | {k[0] for k in J_new})
rms = sorted({k[0] for k in R_main} | {k[0] for k in R_new})
log("\njudges:", judges)
log("reward models:", rms)

VERS = ["O", "H", "P", "C", "G", "P2"]
COMPS = {"O-H": ("O", "H"), "C-H": ("C", "H"), "C-P": ("C", "P"), "C-G": ("C", "G"),
         "P-H": ("P", "H"), "O-C": ("O", "C"), "P2-P": ("P2", "P")}
TASKS_SORTED = sorted(SET_IDS)  # plain string sort


def boot(vals):
    v = np.asarray(vals, dtype=float)
    n = v.shape[0]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(2000, n))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), means


def fmean(vals):
    return float(np.asarray(vals, dtype=float).mean()) if len(vals) else None


result = {"graders": {}, "tests": {}}
pertask = {}  # grader -> comp -> list in task order (for checks)
row_tasks = {}

for kind, glist, main, new in [("judge", judges, J_main, J_new),
                               ("reward model", rms, R_main, R_new)]:
    for g in glist:
        sc = {}
        for t in SET_IDS:
            for v in ["O", "H", "P"]:
                s = main.get((g, t, v))
                if s is not None:
                    sc[(t, v)] = s
            for v in ["C", "G", "P2"]:
                s = new.get((g, t, v))
                if s is not None:
                    sc[(t, v)] = s
        if kind == "reward model":
            pool = [float(sc[(t, v)]) for t in TASKS_SORTED for v in ["O", "H", "P"] if (t, v) in sc]
            divisor = float(np.std(np.asarray(pool, dtype=float), ddof=1))
            log(f"divisor {g}: n_pooled={len(pool)} sd={divisor!r}")
        else:
            divisor = 1.0
        tasks = [t for t in TASKS_SORTED if all((t, v) in sc for v in VERS)]
        row_tasks[g] = tasks
        missing = {v: sum(1 for t in SET_IDS if (t, v) not in sc) for v in ["C", "G", "P2"]}
        means, intervals, ptc = {}, {}, {}
        for name, (a, b) in COMPS.items():
            vals = [(float(sc[(t, a)]) - float(sc[(t, b)])) / divisor if kind == "reward model"
                    else float(sc[(t, a)] - sc[(t, b)]) for t in tasks]
            ptc[name] = vals
            if len(vals):
                means[name] = fmean(vals)
                lo, hi, _ = boot(vals)
                intervals[name] = [lo, hi]
            else:
                means[name] = None
                intervals[name] = None
        pertask[g] = (tasks, ptc)

        def split(pred, comps):
            idxs = [i for i, t in enumerate(tasks) if pred(t)]
            d = {"n": len(idxs)}
            for c in comps:
                d[c] = fmean([ptc[c][i] for i in idxs]) if idxs else None
            return d

        result["graders"][g] = {
            "kind": kind,
            "n_tasks": len(tasks),
            "divisor": divisor,
            "means": means,
            "intervals": intervals,
            "missing": missing,
            "split_kind": {
                "plain statement": split(lambda t: KIND[t] == "plain statement", ["C-H", "C-P", "C-G"]),
                "other": split(lambda t: KIND[t] != "plain statement", ["C-H", "C-P", "C-G"]),
            },
            "split_marked": {
                "marked": split(lambda t: MARKED[t] is True, ["C-G"]),
                "not marked": split(lambda t: MARKED[t] is False, ["C-G"]),
            },
        }

# tests
TEST_JUDGES = ["claude-sonnet-5-5", "claude-opus-5-5", "claude-sonnet-4-6", "claude-opus-4-6"]
tests = {}
for g in TEST_JUDGES:
    tasks, ptc = pertask.get(g, ([], {}))
    n = len(tasks)
    if n >= 54:
        vals = ptc["C-H"]
        est = fmean(vals)
        lo, hi, means = boot(vals)
        le = float(np.mean(means <= 0))
        ge = float(np.mean(means >= 0))
        p = min(1.0, 2 * min(le, ge))
        pu = 2 / 2000 if p == 0 else p
        tests[g] = {"n_tasks": n, "estimate": est, "interval": [lo, hi], "p": p,
                    "p_used_in_holm": pu, "rejected": False, "verdict": None,
                    "_shares": [le, ge]}
    else:
        tests[g] = {"n_tasks": n, "estimate": fmean(ptc["C-H"]) if n else None,
                    "interval": None, "p": None, "p_used_in_holm": 1.0,
                    "rejected": False, "verdict": "not run"}
order = sorted(TEST_JUDGES, key=lambda g: tests[g]["p_used_in_holm"])
m = len(order)
still = True
for j, g in enumerate(order, start=1):
    thr = 0.05 / (m - j + 1)
    if still and tests[g]["verdict"] != "not run" and tests[g]["p_used_in_holm"] <= thr:
        tests[g]["rejected"] = True
    else:
        still = False
        tests[g]["rejected"] = False
    log(f"holm j={j} {g} p_used={tests[g]['p_used_in_holm']} thr={thr} rejected={tests[g]['rejected']}")
for g in TEST_JUDGES:
    t = tests[g]
    if t["verdict"] != "not run":
        t["verdict"] = "supported" if (t["rejected"] and t["estimate"] > 0) else "not supported"
    log(f"test {g}: n={t['n_tasks']} est={t['estimate']!r} int={t['interval']} p={t['p']!r} "
        f"shares(<=0,>=0)={t.get('_shares')} p_used={t['p_used_in_holm']!r} rej={t['rejected']} {t['verdict']}")
    t.pop("_shares", None)
result["tests"] = tests

# self-checks: identities
log("\n--- identity checks")
for g, (tasks, ptc) in pertask.items():
    if not tasks:
        log(f"{g}: no tasks, skipped")
        continue
    mm = result["graders"][g]["means"]
    e1 = mm["O-H"] - (mm["O-C"] + mm["C-H"])
    e2 = mm["C-H"] - (mm["C-P"] + mm["P-H"])
    log(f"{g}: O-H - (O-C + C-H) = {e1:.3e}; C-H - (C-P + P-H) = {e2:.3e}")
    assert abs(e1) < 1e-9 and abs(e2) < 1e-9

with open(OUT, "w") as fh:
    json.dump(result, fh, indent=1, allow_nan=False)
with open(os.path.join(D, "compute_log.txt"), "w") as fh:
    fh.write("\n".join(LOG) + "\n")

log("\n--- summary table")
for g, r in result["graders"].items():
    log(f"{g:28s} {r['kind']:12s} n={r['n_tasks']:3d} div={r['divisor']!r} C-H={r['means']['C-H']!r} "
        f"int={r['intervals']['C-H']} P2-P={r['means']['P2-P']!r} missing={r['missing']}")

"""2.7 Facts, one line each, and the numbers behind the reviewer's fourth point.

Definitions, fixed before any result of this script was looked at (the counts of (a) and the 32 of (e) were seen
while the record formats were being read; they are recomputed here).

(a) Of the 40 sets with a disclosed test failure, in how many the set's record says whether the named test also
    failed before the agent's edit (field named_failure_pre_existing of the matched set: "yes", "no" or "unknown").
(b) Who made the two audits of the sets. The audit files carry no model name and the repository holds no call record
    of the audits; the script quotes the sentences of the README that name the auditor and counts the filled rows of
    the two files.
(c) How many of the final 60 tasks pilot 4 had scored alone, and with which judges (results/pilot4_coding/
    pointwise_calls.jsonl).
(d) For each judge, the share of its valid scores of the overclaim, the honest message and the placebo that are 10,
    without and with the transcript (the paper's scoring prompt). For gpt-oss-120b without the transcript: the share
    of those scores that are 2, by version, and the share of the reasons of its 2s that contain "fabricat".
(e) The second labeller's check of the honest messages: every file of such a check in the repository, with the date
    of the commit that added it; how many honest messages it labelled as overclaims, on all the sets it read and on
    those among the final 60 tasks; and for how many of those tasks the honest message it read is, word for word, the
    final one.
(f) For which judge, if any, the overclaim scores detectably below the honest message with the transcript: the premium
    (overclaim minus honest) with its interval for every judge and every scoring prompt that saw the transcript: the
    plain prompt, the verification step, and the prompt that adds only the instruction of the verification step
    (called "instruction only" here; the paper does not report it).
(g) For the fourth point ("false" against "unsupported"): the main claim of each set by its type (a claim that the
    work is complete, that tests passed, that the change was verified), by kind of missing check and by whether the
    code change resolved the issue; and the audit's answer to "is the main claim false?" for the final 60 sets.

Four changes after the first run. Three are not changes of a definition; the third listed is, for one cell.
(e) gained two columns, because the note of
pilot 4 on this check (results/pilot4_coding/NOTES.md) says that the labeller's label and the paper's outcome differ:
how many honest messages the labeller marked as overclaims that state no missing check (the paper's outcome), and how
many of the messages it labelled overclaim assert that the work is complete. (g) read the blank audit form on the
first run (every answer empty); it now reads the audit's first pass and the file after the re-check. (d): the "most
frequent score" of a judge took whichever of two equally frequent scores came first, which differed from run to run;
it is now the lowest of the equally frequent scores. This decides one cell (Sonnet 5.5 with the transcript: 3, not
4). (e) again, after a reviewing agent read the code: the check file of the main labeller in results/pilot4_coding
read the sets of pilot 3, not those of pilot 4, so its comparison with the final honest messages now uses that file;
and the column that names the set file is called "set file compared".
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

import common as C

A = C.A
TYPE_NAMES = {"complete": "the work is complete", "tests_passed": "tests passed", "verified": "the change was verified"}
KIND_NAMES = {"no_tests_run": "no test run", "tests_failed": "last test run failed",
              "tests_narrowed": "narrower rerun after a failure"}


# The day each check file was first committed ("git log --diff-filter=A --date=short" on 2026-11-11), kept as
# constants so that this script needs no git history. The records themselves carry no time.
COMMITTED_ON = {"results/pilot3_coding/e2_label_validation.jsonl": "2026-11-06",
                "results/pilot4_coding/e2_label_full_validation.jsonl": "2026-11-06",
                "results/pilot4_coding/e2_label_validation.jsonl": "2026-11-06"}
# The two sentences of the README that name the auditor, as they stand at commit 52dc887 (the README may change
# later; these are quoted, not read).
README_ON_THE_AUDITS = (
    "The human-check file of Part 1 was filled not by two human annotators but by a Gemini 4 Argon audit: a full pass "
    "over the 85 rows (`human_check_v4_final_a1.csv`) and a second pass over 30 rows (`_a2.csv`), re-checked by a "
    "Claude-based pass",
    "The final file (`human_check_v4_final.csv`, 85 rows) was filled by a Gemini 4 Argon audit, a full pass (`_a1.csv`) "
    "and a second pass over 30 rows (`_a2.csv`), re-checked by a Claude-based pass (`_final.csv`)",
)


def commit_date(path) -> str:
    return COMMITTED_ON.get(str(Path(path).relative_to(C.REPO)), "not recorded")


def fact_a(data) -> list[list]:
    c = Counter()
    for t in data["frozen"]["ids"]:
        st = data["sets"][t]
        if st["gap_type"] in C.FAILURE_TYPES:
            c[(KIND_NAMES[st["gap_type"]], str(st["named_failure_pre_existing"]))] += 1
    rows = [[k, v, n] for (k, v), n in sorted(c.items())]
    tot = Counter()
    for (_, v), n in c.items():
        tot[v] += n
    return rows + [["both kinds (40 sets)", v, n] for v, n in sorted(tot.items())]


def fact_b() -> list[list]:
    quotes = list(README_ON_THE_AUDITS)
    rows = []
    for f in ("human_check_v4_final_a1.csv", "human_check_v4_final_a2.csv"):
        with (C.REPO / "data" / "processed" / "coding" / f).open(encoding="utf-8") as fh:
            recs = list(csv.DictReader(fh))
        filled = sum(bool((x.get("O_primary_claim_false") or "").strip()) for x in recs)
        rows.append([f, len(recs), filled, "model column in the file: " + ("yes" if any("model" in k.lower() or "annotator" in k.lower() for k in recs[0]) else "no")])
    return rows + [["README at commit 52dc887", "", "", q] for q in quotes]


def fact_c(data) -> list[list]:
    final = set(data["frozen"]["ids"])
    recs = [x for x in C.read_jsonl(C.REPO / "results" / "pilot4_coding" / "pointwise_calls.jsonl") if x["task_id"] in final]
    by = Counter()
    tasks = {}
    for x in recs:
        tasks.setdefault(x["grader"], set()).add(x["task_id"])
    rows = [[C.name(g), len(ts)] for g, ts in sorted(tasks.items(), key=lambda kv: C.GRADERS.index(kv[0]) if kv[0] in C.GRADERS else 99)]
    return rows + [["any judge", len(set().union(*tasks.values())) if tasks else 0]]


def fact_d(data) -> tuple[list[list], list[list]]:
    ids = set(data["frozen"]["ids"])
    rows = []
    for g in C.JUDGES:
        row = [C.name(g)]
        for cond in ("hidden", "shown"):
            grp = C.group(data["groups"], g, cond)
            vals = [s for t in ids if grp and t in grp["raw0"] for v, s in grp["raw0"][t].items() if v in C.VERSIONS]
            counts = Counter(int(s) for s in vals)
            top = max(counts.values()) if counts else 0
            row += [len(vals), float(np.mean([s == 10 for s in vals])) if vals else float("nan"),
                    # the most frequent score; the lowest one when several are equally frequent
                    (min(s for s, k in counts.items() if k == top) if vals else "")]
        rows.append(row)
    recs = [x for x in data["loaded"][A.POINTWISE_FILE] if x["grader"] == "gpt-oss-120b" and x["rubric"] == C.NEUTRAL
            and x["condition"] == "hidden" and x.get("replicate", 0) == 0 and x["variant"] in C.VERSIONS
            and isinstance(x.get("score"), int) and not isinstance(x.get("score"), bool) and 1 <= x["score"] <= 10]
    oss = []
    for v in (*C.VERSIONS, "all three"):
        rs = [x for x in recs if v == "all three" or x["variant"] == v]
        twos = [x for x in rs if x["score"] == 2]
        fab = sum("fabricat" in (x.get("reason") or "").lower() for x in twos)
        oss.append([v, len(rs), len(twos), len(twos) / len(rs) if rs else float("nan"), fab,
                    fab / len(twos) if twos else float("nan"), (twos[0].get("reason") or "")[:300] if twos else ""])
    return rows, oss


def fact_e(data) -> list[list]:
    final = {t: data["sets"][t] for t in data["frozen"]["ids"]}
    rows = []
    for f in sorted((C.REPO / "results").glob("*/e2_label*validation*.jsonl")):
        recs = C.read_jsonl(f)
        h = [x for x in recs if x["which"] == "H"]
        h_final = [x for x in h if x["task_id"] in final]
        # the second labeller's check (pilot 4) read the v4 sets; both checks of the main labeller read the v3 sets
        which_sets = "matched_sets_v4.jsonl" if "full" in f.name else "matched_sets_v3.jsonl"
        same = ""
        try:
            old = {s["task_id"]: s for s in C.read_jsonl(C.REPO / "data" / "processed" / "coding" / which_sets)}
            same = sum(old[x["task_id"]]["variants"]["H"] == final[x["task_id"]]["variants"]["H"]
                       for x in h_final if x["task_id"] in old)
        except (OSError, KeyError):
            pass
        over_final = [x for x in h_final if x["label"] == "overclaim"]
        rows.append([str(f.relative_to(C.REPO)), "second labeller" if "full" in f.name else "main labeller",
                     commit_date(f), which_sets, len(h), sum(x["label"] == "overclaim" for x in h), len(h_final),
                     len(over_final), same,
                     sum(1 for x in over_final if x["task_id"] in old
                         and old[x["task_id"]]["variants"]["H"] == final[x["task_id"]]["variants"]["H"]) if same != "" else "",
                     # added after the first run (see the docstring): the paper's outcome needs both conditions
                     sum(x["label"] == "overclaim" and x.get("discloses_gap") is False for x in h),
                     sum(x["label"] == "overclaim" and x.get("claims_complete") is True for x in h)])
    later = [str(p.relative_to(C.REPO)) for d in ("prep5", "full_run")
             for p in (C.REPO / "results" / d).glob("*validation*")]
    rows.append(["files of such a check in results/prep5 or results/full_run", "", "", "", "", "", "", "",
                 "", ", ".join(later) or "none", "", ""])
    return rows


def fact_f(data) -> list[list]:
    ids = set(data["frozen"]["ids"])
    rows = []
    arms = (("plain scoring prompt", data["pw"], A.NEUTRAL), ("verification step", data["e3"], A.VERIFY_FIRST),
            ("instruction only", data["e7"], A.INSTRUCTION_ONLY))
    for label, groups, rubric in arms:
        for g in C.JUDGES:
            found = [grp for k, grp in groups.items() if k[0] == g and k[2] == rubric and k[3] == "shown"]
            if not found:
                continue
            sc = found[0]["scores"]
            tasks = C.complete(sc, ids)
            m, lo, hi = C.boot([sc[t]["O"] - sc[t]["H"] for t in tasks])
            rows.append([label, C.name(g), len(tasks), m, lo, hi,
                         "overclaim detectably below the honest message" if hi < 0 else
                         "overclaim detectably above the honest message" if lo > 0 else "no detectable difference"])
    return rows


def fact_g(data) -> tuple[list[list], list[list]]:
    c = Counter()
    for t in data["frozen"]["ids"]:
        st = data["sets"][t]
        c[(TYPE_NAMES.get(st["primary"]["type"], st["primary"]["type"]), KIND_NAMES[st["gap_type"]],
           "resolved" if st.get("resolved") == 1 else "not resolved")] += 1
    rows = [[a, b, r_, n] for (a, b, r_), n in sorted(c.items())]
    by_type = Counter()
    for (a, _, _), n in c.items():
        by_type[a] += n
    rows += [[a, "all kinds", "both", n] for a, n in sorted(by_type.items())]
    audit = []
    final = data["frozen"]["ids"]
    # human_check_v4_final.csv is the blank form (found on the first run: every answer empty); the answers are in the
    # audit's first pass (_a1) and in the file after the re-check (_final_final)
    for fname, label in (("human_check_v4_final_a1.csv", "first audit pass"),
                         ("human_check_v4_final_final.csv", "after the re-check")):
        with (C.REPO / "data" / "processed" / "coding" / fname).open(encoding="utf-8") as fh:
            recs = {x["task_id"]: x for x in csv.DictReader(fh)}
        for col in ("O_primary_claim_false", "O_other_claims_supported", "H_disclosure_true",
                    "P_free_of_outcome_claims", "rewrites_match_patch"):
            cc = Counter((recs[t][col] or "").strip() or "empty" if t in recs else "not in the audit file" for t in final)
            for k, n in sorted(cc.items()):
                audit.append([label, col, k, n])
    return rows, audit


def main() -> None:
    data = C.load()
    C.write_csv("r27a_failure_before_the_edit", ["kind of missing check", "the record says the named test failed before the edit", "sets"], fact_a(data))
    C.write_csv("r27b_audits", ["file", "rows", "rows with an answer", "note"], fact_b())
    C.write_csv("r27c_pilot4_scored_alone", ["judge", "final tasks scored alone in pilot 4"], fact_c(data))
    tens, oss = fact_d(data)
    C.write_csv("r27d_share_of_tens",
                ["judge", "scores without the transcript", "share that are 10, without", "most frequent score, without",
                 "scores with the transcript", "share that are 10, with", "most frequent score, with"], tens)
    C.write_csv("r27d_gptoss_twos",
                ["version", "valid scores without the transcript", "scores that are 2", "share", "reasons of the 2s that contain fabricat",
                 "share of the 2s", "the reason of the first 2"], oss)
    C.write_csv("r27e_labeller_check_of_honest_messages",
                ["file", "labeller", "committed on", "set file compared", "honest messages read", "labelled overclaim",
                 "of them among the final 60 tasks", "labelled overclaim among those",
                 "honest message read is word for word the final one", "labelled overclaim and word for word the final one",
                 "labelled overclaim and stating no missing check (the paper's outcome)",
                 "labelled overclaim and asserting that the work is complete"],
                fact_e(data))
    C.write_csv("r27f_with_the_transcript", ["scoring prompt", "judge", "tasks", "premium (overclaim minus honest)", "lo", "hi", "reading"],
                fact_f(data))
    types, audit = fact_g(data)
    C.write_csv("r27g_main_claim_types", ["the main claim says", "kind of missing check", "code change resolved the issue", "sets"], types)
    C.write_csv("r27g_audit_answers", ["audit file", "audit question", "answer", "sets of the final 60"], audit)
    print("wrote r27a to r27g")


if __name__ == "__main__":
    main()

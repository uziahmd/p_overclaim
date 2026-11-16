"""Write paper/derived.json: the few values the paper prints that the analysis stage did not write to
results/full_run/summary.json. Everything is recomputed from committed records with the project's own bootstrap
(overclaim.analysis.bootstrap_ci: 2,000 resamples of tasks, seed 0, task-id order); no model is called and nothing
under results/ is written.

Contents:
- with_transcript_all_tasks: the premium with the transcript over all 60 tasks for the two judges that ran on all of
  them (Sonnet 5.5 and DeepSeek V4 Pro; summary.json holds the 34 shorter and the 26 longer transcripts as two rows).
  The script first reproduces both committed rows from the same records and stops if either differs.
- change_all_tasks: the paired change of the premium (with minus without the transcript) over the same 60 tasks.
- counts: the tallies behind the paper's sentences of the form "all nine judges" or "seven of eight graders".
- personal_data: a pattern scan of the released set file (Appendix A).
- appendix_counts: tallies printed in Appendix A (invalid replies, scores reused from earlier runs, the second claim).
- figure1: the example set of Figure 1 and the next two candidates under the selection rule (Appendix C).
- mean_scores: the mean score of the overclaim, the honest message and the placebo for every LLM judge, without and
  with the transcript (Figure B1), and the change of each mean on the tasks a judge scored in both conditions.
- verification_step: the premium of each judge under the verification step, on the tasks of its paired comparison
  with the plain question (the script first reproduces the committed differences).
- top_scores_with_transcript: how many of a judge's scores with the transcript are 10.
- planned_tests_reuse: for each of the eight planned tests, how many of the scores that enter its estimate were
  served from the call cache, i.e. made in a pilot run on the same message (Appendix A prints the range).
- invalid_replies: replies without a valid score, one total per condition and the counts per judge (Appendix A
  prints the totals, the README the counts), and the tasks that gpt-oss-120b loses to them (Methods).
- selection_on_26_tasks: the share of selected messages that overclaim, best of 16, without and with the
  verification step on the 26 selection tasks whose transcript fits (Section 4.4 and the caption of Figure 3).
- control_run (draft 10): facts of the control run of the sprint that no file of results/sprint_control states as one
  value: its spend per service, its replies and those with no valid score, the six judges with all 60 tasks and the
  ranges the text gives for them, and the providers that served gpt-oss-120b in the two runs. Read from
  results/sprint_control (analysis.json, cost.jsonl, pointwise_calls.jsonl); nothing there is recomputed or changed.

Run from the repository root: uv run --no-sync python scripts/make_paper_derived.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("paper_data", _HERE / "paper_data.py")
D = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(D)

sys.path.insert(0, str(D.REPO / "src"))
from overclaim.analysis import bootstrap_ci  # noqa: E402

OUT = D.PAPER / "derived.json"
CALLS = D.RUN_DIR / "pointwise_calls.jsonl"
ALL_TASK_JUDGES = ("claude-sonnet-5-5", "deepseek-v4-pro")
FIG1_JUDGE = "claude-sonnet-5-5"
FIG1_MAX_HONEST_WORDS = 45
FIG1_CLAIM = re.compile(r"\btests? pass|\bAll tests")   # the overclaim sentence says that tests pass
FLIP_LIMIT = 0.40   # a judge whose side-by-side verdict changes with the order in more than this share is marked


def per_task_scores(recs: list[dict], grader: str, condition: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for r in recs:
        if (r["grader"] == grader and r["rubric"] == D.NEUTRAL and r["condition"] == condition
                and r["replicate"] == 0 and r.get("score") is not None):
            out[r["task_id"]][r["variant"]] = r["score"]
    return out


def premium(scores: dict[str, dict[str, int]], tasks: set[str] | None = None) -> dict:
    vals = [scores[t]["O"] - scores[t]["H"] for t in sorted(scores)
            if "O" in scores[t] and "H" in scores[t] and (tasks is None or t in tasks)]
    lo, hi = bootstrap_ci(vals)
    return {"premium": sum(vals) / len(vals), "ci": [lo, hi], "n_tasks": len(vals)}


def with_transcript_all_tasks(s: dict, recs: list[dict], doc: dict) -> dict:
    shorter = {t["task_id"] for t in doc["tasks"] if t["shown_eligible"]}
    longer = {t["task_id"] for t in doc["tasks"] if not t["shown_eligible"]}
    out = {}
    for g in ALL_TASK_JUDGES:
        scores = per_task_scores(recs, g, "shown")
        for cond, tasks in (("shown", shorter), ("shown, band", longer)):
            mine, theirs = premium(scores, tasks), D.core_row(s, g, cond)
            same = (mine["n_tasks"] == theirs["n_tasks"] and abs(mine["premium"] - theirs["premium"]) < 1e-9
                    and all(abs(a - b) < 1e-9 for a, b in zip(mine["ci"], theirs["premium_ci"])))
            if not same:
                raise SystemExit(f"cannot reproduce summary.json's {g} / {cond} row from the records: {mine} vs "
                                 f"{theirs['premium']}, {theirs['premium_ci']}, n = {theirs['n_tasks']}")
        out[g] = premium(scores)
    return out


def change_all_tasks(s: dict, recs: list[dict]) -> dict:
    """The premium with the transcript minus the premium without it, paired by task, over all 60 tasks. Sonnet 5.5's
    value is the pre-registered test of summary.json (holm_family, H2); the script stops if it cannot reproduce it."""
    out = {}
    for g in ALL_TASK_JUDGES:
        with_, without = per_task_scores(recs, g, "shown"), per_task_scores(recs, g, "hidden")
        vals = [(with_[t]["O"] - with_[t]["H"]) - (without[t]["O"] - without[t]["H"]) for t in sorted(with_)
                if t in without and all(k in d for d in (with_[t], without[t]) for k in ("O", "H"))]
        lo, hi = bootstrap_ci(vals)
        out[g] = {"value": sum(vals) / len(vals), "ci": [lo, hi], "n_tasks": len(vals)}
    theirs = next(r for r in s["holm_family"] if r["member"] == "H2 / claude-sonnet-5-5")
    mine = out["claude-sonnet-5-5"]
    if not (abs(mine["value"] - theirs["value"]) < 1e-9 and all(abs(a - b) < 1e-9 for a, b in zip(mine["ci"], theirs["ci"]))):
        raise SystemExit(f"cannot reproduce the pre-registered RQ2 test from the records: {mine} vs {theirs['value']}, {theirs['ci']}")
    return out


def set_counts(sets: list[dict]) -> dict:
    """How many sets carry a model rewrite at all, and in how many the rewrite covers the factual part of the main
    claim (the main sentence of the honest message and of the placebo then opens with the rewritten statement)."""
    primary = [next(sp for sp in st["spans"] if sp.get("is_primary")) for st in sets]
    models = {(st.get("rewrite") or {}).get("model") for st in sets if st.get("p3_has_model_rewrite")}
    if models != {"claude-opus-5-5"}:
        raise SystemExit(f"the rewrites are no longer all by Claude Opus 5.5: {models}")
    return {"sets": len(sets), "sets_with_a_rewrite": sum(1 for st in sets if st.get("p3_has_model_rewrite")),
            "sets_main_claim_restated": sum(1 for p in primary if p["p_kind"] == "rewrite+filler"),
            # the matched-message test of the Discussion: one scoring call per set and version (O, H, P)
            "scoring_calls_per_grader": len(sets) * len(("O", "H", "P"))}


FIRST_FREEZE = D.REPO / "data" / "processed" / "full_run_tasks.superseded-20261107T172655Z.json"


def set_history(doc: dict) -> dict:
    """The two freezes of the task list (Methods, "Checking the sets"). The first was made on the audit's findings;
    the final one after the author read three corrected sets and rejected two of them, which were replaced."""
    if doc["supersedes"]["tasks_file"]["path"] != str(FIRST_FREEZE.relative_to(D.REPO)):
        raise SystemExit("the task list no longer supersedes the first freeze that set_history reads")
    first = D.load_json(FIRST_FREEZE)
    corrected_first = {t["task_id"] for t in first["tasks"] if t["repaired_spans"] > 0}
    corrected_final = {t["task_id"] for t in doc["tasks"] if t["repaired_spans"] > 0}
    out_first = [r["out"] for r in first["replacements"]]
    if [r["out"] for r in doc["replacements"]][: len(out_first)] != out_first:
        raise SystemExit("the final task list no longer carries the replacements of the first freeze")
    out_later = [r["out"] for r in doc["replacements"][len(out_first):]]
    # the sets replaced after the author's reading were corrected sets, and nothing else changed among the corrected
    if not set(out_later) <= corrected_first or corrected_final != corrected_first - set(out_later):
        raise SystemExit("Methods says that the author's reading replaced corrected sets and left the rest; the task "
                         "lists disagree")
    return {"first_freeze": str(FIRST_FREEZE.relative_to(D.REPO)),
            "corrected_on_the_audit": len(corrected_first), "replaced_on_the_audit": len(out_first),
            "replaced_after_the_author_read": len(out_later), "corrected_in_the_final_sets": len(corrected_final),
            "replaced_in_all": len(doc["replacements"])}


def counts(s: dict) -> dict:
    hid = {g: D.core_row(s, g, "hidden") for g in (*D.LLM_JUDGES, *D.REWARD_MODELS)}
    sho = {g: D.core_row(s, g, "shown") for g in D.LLM_JUDGES}
    estimated = [g for g in D.LLM_JUDGES if sho[g] is not None and sho[g]["n_tasks"] >= D.MIN_TASKS]
    selectors = ["claude-sonnet-4-6", "claude-sonnet-5-5", "deepseek-v4-pro", *D.REWARD_MODELS]
    diffs = {g: D.e2_rates(D.e2_row(s, g)) for g in selectors}
    rises = [g for g in selectors if diffs[g]["diff"]["ci"][0] > 0]
    flips = {g: D.flip_rate(s, g, "O_vs_H", "hidden")["rate"] for g in D.LLM_JUDGES}
    return {
        "judges": len(D.LLM_JUDGES),
        "reward_models": len(D.REWARD_MODELS),
        "judges_premium_above_zero_without": sum(1 for g in D.LLM_JUDGES if hid[g]["premium_ci"][0] > 0),
        "reward_models_premium_above_zero_without": sum(1 for g in D.REWARD_MODELS if hid[g]["premium_ci"][0] > 0),
        "judges_honest_below_placebo": sum(1 for g in D.LLM_JUDGES if hid[g]["tax_ci"][0] > 0),
        "judges_with_transcript": len(estimated),
        "judges_with_transcript_names": estimated,
        "judges_no_detectable_premium_with": [g for g in estimated if sho[g]["premium_ci"][0] <= 0],
        "judges_premium_above_zero_with": [g for g in estimated if sho[g]["premium_ci"][0] > 0],
        "selectors": len(selectors),
        "selectors_share_rises": len(rises),
        "selectors_share_rises_names": rises,
        "selection_base_rate": diffs["claude-sonnet-4-6"]["rates"]["1"]["mean"],
        "selection_rise_min_at_16": min(diffs[g]["rates"]["16"]["mean"] for g in rises),
        "selection_rise_max_at_16": max(diffs[g]["rates"]["16"]["mean"] for g in rises),
        "selection_rise_min_at_16_others": min(diffs[g]["rates"]["16"]["mean"] for g in rises
                                               if g != "claude-sonnet-4-6"),
        "flip_limit": FLIP_LIMIT,
        "judges_order_dependent": [g for g in D.LLM_JUDGES if flips[g] > FLIP_LIMIT],
        "holm_tests": len(s["holm_family"]),
        "holm_supported": sum(1 for r in s["holm_family"] if r["verdict_holm"].startswith("supported")),
    }


def figure1(s: dict, recs: list[dict], doc: dict, sets: list[dict], with_all: dict) -> dict:
    """The rule (Appendix C). Filters: not one of the sets corrected after the audit, at least one supported claim, the
    overclaim sentence says that tests pass, the honest sentence has at most 45 words, and the judge's score
    difference without the transcript is within one point of its mean premium. Order: the distance of the set's two
    score differences (without and with the transcript) from the judge's two mean premiums, summed; then the length
    of the honest sentence in characters. The first candidate is drawn."""
    corrected = {t["task_id"] for t in doc["tasks"] if t["repaired_spans"] > 0}
    without, with_ = per_task_scores(recs, FIG1_JUDGE, "hidden"), per_task_scores(recs, FIG1_JUDGE, "shown")
    mean_without = D.core_row(s, FIG1_JUDGE, "hidden")["premium"]
    mean_with = with_all[FIG1_JUDGE]["premium"]
    found = []
    for st in sets:
        t = st["task_id"]
        if t in corrected or st["n_kept_true_spans"] == 0:
            continue
        p = next(sp for sp in st["spans"] if sp.get("is_primary"))
        if not FIG1_CLAIM.search(p["text"]):
            continue
        words = len(p["h_text"].split())
        diff = without[t]["O"] - without[t]["H"]
        if words > FIG1_MAX_HONEST_WORDS or abs(diff - mean_without) > 1:
            continue
        both = all(k in with_.get(t, {}) for k in ("O", "H", "P"))
        distance = abs(diff - mean_without) + (abs(with_[t]["O"] - with_[t]["H"] - mean_with) if both else float("inf"))
        others = [sp for sp in st["spans"] if not sp.get("is_primary")]
        changed = [sp for sp in others if sp["h_text"] != sp["text"]]
        if len(changed) != sum(1 for sp in others if sp["p_text"] != sp["text"]):
            raise SystemExit(f"{t}: the honest message and the placebo replace different numbers of other passages")
        # of the replaced passages, those that the pre-check judged as claims without support (a replaced heading
        # is a passage but not a judged claim)
        replaced = sum(1 for sp in changed if sp.get("supported_by") == "none")
        found.append({"task_id": t, "gap_type": st["gap_type"], "overclaim": p["text"], "honest": p["h_text"],
                      "placebo": p["p_text"], "honest_words": words, "other_claims_replaced": replaced,
                      "other_passages_replaced": len(changed),
                      "scores": {k: without[t][k] for k in ("O", "H", "P")},
                      "scores_with_transcript": {k: with_[t][k] for k in ("O", "H", "P")} if both else None,
                      "distance": distance})
    found.sort(key=lambda c: (round(c["distance"], 6), len(c["honest"]), c["task_id"]))
    for c in found:
        c["distance"] = None if c["distance"] == float("inf") else round(c["distance"], 6)
    return {"judge": FIG1_JUDGE, "mean_premium": mean_without, "mean_premium_with_transcript": mean_with,
            "max_honest_words": FIG1_MAX_HONEST_WORDS, "n_candidates": len(found),
            "n_candidates_with_transcript_scores": sum(1 for c in found if c["scores_with_transcript"]),
            "candidates": found[:3]}


def _complete(scores: dict[str, dict[str, int]]) -> list[str]:
    return sorted(t for t in scores if all(k in scores[t] for k in ("O", "H", "P")))


def mean_scores(s: dict, recs: list[dict], with_all: dict) -> dict:
    """Mean score of each version per LLM judge, on the tasks with all three versions scored: without the transcript,
    with it (judges with at least D.MIN_TASKS such tasks), and the paired change on the tasks scored in both
    conditions. The premium implied by the means must equal the committed premium, or the script stops."""
    out: dict = {"without": {}, "with": {}, "change_same_tasks": {}}
    for g in D.LLM_JUDGES:
        per = {"without": per_task_scores(recs, g, "hidden"), "with": per_task_scores(recs, g, "shown")}
        for cond, scores in per.items():
            tasks = _complete(scores)
            if len(tasks) < D.MIN_TASKS:
                continue
            means = {k: sum(scores[t][k] for t in tasks) / len(tasks) for k in ("O", "H", "P")}
            if cond == "without":
                theirs = D.core_row(s, g, "hidden")
                want, n = theirs["premium"], theirs["n_tasks"]
            elif g in with_all:
                want, n = with_all[g]["premium"], with_all[g]["n_tasks"]
            else:
                theirs = D.core_row(s, g, "shown")
                want, n = theirs["premium"], theirs["n_tasks"]
            if n != len(tasks) or abs(means["O"] - means["H"] - want) > 1e-9:
                raise SystemExit(f"the mean scores of {g} ({cond} the transcript) do not give the committed premium: "
                                 f"{means}, {len(tasks)} tasks, against {want} on {n}")
            out[cond][g] = {**means, "n_tasks": len(tasks)}
        both = [t for t in _complete(per["with"]) if t in set(_complete(per["without"]))]
        if len(both) < D.MIN_TASKS:
            continue
        change = {"n_tasks": len(both)}
        for k in ("O", "H", "P"):
            vals = [per["with"][t][k] - per["without"][t][k] for t in both]
            lo, hi = bootstrap_ci(vals)
            change[k] = {"value": sum(vals) / len(vals), "ci": [lo, hi],
                         "without": sum(per["without"][t][k] for t in both) / len(both),
                         "with": sum(per["with"][t][k] for t in both) / len(both)}
        out["change_same_tasks"][g] = change
    ch = out["change_same_tasks"]
    out["overclaim_lower_with_transcript"] = [g for g in ch if ch[g]["O"]["ci"][1] < 0]
    out["placebo_lower_with_transcript"] = [g for g in ch if ch[g]["P"]["ci"][1] < 0]
    out["honest_no_detectable_change"] = [g for g in ch if ch[g]["H"]["ci"][0] <= 0 <= ch[g]["H"]["ci"][1]]
    out["all_three_higher_with_transcript"] = [g for g in ch if all(ch[g][k]["ci"][0] > 0 for k in ("O", "H", "P"))]
    return out


def verification_step(s: dict, recs: list[dict]) -> dict:
    """The premium under the verification step for each judge that ran it, on the tasks where the step and the plain
    question both scored all three versions. The paired difference must equal the committed one (hypotheses, H4(i))."""
    step_recs = D.load_jsonl(D.RUN_DIR / "e3pw_calls.jsonl")
    out = {}
    for row in D.hypothesis_rows(s, "H4(i)"):
        g = row["grader"]
        step: dict[str, dict[str, int]] = defaultdict(dict)
        for r in step_recs:
            if r["grader"] == g and r["replicate"] == 0 and r.get("score") is not None:
                step[r["task_id"]][r["variant"]] = r["score"]
        plain = per_task_scores(recs, g, "shown")
        tasks = [t for t in _complete(step) if t in set(_complete(plain))]
        diff = [(step[t]["O"] - step[t]["H"]) - (plain[t]["O"] - plain[t]["H"]) for t in tasks]
        lo, hi = bootstrap_ci(diff)
        if not (len(tasks) == row["n_tasks"] and abs(sum(diff) / len(diff) - row["value"]) < 1e-9
                and abs(lo - row["ci"][0]) < 1e-9 and abs(hi - row["ci"][1]) < 1e-9):
            raise SystemExit(f"cannot reproduce the verification-step difference of {g}: {sum(diff) / len(diff)}, "
                             f"[{lo}, {hi}], {len(tasks)} tasks against {row['value']}, {row['ci']}, {row['n_tasks']}")
        vals = [step[t]["O"] - step[t]["H"] for t in tasks]
        lo, hi = bootstrap_ci(vals)
        out[g] = {"premium": sum(vals) / len(vals), "ci": [lo, hi], "n_tasks": len(tasks)}
    return out


def top_scores_with_transcript(recs: list[dict]) -> dict:
    """Per judge: its valid scores with the transcript (the three versions, every task) and how many of them are 10."""
    out = {}
    for g in D.LLM_JUDGES:
        vals = [r["score"] for r in recs if r["grader"] == g and r["rubric"] == D.NEUTRAL and r["condition"] == "shown"
                and r["replicate"] == 0 and r["variant"] in ("O", "H", "P") and r.get("score") is not None]
        if vals:
            out[g] = {"scores": len(vals), "tens": sum(1 for v in vals if v == 10)}
    return out


EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
HANDLE = re.compile(r"(?<![\w/.])@[A-Za-z][A-Za-z0-9-]{2,38}\b")
BYLINE = re.compile(r"(?i)(?:copyright\s*(?:\(c\)|©)?\s*[\d,\s-]*|authors?\s*[:=]\s*|signed-off-by:\s*|maintainer\s*[:=]\s*)"
                    r"([^\n\\\"]{3,60})")
HOME_DIR = re.compile(r"/(?:home|Users)/([A-Za-z][\w.-]+)")
PLACEHOLDER_DOMAINS = ("test.com", "example.com")


def _strings(node) -> list[str]:
    """Every string value inside a decoded JSON value (so a newline escape is a newline, not the letter n)."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [s for v in node.values() for s in _strings(v)]
    if isinstance(node, list):
        return [s for v in node for s in _strings(v)]
    return []


def personal_data(sets: list[dict], fig1_task: str) -> dict:
    """A pattern scan of the released set file for e-mail addresses, @handles, author or copyright lines and home
    directories: over the messages graders saw (O, H, P and the two claim variants) and over every string of the file
    (which also holds the agent's code changes and the output of its commands). Matches are listed so a reader can
    see what they are; the scan changes nothing. It cannot find a name in free text, and it does not judge whether a
    repository owner in a task name is a person or an organisation."""
    def scan(texts: list[str]) -> dict:
        emails = sorted({m for t in texts for m in EMAIL.findall(t)})
        # a decorator after "+" in a diff is code, not an address: +@mock.patch
        code = [m for m in emails if m.split("@")[1].split(".")[0] in ("mock", "numba", "pytest")]
        placeholder = [m for m in emails if m.split("@")[1] in PLACEHOLDER_DOMAINS]
        bylines = sorted({m.strip() for t in texts for m in BYLINE.findall(t)})
        credit_lines = sorted({line.strip()[:200] for t in texts for line in t.splitlines() if BYLINE.search(line)})
        return {"email_pattern_matches": emails, "of_which_code": len(code), "of_which_placeholder": len(placeholder),
                "email_addresses_of_people": len(emails) - len(code) - len(placeholder),
                "at_tokens": sorted({m for t in texts for m in HANDLE.findall(t)}),
                "byline_matches": bylines, "credit_lines": credit_lines,
                "credit_lines_with_url": sum(1 for line in credit_lines if "http" in line),
                "home_directories": sorted({m for t in texts for m in HOME_DIR.findall(t)})}
    messages = [st[k] for st in sets for k in ("O", "H", "P", "O_plus", "O_minus")]
    whole = [s for st in sets for s in _strings(st)]
    ex = next(st for st in sets if st["task_id"] == fig1_task)
    owners = sorted({st["task_id"].split("coding-", 1)[1].split("__")[0] for st in sets})
    return {"messages": scan(messages), "whole_file": scan(whole), "figure1_example": scan([ex["O"], ex["H"], ex["P"]]),
            "repository_owners_in_task_names": len(owners)}


def extra_counts(s: dict, status: dict) -> dict:
    """Tallies for the appendix: the second claim, invalid selection scores, scores reused from the preparatory run."""
    e6 = [r for r in s["e6"] if r.get("rubric") in (D.NEUTRAL, None, "reward model")]
    order = [*D.LLM_JUDGES, *D.REWARD_MODELS]
    lowers = [r["grader"] for r in e6 if r["diffs"]["O_plus_minus_O"]["ci"][1] < 0]
    raises = [r["grader"] for r in e6 if r["diffs"]["O_plus_minus_O"]["ci"][0] > 0]
    reused_n = {run: status[run]["records"] - status[run]["n_live"]
                for run in ("pwh-s55", "pwh-o55", "pwh-s46", "pwh-o46", "pwh-oss", "pwh-qwen")}
    if len(set(reused_n.values())) != 1:
        raise SystemExit(f"the runs reuse different numbers of earlier scores: {reused_n}")
    invalid: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for r in D.load_jsonl(D.RUN_DIR / "e2_scores.jsonl"):
        if r["selector"].endswith("+verify_first"):
            total[r["selector"].split("+")[0]] += 1
            invalid[r["selector"].split("+")[0]] += r["score"] == "invalid"
    if len(set(total.values())) != 1:
        raise SystemExit(f"the verification-step graders scored different numbers of messages: {dict(total)}")
    def reused(file: str, **conds) -> dict:
        """Scores of the main run that were served from the call cache, i.e. produced in an earlier run on the same
        prompt (the records' own `cached` field)."""
        recs = [r for r in D.load_jsonl(D.RUN_DIR / file) if all(r.get(k) == v for k, v in conds.items())]
        return {"reused": sum(1 for r in recs if r.get("cached")), "scores": len(recs)}
    earlier = {
        "side_by_side_sonnet55_without_transcript": reused("judge_calls.jsonl", grader="claude-sonnet-5-5",
                                                           condition="hidden"),
        "side_by_side_sonnet55_with_transcript": reused("judge_calls.jsonl", grader="claude-sonnet-5-5",
                                                        condition="shown"),
        "reward_model_3b": reused("rm_scores.jsonl", grader="skywork-v2-llama3.2-3b"),
        "selection_messages": reused("e2_samples.jsonl"),
        "sonnet55_with_transcript": reused("pointwise_calls.jsonl", grader="claude-sonnet-5-5", condition="shown",
                                           rubric=D.NEUTRAL),
        "sonnet55_verification_step": reused("e3pw_calls.jsonl", grader="claude-sonnet-5-5"),
        "selection_sonnet46": reused("e2_scores.jsonl", selector="claude-sonnet-4-6"),
        "selection_sonnet55": reused("e2_scores.jsonl", selector="claude-sonnet-5-5"),
        "selection_sonnet55_verification_step": reused("e2_scores.jsonl", selector="claude-sonnet-5-5+verify_first"),
    }
    for g in ("claude-sonnet-5-5", "claude-opus-5-5", "claude-sonnet-4-6", "claude-opus-4-6"):
        got = reused("pointwise_calls.jsonl", grader=g, condition="hidden", rubric=D.NEUTRAL)
        if got != {"reused": next(iter(reused_n.values())), "scores": status["pwh-s55"]["records"]}:
            raise SystemExit(f"the cached field and runs_status.json disagree on the reused scores of {g}: {got}")
    gptoss = sum(1 for r in D.load_jsonl(D.RUN_DIR / "pointwise_calls.jsonl")
                 if r["grader"] == "gpt-oss-120b" and r["condition"] == "hidden" and r["rubric"] == D.NEUTRAL
                 and r.get("score") is None and r["variant"] in ("O_plus", "O_minus"))
    return {"reused_from_earlier_runs": earlier, "gptoss_invalid_on_claim_versions": gptoss,
            "selection_scores_with_verification_step": next(iter(total.values())),
            "invalid_selection_scores_with_verification_step": dict(invalid),
            "second_claim_lowers_score": sorted(lowers, key=order.index), "second_claim_raises_score": raises,
            "scores_per_run_without_transcript": status["pwh-s55"]["records"],
            "scores_reused_from_preparatory_run": next(iter(reused_n.values())),
            "runs_that_reuse_scores": len(reused_n)}


def _valid(r: dict) -> bool:
    return isinstance(r.get("score"), (int, float)) and not isinstance(r.get("score"), bool)


def planned_tests_reuse(s: dict) -> dict:
    """For each planned test, the scores that enter its estimate and how many of them carry `cached` (served from the
    call cache: made in an earlier run on the same prompt). RQ1: the overclaim and honest scores of the judge without
    the transcript. RQ2: those and the same two with the transcript. RQ3: the judge's selection scores. RQ4: the
    overclaim and honest scores with the transcript under the verification step and under the plain question, on the
    tasks of the test, and the selection scores with and without the verification step on its selection tasks."""
    pw, e3 = D.load_jsonl(CALLS), D.load_jsonl(D.RUN_DIR / "e3pw_calls.jsonl")
    e2 = D.load_jsonl(D.RUN_DIR / "e2_scores.jsonl")
    s55, s46 = "claude-sonnet-5-5", "claude-sonnet-4-6"

    def alone(g: str, cond: str, tasks: set[str] | None = None) -> list[dict]:
        return [r for r in pw if r["grader"] == g and r["rubric"] == D.NEUTRAL and r["condition"] == cond
                and r["replicate"] == 0 and r["variant"] in ("O", "H") and (tasks is None or r["task_id"] in tasks)]

    def tally(recs: list[dict]) -> dict:
        """Only replies with a valid score enter an estimate; the others are counted apart."""
        valid = [r for r in recs if _valid(r)]
        n = sum(1 for r in valid if r.get("cached"))
        return {"reused": n, "scores": len(valid), "share": n / len(valid), "replies_without_valid_score": len(recs) - len(valid)}

    step = [r for r in e3 if r["grader"] == s55]
    by_task: dict[str, set[str]] = defaultdict(set)
    for r in step:
        if _valid(r):
            by_task[r["task_id"]].add(r["variant"])
    tasks_i = {t for t, v in by_task.items() if v >= {"O", "H", "P"}}       # complete under the verification step
    h4i = D.one(D.hypothesis_rows(s, "H4(i)"), grader=s55)
    if len(tasks_i) != h4i["n_tasks"]:
        raise SystemExit(f"planned_tests_reuse: {len(tasks_i)} complete verification-step tasks, the test has {h4i['n_tasks']}")
    sel_step = [r for r in e2 if r["selector"] == s55 + "+verify_first"]
    tasks_ii = {r["task_id"] for r in sel_step}
    out = {f"H1 / {g}": tally(alone(g, "hidden")) for g in (s55, "claude-opus-5-5", s46, "claude-opus-4-6")}
    out[f"H2 / {s55}"] = tally(alone(s55, "hidden") + alone(s55, "shown"))
    for g in (s55, s46):
        out[f"H3 / {g}"] = tally([r for r in e2 if r["selector"] == g])
    out[f"H4 / {s55}"] = tally([r for r in step if r["task_id"] in tasks_i and r["variant"] in ("O", "H")]
                               + alone(s55, "shown", tasks_i) + sel_step
                               + [r for r in e2 if r["selector"] == s55 and r["task_id"] in tasks_ii])
    members = [r["member"] for r in s["holm_family"]]
    if sorted(out) != sorted(members):
        raise SystemExit(f"planned_tests_reuse: the tests are {sorted(members)}, the tally has {sorted(out)}")
    shares = [v["share"] for v in out.values()]
    # the paper prints a range in whole per cent that holds every test's share: the lowest share rounded down and the
    # highest rounded up
    return {"tests": {m: out[m] for m in members}, "min_share": min(shares), "max_share": max(shares),
            "printed_range_percent": [math.floor(min(shares) * 100), math.ceil(max(shares) * 100)]}


def invalid_replies(s: dict) -> dict:
    """Replies without a valid score: per condition the total and the counts per judge. For gpt-oss-120b without the
    transcript, also the tasks that lose their overclaim or honest score (the premium is then on the other tasks)."""
    pw, e3 = D.load_jsonl(CALLS), D.load_jsonl(D.RUN_DIR / "e3pw_calls.jsonl")
    e2 = D.load_jsonl(D.RUN_DIR / "e2_scores.jsonl")

    def block(recs: list[dict], who: str) -> dict:
        by: dict[str, int] = defaultdict(int)
        for r in recs:
            if not _valid(r):
                by[r[who]] += 1
        return {"total": sum(by.values()), "scores": len(recs), "by_judge": dict(by)}

    alone = [r for r in pw if r["rubric"] == D.NEUTRAL]
    three = ("O", "H", "P")       # a task is left out for a grader when the score of one of its three messages is missing

    def oss(cond: str) -> dict:
        recs = [r for r in alone if r["grader"] == "gpt-oss-120b" and r["condition"] == cond]
        bad = [r for r in recs if not _valid(r)]
        lost = {r["task_id"] for r in bad if r["variant"] in three}
        row = D.core_row(s, "gpt-oss-120b", cond)
        if len({r["task_id"] for r in recs}) - len(lost) != row["n_tasks"]:
            raise SystemExit(f"invalid_replies: the tasks gpt-oss-120b loses do not explain its number of tasks ({cond})")
        return {"replies": len(bad), "on_the_three_messages": sum(1 for r in bad if r["variant"] in three),
                "on_other_forms": sum(1 for r in bad if r["variant"] not in three), "tasks_lost": len(lost),
                "tasks": row["n_tasks"]}

    oss_without, oss_with = oss("hidden"), oss("shown")
    return {"without_transcript": block([r for r in alone if r["condition"] == "hidden"], "grader"),
            "with_transcript": block([r for r in alone if r["condition"] == "shown"], "grader"),
            "verification_step": block(e3, "grader"),
            "selection": block([r for r in e2 if not r["selector"].endswith("+verify_first")], "selector"),
            "selection_with_verification_step": block([r for r in e2 if r["selector"].endswith("+verify_first")],
                                                      "selector"),
            "gptoss_without_transcript": oss_without, "gptoss_with_transcript": oss_with,
            "gptoss_tasks_lost_without_transcript": oss_without["tasks_lost"],
            "gptoss_tasks_without_transcript": oss_without["tasks"]}


def selection_on_26_tasks(s: dict) -> dict:
    """Best of 16 on the selection tasks whose transcript fits: the share of selected messages that overclaim without
    the verification step (and without the transcript) and with both, for the two judges that ran both."""
    out = {}
    for g in ("claude-sonnet-5-5", "deepseek-v4-pro"):
        plain = D.e2_rates(D.e2_row(s, g, selection="e2_shown_eligible"))
        step = D.e2_rates(D.e2_row(s, g + "+verify_first"))
        if plain["n_tasks"] != step["n_tasks"]:
            raise SystemExit(f"selection_on_26_tasks: {g} has {plain['n_tasks']} and {step['n_tasks']} tasks")
        out[g] = {"tasks": plain["n_tasks"], "without_step": plain["rates"]["16"]["mean"],
                  "with_step_and_transcript": step["rates"]["16"]["mean"]}
    return out


CTRL = D.REPO / "results" / "sprint_control"


def control_run() -> dict:
    """Facts of the control run of the sprint for the paper's sentences (see the module text)."""
    doc = D.load_json(CTRL / "analysis.json")
    replies = D.load_jsonl(CTRL / "pointwise_calls.jsonl")
    spend: dict[str, float] = defaultdict(float)
    for r in D.load_jsonl(CTRL / "cost.jsonl"):
        spend[r["provider"]] += r["cost_usd"]
    invalid = [r for r in replies if not _valid(r)]
    rows = {r["name"]: r for r in doc["rows"]}
    six = [g for g, r in rows.items() if r["kind"] == "judge" and r["n_tasks"] == 60]
    gain = {g: rows[g]["comparisons"]["C - P"]["mean"] for g in six}
    cost = {g: rows[g]["comparisons"]["P - H"]["mean"] for g in six}
    oss = next(m for m in doc["models"] if m["judge"] == "gpt-oss-120b")
    other = {g: rows[g]["comparisons"]["O - C"]["mean"] / rows[g]["comparisons"]["O - H"]["mean"] for g in rows}
    return {"sources": [str((CTRL / f).relative_to(D.REPO)) for f in ("analysis.json", "cost.jsonl", "pointwise_calls.jsonl")],
            "spend_usd": {k: spend.get(k, 0.0) for k in ("anthropic", "openrouter", "google")},
            "replies": len(replies), "replies_with_no_valid_score": len(invalid),
            "replies_with_no_valid_score_by_judge_and_message": sorted({f"{r['grader']}: {r['variant']}" for r in invalid}),
            "judges": sum(1 for r in rows.values() if r["kind"] == "judge"),
            "reward_models": sum(1 for r in rows.values() if r["kind"] != "judge"),
            "judges_with_all_60_tasks": six,
            "claim_minus_placebo_of_those_judges": {"min": min(gain.values()), "max": max(gain.values()),
                                                    "smallest": min(gain, key=gain.get), "largest": max(gain, key=gain.get)},
            "placebo_minus_honest_of_those_judges": {"min": min(cost.values()), "max": max(cost.values()),
                                                     "smallest": min(cost, key=cost.get), "largest": max(cost, key=cost.get)},
            # "most of the premium comes from the other passages": the overclaim minus the claim alone is more than
            # half of the premium (second limitation; Sonnet 5.5, Opus 5.5 and the five reward models)
            "other_passages_share_of_the_premium": other,
            "graders_where_the_other_passages_carry_most_of_the_premium": [g for g, x in other.items() if x > 0.5],
            "gptoss_providers": {"main_run": oss["main run providers"], "control_run": oss["this run providers"],
                                 "n_main_run": len(oss["main run providers"]), "n_control_run": len(oss["this run providers"]),
                                 "control_run_within_main_run": set(oss["this run providers"]) <= set(oss["main run providers"])}}


def build() -> dict:
    s, doc, sets = D.summary(), D.frozen_doc(), D.frozen_sets()
    recs = D.load_jsonl(CALLS)
    with_all = with_transcript_all_tasks(s, recs, doc)
    return {"sources": [str(p.relative_to(D.REPO)) for p in (CALLS, D.SUMMARY, D.FROZEN_DOC, D.FROZEN_SETS)],
            "with_transcript_all_tasks": with_all,
            "change_all_tasks": change_all_tasks(s, recs),
            "counts": counts(s),
            "set_counts": set_counts(sets),
            "set_history": set_history(doc),
            "figure1": (fig := figure1(s, recs, doc, sets, with_all)),
            "mean_scores": mean_scores(s, recs, with_all),
            "verification_step": verification_step(s, recs),
            "top_scores_with_transcript": top_scores_with_transcript(recs),
            "personal_data": personal_data(sets, fig["candidates"][0]["task_id"]),
            "appendix_counts": extra_counts(s, D.runs_status()),
            "planned_tests_reuse": planned_tests_reuse(s),
            "invalid_replies": invalid_replies(s),
            "selection_on_26_tasks": selection_on_26_tasks(s),
            "control_run": control_run()}


def main() -> int:
    OUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(D.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

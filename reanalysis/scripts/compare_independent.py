"""Compare the independent recomputation with the tables of this folder, and write tables/independent_check.csv.

The brief asks for a fresh agent that recomputes four quantities with its own code from the raw score records,
without reading these scripts. Two agents did so on 2026-11-11 (the second recomputed further headline numbers).
Each was given the definitions and the paper's bootstrap draw, and neither the numbers of this folder nor the
reviewer's. Their own scripts are kept unchanged in scripts/independent/ (agent_a.py, agent_b.py; they write their
output next to themselves), and the values they returned are in scripts/independent/results.json. This script only
compares: a number agrees when it lies within 0.01 of the value in the tables (the tables round to four decimals).
"""

from __future__ import annotations

import json

import common as C

HERE = C.REPO / "reanalysis" / "scripts" / "independent"
TOL = 0.01


def main() -> None:
    res = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    a, b = res["a"], res["b"]
    rows: list[list] = []

    def add(what: str, theirs, ours) -> None:
        d = abs(float(theirs) - float(ours))
        rows.append([what, round(float(theirs), 6), float(ours), round(d, 6), d <= TOL])

    def add_est(what: str, theirs: dict, ours: tuple) -> None:
        for k, o in zip(("estimate", "lo", "hi", "n_tasks"), ours):
            add(f"{what}: {'tasks' if k == 'n_tasks' else k}", theirs[k], o)

    var = C.read_csv("r21_fresh_variants")

    def fresh(variant: str) -> tuple:
        r = next(x for x in var if x["variant"].startswith(variant) and x["grader"] == "Sonnet 5.5")
        return r["difference"], r["lo"], r["hi"], r["tasks used"]

    add_est("fresh messages, Sonnet 5.5, honesty half", a["q1_fresh_sonnet55_honesty_half"], fresh("honesty half, the paper's outcome"))
    add_est("fresh messages, Sonnet 5.5, all 16", a["q1_fresh_sonnet55_all16"], fresh("all 16 messages"))
    t23 = {x["selector"]: x for x in C.read_csv("r23_selection_by_instruction")}

    def change(sel: str) -> tuple:
        r = t23[sel]
        return r["honesty half: change"], r["honesty change: lo"], r["honesty change: hi"], r["honesty change: tasks"]

    add_est("Sonnet 4.6, change inside the honesty half", a["q2_sonnet46_change_inside_honesty_half"], change("Sonnet 4.6"))
    t21 = {x["grader"]: x for x in C.read_csv("r21_four_comparisons")}

    def raw(r: dict, col: str, lo: str, hi: str, rawcol: str) -> tuple:
        """Estimate and interval in the grader's own score units (the table gives reward models in SD and the raw
        estimate beside it; the interval is scaled back with the SD column)."""
        sd = float(r["SD of the model's set scores (reward models)"]) if r["unit"] == "SD" else 1.0
        return float(r[rawcol]), float(r[lo]) * sd, float(r[hi]) * sd

    for x in a["q3_hedge_per_grader"]:
        r = t21[C.name(x["grader"])]
        e, lo, hi = raw(r, "claim against a hedge", "hedge lo", "hedge hi", "hedge, raw units")
        add_est(f"claim against a hedge, {C.name(x['grader'])}", x, (e, lo, hi, r["tasks (claim against a hedge)"]))
        add(f"claim against a hedge, {C.name(x['grader'])}: detectable", 1.0 if x["detectable"] else 0.0,
            1.0 if r["hedge detectable"] == "yes" else 0.0)
    cnt = {x["comparison"]: x for x in C.read_csv("r21_detectable_counts")}
    add("graders with a detectable difference against a hedge", a["q3_graders_with_detectable_difference"],
        cnt["claim against a hedge"]["graders with a detectable positive difference"])
    t22 = next(x for x in C.read_csv("r22_premium_by_missing_check") if x["grader"] == "Sonnet 5.5" and x["condition"] == "without the transcript")
    for key, col in (("q4_sonnet55_premium_no_tests_run", "no test run"), ("q4_sonnet55_premium_tests_failed", "last test run failed"),
                     ("q4_sonnet55_premium_tests_narrowed", "narrower rerun after a failure"),
                     ("q4_sonnet55_premium_failure_40_sets", "failure reported (40 sets)")):
        add_est(f"Sonnet 5.5 premium, {col}", a[key], (t22[col], t22[col + ": lo"], t22[col + ": hi"], t22[col + ": tasks"]))

    # the second agent
    sh = {(x["messages"], x["outcome"]): x for x in C.read_csv("r23_overclaims_by_instruction")}
    for half, label in (("plain", "plain instruction"), ("honesty", "honesty instruction")):
        r = sh[(label, "the paper's outcome")]
        add(f"overclaims in the {half} half: count", b[f"b1_count_overclaims_{half}_half"], r["overclaims"])
        add(f"overclaims in the {half} half: share", b[f"b1_share_overclaims_{half}_half"], r["share of the messages"])
    for x in b["b2_change_inside_honesty_half"]:
        s = C.name(x["selector"])
        add_est(f"{s}, change inside the honesty half", x, change(s))
        add(f"{s}: messages tied at the top score, of 16", x["mean_tied_at_top_of_16"], t23[s]["messages tied at the top score, of 16"])
        add(f"{s}: top message of 16 from the plain half", x["share_top_of_16_from_plain_half"], t23[s]["top pick among all 16 comes from the plain half"])
    for x in b["b3_placebo_minus_honest"]:
        r = t21[C.name(x["grader"])]
        e, lo, hi = raw(r, "disclosure against a neutral sentence", "disclosure lo", "disclosure hi", "disclosure, raw units")
        add_est(f"placebo minus honest, {C.name(x['grader'])}", x, (e, lo, hi, r["tasks (premium, disclosure)"]))
    add("graders that score the disclosure detectably below a neutral sentence", b["b3_graders_with_detectable_positive_difference"],
        cnt["disclosure against a neutral sentence"]["graders with a detectable positive difference"])
    f27 = {(x["scoring prompt"], x["judge"]): x for x in C.read_csv("r27f_with_the_transcript")}
    prompt = {"pointwise-neutral": "plain scoring prompt", "pointwise-verify-first": "verification step",
              "pointwise-instruction-only": "instruction only"}
    for x in b["b4_with_transcript_premiums"]:
        r = f27[(prompt[x["rubric"]], C.name(x["grader"]))]
        add_est(f"with the transcript, {prompt[x['rubric']]}, {C.name(x['grader'])}", x,
                (r["premium (overclaim minus honest)"], r["lo"], r["hi"], r["tasks"]))
    s25 = {x["what"]: x["value"] for x in C.read_csv("r25_summary")}
    add("replaced passages, all sets", b["b5_replaced_passages_total"], s25["replaced passages, all sets"])
    add("replaced passages per set: mean", b["b5_replaced_passages_mean"], s25["replaced passages per set: mean"])
    add("replaced passages per set: smallest", b["b5_replaced_passages_min"], s25["replaced passages per set: smallest"])
    add("replaced passages per set: largest", b["b5_replaced_passages_max"], s25["replaced passages per set: largest"])
    add("share of the overclaim's characters replaced: mean", b["b5_mean_share_of_characters_replaced"],
        s25["share of the overclaim's characters replaced: mean"])
    add("honest messages with a check mark", b["b5_honest_messages_with_a_check_mark"],
        s25["honest messages that keep at least one check-marked line of the agent"])
    o27 = {x["version"]: x for x in C.read_csv("r27d_gptoss_twos")}
    add("gpt-oss-120b without the transcript: share of scores that are 2", b["b6_gptoss_hidden_share_of_scores_equal_2"], o27["all three"]["share"])
    d27 = {x["judge"]: x for x in C.read_csv("r27d_share_of_tens")}
    for x in b["b6_share_of_tens_with_transcript"]:
        r = d27[C.name(x["grader"])]
        add(f"share of 10s with the transcript, {C.name(x['grader'])}", x["share"], r["share that are 10, with"])
        add(f"scores with the transcript, {C.name(x['grader'])}", x["n_scores"], r["scores with the transcript"])
    t24 = next(x for x in C.read_csv("r24_planned_estimates") if x["test"] == "RQ1" and x["judge"] == "Sonnet 5.5")
    lab = "without the 6 resolved tasks"
    add_est("Sonnet 5.5 premium without the 6 resolved tasks", b["b7_sonnet55_premium_without_resolved_tasks"],
            (t24[lab], t24[lab + ": lo"], t24[lab + ": hi"], t24[lab + ": tasks"]))

    C.write_csv("independent_check", ["quantity", "independent", "this report", "difference", "agrees to 0.01"], rows)
    bad = [x for x in rows if not x[4]]
    print(f"wrote independent_check: {len(rows)} numbers compared, {len(bad)} differ by more than {TOL}")
    for x in bad:
        print("  ", x)


if __name__ == "__main__":
    main()

"""Write paper/numbers_manifest.csv: one row per number printed in paper/sections/*.tex, with the committed file and
key it comes from (rule 1.3 of the write-up brief), and one row per mean drawn in Figure B1
(paper/figures/figB1_mean_scores.csv). Two numbers of the introduction are not ours: they are quoted from a cited
source (about 1,200 agents and the year 2026), and their rows point to paper/refs_check.json, where
scripts/check_paper_refs.py wrote what it matched on the fetched pages. scripts/check_paper_numbers.py reads the manifest and fails when
a number in the prose is missing from it or disagrees with its source at the printed precision.

Draft 7 adds the numbers of the reanalysis (reanalysis/report.md): each row points to its CSV in reanalysis/tables.
The column `mark` says "after the fact" for a number of an analysis added after the results were known: the premium by
kind of missing check, every result within the 8 messages written under the added instruction, the difference between
fresh messages, the share of the premium, and the samples that leave tasks out. The checker then requires the words in the
part of the text that prints the number.

Draft 10 adds the numbers of the control run of the sprint: each row points to a file of results/sprint_control
(written by sprint_control/analyze.py at commit 0d1ed8b from the scores of 3cf890b and 8320b5b) or to the block
`control_run` of paper/derived.json, which is read from those files. The control run was declared before its first
call, so its numbers do not carry the mark "after the fact". The comparison against a hedge left Table 2 and
Section 4.1, and the keyword count of the judges' reasons moved from the caption of Table B9 into the table, so
their rows left the manifest (the table script reads the same files).

The `printed` values below are typed from the prose, not computed from the sources: the check is only a check if the
two sides are independent. When a sentence changes, change its row here and run the checker.

Run: uv run --no-sync python scripts/make_numbers_manifest.py && uv run --no-sync python scripts/check_paper_numbers.py
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "numbers_manifest.csv"
COMMIT = "4965785"
# paper/derived.json is generated from the records at COMMIT and committed with the draft that uses it.
DERIVED_COMMIT = "generated from 4965785 (scripts/make_paper_derived.py)"

S = "results/full_run/summary.json"
RS = "results/full_run/runs_status.json"
NOTES = "results/full_run/NOTES.md"
KAPPA = "results/full_run/human_check_kappa_a1_a2.txt"
CALLS = "results/full_run/pointwise_calls.jsonl"
DOC = "data/processed/full_run_tasks.json"
SETS = "data/processed/coding/matched_sets_v4_full_run.jsonl"
CFG = "configs/full_run.yaml"
ANALYSIS = "src/overclaim/analysis.py"
POINTWISE = "src/overclaim/graders/pointwise.py"
E2SEL = "src/overclaim/experiments/e2_selection.py"
RENDER = "src/overclaim/render.py"
README = "README.md"
REAN_COMMIT = "639cc30"   # the commit of reanalysis/tables (scripts with their definitions: 05f7af5)
_R = "reanalysis/tables/"
R21, R21C, R21F = _R + "r21_four_comparisons.csv", _R + "r21_detectable_counts.csv", _R + "r21_fresh_variants.csv"
R22, R22P = _R + "r22_premium_by_missing_check.csv", _R + "r22_parts_by_missing_check.csv"
R23O, R23S = _R + "r23_overclaims_by_instruction.csv", _R + "r23_selection_by_instruction.csv"
R24F, R25 = _R + "r24_pilot_score_files.csv", _R + "r25_summary.csv"
R27A, R27C = _R + "r27a_failure_before_the_edit.csv", _R + "r27c_pilot4_scored_alone.csv"
R27E, R27F = _R + "r27e_labeller_check_of_honest_messages.csv", _R + "r27f_with_the_transcript.csv"
R28 = _R + "r28_contradiction_in_reasons.csv"      # the count of 2026-11-13 (draft 9), committed with that draft
R28_COMMIT = "f0c4f11"     # draft 9: the script (reanalysis/scripts/r28_contradiction_in_reasons.py) and its tables
# (no row reads R28 since draft 10: its counts are printed in the lower part of Table B9, which the table script writes)
# The control run of the sprint (draft 10): the tables that sprint_control/analyze.py wrote, and its analysis file.
_C = "results/sprint_control/"
CC, CS, CM, CR, CA = (_C + "comparisons.csv", _C + "split.csv", _C + "marked.csv", _C + "reasons.csv",
                      _C + "analysis.json")
CTRL_COMMIT = "0d1ed8b"    # the declared analysis of the control run (scores: 3cf890b and 8320b5b)
CTRL_RULE = "sprint_control/analyze.py"      # the rule of 54 tasks (declared in 2175c36)
AF = "after the fact"      # the value of the column `mark`
DER = "paper/derived.json"   # written by scripts/make_paper_derived.py from the committed records
REFS = "paper/refs_check.json"   # written by scripts/check_paper_refs.py from the pages of the cited sources
REFS_COMMIT = "pages fetched on 2026-11-13 (scripts/check_paper_refs.py)"

ABS, INT, PCT = "abstract.tex", "introduction.tex", "pct"
MET, RES, DIS, CON, APP = "methods.tex", "results.tex", "discussion.tex", "conclusion.tex", "appendix.tex"
COD = "code_and_data.tex"
FIG_MEANS = "figB1_mean_scores.csv"   # the values drawn in Figure B1 (paper/figures); see DATA_FILES in the checker


def core(g: str, cond: str = "hidden") -> str:
    return f"core[grader={g};condition={cond};rubric=pointwise-neutral]"


def holm(member: str) -> str:
    return f"holm_family[member={member}]"


def hyp(h: str, g: str) -> str:
    return f"hypotheses[hypothesis={h};grader={g}]"


def e2(sel: str, n: str, table: bool = False) -> str:
    if table:
        return f"e2[selector={sel};labeller=e2_tasks/session3;outcome=clean_overclaim;source=e2_table.json].tie_averaged.rates.{n}.mean"
    return f"e2[selector={sel};labeller=session3;outcome=clean_overclaim;source=records].tie_averaged.rates.{n}.mean"


def pair(g: str, cond: str = "hidden") -> str:
    return f"pairwise[table=judge;grader={g};condition={cond}]"


def flip(g: str, p: str) -> str:
    return f"pairwise[table=position_bias;grader={g};pair={p};condition=hidden].rate"


def sens(rubric: str, g: str, part: str) -> str:
    return f"sensitivity[section=Holm family;rubric={rubric};grader={g}].{part}"


E7S = "e7[grader=claude-sonnet-5-5]"
S55, O55, S46, O46 = "claude-sonnet-5-5", "claude-opus-5-5", "claude-sonnet-4-6", "claude-opus-4-6"
FLASH, ARGON, DS, OSS, QWEN = "gemini-3.8-flash", "gemini-4-argon", "deepseek-v4-pro", "gpt-oss-120b", "qwen3-32b"
H4II = "claude-sonnet-5-5+verify_first vs claude-sonnet-5-5"
ALL = (ABS, INT, MET, RES, DIS, CON, COD, APP)

# (printed, files, source, key, transform, note), and for a number of an analysis after the fact a seventh item, AF
Row = tuple

DESIGN: list[Row] = [
    ("60", ALL, S, "header.counts.sets", "", "tasks"),
    ("20", (MET, RES, APP), S, "header.counts.per_gap.no_tests_run", "",
     "tasks per kind of missing check; also the sets whose honest message says that no test ran"),
    ("34", (MET, DIS, APP), S, "header.counts.shown_eligible", "", "transcripts within 40,000 tokens"),
    ("26", (APP,), S, "header.counts.band", "", "longer transcripts"),
    ("26", (RES, APP), S, "header.stages.e2.n_shown_eligible", "", "selection tasks whose transcript fits in 40,000 tokens"),
    ("57", (RES, APP), S, core(OSS) + ".n_tasks", "",
     "gpt-oss-120b complete tasks without the transcript (caption of the table of comparisons; caption of Figure B2)"),
    ("4", (APP,), CFG, "e2.ns[#1]", "", "best of 4"),
    ("40", (MET, RES, APP), S, "header.stages.e2.n_tasks", "", "selection tasks"),
    ("17", (MET, DIS, APP), S, "header.counts.repaired_sets", "", "sets corrected after the audit"),
    ("19", (MET,), DER, "set_history.corrected_on_the_audit", "",
     "sets corrected on the audit's findings (the first freeze, data/processed/full_run_tasks.superseded-20261107T172655Z.json)"),
    ("2", (MET,), DER, "set_history.replaced_on_the_audit", "", "sets replaced on the audit's findings (the first freeze)"),
    ("2", (MET,), DER, "set_history.replaced_after_the_author_read", "",
     "corrected sets replaced after the author read three of them (19 - 2 = 17 corrected sets remain)"),
    ("7", (APP,), SETS, "count[n_kept_true_spans=0]", "", "sets in which the transcript supports none of the other claims"),
    ("1", (ABS, MET, APP), POINTWISE, r"regex:from (1) \(very poor\)", "", "score scale, low end"),
    ("10", (ABS, MET, APP), POINTWISE, r"regex:to (10) \(excellent\)", "", "score scale, high end"),
    ("16", (ABS, MET, RES, APP), CFG, "e2.ns[#2]", "", "best of 16"),
    ("40,000", (MET, RES, APP), CFG, "shown.max_transcript_tokens", "", "transcript limit for the runs with the transcript"),
    ("2,000", (MET, APP), S, "n_bootstrap", "", "bootstrap resamples"),
    ("0", (APP,), S, "seed", "", "bootstrap seed"),
    ("95%", (MET, RES, APP), ANALYSIS, r"regex:alpha: float = (0\.05)", "pct1m", "interval level, 100 (1 - alpha)"),
    ("0.05", (MET, RES, APP), ANALYSIS, r"regex:alpha: float = (0\.05)", "", "level of the Holm correction"),
    ("45", (APP,), DER, "figure1.max_honest_words", "", "Figure 1 rule: longest honest sentence allowed"),
    ("6", (INT,), DER, "figure1.candidates[#0].other_claims_replaced", "", "Figure 1 caption: other unsupported claims replaced in the example set (a heading is replaced too)"),
    ("22", (APP,), DER, "set_counts.sets_main_claim_restated", "", "sets in which Claude Opus 5.5 restated the factual part of the main claim"),
    ("59", (APP,), DER, "set_counts.sets_with_a_rewrite", "", "sets that carry a rewrite by Claude Opus 5.5"),
    ("100", (APP,), S, "n_tie_seeds", "", "random draws over which a tie for the highest score is averaged"),
    ("3", (MET,), DER, "invalid_replies.gptoss_tasks_lost_without_transcript", "",
     "tasks that gpt-oss-120b loses without the transcript to a reply with no valid score (60 - 3 = 57)"),
    ("30", (RES, APP), S, core(O55, "shown") + ".n_tasks", "", "Opus 5.5 tasks with the transcript"),
    ("34", (RES, APP), S, core(FLASH, "shown") + ".n_tasks", "", "Gemini 3.8 Flash tasks with the transcript"),
    ("33", (APP,), S, core(OSS, "shown") + ".n_tasks", "", "gpt-oss-120b tasks with the transcript"),
    ("2", (RES, APP), S, core(QWEN, "shown") + ".n_tasks", "", "Qwen3-32B complete tasks with the transcript when its run failed"),
]

# Numbers quoted from cited sources in the introduction's paragraph on systems of several agents (not our data).
CITED: list[Row] = [
    ("1,200", (INT,), REFS, "[key=greenblatt2026incident].numbers.agents_that_communicated", "",
     "quoted from the cited investigation: agents that found a way to communicate (the page says roughly 1200)"),
    ("2026", (INT,), REFS, "[key=greenblatt2026incident].numbers.year", "", "year of the cited investigation and of the incident"),
]

WITHOUT: list[Row] = [
    ("0.65", (ABS,), S, core(QWEN) + ".premium", "", "smallest judge premium without the transcript: Qwen3-32B"),
    ("3.33", (ABS,), S, core(FLASH) + ".premium", "", "largest judge premium without the transcript: Gemini 3.8 Flash"),
    ("1.57", (INT, RES), S, holm(f"H1 / {S55}") + ".value", "", "Sonnet 5.5 premium without the transcript (planned test; Figure 1 caption)"),
]

WITH: list[Row] = [
    ("0.07", (INT, RES), DER, f"with_transcript_all_tasks.{S55}.premium", "", "Sonnet 5.5 premium with the transcript, all 60 tasks (also the Figure 1 caption)"),
    ("-1.50", (RES,), S, holm(f"H2 / {S55}") + ".value", "", "planned change, Sonnet 5.5, 60 tasks"),
    ("-1.80", (RES,), S, holm(f"H2 / {S55}") + ".ci[#0]", "", ""),
    ("-1.20", (RES,), S, holm(f"H2 / {S55}") + ".ci[#1]", "", ""),
    ("11", (RES, APP), S, core(ARGON, "shown") + ".n_tasks", "", "Gemini 4 Argon tasks with the transcript"),
    ("34", (APP,), RS, "pws-gema.n_tasks.'pointwise_calls.jsonl'", "", "tasks planned for Gemini 4 Argon with the transcript"),
]

SELECTION: list[Row] = [
    ("68%", (RES, APP), S, e2(S46, "1"), PCT, "share of overclaims when one message is taken at random"),
    ("12", (RES,), S, holm(f"H3 / {S46}") + ".value", PCT, "planned rise under Sonnet 4.6, percentage points"),
    ("3", (RES,), S, holm(f"H3 / {S46}") + ".ci[#0]", PCT, ""),
    ("20", (RES,), S, holm(f"H3 / {S46}") + ".ci[#1]", PCT, ""),
    ("58%", (RES,), S, e2(S55, "16"), PCT, "Sonnet 5.5, best of 16"),
    ("89%", (RES,), DER, "counts.selection_rise_min_at_16_others", PCT, "lowest share among DeepSeek V4 Pro and the reward models"),
    ("98%", (RES,), DER, "counts.selection_rise_max_at_16", PCT, "highest share, best of 16"),
    ("81%", (APP,), S, "e2[selector=claude-sonnet-4-6;labeller=session3;outcome=overclaim;source=records].tie_averaged.rates.1.mean", PCT,
     "share at random when every overclaim is counted, with or without a disclosed gap"),
    ("-10", (RES,), S, holm(f"H3 / {S55}") + ".value", PCT, "planned test under Sonnet 5.5: change of the share, percentage points"),
    ("-20", (RES,), S, holm(f"H3 / {S55}") + ".ci[#0]", PCT, ""),
    ("-1", (RES,), S, holm(f"H3 / {S55}") + ".ci[#1]", PCT, ""),
    ("53%", (RES,), DER, f"selection_on_26_tasks.{S55}.without_step", PCT,
     "Sonnet 5.5, best of 16 on the 26 tasks, without the verification step and the transcript (Section 4.4 and Figure 3)"),
    ("33%", (RES,), DER, f"selection_on_26_tasks.{S55}.with_step_and_transcript", PCT,
     "the same with the verification step and the transcript (Section 4.4)"),
    ("89%", (RES,), DER, f"selection_on_26_tasks.{DS}.without_step", PCT,
     "DeepSeek V4 Pro, best of 16 on the 26 tasks, without the verification step and the transcript (caption of Figure 3)"),
]

VERIFICATION: list[Row] = [
    ("-0.84", (RES,), S, hyp("H4(i)", S55) + ".value", "", "verification step minus plain question, premium with the transcript"),
    ("-1.23", (RES,), S, hyp("H4(i)", S55) + ".ci[#0]", "", ""),
    ("-0.48", (RES,), S, hyp("H4(i)", S55) + ".ci[#1]", "", ""),
    ("-20", (RES,), S, hyp("H4(ii)", H4II) + ".value", PCT, "change in the share of overclaims selected, percentage points"),
    ("-38", (RES,), S, hyp("H4(ii)", H4II) + ".ci[#0]", PCT, ""),
    ("-2", (RES,), S, hyp("H4(ii)", H4II) + ".ci[#1]", PCT, ""),
    ("0.029", (RES, APP), S, holm(f"H4 / {S55}") + ".p", "", "RQ4 uncorrected p (also the caption of Table B1)"),
    ("0.058", (RES, APP), S, holm(f"H4 / {S55}") + ".p_holm", "", "RQ4 corrected p (also the caption of Table B1)"),
    ("31", (APP,), S, hyp("H4(i)", S55) + ".n_tasks", "", "tasks in the first part of the RQ4 test"),
    ("-0.68", (APP,), DER, f"verification_step.{S55}.premium", "", "Sonnet 5.5 premium under the verification step, on the 31 tasks of the test"),
    ("-0.97", (APP,), DER, f"verification_step.{S55}.ci[#0]", "", ""),
    ("-0.39", (APP,), DER, f"verification_step.{S55}.ci[#1]", "", ""),
    ("30", (APP,), S, E7S + ".diffs.instruction_only_minus_plain.n_tasks", "", "tasks of the three ways of scoring"),
    ("28", (APP,), S, E7S + ".scorers.verify_first.n_tasks", "", "Sonnet 5.5 verification-step tasks among those 30"),
    ("4", (APP,), RS, "pws-s55.n_invalid.'e3pw_calls.jsonl'", "", "invalid replies, Sonnet 5.5 verification step"),
    ("6", (APP,), RS, "pwh-oss.n_invalid", "", "invalid replies, gpt-oss-120b without the transcript"),
    ("1", (APP,), RS, "pws-oss.n_invalid.'pointwise_calls.jsonl'", "", "invalid replies, gpt-oss-120b with the transcript"),
    ("6", (APP,), DER, "invalid_replies.without_transcript.total", "", "replies with no valid score without the transcript, all judges"),
    ("1", (APP,), DER, "invalid_replies.with_transcript.total", "", "the same with the transcript"),
    ("4", (APP,), DER, "invalid_replies.verification_step.total", "", "the same under the verification step, messages scored alone"),
    ("25", (APP,), DER, "invalid_replies.selection_with_verification_step.total", "", "the same in selection under the verification step"),
    ("-0.40", (APP,), S, E7S + ".diffs.instruction_only_minus_plain.value", "", "Sonnet 5.5, 30 tasks scored three ways: a short instruction alone minus the plain question"),
    ("-0.73", (APP,), S, E7S + ".diffs.instruction_only_minus_plain.ci[#0]", "", ""),
    ("-0.10", (APP,), S, E7S + ".diffs.instruction_only_minus_plain.ci[#1]", "", ""),
    ("-0.75", (APP,), S, E7S + ".diffs.verify_first_minus_plain.value", "", "the same tasks: the verification step minus the plain question"),
    ("-1.14", (APP,), S, E7S + ".diffs.verify_first_minus_plain.ci[#0]", "", ""),
    ("-0.36", (APP,), S, E7S + ".diffs.verify_first_minus_plain.ci[#1]", "", ""),
]

SIDE_BY_SIDE: list[Row] = [
    ("21%", (RES,), S, pair(S55) + ".pairs.O_vs_H.rate", PCT, "Sonnet 5.5 prefers the overclaim side by side"),
    ("98%", (RES,), S, f"pairwise[table=reasons;grader={O55};condition=hidden].O_vs_H_only.against_O.share_cites_disclosure", PCT,
     "Opus 5.5: verdicts for the honest message over the overclaim whose reason cites the disclosure"),
    ("98%", (RES,), S, f"pairwise[table=reasons;grader={S55};condition=hidden].O_vs_H_only.against_O.share_cites_disclosure", PCT,
     "Sonnet 5.5: the same share"),
    ("40%", (RES, APP), DER, "counts.flip_limit", PCT, "limit above which a judge's side-by-side rate is not read as a preference"),
]

APPENDIX_ONLY: list[Row] = [
    ("7", (APP,), DER, "personal_data.whole_file.email_pattern_matches", "len", "strings shaped like e-mail addresses in the set file"),
    ("2", (APP,), DER, "personal_data.whole_file.byline_matches", "len", "author or copyright lines"),
    ("16%", (APP,), DER, "planned_tests_reuse.printed_range_percent[#0]", "",
     "lower end of the printed range of reused scores: the smallest share (the RQ4 test: 158 of 938, 16.8%) rounded down"),
    ("33%", (APP,), DER, "planned_tests_reuse.printed_range_percent[#1]", "",
     "upper end of the printed range: the largest share (each RQ1 test: 39 of 120, 32.5%) rounded up"),
    ("55", (APP,), S, "e6[grader=gpt-oss-120b].n_tasks", "", "gpt-oss-120b tasks in Table B9 (every reply valid), named in its caption"),
    ("30", (APP,), KAPPA, r"regex:O_other_claims_supported: both filled (\d+)", "", "sets audited twice"),
    ("0.933", (APP,), KAPPA, r"regex:O_other_claims_supported: both filled \d+, either empty \d+, observed agreement ([0-9.]+)", "",
     "agreement of the two audits on whether the other claims are supported"),
    ("0.83", (APP,), KAPPA, r"regex:O_other_claims_supported: both filled \d+, either empty \d+, observed agreement [0-9.]+, kappa ([0-9.]+)", "",
     "Cohen's kappa of the two audits on that question"),
    ("2,000", (APP,), RENDER, r"regex:max_tokens: int = (2000)", "", "tool outputs longer than this are cut in the transcript graders see"),
    ("71.76", (APP,), S, "spend[table=provider;provider=anthropic].usd", "", "main run, Anthropic API, USD"),
    ("7.06", (APP,), S, "spend[table=provider;provider=openrouter].usd", "", "main run, OpenRouter, USD"),
    ("12.76", (APP,), S, "spend[table=provider;provider=google].usd", "", "main run, Google (Vertex AI and the AI Studio API), USD"),
]


def r22(g: str, col: str, cond: str = "without the transcript") -> str:
    return f"[grader={g};condition={cond}].{col}"


def r23s(sel: str, col: str) -> str:
    return f"[selector={sel}].{col}"


def r27f(prompt: str, judge: str, col: str) -> str:
    return f"[scoring prompt={prompt};judge={judge}].{col}"


_PARTS = "[comparison=placebo minus honest message;grader=Sonnet 5.5;condition=without the transcript]"
_DETECT = "graders with a detectable positive difference"
_VS_HALF = "[variant=with the transcript and the verification step, honesty half;grader={}, verification step].tasks used"

_SECOND = "[variant=honesty half, every message labelled overclaim;grader={}]"


# The numbers of draft 7 that come from the reanalysis of 2026-11-11 or from the facts it brought into the paper.
REANALYSIS: list[Row] = [
    # the sets
    ("6", (MET, DIS, APP), S, "header.counts.resolved", "", "runs whose code change did resolve the issue"),
    ("8", (ABS, RES, APP), CFG, "e2.n_per_instruction", "", "fresh messages per task written under the added instruction"),
    ("8", (RES, DIS), R25, "[what=replaced passages per set: mean].value", "",
     "passages in which the honest message differs from the overclaim, mean over the 60 sets"),
    ("40", (RES, APP), R22, r22("Sonnet 5.5", "failure reported (40 sets): tasks"), "",
     "sets whose honest message reports a failed test run (the two kinds of missing check with a failure)"),
    # the plan and the pilots
    ("18", (APP,), R22, r22("gpt-oss-120b", "last test run failed: tasks"), "",
     "gpt-oss-120b tasks in the sets whose last test run failed (caption of Table B10)"),
    ("19", (APP,), R22, r22("gpt-oss-120b", "narrower rerun after a failure: tasks"), "",
     "gpt-oss-120b tasks in the sets with a narrower rerun after a failure (caption of Table B10)"),
    ("33", (APP,), R27A, "[kind of missing check=both kinds (40 sets);the record says the named test failed before the edit=unknown].sets", "",
     "sets with a reported failure for which the transcript does not show whether the test failed before the edit"),
    # the table of comparisons (the hedge and the disclosure were run with the main run: not after the fact)
    # draft 10: the comparison against a hedge left Table 2 and Section 4.1 (its numbers are in Table B9)
    ("12", (RES,), R21C, f"[comparison=disclosure against a neutral sentence].{_DETECT}", "",
     "graders whose score falls detectably when the disclosure stands in place of the placebo's neutral sentence"),
    ("14", (RES,), R21C, "[comparison=disclosure against a neutral sentence].of", "", "graders in that comparison"),
    ("29", (RES, APP), R21, "[grader=Sonnet 5.5].fresh tasks used", "",
     "tasks whose 8 messages under the added instruction hold both overclaims and other messages", AF),
    # draft 9: the second count of an overclaim (r21). The keyword count of the judges' stated reasons (r28) is
    # printed in the lower part of Table B9 since draft 10, not in a sentence, so it has no row here.
    ("-0.06", (APP,), R21F, _SECOND.format("Sonnet 4.6") + ".difference", "",
     "Sonnet 4.6: overclaims minus the other messages among the 8, every message labelled overclaim counted (caption of Table B12)", AF),
    ("-0.29", (APP,), R21F, _SECOND.format("Sonnet 4.6") + ".lo", "", "", AF),
    ("0.19", (APP,), R21F, _SECOND.format("Sonnet 4.6") + ".hi", "", "", AF),
    ("32", (APP,), R21F, _SECOND.format("Sonnet 4.6") + ".tasks used", "",
     "tasks whose 8 messages hold both kinds under the second count (caption of Table B12)", AF),
    # the revision of 2026-11-16: the largest fall for the disclosure among the judges (Section 4.1; Table 3)
    ("1.21", (RES,), R21, "[grader=gpt-oss-120b].disclosure against a neutral sentence", "",
     "the placebo minus the honest message for gpt-oss-120b (57 tasks), the largest among the nine judges"),
    # the author's second list of 2026-11-16: what the last column of Table 3 found, stated in Section 4.1
    ("0.27", (RES,), R21, "[grader=Sonnet 4.6].fresh messages (honesty half)", "",
     "Sonnet 4.6: overclaims minus the other messages among the 8 written under the added instruction", AF),
    ("0.41", (RES,), R21, "[grader=DeepSeek V4 Pro].fresh messages (honesty half)", "", "the same for DeepSeek V4 Pro", AF),
    ("0.47", (RES,), R21, "[grader=Skywork 1.7B].fresh messages (honesty half)", "",
     "the same for Skywork 1.7B, the smallest value among the five reward models (standard deviations of its scores)", AF),
    ("0.73", (RES,), R21, "[grader=Skywork 4B].fresh messages (honesty half)", "",
     "the same for Skywork 4B, the largest value among the five reward models", AF),
    # the revision of 2026-11-16: why Sonnet 5.5 selects fewer overclaims (Section 4.3; Table B12)
    ("64%", (RES,), R23S, r23s("Sonnet 5.5", "top pick among all 16 comes from the plain half"), "pct1m",
     "Sonnet 5.5: share of its selections from all 16 that were written under the added instruction (100% - 36%)", AF),
    ("37%", (RES,), R23S, r23s("Sonnet 5.5", "honesty half: top pick"), PCT,
     "Sonnet 5.5: share of overclaims among its selections from the 8 written under the added instruction", AF),
    # the premium by kind of missing check
    # draft 10: these two moved from Section 4.1 to the caption of Table B10 (the brief's sixth cut for room)
    ("0.47", (APP,), R22P, _PARTS + ".failure reported (40 sets)", "",
     "Sonnet 5.5: the placebo minus the honest message in the 40 sets with a reported failure (0.475)", AF),
    ("-0.20", (APP,), R22P, _PARTS + ".no test run", "",
     "Sonnet 5.5: the placebo minus the honest message in the 20 sets where no test ran (40 x 0.475 - 20 x 0.20 = 60 x 0.25)", AF),
    ("13", (RES,), R22, "count[condition=without the transcript;no test run: lo>0]", "",
     "graders with a detectable premium in the 20 sets where no test ran", AF),
    ("14", (RES,), R22, "count[condition=without the transcript]", "", "graders in that count", AF),
    # selection by instruction
    ("99%", (RES,), R23O, "[messages=plain instruction;outcome=the paper's outcome].share of the messages", PCT,
     "overclaims among the messages written without the added instruction", AF),
    ("38%", (RES,), R23O, "[messages=honesty instruction;outcome=the paper's outcome].share of the messages", PCT,
     "overclaims among the messages written under the added instruction", AF),
    ("16", (APP,), R21F, _VS_HALF.format("Sonnet 5.5"), "",
     "Sonnet 5.5 under the verification step: tasks with both kinds of message among the 8 (caption of Table B12)", AF),
    ("17", (APP,), R21F, _VS_HALF.format("DeepSeek V4 Pro"), "", "the same for DeepSeek V4 Pro", AF),
    # the verification step, each message scored alone (in draft 6's tables; printed in Section 4.4 from draft 7)
    ("-0.68", (RES,), R27F, r27f("verification step", "Sonnet 5.5", "premium (overclaim minus honest)"), "",
     "Sonnet 5.5 premium under the verification step, 31 tasks"),
    ("-0.97", (RES,), R27F, r27f("verification step", "Sonnet 5.5", "lo"), "", ""),
    ("-0.39", (RES,), R27F, r27f("verification step", "Sonnet 5.5", "hi"), "", ""),
    ("-1.13", (RES,), R27F, r27f("verification step", "Opus 5.5", "premium (overclaim minus honest)"), "",
     "Opus 5.5 premium under the verification step, 30 tasks"),
    ("-1.50", (RES,), R27F, r27f("verification step", "Opus 5.5", "lo"), "", ""),
    ("-0.80", (RES,), R27F, r27f("verification step", "Opus 5.5", "hi"), "", ""),
    ("31", (RES,), R27F, r27f("verification step", "Sonnet 5.5", "tasks"), "", "tasks of Sonnet 5.5 under the verification step"),
    ("30", (RES,), R27F, r27f("verification step", "Opus 5.5", "tasks"), "", "tasks of Opus 5.5 under the verification step"),
]


def _cmp(grader: str, col: str) -> str:
    return f"[grader={grader}].{col}"


def _split(grader: str, sets: str, col: str, comparison: str = "C - H") -> str:
    return f"[grader={grader};comparison={comparison};sets={sets}].{col}"


def _reasons(judge: str, version: str) -> str:
    return f"[judge={judge};version={version}].list of the first count"


_CTRL = "control_run."
_G, _P2 = "hedge alone (G)", "placebo, second score (P2)"
_MARKED, _OTHER = "marked by the keyword rule", "not marked"

# Draft 10: the control run of the sprint (declared before its first call; no mark).
CONTROL: list[Row] = [
    # Section 4.1 and the caption of the table of comparisons (Table 3 since draft 13)
    ("0.02", (RES,), DER, _CTRL + "claim_minus_placebo_of_those_judges.min", "",
     "control run: the smallest gain of the claim against the placebo's neutral sentence among the six judges with "
     "all 60 tasks (DeepSeek V4 Pro)"),
    ("0.30", (RES,), DER, _CTRL + "claim_minus_placebo_of_those_judges.max", "",
     "the largest such gain (Sonnet 5.5 and Opus 4.6)"),
    ("54", (RES, APP), CC, _cmp("gpt-oss-120b", "tasks"), "",
     "control run: gpt-oss-120b tasks with a valid score on all six messages (caption of the table of comparisons; Appendix A)"),
    # Appendix A, the paragraph on the control run, and the captions of Tables B13 and B14
    ("54", (APP,), CTRL_RULE, r"regex:MIN_TASKS = (54)\b", "", "a test of the control run needs at least 54 of the 60 tasks"),
    ("22", (APP,), CS, _split("Sonnet 5.5", "plain statement", "tasks"), "",
     "sets whose changed passage also holds a plain statement of the code change"),
    ("38", (APP,), CS, _split("Sonnet 5.5", "no plain statement", "tasks"), "", "the other sets"),
    ("0.26", (APP,), CS, _split("Opus 5.5", "no plain statement", "mean"), "",
     "control run: the claim alone minus the honest message for Opus 5.5 on the 38 sets without a plain statement"),
    ("0.05", (APP,), CS, _split("Opus 5.5", "no plain statement", "lo"), "", ""),
    ("0.50", (APP,), CS, _split("Opus 5.5", "no plain statement", "hi"), "", ""),
    ("0.15", (APP,), CC, _cmp("Opus 4.6", "P2 - P mean"), "",
     "control run: the placebo scored again minus its stored score, Opus 4.6 (the only judge whose interval excludes zero)"),
    ("0.02", (APP,), CC, _cmp("Opus 4.6", "P2 - P lo"), "", ""),
    ("0.30", (APP,), CC, _cmp("Opus 4.6", "P2 - P hi"), "", ""),
    ("0.70", (APP,), CC, _cmp("Opus 4.6", "C - H, same day mean"), "",
     "control run: the claim alone minus the honest message for Opus 4.6 with claim and placebo both scored in this run"),
    ("0.40", (APP,), CC, _cmp("Opus 4.6", "C - H, same day lo"), "", ""),
    ("1.03", (APP,), CC, _cmp("Opus 4.6", "C - H, same day hi"), "", ""),
    ("3", (APP,), DER, _CTRL + "replies_with_no_valid_score", "",
     "control run: replies with no valid score (all gpt-oss-120b on the claim alone)"),
    ("1,260", (APP,), DER, _CTRL + "replies", "", "control run: replies of the seven judges (7 x 180)"),
    ("36", (APP,), CM, f"[grader=Sonnet 5.5;sets={_MARKED}].tasks", "", "sets that the keyword rule marks (caption of Table B14)"),
    ("24", (APP,), CM, f"[grader=Sonnet 5.5;sets={_OTHER}].tasks", "", "the other sets"),
    ("34", (APP,), CS, _split("gpt-oss-120b", "no plain statement", "tasks"), "",
     "gpt-oss-120b tasks among the 38 sets (caption of Table B14)"),
    ("20", (APP,), CS, _split("gpt-oss-120b", "plain statement", "tasks"), "", "gpt-oss-120b tasks among the 22 sets"),
    ("31", (APP,), CM, f"[grader=gpt-oss-120b;sets={_MARKED}].tasks", "", "gpt-oss-120b tasks among the 36 marked sets"),
    ("23", (APP,), CM, f"[grader=gpt-oss-120b;sets={_OTHER}].tasks", "", "gpt-oss-120b tasks among the 24 other sets"),
    ("4.40", (APP,), DER, _CTRL + "spend_usd.anthropic", "", "control run, Anthropic API, USD"),
    ("0.13", (APP,), DER, _CTRL + "spend_usd.openrouter", "", "control run, OpenRouter, USD"),
]


def _mean_rows() -> list[Row]:
    """The means drawn in Figure B1, as printed in paper/figures/figB1_mean_scores.csv (two decimals). The values are
    typed here from that file, one line per judge and panel in the order overclaim, honest message, placebo."""
    typed = {
        "without": {S55: ("5.33", "3.77", "4.02"), O55: ("5.35", "3.93", "3.83"), S46: ("5.75", "5.02", "5.30"),
                    O46: ("6.50", "5.43", "5.98"), FLASH: ("9.10", "5.77", "6.88"), ARGON: ("8.30", "5.18", "6.10"),
                    DS: ("9.52", "7.80", "8.83"), OSS: ("4.26", "3.18", "4.39"), QWEN: ("9.70", "9.05", "9.35")},
        "with": {S55: ("3.97", "3.90", "3.80"), O55: ("4.07", "4.30", "3.50"), FLASH: ("6.94", "7.18", "7.26"),
                 ARGON: ("5.18", "5.55", "5.64"), DS: ("9.98", "9.18", "9.80"), OSS: ("9.94", "8.85", "9.64")},
    }
    out: list[Row] = []
    for cond, by_judge in typed.items():
        for g, vals in by_judge.items():
            for version, printed in zip(("O", "H", "P"), vals):
                out.append((printed, (FIG_MEANS,), DER, f"mean_scores.{cond}.'{g}'.{version}", "",
                            f"Figure B1: mean score of {version}, {g}, {cond} the transcript"))
    return out


GROUPS = [DESIGN, CITED, WITHOUT, WITH, SELECTION, VERIFICATION, SIDE_BY_SIDE, APPENDIX_ONLY, REANALYSIS, CONTROL,
          _mean_rows()]


def rows() -> list[dict]:
    out = []
    for group in GROUPS:
        for printed, files, source, key, transform, note, *mark in group:
            commit = (R28_COMMIT if source == R28 else REAN_COMMIT if source.startswith(_R)
                      else CTRL_COMMIT if source.startswith(_C) or key.startswith(_CTRL)
                      else "2175c36" if source == CTRL_RULE
                      else {DER: DERIVED_COMMIT, REFS: REFS_COMMIT}.get(source, COMMIT))
            for f in ((files,) if isinstance(files, str) else files):
                out.append({"printed": printed, "file": f, "source": source, "key": key, "transform": transform,
                            "commit": commit, "note": note, "mark": mark[0] if mark else ""})
    return out


def main() -> int:
    data = rows()
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["printed", "file", "source", "key", "transform", "commit", "note", "mark"])
        w.writeheader()
        w.writerows(data)
    print(f"wrote {OUT.relative_to(REPO)}: {len(data)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

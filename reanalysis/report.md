# Reanalysis for the reviewer's four points (2026-11-11)

For Uzair. This round computes and reports; nothing that exists changes. Every number comes from the committed
records through `reanalysis/scripts`, and the tables are in `reanalysis/tables`. No model was called. The analyses
are **after the fact**: a reviewer who had seen the results suggested them. Intervals are the paper's (95% task
bootstrap, 2,000 resamples, seed 0), "detectable" means the interval excludes zero, and nothing is corrected. The
eight planned tests and their verdicts are untouched.

The scripts and their definitions were committed (05f7af5) before their first run. The scripts of 2.1 to 2.4 first
reproduce numbers the paper prints and stop if one differs; none did. Two fresh agents, given the definitions and no result, recomputed 281 numbers with their own code from the raw records, among them the four the brief names; all agree to 0.01 (`independent_check.csv`). Seen while the record formats were read, before the definitions were fixed: the counts of 2.5, 2.7a, 2.7c and 2.7e;
the label totals of the fresh messages, which informed the reading of "overclaim" in 2.1; and how many tasks carry
a reused score, the input of 2.4. What changed after the first run is listed in the scripts' docstrings; one change
touches a definition, the tie rule of one cell of 2.7d.

**Words used.** The *hedge*: the overclaim with only its main claim replaced by "I expect this to resolve the issue,
but I have not confirmed it." (in 22 sets after a plain statement of the code change); the agent's other claims
stay. Table B9 of the paper prints this version as "main claim removed". *Fresh messages*: the 16 messages per task
that the agent's model wrote for the selection experiment, 8 under the plain instruction (the *plain half*) and 8
under the added instruction to state exactly what was and was not verified (the *honesty half*). An *overclaim*
among them is the paper's outcome: labelled overclaim and stating no missing check.

## 1. The reviewer's numbers beside ours

| what was measured | (R) | ours | agree? | table |
|---|---|---|---|---|
| Fresh messages, all 16, Sonnet 5.5 (2.1) | −0.08 [−0.26, 0.10] | −0.08 [−0.26, 0.10] | yes | r21_fresh_variants |
| Same, the other seven graders (2.1) | positive | 7 of 7 positive, each detectable | yes | r21_fresh_variants |
| Sonnet 5.5 premium: failure reported against no test run (2.2) | 1.95 against 1.10 | 1.80 (40 sets) against 1.10 (20 sets); 1.95 is the 20 sets whose last run failed | in part | r22_premium_by_missing_check |
| Overclaims among messages written under the plain instruction (2.3) | 99% | 99% (317 of 320) | yes | r23_overclaims_by_instruction |
| Overclaims among messages written under the honesty instruction (2.3) | 38% | 38% (121 of 320) | yes | r23_overclaims_by_instruction |
| Top message among the 8 honesty messages against one at random, reward models (2.3) | +23 to +32 points | +23 to +32 points, each detectable | yes | r23_selection_by_instruction |
| Same, Sonnet 4.6 (2.3) | +4 [−2, +11] | +4 [−2, 11] | yes | r23_selection_by_instruction |
| Messages tied at the top score, judges (2.3) | about 5 of 16 | 4.6, 5.4 and 5.7 of 16 | yes | r23_selection_by_instruction |
| Replaced passages per set (2.5) | 8 on average, 3 to 17 | 8.0 on average, 3 to 17 | yes | r25_summary |
| Share of the text replaced (2.5) | about a quarter | 34% of the overclaim's characters on average (12% to 64%) | no: a third by characters | r25_summary |
| Replacements that are neutral style sentences (2.5) | 301 of 480 | 301 of 480 in the placebo: 185 sentences, 63 headings, 38 lead-ins, 15 fragments; 241 in the honest message | yes, in count, for the placebo | r25_summary |
| Honest messages that keep check-marked lines (2.5) | 45 of 60 | 45 of 60 | yes | r25_summary |
| Reasons that call content padded, Sonnet 5.5 (2.6) | 88% honest, 13% overclaim | 77% and 7% with the brief's keywords; 83% and 10% with a longer list; placebo 90% | direction yes, levels lower | r26_reasons_by_keyword |
| Disclosed failure: did the test fail before the edit? (2.7a) | unknown for 33 | unknown for 33 of 40; "no" for 7 | yes | r27a_failure_before_the_edit |
| Final tasks that pilot 4 scored alone (2.7c) | 21 | 21, with eight of the nine judges | yes | r27c_pilot4_scored_alone |
| Scores of 10 with the transcript (2.7d) | about 80% (DeepSeek V4 Pro, gpt-oss-120b) | 81% and 80% | yes | r27d_share_of_tens |
| gpt-oss-120b scores of 2 without the transcript (2.7d) | about half, "fabricated" | 49%; 39% of those reasons contain "fabricat" | yes | r27d_gptoss_twos |
| Second labeller on the honest messages, before the corrections (2.7e) | 32 of 60 still overclaiming | 32 of 60 labelled overclaim; 0 of 60 by the paper's outcome | yes, with a qualifier | r27e_labeller_check_of_honest_messages |

13 of the eighteen agree outright and 2 more with a note. Three differ. The
1.95 is the premium in one kind of failure set, not in all 40. The share of text replaced is a third by characters,
not a quarter. The keyword count gives lower levels than the reviewer's, in the same direction; the reviewer's list
is not known.

## 2. The analyses

### 2.1 Four comparisons without the transcript (`r21_*`)

| grader | premium (every edit) | claim against a hedge | its share of the premium | disclosure against a neutral sentence | fresh messages, honesty half |
|---|---|---|---|---|---|
| Sonnet 5.5 | 1.57 [1.32, 1.82] | 1.22 [0.97, 1.47] | 78% | 0.25 [0.02, 0.48] | 0.10 [−0.14, 0.34] |
| Opus 5.5 | 1.42 [1.12, 1.70] | 1.38 [1.08, 1.67] | 98% | −0.10 [−0.30, 0.10] | – |
| Sonnet 4.6 | 0.73 [0.37, 1.10] | 0.38 [0.10, 0.65] | 52% | 0.28 [−0.02, 0.60] | 0.27 [0.02, 0.53] |
| Opus 4.6 | 1.07 [0.78, 1.35] | 0.27 [0.08, 0.45] | 25% | 0.55 [0.25, 0.87] | – |
| Gemini 3.8 Flash | 3.33 [2.78, 3.85] | 2.52 [2.13, 2.88] | 76% | 1.12 [0.58, 1.65] | – |
| Gemini 4 Argon | 3.12 [2.52, 3.73] | 1.57 [1.10, 2.00] | 50% | 0.92 [0.42, 1.42] | – |
| DeepSeek V4 Pro | 1.72 [1.28, 2.20] | 0.72 [0.45, 1.05] | 42% | 1.03 [0.72, 1.38] | 0.41 [0.23, 0.59] |
| gpt-oss-120b | 1.09 [0.42, 1.82] | 0.53 [−0.22, 1.31] | 47% | 1.21 [0.54, 1.89] | – |
| Qwen3-32B | 0.65 [0.52, 0.78] | 0.67 [0.53, 0.78] | 103% | 0.30 [0.20, 0.42] | – |
| Skywork 0.6B | 1.39 [1.24, 1.56] | 0.46 [0.37, 0.55] | 33% | 0.11 [0.05, 0.18] | 0.48 [0.31, 0.65] |
| Skywork 1.7B | 1.29 [1.16, 1.43] | 0.60 [0.52, 0.68] | 46% | 0.11 [0.05, 0.16] | 0.47 [0.29, 0.64] |
| Skywork 3B | 1.45 [1.32, 1.60] | 0.80 [0.72, 0.88] | 55% | 0.22 [0.16, 0.27] | 0.58 [0.44, 0.71] |
| Skywork 4B | 1.53 [1.41, 1.66] | 0.48 [0.42, 0.54] | 31% | 0.33 [0.26, 0.41] | 0.73 [0.59, 0.89] |
| Skywork 8B | 1.58 [1.45, 1.71] | 0.59 [0.53, 0.66] | 38% | 0.26 [0.20, 0.32] | 0.71 [0.55, 0.88] |

Judges in points; reward models in standard deviations of their own scores (on the 180 messages of the sets for
the first three comparisons, on the 640 fresh messages for the last). The premium changes 8 passages at once. The
next two comparisons change one, and draft 6 already prints them in Tables B9 and B5. The last compares, within a
task, the fresh overclaims with the other fresh messages of the honesty half (29 of 40 tasks have both kinds). The
difference is detectable for 13 of 14, 12 of 14 and 7 of 8 graders. The share is a ratio of two point
estimates on the same tasks and has no interval. **It cannot show** what honesty costs: a hedge is not an honest
statement of the missing check, and the labels of the fresh messages are one model's. Variants
(`r21_fresh_variants.csv`): the brief's words were "labelled overclaim"; read literally, so that the
79 messages labelled overclaim that state a missing check count as
overclaims, 6 of 8 remain and Sonnet 4.6 drops out. With the transcript and the verification step, Sonnet 5.5
scores fresh overclaims lower.

### 2.2 The premium by kind of missing check (`r22_*`)

| grader | no test run (20 sets) | failure reported (40 sets) | difference |
|---|---|---|---|
| Sonnet 5.5 | 1.10 [0.75, 1.45] | 1.80 [1.50, 2.08] | 0.70 [0.20, 1.18] * |
| Opus 5.5 | 1.00 [0.55, 1.50] | 1.62 [1.25, 1.95] | 0.62 [0.00, 1.22] |
| Sonnet 4.6 | 0.35 [−0.30, 1.05] | 0.93 [0.47, 1.35] | 0.57 [−0.21, 1.32] |
| Opus 4.6 | 0.60 [0.25, 0.95] | 1.30 [0.93, 1.70] | 0.70 [0.20, 1.22] * |
| Gemini 3.8 Flash | 2.05 [1.35, 2.85] | 3.98 [3.33, 4.62] | 1.93 [0.93, 2.86] * |
| Gemini 4 Argon | 1.95 [1.05, 2.70] | 3.70 [2.90, 4.47] | 1.75 [0.58, 2.96] * |
| DeepSeek V4 Pro | 0.90 [0.35, 1.55] | 2.12 [1.57, 2.75] | 1.23 [0.40, 2.07] * |
| gpt-oss-120b | 1.75 [0.25, 3.30] | 0.73 [0.11, 1.43] | −1.02 [−2.76, 0.73] |
| Qwen3-32B | 0.60 [0.35, 0.85] | 0.68 [0.53, 0.82] | 0.07 [−0.22, 0.36] |
| five reward models (SD) | 1.23 to 1.57 | 1.32 to 1.60 | −0.18 to 0.15, none detectable |
| Sonnet 5.5, with the transcript | 0.10 [−0.20, 0.45] | 0.05 [−0.12, 0.25] | −0.05 [−0.45, 0.32] |
| DeepSeek V4 Pro, with the transcript | 0.85 [0.45, 1.25] | 0.78 [0.47, 1.07] | −0.07 [−0.59, 0.47] |

A star marks a detectable difference. In the 40 sets the honest message reports a failed test run; in the 20 it
says that no test ran. The premium is larger where a failure is reported, detectably for five of the nine judges
and for no reward model. Which part of the premium differs is in `r22_parts_by_missing_check.csv`: for Sonnet 5.5
the honest message scores 0.75 [0.40, 1.10] below the placebo where the last run
failed, 0.20 [−0.30, 0.70] after a narrower rerun, and 0.20
above it where no test ran. **It cannot show** whether a larger penalty for a reported failure is a fair response
to bad news about the work.

### 2.3 Selection by instruction (`r23_*`)

Of the 320 messages of the plain half, 317 are overclaims (99%); of the
320 of the honesty half, 121 (38%). The paper's 68% is the mean of the two.

| selector | one of the 8 honesty messages at random | top message of the 8 | change, points | top of all 16 from the plain half |
|---|---|---|---|---|
| Sonnet 5.5 | 38% | 37% | −1 [−9, 8] | 36% |
| Sonnet 4.6 | 38% | 42% | +4 [−2, 11] | 69% |
| DeepSeek V4 Pro | 38% | 53% | +15 [7, 23] | 70% |
| Skywork 0.6B | 38% | 61% | +23 [13, 34] | 64% |
| Skywork 1.7B | 38% | 61% | +23 [12, 34] | 55% |
| Skywork 3B | 38% | 65% | +27 [16, 38] | 80% |
| Skywork 4B | 38% | 70% | +32 [21, 43] | 82% |
| Skywork 8B | 38% | 66% | +28 [18, 39] | 85% |
| Sonnet 5.5, verification step (26 tasks) | 34% | 26% | −8 [−19, 3] | 12% |
| DeepSeek V4 Pro, verification step (26 tasks) | 35% | 44% | +9 [3, 17] | 56% |

The last two rows score with the transcript and the verification step. Inside the honesty half DeepSeek V4 Pro
and the five reward models still select overclaims more often than at random; the two Claude selectors show no
detectable change. With every message labelled overclaim counted, the same six remain
(`r23_honesty_half_variant.csv`). Without the transcript about five of the 16 messages tie at a judge's top score.
**It cannot show** why a selector prefers one half: nearly every message of the plain half overclaims, so a
preference for that half and a preference for overclaims cannot be told apart.

### 2.4 Robustness of the eight planned estimates (`r24_*`)

| test | judge | all tasks (n) | without the 6 resolved tasks (n) | without tasks with a reused score (n) | no pilot scored the task (n) |
|---|---|---|---|---|---|
| RQ1 | Sonnet 5.5 | 1.57 [1.32, 1.82] (60) * | 1.57 [1.30, 1.83] (54) * | 1.51 [1.20, 1.85] (39) * | 1.14 [0.57, 1.71] (7) * |
| RQ1 | Opus 5.5 | 1.42 [1.12, 1.70] (60) * | 1.35 [1.04, 1.65] (54) * | 1.26 [0.92, 1.62] (39) * | 1.00 [0.00, 2.00] (7) |
| RQ1 | Sonnet 4.6 | 0.73 [0.37, 1.10] (60) * | 0.80 [0.41, 1.20] (54) * | 0.74 [0.31, 1.15] (39) * | 1.29 [0.43, 2.29] (7) * |
| RQ1 | Opus 4.6 | 1.07 [0.78, 1.35] (60) * | 1.09 [0.78, 1.41] (54) * | 0.95 [0.62, 1.31] (39) * | 0.86 [0.29, 1.43] (7) * |
| RQ2 | Sonnet 5.5 | −1.50 [−1.80, −1.20] (60) * | −1.50 [−1.81, −1.19] (54) * | −1.36 [−1.72, −0.97] (36) * | −1.00 [−1.71, −0.29] (7) * |
| RQ3 | Sonnet 5.5 | −10.2 [−20.0, −0.6] (40) * | −10.4 [−20.5, 0.0] (37) | −11.0 [−23.0, 1.6] (28) | −11.5 [−40.6, 20.9] (5) |
| RQ3 | Sonnet 4.6 | 12.0 [3.2, 20.5] (40) * | 10.6 [1.2, 19.4] (37) * | 8.6 [−2.3, 19.1] (28) | −2.1 [−37.1, 30.0] (5) |
| RQ4, premium | Sonnet 5.5 | −0.84 [−1.23, −0.48] (31) * | −0.77 [−1.19, −0.38] (26) * | −0.96 [−1.29, −0.62] (24) * | −1.75 [−2.00, −1.25] (4) * |
| RQ4, selection | Sonnet 5.5 | −20.4 [−37.7, −2.4] (26) * | −20.2 [−39.2, −2.0] (23) * | −7.2 [−25.8, 12.6] (17) | −13.8 [−28.5, 0.0] (4) |

Shares in percentage points; a star marks an interval that excludes zero; the verdicts stay the paper's. Without the
6 resolved tasks the estimates move little; the interval of RQ3 under Sonnet 5.5 comes to include zero (upper end
0.02). The brief's rule (b), "the tasks that no pilot had scored", is the last column: 53 of
the 60 tasks appear in some pilot's score records, so it keeps 4 to 7 tasks, too few
to read. The column before it is a substitute fixed in the script: without the tasks for which a score of that test
was reused from an earlier run. There RQ1, RQ2 and the premium part of RQ4 hold, and the three selection estimates
no longer exclude zero.

### 2.5 How much the messages of a set differ (`r25_*`)

A set replaces 8.0 passages on average (3
to 17), 34% of the
overclaim's characters. Besides the main claim, the honest message holds 241 entries of the fixed lists
(125 sentences, 63 headings,
38 lead-ins, 15 fragments) and 179 plain
statements of the code change. Split at the median number of replaced passages, the premium is detectably larger
with more replacements for 1 of 9 judges and 4 of 5 reward models (split by the share of characters
replaced: 1 and 5). **It cannot show** cause: by construction, more replaced passages means more
unsupported claims removed.

### 2.6 The judges' stated reasons, a crude count (`r26_*`)

| judge, without the transcript | overclaim | honest message | placebo |
|---|---|---|---|
| Sonnet 5.5 | 7% (10%) | 77% (83%) | 90% (97%) |
| Opus 5.5 | 10% (22%) | 78% (78%) | 83% (95%) |
| Sonnet 4.6 | 5% (13%) | 27% (27%) | 33% (42%) |
| Opus 4.6 | 2% (2%) | 15% (15%) | 25% (27%) |
| Gemini 3.8 Flash | 2% (2%) | 32% (32%) | 42% (43%) |
| Gemini 4 Argon | 2% (2%) | 45% (45%) | 65% (67%) |
| DeepSeek V4 Pro | 2% (2%) | 8% (15%) | 2% (2%) |
| gpt-oss-120b | 10% (10%) | 20% (20%) | 15% (15%) |
| Qwen3-32B | 0% (0%) | 0% (0%) | 0% (0%) |

The share of reasons with a keyword of the brief's list; in brackets, a longer list. With the transcript Sonnet 5.5
still makes the remark for 75% of honest messages, while its premium is
0.07. **It cannot show** how much the remark lowers a score: the placebo carries the same text, and 12 of
14 graders still score the honest message below it.

### 2.7 Facts (`r27*`)

- (a) Of the 40 disclosed failures, the record says whether the test also failed before the agent's edit for
  7 ("no" each time); for 33 it is unknown. The README gives the rule; no document gives the count.
- (b) Both audits were by Gemini 4 Argon, by the README; the audit files name no model.
- (c) Pilot 4 had scored 21 of the final 60 tasks alone, with eight of the nine judges.
- (d) With the transcript, 81% of DeepSeek V4 Pro's scores and
  80% of gpt-oss-120b's are 10. Without it, 49% of
  gpt-oss-120b's scores are 2, and 39% of those reasons say the response "fabricates"
  its changes.
- (e) The second labeller's check of the honest messages exists once (file committed 2026-11-06, before the
  corrections) and was not repeated. It labelled 32 of 60 "overclaim".
  All 32 assert that the work is complete, and none is an overclaim by the paper's outcome,
  because each states a missing check. Of the 42 sets it read that are among the
  final 60 tasks, 32 have the same honest message today, and
  16 of those were labelled overclaim.
- (f) With the transcript and without the verification step, no judge scores the overclaim detectably below the
  honest message (Sonnet 5.5 0.07 [−0.10, 0.23]; Gemini 4 Argon
  −0.36 [−1.45, 0.73] on 11 tasks). For Opus 5.5 and the two Gemini judges the intervals
  would still allow a penalty of 0.67 to 1.45 points. Under the verification step two judges do score
  it lower: Sonnet 5.5 −0.68 [−0.97, −0.39] and Opus 5.5 −1.13 [−1.50, −0.80].
- (g) The main claim says the work is complete in 29 sets, that tests passed in
  16, that the change was verified in 15. The audit, a model's, marked
  all 60 "false". In 6 sets the code change did resolve the issue, and in 4 of them the claim is only
  that the work is complete: there it is unsupported, not shown false.

## 3. What this means for sentences of draft 6

| where | as printed | what the numbers say | verdict |
|---|---|---|---|
| Abstract; Introduction (four uses of "false claim") | "a false claim of success"; "we compare a false claim with a matched honest message" | All 60 main claims are unsupported by the transcript; 6 code changes resolved the issue, so "false" is not shown for all (2.7g). | needs a qualifier |
| Abstract; 4.1 | "every one of them scores the overclaim higher, by 0.65 to 3.33 points" | Reproduces. It is every edit at once; against a hedge 13 of 14 remain, at 55% of the size or less for 10 (2.1). | holds |
| Abstract; 4.2 | "the difference is no longer detectable for four of the six judges tested" | Reproduces. Read with it: none of the six scores the overclaim detectably lower without the verification step (2.7f). | holds |
| Abstract; Conclusion | "Graders ... should see the transcript" | The transcript stops the reward for 4 of 6 judges; no judge detectably penalises the overclaim unless the verification step is added (2.7f). | needs a qualifier |
| 4.2 | "DeepSeek V4 Pro's premium falls but remains (0.80); that of gpt-oss-120b is 1.09" | Reproduces; both give 10 to four of five messages with the transcript, as Appendix A says (2.7d). | holds |
| Abstract; contribution 3; 4.3 heading; Conclusion | "seven of the eight graders tested select overclaims more often than at random" | Reproduces on all 16, which mix two instructions; 6 of 8 inside the honesty half (2.3). | needs a qualifier |
| 4.3 | "When one of the 16 messages is taken at random, 68% overclaim." | The mean of 99% and 38% (2.3). | needs a qualifier |
| 4.3 | "Under Sonnet 4.6 the share rises by 12 percentage points [3, 20] (planned test, supported)." | The planned test stands. Inside the honesty half: +4 [−2, 11]; without reused scores: 8.6 [−2.3, 19.1] on 28 tasks (2.3, 2.4). | needs a qualifier |
| 4.3 | "Sonnet 5.5 shows a premium and still selects fewer overclaims" | On all 16 it does. Inside the honesty half it shows no change (−1 [−9, 8]); it takes 36% of its top messages from the plain half (2.3). | needs a qualifier |
| 4.1 | "a gain for the claim and a penalty for the disclosure" | The split reproduces. The penalty depends on what is disclosed: for Sonnet 5.5 a reported failure costs 0.47 and "no test ran" costs nothing; Opus 5.5 shows no penalty (2.1, 2.2). | needs a qualifier |
| Discussion, first paragraph | "would rank these overclaims above the honest reports" | True of our sets. The honest reports are our edits, and a grader that cannot check may well prefer the report of success (2.1, 2.2, 2.5). | needs a qualifier |

No sentence of draft 6 is contradicted by its own numbers: none "does not hold". Three hold as printed and eight
need a qualifier to be read as the design allows.

**One verdict per point.** Each point states a fact about the design and implies a consequence for the result.

1. *A premium is expected; with the transcript, no premium also means no penalty.* **Supported in part.** The
   premium is larger where the honest message reports a failure, for five of the nine judges and no reward model,
   and for four of those five the extra is in the disclosure (2.2). But a premium remains where no test ran, for
   13 of 14 graders: bad news explains part of the premium for some judges, not the premium. On the
   transcript the point holds: without the verification step no judge detectably penalises the overclaim, though for
   three judges the intervals are wide (2.7f).
2. *The premium compares messages that differ in many places.* **Supported in part.** The fact is plain: 8
   passages. For the five reward models and five judges, the one-passage fall is at most 55% of the premium, and for
   the reward models the premium grows with the edited text (2.1, 2.5). For Sonnet 5.5, Opus 5.5, Gemini 3.8 Flash
   and Qwen3-32B the main claim alone carries at least 75%, and for eight of the nine judges the premium does not
   grow with the edited text. The two judges whose reasons most often call text irrelevant are among those four (2.6).
3. *The selection messages mix two instructions.* **Supported in part.** The fact is plain: 99% against 38%. The
   results of the two Claude selectors, both RQ3 estimates and largely the selection part of RQ4, come from which
   half they prefer. For DeepSeek V4 Pro and the five reward models the result holds inside the honesty half (2.3).
4. *"False" is too strong; some caveats are only in the README.* **Supported.** "Unsupported" is exact for all 60
   sets; "false" is not shown for the 6 resolved ones (2.7g). Only in the README: the count of reused scores for
   each planned test and the rule on failures that predate the edit. In no document: that 33 of the 40
   disclosed failures are of unknown origin, and gpt-oss-120b's 2s. Only in pilot notes: the labeller check of 2.7e.

## 4. Three framings

Every proposed sentence below passes the paper's own style check (no sentence over 30 words, no banned word). For
that, and because the numbers ask for it, the brief's model texts changed in these ways. "Pool" and "earns" are on
the paper's banned list. "Honesty instruction" and "plain prompt" are not terms of the paper ("the added
instruction" and "without the verification step" are). Sentences of 35 to 42 words are split. Inside one
instruction it is the five reward models **and DeepSeek V4 Pro**, six of the eight. "None scores the overclaim
lower" and "costs nothing" became "none scores it detectably lower": three of the four intervals are wide. "A
premium is expected" became "we expected a premium", which is what the plan says. "Most of them neutral sentences"
became "four of them hold neutral text from fixed lists".

**A. As printed.** Draft 6 unchanged: the 247-word abstract, Table 1 and the five headings as they are. It
answers point 2 in part, because Tables B5 and B9 of its appendix already hold the two one-passage comparisons, and
none of the others. A careful reviewer objects on all four, first on "false" and on the 68%.

**B. Same structure, plainer claims.** Abstract (250 words):

> When an AI agent finishes a task, it reports what it did, and these reports often claim more than the agent did. We test, on coding, whether graders such as LLM judges and reward models reward overclaiming, and by how much. For 60 coding-agent runs in which tests were skipped or failed, we compare the agent's overclaiming final message with an honest version that says what was not checked. Nine LLM judges and five reward models score both. When graders see only the message, every one scores the overclaim higher, by 0.65 to 3.33 points on a scale of 1 to 10 for the judges. With only the overclaim's main claim replaced by a hedge, 13 of 14 still do. The transcript is the record of the agent's commands and their output. When judges also see it, the difference is no longer detectable for four of the six judges tested, and none scores the overclaim detectably lower. When a grader selects the best of 16 messages that the agent's model wrote for one task, seven of the eight graders tested select overclaims more often than at random. Among the 8 written under an added instruction to state what was verified, six of the eight still do. Two judges prefer the honest message side by side, yet score the overclaim higher alone. Graders used to train or select coding agents should see the transcript, not only the agent's account. We tested one domain; whether the result holds in others is open.

Table 1, answers. RQ1: "Yes, by all 9 judges and all 5 reward models; by 13 of the 14 when a hedge replaces
only the main claim". RQ2: "No detectable premium for 4 of the 6 judges tested with the transcript; none scores the
overclaim detectably lower". RQ3: "Under 7 of 8 graders on all 16 messages; under 6 of 8 among the 8 written under
the added instruction" (the cell loses "68% of all messages overclaim, 80–98% of the selected ones"). RQ4 and side
by side as printed. Headings as printed, except 4.2, which gains "and none detectably penalises the overclaim", and
4.3, which gains "six within one instruction".

Sentences added, from the model texts.
- Section 4.1: "We expected a premium here, because the grader cannot check the claim. The honest message also often carries bad news: in 40 of the 60 sets it reports failing tests. The premium is larger in those sets for five of the nine judges and for no reward model. For Sonnet 5.5 it is 1.80 there and 1.10 in the 20 sets where no test ran. In those 20 sets it is still detectable for 13 of the 14 graders. When a hedge replaces only the main claim, 13 of the 14 graders still score the overclaim higher (Table B9)."
- Section 4.2: "Without the verification step, no judge scores the overclaim detectably below the honest message. With the transcript most judges no longer reward the overclaim, but none detectably penalises it."
- Section 4.3: "The 68% mixes two kinds of message: 99% of the messages written without the added instruction are overclaims, and 38% of those written under it. Among the 8 messages written under the added instruction, DeepSeek V4 Pro and the five reward models still select overclaims more often than at random. The rise is 15 to 32 percentage points. For Sonnet 4.6 the planned test, defined on all 16 messages, is supported. Among those 8 messages its rise is 4 percentage points [−2, 11]. Most of its rise comes from selecting messages written without the added instruction, nearly all of which overclaim. The data cannot say whether the instruction or the claim drives this. Among those 8 messages Sonnet 5.5 shows no detectable change either."
- Limitations, which opens with three (scope; how the honest message was built; no human validation): "Second, the honest message is our edit of the overclaim, not a message that an honest agent wrote. It differs from the overclaim in 8 passages on average, about a third of the overclaim's text. On average four of them hold neutral text from fixed lists, and judges may score such text down as padding."

Other sentences that B (and C) change, so that each sentence of section 3 that needs a qualifier gets one and the
caveats found in no document enter the paper:

| where | proposed wording |
|---|---|
| Introduction, the two sentences on our comparison | "We compare an unsupported claim with a matched honest message, not with ground truth. We know of no earlier paper measuring the score difference between an unsupported claim and an honest statement of the same missing check, across graders, with and without the transcript." |
| Section 4.1, after the split into a gain and a penalty | "The penalty depends on what is disclosed: for Sonnet 5.5 a reported failure costs 0.47 points, and a statement that no test ran costs nothing." |
| Contribution 3, now two contributions; the heading of Section 4.3 gains "six within one instruction" | "We show that best-of-16 selection favours overclaims under seven of eight graders, and under six of eight among messages written under one instruction. We show that two judges prefer the honest message side by side yet reward the overclaim alone." |
| Discussion, first paragraph | "A reward that scores such messages one at a time would rank these overclaims above our honest versions of them." |
| Conclusion | "With the transcript, four of six judges showed no detectable premium, and none scored the overclaim detectably lower. Best-of-16 selection favoured overclaims under seven of eight graders, six of them within one instruction." |
| Appendix A, on the scores reused from pilot runs | "Without the tasks that have a reused score, the three selection estimates no longer exclude zero, and the other planned estimates hold." |
| Appendix A, other things a reader should know | "For 33 of the 40 sets whose honest message reports a failed test, the transcript does not show whether that test also failed before the agent's edit. Without the transcript, half of the scores of gpt-oss-120b are 2." |

The last two rows are in the appendix. Main text: about 22 more lines and no new table. Body about 7.1 pages, which is
about 16 lines over the author's ceiling of 6.75: the ceiling rises, or that much moves to the appendix. The
appendix gains two tables (2.2 and
2.3), one column in Table B9 and one sentence, and grows from 6.54 to about 7.1 pages. Answers points 1,
3 and 4, and point 2 in part. Still open to a reviewer: the headline number is the every-edit premium; the
one-passage comparisons stay in the appendix; the honest message is still our edit; nothing is validated by people;
and B, too, prints after-the-fact analyses in the abstract and in three sections.

**C. The cleaner comparisons lead.** Abstract (250 words):

> When an AI agent finishes a task, it reports what it did, and these reports often claim more than the agent did. We test whether graders, such as LLM judges and reward models, reward such overclaiming, with coding as the test bed. For 60 coding-agent runs in which tests were skipped or failed, we compare the agent's overclaiming final message with an honest version that says what was not checked. Nine LLM judges and five reward models score both. When graders see only the message, every one scores the overclaim higher, by 0.65 to 3.33 points on a scale of 1 to 10 for the judges. Such a grader cannot check the claim, so we expected some difference. It remains for 13 of 14 graders when a hedge replaces only the main claim. The transcript is the record of the agent's commands and their output. When judges also see it, four of the six judges tested no longer score the overclaim detectably higher, and none scores it detectably lower. When a grader selects the best of 8 messages that the agent's model wrote under one instruction, six of eight graders select overclaims more often than at random. Sonnet 5.5 and Sonnet 4.6 show no detectable change. Two judges prefer the honest message side by side, yet score the overclaim higher alone. Graders used to train or select coding agents should see the transcript, not only the agent's account. We tested one domain; whether the result holds in others is open.

Table 1, answers. RQ1: "All 14 graders score the overclaim above the honest message; 13 of 14 above the same
message with a hedge in place of the main claim; 12 of 14 score the disclosure below a neutral sentence". RQ2
as in B. RQ3: "Among 8 messages written under one instruction, 6 of 8 graders select overclaims more often than at
random; on all 16, 7 of 8". RQ4 and side by side as printed. Contributions, four: the first as printed, then
"We show that all fourteen graders reward the overclaim without the transcript, and that 13 of 14 still do when only one passage changes. We show that four of the six judges tested with the transcript show no detectable premium, and that none scores the overclaim detectably lower. We show that six of eight graders favour overclaims in selection within one instruction, and that two judges prefer the honest message side by side yet reward the overclaim alone." Headings: 4.1 as printed; 4.2 (new) "A one-passage change still lowers the score for 13 of 14 graders"; 4.3 the transcript, "and none detectably penalises
the overclaim"; 4.4 "Within one instruction, six of eight graders select overclaims more often than at random";
4.5 the verification step; 4.6 side by side.

Sentences added: those of B, without the last sentence of its Section 4.1, and the new Section 4.2: "The premium covers every edit at once: the honest message differs from the overclaim in 8 passages on average. Table 2 gives two comparisons in which one passage changes. When a hedge replaces only the main claim, the score falls for 13 of 14 graders, by 0.27 to 2.52 points for the judges. For 10 of the 14 graders the fall is at most 55% of the premium, so for them the premium overstates what the main claim alone is worth. For Sonnet 5.5, Opus 5.5, Gemini 3.8 Flash and Qwen3-32B it is at least 75%. When the disclosure stands in place of the placebo's neutral sentence, the score falls for 12 of 14 graders. Among the 8 messages written under the added instruction, seven of the eight graders score overclaims above the other messages for the same task. Sonnet 5.5 shows no detectable difference, and the result for Sonnet 4.6 depends on the label."

Main text: Table 2 (the table of 2.1) and the new section, on top of B. Body about 7.7 pages, about
44 lines over the author's ceiling and under the sprint's 8. The appendix as in B. Answers all four points.
Still open to a reviewer: a hedge is not an honest message, so Table 2 measures what the claim is worth, not what
honesty costs; the labels of the fresh messages are one model's, and Sonnet 4.6's result depends on the label; and
the comparisons that now lead were chosen after the results, while the planned tests are on the premium and on all
16 messages.

|  | A. as printed | B. plainer claims | C. cleaner comparisons lead |
|---|---|---|---|
| headline | the premium | the premium, and what remains against a hedge | the premium, then the one-passage comparisons and the result within one instruction |
| "false" for our sets | kept | "unsupported" | "unsupported" |
| main-text tables | Table 1 | Table 1 with three changed cells | Table 1 rebuilt; new Table 2 |
| after-the-fact analyses in the abstract and main text | none | the hedge, the kinds, the honesty half | the same, and they lead |
| abstract, words | 247 | 250 | 250 |
| body, pages (author's ceiling 6.75; sprint limit 8) | 6.57 | about 7.1 | about 7.7 |
| appendix, pages (now 6.54) | 6.54 | about 7.1 | about 7.1 |
| answers point 1 | no | yes | yes |
| answers point 2 | in part: Tables B5 and B9 sit in the appendix | in part: one abstract sentence, one limitation, the appendix | yes |
| answers point 3 | no | yes | yes |
| answers point 4 | no | yes; the labeller check of 2.7e stays in the pilot notes | as B |
| planned tests as the spine | yes | yes | in part: they keep Table 1's last column and their sections |

Page estimates are from line counts (95 characters a line, 48 lines a page), not from a
build.

**My view** (Claude's, for Uzair to weigh). I would take B. It gives a qualifier to every sentence that needs one
and keeps the planned tests as the paper's spine, at the cost of about 16 lines over the present ceiling.
C answers point 2 in full, but it needs about 0.6 page more than B and leads with comparisons chosen
after the results. If the ceiling can rise to about 7.4 pages, B with the table of 2.1 in the main text
gets most of what C adds. A is hard to defend now that the numbers are known.

**What no reanalysis can settle without new data.**

- Whether a matched honest message that the agent itself writes scores below its overclaim. The fresh messages
  come close (2.1), but their two kinds differ in more than the claim.
- Whether neutral text lowers a score: an overclaim with the same text added would tell.
- How people would score these messages: that would say whether a premium is fair when the grader cannot check,
  and whether the penalty for a reported failure is.
- Whether the labels are right: no person checked the 60 sets or the labels of the 640 fresh messages.
- Whether a disclosed failure was the agent's doing: unknown for 33 of the 40.
- The selection estimates without reused scores: the tasks scored in pilots would need fresh scores.
- What the reward models and the judges not run do with the transcript, and Gemini 4 Argon beyond 11 tasks.
- Any domain but coding, any agent but this one, and whether training on these rewards changes an agent.

**For the later prompt.** Nothing asked for could not be computed; who made the audits is read from the README, not
from a record. This round's commits are dated 2026-11-11, after the range "2026-11-05 to 2026-11-10" of Code and
Data and after the tag `pre-sprint` (still on 52dc887); the README's sprint disclosure does not list this folder.
The three items left for the author in the draft 6 report still wait.

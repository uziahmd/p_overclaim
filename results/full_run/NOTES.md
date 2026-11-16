# Session 5, Part 2: notes

Running notes of the full run. Section 1 is the state; section 2 the freeze (Step 8, done on 2026-11-08 on Uzair's
start line); section 3 the plan; section 4 the questions for Uzair; section 5 the code of this step. Everything here
is from files on disk (`results/full_run/freeze.md`, `results/full_run/projection.md`,
`data/processed/full_run_tasks.json`); nothing in sections 2 to 5 is a result of the full run. Section 6 is the run log of
Step 2, written as the stages run.

## 1. State

- **2026-11-08, 23:20 local: the memo's rows are done and analysed** (1a71dfb): the six Gemini rows on the aistudio
  route, rows 1 to 5 complete and the Argon shown row partial at 11 of 34 tasks, 11.59 of the 11.70 USD cap; the
  DeepSeek band row complete; the qwen shown row still partial (16 of 102; SiliconFlow 429 on every try). The
  analysis was rerun: the Holm family, every Claude row, E5, E6 and E4 unchanged (within 1e-9); new: Flash shown
  -0.24 [-0.88, 0.35] (n = 34) and Argon shown -0.36 [-1.45, 0.73] (n = 11) against hidden premiums above 3, E7 on
  Flash flat (plain -0.17, instruction-only +0.47, verification-first +0.13, H4a no difference), DeepSeek band 0.73
  [0.35, 1.08] (pays with the transcript shown), Gemini hidden pairwise O-vs-H 0.98 (Flash) and 0.61 (Argon). Spend:
  Anthropic 71.76 (project 149.58 of 180), Google 12.76 (1.17 credit + 11.59 cash), OpenRouter 7.06 of 19.00. Still
  open: 23 Argon shown tasks (Vertex, if the appeal clears before 2026-11-12 09:00 UTC) and 86 qwen records.
- **2026-11-08, 21:40 local: Uzair's memo on the open rows arrived** (README amendment 3, item 13): the Vertex
  project is restricted (appeal filed, no date); the six open Gemini rows move to the `aistudio` route under a hard
  cash cap of 11.70 USD on that route's cost lines (the AI Studio balance of about 12.9 USD less a 10 percent
  margin), in the memo's priority order (Flash shown with the two E7 scorers; Flash and Argon hidden pairwise; Argon
  shown, whatever fits), each row priced from its projection row until 15 live calls measure it, the longest task
  prefix that fits otherwise; the qwen shown row resumes with one worker, the 10-to-320 s backoff and the fp8/bf16,
  65,536-token provider rule; the DeepSeek band row (26 tasks, about 2 USD) is added as replication. The code landed
  in bf9b36c (route caps in the cost logger, `route_cap.py`, partial statuses, a quota-429 stop, `--only`), the
  decisions are in item 13, the run log is section 6 ("Memo of 2026-11-08").
- **2026-11-08, 18:50 local: the full run is done through the analysis** (`results/full_run/summary.md`, the figures
  in `results/full_run/figures/`), with two rows open: the Gemini shown rows (Flash plain, E7 and verification-first;
  Argon plain) and the two Gemini hidden pairwise rows did not run because the Vertex project is refused since 16:17
  (HTTP 403, policy), and the qwen3-32b shown row is partial (15 of 102 records; provider 429s). When the project is
  back: `--stage pointwise_shown`, `--stage pairwise`, `--stage reasons`, `--stage analysis`, in that order, each
  resuming from its records; the Holm family does not involve any of these rows. Spend: Anthropic 71.68 USD of the
  102.18 remainder (margin 30.50; the plan said 82.41), OpenRouter 5.87 of 19.00, Google 1.17 on the credit; project
  total on Anthropic 149.50 of 180. Headline: every grader pays the hidden premium (the four Claude judges 0.73 to
  1.57 score points, all Holm-supported); showing the transcript removes it for Sonnet 5.5 (H2 supported, shown
  premium 0.15 [-0.09, 0.41]) and Opus 5.5; best-of-16 under Sonnet 4.6 raises the clean-overclaim rate (+0.12, Holm-
  supported) and under Sonnet 5.5 lowers it (-0.10); the verification-first scorer lowers the shown premium (-0.84)
  and the mitigated best-of-16's clean-overclaim rate (-0.20 [-0.38, -0.02]), so H4 holds unadjusted (p 0.029) but
  not after Holm (adjusted 0.058). Pairwise hidden, Sonnet 5.5 and Opus 5.5 prefer H over O (O-vs-H 0.21 and 0.00)
  while their pointwise scores pay the premium: the hidden-pair disagreement, in the appendix.
- 2026-11-08, 13:59 local: **Uzair's run line arrived** ("Part 2 run. Replace: pyiron_base-641, ParlAI-1869 ...
  smatch-30 stays."), with the answers to questions 1, 2 and 7 (README amendment 3, items 10 and 11). The re-freeze
  ran with `--supersede` (section 2, item 11): the first freeze is kept under `*.superseded-20261107T172655Z.*`;
  the new sets file's sha256 is `fb4f87be8afecc1cba49c8fae3610511f4dd3c5fd64c0a09c3b5e24e5e1311d9`, the task
  file's `6626210afe1c5943ba04cbfb92c6d97a7e2effa1f86969f84d129676968aebff`. The plan re-priced: final 82.41 USD,
  margin 19.77, A2 the only cut. The pinned tests were updated. Step 2 (the runs) starts from this freeze; the run
  log is section 6.
- 2026-11-08, later: Uzair's memo answered questions 3 to 6 (README amendment 3, item 9) and left the set choice to
  their read of the three printed sets; the choice is pending (section 4, question 1). The runner stages are being
  built offline meanwhile; `bitsandbytes` 0.50.2 is installed and its nf4 kernels run on the GPU.
- 2026-11-08, later still: the runner is built and committed (section 5). Every stage dry-runs on the frozen files
  without a call: pointwise_hidden (10 runs, 2,880 records), tie_rule, rm (5 models), pointwise_shown (7 runs: 795
  plain, 180 E7, 285 E3 records), e2 (640 samples, 1,280 labels, 3 plain and 1 mitigated LLM selectors, 5 reward
  models; the optional open-weight mitigated selector skipped by the projection), pairwise (10 runs, 3,450
  records; the shown Sonnet 5.5 row kept, A3 not cut), reasons, analysis. `accelerate` 1.15.0 (transformers' 4-bit
  loader needs it) and `matplotlib` 3.11.2 (the figures) are pinned dependencies now.
- Step 1 of the two-step start (addendum section 7) is done: the audit file fixed, the repairs and replacements
  applied, the task list frozen with hashes, amendment 3 written, the plan priced row by row. No scoring call, no
  model call, no GPU use. Step 2 starts on "Part 2 run" (or "Part 2 run. Replace: <task ids>").
- Spend of Part 2 so far: 0 on every provider (`results/full_run/cost.jsonl` does not exist yet).
- The audit files: `data/processed/coding/human_check_v4_final_{a1,a2,final}.csv`, committed as received (md5 of
  `_final` 03d9b794003d7d24f5f4ca78b404bd69), then "do not use" added to the notes of the four reserve rows the
  addendum names (md5 after 0dcae356991352c93f47e368dd1c51b1). A duplicate directory
  `data/processed/coding/human_check_v4_final/` holds identical copies and is left uncommitted. Kappa of the two Argon
  passes: `results/full_run/human_check_kappa_a1_a2.txt`.

## 2. The freeze: decisions and surprises

1. **The audit file matched the addendum's lists exactly**: 21 main rows with a `no`, 19 of them only on
   `O_other_claims_supported` (the repairs), 2 the replacements; the four do-not-use rows; span numbers identical.
   Six reserve rows fail only `O_other_claims_supported` and are in neither list (bork-154, PyPSA-924,
   mne-python-12537, HARK-1039, stellarphot-298, pylutron-caseta-20): under the replacement rule's letter they are
   not sound; they would be repairable under the repair rule (question 6 below).
2. **Repairs without a model call.** A relabelled span takes a bank filler by the loop of `build_v4_set` step 11,
   with every filler already in the set excluded, so no entry repeats within a message; the worst length deviation is
   0.087 of the span's length (tolerance 0.15). Deviation from the v4 construction, recorded in amendment 3: a
   relabelled list item under a line that announces edits would have been a fact item and received an Opus rewrite
   from the patch; the addendum's rule gives it a filler. `style_in_edit_list` rose in four sets: ParlAI-1869 1 to 8,
   dwave-system-279 3 to 5, pyiron_base-641 and hashin-88 0 to 1. The post-check was not re-run (the addendum asks
   for lint and length match only). Lint and the four assembly invariants of step 13 passed on all 19 sets.
3. **Sets with no supported span left.** After the repair five sets keep no kept span (pyiron_base-641, borg-7517,
   ParlAI-1869, jdaviz-2485, python-firestore-16); three main sets had none before the repair (azure-cli-4265,
   borg-7492, scanpy-2893), so 8 of the 60 sets carry no supported claim. Their H still holds the agent's description
   of the edit and the problem (text outside the spans); whether that is "a content-bearing sentence beyond the edit
   description" is Uzair's read (question 1).
4. **E4.** `resolved` is 1 for 6 of the 60 tasks, all `no_tests_run`; the `tests_failed` and `tests_narrowed` tasks
   are unresolved by construction (their last test run failed). The split 6 versus 54 gives the resolved stratum a wide
   CI. Second finding: every Experiment 2 sample of a task is a final message for the same trajectory and patch, so
   the "resolved rate of the selected sample at N = 1, 4 and 16" cannot vary with N or selector; it is one number
   (the resolved share of the Experiment 2 tasks) and is reported as such. The task files carry no `model_patch`, so
   the freeze checks the set's patch against the trajectory's raw row instead (all 60 match).
5. **Picks.** Experiment 2: 40 tasks by `prep5_e2.pick_tasks` on the frozen 60 (14, 13, 13 per gap type); all 10 of
   the pilot's tasks are among them, so their 160 samples come from the disk cache; 26 of the 40 are shown-eligible
   (the mitigated best-of-16). E7: `tests_failed` has only 8 shown-eligible tasks, so the pick is 11, 8, 11 (the
   shortfall split as Step 1.4 of the brief splits one). Shown-eligible 35 (36 before the replacements: nilearn-3823
   is eligible, cmdstanpy-677 is not, and both outgoing sets were eligible); band 25.
6. **Hashes.** `data/processed/full_run_tasks.json` holds the sha256 of O, H, P, O+, O- and of the record of every
   set, the sets file's sha256 (`303ebf8f9619d8e464bd99f757bc20732dc5bed1a238173ef078a93c5ea1a41d`) and the hashes of
   the four inputs. A later freeze run that would write anything different fails and writes nothing.

7. **The reserve rows' audit notes are partly misattributed** (found 2026-11-08 while adding the repairable reserve).
   Checked every audit row with a `no` or a note against its own set: the main rows are consistent (all 52 repaired
   spans are kept spans; pygmt-619's quote is script output; dwave-system-279's quote paraphrases span 4; codalab-3393's
   note says kept 10 where the kept spans are 2, 3, 8, 9, 11, a flagged-but-kept row). Of the six repairable reserve
   rows, four name spans that are not kept spans (mne-python-12537, HARK-1039, stellarphot-298, pylutron-caseta-20),
   PyPSA-924's quoted sentence is in MetPy-3103's message, stellarphot-298's in py-geth-221's, and bork-154's quotes are
   in no message. Only bork-154 [4] and PyPSA-924 [5] have usable lists. The freeze refuses a repairable row whose
   spans are not kept spans; none is used by either run line. Uzair should re-read those six rows before any is used.
8. **Freeze, second edition** (`--supersede`): a re-freeze renames the earlier freeze under its UTC stamp
   (`*.superseded-20261107T172655Z.*`), records it in the new task file, and carries the E2 and E7 picks with a
   replacement substituted where the replaced task was picked (an E7 substitute must be shown-eligible). Dry run of
   the recommended line: pyiron_base-641 is replaced by mne-python-12033 and ParlAI-1869 by sunpy-6581; 17 sets stay
   repaired (36 spans); shown-eligible 34, band 26; ParlAI's E2 slot goes to sunpy-6581, pyiron's E7 slot to
   scanpy-2893 (mne-12033 is not shown-eligible); E2's shown-eligible count stays 26 and E7's split 11, 8, 11.

9. **Rules the runner fixes, recorded** (the implementers' readings, accepted by the lead): a grader's tie rate is
   decided only once every frozen task has its hidden O and H records (after its first hidden pass); the shown
   stage refuses to start until `tie_rule.json` exists and names the judges still undecided; the budget clause of
   amendment 3 item 9 is the config key `pointwise_shown.second_sample_hidden_only` (the lead lists a qualifying
   Claude judge there when the margin does not cover its shown second samples); a failed probe or a failed parse
   smoke test fails that run only, not the stage, and is never relaunched; a reward model that fails to load or
   score is named and the stage goes on (exit 0); records outside the frozen 60, or shown records outside the
   shown-eligible and band tasks, are counted and left out of every table; E7 on Flash follows the E7 task list,
   so cut C2 would shorten both judges' lists together; OpenRouter runs are gated on the 17.00 stop line through the
   run's own cost file; a Gemini run without `route: vertex` is refused before any call.
10. **Open-judge context.** The longest rendered shown prompt is 40,128 o200k tokens (nomenclature-320's H); 18 of the
    35 shown-eligible prompts exceed 32,768. qwen/qwen3-32b's fp8 endpoints serve 40,960 (DeepInfra), 20,000
    (Tenstorrent) and 131,072 (SiliconFlow) tokens; gpt-oss-120b's serve 131,072. The Qwen tokenizer counted 0.61 of
    Sonnet's tokens in pilot 3, so the prompts should fit the 40,960 endpoint; a call that does not fit fails its
    run, which the stage records and the next invocation resumes past.
11. **The re-freeze, as done (2026-11-08, 13:59 local; `frozen_at` 2026-11-08T04:59:42+00:00).** Exactly the dry run
    of item 8: pyiron_base-641 -> mne-python-12033 (rank 2, sound, not shown-eligible, 0 kept spans, 2 fillers),
    ParlAI-1869 -> sunpy-6581 (rank 5, sound, shown-eligible, 5 kept, 2 fillers); 17 repaired sets, 36 spans;
    shown-eligible 34, band 26; E2's slot of ParlAI to sunpy-6581 (26 shown-eligible among the 40), E7's slot of
    pyiron to scanpy-2893 (11, 8, 11). Seven sets keep no supported span (azure-cli-4265, mne-python-12033,
    borg-7517, borg-7492, jdaviz-2485, python-firestore-16, scanpy-2893). Filler dose: mean 4.0, median 3.5, max 10
    (borg-7492, smatch-30, MetPy-3103); by gap type 4.95 / 3.5 / 3.6. One runner change made for it: Opus 5.5's 30
    shown tasks are read from the frozen E7 pick (the same seeded pick at the first freeze) instead of being
    recomputed, so the list is carried like the E7 list (README amendment 3, item 10). The superseded files are
    committed next to the new ones.

## 3. The plan (results/full_run/projection.md)

- **Re-priced after the re-freeze (2026-11-08)**: full plan 93.22 USD; A2 (10.81) applied; final 82.41 (Tier 1
  63.74, Tier 2 18.66), margin 19.77; OpenRouter 4.71 of 19.00; Vertex 19.08 (29.61 after Argon's price doubles).
  The lines below are the first freeze's figures, kept for the record.
- Anthropic: full plan 93.49 USD (Tier 1 63.87, Tier 2 29.63); threshold 87.18; cut A2 (pairwise shown Opus 5.5,
  10.80) applied; final 82.68, margin 19.50. The section-5 identity: 84.58 - 7.28 + 8.98 + 2.88 (E6, Claude) + 4.54
  (E7) + 0.04 (smoke test) + 0.02 (144 more reason calls) = 93.75 on the Part 1 counts; 93.49 on the frozen counts
  (35 shown-eligible, 25 band).
- OpenRouter: 4.79 USD of the 19.00 remaining. The optional mitigated open-weight selector is priced at 14.54 USD on
  Sonnet 5.5's token profile at DeepSeek list prices (no prompt cache); with it the total would be 19.33, above the
  17.00 stop line, so it is not counted. The basis is conservative: on qwen3-32b's tokenizer ratio the reviewer's hand
  figure is about 9 USD, which would fit. Decision: re-price it from the judge's own measured calls once its hidden
  rows have run, and launch it only if the total with it stays under 17.00.
- Google (vertex): 19.40 USD at current prices, 30.16 after Argon's price doubles on 2026-12-01, against the credit of
  about 280 USD; zero cash. Every Argon row is scheduled before the end of November.
- Tie rule: no pilot judge is above 0.5 (highest gpt-oss-120b, 0.481); the contingency for a second sample on every
  pointwise row is 48.46 USD on Anthropic, which would not fit the margin for the Claude judges; if a Claude judge
  qualifies on Part 2's hidden scores, the second sample runs on its hidden rows first and the plan is re-printed
  before the shown rows (amendment 3, item 6).
- Rulings on the reviewers' points: cut B3 removes Opus 4.6's E6 rows with its hidden pointwise row (the levels are
  read against that row's O and H; the cut is not reached); `pointwise.check_question` stays (a rubric name that does
  not match its question raises before any call).

## 4. Questions for Uzair

1. **Answered by the run line of 2026-11-08: pyiron_base-641 and ParlAI-1869 rejected, smatch-30 stays** (amendment
   3, item 10). The question as put: **The three printed sets** (`results/full_run/freeze.superseded-20261107T172655Z.md`). Facts for the read, added 2026-11-08: the filler dose
   (non-primary spans that are bank fillers) is 12 in pyiron_base-641 and 9 in ParlAI-1869, against a mean of 4.3
   and a median of 4 over the 60; the next highest are borg-7492, smatch-30 and MetPy-3103 at 10, so replacing the
   first two lowers the maximum dose to 10, not 6. smatch-30's numbers come from the repository's own code: the
   agent ran `python smatch.py -f ...` (F-score 0.81 before, 0.85 after) and its test scripts `import smatch`,
   which imports `amr.py`; no reimplementation was found in the script runs. The two sound `no_tests_run` reserve
   rows: mne-python-12033 (rank 2) is not shown-eligible (42,199 tokens, in the band) and already keeps no
   supported span (2 fillers, 10 rewrites); sunpy-6581 (rank 5) is shown-eligible and keeps 5. pyiron_base-641 is
   in the E7 pick (not in Experiment 2); ParlAI-1869 is in the Experiment 2 pick (not in E7). pyiron_base-641 and ParlAI-1869 keep no supported
   span; their H holds the problem analysis and the list of edits (text outside the spans) and then style remarks;
   smatch-30 keeps 5 supported spans. Which, if any, do you reject?
2. **Answered: moot with two**; a third would come from the repairable reserve when it has a `no_tests_run` row,
   otherwise from another gap type by Step 1.4, keeping 60 (amendment 3, item 9). As put: **If all three are rejected.** All three are `no_tests_run`, and the sound unused `no_tests_run` reserve rows are
   two: mne-python-12033 (rank 2), then sunpy-6581 (rank 5). A third rejection needs a decision: 59 sets, a repair of
   one of the six failing reserve rows (question 6), or a replacement from another gap type.
3. **The optional mitigated open-weight selector**: as in section 3, decided after the judge's measured profile; say
   if you want it dropped outright or run regardless of the stop line.
4. **The 8B reward model in 4-bit** needs the `bitsandbytes` package, which is not installed (torch 2.15 cu132,
   transformers 5.22 are). Installing it is a dependency change; without it the 8B is skipped with that reason and the
   scan is 0.6B, 1.7B, 3B, 4B. Install, or skip?
5. **Cut B3's scope** (section 3): keep the wider reading, or the literal one?
6. **The six reserve rows** that fail only `O_other_claims_supported`: leave them out (current) or make them
   repairable reserve under the repair rule, should a replacement be needed?
7. **Answered: the sentence stays and amendment 3, item 11, declares one family of eight tests** (H1 for the four
   Claude judges, H2 for Sonnet 5.5 on its 60 shown tasks, H3 for the two plain Claude selectors, H4 for Sonnet 5.5
   as an intersection-union test of H4(i) and H4(ii)), CI-inversion p-values from the same bootstrap, a Holm
   adjusted-verdict column next to the unadjusted CI; the analysis code for it is being added (section 5). As put:
   **Holm correction.** The original pre-registration applies a Holm correction across the four primary tests. The
   per-grader pointwise rules of amendment 2 (a CI per judge) do not say what the family is. Until a decision, the
   analysis prints the verdicts uncorrected and says so. Options: withdraw the sentence in an amendment, or declare
   the family (for example H1 hidden per Claude judge, H2 for Sonnet 5.5, H3 for Sonnet 4.6, H4a for Sonnet 5.5) and
   add an adjusted-verdict column.

## 5. Code of this step

- `src/overclaim/repair_v4.py` (new): `relabel_spans`, `dose_variants`, the two E6 sentences.
- `src/overclaim/experiments/full_run.py` (new): the Part 2 runner, stage `freeze` (later stages are added before
  Step 2); `configs/full_run.yaml`.
- `src/overclaim/projection_v4.py` (additions): the `profile_of` price method, constant factors, E7 counts, the Vertex
  credit and OpenRouter stop-line lines; `configs/projection_full_run.yaml`; Part 1's projection output is
  byte-identical under the new code.
- `src/overclaim/graders/pointwise.py`, `src/overclaim/experiments/e1b_pointwise.py` (additions): the full-rubric
  question, the E7 instruction, `question_for`, `check_question`, `prompt_hashes`, the resume key with the rubric
  (`_rubric_key`); today's prompt, schema and cache key unchanged.
- `configs/models.yaml`: the judge `deepseek-v4-pro` and its price row; `reward_models_part2` (the E5 scan).
- The runner stages, each in its own module: `full_run_pointwise.py` (pointwise_hidden with E6 and the full-rubric
  smoke test, tie_rule, rm), `full_run_shown.py` (pointwise_shown with the E3 and E7 scorers), `full_run_e2.py`
  (Experiment 2 with the mitigated, open-weight and reward-model selectors; `e3_verification_pointwise.make_vfp_scorer`),
  `full_run_pairwise.py` (pairwise, reasons; `analysis_reasons_v5.py`, the five-field classifier),
  `full_run_analysis.py` (`analysis_full_run.py`, `figures_full_run.py`). `full_run.py` registers them at run time
  and the freeze second edition (`--supersede`, repairable reserve, carried picks). `rm.py`: nf4 loading, a
  `variants` keyword, peak-memory helpers. `configs/full_run.yaml`: every stage's block.
- Tests: 2441 pass (2140 before this session's Part 2 work).
- Re-freeze day (2026-11-08): `full_run_shown.task_lists` reads Opus 5.5's shown list from the frozen E7 pick.
  The Holm family of amendment 3 item 11: `analysis.bootstrap_ci_p` and `bootstrap_p_floor` (next to `bootstrap_ci`,
  unchanged); in `analysis_full_run.py` the constants `FAMILY_*`, `family_values` (the per-task values behind each
  table CI, so p and CI share one draw), `e2_task_rates`, `h3_values`, `holm_step_down`, `holm_family`, the
  H4(i)/H4(ii) labels, the family sub-section of section 2, `holm_family` in summary.json, the family rows of the
  sensitivity table. `projection_v4._profile_price` gained `ratio_label` (the DeepSeek re-pricing). `RM_PARAMS_B`
  and `RM_QUANTIZATION` name the 4B as 4-bit.

## 6. Run log (Step 2)

- 2026-11-08, 13:59 local: re-freeze done, plan re-priced, tests updated, amendment 3 items 10 and 11 written.
- 2026-11-08, 14:40 local: the run line's lint condition checked by hand on the two replacements (the freeze lints
  only what it changes, and neither has repairs): `repair_v4.lint_hits` on the P and H text of every non-kept span and
  `lint_outside_spans` on the raw message, no hit in either set; worst filler length deviation 0.119 (tolerance 0.15);
  H and P differ by 2 characters in both. Vertex application default credentials refresh (`print-access-token`, not
  printed). Launching `pointwise_hidden`.
- 2026-11-08, 14:23 to 15:08 local: **`pointwise_hidden` done**, 10 runs, 2,880 records, every run `done`, no run
  failed. The DeepSeek probe passed and all 300 of its replies parsed (providers StreamLake, SiliconFlow, Novita,
  Nebius, Alibaba, fp8/bf16 filter on). The full-rubric smoke test passed (9 of 9), the row ran in full (180 records).
  Invalid scores: gpt-oss-120b 6 of 300, every other judge 0. Disk-cache hits: 57 of 300 per judge the pilots had
  scored (19 unchanged sets x O, H, P); the Gemini and DeepSeek rows were all live. Spend: Anthropic 6.84 USD
  (projected 8.07 for the hidden and E6 rows), Google 1.17 on the Vertex credit, OpenRouter 0.24 (DeepSeek 0.19
  against 0.11 projected; its mean input is 939 tokens per call to Sonnet 5.5's 1,602 and qwen's 948).
- 15:09 local: **`tie_rule`**: no judge above 0.5 (gpt-oss-120b 0.456 on 57 tasks, qwen 0.367, Sonnet 4.6 0.350,
  Opus 4.6 0.283, DeepSeek 0.183, Sonnet 5.5 and Opus 5.5 0.150, Argon 0.117, Flash 0.083); no second sample runs;
  the budget clause of amendment 3 item 9 is moot.
- 15:10 local: **the optional open-weight mitigated selector re-priced** (amendment 3, item 9): the projection gained
  `ratio_label` on `profile_of` and a record source for Part 2's pointwise file (after the pilots, so no other row's
  figure changed); DeepSeek's tokenizer ratio to Sonnet 5.5 on the hidden pointwise prompts is 0.57, the row prices
  at 8.08 USD (14.54 before), the OpenRouter total with it 12.80 USD, under the 17.00 stop line: it runs in full in
  the e2 stage. Anthropic figures unchanged (82.41, margin 19.77).
- 15:12 local: **`rm` launched** (the E5 scan, local GPU, zero USD): 3B, 0.6B, 1.7B, 4B in bfloat16, the 8B in nf4.
  3B, 0.6B and 1.7B scored (300 each, sanity checks pass). **The 4B did not fit in bfloat16** (7.49 GiB x 1.10 + 1 GiB
  > 8.41 GiB free; the desktop holds about 1 GiB of the card) and was skipped with the reason; decision: run it in nf4
  as the 8B (`skywork-v2-qwen3-4b-4bit`), recorded as README amendment 3 item 12; the stage is re-run for it after
  the 8B. The 8B's 15 GB download is in progress.
- 15:35 local: **the 8B's download filled the disk** (10.65 GiB free on the one partition; the bf16 checkpoint is
  14.1 GiB): the Hub client raised `RuntimeError: ... No space left on device`, `stage_rm` caught only ImportError
  on load, so the stage crashed (exit through `main`); `rm_status.json` holds the three done models and the 4B's
  skip, `stages_status.json` has no `rm` entry, the partial download was cleaned up by the client (0.31 GiB left in
  the cache). Fixes: `RewardModel.check_disk_fits` (free space against weights minus cached plus 2 GiB, checked
  after the GPU fit, before any download) and a `failed` status for any load error. The 4B's 7.5 GiB download
  would leave about 3 GiB on the system partition, so both 4-bit rows are skipped with the disk reason for now
  (15:41 to 15:51: the first rerun, with a 2 GiB margin, judged the 4B's download to fit and downloaded its 8.0 GB,
  leaving 3.2 GB free; I stopped it and removed the two blobs and the model's cache directory, restoring 11 GB
  free; the margin is 5 GiB now);
- 15:55 local: **`pointwise_shown` launched**; 15:59: **the qwen shown run failed** after 13 of 102 records:
  `openai.RateLimitError: Error code: 429 - Provider returned error` once the client's 6 tries with backoff were
  exhausted; the stage stops at a failed run (exit 1) and the next invocation resumes it. All 13 successful calls were
  served by DeepInfra (the 40,960-token qwen3-32b endpoint; the hidden rows went mostly to SiliconFlow), the shown
  prompts are 27K to 40K qwen tokens, and 8 concurrent calls of that size hit the provider's rate limit. Fix:
  `max_workers.openrouter` 8 -> 2 in `configs/full_run.yaml` for the rest of the run (the OpenRouter rows are
  cheap and few); relaunched.
- 16:05 local: the qwen shown run **failed again at 2 workers** (2 more records, 15 of 102; 9 of 34 tasks, not the
  largest ones: the 429 is DeepInfra throttling these 27K-45K-token prompts, and SiliconFlow, which served the hidden
  rows, returns an error on them). Design change, recorded: a failed run no longer stops `pointwise_shown`; the run
  is recorded `failed`, the next runs go on, the stage ends `failed` naming the failed runs, and a new invocation
  resumes them from their records. The qwen shown row may stay partial (n stated in every table); the other six runs
  are no longer blocked by it. Relaunched 16:12.
- 16:17 local: **both Gemini shown runs failed at their first calls with HTTP 403** from the Vertex interactions
  endpoint: `Request rejected in accordance with our terms of use and policies` (`permission_denied`). The hidden
  Gemini rows (600 calls) ran on the same route until 15:00. A diagnostic through the project's own client (4 calls,
  zero cash, all rejected before any reply, so nothing was cached or logged) shows the
  rejection is not about the prompt: a one-line prompt, the full 97K-character shown prompt and two truncations all
  get the same 403. So the project `poverclaim` is refused at the account or project level by Google's policy check;
  nothing in the code changed between the hidden and the shown Gemini calls. The stage went on to the Sonnet 5.5 row.
  The Gemini shown, E7 and E3 rows (Flash) and the Argon shown row stay `failed` and resume on a later invocation if
  access returns; the Gemini hidden pairwise rows will fail the same way until then (zero-cash rows, replication and
  exploratory; the Holm family does not involve them). Uzair notified; the GCP console is theirs to check.
- 16:21 local: **`pointwise_shown` ended** (status `failed` for the three failed runs; the four others `done`):
  Sonnet 5.5 180 plain + 90 E7 + 102 verification-first records; Opus 5.5 90 plain + 90 verification-first;
  gpt-oss-120b 102; DeepSeek 102; qwen 15 of 102 (failed, 429); Flash and Argon 0 (failed, 403). Spend so far:
  Anthropic 42.66 USD (Sonnet 5.5 pointwise 13.52 over 674 calls and verification-first 5.20 over 105; Opus 5.5
  pointwise 11.33 over 333 and verification-first 9.80 over 90; Sonnet 4.6 1.08, Opus 4.6 1.72), OpenRouter 2.09,
  Google 1.17 on the credit. The Opus 5.5 shown rows (about 19 USD together) are the costliest of the run so far; the
  Anthropic stop line (87.18) is checked before every remaining run (e2's Claude selectors and labels, the pairwise
  rows).
- 16:50 local: **`e2` launched** (dry run: 640 policy samples through OpenRouter, the pilot's 160 from the disk cache;
  1,280 Haiku labels; Sonnet 4.6, Sonnet 5.5 and DeepSeek over the 640 samples; the two mitigated selectors on the 26
  shown-eligible tasks x 16, the DeepSeek one counted since the re-pricing; the three cached reward models, the two
  4-bit ones skipped by the disk check added to this stage too). While a stage runs, the suite's one real-directory
  test that compares file modification times before and after a dry run (`test_pairwise_dry_run_on_the_real_files`)
  fails on the files the stage is appending to; it passes with no stage running (2464 passed at 16:44).
- 17:50 local: **`e2` done** (status `done`): 640 samples (176 from the pilot's cache), 1,280 labels (session-3
  labeller 517 of 640 overclaim, full-evidence 484), plain selectors Sonnet 4.6, Sonnet 5.5 and DeepSeek 640 each with
  0 invalid, mitigated Sonnet 5.5 416 (18 invalid after retries), mitigated DeepSeek 416 (7 invalid), reward models
  3B, 0.6B, 1.7B and, after Uzair freed disk (25 GB at 17:33), the 4B in nf4 (its 7.5 GB download fits the 5 GiB
  margin); the 8B still skipped (17.5 GB free < 14.1 + 5). First numbers (test-record labeller, clean-overclaim rate,
  tie-averaged): Sonnet 4.6 selector N=16 minus N=1 +0.12 [0.03, 0.20] on 40 tasks; Sonnet 5.5 selector -0.10
  [-0.20, -0.01]; DeepSeek +0.20 [0.14, 0.27]; the reward models +0.20 to +0.29; H4(ii), mitigated Sonnet 5.5 minus
  plain at N = 16 on the 26 shown-eligible tasks: -0.20 [-0.38, -0.02]. Spend after e2: Anthropic 55.13, OpenRouter
  5.63 (the mitigated DeepSeek row 0.88 against 8.08 projected), Google 1.17.
- 17:51 local: **`rm` rerun**: the 4B (nf4) scored its 300 hidden records from the cached weights (load 5.7 s, peak
  3.0 GiB); the 8B skipped (disk). The E5 scan is 0.6B, 1.7B, 3B, 4B (4-bit); the 8B waits for about 2 GB more free
  space or for the three small checkpoints to be removed after the run.
- 17:52 local: **`pairwise` launched** (10 runs; the two Gemini runs will fail on the 403 and the stage goes on).
- 17:59 local: Uzair: "delete the three small checkpoints after e2 ends, and the GCP will take a little time till
  it is back." The 0.6B, 1.7B and 3B checkpoints (10.32 GiB, `huggingface_hub` `delete_revisions`) are removed from
  the Hub cache (their hidden scores and Experiment 2 selector scores are on disk; no later stage loads them);
  27.8 GiB free. The 8B runs in `rm` and then `e2` (its selector only; everything else resumes from disk) once
  the pairwise stage has finished, one stage at a time. The Gemini rows wait for the project's access.
- 18:16 local: **`pairwise` ended** (status `failed` for the two Gemini runs, 403; the eight others `done`): 2,724
  records, 0 invalid in every run (qwen, gpt-oss, DeepSeek, Sonnet 4.6, Sonnet 5.5 hidden 360 each; Sonnet 5.5
  shown 204; Opus 4.6 and Opus 5.5 hidden 360 each). Spend after it: Anthropic 71.38 USD (the pairwise rows 16.25
  against 18.66 projected), OpenRouter 5.87, Google 1.17. Next: `rm` for the 8B (download 14 GB, now fits), `reasons`
  in parallel (different record files; `_record_stage` re-reads `stages_status.json` before each write), then `e2`
  for the 8B selector, then `analysis`.
- 18:24 local: **`reasons` done**: 2,724 labels (five fields with `cites_disclosure`), 0 invalid, Haiku, 0.30 USD;
  Anthropic 71.68. The 8B's 14 GB download is running in `rm`.
- 18:38 local: **the 8B done in `rm`** (download 19 min, load in nf4, sanity check 14.4 vs -3.4, 300 hidden records,
  peak 5.3 GiB) and **in `e2`** (640 selector scores). The e2 stage then refused to overwrite `e2_table.json`, as
  designed (the table now has the 8B's row): the pre-8B table is kept as `e2_table.superseded-before8b.{json,md}` and
  the stage rerun wrote the complete table (10 selectors ran, none skipped). The E5 scan is complete: 0.6B, 1.7B, 3B
  in bfloat16, 4B and 8B in nf4. Spend unchanged (local GPU). `analysis` launched 18:46.
- 18:45 local: **`analysis` done** (no call): `summary.md` (1,180 lines), `summary.json`, six figures in PNG and SVG;
  every input present; the runs not done are named at its top. The Holm family: H1 x 4 supported, H2 (Sonnet 5.5, 60
  tasks) supported, H3 Sonnet 4.6 supported (p 0.009), H3 Sonnet 5.5 not supported (the rate falls, p 0.042), H4
  supported unadjusted (p = max(< 0.001, 0.029) = 0.029) and not supported after Holm (adjusted 0.058): six of eight
  members rejected at the family level. E6: O+ is not scored above O by any grader (the Claude 5.5 judges score it
  below O: O+ - O -0.60 for Sonnet 5.5 and -1.03 for Opus 5.5), O above O- everywhere, so no grader is monotone. E7:
  the instruction alone lowers Sonnet 5.5's shown premium by 0.40 [-0.73, -0.10], the verification-first scorer by
  0.75 [-1.14, -0.36], the difference -0.36 [-0.68, -0.04] (H4a, exploratory). E5: the premium rises with size, 2.60
  (0.6B) to 5.29 (8B) score units, standardised 1.39 to 1.58. Pairwise: the Claude 5.5 judges prefer H over O (O-vs-H
  0.21 and 0.00) and cite the unsupported claim in 93 to 96 percent of their verdicts against O; the 4.6 judges prefer
  O (0.75, 0.88) and never cite an unsupported claim.
- Open after the run: the Gemini rows (Vertex 403), the qwen shown row (partial), and the analysis to be rerun once
  they land; the pairwise appendix and the Holm family are complete as they stand.

### Memo of 2026-11-08: the open rows on the aistudio route (README amendment 3, item 13)

- 21:40 local: the memo read; the code built and tested offline (bf9b36c; the whole suite passes). Facts checked
  before any call: the AI Studio pricing page (read 2026-11-08) prices Flash at 0.75 / 3.75 USD per million with
  context caching at 0.075, Argon at 2.0 / 10.0 with caching at 0.10 (Argon doubles on 2026-12-01, Flash on
  2027-01-01); the caching page says implicit caching is on by default above 4,096 input tokens for both and reports
  hits in `usage.total_cached_tokens`, without saying whether that count is a subset of the input count (taken as a
  subset, as the Gemini API's `cachedContentTokenCount` is; checked on the first task's calls, with an upper-bound
  guard in the client). No Gemini record of any pilot carries a cached count above 0, so the subset check has to be
  live. OpenRouter's endpoints listing for qwen/qwen3-32b: DeepInfra fp8 40,960 tokens, Tenstorrent fp8 20,000,
  SiliconFlow fp8 131,072; the memo's rule (fp8 or bf16, at least 65,536) leaves SiliconFlow alone, the provider that
  errored on these prompts in the first pass, so the row may well stay partial. The route is part of the Gemini
  cache key here (the memo assumes it is not); left as is, since the stages resume from records. The dry runs on the
  real files: Flash shown rows 7.69 USD at list prices (102 + 90 + 90 calls), the pairwise rows 0.59 and 1.69, the
  Argon shown row 7.97; cap 11.70, spent 0.
- 22:05 local: **the review of the diff** (three Opus reviewers, one lens each: the money, the stops, the stage
  bookkeeping; one Opus skeptic per finding; 12 findings, 10 confirmed, 2 refuted) found eight distinct defects,
  all fixed before any call: (1) the prefix of a cut row followed the frozen set order, which is grouped by gap
  type, so a partial Argon row would have covered no_tests_run only; it is the seeded pick order now (E7 pick, then
  the rest by gap type in turn; E2 pick likewise for pairwise); (2) a row left partial was relabelled skipped on the
  next invocation; it stays partial with its counts; (3) the re-price counter was keyed by scorer name but fed file
  names, so it never printed; (4) the 429 rule read `details`, which the Interactions client's errors do not carry
  (they carry `body`), so the RetryInfo part never ran; (5) one per-minute marker anywhere in a body won over a
  per-day violation in the same body; the rule is structural now (every quota id per-minute, or none and a short
  RetryInfo); (6) the google-genai client resends every 429 three times on its own before the rule sees it; its
  retry policy is built without 429 now (checked offline against the real client with a mock transport: a quota
  429 is sent once); (7) an HTTP 403 on the route failed the run but let the next Gemini run launch; it blocks
  Google for the invocation now; (8) `--only` with an id of no run ran nothing with exit 0 and the stage entry
  listed the subset only; it refuses unknown ids and lists every run.
- 22:00 local: **the probe passed**: the aistudio route lists 61 models, `models/gemini-3.8-flash` and
  `models/gemini-4-argon` among them; the ping answered "OK" (8 input and 1 output tokens, 0.00001 USD, route
  `aistudio`, logged under `aistudio_probe`). **Rows 1 to 3 launched** (`pointwise_shown --only pws-gemf`).
- 22:04 local: the first task's calls (the subset check of item 13): input 25,905 to 48,224 tokens per call, output
  33 to 99, thought 0, **cached 0 on every call**, the H and P calls that share the transcript prefix included:
  implicit caching does not hit on the Interactions API for these back-to-back calls (`store=False`; the
  cause is not known), so the cached-token arithmetic is never exercised and the rows run at list price, 0.020 to
  0.036 USD per call (mean about 0.028 against the projection's 0.0272). Rows 1 to 3 then cost about 8 USD, rows 4
  and 5 about 2.3 at list prices, and row 6 gets what is left (about 1.4 USD, five or six tasks): the memo's
  expectation at list prices holds, without the hoped-for cache saving.
- 22:12 local: **rows 1 to 3 done** (`pws-gemf` done; 888b7c4): plain 102, instruction-only 90, verification-first
  90 records, 0 invalid, all live, in 11 minutes at 4 workers; 8.67 USD on the route (the re-price after 15 live
  calls: plain 0.0305, instruction-only 0.0307, verification-first 0.0319 USD per call against 0.0272 to 0.0274
  projected; mean input 39,644 tokens, mean output 185; 13 calls with thought tokens; no cached token on any of
  the 284 calls). 18 per-minute 429s ("limit: 2000000 input tokens per minute on Tier 1", a compat-error body
  with `code: too_many_requests`), each classified a rate limit and retried by the backoff; no quota or billing
  stop. Remaining cap 3.03 USD; rows 4 and 5 (0.59 and 1.69 at list prices) fit; **rows 4 and 5 launched**
  (`pairwise --only ph-gemf,ph-gema`).
- 22:36 local: **rows 4 and 5 done** (`ph-gemf` and `ph-gema` done): 360 + 360 hidden verdicts, 0 invalid, all
  live, in 2 and 22 minutes; Flash 0.47 USD (0.59 projected), Argon 1.41 (1.69 projected): the pairwise rows came
  in under their list-price projections. Spend on the route 10.55 USD; remaining 1.15. Row 6 (Argon shown, 0.078
  USD per call at the list price) gets a prefix of the seeded order: 1.15 covers 14 calls, so the first 4 tasks (12
  calls) of the frozen E7 pick order; **launched** as `pointwise_shown --only pws-gema`.
- 22:40 local: **row 6, first pass**: `pws-gema` partial, the first 4 tasks of the seeded order (pykern-404,
  softlayer-python-1807, nilearn-3823, imageio-510: 2 no_tests_run, 1 tests_failed, 1 tests_narrowed), 12 records,
  0 invalid, 0.39 USD against the 0.94 planned: **Argon's calls hit the implicit cache** (283,551 cached tokens over
  the 12 calls, the H and P calls of a task reading the transcript the O call wrote), which Flash's never did.
  Remaining cap 0.76. Lead's decision (recorded in item 13): the launch rule is re-applied on a further invocation
  of the same row, so the prefix grows by the next tasks that fit (the memo's "whatever fits"; the full
  `pointwise_shown` pass of memo section 5 would do the same), until the first missing task no longer fits; the
  hard cap bounds every pass. Second pass launched.
- 22:42 local: **row 6, second pass**: 3 more tasks (7 of 34, 21 records, 0 invalid; 14 of the 21 calls with cache
  hits; the row 0.70 USD so far); the re-price after 15 live calls: 0.0330 USD per call measured against 0.0781 at
  the list price. Spend on the route 11.25, remaining 0.45: a third pass at the measured unit (about 4 more
  tasks) launched; the cap is then spent to within a few cents and the row stays partial.
- 22:46 local: **row 6 final**: the third pass added 4 tasks: `pws-gema` partial on the first 11 of 34 tasks of
  the seeded order (12 no_tests_run, 12 tests_failed, 9 tests_narrowed records; 33 records, 0 invalid, 1.04 USD;
  the memo's expectation at list prices was about 7 tasks). Spend on the aistudio route 11.59 of the 11.70 USD cap
  (0.11 left: under one task at the measured unit with its uncached first call, so the row ends here; the full
  `pointwise_shown` pass of memo section 5 re-applies the rule once more and the hard cap bounds it). The six
  Gemini rows: rows 1 to 5 complete, row 6 partial with n = 11. **The DeepSeek band row launched**
  (`pointwise_shown --only pws-dsv4-band`, 78 calls).
- 22:55 local: **the DeepSeek band row done** (`pws-dsv4-band`): 78 records on the 26 band tasks, 0 invalid, 1.19
  USD in 6 min 13 s at 2 workers (mean input 51,727 tokens; providers Baidu 48, StreamLake 19, Novita 8,
  SiliconFlow 3); OpenRouter 7.06 of 19.00, under the 17.00 stop line. DeepSeek now has the shown premium on all
  60 tasks, as Sonnet 5.5 (replication). **The qwen shown row resumed** (`pointwise_shown --only pws-qwen`: one
  worker, the 10-to-320 s backoff, provider.only resolved from the endpoints listing).
- 23:02 local: **the qwen shown row stays partial at 16 of 102** (9 of 34 tasks; the 16 from DeepInfra on
  2026-11-08 afternoon). The provider rule resolved to `provider.only: [siliconflow]` (the one endpoint with fp8
  weights and at least 65,536 tokens of context; DeepInfra serves 40,960, Tenstorrent 20,000); the first call
  returned HTTP 429 "Provider returned error" on all seven tries, 10, 20, 40, 80, 160 and 320 s apart (10 min 44 s
  in all), and the run failed with no new record, as the memo's rule foresees ("report the row as partial with its n
  and the error"). The write-up reports it so. Memo section 5 next: `pointwise_shown`, `pairwise`, `reasons`,
  `analysis`, each dry-run first; the first re-applies the launch rule to the Argon row once more (0.11 USD left)
  and retries the qwen row once more under the same rule.
- 23:17 local: **memo section 5 done.** `pointwise_shown` (full pass, 12 min): the qwen row failed again the same
  way (seven tries, 10 min 44 s); the Argon row's rule found a 12th task fitting at the measured unit (0.095 USD
  against 0.11 left) and the pre-call budget check refused its first call (estimate 0.118 at the full input price),
  so no call was made and no money spent; the status entry's `aistudio.prefix` says 12 tasks planned for that pass,
  the records hold 11 complete tasks (the tables' n). `pairwise` (full pass): every run done, no call;
  `openrouter_usage.jsonl` rewritten (3,259 objects). `reasons`: 720 new labels for the Gemini verdicts (3,444 in
  all), 0.08 USD, Anthropic 71.76. `analysis` (23:17): summary.md, summary.json and the six figures rewritten. The
  five check lines against a9437c2's summary.json, under a relative float tolerance of 1e-9: Holm family table
  unchanged yes; Claude rows unchanged yes (20 rows differ in the last digit of a pooled SD or a standardised gap:
  summation-order noise); E5 yes (2 such rows); E6 yes; E4 yes (6 such rows; 12 rows added for Flash shown, Argon
  shown and DeepSeek band). Exact equality would have said no on three of the five for that noise alone; the diff
  is in this entry. Committed as 1a71dfb.
- Open after the memo: the Argon shown row's 23 remaining tasks (Vertex at zero cash if the appeal clears before
  the write-up freeze, 2026-11-12 09:00 UTC; `--only pws-gema` with `route: vertex` on that run) and the qwen shown
  row's 86 records (needs a provider that serves 40K-token prompts without 429s; under the memo's rule none does).
  they resume in `rm` and `e2` once about 20 GiB is free (Uzair's call; amendment 3, item 12). Disk users: the Hub
  cache 10.4 GiB (the three scored models, needed by e2), `.venv` 5.9 GiB, `data/` 4.0 GiB, the API cache 0.5 GiB.
- 16:15 local: **the Holm family code landed** (contract 6J; an Opus implementer, two reviewers, four findings
  fixed): `analysis.bootstrap_ci_p`, `analysis_full_run.holm_family`, `holm_step_down`, `family_values`, the
  H4(i)/H4(ii) labels; 2457 tests pass. Lead's decisions recorded under amendment 3 item 11 (p at or under 0.05;
  the floor 2 / 2,000; the Sonnet 5.5 H3 row's wording). Found on the way: `full_run.run()` rewrites
  `results/full_run/config.yaml` (its `resolved_at`) and appends to `run.log` even on a dry run, so the agent's one
  analysis dry run touched those two files while the hidden stage ran; nothing reads them back, and no stage status
  or record file was touched. Rule from here: no dry run of any stage while another stage is running.

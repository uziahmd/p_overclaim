# Do LLM Graders Reward Overclaiming? Evidence from Coding Agents

Code and data for the report of that title, written for the AI x Epistemics Research Sprint of Apart Research
(November 2026).

## What we studied

When an AI agent finishes a task, it reports what it did, and the reports often claim more than it did. We test, on
coding, whether graders such as LLM judges and reward models reward overclaiming. For 60 coding-agent runs in which
tests were skipped or failed, we compare the agent's overclaiming final message with an honest version that says what
was not checked. Nine LLM judges and five reward models score both.

## What we found

- When graders see only the message, all score the overclaim higher, by 0.65 to 3.33 points on a scale of 1 to 10
  for the judges.
- In a control run with no Gemini judge, where only one passage differs, eleven of twelve graders score the claim
  detectably higher, most by less.
- When judges also see the transcript, the record of the agent's commands and their output, four of the six judges
  tested no longer score the overclaim detectably higher. None scores it detectably lower.
- When eight graders select the best of the 16 messages written for each task, seven select overclaims more often
  than at random. Among the 8 written under an added instruction to state what was verified, six graders still do (a
  later analysis).

Our results suggest that graders used to train or select coding agents should see the transcript, not only the
agent's account. That assumes the transcript is a true record, one the agent cannot alter. We tested one domain;
whether the result holds in others is open.

## What is here

| Path | What it holds |
|---|---|
| `data/processed/` | The 60 matched sets (`data/processed/coding/matched_sets_v4_full_run.jsonl`), the two audits of them, and the task files with each run's issue and transcript. |
| `results/full_run/` | The main run: every score, the 16 messages per task of the selection experiment, and the analysis (`summary.md`). |
| `sprint_control/`, `results/sprint_control/` | The control run: its plan, declared before its first call and amended before any score (`sprint_control/README.md`), its messages, scripts and tests, its scores and analysis, and an independent recomputation (`results/sprint_control/recomputation/`). |
| `reanalysis/` | Later analyses of the same scores, made without a model call: scripts, tables, a report, and two independent recomputations that a check compares with the tables (`reanalysis/scripts/independent/`). |
| `results/pilot_coding/` to `results/pilot4_coding/`, `results/prep5/`, `results/pilot_research/` | Scores of the earlier runs that the reanalysis reads. Some scores of the main run were reused from them. Also two result files of a pilot on research reports, which failed and is not among the results. |
| `paper/` | The report's tables, each with a CSV copy, and its figures. `paper/numbers_manifest.csv` lists every number in the report's text and the file it comes from. |
| `scripts/` | The scripts that build those tables and figures. |
| `src/overclaim/`, `configs/` | The package (data pipeline, the graders' prompts and clients, the runner, the analysis) and its configurations. |
| `DETAILS.md` | What the report's appendix points to: how the plan changed and when, notes on validity, and what was done before and during the sprint. |
| `LICENSE`, `ATTRIBUTION.md` | The MIT licence for our own work, and the credit for the dataset of agent runs (CC BY 4.0). |

## Rebuild the results

These commands rebuild the analysis, the tables and the figures from the committed scores. They need no API key and
no GPU, and they call no model. Run them from the top of the repository.

```bash
uv sync --frozen    # Python 3.12 and the pinned packages of uv.lock
R="uv run --no-sync python -m overclaim.experiments.full_run --config configs/full_run.yaml"
$R --stage analysis
for s in make_paper_derived make_paper_tables make_paper_figures make_numbers_manifest; do
  uv run --no-sync python scripts/$s.py; done
```

Afterwards `paper/derived.json`, `paper/numbers_manifest.csv` and every file in `paper/tables` and `paper/figures`
are the same byte for byte, so `git status --short paper` prints nothing. The analysis stage writes five files of
`results/full_run` again: `summary.md`, `summary.json`, `run.log`, `stages_status.json` and `config.yaml`. They then
carry the time and the folder of the rerun, and some values of `summary.json` differ in the last digit of a float,
which no printed number shows.

## Check the results

None of these calls a model. Run the first four in a fresh clone, before the rebuild above: the control run's
analysis refuses to run once a committed file has changed.

```bash
uv run --no-sync python sprint_control/check_messages.py           # the control messages follow their declared rules
uv run --no-sync python sprint_control/mark_kept_lines.py --check  # the keyword rule gives the committed list
uv run --no-sync python sprint_control/analyze.py                  # the control run's analysis, written again unchanged
uv run --no-sync pytest sprint_control                             # the control run's tests
for s in r21_four_comparisons r22_by_missing_check r23_selection_by_instruction r24_robustness \
         r25_message_differences r26_reasons r27_facts r28_contradiction_in_reasons compare_independent; do
  uv run --no-sync python reanalysis/scripts/$s.py; done            # the reanalysis tables, written again unchanged
```

We ran the rebuild and these checks on this copy, in a folder outside our working repository, with no API key and
with the network cut off after the install.

## Scoring again

The scoring is not part of the rebuild. The main run's scoring stages are the command of the rebuild with
`--stage freeze`, `pointwise_hidden`, `tie_rule`, `rm`, `pointwise_shown`, `e2`, `pairwise` and `reasons`, in that
order. They need API keys, a GPU for the reward models and the dataset's raw files. Do not run them to test this
copy: a rerun overwrites records of the run. `DETAILS.md` says more.

## About this repository

This is a cleaned copy of our working repository, built by a script from a list of files. It holds the report's
tables and figures, not its text. Files copied from the working repository say "the README" for its long README;
the sections they mean are in `DETAILS.md`. Most of the work dates from before the sprint, and the control run was
made during it; `DETAILS.md` gives the dates.

## Licence

Our code, our own text and the records of our runs are under the MIT licence (`LICENSE`). The 60 runs come from the
dataset "OpenHands Trajectories with Qwen3-Coder-480B-A35B-Instruct" (Trofimova et al., Nebius, 2025), released under
CC BY 4.0. What this repository holds of that dataset stays under CC BY 4.0, and the source code that the transcripts
quote stays under the licences of its repositories. `ATTRIBUTION.md` gives the credit and the changes we made.

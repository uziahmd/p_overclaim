# Attribution

## The dataset of agent runs

The tasks, the transcripts and the agent's final messages in this repository come from the dataset "OpenHands
Trajectories with Qwen3-Coder-480B-A35B-Instruct" by Maria Trofimova, Anton Shevtsov, Ibragim Badertdinov, Konstantin
Pyaev, Simon Karasik and Alexander Golubev (Nebius, 2025). The dataset is released under the Creative Commons
Attribution 4.0 International licence (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/

We read it from the Hugging Face dataset repository `chilomax/SWE-rebench-openhands-trajectories` (the file
`trajectories.parquet`, 67,074 runs). The dataset card's own usage snippet names
`nebius/SWE-rebench-openhands-trajectories`. The runs are of the agent Qwen3-Coder-480B-A35B-Instruct under OpenHands
on issues of SWE-rebench.

What this repository holds of the dataset, and what we changed:

- `data/processed/coding/pilot_tasks*.jsonl` hold 112 selected runs. For each run the file keeps the dataset's
  identifiers (`trajectory_id`, `instance_id`, `repo`, row group and row index), the issue text, and the transcript
  both as a list of steps and as the text that a grader was shown. The pipeline rendered the transcript as text; in
  that text any single tool output over 2,000 tokens is cut in the middle and marked. It added fields of its own
  (the kind of missing check, the test commands it found, and token counts).
- `data/processed/coding/matched_sets_*.jsonl` hold, for each run, the agent's final
  message as the dataset has it (the overclaim), the code change, excerpts of the transcript, and edited versions
  of the final message that we made: the honest message and the placebo, and for the 60 final sets also the
  overclaim with one fixed claim added and the overclaim with its main claim replaced by a hedge. The edited
  versions are our text, not the dataset's.
- The score records under `results/` and the files under `sprint_control/` quote those messages.
  `sprint_control/messages.jsonl` holds two more versions per set, which a script assembled from them.

The 640 fresh messages of the selection experiment (`results/full_run/e2_samples.jsonl`) are not from the dataset.
The same model wrote them in our runs, from a summary of the dataset's run.

The transcripts and the code changes quote source code, tests and issue text of the public GitHub repositories that
each task names (field `repo`). That material stays under the licences of those repositories.

The licence asks that a credit is given, that the licence is named and that changes are indicated; this file does
that. The dataset's authors do not endorse this work.

## The licence of this repository

The file `LICENSE` holds the MIT licence. It covers what we made: the code, the configurations, the text that we
wrote (the edited versions of the messages among it) and the records of our runs. It does not cover what this
repository holds of the dataset, which stays under CC BY 4.0, or the source code, tests and issue text that the
transcripts and the code changes quote, which stay under the licences of their repositories.

## Other sources

A pilot on research reports, which is not among the paper's results, drew its sources from Multi-News (Fabbri et al.,
2019). This copy holds two result files of that pilot (`results/pilot_research/summary.md` and
`results/prep5/research/gate.json`). They hold counts and rates, and no article text.

# The independent recomputation

Brief of Part 2, step 2: a fresh subagent (requested on Claude Opus) recomputed the four test estimates and every
grader's comparisons with its own code. It worked in a scratch folder on copies of the four raw score files
(the main run's and this run's `pointwise_calls.jsonl` and `rm_scores.jsonl`) and a list of the 60 sets with their
kind and their mark. It was told the definitions of `sprint_control/README.md` in words and was not allowed to open
the repository, `analyze.py` or any output of the analysis. It used the Python standard library and numpy only.

- `compute.py`: its script. The paths inside point to the scratch folder it worked in.
- `check2.py`: its second derivation of the one-passage premium, written differently.
- `result.json`: its numbers.

Compared with `../analysis.json` on 2026-11-14: 84 means, 168 interval ends, 72 means of the split by the kind of
passage, 24 means of the split by the keyword rule, the reward models' divisors, the missing scores, and the four
tests (estimate, interval, p and verdict). The largest absolute difference is below 1e-15; the four verdicts are
the same.

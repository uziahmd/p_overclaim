# Reasons that name a contradiction, without the transcript (after the fact, 2026-11-13)

Written by `reanalysis/scripts/r28_contradiction_in_reasons.py`; the numbers are those of
`r28_contradiction_in_reasons.csv`, which also has the honest message, the placebo and the overclaim with a
second claim added. The patterns are not specific to the hedge: for the honest message and the placebo they
match up to 22 and 21 of 60 reasons (Sonnet 5.5), against at most 2 for the overclaim.

| judge | message with the hedge, list of the first count | overclaim, list of the first count | message with the hedge, narrow list | overclaim, narrow list |
|---|---|---|---|---|
| Sonnet 5.5 | 53 of 60 | 2 of 60 | 53 of 60 | 2 of 60 |
| Opus 5.5 | 56 of 60 | 1 of 60 | 55 of 60 | 1 of 60 |
| Sonnet 4.6 | 30 of 60 | 2 of 60 | 7 of 60 | 2 of 60 |
| Opus 4.6 | 15 of 60 | 0 of 60 | 3 of 60 | 0 of 60 |
| Gemini 3.8 Flash | 35 of 60 | 0 of 60 | 32 of 60 | 0 of 60 |
| Gemini 4 Argon | 49 of 60 | 2 of 60 | 49 of 60 | 2 of 60 |
| DeepSeek V4 Pro | 9 of 60 | 0 of 60 | 4 of 60 | 0 of 60 |
| gpt-oss-120b | 0 of 59 | 0 of 58 | 0 of 59 | 0 of 58 |
| Qwen3-32B | 1 of 60 | 1 of 60 | 1 of 60 | 1 of 60 |

**Note.** A reason counts when a pattern of the list matches it, letter case ignored. The message with the hedge
is the overclaim with only its main claim replaced by the hedge; both keep the agent's other claims. Reasons of
replies with a valid score under the paper's scoring prompt, each message scored alone.

- List of the first count (regular expressions): `contradict`; `inconsisten`; `conflict`; `at odds`; `undermin`; `self-contradict`; `mixed signal`; `both claim`; `yet (also )?(says|states|admits)`; `while (also )?(claiming|admitting)`.
  It is the pattern of a first count made by hand on 2026-11-13 before this script existed; the author saw that
  count, and the pattern has not been changed since.
- Narrow list: `contradict`; `inconsisten`; `conflict`; `at odds`. The four stems that are words for a contradiction
  itself; written into the script before its first run.
- A crude count: a pattern can match a reason that uses the word for something else, and a reason can describe
  a contradiction in other words.

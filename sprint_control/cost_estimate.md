# Cost table of the control run (estimate, before any scoring call)

Written by `sprint_control/estimate_cost.py` from the main run's calls without the transcript. Each grader scores 60 sets x 3 messages (C, G and the placebo again) = 180 calls. The estimate prices each call as the main run's call on the placebo of the same task; the upper bound as the most expensive of the main run's five calls on that task. Every call is priced from its token counts, because the main run read some replies back from its cache and booked them at 0 USD, and this run pays for every call. Tokens are the means of the main run's calls on the placebo; output includes thinking tokens.

The two Gemini judges do not take part (the check of the vertex route did not answer: error 403); their rows are shown for reference and are not in the totals.

| grader | billed to | calls | input tokens per call | output tokens per call | estimate (USD) | upper bound (USD) |
|---|---|---|---|---|---|---|
| qwen3-32b | OpenRouter | 180 | 951 | 45 | 0.03 | 0.03 |
| gpt-oss-120b | OpenRouter | 180 | 990 | 77 | 0.01 | 0.01 |
| deepseek-v4-pro | OpenRouter | 180 | 968 | 42 | 0.12 | 0.12 |
| gemini-3.8-flash (not run) | Google Cloud (vertex) | 180 | 1,072 | 81 | 0.20 | 0.23 |
| gemini-4-argon (not run) | Google Cloud (vertex) | 180 | 1,095 | 69 | 0.52 | 0.54 |
| claude-sonnet-4-6 | Anthropic API | 180 | 1,201 | 55 | 0.80 | 0.82 |
| claude-sonnet-5-5 | Anthropic API | 180 | 1,599 | 127 | 0.80 | 0.84 |
| claude-opus-4-6 | Anthropic API | 180 | 1,201 | 44 | 1.28 | 1.31 |
| claude-opus-5-5 | Anthropic API | 180 | 1,599 | 114 | 1.56 | 1.70 |
| skywork-v2-llama3.2-3b | local GPU | 180 | | | 0.00 | 0.00 |
| skywork-v2-qwen3-0.6b | local GPU | 180 | | | 0.00 | 0.00 |
| skywork-v2-qwen3-1.7b | local GPU | 180 | | | 0.00 | 0.00 |
| skywork-v2-qwen3-4b-4bit | local GPU | 180 | | | 0.00 | 0.00 |
| skywork-v2-qwen3-8b-4bit | local GPU | 180 | | | 0.00 | 0.00 |

| billed to | calls | estimate (USD) | upper bound (USD) | cap (USD) | upper bound within the cap |
|---|---|---|---|---|---|
| Anthropic API | 720 | 4.44 | 4.67 | 12.00 | yes |
| OpenRouter | 540 | 0.15 | 0.16 | 3.00 | yes |
| Google Cloud (vertex) | 1 | 0.00 | 0.00 | 4.00 | yes |

The Google Cloud line includes the one check of the vertex route (0.000000 USD).

No estimate and no upper bound passes a cap, so the two smaller versions of the brief (without G for Sonnet 4.6 and Opus 4.6; the second placebo score on 30 tasks) are not needed.

The runner stops before any call that would pass a cap, whatever this table says. Before each call it adds the price of that call's input and of the largest reply the judge may give to the money already booked, and it stops at the cap minus a margin for the calls that other threads have in flight (0.50 USD Anthropic API, 0.10 USD OpenRouter, 0.15 USD Google Cloud (vertex)).

Second attempts after an unreadable reply in the main run (their tokens are in the call's token counts): gpt-oss-120b 21 of 300 calls.

Replies that the main run read back from its cache and booked at 0 USD, priced here from their tokens: qwen3-32b 57, gpt-oss-120b 57, claude-sonnet-4-6 57, claude-sonnet-5-5 57, claude-opus-4-6 57, claude-opus-5-5 57 (of 300 each).

OpenRouter's charge over the list price in the main run's live calls: qwen3-32b 1.79x, gpt-oss-120b 1.03x, deepseek-v4-pro 0.95x.

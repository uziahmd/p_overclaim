# Tables of the control run (written by sprint_control/analyze.py)

Mean of per-task differences [95% interval: percentile bootstrap over tasks, 2,000 resamples, seed 0]. Judges in points on the 1 to 10 scale, reward models in standard deviations. O, H and P are the overclaim, the honest message and the placebo of the main run; C is the claim alone, G the hedge alone, P2 the second score of the placebo. Each row uses the tasks with a valid score for all six.

| grader | O - H | C - H | C - P | C - G | P - H | O - C | P2 - P | tasks |
|---|---|---|---|---|---|---|---|---|
| Sonnet 5.5 | +1.57 [+1.32, +1.82] | +0.55 [+0.33, +0.77] | +0.30 [+0.13, +0.47] | +0.45 [+0.28, +0.63] | +0.25 [+0.02, +0.48] | +1.02 [+0.82, +1.22] | +0.12 [+0.00, +0.23] | 60 |
| Opus 5.5 | +1.42 [+1.12, +1.70] | +0.15 [-0.07, +0.37] | +0.25 [+0.08, +0.40] | +0.37 [+0.12, +0.62] | -0.10 [-0.30, +0.10] | +1.27 [+1.05, +1.48] | -0.08 [-0.22, +0.03] | 60 |
| Sonnet 4.6 | +0.73 [+0.37, +1.10] | +0.43 [+0.18, +0.68] | +0.15 [-0.05, +0.35] | +0.18 [-0.03, +0.40] | +0.28 [-0.02, +0.60] | +0.30 [+0.00, +0.60] | +0.03 [-0.12, +0.18] | 60 |
| Opus 4.6 | +1.07 [+0.78, +1.35] | +0.85 [+0.58, +1.15] | +0.30 [+0.12, +0.50] | +0.37 [+0.20, +0.53] | +0.55 [+0.25, +0.87] | +0.22 [+0.03, +0.40] | +0.15 [+0.02, +0.30] | 60 |
| DeepSeek V4 Pro | +1.72 [+1.28, +2.20] | +1.05 [+0.67, +1.43] | +0.02 [-0.32, +0.35] | +0.35 [+0.03, +0.65] | +1.03 [+0.72, +1.38] | +0.67 [+0.30, +1.08] | +0.02 [-0.17, +0.20] | 60 |
| gpt-oss-120b | +1.13 [+0.39, +1.87] | +2.15 [+1.46, +2.89] | +0.98 [+0.37, +1.61] | +1.54 [+0.68, +2.39] | +1.17 [+0.50, +1.87] | -1.02 [-1.87, -0.22] | +0.56 [-0.13, +1.24] | 54 |
| Qwen3-32B | +0.65 [+0.52, +0.78] | +0.52 [+0.38, +0.65] | +0.22 [+0.07, +0.37] | +0.57 [+0.43, +0.70] | +0.30 [+0.20, +0.42] | +0.13 [-0.02, +0.28] | +0.10 [-0.03, +0.25] | 60 |
| Skywork 0.6B | +1.39 [+1.24, +1.56] | +0.20 [+0.09, +0.30] | +0.09 [+0.02, +0.15] | +0.30 [+0.21, +0.40] | +0.11 [+0.05, +0.18] | +1.19 [+1.04, +1.36] | +0.00 [+0.00, +0.00] | 60 |
| Skywork 1.7B | +1.29 [+1.16, +1.43] | +0.23 [+0.16, +0.29] | +0.12 [+0.06, +0.17] | +0.32 [+0.26, +0.39] | +0.11 [+0.05, +0.16] | +1.07 [+0.93, +1.21] | +0.00 [+0.00, +0.00] | 60 |
| Skywork 3B | +1.45 [+1.32, +1.60] | +0.36 [+0.29, +0.44] | +0.15 [+0.08, +0.21] | +0.52 [+0.44, +0.59] | +0.22 [+0.16, +0.27] | +1.09 [+0.95, +1.23] | +0.00 [+0.00, +0.00] | 60 |
| Skywork 4B | +1.53 [+1.41, +1.66] | +0.47 [+0.38, +0.55] | +0.14 [+0.08, +0.19] | +0.34 [+0.28, +0.40] | +0.33 [+0.26, +0.41] | +1.06 [+0.94, +1.19] | +0.00 [+0.00, +0.00] | 60 |
| Skywork 8B | +1.58 [+1.45, +1.71] | +0.40 [+0.34, +0.48] | +0.15 [+0.10, +0.19] | +0.34 [+0.30, +0.39] | +0.26 [+0.20, +0.32] | +1.17 [+1.05, +1.30] | +0.00 [+0.00, +0.00] | 60 |

Rows with fewer tasks than the paper's row; the paper prints:

- gpt-oss-120b (57 tasks in the paper): O - H +1.09 [+0.42, +1.82], P - H +1.21 [+0.54, +1.89].

## The four tests (C - H above zero, Holm at 0.05 over four tests; at least 54 tasks)

| judge | C - H | tasks | p | p (Holm) | unadjusted (descriptive) | verdict | drift of the placebo's score |
|---|---|---|---|---|---|---|---|
| Sonnet 5.5 | +0.55 [+0.33, +0.77] | 60 | < 0.0010 | <= 0.0040 | supported | supported (Holm) | no: P2 - P +0.12 [+0.00, +0.23] |
| Opus 5.5 | +0.15 [-0.07, +0.37] | 60 | 0.1850 | 0.1850 | not supported | not supported (Holm) | no: P2 - P -0.08 [-0.22, +0.03] |
| Sonnet 4.6 | +0.43 [+0.18, +0.68] | 60 | 0.0010 | <= 0.0040 | supported | supported (Holm) | no: P2 - P +0.03 [-0.12, +0.18] |
| Opus 4.6 | +0.85 [+0.58, +1.15] | 60 | < 0.0010 | <= 0.0040 | supported | supported (Holm) | yes: P2 - P +0.15 [+0.02, +0.30]; the verdict compares scores of two dates |

## The same day, the other claims and the share (declared, descriptive)

"C - H, same day" is (C - P2) + (P - H): the one-passage premium with the claim and the placebo both scored in this run. The share is C - H divided by O - H, given where the interval of O - H lies above zero.

| grader | C - P2 | C - H, same day | hedge in O - G | C - H as a share of O - H |
|---|---|---|---|---|
| Sonnet 5.5 | +0.18 [+0.03, +0.33] (n 60) | +0.43 [+0.17, +0.70] (n 60) | +0.25 [+0.02, +0.50] (n 60) | 0.35 |
| Opus 5.5 | +0.33 [+0.18, +0.48] (n 60) | +0.23 [-0.02, +0.48] (n 60) | +0.25 [+0.05, +0.45] (n 60) | 0.11 |
| Sonnet 4.6 | +0.12 [-0.08, +0.32] (n 60) | +0.40 [+0.08, +0.73] (n 60) | +0.10 [-0.17, +0.37] (n 60) | 0.59 |
| Opus 4.6 | +0.15 [-0.02, +0.33] (n 60) | +0.70 [+0.40, +1.03] (n 60) | +0.32 [+0.12, +0.52] (n 60) | 0.80 |
| DeepSeek V4 Pro | +0.00 [-0.33, +0.33] (n 60) | +1.03 [+0.60, +1.47] (n 60) | +0.30 [-0.02, +0.62] (n 60) | 0.61 |
| gpt-oss-120b | +0.43 [-0.17, +1.07] (n 54) | +1.59 [+0.67, +2.61] (n 54) | -0.04 [-0.70, +0.62] (n 53) | 1.90 |
| Qwen3-32B | +0.12 [+0.02, +0.22] (n 60) | +0.42 [+0.28, +0.55] (n 60) | +0.03 [-0.05, +0.12] (n 60) | 0.79 |
| Skywork 0.6B | +0.09 [+0.02, +0.15] (n 60) | +0.20 [+0.09, +0.30] (n 60) | +1.04 [+0.89, +1.19] (n 60) | 0.14 |
| Skywork 1.7B | +0.12 [+0.06, +0.17] (n 60) | +0.23 [+0.16, +0.29] (n 60) | +0.79 [+0.67, +0.93] (n 60) | 0.17 |
| Skywork 3B | +0.15 [+0.08, +0.21] (n 60) | +0.36 [+0.29, +0.44] (n 60) | +0.81 [+0.68, +0.94] (n 60) | 0.25 |
| Skywork 4B | +0.14 [+0.08, +0.19] (n 60) | +0.47 [+0.38, +0.55] (n 60) | +0.92 [+0.80, +1.04] (n 60) | 0.31 |
| Skywork 8B | +0.15 [+0.10, +0.19] (n 60) | +0.40 [+0.34, +0.48] (n 60) | +0.92 [+0.81, +1.03] (n 60) | 0.26 |

## The split by the kind of passage (declared, descriptive)

First value: the sets whose passage holds a plain statement of the code change (22 when no task is missing). Second value: the other sets (38). The result of a comparison is its value on all the row's tasks, in the first table.

| grader | O - H | C - H | C - P | C - G | P - H | O - C | P2 - P |
|---|---|---|---|---|---|---|---|
| Sonnet 5.5 | +1.50 [+1.09, +1.86] / +1.61 [+1.29, +1.92] | +0.64 [+0.32, +1.00] / +0.50 [+0.24, +0.79] | +0.27 [+0.00, +0.55] / +0.32 [+0.13, +0.53] | +0.45 [+0.18, +0.77] / +0.45 [+0.26, +0.63] | +0.36 [+0.09, +0.73] / +0.18 [-0.16, +0.53] | +0.86 [+0.55, +1.18] / +1.11 [+0.87, +1.32] | +0.09 [+0.00, +0.23] / +0.13 [-0.03, +0.32] |
| Opus 5.5 | +1.14 [+0.64, +1.59] / +1.58 [+1.21, +1.95] | -0.05 [-0.41, +0.36] / +0.26 [+0.05, +0.50] | +0.18 [-0.05, +0.41] / +0.29 [+0.08, +0.47] | +0.09 [-0.23, +0.41] / +0.53 [+0.21, +0.87] | -0.23 [-0.50, +0.05] / -0.03 [-0.29, +0.24] | +1.18 [+0.77, +1.59] / +1.32 [+1.05, +1.58] | -0.09 [-0.23, +0.00] / -0.08 [-0.26, +0.08] |
| Sonnet 4.6 | +0.59 [-0.09, +1.27] / +0.82 [+0.42, +1.21] | +0.09 [-0.27, +0.45] / +0.63 [+0.34, +0.95] | +0.36 [-0.05, +0.82] / +0.03 [-0.18, +0.24] | -0.18 [-0.50, +0.09] / +0.39 [+0.13, +0.68] | -0.27 [-0.68, +0.14] / +0.61 [+0.26, +0.97] | +0.50 [-0.09, +1.14] / +0.18 [-0.13, +0.47] | +0.18 [+0.00, +0.45] / -0.05 [-0.26, +0.13] |
| Opus 4.6 | +1.09 [+0.59, +1.59] / +1.05 [+0.71, +1.42] | +0.95 [+0.55, +1.45] / +0.79 [+0.45, +1.18] | +0.23 [-0.05, +0.50] / +0.34 [+0.08, +0.61] | +0.27 [+0.00, +0.55] / +0.42 [+0.21, +0.61] | +0.73 [+0.32, +1.23] / +0.45 [+0.08, +0.87] | +0.14 [-0.23, +0.50] / +0.26 [+0.05, +0.47] | +0.09 [-0.23, +0.41] / +0.18 [+0.05, +0.34] |
| DeepSeek V4 Pro | +2.18 [+1.36, +3.18] / +1.45 [+1.00, +1.89] | +0.91 [+0.00, +1.77] / +1.13 [+0.84, +1.45] | -0.14 [-0.86, +0.64] / +0.11 [-0.16, +0.37] | +0.09 [-0.68, +0.73] / +0.50 [+0.29, +0.76] | +1.05 [+0.45, +1.68] / +1.03 [+0.63, +1.42] | +1.27 [+0.45, +2.18] / +0.32 [+0.00, +0.63] | -0.05 [-0.32, +0.23] / +0.05 [-0.16, +0.29] |
| gpt-oss-120b | +1.40 [+0.25, +2.70] / +0.97 [+0.12, +1.94] | +1.45 [+0.60, +2.45] / +2.56 [+1.56, +3.62] | +1.15 [+0.40, +2.00] / +0.88 [+0.00, +1.82] | +0.35 [-0.95, +1.65] / +2.24 [+1.21, +3.29] | +0.30 [-0.40, +1.05] / +1.68 [+0.68, +2.68] | -0.05 [-1.60, +1.50] / -1.59 [-2.50, -0.74] | +1.70 [+0.70, +2.80] / -0.12 [-0.91, +0.68] |
| Qwen3-32B | +0.64 [+0.45, +0.82] / +0.66 [+0.47, +0.82] | +0.64 [+0.45, +0.82] / +0.45 [+0.29, +0.61] | +0.27 [+0.00, +0.55] / +0.18 [+0.00, +0.37] | +0.64 [+0.45, +0.82] / +0.53 [+0.37, +0.68] | +0.36 [+0.18, +0.55] / +0.26 [+0.13, +0.42] | +0.00 [-0.27, +0.27] / +0.21 [+0.03, +0.39] | +0.00 [-0.18, +0.18] / +0.16 [-0.05, +0.37] |
| Skywork 0.6B | +1.42 [+1.14, +1.73] / +1.37 [+1.19, +1.55] | -0.04 [-0.22, +0.16] / +0.33 [+0.24, +0.42] | -0.02 [-0.16, +0.13] / +0.15 [+0.09, +0.21] | +0.06 [-0.12, +0.26] / +0.44 [+0.38, +0.51] | -0.02 [-0.11, +0.09] / +0.19 [+0.12, +0.26] | +1.46 [+1.20, +1.72] / +1.04 [+0.86, +1.23] | +0.00 [+0.00, +0.00] / +0.00 [+0.00, +0.00] |
| Skywork 1.7B | +1.44 [+1.21, +1.71] / +1.21 [+1.07, +1.35] | +0.18 [+0.06, +0.32] / +0.25 [+0.18, +0.33] | +0.17 [+0.05, +0.29] / +0.08 [+0.04, +0.13] | +0.25 [+0.10, +0.40] / +0.37 [+0.31, +0.43] | +0.01 [-0.10, +0.12] / +0.17 [+0.11, +0.23] | +1.26 [+1.02, +1.52] / +0.96 [+0.80, +1.11] | +0.00 [+0.00, +0.00] / +0.00 [+0.00, +0.00] |
| Skywork 3B | +1.59 [+1.36, +1.83] / +1.38 [+1.22, +1.54] | +0.38 [+0.25, +0.54] / +0.35 [+0.27, +0.44] | +0.25 [+0.12, +0.39] / +0.09 [+0.04, +0.14] | +0.51 [+0.37, +0.66] / +0.52 [+0.44, +0.61] | +0.13 [+0.03, +0.23] / +0.27 [+0.21, +0.33] | +1.20 [+1.01, +1.41] / +1.02 [+0.85, +1.20] | +0.00 [+0.00, +0.00] / +0.00 [+0.00, +0.00] |
| Skywork 4B | +1.61 [+1.46, +1.80] / +1.48 [+1.33, +1.63] | +0.42 [+0.28, +0.57] / +0.49 [+0.39, +0.60] | +0.10 [-0.02, +0.22] / +0.16 [+0.11, +0.21] | +0.29 [+0.16, +0.42] / +0.37 [+0.32, +0.42] | +0.33 [+0.22, +0.44] / +0.34 [+0.24, +0.43] | +1.19 [+1.01, +1.37] / +0.99 [+0.83, +1.14] | +0.00 [+0.00, +0.00] / +0.00 [+0.00, +0.00] |
| Skywork 8B | +1.70 [+1.51, +1.94] / +1.50 [+1.35, +1.66] | +0.41 [+0.30, +0.54] / +0.40 [+0.31, +0.49] | +0.19 [+0.09, +0.28] / +0.12 [+0.08, +0.17] | +0.39 [+0.30, +0.48] / +0.32 [+0.27, +0.37] | +0.23 [+0.15, +0.32] / +0.28 [+0.20, +0.35] | +1.29 [+1.09, +1.50] / +1.10 [+0.95, +1.26] | +0.00 [+0.00, +0.00] / +0.00 [+0.00, +0.00] |

## The split, for the three comparisons with the claim

The same numbers as above, side by side: the 38 sets without a plain statement, the 22 sets with one, and all the row's tasks (the result). Note the order: the 38 sets come first here, and the 22 sets come first in the table above.

| grader | comparison | 38 sets | 22 sets | all |
|---|---|---|---|---|
| Sonnet 5.5 | C - H | +0.50 [+0.24, +0.79] (n 38) | +0.64 [+0.32, +1.00] (n 22) | +0.55 [+0.33, +0.77] (n 60) |
| Sonnet 5.5 | C - P | +0.32 [+0.13, +0.53] (n 38) | +0.27 [+0.00, +0.55] (n 22) | +0.30 [+0.13, +0.47] (n 60) |
| Sonnet 5.5 | C - G | +0.45 [+0.26, +0.63] (n 38) | +0.45 [+0.18, +0.77] (n 22) | +0.45 [+0.28, +0.63] (n 60) |
| Opus 5.5 | C - H | +0.26 [+0.05, +0.50] (n 38) | -0.05 [-0.41, +0.36] (n 22) | +0.15 [-0.07, +0.37] (n 60) |
| Opus 5.5 | C - P | +0.29 [+0.08, +0.47] (n 38) | +0.18 [-0.05, +0.41] (n 22) | +0.25 [+0.08, +0.40] (n 60) |
| Opus 5.5 | C - G | +0.53 [+0.21, +0.87] (n 38) | +0.09 [-0.23, +0.41] (n 22) | +0.37 [+0.12, +0.62] (n 60) |
| Sonnet 4.6 | C - H | +0.63 [+0.34, +0.95] (n 38) | +0.09 [-0.27, +0.45] (n 22) | +0.43 [+0.18, +0.68] (n 60) |
| Sonnet 4.6 | C - P | +0.03 [-0.18, +0.24] (n 38) | +0.36 [-0.05, +0.82] (n 22) | +0.15 [-0.05, +0.35] (n 60) |
| Sonnet 4.6 | C - G | +0.39 [+0.13, +0.68] (n 38) | -0.18 [-0.50, +0.09] (n 22) | +0.18 [-0.03, +0.40] (n 60) |
| Opus 4.6 | C - H | +0.79 [+0.45, +1.18] (n 38) | +0.95 [+0.55, +1.45] (n 22) | +0.85 [+0.58, +1.15] (n 60) |
| Opus 4.6 | C - P | +0.34 [+0.08, +0.61] (n 38) | +0.23 [-0.05, +0.50] (n 22) | +0.30 [+0.12, +0.50] (n 60) |
| Opus 4.6 | C - G | +0.42 [+0.21, +0.61] (n 38) | +0.27 [+0.00, +0.55] (n 22) | +0.37 [+0.20, +0.53] (n 60) |
| DeepSeek V4 Pro | C - H | +1.13 [+0.84, +1.45] (n 38) | +0.91 [+0.00, +1.77] (n 22) | +1.05 [+0.67, +1.43] (n 60) |
| DeepSeek V4 Pro | C - P | +0.11 [-0.16, +0.37] (n 38) | -0.14 [-0.86, +0.64] (n 22) | +0.02 [-0.32, +0.35] (n 60) |
| DeepSeek V4 Pro | C - G | +0.50 [+0.29, +0.76] (n 38) | +0.09 [-0.68, +0.73] (n 22) | +0.35 [+0.03, +0.65] (n 60) |
| gpt-oss-120b | C - H | +2.56 [+1.56, +3.62] (n 34) | +1.45 [+0.60, +2.45] (n 20) | +2.15 [+1.46, +2.89] (n 54) |
| gpt-oss-120b | C - P | +0.88 [+0.00, +1.82] (n 34) | +1.15 [+0.40, +2.00] (n 20) | +0.98 [+0.37, +1.61] (n 54) |
| gpt-oss-120b | C - G | +2.24 [+1.21, +3.29] (n 34) | +0.35 [-0.95, +1.65] (n 20) | +1.54 [+0.68, +2.39] (n 54) |
| Qwen3-32B | C - H | +0.45 [+0.29, +0.61] (n 38) | +0.64 [+0.45, +0.82] (n 22) | +0.52 [+0.38, +0.65] (n 60) |
| Qwen3-32B | C - P | +0.18 [+0.00, +0.37] (n 38) | +0.27 [+0.00, +0.55] (n 22) | +0.22 [+0.07, +0.37] (n 60) |
| Qwen3-32B | C - G | +0.53 [+0.37, +0.68] (n 38) | +0.64 [+0.45, +0.82] (n 22) | +0.57 [+0.43, +0.70] (n 60) |
| Skywork 0.6B | C - H | +0.33 [+0.24, +0.42] (n 38) | -0.04 [-0.22, +0.16] (n 22) | +0.20 [+0.09, +0.30] (n 60) |
| Skywork 0.6B | C - P | +0.15 [+0.09, +0.21] (n 38) | -0.02 [-0.16, +0.13] (n 22) | +0.09 [+0.02, +0.15] (n 60) |
| Skywork 0.6B | C - G | +0.44 [+0.38, +0.51] (n 38) | +0.06 [-0.12, +0.26] (n 22) | +0.30 [+0.21, +0.40] (n 60) |
| Skywork 1.7B | C - H | +0.25 [+0.18, +0.33] (n 38) | +0.18 [+0.06, +0.32] (n 22) | +0.23 [+0.16, +0.29] (n 60) |
| Skywork 1.7B | C - P | +0.08 [+0.04, +0.13] (n 38) | +0.17 [+0.05, +0.29] (n 22) | +0.12 [+0.06, +0.17] (n 60) |
| Skywork 1.7B | C - G | +0.37 [+0.31, +0.43] (n 38) | +0.25 [+0.10, +0.40] (n 22) | +0.32 [+0.26, +0.39] (n 60) |
| Skywork 3B | C - H | +0.35 [+0.27, +0.44] (n 38) | +0.38 [+0.25, +0.54] (n 22) | +0.36 [+0.29, +0.44] (n 60) |
| Skywork 3B | C - P | +0.09 [+0.04, +0.14] (n 38) | +0.25 [+0.12, +0.39] (n 22) | +0.15 [+0.08, +0.21] (n 60) |
| Skywork 3B | C - G | +0.52 [+0.44, +0.61] (n 38) | +0.51 [+0.37, +0.66] (n 22) | +0.52 [+0.44, +0.59] (n 60) |
| Skywork 4B | C - H | +0.49 [+0.39, +0.60] (n 38) | +0.42 [+0.28, +0.57] (n 22) | +0.47 [+0.38, +0.55] (n 60) |
| Skywork 4B | C - P | +0.16 [+0.11, +0.21] (n 38) | +0.10 [-0.02, +0.22] (n 22) | +0.14 [+0.08, +0.19] (n 60) |
| Skywork 4B | C - G | +0.37 [+0.32, +0.42] (n 38) | +0.29 [+0.16, +0.42] (n 22) | +0.34 [+0.28, +0.40] (n 60) |
| Skywork 8B | C - H | +0.40 [+0.31, +0.49] (n 38) | +0.41 [+0.30, +0.54] (n 22) | +0.40 [+0.34, +0.48] (n 60) |
| Skywork 8B | C - P | +0.12 [+0.08, +0.17] (n 38) | +0.19 [+0.09, +0.28] (n 22) | +0.15 [+0.10, +0.19] (n 60) |
| Skywork 8B | C - G | +0.32 [+0.27, +0.37] (n 38) | +0.39 [+0.30, +0.48] (n 22) | +0.34 [+0.30, +0.39] (n 60) |

## C - G on the sets that the keyword rule marks (amendment of 2026-11-14, descriptive)

The keyword rule of mark_kept_lines.py marks 36 of the sets: G keeps a line of the agent that says the case was reproduced, tested or verified, that it works now, or that the problem is fixed, so the hedge may still contradict its surroundings. The difference between the two groups is not tested.

| grader | C - G, marked sets | C - G, other sets | C - G, all |
|---|---|---|---|
| Sonnet 5.5 | +0.67 [+0.47, +0.89] (n 36) | +0.12 [-0.08, +0.38] (n 24) | +0.45 [+0.28, +0.63] (n 60) |
| Opus 5.5 | +0.69 [+0.36, +1.03] (n 36) | -0.12 [-0.42, +0.12] (n 24) | +0.37 [+0.12, +0.62] (n 60) |
| Sonnet 4.6 | +0.33 [+0.03, +0.61] (n 36) | -0.04 [-0.29, +0.21] (n 24) | +0.18 [-0.03, +0.40] (n 60) |
| Opus 4.6 | +0.44 [+0.22, +0.67] (n 36) | +0.25 [+0.00, +0.50] (n 24) | +0.37 [+0.20, +0.53] (n 60) |
| DeepSeek V4 Pro | +0.28 [-0.19, +0.69] (n 36) | +0.46 [+0.17, +0.75] (n 24) | +0.35 [+0.03, +0.65] (n 60) |
| gpt-oss-120b | +1.81 [+0.87, +2.84] (n 31) | +1.17 [-0.30, +2.74] (n 23) | +1.54 [+0.68, +2.39] (n 54) |
| Qwen3-32B | +0.53 [+0.36, +0.69] (n 36) | +0.62 [+0.42, +0.83] (n 24) | +0.57 [+0.43, +0.70] (n 60) |
| Skywork 0.6B | +0.35 [+0.23, +0.45] (n 36) | +0.23 [+0.07, +0.40] (n 24) | +0.30 [+0.21, +0.40] (n 60) |
| Skywork 1.7B | +0.37 [+0.28, +0.46] (n 36) | +0.26 [+0.17, +0.36] (n 24) | +0.32 [+0.26, +0.39] (n 60) |
| Skywork 3B | +0.53 [+0.44, +0.63] (n 36) | +0.49 [+0.39, +0.62] (n 24) | +0.52 [+0.44, +0.59] (n 60) |
| Skywork 4B | +0.35 [+0.28, +0.42] (n 36) | +0.32 [+0.21, +0.42] (n 24) | +0.34 [+0.28, +0.40] (n 60) |
| Skywork 8B | +0.35 [+0.30, +0.41] (n 36) | +0.33 [+0.25, +0.41] (n 24) | +0.34 [+0.30, +0.39] (n 60) |

## Reasons that name a contradiction (keyword count of reanalysis r28, both lists)

Count by the list of the first count, with the narrow list in brackets, of the reasons with a valid score. The first rows are this run's; the rest are reanalysis r28's counts for the main run.

| judge | version | reasons | count |
|---|---|---|---|
| Sonnet 5.5 | claim alone (C) | 60 | 16 (16) |
| Sonnet 5.5 | hedge alone (G) | 60 | 42 (41) |
| Sonnet 5.5 | placebo, second score (P2) | 60 | 19 (18) |
| Opus 5.5 | claim alone (C) | 60 | 8 (8) |
| Opus 5.5 | hedge alone (G) | 60 | 41 (40) |
| Opus 5.5 | placebo, second score (P2) | 60 | 9 (8) |
| Sonnet 4.6 | claim alone (C) | 60 | 5 (5) |
| Sonnet 4.6 | hedge alone (G) | 60 | 15 (13) |
| Sonnet 4.6 | placebo, second score (P2) | 60 | 8 (7) |
| Opus 4.6 | claim alone (C) | 60 | 2 (2) |
| Opus 4.6 | hedge alone (G) | 60 | 6 (4) |
| Opus 4.6 | placebo, second score (P2) | 60 | 2 (2) |
| DeepSeek V4 Pro | claim alone (C) | 60 | 4 (4) |
| DeepSeek V4 Pro | hedge alone (G) | 60 | 3 (2) |
| DeepSeek V4 Pro | placebo, second score (P2) | 60 | 3 (3) |
| gpt-oss-120b | claim alone (C) | 57 | 1 (1) |
| gpt-oss-120b | hedge alone (G) | 60 | 0 (0) |
| gpt-oss-120b | placebo, second score (P2) | 60 | 0 (0) |
| Qwen3-32B | claim alone (C) | 60 | 0 (0) |
| Qwen3-32B | hedge alone (G) | 60 | 1 (1) |
| Qwen3-32B | placebo, second score (P2) | 60 | 1 (1) |
| Sonnet 5.5 | main claim replaced by the hedge | 60 | 53 (53) |
| Sonnet 5.5 | overclaim | 60 | 2 (2) |
| Sonnet 5.5 | honest message | 60 | 22 (20) |
| Sonnet 5.5 | placebo | 60 | 21 (20) |
| Opus 5.5 | main claim replaced by the hedge | 60 | 56 (55) |
| Opus 5.5 | overclaim | 60 | 1 (1) |
| Opus 5.5 | honest message | 60 | 11 (11) |
| Opus 5.5 | placebo | 60 | 10 (10) |
| Sonnet 4.6 | main claim replaced by the hedge | 60 | 30 (7) |
| Sonnet 4.6 | overclaim | 60 | 2 (2) |
| Sonnet 4.6 | honest message | 60 | 22 (11) |
| Sonnet 4.6 | placebo | 60 | 5 (5) |
| Opus 4.6 | main claim replaced by the hedge | 60 | 15 (3) |
| Opus 4.6 | overclaim | 60 | 0 (0) |
| Opus 4.6 | honest message | 60 | 8 (2) |
| Opus 4.6 | placebo | 60 | 1 (1) |
| Gemini 3.8 Flash | main claim replaced by the hedge | 60 | 35 (32) |
| Gemini 3.8 Flash | overclaim | 60 | 0 (0) |
| Gemini 3.8 Flash | honest message | 60 | 2 (2) |
| Gemini 3.8 Flash | placebo | 60 | 2 (2) |
| Gemini 4 Argon | main claim replaced by the hedge | 60 | 49 (49) |
| Gemini 4 Argon | overclaim | 60 | 2 (2) |
| Gemini 4 Argon | honest message | 60 | 3 (3) |
| Gemini 4 Argon | placebo | 60 | 0 (0) |
| DeepSeek V4 Pro | main claim replaced by the hedge | 60 | 9 (4) |
| DeepSeek V4 Pro | overclaim | 60 | 0 (0) |
| DeepSeek V4 Pro | honest message | 60 | 8 (6) |
| DeepSeek V4 Pro | placebo | 60 | 3 (3) |
| gpt-oss-120b | main claim replaced by the hedge | 59 | 0 (0) |
| gpt-oss-120b | overclaim | 58 | 0 (0) |
| gpt-oss-120b | honest message | 59 | 0 (0) |
| gpt-oss-120b | placebo | 60 | 1 (1) |
| Qwen3-32B | main claim replaced by the hedge | 60 | 1 (1) |
| Qwen3-32B | overclaim | 60 | 1 (1) |
| Qwen3-32B | honest message | 60 | 1 (1) |
| Qwen3-32B | placebo | 60 | 1 (1) |

## Missing scores

Per grader: the records present, and per version the scores that are missing (no record, or a reply with no valid score after the second attempt).

| grader | records | C | G | P2 |
|---|---|---|---|---|
| Sonnet 5.5 | 180 of 180 | 0 | 0 | 0 |
| Opus 5.5 | 180 of 180 | 0 | 0 | 0 |
| Sonnet 4.6 | 180 of 180 | 0 | 0 | 0 |
| Opus 4.6 | 180 of 180 | 0 | 0 | 0 |
| DeepSeek V4 Pro | 180 of 180 | 0 | 0 | 0 |
| gpt-oss-120b | 180 of 180 | 3 | 0 | 0 |
| Qwen3-32B | 180 of 180 | 0 | 0 | 0 |
| Skywork 0.6B | 180 of 180 | 0 | 0 | 0 |
| Skywork 1.7B | 180 of 180 | 0 | 0 | 0 |
| Skywork 3B | 180 of 180 | 0 | 0 | 0 |
| Skywork 4B | 180 of 180 | 0 | 0 | 0 |
| Skywork 8B | 180 of 180 | 0 | 0 | 0 |

No score from: Gemini 3.8 Flash, Gemini 4 Argon.

## Model names and providers, main run and this run

| judge | main run | this run | same name | providers, main run | providers, this run | same providers |
|---|---|---|---|---|---|---|
| Sonnet 5.5 | claude-sonnet-5-5 | claude-sonnet-5-5 | yes |  |  | yes |
| Opus 5.5 | claude-opus-5-5 | claude-opus-5-5 | yes |  |  | yes |
| Sonnet 4.6 | claude-sonnet-4-6 | claude-sonnet-4-6 | yes |  |  | yes |
| Opus 4.6 | claude-opus-4-6 | claude-opus-4-6 | yes |  |  | yes |
| DeepSeek V4 Pro | deepseek/deepseek-v4-pro | deepseek/deepseek-v4-pro | yes | Alibaba, Nebius, Novita, SiliconFlow, StreamLake | Baidu, GMICloud, Novita, SiliconFlow, StreamLake | NO |
| gpt-oss-120b | openai/gpt-oss-120b | openai/gpt-oss-120b | yes | Cerebras, Crusoe, DeepInfra, DekaLLM, Mancer 2 | Crusoe, DekaLLM | NO |
| Qwen3-32B | qwen/qwen3-32b | qwen/qwen3-32b | yes | DeepInfra, SiliconFlow | DeepInfra, Tenstorrent | NO |

## The reward models' second score of the placebo

| reward model | tasks whose second score differs | largest difference (model's own units) |
|---|---|---|
| Skywork 0.6B | 0 of 60 | 0 |
| Skywork 1.7B | 0 of 60 | 0 |
| Skywork 3B | 0 of 60 | 0 |
| Skywork 4B | 0 of 60 | 0 |
| Skywork 8B | 0 of 60 | 0 |

## The reward models' weights and the placebo scored first (amendment of 2026-11-14)

The main run recorded the repository and the byte count of the weights, no revision and no file hash. "Files in the trace": files of the loaded snapshot whose content hash is among the hashes that the Hub client's lock files of 2026-11-05 and 2026-11-08 name (main_run_weights.json). The placebo was scored before C and G and compared with the main run's stored scores, in the model's own score units; above 0.05 the model stops.

| reward model | revision | files in the trace | weights bytes, now and main run | placebo: tasks that differ | placebo: largest absolute difference | placebo: tasks with another token count | passed |
|---|---|---|---|---|---|---|---|
| Skywork 0.6B | 8c14a4e9e632 | 9 of 9 | 1192137232 and 1192137232 | 0 of 60 | 0 | 0 | yes |
| Skywork 1.7B | e51ea3e08fb8 | 9 of 9 | 3441189792 and 3441189792 | 0 of 60 | 0 | 0 | yes |
| Skywork 3B | beafe5f2e6a2 | 8 of 8 | 6425505792 and 6425505792 | 0 of 60 | 0 | 0 | yes |
| Skywork 4B | fd958fef475f | 11 of 11 | 8044941312 and 8044941312 | 0 of 60 | 0 | 0 | yes |
| Skywork 8B | 6f19fdefb933 | 13 of 13 | 15136819200 and 15136819200 | 0 of 60 | 0 | 0 | yes |

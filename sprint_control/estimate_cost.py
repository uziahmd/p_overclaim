"""The cost table of the control run, before any call: calls and estimated cost per grader. No model is called.

The estimate comes from the token counts of the main run's own calls without the transcript
(results/full_run/pointwise_calls.jsonl, the scoring prompt of the paper's main rows, first sample): for each judge,
its five calls on each of the 60 tasks. This run sends the same prompt for the same tasks, three calls per task: C,
G and the placebo. All three are the edited message, so each is costed as the main run's call on the placebo of
that task.

Every call is priced from its tokens, not from the cost the main run booked. The main run read 57 of each of six
judges' 300 replies back from its disk cache (replies of the pilot of 2026-11-06 to the identical request) and
booked those at 0 USD; this run starts with an empty cache and pays for every call. The price of a call is its
tokens at the list prices of configs/models.yaml, times the judge's ratio of booked cost to list-price cost over
the main run's live calls. That ratio is 1 for the Claude and Gemini judges; for the OpenRouter judges it is what
OpenRouter charged over the list price.

- estimate: per task, 3 x the price of the judge's call on the placebo; summed over the built sets. A call's token
  counts include a second attempt after an unreadable reply (gpt-oss-120b needed one in 21 of its 300 calls).
- upper bound: per task, 3 x the most expensive of the judge's five calls on that task (the five versions of the
  message differ in length and in how long the judge thought); summed over the built sets.

Prices are those of configs/models.yaml, in force on the main run's day. The reward models run on the local GPU and
cost nothing. The Gemini rows are counted only if the one check of the vertex route answered; before the check the
table shows them and says so.

Output: sprint_control/cost_estimate.md and the same text on the console.
Run: uv run python sprint_control/estimate_cost.py
"""

from __future__ import annotations

import statistics as st

import control_common as K
from run_control import COST_PROVIDER, judge_spec, load_config, vertex_answered

PROVIDER_NAMES = {"anthropic": "Anthropic API", "openrouter": "OpenRouter", "google": "Google Cloud (vertex)"}


def main_run_calls() -> dict[str, list[dict]]:
    """{judge: its calls of the main run without the transcript, scoring prompt of the main rows, first sample}."""
    out: dict[str, list[dict]] = {}
    for r in K.read_jsonl(K.MAIN_RUN / "pointwise_calls.jsonl"):
        if (r["condition"] == K.CONDITION and r["rubric"] == "pointwise-neutral" and r["set"] == K.SET_NAME
                and int(r.get("replicate") or 0) == 0):
            out.setdefault(r["grader"], []).append(r)
    return out


def list_price(rec: dict, price: dict, gemini: bool) -> float:
    """The call's tokens at list prices (USD). Claude and OpenRouter count thinking inside the output tokens; Gemini
    reports thought tokens beside them and bills them as output, and bills cached input at its own price."""
    u = rec["usage"]
    cached = int(u.get("cache_read_input_tokens") or 0) if gemini else 0
    out = u["output_tokens"] + ((u.get("thought_tokens") or 0) if gemini else 0)
    return ((u["input_tokens"] - cached) * price["input"] + cached * price.get("implicit_cache_read", price["input"])
            + out * price["output"]) / 1e6


def rows(cfg: dict) -> list[dict]:
    calls = main_run_calls()
    built = [r["task_id"] for r in K.read_jsonl(K.MESSAGES_FILE)]
    k = len(cfg["variants"])
    out = []
    for run in cfg["judges"]:
        spec = judge_spec(cfg, run)
        gemini = spec["provider"] == "gemini"
        price = cfg["prices"][spec["model"]]
        recs = [r for r in calls[run["judge"]] if r["task_id"] in set(built)]
        live = [r for r in recs if not r["cached"]]
        ratio = sum(float(r["cost_usd"]) for r in live) / sum(list_price(r, price, gemini) for r in live)
        by_task: dict[str, dict[str, float]] = {}
        for r in recs:
            by_task.setdefault(r["task_id"], {})[r["variant"]] = ratio * list_price(r, price, gemini)
        if set(by_task) != set(built) or any(K.PLACEBO not in v for v in by_task.values()):
            raise SystemExit(f"{run['judge']}: the main run has no call on the placebo of some built set")
        placebo = [r for r in recs if r["variant"] == K.PLACEBO]
        out.append({
            "judge": run["judge"], "provider": COST_PROVIDER[spec["provider"]], "calls": k * len(built),
            "mean_input_tokens": st.mean(r["usage"]["input_tokens"] for r in placebo),
            "mean_output_tokens": st.mean(r["usage"]["output_tokens"]
                                          + ((r["usage"].get("thought_tokens") or 0) if gemini else 0)
                                          for r in placebo),
            "second_attempts": sum(r["parse_attempts"] > 1 for r in recs), "main_run_calls": len(recs),
            "read_from_cache": len(recs) - len(live), "charged_over_list": ratio,
            "placebo_costs": [by_task[t][K.PLACEBO] for t in built],
            "estimate": k * sum(by_task[t][K.PLACEBO] for t in built),
            "upper": k * sum(max(by_task[t].values()) for t in built),
        })
    return out


def gemini_counted() -> bool:
    """Whether the Gemini rows are in the totals: before the check of the vertex route, or when it answered."""
    return K.read_json(K.VERTEX_CHECK_FILE) is None or vertex_answered()[0]


def smaller_versions(cfg: dict, table: list[dict]) -> list[str]:
    """The estimate of the brief's two smaller versions, per provider (only printed when a cap would be passed):
    (a) without G for Sonnet 4.6 and Opus 4.6; (b) with the second placebo score on the first 30 built sets."""
    lines = ["| smaller version | Anthropic API | OpenRouter | Google Cloud (vertex) |", "|---|---|---|---|"]
    for label, saved in (("without G for Sonnet 4.6 and Opus 4.6",
                          lambda r: sum(r["placebo_costs"]) if r["judge"] in ("claude-sonnet-4-6",
                                                                              "claude-opus-4-6") else 0.0),
                         ("second placebo score on 30 tasks", lambda r: sum(r["placebo_costs"][30:]))):
        cells = []
        for provider in ("anthropic", "openrouter", "google"):
            rows_ = [r for r in table if r["provider"] == provider and (provider != "google" or gemini_counted())]
            cells.append(f"{sum(r['estimate'] - saved(r) for r in rows_):.2f}")
        lines.append(f"| {label} (estimate, USD) | " + " | ".join(cells) + " |")
    return lines + [""]


def render(cfg: dict) -> str:
    table = rows(cfg)
    ok, why = vertex_answered()
    checked = K.read_json(K.VERTEX_CHECK_FILE) is not None
    check_cost = float((K.read_json(K.VERTEX_CHECK_FILE) or {}).get("cost_usd") or 0.0)
    lines = ["# Cost table of the control run (estimate, before any scoring call)", "",
             "Written by `sprint_control/estimate_cost.py` from the main run's calls without the transcript. "
             f"Each grader scores {table[0]['calls'] // len(cfg['variants'])} sets x {len(cfg['variants'])} messages "
             f"(C, G and the placebo again) = {table[0]['calls']} calls. The estimate prices each call as the main "
             "run's call on the placebo of the same task; the upper bound as the most expensive of the main run's "
             "five calls on that task. Every call is priced from its token counts, because the main run read some "
             "replies back from its cache and booked them at 0 USD, and this run pays for every call. Tokens are "
             "the means of the main run's calls on the placebo; output includes thinking tokens.", ""]
    if not checked:
        lines += ["The check of the vertex route has not been made yet: the two Gemini rows are shown and counted "
                  "as if it answers.", ""]
    elif ok:
        lines += ["The check of the vertex route answered: the two Gemini judges take part.", ""]
    else:
        lines += [f"The two Gemini judges do not take part ({why}); their rows are shown for reference and are not "
                  "in the totals.", ""]
    lines += ["| grader | billed to | calls | input tokens per call | output tokens per call | estimate (USD) | "
              "upper bound (USD) |", "|---|---|---|---|---|---|---|"]
    totals: dict[str, list[float]] = {}
    for r in table:
        counted = r["provider"] != "google" or ok or not checked
        mark = "" if counted else " (not run)"
        lines.append(f"| {r['judge']}{mark} | {PROVIDER_NAMES[r['provider']]} | {r['calls']} | "
                     f"{r['mean_input_tokens']:,.0f} | {r['mean_output_tokens']:,.0f} | {r['estimate']:.2f} | "
                     f"{r['upper']:.2f} |")
        if counted:
            t = totals.setdefault(r["provider"], [0.0, 0.0, 0])
            t[0] += r["estimate"]
            t[1] += r["upper"]
            t[2] += r["calls"]
    for spec in cfg["rm_specs"]:
        lines.append(f"| {spec['name']} | local GPU | {table[0]['calls']} | | | 0.00 | 0.00 |")
    lines += ["", "| billed to | calls | estimate (USD) | upper bound (USD) | cap (USD) | upper bound within the cap |",
              "|---|---|---|---|---|---|"]
    over = []
    for provider in ("anthropic", "openrouter", "google"):
        est, upper, n = totals.get(provider, [0.0, 0.0, 0])
        if provider == "google":
            est, upper, n = est + check_cost, upper + check_cost, n + (1 if checked else 0)
        cap = float(cfg["budget"][f"{provider}_usd_cap"])
        lines.append(f"| {PROVIDER_NAMES[provider]} | {n} | {est:.2f} | {upper:.2f} | {cap:.2f} | "
                     f"{'yes' if upper <= cap else 'NO'} |")
        if upper > cap:
            over.append(provider)
    lines += [""]
    if checked:
        lines += [f"The Google Cloud line includes the one check of the vertex route ({check_cost:.6f} USD).", ""]
    if over:
        lines += [f"The upper bound passes the cap of: {', '.join(PROVIDER_NAMES[p] for p in over)}. The brief's two "
                  "smaller versions must be costed before the run.", ""]
        lines += smaller_versions(cfg, table)
    else:
        lines += ["No estimate and no upper bound passes a cap, so the two smaller versions of the brief (without G "
                  "for Sonnet 4.6 and Opus 4.6; the second placebo score on 30 tasks) are not needed.", "",
                  "The runner stops before any call that would pass a cap, whatever this table says. Before each "
                  "call it adds the price of that call's input and of the largest reply the judge may give to the "
                  "money already booked, and it stops at the cap minus a margin for the calls that other threads "
                  f"have in flight ({', '.join(f'{v:.2f} USD {PROVIDER_NAMES[p2]}' for p2, v in cfg['in_flight_margin_usd'].items())}).", ""]
    second = {r["judge"]: (r["second_attempts"], r["main_run_calls"]) for r in table if r["second_attempts"]}
    if second:
        lines += ["Second attempts after an unreadable reply in the main run (their tokens are in the call's "
                  "token counts): " + ", ".join(f"{j} {n} of {r_n} calls" for j, (n, r_n) in second.items()) + ".",
                  ""]
    cached = {r["judge"]: r["read_from_cache"] for r in table if r["read_from_cache"]}
    if cached:
        lines += ["Replies that the main run read back from its cache and booked at 0 USD, priced here from their "
                  "tokens: " + ", ".join(f"{j} {n}" for j, n in cached.items()) + " (of 300 each).", ""]
    lines += ["OpenRouter's charge over the list price in the main run's live calls: "
              + ", ".join(f"{r['judge']} {r['charged_over_list']:.2f}x" for r in table if r["provider"] == "openrouter")
              + ".", ""]
    return "\n".join(lines)


def main() -> None:
    text = render(load_config())
    K.ESTIMATE_FILE.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

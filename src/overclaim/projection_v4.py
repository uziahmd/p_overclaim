"""Session-5 cost projection (session-4 brief, Step 10): arithmetic on recorded costs, nothing is run.

python -m overclaim.projection_v4 --config configs/projection_session5.yaml writes the configured JSON and markdown
files and prints the markdown. Pricing, running totals, the Anthropic budget and the cumulative cut totals are
overclaim.projection's (unit_stats, unit_cost, project_rescoped); this module builds the per-domain stats, turns task
counts into calls, decides which cuts are needed and adds the OpenRouter and Google lines.

Stats keys are (name, condition label). A measured label carries the domain and the call kind: "hidden@coding",
"shown@research", "full/hidden@coding" (full rubric), "pointwise/hidden@research", "e2_score/hidden@coding". The bare
labels "hidden" and "shown" are the basis that unit_cost and project_rescoped's e3_minus_plain read literally: the
neutral pairwise calls of the one record source marked ratio_basis, where every judge saw the same prompts. So
unit_cost prices a missing (grader, label) from the reference judge's profile under the same label (same domain and
kind), scaled by the two graders' tokenizer ratio on the same prompts, with the grader's own output tokens.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import yaml

from overclaim import projection
from overclaim.paths import REPO_ROOT
from overclaim.verify_variants import _fill

MISSING = "missing:"   # condition prefix that no stats key has: forces unit_cost to return None


def read_jsonl(path: Path) -> list[dict]:
    """Complete lines of a JSONL file ([] if missing). Stages append to these files while this runs, so the text
    after the last newline (a line still being written) is left out, as pilot4.sets_spend does."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")[:-1] if line.strip()]


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def label(condition: str, domain: str, kind: str = "") -> str:
    return f"{kind}/{condition}@{domain}" if kind else f"{condition}@{domain}"


def _add(stats: dict, new: dict, source: str) -> None:
    """Add stats entries; a key already present (from a preferred source) is kept."""
    for key, value in new.items():
        stats.setdefault(key, {**value, "source": source})


def purposes() -> dict[str, str]:
    """cost.jsonl purposes, read from the modules that log them."""
    from overclaim.domains.research import harness, sets
    from overclaim.experiments import e2_selection as e2
    from overclaim.experiments import e3_verification as e3
    from overclaim.experiments import pilot4

    return {"e2_policy": e2.PACKET_PURPOSE, "e2_score": e2.POINTWISE_PURPOSE,
            "e2_label": inspect.signature(e2.label_message).parameters["purpose"].default,
            "e2_label_full": e2.LABEL_FULL_PURPOSE, "e3_judge3": e3.JUDGE3_PURPOSE,
            "research_agent": harness.PURPOSE, "coding_sets": sorted(pilot4.sets_purposes()),
            "research_sets": [sets.PURPOSE_DETECTION, sets.PURPOSE_PRECHECK, sets.PURPOSE_REWRITE,
                              sets.PURPOSE_VERIFICATION]}


def _purpose_usd(cost: list[dict], names: list[str]) -> float:
    return math.fsum(float(r["cost_usd"]) for r in cost if r.get("purpose") in names and r.get("cost_usd") is not None)


def build_stats(cfg: dict, root: Path, cost: list[dict], base: dict) -> dict:
    """{(name, label): {"n", "usd", "tokens", "source"[, "basis"]}}, live calls only.

    Record files (cfg["record_sources"], in order of preference: the first source with live calls for a key wins),
    one label per rubric through the source's `rubrics` map, so the neutral and full rubrics are never pooled; the
    neutral calls of the source marked ratio_basis also give the bare labels. Then cost.jsonl by purpose (it logs
    uncached calls only): the Experiment 2 policy, selectors (per model), both labellers and the three-label
    Experiment 3 judge (name <model>+verify_first, also under the bare "shown", which e3_minus_plain reads: it ran
    on pilot 3's v3 sets). Then the per-unit costs: coding and research set building per built set and the research
    agent per trajectory."""
    thought = tuple(cfg.get("thought_separate") or ())
    stats: dict = {}
    for src in cfg["record_sources"]:
        kinds, sets = src["rubrics"], src.get("sets")
        recs = [r for r in read_jsonl(root / src["path"])
                if r.get("rubric") in kinds and (sets is None or r.get("set") in sets)]
        relabelled = [{**r, "condition": label(r["condition"], src["domain"], kinds[r["rubric"]])} for r in recs]
        _add(stats, projection.unit_stats(relabelled, thought_separate=thought), src["source"])
        if src.get("ratio_basis"):
            neutral = [r for r in recs if kinds[r["rubric"]] == ""]
            _add(stats, projection.unit_stats(neutral, thought_separate=thought), f"{src['source']}, ratio basis")
    p = purposes()
    by_purpose = [("e2_policy", lambda m: "e2_policy", ["packet@coding"]),
                  ("e2_score", lambda m: m, ["e2_score/hidden@coding"]),
                  ("e2_label", lambda m: "e2_label", ["label@coding"]),
                  ("e2_label_full", lambda m: "e2_label_full", ["label@coding"]),
                  ("e3_judge3", lambda m: f"{m}+verify_first", ["shown@coding", "shown"])]
    source = f"{cfg['cost_log']} by purpose"
    for key, name, conds in by_purpose:
        for cond in conds:
            pseudo = [{"grader": name(r["model"]), "condition": cond, "cost_usd": r.get("cost_usd"), "usage": r}
                      for r in cost if r.get("purpose") == p[key]]
            _add(stats, projection.unit_stats(pseudo), f"{source} {p[key]}")
    zero = {b: 0.0 for b in projection.BUCKETS}
    units = [(("coding_set", "build@coding"), p["coding_sets"], base["coding_sets"],
              f"{cfg['sets']['coding']} sets"),
             (("research_set", "build@research"), p["research_sets"], base["research_sets"],
              f"{cfg['sets']['research']} sets"),
             (("research_trajectory", "agent@research"), [p["research_agent"]], base["trajectories"],
              f"trajectories in {cfg['research_trajectories']}")]
    for key, names, n, what in units:
        total = _purpose_usd(cost, names)
        if n and total > 0:
            stats[key] = {"n": n, "usd": total / n, "tokens": dict(zero), "source": source,
                          "basis": (f"measured: {total:.4f} USD of purposes {', '.join(names)} in {cfg['cost_log']} "
                                    f"/ {n} {what}")}
    return stats


def base_counts(cfg: dict, root: Path) -> dict:
    """Counts read from the set files, the research funnel and the trajectories file."""
    out: dict = {}
    for dom in ("coding", "research"):
        sets = read_jsonl(root / cfg["sets"][dom])
        out[f"{dom}_sets"] = len(sets)
        out[f"{dom}_eligible"] = sum(1 for s in sets if s.get("shown_eligible") is True)
    funnel = read_json(root / cfg["research_funnel"]) or {}
    batch = (funnel.get("gaps") or {}).get(cfg["research_funnel_batch"]) or {}
    rs = funnel.get("research_sets") or {}
    out["funnel_trajectories"] = batch.get("trajectories")
    labels = batch.get("labels") or {}
    out["funnel_gap_trajectories"] = sum(v for k, v in labels.items() if k != "no_gap") if labels else None
    out["funnel_built"] = (rs.get("outcomes") or {}).get("built")
    out["funnel_tried"] = sum(g.get("tried", 0) for g in (rs.get("by_gap") or {}).values()) or None
    out["funnel_by_gap"] = {g: {"trajectories": labels.get(g), "tried": v.get("tried"), "built": v.get("built"),
                                "list_exhausted": v.get("list_exhausted")} for g, v in (rs.get("by_gap") or {}).items()}
    out["pool_size"] = (funnel.get("pool") or {}).get("pool_size")
    out["trajectories"] = len(read_jsonl(root / cfg["research_trajectories"]))
    return out


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def scenario_counts(base: dict, coding: int, research: int, e2_tasks: int) -> dict:
    """Task counts for one scenario. Shown-eligible = ceil(eligible sets x tasks / sets in the file). Trajectories
    for the research sets still owed = ceil(owed / (gap rate x build yield)), gap rate = gap trajectories /
    trajectories of the funnel batch, build yield = sets built / gap tasks tried (None when a count is missing)."""
    out = {"coding": coding, "research": research, "e2_tasks": e2_tasks,
           "coding_sets_owed": max(0, coding - base["coding_sets"]),
           "research_sets_owed": max(0, research - base["research_sets"])}
    for dom in ("coding", "research"):
        if not base[f"{dom}_sets"]:
            raise ValueError(f"no {dom} sets: the shown-eligible count cannot be scaled")
        out[f"shown_{dom}"] = _ceil_div(base[f"{dom}_eligible"] * out[dom], base[f"{dom}_sets"])
    parts = [base.get(k) for k in ("funnel_gap_trajectories", "funnel_trajectories", "funnel_built", "funnel_tried")]
    if all(parts):
        rate = Fraction(parts[0], parts[1]) * Fraction(parts[2], parts[3])
        out["research_trajectories"] = math.ceil(out["research_sets_owed"] / rate)
    else:
        out["research_trajectories"] = None
    return out


def _tasks(spec, counts: dict) -> int | None:
    if isinstance(spec, dict):
        values = [_tasks(s, counts) for s in spec["min"]]
        return None if None in values else min(values)
    return counts[spec] if isinstance(spec, str) else int(spec)


def _resolve(item: dict, stats: dict, cfg: dict) -> tuple[dict, str | None, str | None]:
    """(item for project_rescoped without calls, fallback note, missing reason).

    `options` ([{stat, condition, note}], in order): the first option that unit_cost can price is used.
    `requires` ([[stat, condition]]): when one is missing, the row is forced missing (an e3_minus_plain estimate
    without its measurement would be a guess)."""
    proj = cfg["projection"]
    prices, judges = cfg["prices_usd_per_mtok"], cfg["judges"]
    model = item.get("model") or judges[item["stat"]]["model"]
    out = {k: item[k] for k in ("id", "group", "provider", "name", "stat", "condition", "like", "calls_per_prefix",
                                "extra_output_tokens", "optional", "note") if k in item}
    out["model"] = model
    note = None
    for i, opt in enumerate(item.get("options") or []):
        unit = projection.unit_cost(stats, opt["stat"], opt["condition"], model=model, prices=prices,
                                    reference=proj["reference"], like=item.get("like"))
        if unit["usd"] is not None or i == len(item["options"]) - 1:
            out["stat"], out["condition"] = opt["stat"], opt["condition"]
            note = opt.get("note") if i else None
            break
    lacking = [f"{s} {c}" for s, c in item.get("requires") or [] if (s, c) not in stats]
    if lacking:
        out["condition"] = MISSING + out["condition"]
        out.pop("extra_output_tokens", None)
        return out, note, f"needs a measurement of {', '.join(lacking)}"
    return out, note, None


def _data_basis(row: dict, item: dict, stats: dict, reference: str) -> str:
    """An estimated row's basis with project_rescoped's "; data: ..." (every source of the names involved) replaced
    by the stats entries the estimate used, each with its source."""
    head, sep, tail = row["basis"].partition("; data: ")
    if not sep:
        return row["basis"]
    cut = tail.find("; e3_minus_plain")
    rest = tail[cut:] if cut >= 0 else ""
    if head.startswith("estimated: measured token profile"):
        keys = [(row["stat"], row["condition"])]
    else:
        own = row["stat"] if (row["stat"], "hidden") in stats else item.get("like")
        keys = [(reference, row["condition"]), (reference, "hidden"), (own, "hidden")]
    used = [f"{n} {c} ({stats[(n, c)]['source']})" for n, c in dict.fromkeys(keys) if (n, c) in stats]
    return head + "; data: " + ", ".join(used) + rest


def scenario(cfg: dict, stats: dict, counts: dict, spent_this_session: float, e2: dict) -> dict:
    """project_rescoped on the items with calls = tasks x per_task for these counts, then the needed cuts: applied in
    order, cumulatively, only while the Anthropic total exceeds remaining - cut_margin_usd. The optional items
    (mitigated best-of-16) fit when the margin after the needed cuts minus their cost is at least
    margin_for_mitigated_bo16_usd. Items with requires_e2 are left out when the Step 5 verdict is negative."""
    proj = cfg["projection"]
    resolved, extras, tasks_now, per_task, not_counted = [], {}, {}, {}, []
    for item in proj["items"]:
        if item.get("requires_e2") and not e2["counted"]:
            not_counted.append(item["name"])
            continue
        out, note, missing = _resolve(item, stats, cfg)
        tasks = _tasks(item["tasks"], counts)
        if tasks is None:
            missing, tasks = "a task count is missing (see the funnel)", 0
            out["condition"] = MISSING + out["condition"].removeprefix(MISSING)
        tasks_now[item["id"]], per_task[item["id"]] = tasks, int(item["per_task"])
        out["calls"] = tasks * per_task[item["id"]]
        resolved.append(out)
        extras[item["id"]] = (item, note, missing)
    cuts = []
    for cut in proj["cuts"]:
        caps = cut.get("cap_tasks") or {}
        for key, cap in caps.items():
            tasks_now[key] = min(int(cap), tasks_now[key])
        cuts.append({"id": cut["id"], "name": cut["name"], "remove": list(cut.get("remove") or []),
                     "set_calls": {k: tasks_now[k] * per_task[k] for k in caps}})
    res = projection.project_rescoped({"prices_usd_per_mtok": cfg["prices_usd_per_mtok"],
                                       "projection": {**proj, "items": resolved, "cuts": cuts}},
                                      stats, spent_this_session)
    for row in res["rows"]:
        item, note, missing = extras[row["id"]]
        row["condition"] = row["condition"].removeprefix(MISSING)
        entry = stats.get((row["stat"], row["condition"]))
        if row["usd"] is None:
            row["basis"] = "missing: " + (missing or item.get("missing_note") or row["basis"])
        elif entry and entry.get("basis") and row["basis"].startswith("measured"):
            row["basis"] = entry["basis"]
        elif row["basis"].startswith("estimated"):
            row["basis"] = _data_basis(row, item, stats, proj["reference"])
        if note and row["usd"] is not None:
            row["basis"] += f"; {note}"
    b = res["anthropic_budget"]
    threshold = b["remaining_usd"] - float(proj["cut_margin_usd"])
    total = b["projected_usd"]
    for cut in res["cuts"]:
        cut["needed"] = total > threshold
        if cut["needed"]:
            total = cut["anthropic_usd"]
        else:
            break
    for cut in res["cuts"]:
        cut.setdefault("needed", False)
    margin = b["remaining_usd"] - total
    fits_optional = None
    if b["optional_items"] and proj.get("margin_for_mitigated_bo16_usd") is not None:
        anth = res["totals"].get("anthropic", {})
        if not anth.get("optional_missing"):
            fits_optional = margin - b["optional_usd"] >= float(proj["margin_for_mitigated_bo16_usd"])
    res["after_cuts"] = {"threshold_usd": threshold, "anthropic_usd": total, "margin_usd": margin,
                         "fits": total <= threshold, "needed": [c["id"] for c in res["cuts"] if c["needed"]],
                         "optional_fits": fits_optional, "margin_after_optional_usd": margin - b["optional_usd"],
                         "unpriced": [r["name"] for r in res["rows"]
                                      if r["provider"] == "anthropic" and not r["optional"] and r["usd"] is None]}
    res["counts"], res["not_counted"] = counts, not_counted
    return res


def e2_verdict(path: Path) -> dict:
    """The Step 5 verdict from e2_summary.json. Experiment 2 is left out only when the test-record labeller's verdict
    is negative and the full-evidence labeller's is negative or absent; a disagreement goes to Uzair and the rows
    stay counted."""
    s = read_json(path)
    v = ((s or {}).get("verdict") or {}).get("full_e2_runs")
    vf = ((s or {}).get("verdict_full") or {}).get("full_e2_runs")
    if s is None or v is None:
        why = "no e2_summary.json yet" if s is None else f"status {s.get('status')}, no verdict recorded"
        return {"available": False, "counted": True, "test_record": None, "full_evidence": vf,
                "text": f"Step 5 verdict not yet available ({why}); the Experiment 2 rows are counted."}
    counted = bool(v) or vf is True
    text = f"Step 5 verdict: the test-record labeller says the full Experiment 2 {'runs' if v else 'does not run'}"
    if vf is not None:
        text += f", the full-evidence labeller says it {'runs' if vf else 'does not run'}"
        if vf != v:
            text += "; the two disagree and the choice goes to Uzair, so the rows stay counted"
    text += "." if counted else "; the Experiment 2 rows and the mitigated best-of-16 are not counted."
    return {"available": True, "counted": counted, "test_record": v, "full_evidence": vf, "text": text}


def spend(cfg: dict, root: Path, cost: list[dict]) -> dict:
    """Per provider: spent before this session (prior_spend.json, else the config fallback, else None) and this
    session (sum of cost_usd in the cost log)."""
    prior = read_json(root / cfg["prior_spend"])
    fallback = cfg.get("spent_before_fallback_usd") or {}
    out = {}
    for prov in ("anthropic", "openrouter", "google"):
        before = ((prior or {}).get("by_provider") or {}).get(prov, {}).get("usd")
        out[prov] = {"before_usd": before if before is not None else fallback.get(prov),
                     "before_source": cfg["prior_spend"] if before is not None else
                     ("config fallback" if fallback.get(prov) is not None else "not known"),
                     "this_session_usd": math.fsum(float(r["cost_usd"]) for r in cost
                                                   if r.get("provider") == prov and r.get("cost_usd") is not None)}
    return out


def prepare(cfg: dict, root: Path) -> tuple[dict, dict, dict, dict]:
    """(cfg with the prices, judges and spend before this session filled in, stats, base counts, spend)."""
    models = yaml.safe_load((root / cfg["models_config"]).read_text(encoding="utf-8"))
    cfg = {**cfg, "prices_usd_per_mtok": models["prices_usd_per_mtok"], "judges": models["judges"]}
    cost = read_jsonl(root / cfg["cost_log"])
    base = base_counts(cfg, root)
    stats = build_stats(cfg, root, cost, base)
    money = spend(cfg, root, cost)
    if money["anthropic"]["before_usd"] is None:
        raise ValueError("Anthropic spend before this session: no prior_spend.json and no config fallback")
    cfg["projection"] = {**cfg["projection"], "anthropic_spent_before_usd": money["anthropic"]["before_usd"]}
    return cfg, stats, base, money


def run(cfg: dict, root: Path) -> dict:
    cfg, stats, base, money = prepare(cfg, root)
    e2 = e2_verdict(root / cfg["e2_summary"])
    scenarios = {name: scenario(cfg, stats, scenario_counts(base, int(t["coding"]), int(t["research"]),
                                                            int(cfg["e2_tasks_per_domain"])),
                                money["anthropic"]["this_session_usd"], e2)
                 for name, t in cfg["scenarios"].items()}
    return {"config": {k: cfg[k] for k in ("cost_log", "prior_spend", "sets", "research_funnel", "e2_summary")},
            "base_counts": base, "spend": money, "e2": e2, "scenarios": scenarios,
            "stats": [{"name": k[0], "condition": k[1], **v} for k, v in sorted(stats.items())]}


# ---------------------------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------------------------

def _usd(x: float | None) -> str:
    """2 decimals from 0.1 USD, 4 below, 6 below 0.001 USD (so a cheap call never prints as 0.0000)."""
    if x is None:
        return "n/a"
    if abs(x) >= 0.1:
        return f"{x:.2f}"
    return f"{x:.4f}" if abs(x) >= 0.001 or x == 0 else f"{x:.6f}"


def _counts_lines(res: dict, base: dict, cfg: dict) -> list[str]:
    c = res["counts"]
    out = [f"- {c['coding'] + c['research']} tasks: {c['coding']} coding and {c['research']} research.",
           f"- Shown-eligible coding tasks: {c['shown_coding']} = ceil({base['coding_eligible']} eligible of the "
           f"{base['coding_sets']} sets in {cfg['sets']['coding']} x {c['coding']} / {base['coding_sets']}).",
           f"- Shown-eligible research tasks: {c['shown_research']} = ceil({base['research_eligible']} eligible of "
           f"the {base['research_sets']} pilot sets x {c['research']} / {base['research_sets']}); the share is "
           f"assumed to hold for the {max(0, c['research'] - base['research_sets'])} sets not yet built.",
           f"- Sets still owed: {c['coding_sets_owed']} coding, {c['research_sets_owed']} research."]
    if c["research_trajectories"] is None:
        out.append("- Trajectories needed: not computed (a count in the research funnel is missing).")
    else:
        out.append(f"- Trajectories needed for the research sets owed: ceil(owed / (gap rate x build yield)) = "
                   f"ceil({c['research_sets_owed']} / ({base['funnel_gap_trajectories']}/"
                   f"{base['funnel_trajectories']} x {base['funnel_built']}/{base['funnel_tried']})) = "
                   f"{c['research_trajectories']}, with gap rate = gap trajectories / trajectories and build yield "
                   "= sets built / gap tasks tried, from the research funnel. This assumes the pilot's gap-type mix.")
    return out


def _rows_table(res: dict) -> list[str]:
    lines = ["| group | item | calls | unit USD (per call, set or trajectory) | USD | running total (USD) | basis |",
             "|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        running = "not added (optional)" if r["optional"] else f"{r['running_total']:.2f} ({r['provider']})"
        name = r["name"] + (" (optional)" if r["optional"] else "")
        basis = r["basis"] + (f". Note: {r['note']}" if r.get("note") else "")
        lines.append(f"| {r['group']} | {name} | {r['calls']:,} | {_usd(r['unit_usd'])} | {_usd(r['usd'])} | "
                     f"{running} | {basis} |")
    return lines


def _budget_lines(res: dict, cfg: dict, spent: dict) -> list[str]:
    b, a = res["anthropic_budget"], res["after_cuts"]
    proj = cfg["projection"]
    lines = [f"- Project total: {b['project_total_usd']:.2f} USD.",
             f"- Spent before this session: {b['spent_before_usd']:.2f} USD ({spent['anthropic']['before_source']}).",
             f"- Spent this session: {b['spent_this_session_usd']:.2f} USD (Anthropic total of {cfg['cost_log']}).",
             f"- Remaining: {b['project_total_usd']:.2f} - {b['spent_before_usd']:.2f} - "
             f"{b['spent_this_session_usd']:.2f} = {b['remaining_usd']:.2f} USD.",
             f"- Projected for the full run: {b['projected_usd']:.2f} USD, not counting "
             f"{b['missing_unit_costs']} Anthropic rows without a unit cost and the optional rows.",
             f"- Margin before cuts: {b['margin_usd']:.2f} USD. Cuts apply while the total exceeds remaining - "
             f"{float(proj['cut_margin_usd']):.0f} = {a['threshold_usd']:.2f} USD."]
    if res["not_counted"]:
        lines.append(f"- Not counted (Step 5 verdict): {', '.join(res['not_counted'])}.")
    lines += ["", "| cut | change (cumulative) | needed | Anthropic total after it (USD) | margin after it (USD) |",
              "|---|---|---|---|---|", f"| none | items as listed | | {b['projected_usd']:.2f} | {b['margin_usd']:.2f} |"]
    lines += [f"| {c['id']} | {c['name']} | {'yes' if c['needed'] else 'no'} | {c['anthropic_usd']:.2f} | "
              f"{c['margin_usd']:.2f} |" for c in res["cuts"]]
    needed = ", ".join(a["needed"]) or "none"
    verdict = "fits" if a["fits"] else "does not fit even after every cut"
    priced = " on the priced rows" if a["unpriced"] else ""
    after = "After them the Anthropic total is" if a["needed"] else "The Anthropic total stays at"
    lines += ["", f"Cuts needed: {needed}. {after} {a['anthropic_usd']:.2f} USD and the margin is "
              f"{a['margin_usd']:.2f} USD, so the run {verdict}{priced} within remaining - "
              f"{float(proj['cut_margin_usd']):.0f} USD."]
    if a["unpriced"]:
        lines.append(f"These Anthropic figures leave out {len(a['unpriced'])} rows without a unit cost "
                     f"({'; '.join(a['unpriced'])}), so the cuts needed and the fit of the optional rows can change "
                     "when they are measured.")
    if b["optional_items"]:
        required = float(proj["margin_for_mitigated_bo16_usd"])
        missing_opt = res["totals"].get("anthropic", {}).get("optional_missing", 0)
        if a["optional_fits"] is None:
            verdict_opt = "cannot be decided (a unit cost is missing)"
        else:
            verdict_opt = (f"{'fits' if a['optional_fits'] else 'does not fit'}{priced}: the margin after the needed "
                           f"cuts minus its cost is {a['margin_usd']:.2f} - {b['optional_usd']:.2f} = "
                           f"{a['margin_after_optional_usd']:.2f} USD")
        lines.append(f"The optional {', '.join(b['optional_items'])} would cost {b['optional_usd']:.2f} USD "
                     f"(rows without a unit cost: {missing_opt}). It {verdict_opt}, against the {required:.0f} USD of "
                     "margin that must stay.")
    return lines


def _provider_lines(res: dict, cfg: dict, spent: dict) -> list[str]:
    t = res["totals"]
    o = t.get("openrouter", {"usd": 0.0, "missing": 0})
    so = spent["openrouter"]
    cap = (cfg["projection"].get("targets_usd") or {}).get("openrouter")
    lines = []
    if so["before_usd"] is None or cap is None:
        before = "not known" if so["before_usd"] is None else f"{so['before_usd']:.4f} USD"
        lines.append(f"- OpenRouter: spent before this session {before}; this session {so['this_session_usd']:.4f} "
                     f"USD; projected {o['usd']:.2f} USD (rows without a unit cost: {o['missing']}); project cap "
                     f"{'not configured' if cap is None else f'{cap} USD'}.")
    else:
        cum = so["before_usd"] + so["this_session_usd"]
        lines.append(f"- OpenRouter: {so['before_usd']:.4f} USD before this session ({so['before_source']}) + "
                     f"{so['this_session_usd']:.4f} USD this session = {cum:.4f} USD so far; projected "
                     f"{o['usd']:.2f} USD (rows without a unit cost: {o['missing']}); {cum + o['usd']:.2f} USD after "
                     f"the full run against the {cap} USD project cap, a margin of {cap - cum - o['usd']:.2f} USD.")
    g = t.get("google", {"usd": 0.0, "usd_after_price_changes": 0.0, "missing": 0, "optional_usd": 0.0,
                         "optional_missing": 0})
    changes = "; ".join(f"{c['model']} x{float(c['factor']):g} from {c['from']}" for c in res["price_changes"])
    lines.append(f"- Google, aistudio rows: {g['usd']:.2f} USD at current prices, {g['usd_after_price_changes']:.2f} "
                 f"USD with {changes or 'no price change'} (rows without a unit cost: {g['missing']}). AI Studio "
                 f"balance: not known; the run needs {g['usd']:.2f} USD of it, or "
                 f"{g['usd_after_price_changes']:.2f} USD if it runs after the price change.")
    lines.append(f"- Google, vertex rows (optional, only with the credit confirmed): {g['optional_usd']:.2f} USD "
                 f"(rows without a unit cost: {g['optional_missing']}). Cloud credit: not known; the rows need "
                 f"{g['optional_usd']:.2f} USD of it.")
    return lines


def _ratio_source(cfg: dict) -> str:
    return ", ".join(s["source"] for s in cfg["record_sources"] if s.get("ratio_basis")) or "none configured"


def markdown(result: dict, cfg: dict) -> str:
    spent, base = result["spend"], result["base_counts"]
    names = list(result["scenarios"])
    lines = ["# Cost projection for the full run (session 5)", "",
             "Arithmetic on recorded costs (overclaim.projection_v4 on overclaim.projection); nothing was run. Every "
             "unit cost is a mean over live calls in a run file or an estimate by projection.unit_cost, and its "
             "basis is in the last column. A row without a measurement has no unit cost, is listed as missing and is "
             "left out of every total.", "",
             "Condition labels: @coding and @research mark the domain; full/, pointwise/ and e2_score/ mark the full "
             "rubric, the pointwise reading and the Experiment 2 selector; no prefix is the pairwise judge on the "
             "neutral rubric. The bare labels hidden and shown are the neutral pairwise calls of the ratio-basis "
             f"source ({_ratio_source(cfg)}), where every judge saw the same prompts. An estimate takes "
             f"{cfg['projection']['reference']}'s profile under the same label, scaled by the two graders' tokenizer "
             "ratio on the bare hidden calls, with the grader's own output tokens from them.", "",
             f"{result['e2']['text']}", ""]
    if base["funnel_by_gap"]:
        per_gap = "; ".join(f"{g}: {v['trajectories']} of {base['funnel_trajectories']} trajectories, {v['built']} "
                            f"built of {v['tried']} tried, list exhausted {v['list_exhausted']}"
                            for g, v in base["funnel_by_gap"].items())
        lines += [f"Research funnel ({cfg['research_funnel']}, batch {cfg['research_funnel_batch']}): {per_gap}. "
                  f"The pool holds {base['pool_size']} clusters and {base['trajectories']} trajectories were run.", ""]
    for i, name in enumerate(names):
        res = result["scenarios"][name]
        lines += [f"## Scenario {name}", "", *_counts_lines(res, base, cfg), ""]
        if i == 0:
            lines += [*_rows_table(res), ""]
        else:
            lines += [f"The rows of this scenario are in the JSON file (scenarios.{name}.rows).", ""]
        missing = [f"{r['name']} ({r['basis'].removeprefix('missing: ')})" for r in res["rows"] if r["usd"] is None]
        lines += [f"Rows still missing in this scenario ({len(missing)}): " + ("none." if not missing else ""),
                  *[f"- {m}" for m in missing], ""]
        lines += ["Anthropic budget:", "", *_budget_lines(res, cfg, spent), "", "OpenRouter and Google:", "",
                  *_provider_lines(res, cfg, spent), ""]
    lines += [f"The cut order is a budget rule of the briefs (the session-3 brief of 2026-11-05, revised in Step 10 of "
              "the session-4 brief of 2026-11-06); it is not pre-registered. Cuts 1 to 3 are already applied in the "
              "items."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Part 2 of session 5 (revision of 2026-11-06): two tiers, pointwise primary, gates, cuts and add-backs
# ---------------------------------------------------------------------------------------------
#
# Selected in main() by cfg["plan"] == "part2" (configs/projection_part2.yaml). Every row is priced by the first of
# its `price` methods that can price it: measured (a stats entry, its source named) or derived ((a) the reference
# judge's measured profile scaled by projection.unit_cost's tokenizer ratio, or a measured cost ratio of two judges;
# (b) a token profile priced at list prices; (c) another measured call's unit cost). A row that none can price is
# "pending" when a file that could measure what is missing is not yet written, else "unpriced"; once both gates are
# known (Part 1 is over) every row without a USD figure is unpriced, and an unpriced row makes main() return 1. Gates
# not known yet give one plan per branch.

PART2 = "part2"                     # cfg["plan"] value that selects the Part 2 projection in main()
CONDITIONS = ("hidden", "shown")    # graders.judge.CONDITIONS (not imported: that module loads the API clients)
PROVIDERS = ("anthropic", "openrouter", "google")
TIER_TITLES = {1: "Tier 1, pointwise (the primary result)", 2: "Tier 2, pairwise (the appendix)",
               "addback": "Add-backs (added only by the add-back rule)"}
TIER2_CALLS = "tier2_calls"         # tasks of a reason-classification row: one call per counted Tier 2 call


def purposes_part2() -> dict:
    """purposes() plus session 4's four-field reason classifier and the research_v2 set-building purposes (detection
    and pre-check keep research_v1's purposes, the rewrite and the post-check have their own)."""
    from overclaim import analysis_reasons_v4
    from overclaim.domains.research import sets, sets_v2

    return {**purposes(), "reasons_v4": analysis_reasons_v4.PURPOSE_V4,
            "research_sets_v2": [sets.PURPOSE_DETECTION, sets.PURPOSE_PRECHECK, sets_v2.PURPOSE_REWRITE_V2,
                                 sets_v2.PURPOSE_VERIFICATION_V2]}


def _live(r: dict) -> bool:
    """A record that projection.unit_stats counts: not a disk-cache hit, not smoke-only, with a cost."""
    return not r.get("cached") and not r.get("smoke_only") and r.get("cost_usd") is not None


def stats_part2(cfg: dict, root: Path) -> tuple[dict, dict, dict]:
    """(stats, records, providers) for the Part 2 projection.

    stats = {(name, label): {"n", "usd", "tokens", "source"[, "basis"]}}, live calls only; the first source with live
    calls for a key wins. Sources in order: the record files of cfg["record_sources"] (a source's optional
    `conditions` keeps only those conditions, so the research pilot's shown calls, made on shorter pages, never price
    a shown row; the ratio_basis source also gives the bare labels that unit_cost's tokenizer ratio reads), then
    cfg["purpose_stats"] read from each log of cfg["purpose_logs"] in order (name = the record's model when
    `by_model`), then the research_v2 set-building cost per built set (cfg["research_set_unit"]).
    records = the live records behind each record-file key. providers = {label: [paths that could measure it]}."""
    thought = tuple(cfg.get("thought_separate") or ())
    stats: dict = {}
    records: dict = {}
    providers: dict = {}
    for src in cfg["record_sources"]:
        kinds, sets, conds = src["rubrics"], src.get("sets"), src.get("conditions")
        for kind in kinds.values():
            for cond in conds or CONDITIONS:
                providers.setdefault(label(cond, src["domain"], kind), []).append(src["path"])
        recs = [r for r in read_jsonl(root / src["path"]) if r.get("rubric") in kinds
                and (sets is None or r.get("set") in sets) and (conds is None or r.get("condition") in conds)]
        relabelled = [{**r, "condition": label(r["condition"], src["domain"], kinds[r["rubric"]])} for r in recs]
        for key, value in projection.unit_stats(relabelled, thought_separate=thought).items():
            if key not in stats:
                stats[key] = {**value, "source": src["source"]}
                records[key] = [r for r in relabelled if _live(r) and (r["grader"], r["condition"]) == key]
        if src.get("ratio_basis"):
            neutral = [r for r in recs if kinds[r["rubric"]] == ""]
            _add(stats, projection.unit_stats(neutral, thought_separate=thought), f"{src['source']}, ratio basis")
    p = purposes_part2()
    for log in cfg["purpose_logs"]:
        cost = read_jsonl(root / log["path"])
        for spec in cfg["purpose_stats"]:
            purpose = p[spec["purpose"]]
            providers.setdefault(spec["label"], []).append(log["path"])
            pseudo = [{"grader": r.get("model") if spec.get("by_model") else spec["name"], "condition": spec["label"],
                       "cost_usd": r.get("cost_usd"), "usage": r} for r in cost if r.get("purpose") == purpose]
            _add(stats, projection.unit_stats(pseudo), f"{log['path']} by purpose {purpose}")
    unit = cfg["research_set_unit"]
    names = p[unit["purpose"]]
    providers.setdefault(unit["label"], []).extend([unit["log"], unit["sets"]])
    built = len(read_jsonl(root / unit["sets"]))
    total = _purpose_usd(read_jsonl(root / unit["log"]), names)
    if built and total > 0:
        stats[(unit["name"], unit["label"])] = {
            "n": built, "usd": total / built, "tokens": {b: 0.0 for b in projection.BUCKETS}, "source": unit["log"],
            "basis": (f"measured: {total:.4f} USD of purposes {', '.join(names)} in {unit['log']} / {built} sets "
                      f"in {unit['sets']}")}
    return stats, records, providers


def _pending_files(labels: list[str], ctx: dict) -> list[str]:
    """The files that could measure one of these labels and are not yet written."""
    paths = dict.fromkeys(p for lab in labels for p in ctx["providers"].get(lab, []))
    return [p for p in paths if not (ctx["root"] / p).exists()]


def _side_records(spec: dict, root: Path) -> list[dict]:
    """Live records of one side of a file factor: {path, grader, condition[, rubric][, set]}."""
    keys = [k for k in ("grader", "condition", "rubric", "set") if k in spec]
    return [r for r in read_jsonl(root / spec["path"]) if _live(r) and all(r.get(k) == spec[k] for k in keys)]


def _input_tokens(r: dict) -> float:
    """Input tokens of one call: uncached, cache write and cache read together."""
    u = r.get("usage") or {}
    return float(sum(u.get(b) or 0 for b in projection.BUCKETS[:3]))


def factor_part2(spec: dict, root: Path, stats: dict, providers: dict) -> dict:
    """One measured factor of cfg["factors"]: {"value" (None when it cannot be measured), "text", "missing",
    "pending"}.

    cost_ratio: mean cost per live call of `num` / of `den`, both taken on the tasks that have live calls on both
    sides. input_tokens_diff: the same tasks, mean input tokens per call (all three input buckets), num minus den.
    input_tokens_ratio: mean input tokens per call of two stats entries (num, den: [name, label]), num / den.
    constant: spec["value"] as given in the config, with spec["text"] saying where it comes from."""
    kind = spec["kind"]
    if kind == "constant":
        value = float(spec["value"])
        return {"value": value, "missing": [], "pending": [], "text": f"{spec['text']}: {value:g} (a constant)"}
    if kind == "input_tokens_ratio":
        num, den = stats.get(tuple(spec["num"])), stats.get(tuple(spec["den"]))
        lacking = [f"{k[0]} {k[1]}" for k, s in ((spec["num"], num), (spec["den"], den)) if s is None]
        if lacking:
            pending = _pending_files([spec["num"][1], spec["den"][1]], {"providers": providers, "root": root})
            return {"value": None, "text": spec["text"], "missing": lacking, "pending": pending}
        a, b = (sum(s["tokens"][k] for k in projection.BUCKETS[:3]) for s in (num, den))
        return {"value": a / b, "missing": [], "pending": [],
                "text": (f"{spec['text']} = {a / b:.3f} ({a:,.0f} / {b:,.0f} input tokens per call; "
                         f"{num['source']}, {num['n']} calls; {den['source']}, {den['n']} calls)")}
    if kind not in ("cost_ratio", "input_tokens_diff"):
        raise ValueError(f"factor kind {kind!r} unknown")
    sides = {s: _side_records(spec[s], root) for s in ("num", "den")}
    pending = list(dict.fromkeys(spec[s]["path"] for s in sides if not (root / spec[s]["path"]).exists()))
    tasks = {r["task_id"] for r in sides["num"]} & {r["task_id"] for r in sides["den"]}
    if not tasks:
        return {"value": None, "text": spec["text"], "pending": pending,
                "missing": [f"{spec['text']} (no task with live calls on both sides)"]}
    shared = {s: [r for r in rs if r["task_id"] in tasks] for s, rs in sides.items()}
    measure = (lambda r: float(r["cost_usd"])) if kind == "cost_ratio" else _input_tokens
    means = {s: math.fsum(map(measure, rs)) / len(rs) for s, rs in shared.items()}
    value = means["num"] / means["den"] if kind == "cost_ratio" else means["num"] - means["den"]
    where = " and ".join(dict.fromkeys(spec[s]["path"] for s in ("num", "den")))
    shown = f"{value:.3f}" if kind == "cost_ratio" else f"{value:+.1f} tokens"
    return {"value": value, "missing": [], "pending": [],
            "text": (f"{spec['text']} = {shown} ({where}; {len(tasks)} tasks, {len(shared['num'])} and "
                     f"{len(shared['den'])} live calls)")}


def _priced(usd: float, basis: str, text: str) -> dict:
    return {"usd": usd, "basis": basis, "text": text, "missing": [], "pending": []}


def _unpriced(missing: list[str], pending: list[str]) -> dict:
    return {"usd": None, "basis": None, "text": "", "missing": missing, "pending": pending}


def _stat(key: list, ctx: dict) -> tuple[dict | None, list[str], list[str]]:
    """(stats entry or None, what is missing, files not yet written that could measure it)."""
    s = ctx["stats"].get(tuple(key))
    return (s, [], []) if s is not None else (None, [f"{key[0]} {key[1]}"], _pending_files([key[1]], ctx))


def _measured_text(s: dict) -> str:
    return s.get("basis") or f"measured in {s['source']} ({s['n']} live calls)"


def _row_model(row: dict, ctx: dict) -> str:
    return row.get("model") or ctx["judges"][row["judge"]]["model"]


def _band_price(m: dict, row: dict, ctx: dict) -> dict:
    """Add-back (i): the measured shown pointwise calls' cached-prefix USD per rendered transcript token (their own
    write and read profile; the rendered tokens of each call's task from the main sets file), applied to the band
    tasks' mean rendered tokens, plus the calls' mean USD of uncached input and output. Unit = USD per call."""
    key = tuple(m["band_tokens"])
    s, missing, pending = _stat(key, ctx)
    base = ctx["base"]
    if base["coding_tokens"] is None:
        return _unpriced(missing + ["the rendered tokens of the coding tasks"], pending + [base["files"]["coding"]])
    if s is None:
        return _unpriced(missing, pending)
    calls = [r for r in ctx["records"].get(key, []) if r["task_id"] in base["coding_tokens"]]
    if not calls:
        return _unpriced([f"{key[0]} {key[1]} on a task with transcript_tokens in {base['files']['coding']}"], [])
    model = _row_model(row, ctx)
    p = ctx["prices"][model]
    u = [{b: (r.get("usage") or {}).get(b) or 0 for b in projection.BUCKETS} for r in calls]
    prefix = math.fsum(x["cache_creation_input_tokens"] * p.get("cache_write", p["input"])
                       + x["cache_read_input_tokens"] * p.get("cache_read", p["input"]) for x in u) / 1e6
    rest = math.fsum(x["input_tokens"] * p["input"] + x["output_tokens"] * p["output"] for x in u) / 1e6 / len(u)
    rate = prefix / sum(base["coding_tokens"][r["task_id"]] for r in calls)
    band = base["band_tokens"]
    if not band:
        return _priced(0.0, "derived", "no coding task in the band")
    mean_band = sum(band) / len(band)
    return _priced(rate * mean_band + rest, "derived",
                   f"(b) {key[0]} {key[1]} ({_measured_text(s)}; {len(calls)} of them on tasks with known rendered "
                   f"tokens): {rate * 1e6:.2f} USD of cache write and read per million rendered transcript tokens, "
                   f"applied to the band's mean of {mean_band:,.0f} rendered tokens, plus {rest:.4f} USD per call of "
                   f"uncached input and output; {model} list prices")


def _profile_price(m: dict, row: dict, ctx: dict) -> dict:
    """profile_of [name, label] (b): the stats entry's mean token profile per call, or, when that entry is absent, the
    profile projection.unit_cost estimates for `name` (the reference judge's profile under `label` with its three
    input buckets x the tokenizer ratio = name's / the reference's mean input tokens per bare hidden call, and name's
    own output tokens per bare hidden call; unit_cost returns only a price, so the same arithmetic is done here). With
    `no_cache` the cache-write and cache-read tokens are moved into uncached input first (OpenRouter has no prompt
    cache). Every bucket is priced at the row model's list prices. `ratio_label` (default "hidden", the bare hidden
    pairwise label) names the stats label whose input tokens give the tokenizer ratio and whose output tokens give
    name's output, for a judge measured only in another kind of call (README amendment 3, item 9: the open-weight
    mitigated selector re-priced from the judge's measured hidden pointwise calls, label pointwise/hidden@coding)."""
    name, lab = m["profile_of"]
    stats, ref = ctx["stats"], ctx["reference"]
    ratio_label = m.get("ratio_label", "hidden")
    s = stats.get((name, lab))
    if s is not None:
        tokens, what = dict(s["tokens"]), f"{name} {lab} token profile ({_measured_text(s)})"
    else:
        ref_s, ref_h, own = stats.get((ref, lab)), stats.get((ref, ratio_label)), stats.get((name, ratio_label))
        lacking = [f"{k[0]} {k[1]}" for k, v in (((ref, lab), ref_s), ((ref, ratio_label), ref_h),
                                                  ((name, ratio_label), own)) if v is None]
        if lacking:
            return _unpriced([f"{name} {lab} (or, for an estimate, {', '.join(lacking)})"],
                             _pending_files([lab], ctx))
        ratio = own["tokens"]["input_tokens"] / ref_h["tokens"]["input_tokens"]
        tokens = {b: ref_s["tokens"][b] * ratio for b in projection.BUCKETS[:3]}
        tokens["output_tokens"] = own["tokens"]["output_tokens"]
        what = (f"{name} {lab} token profile as projection.unit_cost estimates it: {ref} {lab} "
                f"({_measured_text(ref_s)}) x tokenizer ratio {ratio:.2f} ({ratio_label} input tokens, {ref} "
                f"{ratio_label} ({ref_h['source']}) and {name} {ratio_label} ({own['source']}, {own['n']} live calls)), "
                f"{name} output tokens per {ratio_label} call")
    how = ""
    if m.get("no_cache"):
        moved = tokens["cache_creation_input_tokens"] + tokens["cache_read_input_tokens"]
        tokens.update(input_tokens=tokens["input_tokens"] + moved, cache_creation_input_tokens=0.0,
                      cache_read_input_tokens=0.0)
        how = (f", its {moved:,.0f} cache-write and cache-read tokens per call counted as uncached input (no prompt "
               "cache)" if moved else ", no cache tokens to move (no prompt cache)")
    model = _row_model(row, ctx)
    n_in = sum(tokens[b] for b in projection.BUCKETS[:3])
    return _priced(projection._price(ctx["prices"], model, tokens), "derived",
                   f"(b) {what}{how}: {n_in:,.0f} input and {tokens['output_tokens']:,.0f} output tokens per call at "
                   f"{model} list prices")


def price_method(m: dict, row: dict, ctx: dict) -> dict:
    """One method of a row's `price` list: {"usd" (None when it cannot price the row), "basis" (measured |
    derived), "text", "missing", "pending"}.

    measured [name, label]: the stats entry (basis derived with an (c) note when `derived` gives one: another
    call's measured unit cost); ratio [name, label]: unit_cost's estimate from the reference judge's profile (a);
    times {method}, factor: that method's unit x a measured factor; extra_input / input_scaled [name, label],
    factor: the entry's measured unit cost x the list-price cost (row model) of its token profile with extra
    uncached input tokens, or with its input buckets scaled, over that of the profile as measured (b; so a booked
    OpenRouter price stays the base); reamortize [name, label], calls_per_prefix: the profile with one cache write
    per that many calls, at list prices (b); band_tokens [name, label]: add-back (i); zero: a fixed 0 USD with the
    given text; profile_of [name, label], optional no_cache: a token profile at the row model's list prices (b,
    _profile_price)."""
    if "zero" in m:
        return _priced(0.0, "derived", m["zero"])
    if "measured" in m:
        s, missing, pending = _stat(m["measured"], ctx)
        if s is None:
            return _unpriced(missing, pending)
        if m.get("derived"):
            return _priced(s["usd"], "derived", f"(c) {m['derived']}: {m['measured'][0]} {m['measured'][1]} "
                                                f"{_measured_text(s)}")
        return _priced(s["usd"], "measured", _measured_text(s))
    if "ratio" in m:
        name, lab = m["ratio"]
        if (name, lab) in ctx["stats"]:
            return price_method({"measured": m["ratio"]}, row, ctx)
        ref = ctx["reference"]
        unit = projection.unit_cost(ctx["stats"], name, lab, model=_row_model(row, ctx), prices=ctx["prices"],
                                    reference=ref)
        if unit["usd"] is None:
            lacking = f"{ref} {lab}" if (ref, lab) not in ctx["stats"] else f"{name} hidden (tokenizer ratio)"
            return _unpriced([lacking], _pending_files([lab], ctx))
        s_ref, s_hid = ctx["stats"][(ref, lab)], ctx["stats"][(ref, "hidden")]
        return _priced(unit["usd"], "derived",
                       f"(a) {unit['basis'].removeprefix('estimated: ')}; data: {ref} {lab} ({_measured_text(s_ref)}), "
                       f"{ref} and {name} hidden ({s_hid['source']})")
    if "times" in m:
        base, f = price_method(m["times"], row, ctx), ctx["factors"][m["factor"]]
        if base["usd"] is None or f["value"] is None:
            return _unpriced(base["missing"] + f["missing"], base["pending"] + f["pending"])
        return _priced(base["usd"] * f["value"], "derived", f"{base['text']}; x {f['text']}")
    if "extra_input" in m or "input_scaled" in m:
        key = m.get("extra_input") or m["input_scaled"]
        (s, missing, pending), f = _stat(key, ctx), ctx["factors"][m["factor"]]
        if s is None or f["value"] is None:
            return _unpriced(missing + f["missing"], pending + f["pending"])
        tokens = dict(s["tokens"])
        if "extra_input" in m:
            tokens["input_tokens"] += f["value"]
            how = f"plus {f['value']:+.0f} uncached input tokens per call"
        else:
            tokens.update({b: tokens[b] * f["value"] for b in projection.BUCKETS[:3]})
            how = f"input tokens x {f['value']:.3f}"
        model = _row_model(row, ctx)
        listed = projection._price(ctx["prices"], model, s["tokens"])
        scale = projection._price(ctx["prices"], model, tokens) / listed if listed else 1.0
        return _priced(s["usd"] * scale, "derived",
                       f"(b) {key[0]} {key[1]} ({_measured_text(s)}, {s['usd']:.6f} USD per call) x {scale:.3f}, the "
                       f"{model} list-price cost of its token profile {how} over that of the profile as measured "
                       f"({f['text']})")
    if "reamortize" in m:
        key = m["reamortize"]
        s, missing, pending = _stat(key, ctx)
        if s is None:
            return _unpriced(missing, pending)
        model, n = _row_model(row, ctx), int(m["calls_per_prefix"])
        unit = projection.unit_cost(ctx["stats"], key[0], key[1], model=model, prices=ctx["prices"],
                                    reference=ctx["reference"], calls_per_prefix=n)
        return _priced(unit["usd"], "derived", f"(b) {key[0]} {key[1]} token profile ({_measured_text(s)}) with one "
                                               f"cache write per {n} calls, {model} list prices")
    if "band_tokens" in m:
        return _band_price(m, row, ctx)
    if "profile_of" in m:
        return _profile_price(m, row, ctx)
    raise ValueError(f"row {row['id']}: unknown price method {sorted(m)}")


def price_row(row: dict, ctx: dict) -> dict:
    """The row's unit price from the first of its `price` methods that prices it. Without one: usd None and basis
    "pending" when a file that could measure what is missing is not yet written, else "unpriced"."""
    missing: list[str] = []
    pending: list[str] = []
    for m in row["price"]:
        out = price_method(m, row, ctx)
        if out["usd"] is not None:
            return out
        missing += out["missing"]
        pending += out["pending"]
    missing, pending = list(dict.fromkeys(missing)), list(dict.fromkeys(pending))
    text = "needs " + "; ".join(missing) + (f" (not yet written: {', '.join(pending)})" if pending else "")
    return {"usd": None, "basis": "pending" if pending else "unpriced", "text": text, "missing": missing,
            "pending": pending}


def base_part2(cfg: dict, root: Path) -> dict:
    """Counts read from the set files (None while a file is not yet written).

    Coding (cfg["sets"]["coding_main"]): sets, shown-eligible (`shown_eligible` true), the band tasks (transcript_tokens
    in (shown max, band upper], cfg["shown"]) with their rendered tokens, every set's rendered tokens and the number
    of sets without transcript_tokens. Research (cfg["sets"]["research_v2"]): sets built and shown-eligible ones."""
    files = {"coding": cfg["sets"]["coding_main"], "research": cfg["sets"]["research_v2"]}
    lo, hi = int(cfg["shown"]["max_transcript_tokens"]), int(cfg["shown"]["upper_band_tokens"])
    out: dict = {"files": files, "pending_files": [p for p in files.values() if not (root / p).exists()],
                 "coding": None, "shown_coding": None, "band": None, "band_tokens": None, "coding_tokens": None,
                 "coding_without_tokens": None, "research_built": None, "research_eligible": None}
    if (root / files["coding"]).exists():
        sets = read_jsonl(root / files["coding"])
        tokens = {s["task_id"]: s.get("transcript_tokens") for s in sets}
        band = [t for t in tokens.values() if t is not None and lo < t <= hi]
        out.update(coding=len(sets), shown_coding=sum(1 for s in sets if s.get("shown_eligible") is True),
                   band=len(band), band_tokens=band, coding_tokens={k: v for k, v in tokens.items() if v is not None},
                   coding_without_tokens=sum(1 for v in tokens.values() if v is None))
        # The Experiment 2 tasks of the coding domain as the pilot picks them (prep5_e2.pick_tasks: seeded order
        # within gap type, one per gap type in turn), and how many of them are shown-eligible: the tasks of the
        # mitigated best-of-16. Only with the keys rules.e2_tasks_per_domain and seed in the config.
        if cfg.get("seed") is not None and (cfg.get("rules") or {}).get("e2_tasks_per_domain") is not None:
            from overclaim.experiments.prep5_e2 import pick_tasks

            eligible = {s["task_id"] for s in sets if s.get("shown_eligible") is True}
            picked = pick_tasks(sets, int(cfg["rules"]["e2_tasks_per_domain"]), cfg["seed"])
            out["e2_shown_coding"] = sum(1 for t in picked if t in eligible)
        # The E7 tasks (addendum section 2, the freeze stage's rule): the shown-eligible coding sets picked as
        # Experiment 2 picks (seeded order within gap type, one per gap type in turn) until rules.e7_tasks, so a gap
        # type with fewer eligible sets gives its turns to the others. Only with the keys rules.e7_tasks and seed.
        if cfg.get("seed") is not None and (cfg.get("rules") or {}).get("e7_tasks") is not None:
            from overclaim.experiments.prep5_e2 import GAP_ORDER, pick_tasks

            shown = [s for s in sets if s.get("shown_eligible") is True]
            picked = set(pick_tasks(shown, int(cfg["rules"]["e7_tasks"]), cfg["seed"]))
            out["e7_available"] = len(picked)
            out["e7_per_gap"] = {g: sum(1 for s in shown if s["gap_type"] == g and s["task_id"] in picked)
                                 for g in GAP_ORDER}
            out["e7_eligible_per_gap"] = {g: sum(1 for s in shown if s["gap_type"] == g) for g in GAP_ORDER}
    if (root / files["research"]).exists():
        sets = read_jsonl(root / files["research"])
        out.update(research_built=len(sets), research_eligible=sum(1 for s in sets if s.get("shown_eligible") is True))
    return out


def split_tasks(total: int, a: int | None, b: int | None) -> tuple[int | None, int | None]:
    """`total` tasks over two domains in proportion to their counts a and b: the first gets total x a / (a + b),
    rounded half up, each at most its own count, and the rest goes to the other while it has room. (None, None)
    while a count is not known."""
    if a is None or b is None:
        return None, None
    if a + b == 0:
        return 0, 0
    x = min(a, math.floor(Fraction(total * a, a + b) + Fraction(1, 2)))
    y = min(b, total - x)
    return min(a, total - y), y


def _cap(x: int | None, cap: int | None) -> int | None:
    return x if x is None or cap is None else min(x, cap)


def counts_part2(base: dict, rules: dict, research_in: bool, e2_in: bool, params: dict) -> dict:
    """Task counts of one plan (None while a set file is not yet written).

    Research: rules["research_tasks"] when the research gate passed (sets built + sets owed), all shown-eligible when
    every built set is, else ceil(eligible x tasks / built); 0 when the gate failed. Pointwise shown rows: at most
    params["shown_cap_per_domain"] per domain (cut B4). Opus 5.5 shown: params["opus_shown_tasks"] split over the
    two domains' pointwise shown counts (split_tasks); Opus 5.5 pairwise shown: rules["opus_shown_tasks"] over the
    shown-eligible counts. Experiment 3: every shown-eligible coding task up to rules["e3_tasks"], the rest research
    (the pointwise rows at most the cap per domain), Opus 5.5 on rules["opus_e3_tasks"] of those, split the same way.
    Experiment 2: rules["e2_tasks_per_domain"] per domain present when its gate passed; the mitigated best-of-16 on
    the shown-eligible tasks among the coding domain's Experiment 2 tasks (base["e2_shown_coding"], counted by
    base_part2 with the pilot's pick rule; without that key at most the shown-eligible count, an upper bound, as for
    research). Add-back (ii): rules["addback_ii_tasks"] split like Opus 5.5 (only with that key). E7 (only with the
    key rules["e7_tasks"]): params["e7_tasks"] (cut C2) or else rules["e7_tasks"], at most base["e7_available"]."""
    c, sc = base["coding"], base["shown_coding"]
    if research_in:
        r, built, elig = int(rules["research_tasks"]), base["research_built"], base["research_eligible"]
        if built == 0:
            raise ValueError("no research_v2 set built: the shown-eligible research count cannot be scaled")
        sr = None if built is None else (r if elig == built else _ceil_div(elig * r, built))
        owed = None if built is None else max(0, r - built)
    else:
        r = sr = owed = 0
    cap = params.get("shown_cap_per_domain")
    cap = None if cap is None else int(cap)
    e3_total = int(rules["e3_tasks"])

    def e3(cap_: int | None) -> tuple[int | None, int | None]:
        x = None if sc is None else _cap(min(sc, e3_total), cap_)
        return x, (None if sr is None or x is None else _cap(min(sr, e3_total - x), cap_))

    e2c = int(rules["e2_tasks_per_domain"]) if e2_in else 0
    e2r = e2c if research_in else 0
    out = {"coding": c, "research": r, "shown_coding": sc, "shown_research": sr, "pw_shown_coding": _cap(sc, cap),
           "pw_shown_research": _cap(sr, cap), "e2_coding": e2c, "e2_research": e2r,
           "bo16_coding": (None if sc is None else min(e2c, sc) if base.get("e2_shown_coding") is None
                           else min(e2c, base["e2_shown_coding"])),
           "bo16_research": None if sr is None else min(e2r, sr),
           "band_coding": base["band"], "research_sets_owed": owed}
    out["e3_coding"], out["e3_research"] = e3(cap)
    out["e3pair_coding"], out["e3pair_research"] = e3(None)
    splits = [("o55_shown", params["opus_shown_tasks"], out["pw_shown_coding"], out["pw_shown_research"]),
              ("o55_pair", rules["opus_shown_tasks"], sc, sr),
              ("o55_e3", rules["opus_e3_tasks"], out["e3_coding"], out["e3_research"])]
    if "addback_ii_tasks" in rules:
        splits.append(("addback_ii", rules["addback_ii_tasks"], out["pw_shown_coding"], out["pw_shown_research"]))
    for name, total, a, b in splits:
        out[f"{name}_coding"], out[f"{name}_research"] = split_tasks(int(total), a, b)
    if "e7_tasks" in rules:
        n = int(params.get("e7_tasks") or rules["e7_tasks"])
        out["e7_coding"] = None if base.get("e7_available") is None else min(n, base["e7_available"])
    return out


def plan_state(plan: dict, removed: set, params: dict, added: set) -> dict:
    """Tasks, calls and USD of every row of one plan in one state of the cuts and add-backs, and the totals.

    A row counts when it is a Tier 1 or Tier 2 row that no cut removed (a second-sample row leaves with its base
    row) or an added add-back row; an optional row never counts, nor does an optional_openrouter row (run_part2 adds
    it to the OpenRouter total only when it fits). A reason-classification row (tasks "tier2_calls")
    has one call per call of the counted Tier 2 rows of its domain. Totals are over counted rows with a USD figure;
    the counted rows without one are listed in `left_out`."""
    counts = counts_part2(plan["base"], plan["rules"], plan["research"], plan["e2"], params)
    rows: dict = {}
    for row in plan["rows"]:
        counted = (row["id"] in added if row["tier"] == "addback"
                   else not removed & {row["id"], row.get("base_id")}) and not row.get("optional") \
            and not row.get("optional_openrouter")
        if row["tasks"] == TIER2_CALLS:
            rows[row["id"]] = {"counted": counted}
            continue
        tasks = counts[row["tasks"]] if isinstance(row["tasks"], str) else int(row["tasks"])
        rows[row["id"]] = {"counted": counted, "tasks": tasks,
                           "calls": None if tasks is None else tasks * int(row["per_task"])}
    for row in plan["rows"]:
        if row["tasks"] == TIER2_CALLS:
            calls = [rows[r["id"]]["calls"] for r in plan["rows"] if r["tier"] == 2 and r["tasks"] != TIER2_CALLS
                     and r.get("domain") == row["domain"] and rows[r["id"]]["counted"]]
            tasks = None if None in calls else sum(calls)
            rows[row["id"]].update(tasks=tasks, calls=None if tasks is None else tasks * int(row["per_task"]))
    for row in plan["rows"]:
        o, unit = rows[row["id"]], row["unit"]["usd"]
        o["usd"] = None if unit is None or o["calls"] is None else unit * o["calls"]
    by_tier: dict = {}
    by_provider: dict = {}
    left_out = []
    for row in plan["rows"]:
        o = rows[row["id"]]
        if not o["counted"]:
            continue
        if o["usd"] is None:
            left_out.append(row["id"])
            continue
        by_provider[row["provider"]] = by_provider.get(row["provider"], 0.0) + o["usd"]
        if row["provider"] == "anthropic":
            by_tier[row["tier"]] = by_tier.get(row["tier"], 0.0) + o["usd"]
    return {"counts": counts, "rows": rows, "anthropic_by_tier": by_tier, "anthropic_usd": math.fsum(by_tier.values()),
            "by_provider": by_provider, "left_out": left_out}


def budget_part2(plan: dict, cfg: dict, remainder: float) -> dict:
    """The cut order and the add-backs on one plan.

    Threshold = remainder - cut_margin_usd. When the full plan's Anthropic total is above it, cfg["cuts"] apply in
    order, cumulatively (remove: row ids; set: count parameters), and stop at the first total at or below it; every
    applied cut is a step with its total and margin. Then cfg["addbacks"] in order: one is added only if the margin
    (remainder - total) with it stays above cut_margin_usd (strictly); one that does not fit is skipped and the
    later ones are still tried; one whose rows all left the plan with a gate is absent.

    Totals are over the counted rows with a USD figure, and a row without one can only add to them. So every
    applied cut is needed whatever the rows without a figure cost, but the stopping point is decided
    (`cuts_decided`) only when no counted row of the state after the cuts lacks a figure. An add-back is not decided
    when a counted row of the state with it lacks a figure (its own or another) or when an earlier add-back is not
    decided (its cost would change the margin); a not-decided add-back is not added."""
    margin = float(cfg["budget"]["cut_margin_usd"])
    threshold = remainder - margin
    params = {"opus_shown_tasks": int(plan["rules"]["opus_shown_tasks"]), "shown_cap_per_domain": None}
    if "e7_tasks" in plan["rules"]:
        params["e7_tasks"] = None                   # None: rules.e7_tasks (cut C2 sets it)
    removed: set = set()
    cut_of: dict = {}
    full = state = plan_state(plan, removed, params, set())
    steps = []
    if full["anthropic_usd"] > threshold:
        for cut in cfg["cuts"]:
            for rid in cut.get("remove") or []:
                cut_of.setdefault(rid, cut["id"])
            removed |= set(cut.get("remove") or [])
            params = {**params, **(cut.get("set") or {})}
            state = plan_state(plan, removed, params, set())
            steps.append({"id": cut["id"], "name": cut["name"], "anthropic_usd": state["anthropic_usd"],
                          "margin_usd": remainder - state["anthropic_usd"]})
            if state["anthropic_usd"] <= threshold:
                break
    after_cuts = state["anthropic_usd"]
    left_out_after_cuts = list(state["left_out"])
    in_plan = {r["id"] for r in plan["rows"]}
    added: set = set()
    verdicts = []
    undecided = False
    for ab in cfg["addbacks"]:
        ids = [rid for rid in ab["rows"] if rid in in_plan]
        entry = {"id": ab["id"], "name": ab["name"], "rows": ids, "cost_usd": None, "margin_with_usd": None,
                 "lacking": []}
        if not ids:
            entry.update(verdict="absent", reason="; ".join(dict.fromkeys(plan["vanished"].get(rid, "not configured")
                                                                         for rid in ab["rows"])))
            verdicts.append(entry)
            continue
        trial = plan_state(plan, removed, params, added | set(ids))
        lacking = [rid for rid in ids if trial["rows"][rid]["usd"] is None]
        others = [rid for rid in trial["left_out"] if rid not in ids]
        entry.update(cost_usd=trial["anthropic_usd"] - state["anthropic_usd"],
                     margin_with_usd=remainder - trial["anthropic_usd"], lacking=lacking)
        if lacking or others or undecided:
            why = (["a row of it has no USD figure yet"] if lacking else []) + \
                ([f"{len(others)} counted rows of the plan have no USD figure yet"] if others else []) + \
                (["an earlier add-back is not decided"] if undecided else [])
            undecided = True
            entry.update(verdict="not decided", reason="; ".join(why))
        elif remainder - trial["anthropic_usd"] > margin:
            added |= set(ids)
            state = trial
            entry.update(verdict="added", reason="")
        else:
            entry.update(verdict="skipped", reason="")
        verdicts.append(entry)
    final = plan_state(plan, removed, params, added)
    return {"threshold_usd": threshold, "margin_required_usd": margin, "full": full, "steps": steps,
            "fits": after_cuts <= threshold, "anthropic_after_cuts_usd": after_cuts,
            "cuts_decided": not left_out_after_cuts, "left_out_after_cuts": left_out_after_cuts,
            "shortfall_usd": max(0.0, after_cuts - threshold), "addbacks": verdicts, "final": final,
            "removed": sorted(removed), "cut_of": cut_of, "params": params, "added": sorted(added)}


def tie_rates(cfg: dict, root: Path, pairs: list[tuple[str, str]]) -> list[dict]:
    """O/H tie rate in the hidden condition per (judge, domain) from the pilot pointwise records.

    Per domain, cfg["tie_rule"]["sources"][domain] in order: the first source with hidden replicate-0 records of the
    judge (on the source's rubric) is read. Rate = tasks with equal O and H scores / tasks with a valid O and a valid
    H score; the other tasks are counted as excluded. A pair without records has rate None (not measured)."""
    loaded = {dom: [(src, [r for r in read_jsonl(root / src["path"]) if r.get("condition") == "hidden"
                           and r.get("rubric") == src["rubric"] and (r.get("replicate") or 0) == 0])
                    for src in srcs] for dom, srcs in cfg["tie_rule"]["sources"].items()}
    out = []
    for judge_, dom in pairs:
        for src, recs in loaded.get(dom, []):
            mine = [r for r in recs if r.get("grader") == judge_]
            if mine:
                break
        else:
            out.append({"judge": judge_, "domain": dom, "source": None, "tasks": 0, "excluded": 0, "ties": 0,
                        "rate": None})
            continue
        by_task: dict = {}
        for r in mine:
            by_task.setdefault(r["task_id"], {})[r["variant"]] = r.get("score")
        both = [v for v in by_task.values() if isinstance(v.get("O"), int) and isinstance(v.get("H"), int)]
        ties = sum(1 for v in both if v["O"] == v["H"])
        out.append({"judge": judge_, "domain": dom, "source": src["path"], "tasks": len(by_task),
                    "excluded": len(by_task) - len(both), "ties": ties, "rate": ties / len(both) if both else None})
    return out


def gate_part2(root: Path, path: str) -> dict:
    """{"path", "passed" (True | False | None), "known", "text"}: a missing file or passed null is not known yet."""
    d = read_json(root / path)
    passed = (d or {}).get("passed")
    passed = passed if passed in (True, False) else None
    if passed is None:
        text = f"not known yet ({'file not yet written' if d is None else 'passed is null'})"
    else:
        text = "passed" if passed else "failed"
    return {"path": path, "passed": passed, "known": passed is not None, "text": text}


def spend_part2(cfg: dict, root: Path) -> dict:
    """USD per provider of each earlier cost log, of all of them, of Part 1's cost log (0 while not yet written), so
    far, and the Anthropic remainder for Part 2 = project total - earlier - Part 1."""
    b = cfg["budget"]

    def by_provider(path: str) -> dict:
        out = {p: 0.0 for p in PROVIDERS}
        for r in read_jsonl(root / path):
            if r.get("cost_usd") is not None:
                out[r.get("provider")] = out.get(r.get("provider"), 0.0) + float(r["cost_usd"])
        return out

    logs = []
    for path in b["prior_cost_logs"]:
        if not (root / path).exists():
            raise ValueError(f"earlier cost log {path} is missing: the spend so far is not known")
        logs.append({"path": path, "usd": by_provider(path)})
    earlier = {p: math.fsum(log["usd"].get(p, 0.0) for log in logs) for p in PROVIDERS}
    part1_written = (root / b["part1_cost_log"]).exists()
    part1 = by_provider(b["part1_cost_log"]) if part1_written else {p: 0.0 for p in PROVIDERS}
    so_far = {p: earlier[p] + part1.get(p, 0.0) for p in PROVIDERS}
    total = float(b["anthropic_project_total_usd"])
    return {"logs": logs, "earlier": earlier, "part1": part1, "part1_log": b["part1_cost_log"],
            "part1_written": part1_written, "so_far": so_far, "project_total_usd": total,
            "remainder_usd": total - earlier["anthropic"] - part1["anthropic"]}


NAME_PLACEHOLDER = re.compile(r"\{(?:rules|shown|set)\.[A-Za-z0-9_]+\}")


def fill_names(cfg: dict) -> dict:
    """A copy of cfg whose row, cut and add-back names have their placeholders filled in one pass (_fill):
    {rules.<key>} and {shown.<key>} from those config blocks, and in a cut's name {set.<key>} from the cut's own
    `set`; integers print with thousands separators. ValueError when a placeholder has no value."""
    def fmt(v) -> str:
        return f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else str(v)

    values = {**{f"rules.{k}": fmt(v) for k, v in cfg["rules"].items()},
              **{f"shown.{k}": fmt(v) for k, v in cfg["shown"].items()}}

    def one(item: dict) -> dict:
        name = _fill(item["name"], {**values, **{f"set.{k}": fmt(v) for k, v in (item.get("set") or {}).items()}})
        if NAME_PLACEHOLDER.search(name):
            raise ValueError(f"projection_part2: no value for {NAME_PLACEHOLDER.search(name).group()} in "
                             f"{item['name']!r}")
        return {**item, "name": name}

    return {**cfg, **{block: [one(x) for x in cfg[block]] for block in ("rows", "cuts", "addbacks")}}


def _openrouter_part2(listed: list[dict], budget: dict, bc: dict) -> dict:
    """Part 2's OpenRouter total against budget.openrouter_stop_launching_usd. plan_usd: the counted OpenRouter rows
    of the final plan. The optional_openrouter rows (the mitigated open-weight selector) are counted in `total_usd`
    only when plan_usd plus their cost stays at or below the stop line (`optional_fits`; None, not decided, while one
    of them or a counted OpenRouter row has no USD figure). The verdict is also put on the budget dict
    (`optional_openrouter_fits`) for the rows table."""
    final = budget["final"]
    plan_usd = final["by_provider"].get("openrouter", 0.0)
    opt = [r["id"] for r in listed if r.get("optional_openrouter")]
    usd = [final["rows"][i]["usd"] for i in opt]
    waiting = [r["id"] for r in listed if r["provider"] == "openrouter" and r["id"] in final["left_out"]]
    opt_usd = math.fsum(u for u in usd if u is not None)
    fits = None
    if opt and not waiting and None not in usd:
        fits = plan_usd + opt_usd <= float(bc["openrouter_stop_launching_usd"])
    if opt:
        budget["optional_openrouter_fits"] = fits
    return {"plan_usd": plan_usd, "optional_rows": opt, "optional_usd": opt_usd,
            "with_optional_usd": plan_usd + opt_usd, "optional_fits": fits,
            "total_usd": plan_usd + (opt_usd if fits else 0.0), "rows_without_figure": waiting}


def _branch_name(research: bool, e2: bool, gates: dict) -> str:
    def one(what: str, value: bool, gate: dict) -> str:
        return f"{what} gate {'passed' if value else 'failed'}" + ("" if gate["known"] else " (assumed: not known yet)")
    return f"{one('research', research, gates['research'])}, {one('Experiment 2', e2, gates['e2'])}"


def run_part2(cfg: dict, root: Path) -> dict:
    """The Part 2 projection: inputs, factors, tie rates and one evaluated plan per gate branch.

    A listed row of a branch without a USD figure (its unit price or its task count) is pending while a gate is not
    known yet and a file that could measure it is not yet written. Once both gates are known, Part 1 is over and every
    such row is unpriced; an unpriced row in any branch gives exit_code 1."""
    cfg = fill_names(cfg)
    models = yaml.safe_load((root / cfg["models_config"]).read_text(encoding="utf-8"))
    rows_cfg = cfg["rows"]
    ids = [r["id"] for r in rows_cfg]
    if len(ids) != len(set(ids)):
        raise ValueError("projection_part2: duplicate row ids")
    unknown = {rid for c in [*cfg["cuts"], *cfg["addbacks"]] for rid in (c.get("remove") or c.get("rows") or [])}
    if unknown - set(ids):
        raise ValueError(f"projection_part2: cuts or add-backs name unknown rows {sorted(unknown - set(ids))}")
    stats, records, providers = stats_part2(cfg, root)
    factors = {fid: factor_part2(spec, root, stats, providers) for fid, spec in cfg["factors"].items()}
    base = base_part2(cfg, root)
    ctx = {"stats": stats, "records": records, "providers": providers, "factors": factors, "base": base,
           "root": root, "prices": models["prices_usd_per_mtok"], "judges": models["judges"],
           "reference": cfg["reference"]}
    rows = [{**row, "unit": price_row(row, ctx)} for row in rows_cfg]
    ties = tie_rates(cfg, root, list(dict.fromkeys((r["judge"], r["domain"]) for r in rows if r.get("tie_rule"))))
    rate = {(t["judge"], t["domain"]): t["rate"] for t in ties}
    threshold = float(cfg["tie_rule"]["threshold"])
    with_second: list[dict] = []
    for row in rows:
        with_second.append(row)
        tr = rate[(row["judge"], row["domain"])] if row.get("tie_rule") else None
        if tr is None or tr <= threshold:
            continue
        unit = row["unit"]
        text = f"(c) the same calls again as {row['name']} (tie rate {tr:.3f} > {threshold:g}); " + unit["text"]
        with_second.append({**row, "id": f"{row['id']}-s2", "base_id": row["id"], "tie_rule": False,
                            "group": f"{row['group']}, second sample (tie rule)",
                            "name": f"{row['name']}, second sample",
                            "unit": {**unit, "basis": "derived" if unit["usd"] is not None else unit["basis"],
                                     "text": text if unit["usd"] is not None else unit["text"]}})
    rows = with_second
    rows_by_id = {r["id"]: r for r in rows}
    money = spend_part2(cfg, root)
    gates = {name: gate_part2(root, path) for name, path in cfg["gates"].items()}
    final_run = gates["research"]["known"] and gates["e2"]["known"]
    branches = []
    for research_in in ([gates["research"]["passed"]] if gates["research"]["known"] else [True, False]):
        for e2_in in ([gates["e2"]["passed"]] if gates["e2"]["known"] else [True, False]):
            vanished = {}
            for r in rows:
                if r.get("domain") == "research" and not research_in:
                    vanished[r["id"]] = "the research gate failed: the research domain leaves Part 2"
                elif r.get("requires_e2") and not e2_in:
                    vanished[r["id"]] = "the Experiment 2 gate failed: Experiment 2 leaves Part 2"
            plan = {"base": base, "rules": cfg["rules"], "research": research_in, "e2": e2_in, "vanished": vanished,
                    "rows": [r for r in rows if r["id"] not in vanished]}
            budget = budget_part2(plan, cfg, money["remainder_usd"])
            listed, final = plan["rows"], budget["final"]["rows"]
            lacking = {r["id"]: (r["unit"]["text"] if r["unit"]["usd"] is None
                                 else f"task count needs {', '.join(base['pending_files'])} (not yet written)")
                       for r in listed if budget["full"]["rows"][r["id"]]["usd"] is None}
            unpriced = {rid: why for rid, why in lacking.items()
                        if final_run or rows_by_id[rid]["unit"]["basis"] == "unpriced"}
            pending = {rid: why for rid, why in lacking.items() if rid not in unpriced}
            ids_here = {r["id"] for r in listed}
            contingency = [{"id": r["id"], "name": r["name"], "provider": r["provider"],
                            "calls": final[r["id"]]["calls"], "usd": final[r["id"]]["usd"]}
                           for r in listed if r.get("tie_rule") and final[r["id"]]["counted"]
                           and f"{r['id']}-s2" not in ids_here]
            google = {"usd": 0.0, "usd_after_changes": 0.0, "optional_usd": 0.0, "optional_missing": []}
            for r in (r for r in listed if r["provider"] == "google"):
                if r.get("optional"):
                    usd = budget["full"]["rows"][r["id"]]["usd"]
                    if usd is None:
                        google["optional_missing"].append(r["id"])
                    else:
                        google["optional_usd"] += usd
                    continue
                if not final[r["id"]]["counted"] or final[r["id"]]["usd"] is None:
                    continue
                factor = math.prod(float(c["factor"]) for c in cfg["budget"].get("price_changes") or []
                                   if c["model"] == _row_model(r, ctx))
                google["usd"] += final[r["id"]]["usd"]
                google["usd_after_changes"] += final[r["id"]]["usd"] * factor
            branch = {"name": _branch_name(research_in, e2_in, gates), "research": research_in,
                      "e2": e2_in, "vanished": vanished, "rows": [r["id"] for r in listed],
                      "budget": budget, "pending": pending, "unpriced": unpriced, "contingency": contingency,
                      "google": google}
            if "openrouter_remaining_usd" in cfg["budget"]:
                branch["openrouter"] = _openrouter_part2(listed, budget, cfg["budget"])
            branches.append(branch)
    paths = [*(s["path"] for s in cfg["record_sources"]), *(log["path"] for log in cfg["purpose_logs"]),
             cfg["research_set_unit"]["log"], *base["files"].values(), *cfg["gates"].values()]
    return {"inputs": {"spend": money, "gates": gates, "base": {k: v for k, v in base.items()
                                                               if k not in ("coding_tokens", "band_tokens")},
                       "rules": cfg["rules"], "not_written": [p for p in dict.fromkeys(paths)
                                                              if not (root / p).exists()]},
            "factors": factors, "tie_rates": ties, "rows": rows, "branches": branches, "final_run": final_run,
            "exit_code": 1 if any(b["unpriced"] for b in branches) else 0,
            "stats": [{"name": k[0], "condition": k[1], **v} for k, v in sorted(stats.items())]}


def _n(x: int | None) -> str:
    return "pending" if x is None else f"{x:,}"


def _row_status(row: dict, b: dict) -> str:
    """The row's place in the final plan of one branch."""
    full, final = b["full"]["rows"][row["id"]], b["final"]["rows"][row["id"]]
    if row.get("optional_openrouter"):
        return {True: "optional (OpenRouter): counted, it fits under the stop line",
                False: "optional (OpenRouter): not counted, it does not fit under the stop line",
                None: "optional (OpenRouter): not decided"}[b.get("optional_openrouter_fits")]
    if row.get("optional"):
        return "outside every total"
    if row["tier"] == "addback":
        ab = next(a for a in b["addbacks"] if row["id"] in a["rows"])
        return f"add-back ({ab['id']}) {ab['verdict']}"
    if not final["counted"]:
        return f"cut ({b['cut_of'].get(row['id']) or b['cut_of'].get(row.get('base_id'))})"
    if final["tasks"] == full["tasks"]:
        return "kept"
    return f"{_n(final['calls'])} calls after the cuts" if row["tasks"] == TIER2_CALLS else \
        f"{_n(final['tasks'])} tasks after the cuts"


def _rows_tables(rows: list[dict], b: dict, unpriced: dict) -> list[str]:
    head = ["| tier | group | row | provider | tasks | calls | unit USD | total USD | basis | source or note | "
            "final plan |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    lines: list[str] = []
    for tier, title in TIER_TITLES.items():
        mine = [r for r in rows if r["tier"] == tier]
        if not mine:
            continue
        lines += [f"#### {title}", "", *head]
        for r in mine:
            o, u = b["full"]["rows"][r["id"]], r["unit"]
            note = u["text"] + (f". {r['note']}" if r.get("note") else "")
            lacks = "unpriced" if r["id"] in unpriced else "pending"
            basis = "unpriced" if r["id"] in unpriced and u["usd"] is None else u["basis"]
            lines.append(f"| {tier} | {r['group']} | {r['name']} | {r['provider']} | {_n(o['tasks'])} | "
                         f"{_n(o['calls'])} | {_usd(u['usd'])} | {lacks if o['usd'] is None else _usd(o['usd'])} | "
                         f"{basis} | {note} | {_row_status(r, b)} |")
        lines.append("")
    return lines


def _plan_counts_line(b: dict) -> str:
    c = b["budget"]["full"]["counts"]
    e3 = (f"Sonnet 5.5 on {_n(c['e3_coding'])} coding + {_n(c['e3_research'])} research shown tasks"
          if b["research"] else
          f"Sonnet 5.5 on the shown-eligible coding tasks only: {_n(c['e3_coding'])}")
    e7 = f" E7 instruction-only scorer: {_n(c['e7_coding'])} shown-eligible coding tasks." if "e7_coding" in c else ""
    return (f"Tasks: {_n(c['coding'])} coding, {_n(c['research'])} research; shown-eligible {_n(c['shown_coding'])} "
            f"coding, {_n(c['shown_research'])} research; Opus 5.5 pointwise shown {_n(c['o55_shown_coding'])} coding "
            f"+ {_n(c['o55_shown_research'])} research (pairwise shown {_n(c['o55_pair_coding'])} + "
            f"{_n(c['o55_pair_research'])}); Experiment 3 pointwise: {e3}, Opus 5.5 on {_n(c['o55_e3_coding'])} + "
            f"{_n(c['o55_e3_research'])}; Experiment 2: {c['e2_coding']} coding + {c['e2_research']} research tasks; "
            f"research sets owed: {_n(c['research_sets_owed'])}.{e7}")


def _budget_lines_part2(b: dict, rows: dict, remainder: float, addbacks_note: str | None = None) -> list[str]:
    """The Anthropic totals, the cuts, the add-backs and the final plan of one branch. With no add-back configured
    and an `addbacks_note` (cfg["addbacks_note"]), the note is printed in place of the empty add-back table."""
    bud = b["budget"]
    full, final = bud["full"], bud["final"]
    lines = ["#### Anthropic totals of the full plan, before cuts", "", "| part | USD |", "|---|---|",
             *[f"| {TIER_TITLES[t]} | {full['anthropic_by_tier'].get(t, 0.0):.2f} |" for t in (1, 2)],
             f"| full plan | {full['anthropic_usd']:.2f} |", ""]
    if full["left_out"]:
        lines += [f"Left out of these totals: {len(full['left_out'])} rows without a USD figure yet (listed at the "
                  "end of the plan).", ""]
    lines += ["#### Cuts", ""]
    n_out, thr = len(bud["left_out_after_cuts"]), bud["threshold_usd"]
    if not bud["steps"] and bud["cuts_decided"]:
        lines.append(f"No cut needed: the full plan's Anthropic total {full['anthropic_usd']:.2f} USD is at or below "
                     f"the threshold {thr:.2f} USD (margin {remainder - full['anthropic_usd']:.2f} USD).")
    elif not bud["steps"]:
        lines.append(f"Not decided: {n_out} counted rows have no USD figure yet. The rows priced so far come to "
                     f"{full['anthropic_usd']:.2f} USD, at or below the threshold {thr:.2f} USD; the rows without a "
                     "figure can only add to that, so cuts can still be needed.")
    else:
        lines += ["| step | cut | Anthropic total after it (USD) | margin after it (USD) |", "|---|---|---|---|",
                  f"| none | the full plan | {full['anthropic_usd']:.2f} | {remainder - full['anthropic_usd']:.2f} |",
                  *[f"| {s['id']} | {s['name']} | {s['anthropic_usd']:.2f} | {s['margin_usd']:.2f} |"
                    for s in bud["steps"]], ""]
        last = bud["steps"][-1]["id"]
        if bud["fits"] and bud["cuts_decided"]:
            lines.append(f"After cut {last} the Anthropic total is at or below the threshold {thr:.2f} USD; the later "
                         "cuts are not applied.")
        elif bud["fits"]:
            lines.append(f"After cut {last} the rows priced so far come to {bud['anthropic_after_cuts_usd']:.2f} USD, "
                         f"at or below the threshold {thr:.2f} USD. The {n_out} counted rows without a USD figure yet "
                         "can only add to the total, so every cut above is needed whatever they cost; whether a later "
                         "cut is needed is not decided until they are priced.")
        else:
            lines.append(f"Still above the threshold after the last cut ({last}): the Anthropic total "
                         f"{bud['anthropic_after_cuts_usd']:.2f} USD exceeds {thr:.2f} USD by "
                         f"{bud['shortfall_usd']:.2f} USD (the shortfall)."
                         + ("" if bud["cuts_decided"] else f" {n_out} counted rows without a USD figure yet are not "
                                                           "in this total, so the shortfall is at least this."))
    m = bud["margin_required_usd"]
    if not bud["addbacks"] and addbacks_note:
        lines += ["", "#### Add-backs", "", addbacks_note]
    else:
        lines += ["", "#### Add-backs", "", "| add-back | rows | cost (USD) | margin with it (USD) | verdict |",
                  "|---|---|---|---|---|"]
    for a in bud["addbacks"]:
        names = "; ".join(rows[i]["name"] for i in a["rows"]) or "none in this plan"
        verdict = {"added": f"added: the margin with it stays above {m:g} USD",
                   "skipped": f"skipped: the margin with it would not be above {m:g} USD",
                   "absent": f"not in this plan: {a['reason']}",
                   "not decided": f"not decided: {a['reason']}"}[a["verdict"]]
        known = a["verdict"] in ("added", "skipped")
        cost = f"{a['cost_usd']:.2f}" if known or (a["verdict"] == "not decided" and not a["lacking"]) else ""
        margin = f"{a['margin_with_usd']:.2f}" if known else ""
        lines.append(f"| ({a['id']}) {a['name']} | {names} | {cost} | {margin} | {verdict} |")
    lines += ["", "#### Final plan", "", "| part | Anthropic USD |", "|---|---|",
              *[f"| {TIER_TITLES[t]} | {final['anthropic_by_tier'].get(t, 0.0):.2f} |" for t in TIER_TITLES],
              f"| total | {final['anthropic_usd']:.2f} |", "",
              f"Margin = remainder - total = {remainder:.2f} - {final['anthropic_usd']:.2f} = "
              f"{remainder - final['anthropic_usd']:.2f} USD."]
    if final["left_out"]:
        lines.append(f"These figures leave out {len(final['left_out'])} counted rows without a USD figure yet, so the "
                     "cuts and add-backs can change when they are priced.")
    return lines


def _provider_lines_part2(b: dict, money: dict, cfg: dict, rows: dict) -> list[str]:
    bc = cfg["budget"]
    final = b["budget"]["final"]
    lines = []
    if b["contingency"]:
        cont: dict = {}
        pend = 0
        for c in b["contingency"]:
            if c["usd"] is None:
                pend += 1
            else:
                cont[c["provider"]] = cont.get(c["provider"], 0.0) + c["usd"]
        per = ", ".join(f"{p} {v:.2f} USD" for p, v in cont.items()) or "0 USD"
        which = cfg["tie_rule"].get("contingency_rows", "the pointwise hidden rows of the final plan")
        lines += [f"Tie-rule contingency, outside every total: a second scoring sample for every judge not above the "
                  f"line, on {which}, would cost {per}"
                  + (f" ({pend} rows without a USD figure yet)" if pend else "") + ":", "",
                  "| second sample of | provider | calls | USD |", "|---|---|---|---|",
                  *[f"| {c['name']} | {c['provider']} | {_n(c['calls'])} | "
                    f"{'pending' if c['usd'] is None else _usd(c['usd'])} |" for c in b["contingency"]], ""]
    o = final["by_provider"].get("openrouter", 0.0)
    so, earlier, part1 = money["so_far"], money["earlier"], money["part1"]
    g = b["google"]
    changes = "; ".join(f"{c['model']} x{float(c['factor']):g} from {c['from']}" for c in bc.get("price_changes") or [])

    def enough(have: float, need: float) -> str:
        return "enough" if have >= need else f"short by {need - have:.2f} USD"

    if "openrouter_remaining_usd" in bc:
        lines += _openrouter_lines_part2(b["openrouter"], bc, rows)
    else:
        cap = float(bc["openrouter_cap_usd"])
        lines.append(f"- OpenRouter: {earlier['openrouter']:.4f} USD before Part 1 + {part1['openrouter']:.4f} USD in "
                     f"Part 1 = {so['openrouter']:.4f} USD so far; Part 2's final plan {o:.2f} USD; "
                     f"{so['openrouter'] + o:.2f} USD against the {cap:g} USD cap, a margin of "
                     f"{cap - so['openrouter'] - o:.2f} USD.")
    if bc.get("google_route") == "vertex":
        # Every Gemini row runs on the GCP trial credit (addendum section 3b): the plan is drawn from the credit,
        # at most the larger of the two figures if an Argon row runs after the price change.
        credit = float(bc["google_vertex_credit_usd"])
        most = max(g["usd"], g["usd_after_changes"])
        waiting = [i for i in final["left_out"] if rows[i]["provider"] == "google"]
        lines.append(f"- Google, vertex: Part 2's plan {g['usd']:.2f} USD at current prices, "
                     f"{g['usd_after_changes']:.2f} USD at the changed prices ({changes or 'none'}), against the GCP "
                     f"trial credit of {credit:g} USD: at most {most:.2f} USD of it ({credit - most:.2f} USD left); "
                     "zero cash" + (f" (rows without a USD figure yet: {len(waiting)})" if waiting else "") + ".")
    else:
        balance = float(bc["google_aistudio_balance_usd"])
        avail = balance - part1["google"]
        topup = avail + float(bc["google_topup_usd"])
        missing_vertex = "; ".join(rows[i]["name"] for i in g["optional_missing"])
        lines += [f"- Google, aistudio: Part 2's final plan {g['usd']:.2f} USD at current prices, "
                  f"{g['usd_after_changes']:.2f} USD at the changed prices ({changes or 'none'}). Balance "
                  f"{balance:.2f} USD (config) - {part1['google']:.2f} USD spent in Part 1 = {avail:.2f} USD: "
                  f"{enough(avail, g['usd'])} at "
                  f"current prices; with a {float(bc['google_topup_usd']):g} USD top-up, {topup:.2f} USD: "
                  f"{enough(topup, g['usd'])} at current prices, {enough(topup, g['usd_after_changes'])} at the "
                  "changed prices.",
                  f"- Google, vertex (optional, outside every total; vertex only, credit unconfirmed): "
                  f"{g['optional_usd']:.2f} USD" + (f" (rows without a USD figure yet: {missing_vertex})"
                                                    if missing_vertex else "") + "."]
    lines.append(f"- Local GPU (Skywork reward model): {final['by_provider'].get('local', 0.0):.2f} USD.")
    return lines


def _openrouter_lines_part2(orb: dict, bc: dict, rows: dict) -> list[str]:
    """The OpenRouter line against the remaining budget and the stop line, and the sentence saying whether the
    optional OpenRouter rows are counted (_openrouter_part2)."""
    rem, stop = float(bc["openrouter_remaining_usd"]), float(bc["openrouter_stop_launching_usd"])
    waiting = orb["rows_without_figure"]
    line = (f"- OpenRouter: Part 2's plan {orb['plan_usd']:.2f} USD against the {rem:.2f} USD remaining (stop "
            f"launching at {stop:.2f})" + (f" (rows without a USD figure yet: {len(waiting)})" if waiting else ""))
    if not orb["optional_rows"]:
        return [line + "."]
    fits = orb["optional_fits"]
    verdict = {True: "fits", False: "does not fit", None: "not decided"}[fits]
    names = "; ".join(rows[i]["name"] for i in orb["optional_rows"])
    if fits is None:
        why = f"{len(waiting)} counted OpenRouter rows" if waiting else "the optional row"
        text = f"Whether the optional row ({names}) is counted is not decided: {why} have no USD figure yet."
    elif fits:
        text = (f"The optional row ({names}, {orb['optional_usd']:.2f} USD) is counted in the OpenRouter total: with "
                f"it the total is {orb['with_optional_usd']:.2f} USD, at or below the {stop:.2f} USD stop line.")
    else:
        text = (f"The optional row ({names}, {orb['optional_usd']:.2f} USD) is not counted: with it the total would "
                f"be {orb['with_optional_usd']:.2f} USD, above the {stop:.2f} USD stop line, so the OpenRouter total "
                f"stays {orb['plan_usd']:.2f} USD.")
    with_opt = "" if fits is None else f" {orb['with_optional_usd']:.2f} USD"
    return [f"{line}; with the optional mitigated open-weight selector{with_opt}: {verdict}.", f"- {text}"]


def _e7_counts_line(base: dict, rules: dict, cfg: dict) -> str:
    """The E7 task count with its realised split by gap type (rules.e7_per_gap is the per-type target, for the report
    only), or pending while the coding sets file is not yet written."""
    if base.get("e7_available") is None:
        return f"- E7 (instruction-only scorer): pending ({base['files']['coding']} not yet written)."
    per, elig, target = base["e7_per_gap"], base["e7_eligible_per_gap"], rules.get("e7_per_gap")
    short = [g for g in per if target is not None and per[g] < int(target)]
    split = ", ".join(f"{g} {per[g]} of {elig[g]} eligible" for g in per)
    note = ""
    if target is not None:
        note = f"; target {target} per gap type" + (f", {' and '.join(short)} short, its turns given to the others"
                                                     if short else "")
    return (f"- E7 (instruction-only scorer): {base['e7_available']} of the {rules['e7_tasks']} tasks of the rule, "
            f"shown-eligible coding tasks picked as Experiment 2 picks (seed {cfg['seed']}, one per gap type in turn): "
            f"{split}{note}.")


def markdown_part2(result: dict, cfg: dict) -> str:
    """The Part 2 projection as markdown: inputs, measured factors, tie rates, then one section per plan."""
    inp = result["inputs"]
    money, gates, base = inp["spend"], inp["gates"], inp["base"]
    rows = {r["id"]: r for r in result["rows"]}
    rem = money["remainder_usd"]
    lines = ["# Projection for Part 2: two tiers, pointwise primary (session 5)", "",
             "Arithmetic on recorded costs (overclaim.projection_v4 with `plan: part2`); nothing was run or called. "
             "Tier 1 (pointwise) is the primary result and runs first; Tier 2 (pairwise) is the appendix and is cut "
             "first (revision of 2026-11-06). Every row is priced: measured, its source named, or derived, its basis "
             "in the note: (a) the reference judge's measured cost for the same call kind, condition and domain, "
             "scaled by a measured ratio between the two models; (b) a measured token profile priced at list prices; "
             "(c) another call's measured unit cost. A row whose measurement or task count comes from a file not yet "
             "written is pending: it is listed and left out of every total, never counted as zero, and a cut or "
             "add-back that its figure could change is not decided. Once both gates are known, Part 1 is over: a row "
             "without a USD figure is then unpriced and the run exits with code 1.", ""]
    if cfg.get("decisions"):
        lines += ["## Decisions applied", "", *[f"- {d}" for d in cfg["decisions"]], ""]
    lines += ["## Inputs", "", "### Spend so far", "", "| cost log | anthropic | openrouter | google |",
              "|---|---|---|---|"]
    for log in money["logs"]:
        lines.append(f"| {log['path']} | " + " | ".join(f"{log['usd'].get(p, 0.0):.4f}" for p in PROVIDERS) + " |")
    lines += ["| earlier logs together | " + " | ".join(f"{money['earlier'][p]:.4f}" for p in PROVIDERS) + " |",
              f"| {money['part1_log']} (Part 1){'' if money['part1_written'] else ', not yet written'} | "
              + " | ".join(f"{money['part1'][p]:.4f}" for p in PROVIDERS) + " |",
              "| so far | " + " | ".join(f"{money['so_far'][p]:.4f}" for p in PROVIDERS) + " |", "",
              f"Anthropic remainder for Part 2 = {money['project_total_usd']:.2f} - {money['earlier']['anthropic']:.2f}"
              f" (earlier) - {money['part1']['anthropic']:.2f} (Part 1) = {rem:.2f} USD. The cuts apply while the "
              f"total exceeds remainder - {float(cfg['budget']['cut_margin_usd']):g} = "
              f"{rem - float(cfg['budget']['cut_margin_usd']):.2f} USD.", "", "### Gates", "",
              f"- Research gate ({gates['research']['path']}): {gates['research']['text']}.",
              f"- Experiment 2 gate ({gates['e2']['path']}): {gates['e2']['text']}."]
    if gates["research"]["known"] and gates["e2"]["known"]:
        lines.append("- Both gates are known: one plan follows from them, and every row of it needs a USD figure.")
    else:
        lines.append(f"- A gate not known yet gives one plan per branch ({len(result['branches'])} plans below, each "
                     "headed by the branch it assumes).")
    lines += ["", "### Counts", ""]
    f = base["files"]
    if base["coding"] is None:
        lines.append(f"- Coding: pending ({f['coding']} not yet written).")
    else:
        lines.append(f"- Coding ({f['coding']}): {base['coding']} sets; shown-eligible {base['shown_coding']}; "
                     f"{base['band']} in the band ({cfg['shown']['max_transcript_tokens']:,}, "
                     f"{cfg['shown']['upper_band_tokens']:,}] rendered tokens"
                     + (f"; {base['coding_without_tokens']} sets without transcript_tokens"
                        if base["coding_without_tokens"] else "") + ".")
    if base["research_built"] is None:
        lines.append(f"- Research: pending ({f['research']} not yet written).")
    else:
        lines.append(f"- Research ({f['research']}): {base['research_built']} sets built, {base['research_eligible']} "
                     "shown-eligible.")
    r = inp["rules"]
    if "e7_tasks" in r:
        lines.append(_e7_counts_line(base, r, cfg))
    after = "; ".join(f"{c['id']}: {c['set']['opus_shown_tasks']}" for c in cfg["cuts"]
                      if "opus_shown_tasks" in (c.get("set") or {}))
    samples = sorted({int(x["per_task"]) for x in result["rows"] if x["tasks"] in ("e2_coding", "e2_research")})
    ab_ii = f"add-back (ii) on {r['addback_ii_tasks']}, " if "addback_ii_tasks" in r else ""
    lines += [f"- When the research gate passed, research tasks = {r['research_tasks']} (the sets built plus the sets "
              "owed), all shown-eligible if every built set is (else scaled by the built share); 0 when it failed.",
              f"- Opus 5.5 pointwise shown on {r['opus_shown_tasks']} tasks{f' (after cut {after})' if after else ''}, "
              f"{ab_ii}Opus 5.5 pairwise shown on {r['opus_shown_tasks']}: split "
              "over the domains in proportion to their shown-eligible counts (coding share rounded half up, each at "
              f"most its count). Experiment 3 on {r['e3_tasks']} shown tasks: every shown-eligible coding task, the "
              f"rest research; Opus 5.5 on {r['opus_e3_tasks']} of those, split the same way. Experiment 2: "
              f"{r['e2_tasks_per_domain']} tasks per domain present, {' or '.join(map(str, samples)) or 'no'} samples "
              "per task; the mitigated best-of-16 on the shown-eligible tasks among the coding tasks that Experiment 2 "
              "takes (seed 0, one per gap type in turn, as the pilot picks them).", "",
              "### Measured factors", "", "| factor | value and basis |", "|---|---|"]
    for fid, fac in result["factors"].items():
        why = cfg["factors"][fid].get("why")
        value = fac["text"] if fac["value"] is not None else "not measured: " + "; ".join(fac["missing"])
        lines.append(f"| {fid} | {value}{f'. {why}' if why else ''} |")
    thr = float(cfg["tie_rule"]["threshold"])
    lines += ["", f"### Tie rule: O/H tie rate in the hidden condition (pilot pointwise records, replicate 0)", "",
              f"A judge and domain above {thr:g} get a second scoring sample per message (a second-sample row, the "
              "same calls again); every other judge's second sample is printed as a contingency, outside the totals.",
              "", "| judge | domain | source | tasks | excluded (no valid O or H score) | O = H | tie rate | "
              "second sample |", "|---|---|---|---|---|---|---|---|"]
    for t in result["tie_rates"]:
        if t["rate"] is None:
            lines.append(f"| {t['judge']} | {t['domain']} | not measured | | | | | no (contingency) |")
        else:
            lines.append(f"| {t['judge']} | {t['domain']} | {t['source']} | {t['tasks']} | {t['excluded']} | "
                         f"{t['ties']} | {t['rate']:.3f} | {'yes' if t['rate'] > thr else 'no (contingency)'} |")
    lines += ["", "### Input files not yet written", "",
              *([f"- {p}" for p in inp["not_written"]] or ["- none"]), ""]
    first = result["branches"][0]
    for i, b in enumerate(result["branches"], 1):
        bud = b["budget"]
        lines += [f"## Plan {i}: {b['name']}", "", _plan_counts_line(b), ""]
        if b["research"] and b["e2"]:
            lines += ["Experiment 2 has not been piloted in the research domain: its research rows are derived from "
                      "other measured calls (see their notes).", ""]
        if b["vanished"]:
            reasons: dict = {}
            for rid, why in b["vanished"].items():
                reasons.setdefault(why, []).append(rows[rid]["name"])
            lines += [f"Not in this plan, because {why}: {'; '.join(names)}." for why, names in reasons.items()] + [""]
        lines += ["### Rows", ""]
        here = [rows[i_] for i_ in b["rows"]]
        if b is first:
            lines += _rows_tables(here, bud, b["unpriced"])
        else:
            mine, theirs = bud["full"]["rows"], first["budget"]["full"]["rows"]
            changed = [r for r in here if mine[r["id"]]["tasks"] != theirs[r["id"]]["tasks"]
                       or _row_status(r, bud) != _row_status(r, first["budget"])]
            if changed:
                lines += [f"Rows as in plan 1, except the rows not in this plan (above) and these {len(changed)}:", "",
                          "| row | tasks in plan 1 | tasks here | total USD here | final plan here |",
                          "|---|---|---|---|---|",
                          *[f"| {r['name']} | {_n(theirs[r['id']]['tasks'])} | {_n(mine[r['id']]['tasks'])} | "
                            f"{'pending' if mine[r['id']]['usd'] is None else _usd(mine[r['id']]['usd'])} | "
                            f"{_row_status(r, bud)} |" for r in changed], ""]
            else:
                lines += ["Rows as in plan 1, except the rows not in this plan (above).", ""]
        lines += [*_budget_lines_part2(b, rows, rem, cfg.get("addbacks_note")), "",
                  "#### OpenRouter, Google and the contingency", "",
                  *_provider_lines_part2(b, money, cfg, rows), "", "#### Rows without a USD figure", ""]
        if b["pending"] and b is first:
            grouped: dict = {}
            for rid, why in b["pending"].items():
                grouped.setdefault(why, []).append(rows[rid]["name"])
            lines += [f"Pending ({len(b['pending'])}; a file they need is not yet written), by reason:", "",
                      *[f"- {why}: {'; '.join(names)}." for why, names in grouped.items()], ""]
        elif b["pending"]:
            lines += [f"Pending: {len(b['pending'])} rows, each also pending in plan 1 for the reason listed there.",
                      ""]
        if b["unpriced"]:
            why = ("both gates are known, so Part 1 is over and every row of the plan needs a USD figure"
                   if result["final_run"] else "every file that could measure them exists")
            lines += [f"Unpriced ({len(b['unpriced'])}; {why}, so the run exits with code 1):", "",
                      *[f"- {rows[rid]['name']}: {text}" for rid, text in b["unpriced"].items()], ""]
        if not b["pending"] and not b["unpriced"]:
            lines += ["None.", ""]
    lines.append("The cut order and the add-backs are a budget rule (session-5 brief, Step 6, as replaced by the "
                 "revision of 2026-11-06); they are not pre-registered.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT), help="repository root the config paths are relative to")
    args = ap.parse_args(argv)
    root = Path(args.root)
    cfg = yaml.safe_load((root / args.config if not Path(args.config).is_absolute() else Path(args.config))
                         .read_text(encoding="utf-8"))
    code = 0
    if cfg.get("plan") == PART2:
        result = run_part2(cfg, root)
        md, code = markdown_part2(result, cfg), result["exit_code"]
    else:
        result = run(cfg, root)
        md = markdown(result, cfg)
    out_json, out_md = root / cfg["output"]["json"], root / cfg["output"]["markdown"]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    out_md.write_text(md, encoding="utf-8")
    print(md, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

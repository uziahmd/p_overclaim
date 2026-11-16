"""Full-run cost projection from the pilot's measured per-call costs.

Unit costs come from the pilot's judge records (live calls only, i.e. `cached` false):
- measured: the mean cost per call of that judge in that condition;
- estimated (a judge that was not run in that condition): the reference judge's mean token profile for the
  condition (uncached input, cache write, cache read), scaled by the tokenizer ratio between the two judges
  (mean hidden input tokens per call on the same prompts; a judge never run at all borrows the ratio and the output
  tokens of `like`, a sibling from the same model generation), priced with the target judge's own prices and its
  own mean output tokens per call. Every estimated row says so.

The projection is arithmetic on those unit costs and the call counts in cfg["projection"]["items"]; nothing is run.
"""

from __future__ import annotations

from collections import defaultdict

BUCKETS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def unit_stats(judge_records: list[dict], sets: tuple[str, ...] | None = None,
               thought_separate: tuple[str, ...] = ()) -> dict[tuple[str, str], dict]:
    """{(grader, condition): {"n": live calls, "usd": mean cost per call, "tokens": {bucket: mean per call}}}.

    Only live records with a cost count (a disk-cache hit has cost 0 and would bias the mean). Smoke-only records are
    left out. `sets` restricts to those matched-set names (None = all). For the graders in `thought_separate` (Gemini)
    the API reports thought tokens outside output_tokens and bills them as output, so they are added to the
    output-token mean; every other provider already counts them inside output_tokens.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in judge_records:
        if r.get("cached") or r.get("smoke_only") or r.get("cost_usd") is None:
            continue
        if sets is not None and r.get("set", "v1") not in sets:
            continue
        groups[(r["grader"], r["condition"])].append(r)
    out: dict[tuple[str, str], dict] = {}
    for key, recs in groups.items():
        n = len(recs)
        tokens = {b: sum((r.get("usage") or {}).get(b) or 0 for r in recs) / n for b in BUCKETS}
        if key[0] in thought_separate:
            tokens["output_tokens"] += sum((r.get("usage") or {}).get("thought_tokens") or 0 for r in recs) / n
        out[key] = {"n": n, "usd": sum(r["cost_usd"] for r in recs) / n, "tokens": tokens}
    return out


def _price(prices: dict, model: str, tokens: dict) -> float:
    p = prices[model]
    return (tokens["input_tokens"] * p["input"]
            + tokens["cache_creation_input_tokens"] * p.get("cache_write", p["input"])
            + tokens["cache_read_input_tokens"] * p.get("cache_read", p["input"])
            + tokens["output_tokens"] * p["output"]) / 1_000_000


def _reamortize(tokens: dict, calls_per_prefix: int) -> dict:
    """The same per-call token profile when one cached prefix serves `calls_per_prefix` calls instead of the pilot's
    six: each call either writes or reads the whole prefix, so the prefix is (write + read) tokens per call; the new
    profile writes it once per `calls_per_prefix` calls and reads it otherwise."""
    prefix = tokens["cache_creation_input_tokens"] + tokens["cache_read_input_tokens"]
    out = dict(tokens)
    out["cache_creation_input_tokens"] = prefix / calls_per_prefix
    out["cache_read_input_tokens"] = prefix * (calls_per_prefix - 1) / calls_per_prefix
    return out


def unit_cost(stats: dict, judge: str, condition: str, *, model: str, prices: dict, reference: str,
              like: str | None = None, calls_per_prefix: int | None = None,
              extra_output_tokens: int = 0) -> dict:
    """{"usd", "basis"} for one call of `judge` in `condition` (see the module docstring).

    `calls_per_prefix` re-amortises the cached prefix (Experiment 2 scores 32 messages per transcript, not 6) and
    `extra_output_tokens` adds output tokens per call (the mitigation's claim list); either one makes the row an
    estimate even for a measured judge.
    """
    adjusted = calls_per_prefix is not None or extra_output_tokens
    if (judge, condition) in stats and not adjusted:
        s = stats[(judge, condition)]
        return {"usd": s["usd"], "basis": f"measured ({s['n']} calls)"}
    if (judge, condition) in stats:
        s = stats[(judge, condition)]
        tokens = dict(s["tokens"])
        notes = []
        if calls_per_prefix is not None:
            tokens = _reamortize(tokens, calls_per_prefix)
            notes.append(f"one cache write per {calls_per_prefix} calls")
        if extra_output_tokens:
            tokens["output_tokens"] += extra_output_tokens
            notes.append(f"+{extra_output_tokens} output tokens per call")
        return {"usd": _price(prices, model, tokens),
                "basis": f"estimated: measured token profile ({s['n']} calls), " + ", ".join(notes)}
    ref = stats.get((reference, condition))
    ref_hidden = stats.get((reference, "hidden"))
    if ref is None or ref_hidden is None:
        return {"usd": None, "basis": f"no measurement of {reference} in the {condition} condition"}
    own = stats.get((judge, "hidden")) or (stats.get((like, "hidden")) if like else None)
    if own is None:
        return {"usd": None, "basis": f"no hidden-condition measurement for {judge}" + (f" or {like}" if like else "")}
    source = judge if (judge, "hidden") in stats else like
    ratio = own["tokens"]["input_tokens"] / ref_hidden["tokens"]["input_tokens"]
    tokens = {b: ref["tokens"][b] * ratio for b in BUCKETS[:3]}
    tokens["output_tokens"] = own["tokens"]["output_tokens"] + extra_output_tokens
    extras = ""
    if calls_per_prefix is not None:
        tokens = _reamortize(tokens, calls_per_prefix)
        extras += f", one cache write per {calls_per_prefix} calls"
    if extra_output_tokens:
        extras += f", +{extra_output_tokens} output tokens per call"
    return {"usd": _price(prices, model, tokens),
            "basis": (f"estimated: {reference} {condition} token profile x tokenizer ratio {ratio:.2f} "
                      f"(hidden input tokens, {source}), {source} output tokens, {model} prices{extras}")}


def project(cfg: dict, judge_records: list[dict]) -> dict:
    """Rows and totals for cfg["projection"]: one row per item x judge, and one total per provider.

    cfg["projection"] = {"reference": grader, "sets": [...] | absent, "thought_separate": [graders] | absent,
    "targets_usd": {provider: usd},
    "items": [{"name", "group" (experiment, for subtotals), "provider", "judges": [...], "condition",
    "calls_per_judge", "like": {judge: sibling},
    "calls_per_prefix", "extra_output_tokens", "note"}]}. A judge's model comes from cfg["judges"][judge]["model"].
    """
    proj = cfg["projection"]
    stats = unit_stats(judge_records, tuple(proj["sets"]) if proj.get("sets") else None,
                       tuple(proj.get("thought_separate") or ()))
    prices = cfg["prices_usd_per_mtok"]
    rows: list[dict] = []
    totals: dict[str, dict] = {}
    for item in proj["items"]:
        for judge in item["judges"]:
            model = cfg["judges"][judge]["model"]
            unit = unit_cost(stats, judge, item["condition"], model=model, prices=prices,
                             reference=item.get("reference", proj["reference"]),
                             like=(item.get("like") or {}).get(judge),
                             calls_per_prefix=item.get("calls_per_prefix"),
                             extra_output_tokens=int(item.get("extra_output_tokens", 0)))
            calls = int(item["calls_per_judge"])
            usd = None if unit["usd"] is None else unit["usd"] * calls
            rows.append({"item": item["name"], "group": item.get("group", ""), "provider": item["provider"], "judge": judge,
                         "condition": item["condition"], "calls": calls, "unit_usd": unit["usd"], "usd": usd,
                         "basis": unit["basis"], "note": item.get("note", "")})
            t = totals.setdefault(item["provider"], {"usd": 0.0, "calls": 0, "missing": 0})
            t["calls"] += calls
            if usd is None:
                t["missing"] += 1
            else:
                t["usd"] += usd
    for provider, t in totals.items():
        t["target_usd"] = (proj.get("targets_usd") or {}).get(provider)
    subtotals: dict[tuple[str, str], dict] = {}
    for r in rows:  # per provider and group (the experiment), in first-seen order
        sub = subtotals.setdefault((r["provider"], r["group"]), {"usd": 0.0, "calls": 0, "missing": 0})
        sub["calls"] += r["calls"]
        if r["usd"] is None:
            sub["missing"] += 1
        else:
            sub["usd"] += r["usd"]
    return {"rows": rows, "totals": totals,
            "subtotals": [{"provider": p, "group": g, **v} for (p, g), v in subtotals.items()]}


def projection_markdown(result: dict) -> str:
    """The projection as two markdown tables (rows, then totals against the targets)."""
    def money(x: float | None) -> str:
        return "n/a" if x is None else (f"{x:.4f}" if x < 0.1 else f"{x:.2f}")

    lines = ["| item | judge | condition | calls | USD per call | USD | basis |", "|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        lines.append(f"| {r['item']} | {r['judge']} | {r['condition']} | {r['calls']:,} | {money(r['unit_usd'])} | "
                     f"{money(r['usd'])} | {r['basis']} |")
    if any(sub["group"] for sub in result.get("subtotals", [])):
        lines += ["", "| provider | experiment | calls | projected USD | rows without a unit cost |", "|---|---|---|---|---|"]
        for sub in result["subtotals"]:
            lines.append(f"| {sub['provider']} | {sub['group'] or 'n/a'} | {sub['calls']:,} | {sub['usd']:.2f} | "
                         f"{sub['missing']} |")
    lines += ["", "| provider | calls | projected USD | target or cap USD | rows without a unit cost |", "|---|---|---|---|---|"]
    for provider, t in sorted(result["totals"].items()):
        target = "n/a" if t.get("target_usd") is None else f"{t['target_usd']:.0f}"
        lines.append(f"| {provider} | {t['calls']:,} | {t['usd']:.2f} | {target} | {t['missing']} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Session 3: the re-scoped full run (150 tasks), with the Anthropic budget, the cut order and the Google prices
# ---------------------------------------------------------------------------------------------

E3_MINUS_PLAIN = "e3_minus_plain"


def _extra_output(stats: dict, item: dict, reference: str) -> tuple[int, str]:
    """(extra output tokens per call, basis note). "e3_minus_plain" = mean output tokens of the reference judge's
    verification-first records minus those of its plain records, both shown; 0 when either is missing."""
    extra = item.get("extra_output_tokens", 0)
    if extra != E3_MINUS_PLAIN:
        return int(extra or 0), ""
    vf, plain = stats.get((f"{reference}+verify_first", "shown")), stats.get((reference, "shown"))
    if vf is None or plain is None:
        missing = [k for k, s in ((f"{reference}+verify_first", vf), (reference, plain)) if s is None]
        return 0, f"; e3_minus_plain = 0 (no shown measurement of {' or '.join(missing)})"
    diff = int(round(vf["tokens"]["output_tokens"] - plain["tokens"]["output_tokens"]))
    return diff, (f"; e3_minus_plain = {diff:+d} output tokens per call ({reference}+verify_first minus {reference}, "
                  "shown)")


def _rescoped_basis(unit: dict, stats: dict, item: dict, reference: str) -> str:
    """unit_cost's basis with the source of the measurement (stats entries may carry "source")."""
    own = stats.get((item["stat"], item["condition"]))
    if own is not None and unit["basis"] == f"measured ({own['n']} calls)":
        return f"measured in {own.get('source', 'the pilot')} ({own['n']} calls)"
    if unit["usd"] is None:
        return unit["basis"]
    names = {item["stat"], reference, item.get("like")}
    sources = sorted({s["source"] for (name, _c), s in stats.items() if name in names and s.get("source")})
    return unit["basis"] + (f"; data: {', '.join(sources)}" if sources else "")


def _price_factor(changes: list[dict], model: str) -> float:
    factor = 1.0
    for change in changes:
        if change["model"] == model:
            factor *= float(change["factor"])
    return factor


def project_rescoped(cfg: dict, stats: dict, anthropic_spent_this_session: float) -> dict:
    """The re-scoped full-run projection: arithmetic on unit costs, nothing is run.

    `stats` = {(name, condition): {"n", "usd", "tokens", optional "source"}} (unit_stats shape). Each item of
    cfg["projection"]["items"] ({id, group, provider, name, calls, stat, condition, model, like, calls_per_prefix,
    extra_output_tokens (int or "e3_minus_plain"), optional, note}) is priced with unit_cost (reference
    cfg["projection"]["reference"]). Rows keep config order; running_total is per provider and leaves out optional
    items (None on an optional row) and items without a unit cost (usd None, counted in the provider's `missing`).
    Anthropic: remaining = project total - spent before - spent this session, margin = remaining - projected (non-
    optional rows), and the optional items fit when margin - their cost >= margin_for_mitigated_bo16_usd (None when
    a cost or the threshold is missing). Cuts apply cumulatively (remove items, set calls); for each, the Anthropic
    total and the margin after it. Per provider: the total at current prices and with cfg["projection"]
    ["price_changes"] applied ({model, factor, from}), and the target (None = not known).
    """
    proj = cfg["projection"]
    prices = cfg["prices_usd_per_mtok"]
    reference = proj["reference"]
    changes = proj.get("price_changes") or []
    rows: list[dict] = []
    running: dict[str, float] = {}
    for item in proj["items"]:
        extra, extra_note = _extra_output(stats, item, reference)
        unit = unit_cost(stats, item["stat"], item["condition"], model=item["model"], prices=prices,
                         reference=reference, like=item.get("like"), calls_per_prefix=item.get("calls_per_prefix"),
                         extra_output_tokens=extra)
        calls = int(item["calls"])
        usd = None if unit["usd"] is None else unit["usd"] * calls
        optional = bool(item.get("optional", False))
        total = running.setdefault(item["provider"], 0.0)
        if usd is not None and not optional:
            total = running[item["provider"]] = total + usd
        rows.append({"id": item["id"], "group": item.get("group", ""), "provider": item["provider"],
                     "name": item["name"], "stat": item["stat"], "condition": item["condition"], "model": item["model"],
                     "calls": calls, "unit_usd": unit["usd"], "usd": usd, "optional": optional,
                     "running_total": None if optional else total, "note": item.get("note", ""),
                     "basis": _rescoped_basis(unit, stats, item, reference) + extra_note})
    totals: dict[str, dict] = {}
    for r in rows:
        t = totals.setdefault(r["provider"], {"usd": 0.0, "calls": 0, "missing": 0, "optional_usd": 0.0,
                                              "optional_missing": 0, "usd_after_price_changes": 0.0})
        if r["optional"]:
            if r["usd"] is None:
                t["optional_missing"] += 1
            else:
                t["optional_usd"] += r["usd"]
            continue
        t["calls"] += r["calls"]
        if r["usd"] is None:
            t["missing"] += 1
        else:
            t["usd"] += r["usd"]
            t["usd_after_price_changes"] += r["usd"] * _price_factor(changes, r["model"])
    for provider, t in totals.items():
        t["target_usd"] = (proj.get("targets_usd") or {}).get(provider)

    known = {r["id"] for r in rows}
    project_total = float(proj["anthropic_project_total_usd"])
    before = float(proj["anthropic_spent_before_usd"])
    remaining = project_total - before - float(anthropic_spent_this_session)
    anth = totals.get("anthropic", {"usd": 0.0, "missing": 0, "optional_usd": 0.0, "optional_missing": 0})
    margin = remaining - anth["usd"]
    required = proj.get("margin_for_mitigated_bo16_usd")
    optional_rows = [r for r in rows if r["provider"] == "anthropic" and r["optional"]]
    fits = None
    if required is not None and optional_rows and not anth["optional_missing"]:
        fits = margin - anth["optional_usd"] >= float(required)
    budget = {"project_total_usd": project_total, "spent_before_usd": before,
              "spent_this_session_usd": float(anthropic_spent_this_session), "remaining_usd": remaining,
              "projected_usd": anth["usd"], "missing_unit_costs": anth["missing"], "margin_usd": margin,
              "optional_items": [r["name"] for r in optional_rows], "optional_usd": anth["optional_usd"],
              "margin_after_optional_usd": margin - anth["optional_usd"], "required_margin_usd": required,
              "optional_fits": fits}

    cuts = []
    removed: set[str] = set()
    calls_now = {r["id"]: r["calls"] for r in rows}
    for cut in proj.get("cuts") or []:
        unknown = (set(cut.get("remove") or []) | set(cut.get("set_calls") or {})) - known
        if unknown:
            raise ValueError(f"projection cut {cut['id']}: unknown item ids {sorted(unknown)}")
        removed |= set(cut.get("remove") or [])
        calls_now.update({k: int(v) for k, v in (cut.get("set_calls") or {}).items()})
        total = sum(r["unit_usd"] * calls_now[r["id"]] for r in rows if r["provider"] == "anthropic"
                    and not r["optional"] and r["id"] not in removed and r["unit_usd"] is not None)
        cuts.append({"id": cut["id"], "name": cut["name"], "anthropic_usd": total, "margin_usd": remaining - total})
    return {"rows": rows, "totals": totals, "anthropic_budget": budget, "cuts": cuts, "price_changes": changes}


def rescoped_markdown(result: dict) -> str:
    """Four markdown tables: the items with running totals, the Anthropic budget lines, the cut order, and the
    OpenRouter and Google totals (current prices, with the configured price changes, against the targets)."""
    def money(x: float | None) -> str:
        return "n/a" if x is None else (f"{x:.4f}" if abs(x) < 0.1 else f"{x:.2f}")

    lines = ["| item | calls | USD per call | USD | running total | basis |", "|---|---|---|---|---|---|"]
    for r in result["rows"]:
        running = "not added (optional)" if r["optional"] else f"{r['provider']} {r['running_total']:.2f}"
        name = r["name"] + (" (optional)" if r["optional"] else "")
        lines.append(f"| {name} | {r['calls']:,} | {money(r['unit_usd'])} | {money(r['usd'])} | {running} | "
                     f"{r['basis']} |")
    b = result["anthropic_budget"]
    fits = {True: "yes", False: "no", None: "not decided (a cost or the threshold is missing)"}[b["optional_fits"]]
    lines += ["", "| Anthropic budget | USD |", "|---|---|",
              f"| project total | {b['project_total_usd']:.2f} |",
              f"| spent before this session | {b['spent_before_usd']:.2f} |",
              f"| spent this session | {b['spent_this_session_usd']:.2f} |",
              f"| remaining | {b['remaining_usd']:.2f} |",
              f"| projected (rows without a unit cost: {b['missing_unit_costs']}) | {b['projected_usd']:.2f} |",
              f"| margin = remaining - projected | {b['margin_usd']:.2f} |",
              f"| optional: {', '.join(b['optional_items']) or 'none'} | {b['optional_usd']:.2f} |",
              f"| margin after the optional items | {b['margin_after_optional_usd']:.2f} |",
              "| required margin for the optional items | " + ("not configured" if b["required_margin_usd"] is None
                                                               else f"{float(b['required_margin_usd']):.2f}") + " |",
              f"| optional items fit | {fits} |"]
    lines += ["", "| cut | applied cumulatively | Anthropic total after it | margin after it |", "|---|---|---|---|",
              f"| none | items as listed | {b['projected_usd']:.2f} | {b['margin_usd']:.2f} |"]
    lines += [f"| {c['id']} | {c['name']} | {c['anthropic_usd']:.2f} | {c['margin_usd']:.2f} |" for c in result["cuts"]]
    changes = "; ".join(f"{c['model']} x{float(c['factor']):g} from {c['from']}" for c in result["price_changes"])
    lines += ["", f"| provider | projected USD, current prices | with price changes ({changes or 'none'}) | "
              "rows without a unit cost | target USD |", "|---|---|---|---|---|"]
    for provider, t in sorted(result["totals"].items()):
        if provider == "anthropic":
            continue
        target = "not known" if t.get("target_usd") is None else f"{t['target_usd']:.2f}"
        lines.append(f"| {provider} | {t['usd']:.2f} | {t['usd_after_price_changes']:.2f} | {t['missing']} | "
                     f"{target} |")
    return "\n".join(lines) + "\n"

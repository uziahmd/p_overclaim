"""Cost tables for a run directory: spend from cost.jsonl and per-call judge cost from judge_calls.jsonl.

cost.jsonl has one record per uncached API call. Session 1 records carry only input_tokens and
output_tokens (session 1 used no prompt caching), so a missing token bucket counts as 0 tokens.
A null cost_usd (unpriced model) is never counted as 0 USD: it is left out of the sum and counted.
Judge records without cost_parts (session 1) have no cost split: their split is null ("n/a"),
never derived from cost_usd.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

COST_FILE = "cost.jsonl"
JUDGE_FILE = "judge_calls.jsonl"
TOKEN_BUCKETS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens",
                 "thought_tokens")
BUCKET_HEADERS = ("input (uncached)", "cache read", "cache write", "output", "thought")
PARTS = ("cache_write", "cache_read", "input", "output")   # cost_parts keys, in table order
NA = "n/a"


def _read_jsonl(path: Path) -> list[dict]:
    """Records of a JSONL file ([] if missing); a malformed line raises."""
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON line ({exc})") from exc
    return out


def _spend(recs: list[dict]) -> dict:
    """calls, summed token buckets (missing = 0), usd (sum of non-null cost_usd), n_unpriced (null cost_usd)."""
    priced = [float(r["cost_usd"]) for r in recs if r.get("cost_usd") is not None]
    return {"calls": len(recs), **{b: sum(int(r.get(b) or 0) for r in recs) for b in TOKEN_BUCKETS},
            "usd": math.fsum(priced), "n_unpriced": len(recs) - len(priced)}


def _parts(rec: dict) -> dict | None:
    """The record's cost_parts if it has all four parts with numbers, else None."""
    parts = rec.get("cost_parts")
    if not isinstance(parts, dict) or any(parts.get(p) is None for p in PARTS):
        return None
    return {p: float(parts[p]) for p in PARTS}


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _judge_per_call(judge_recs: list[dict]) -> list[dict]:
    """Mean cost and tokens per UNCACHED judge call (cached false), per (grader, condition).

    mean_cost_usd is over records with a cost_usd (n_unpriced without); mean_parts over records with
    complete cost_parts (n_with_parts); mean_tokens over all uncached records (usage bucket missing = 0).
    Cached records are not in any mean and are counted in n_cached_excluded.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in judge_recs:
        groups[(str(r["grader"]), str(r["condition"]))].append(r)
    rows = []
    for (grader, condition), recs in sorted(groups.items()):
        unc = [r for r in recs if not r.get("cached")]
        costs = [float(r["cost_usd"]) for r in unc if r.get("cost_usd") is not None]
        parts = [p for p in map(_parts, unc) if p is not None]
        rows.append({
            "grader": grader, "condition": condition, "n_calls": len(unc), "n_cached_excluded": len(recs) - len(unc),
            "mean_cost_usd": _mean(costs), "n_unpriced": len(unc) - len(costs),
            "mean_parts": {p: _mean([x[p] for x in parts]) for p in PARTS}, "n_with_parts": len(parts),
            "mean_tokens": {b: _mean([float((r.get("usage") or {}).get(b) or 0) for r in unc]) for b in TOKEN_BUCKETS},
        })
    return rows


def _md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    return lines + ["| " + " | ".join(r) + " |" for r in rows]


def _tok(n: float | None, digits: int = 0) -> str:
    return NA if n is None else f"{n:,.{digits}f}"


def _usd(x: float | None, digits: int) -> str:
    return NA if x is None else f"{x:.{digits}f}"


def cost_tables(run_dir: str | Path) -> tuple[dict, str]:
    """Spend per provider and per provider x model x purpose (cost.jsonl), and mean cost per uncached judge
    call per grader x condition split into cost_parts, with mean tokens per bucket (judge_calls.jsonl).

    Either file may be missing. Returns (data, markdown); nothing is written.
    """
    run = Path(run_dir)
    cost_recs = _read_jsonl(run / COST_FILE)
    judge_recs = _read_jsonl(run / JUDGE_FILE)

    by_provider: dict[str, list[dict]] = defaultdict(list)
    by_mp: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in cost_recs:
        by_provider[str(r.get("provider"))].append(r)
        by_mp[(str(r.get("provider")), str(r.get("model")), str(r.get("purpose")))].append(r)
    provider_rows = [{"provider": p, **_spend(recs)} for p, recs in sorted(by_provider.items())]
    mp_rows = [{"provider": p, "model": m, "purpose": u, **_spend(recs)} for (p, m, u), recs in sorted(by_mp.items())]
    total = _spend(cost_recs)
    judge_rows = _judge_per_call(judge_recs)

    def spend_cells(s: dict) -> list[str]:
        return [str(s["calls"]), *[_tok(s[b]) for b in TOKEN_BUCKETS], _usd(s["usd"], 3), str(s["n_unpriced"])]

    spend_head = ["calls", *BUCKET_HEADERS, "USD", "unpriced calls"]
    lines = ["## Spend by provider (cost.jsonl)", ""]
    if cost_recs:
        lines += _md_table(["provider", *spend_head],
                           [[r["provider"], *spend_cells(r)] for r in provider_rows] + [["total", *spend_cells(total)]])
        lines += ["", "## Spend by model and purpose (cost.jsonl)", ""]
        lines += _md_table(["provider", "model", "purpose", *spend_head],
                           [[r["provider"], r["model"], r["purpose"], *spend_cells(r)] for r in mp_rows])
    else:
        lines.append("No cost records.")
    lines += ["", "## Judge cost per uncached call (judge_calls.jsonl)", "",
              "Means over judge records with cached false; USD split from cost_parts (n/a for records without "
              "it); mean tokens per bucket.", ""]
    if judge_rows:
        lines += _md_table(
            ["grader", "condition", "uncached calls", "cached (excluded)", "mean USD", "cache write USD",
             "cache read USD", "input USD", "output USD", *[f"{h} tok" for h in BUCKET_HEADERS]],
            [[r["grader"], r["condition"], str(r["n_calls"]), str(r["n_cached_excluded"]),
              _usd(r["mean_cost_usd"], 5), *[_usd(r["mean_parts"][p], 5) for p in PARTS],
              *[_tok(r["mean_tokens"][b]) for b in TOKEN_BUCKETS]] for r in judge_rows])
    else:
        lines.append("No judge records.")

    data: dict[str, Any] = {"inputs": {COST_FILE: len(cost_recs), JUDGE_FILE: len(judge_recs)},
                            "by_provider": provider_rows, "by_model_purpose": mp_rows, "total": total,
                            "judge_per_call": judge_rows}
    return data, "\n".join(lines) + "\n"

"""The AI Studio cap of Uzair's memo of 2026-11-08 (section 1.3 and 1.4), shared by the pointwise_shown and pairwise
stages: the Gemini rows that Vertex refused run on the `aistudio` route under a hard cash cap
(budget.google_aistudio_cap_usd, counted by the CostLogger as a route cap), with a launch rule per row:

- the remaining cap is the cap less the spend on that route so far (cost.jsonl lines of provider google whose route
  is aistudio);
- a row's unit cost per call is the mean cost of its own live aistudio calls so far when at least MIN_LIVE exist
  (the records of that grader, condition and rubric with route aistudio and cached false), else the list-price unit of
  its projection row (results/full_run/projection.json, rows[].unit.usd), named per output file in the run's config;
- the row's remaining cost is the sum over its tasks of the calls still missing times the unit; it launches whole when
  that fits under the remaining cap, else on the longest prefix of its tasks, in the run's task order, whose missing
  calls fit (prefix_fit), recorded as partial with the n it ran; a prefix of zero tasks skips the row;
- after MIN_LIVE live calls of a scorer within a row, the rest of the row and the later aistudio rows of the invocation
  are re-priced from the measured unit and printed (Reprice).
Nothing here makes a call; the stages pass their records and the cost logger in.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

AISTUDIO = "aistudio"
PROVIDER = "google"            # the cost name of the gemini provider (pilot2.COST_PROVIDER)
CAP_KEY = "google_aistudio_cap_usd"
MIN_LIVE = 15                  # live calls of a scorer before its measured unit replaces the list price
PARTIAL = "partial"            # the status of a row cut to a prefix, or stopped short by the cap, a 402 or a quota 429
PROJECTION_FILE = "projection.json"


def on_route(record: dict, route: str = AISTUDIO) -> bool:
    """Whether a call record or cost line was served on that route (its `route` label, credential suffix ignored)."""
    label = record.get("route")
    return bool(label) and str(label).split(":", 1)[0] == route


def cap_state(cost: Any) -> dict:
    """{"cap", "spent", "remaining"} of the aistudio route from the cost logger (CostLogger.route_cap and
    spent_route). ValueError when the budget has no CAP_KEY."""
    cap = cost.route_cap(PROVIDER, AISTUDIO)
    if cap is None:
        raise ValueError(f"budget.{CAP_KEY} is not set: an aistudio Gemini run needs the cap (memo of 2026-11-08)")
    spent = cost.spent_route(PROVIDER, AISTUDIO)
    return {"cap": cap, "spent": spent, "remaining": cap - spent}


def projection_units(run_dir: Path) -> dict[str, float]:
    """{row id: list-price unit USD per call} of run_dir/projection.json (rows[].unit.usd). ValueError when the file is
    missing or a row has no unit."""
    path = Path(run_dir) / PROJECTION_FILE
    if not path.exists():
        raise ValueError(f"{path} is missing: the aistudio launch rule prices a row from its projection row")
    projection = json.loads(path.read_text(encoding="utf-8"))
    units: dict[str, float] = {}
    for row in projection.get("rows") or []:
        unit = (row.get("unit") or {}).get("usd")
        if isinstance(unit, (int, float)) and not isinstance(unit, bool):
            units[str(row["id"])] = float(unit)
    return units


def live_costs(records: list[dict], keep: Callable[[dict], bool]) -> list[float]:
    """The cost_usd of the records keep() accepts that were live on the aistudio route (cached false, a cost)."""
    return [float(r["cost_usd"]) for r in records
            if keep(r) and on_route(r) and not r.get("cached") and r.get("cost_usd") is not None]


def unit_for(records: list[dict], keep: Callable[[dict], bool], units: dict[str, float], row_id: str,
             min_live: int = MIN_LIVE) -> dict:
    """The unit USD per call of one scorer: {"usd", "basis", "n_live", "row"}: measured (mean of its live aistudio
    calls) when at least min_live exist, else the projection row's list price. ValueError when the row is unpriced."""
    costs = live_costs(records, keep)
    if len(costs) >= min_live:
        return {"usd": math.fsum(costs) / len(costs), "n_live": len(costs), "row": row_id,
                "basis": f"measured: mean of {len(costs)} live aistudio calls"}
    if row_id not in units:
        raise ValueError(f"projection row {row_id!r} has no unit price in {PROJECTION_FILE}")
    return {"usd": units[row_id], "n_live": len(costs), "row": row_id,
            "basis": f"list price of projection row {row_id} ({len(costs)} live aistudio calls so far, under "
                     f"{min_live})"}


def prefix_fit(per_task: list[tuple[str, float]], remaining: float) -> dict:
    """The longest prefix of per_task ([(task id, USD of its missing calls)], the run's task order) whose costs sum to
    at most `remaining`: {"n_tasks", "n_fit", "cost_fit", "cost_all", "fits"} (fits: every task fits)."""
    acc, n_fit = 0.0, 0
    for _, usd in per_task:
        if acc + usd > remaining + 1e-12:
            break
        acc += usd
        n_fit += 1
    total = math.fsum(usd for _, usd in per_task)
    return {"n_tasks": len(per_task), "n_fit": n_fit, "cost_fit": acc, "cost_all": total,
            "fits": n_fit == len(per_task)}


def plan_text(plan: dict) -> str:
    """One line of a launch plan: the cap state, the units and the fit."""
    units = "; ".join(f"{name} {u['usd']:.5f} USD/call ({u['basis']})" for name, u in plan["units"].items())
    fit = plan["fit"]
    verdict = ("fits" if fit["fits"] else f"does NOT fit: the first {fit['n_fit']} of {fit['n_tasks']} tasks fit "
               f"({fit['cost_fit']:.2f} USD)")
    return (f"aistudio cap {plan['cap']:.2f} USD, spent {plan['spent']:.4f}, remaining {plan['remaining']:.4f}; "
            f"units: {units}; remaining calls {plan['remaining_calls']} at {fit['cost_all']:.2f} USD: {verdict}")


def cap_line(cost: Any) -> str:
    """"aistudio: spent X of the Y USD cap, remaining Z" for the console after each row (memo section 2)."""
    state = cap_state(cost)
    return (f"aistudio: spent {state['spent']:.4f} of the {state['cap']:.2f} USD cap, remaining "
            f"{state['remaining']:.4f}")


class Reprice:
    """Counts a row's live calls per scorer as its tasks finish and, when a scorer first reaches min_live live calls,
    prints the rest of the row and the later aistudio rows re-priced from the measured unit (memo section 1.4).
    `later` is [(row id, scorer name, remaining calls, list unit USD, same_scorer: bool)]; a later row of the same
    scorer takes the measured unit, the others keep their list price until they measure their own."""

    def __init__(self, run_id: str, remaining_calls: dict[str, int], later: list[tuple], cost: Any,
                 min_live: int = MIN_LIVE, out: Callable[[str], None] = print) -> None:
        self.run_id, self.remaining, self.later, self.cost = run_id, dict(remaining_calls), later, cost
        self.min_live, self.out = min_live, out
        self.live: dict[str, list[float]] = {name: [] for name in remaining_calls}
        self.printed: set[str] = set()
        self._lock = threading.Lock()

    def task_done(self, records: dict[str, list[dict]]) -> None:
        """Record one task's records ({scorer name: records}); print the re-price when a scorer reaches min_live."""
        with self._lock:
            for name, recs in records.items():
                if name not in self.live:
                    continue
                self.live[name] += [float(r["cost_usd"]) for r in recs
                                    if not r.get("cached") and r.get("cost_usd") is not None]
                self.remaining[name] = max(0, self.remaining.get(name, 0) - len(recs))
                if len(self.live[name]) >= self.min_live and name not in self.printed:
                    self.printed.add(name)
                    self.out(self.line(name))

    def line(self, name: str) -> str:
        unit = math.fsum(self.live[name]) / len(self.live[name])
        rest = self.remaining[name] * unit
        later = []
        for row_id, scorer, calls, list_unit, same in self.later:
            u = unit if same and scorer == name else list_unit
            later.append(f"{row_id}/{scorer} {calls} calls x {u:.5f} = {calls * u:.2f} USD"
                         + (" (measured)" if same and scorer == name else " (list)"))
        return (f"[{self.run_id}] aistudio re-price after {len(self.live[name])} live calls of {name}: measured "
                f"{unit:.5f} USD/call; rest of this row {self.remaining[name]} calls = {rest:.2f} USD; later rows: "
                + ("; ".join(later) if later else "none") + f"; {cap_line(self.cost)}")

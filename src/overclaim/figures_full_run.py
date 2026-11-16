"""Session 5, Part 2 figures (contract 6H, section 2): one figure per file, PNG and SVG, into results/full_run/figures/.

`write_figures(data, out_dir)` draws from the section rows that analysis_full_run.summarize_full_run computed (its
`data` dict, before it is written as JSON) and returns {"written": [file names], "skipped": {figure: reason}}. A figure
whose inputs are missing is skipped with the reason; every figure is skipped when matplotlib is not installed (it is not
among pyproject.toml's dependencies). Matplotlib only: no seaborn and no pyplot state (matplotlib.figure.Figure, saved
through the Agg and SVG canvases, a non-interactive backend), the default font. Colours follow the entity in a fixed
order (the categorical order of the dataviz reference palette); text is neutral ink; one value axis per panel; every
point carries its 95 percent task-bootstrap CI as an error bar.

Figures (FIGURES):
- judge_premium: the hidden and shown premium per judge (neutral rubric, shown on the shown-eligible tasks) with the tax
  beside it, and the reward models' standardised premium in their own panel.
- e2_selection: the clean-overclaim rate of the selected sample at N = 1, 4, 16 per selector (LLM selectors, mitigated
  ones included, in one panel; the reward models, by the curve's kind, in a second panel).
- pairwise_vs_pointwise: the pairwise O vs H rate next to the premium per judge and condition (appendix).
- rm_size: the reward models' standardised premium against the parameter count, log x axis (E5).
- dose_response: the mean score per level O+, O, O-, H, P per judge, and the reward models' level minus H (E6).
- e7_ablation: the shown premium of the plain, instruction-only and verification-first scorers on the E7 tasks (E7).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from overclaim.analysis_v4 import _ok

logger = logging.getLogger(__name__)

FIGURES = ("judge_premium", "e2_selection", "pairwise_vs_pointwise", "rm_size", "dose_response", "e7_ablation")
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd8"
CONDITION_COLOURS = {"hidden": SERIES[0], "shown": SERIES[1]}
SCORER_COLOURS = {"plain": SERIES[0], "instruction_only": SERIES[1], "verify_first": SERIES[2]}
SCORER_NAMES = {"plain": "plain", "instruction_only": "instruction-only", "verify_first": "verification-first"}
LEVELS = ("O_plus", "O", "O_minus", "H", "P")
LEVEL_NAMES = ("O+", "O", "O-", "H", "P")
NEUTRAL = "pointwise-neutral"
NS = (1, 4, 16)


def _err(value: Any, ci: Any) -> tuple[float, float] | None:
    """(distance to the lower bound, distance to the upper bound) of a CI, None when undefined."""
    if ci is None or not _ok(value, *ci):
        return None
    return max(value - ci[0], 0.0), max(ci[1] - value, 0.0)


def _style(ax: Any, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _point(ax: Any, x: float, y: float, xerr: Any = None, yerr: Any = None, *, colour: str, label: str | None = None,
           hollow: bool = False) -> None:
    """One point with its CI bar (xerr / yerr: a (low, high) pair or None)."""
    ax.errorbar([x], [y], xerr=None if xerr is None else [[xerr[0]], [xerr[1]]],
                yerr=None if yerr is None else [[yerr[0]], [yerr[1]]], fmt="o", markersize=6, color=colour,
                ecolor=colour, elinewidth=1.5, capsize=0, markerfacecolor="white" if hollow else colour, label=label)


def _hdot(ax: Any, labels: list[str], points: dict[str, list[tuple[int, float, Any]]], colours: dict[str, str],
          ref: float = 0.0) -> None:
    """Horizontal dot plot: per series name, (row index, value, CI) points offset around each row; a reference line at
    `ref`; row labels on the y axis."""
    step = 0.6 / max(len(points), 1)
    for k, (name, pts) in enumerate(points.items()):
        off = (k - (len(points) - 1) / 2) * step
        for j, (i, v, ci) in enumerate(pts):
            _point(ax, v, i + off, xerr=_err(v, ci), colour=colours[name], label=name if j == 0 else None)
    ax.axvline(ref, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()


def judge_premium(data: dict, figure: Callable) -> Any:
    """Premium and tax per judge, hidden and shown side by side; reward models' standardised premium."""
    core = [r for r in data.get("core") or [] if r["kind"] == "judge" and r["rubric"] == NEUTRAL
            and r["condition"] in CONDITION_COLOURS and _ok(r["premium"])]
    rms = [r for r in data.get("core") or [] if r["kind"] == "reward_model" and r["condition"] == "hidden"
           and _ok(r["premium"], r["pooled_sd"]) and r["pooled_sd"] > 0]
    if not core and not rms:
        return "no pointwise premium yet (pointwise_calls.jsonl and rm_scores.jsonl not run yet)"
    fig = figure(figsize=(12, 0.45 * max(len({r["grader"] for r in core}), len(rms), 4) + 1.6), layout="constrained")
    axes = fig.subplots(1, 3, gridspec_kw={"width_ratios": [2, 2, 1.6]})
    graders = sorted({r["grader"] for r in core})
    for ax, stat, title in ((axes[0], "premium", "Premium O - H"), (axes[1], "tax", "Tax P - H")):
        points = {c: [(graders.index(r["grader"]), r[stat], r[f"{stat}_ci"]) for r in core
                      if r["condition"] == c and _ok(r[stat])] for c in CONDITION_COLOURS}
        _hdot(ax, graders, {c: p for c, p in points.items() if p}, CONDITION_COLOURS)
        _style(ax, f"{title}, judges (neutral rubric)", "score points (1 to 10 scale)")
        if any(points.values()):
            ax.legend(frameon=False, fontsize=8)
    axes[1].set_yticklabels([])
    names = [r["grader"] for r in rms]
    pts = [(i, r["premium"] / r["pooled_sd"], [x / r["pooled_sd"] for x in r["premium_ci"]]
            if _ok(*r["premium_ci"]) else None) for i, r in enumerate(rms)]
    _hdot(axes[2], names, {"hidden": pts} if pts else {}, CONDITION_COLOURS)
    _style(axes[2], "Premium O - H, reward models", "premium / SD of the model's scores")
    return fig


def e2_selection(data: dict, figure: Callable) -> Any:
    """Clean-overclaim rate at N = 1, 4, 16 per selector with CIs."""
    curves = [c for c in data.get("e2_curves") or [] if any(_ok((c["rates"].get(n) or c["rates"].get(str(n)) or {})
                                                                 .get("mean")) for n in NS)]
    if not curves:
        return "no Experiment 2 selection rows yet (e2_scores.jsonl / e2_table.json not run yet)"
    panels = [("LLM selectors", [c for c in curves if c.get("kind") != "reward_model"]),
              ("Reward-model selectors", [c for c in curves if c.get("kind") == "reward_model"])]
    panels = [p for p in panels if p[1]]
    fig = figure(figsize=(5.2 * len(panels), 4.2), layout="constrained")
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    for ax, (title, cs) in zip(axes, panels):
        for k, c in enumerate(sorted(cs, key=lambda c: c["selector"])[:len(SERIES)]):
            colour = SERIES[k]
            cells = [(n, c["rates"].get(n) or c["rates"].get(str(n)) or {}) for n in NS]
            xs = [n for n, d in cells if _ok(d.get("mean"))]
            ax.plot(xs, [d["mean"] for n, d in cells if _ok(d.get("mean"))], color=colour, linewidth=2,
                    label=c["selector"])
            for n, d in cells:
                if _ok(d.get("mean")):
                    _point(ax, n, d["mean"], yerr=_err(d["mean"], d.get("ci")), colour=colour)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(NS))
        ax.set_xticklabels([str(n) for n in NS])
        ax.set_ylim(0, 1)
        _style(ax, f"{title}: best of N", "N (samples the selector chooses from)", "clean-overclaim rate")
        ax.legend(frameon=False, fontsize=8)
    return fig


def pairwise_vs_pointwise(data: dict, figure: Callable) -> Any:
    """Pairwise O vs H rate next to the pointwise premium per judge and condition."""
    rows = [r for r in data.get("pairwise") or [] if r.get("table") == "agreement" and r.get("kind") == "judge"
            and _ok(r.get("pairwise_rate"), r.get("premium"))]
    if not rows:
        return "no judge with both a pairwise and a pointwise row yet (judge_calls.jsonl not run yet)"
    labels = sorted({r["grader"] for r in rows})
    fig = figure(figsize=(10, 0.45 * max(len(labels), 4) + 1.6), layout="constrained")
    axes = fig.subplots(1, 2)
    for ax, (value, ci, ref, title, xlabel) in zip(axes, (
            ("pairwise_rate", "pairwise_ci", 0.5, "Pairwise: P(O over H)", "rate (ties 0.5)"),
            ("premium", "premium_ci", 0.0, "Pointwise: premium O - H", "score points"))):
        points = {c: [(labels.index(r["grader"]), r[value], r[ci]) for r in rows if r["condition"] == c]
                  for c in CONDITION_COLOURS}
        _hdot(ax, labels, {c: p for c, p in points.items() if p}, CONDITION_COLOURS, ref=ref)
        _style(ax, title, xlabel)
        ax.legend(frameon=False, fontsize=8)
    axes[1].set_yticklabels([])
    return fig


def rm_size(data: dict, figure: Callable) -> Any:
    """Standardised premium against the parameter count (log x)."""
    rows = [r for r in data.get("e5") or [] if r.get("table") == "premium" and r.get("params_b")
            and _ok(r.get("standardised_gap"))]
    if not rows:
        return "no reward-model premium with a known parameter count yet (rm_scores.jsonl not run yet)"
    fig = figure(figsize=(6, 4.2), layout="constrained")
    ax = fig.subplots()
    for r in rows:
        _point(ax, r["params_b"], r["standardised_gap"], yerr=_err(r["standardised_gap"], r["standardised_ci"]),
               colour=SERIES[0], hollow=bool(r.get("quantization")))
        ax.annotate(r["grader"] + (" (4-bit)" if r.get("quantization") else ""), (r["params_b"],
                    r["standardised_gap"]), textcoords="offset points", xytext=(6, 4), fontsize=7, color=MUTED)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xscale("log")
    _style(ax, "Reward models: premium against size (E5)", "parameters (billions, log scale)",
           "premium / SD of the model's scores")
    return fig


def dose_response(data: dict, figure: Callable) -> Any:
    """Mean score per level per judge; the reward models' premium of each level over H."""
    rows = [r for r in data.get("e6") or [] if any(r["levels"].get(lv) for lv in ("O_plus", "O_minus"))]
    if not rows:
        return "no E6 variant scored yet (O_plus / O_minus records not run yet)"
    judges = [r for r in rows if r["kind"] == "judge"]
    rms = [r for r in rows if r["kind"] == "reward_model"]
    panels = [p for p in (("Judges: mean score per level", judges, "levels"),
                          ("Reward models: level minus H", rms, "over_h")) if p[1]]
    fig = figure(figsize=(5.6 * len(panels), 4.4), layout="constrained")
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    for ax, (title, rs, field) in zip(axes, panels):
        for k, r in enumerate(sorted(rs, key=lambda r: r["grader"])[:len(SERIES)]):
            pts = []
            for i, lv in enumerate(LEVELS):
                d = r[field].get(lv) if field == "levels" else ({"value": 0.0, "ci": [0.0, 0.0]} if lv == "H"
                                                                  else r[field].get(lv))
                v = None if d is None else d.get("mean", d.get("value"))
                if _ok(v):
                    pts.append((i, v, d.get("ci")))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES[k], linewidth=2, label=r["grader"])
            for i, v, ci in pts:
                _point(ax, i, v, yerr=_err(v, ci), colour=SERIES[k])
        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels(LEVEL_NAMES)
        _style(ax, title + " (E6, hidden)", "level", "score" if field == "levels" else "score units")
        ax.legend(frameon=False, fontsize=7)
    return fig


def e7_ablation(data: dict, figure: Callable) -> Any:
    """Shown premium of the three scorers per grader on the E7 tasks."""
    rows = [r for r in data.get("e7") or [] if any(d and _ok(d.get("premium")) for d in r["scorers"].values())]
    if not rows:
        return "no E7 scorer row yet (e7_calls.jsonl not run yet)"
    labels = [r["grader"] for r in rows]
    fig = figure(figsize=(7, 0.6 * max(len(labels), 3) + 1.8), layout="constrained")
    ax = fig.subplots()
    points = {SCORER_NAMES[k]: [(i, r["scorers"][k]["premium"], r["scorers"][k]["ci"]) for i, r in enumerate(rows)
                                if r["scorers"].get(k) and _ok(r["scorers"][k]["premium"])] for k in SCORER_NAMES}
    _hdot(ax, labels, {k: p for k, p in points.items() if p},
          {SCORER_NAMES[k]: c for k, c in SCORER_COLOURS.items()})
    _style(ax, "E7: shown premium O - H by scorer (E7 tasks)", "score points")
    ax.legend(frameon=False, fontsize=8)
    return fig


BUILDERS: dict[str, Callable[[dict, Callable], Any]] = {
    "judge_premium": judge_premium, "e2_selection": e2_selection, "pairwise_vs_pointwise": pairwise_vs_pointwise,
    "rm_size": rm_size, "dose_response": dose_response, "e7_ablation": e7_ablation}


def write_figures(data: dict, out_dir: str | Path) -> dict:
    """Draw every figure of FIGURES from `data` and save <name>.png (150 dpi) and <name>.svg into out_dir (created when
    a figure is written). Returns {"written": [file names], "skipped": {figure: reason}}; nothing raises for a missing
    input or a missing matplotlib."""
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:
        reason = (f"matplotlib is not installed ({exc}); it is not among pyproject.toml's dependencies, so no figure "
                  "was drawn")
        logger.warning("figures: %s", reason)
        return {"written": [], "skipped": {name: reason for name in FIGURES}}
    out = Path(out_dir)
    written: list[str] = []
    skipped: dict[str, str] = {}
    for name in FIGURES:
        fig = BUILDERS[name](data, Figure)
        if isinstance(fig, str):
            skipped[name] = fig
            continue
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.png", dpi=150, facecolor="white")
        fig.savefig(out / f"{name}.svg", facecolor="white", metadata={"Date": None})
        written += [f"{name}.png", f"{name}.svg"]
    logger.info("figures: wrote %s; skipped %s", written or "none", sorted(skipped) or "none")
    return {"written": written, "skipped": skipped}

"""Shared readers and formatters for the paper scripts (make_paper_tables.py, make_paper_figures.py,
check_paper_numbers.py). Everything is read from committed files; nothing here writes.

Sources (all at the commit named in paper/numbers_manifest.csv):
- results/full_run/summary.json: the analysis stage's rows (core, hypotheses, holm_family, e2, e2_curves, e5, e6, e7,
  filler_dose, sensitivity, pairwise, splits, e4, kappa, spend, tie_rule, header).
- results/full_run/runs_status.json: the status, record counts and API route of every run.
- data/processed/full_run_tasks.json and data/processed/coding/matched_sets_v4_full_run.jsonl: the frozen task list
  and sets (counts, the Figure 1 example).
- configs/full_run.yaml, configs/models.yaml: design constants.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "results" / "full_run"
SUMMARY = RUN_DIR / "summary.json"
RUNS_STATUS = RUN_DIR / "runs_status.json"
FROZEN_DOC = REPO / "data" / "processed" / "full_run_tasks.json"
FROZEN_SETS = REPO / "data" / "processed" / "coding" / "matched_sets_v4_full_run.jsonl"
CONFIG = REPO / "configs" / "full_run.yaml"
MODELS = REPO / "configs" / "models.yaml"
PAPER = REPO / "paper"

NEUTRAL = "pointwise-neutral"

# Display names (section 2.2 of the writing brief: graders by model name, never by run id).
NAMES = {
    "claude-sonnet-5-5": "Sonnet 5.5", "claude-opus-5-5": "Opus 5.5", "claude-sonnet-4-6": "Sonnet 4.6",
    "claude-opus-4-6": "Opus 4.6", "gemini-3.8-flash": "Gemini 3.8 Flash", "gemini-4-argon": "Gemini 4 Argon",
    "deepseek-v4-pro": "DeepSeek V4 Pro", "gpt-oss-120b": "gpt-oss-120b", "qwen3-32b": "Qwen3-32B",
    "skywork-v2-qwen3-0.6b": "Skywork 0.6B", "skywork-v2-qwen3-1.7b": "Skywork 1.7B",
    "skywork-v2-llama3.2-3b": "Skywork 3B", "skywork-v2-qwen3-4b-4bit": "Skywork 4B",
    "skywork-v2-qwen3-8b-4bit": "Skywork 8B",       # both ran with 4-bit weights (the graders table says so)
    "claude-sonnet-5-5+verify_first": "Sonnet 5.5, verification step",
    "deepseek-v4-pro+verify_first": "DeepSeek V4 Pro, verification step",
}
LAB = {"claude-sonnet-5-5": "Anthropic", "claude-opus-5-5": "Anthropic", "claude-sonnet-4-6": "Anthropic",
       "claude-opus-4-6": "Anthropic", "gemini-3.8-flash": "Google", "gemini-4-argon": "Google",
       "deepseek-v4-pro": "DeepSeek", "gpt-oss-120b": "OpenAI", "qwen3-32b": "Qwen"}
# Figure 2's row order: labs in this order, judges within a lab ordered by hidden premium at draw time.
LAB_ORDER = ("Anthropic", "Google", "DeepSeek", "OpenAI", "Qwen")
LLM_JUDGES = ("claude-sonnet-5-5", "claude-opus-5-5", "claude-sonnet-4-6", "claude-opus-4-6", "gemini-3.8-flash",
              "gemini-4-argon", "deepseek-v4-pro", "gpt-oss-120b", "qwen3-32b")
REWARD_MODELS = ("skywork-v2-qwen3-0.6b", "skywork-v2-qwen3-1.7b", "skywork-v2-llama3.2-3b",
                 "skywork-v2-qwen3-4b-4bit", "skywork-v2-qwen3-8b-4bit")
GAP_NAMES = {"no_tests_run": "no test run", "tests_failed": "last test run failed",
             "tests_narrowed": "narrower rerun after a failure"}
# The run of each (grader, condition) cell of the pointwise and pairwise rows (runs_status.json keys).
POINTWISE_RUNS = {("claude-sonnet-5-5", "hidden"): "pwh-s55", ("claude-opus-5-5", "hidden"): "pwh-o55",
                  ("claude-sonnet-4-6", "hidden"): "pwh-s46", ("claude-opus-4-6", "hidden"): "pwh-o46",
                  ("gemini-3.8-flash", "hidden"): "pwh-gemf", ("gemini-4-argon", "hidden"): "pwh-gema",
                  ("deepseek-v4-pro", "hidden"): "pwh-dsv4", ("gpt-oss-120b", "hidden"): "pwh-oss",
                  ("qwen3-32b", "hidden"): "pwh-qwen",
                  ("claude-sonnet-5-5", "shown"): "pws-s55", ("claude-opus-5-5", "shown"): "pws-o55",
                  ("gemini-3.8-flash", "shown"): "pws-gemf", ("gemini-4-argon", "shown"): "pws-gema",
                  ("deepseek-v4-pro", "shown"): "pws-dsv4", ("gpt-oss-120b", "shown"): "pws-oss",
                  ("qwen3-32b", "shown"): "pws-qwen", ("deepseek-v4-pro", "shown, band"): "pws-dsv4-band"}
PAIRWISE_RUNS = {("claude-sonnet-5-5", "hidden"): "ph-s55", ("claude-opus-5-5", "hidden"): "ph-o55",
                 ("claude-sonnet-4-6", "hidden"): "ph-s46", ("claude-opus-4-6", "hidden"): "ph-o46",
                 ("gemini-3.8-flash", "hidden"): "ph-gemf", ("gemini-4-argon", "hidden"): "ph-gema",
                 ("deepseek-v4-pro", "hidden"): "ph-dsv4", ("gpt-oss-120b", "hidden"): "ph-oss",
                 ("qwen3-32b", "hidden"): "ph-qwen", ("claude-sonnet-5-5", "shown"): "ps-s55"}
ROUTE_NAMES = {"vertex": "Vertex AI", "aistudio": "AI Studio API"}
MIN_TASKS = 5   # a premium on fewer complete tasks is not estimated
DERIVED = PAPER / "derived.json"   # written by scripts/make_paper_derived.py from the records


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def summary() -> dict:
    return load_json(SUMMARY)


def runs_status() -> dict:
    return load_json(RUNS_STATUS)


def frozen_doc() -> dict:
    return load_json(FROZEN_DOC)


def frozen_sets() -> list[dict]:
    return load_jsonl(FROZEN_SETS)


def derived() -> dict:
    return load_json(DERIVED)


def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def load_resolved_config() -> dict:
    """The configuration as resolved when the full run started (results/full_run/config.yaml)."""
    return yaml.safe_load((RUN_DIR / "config.yaml").read_text(encoding="utf-8"))


def models_config() -> dict:
    return yaml.safe_load(MODELS.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- row finders


def rows_where(rows: list[dict], **conds: Any) -> list[dict]:
    """The rows whose every named field equals the given value (a missing field never matches)."""
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in conds.items()):
            out.append(r)
    return out


def one(rows: list[dict], **conds: Any) -> dict:
    """Exactly one matching row, else ValueError naming the condition."""
    found = rows_where(rows, **conds)
    if len(found) != 1:
        raise ValueError(f"expected one row for {conds}, found {len(found)}")
    return found[0]


def core_row(s: dict, grader: str, condition: str, rubric: str = NEUTRAL) -> dict | None:
    rows = rows_where(s["core"], grader=grader, condition=condition)
    rows = [r for r in rows if r.get("rubric") == rubric or r.get("kind") == "reward_model"]
    return rows[0] if rows else None


def holm_rows(s: dict) -> list[dict]:
    return list(s["holm_family"])


def hypothesis_rows(s: dict, hypothesis: str) -> list[dict]:
    return rows_where(s["hypotheses"], hypothesis=hypothesis)


def e2_row(s: dict, selector: str, outcome: str = "clean_overclaim", labeller: str = "session3",
           selection: str = "e2_tasks") -> dict | None:
    """The Experiment 2 row of a selector: the records-based row of the 40 tasks for the LLM selectors (source
    'records', labeller 'session3'), the e2_table row otherwise (reward models; the 26-task selection)."""
    rows = rows_where(s["e2"], selector=selector, outcome=outcome)
    if selection == "e2_tasks":
        recs = [r for r in rows if r.get("source") == "records" and r.get("labeller") == labeller]
        if recs:
            return recs[0]
        tab = [r for r in rows if r.get("source") == "e2_table.json" and r.get("labeller") == f"e2_tasks/{labeller}"]
        return tab[0] if tab else None
    tab = [r for r in rows if r.get("source") == "e2_table.json" and r.get("labeller") == f"{selection}/{labeller}"]
    return tab[0] if tab else None


def e2_rates(row: dict) -> dict:
    """The tie-averaged rates of an Experiment 2 row ({N: {"mean", "ci", "n_tasks"}}) and its N = 16 minus N = 1 diff."""
    ta = row["tie_averaged"]
    return {"rates": ta["rates"], "diff": ta["diff"], "n_tasks": ta["n_tasks"]}


def pairwise_row(s: dict, grader: str, condition: str = "hidden") -> dict | None:
    rows = [r for r in s["pairwise"] if r.get("table") == "judge" and r.get("grader") == grader
            and r.get("condition") == condition]
    return rows[0] if rows else None


def flip_rate(s: dict, grader: str, pair: str, condition: str = "hidden") -> dict | None:
    rows = [r for r in s["pairwise"] if r.get("table") == "position_bias" and r.get("grader") == grader
            and r.get("pair") == pair and r.get("condition") == condition]
    return rows[0] if rows else None


def reasons_row(s: dict, grader: str, condition: str = "hidden") -> dict | None:
    rows = [r for r in s["pairwise"] if r.get("table") == "reasons" and r.get("grader") == grader
            and r.get("condition") == condition]
    return rows[0] if rows else None


def cell_status(status: dict, run_id: str | None, expected_file: str = "pointwise_calls.jsonl") -> dict:
    """{"status", "records", "expected", "n_tasks", "route", "text"} of a run from runs_status.json; a run without an
    entry is "not run"."""
    if run_id is None or run_id not in status:
        return {"status": "not run", "records": 0, "expected": None, "n_tasks": None, "route": None,
                "text": "not run"}
    e = status[run_id]
    records = e.get("records")
    expected = e.get("expected_records")
    n_tasks = e.get("n_tasks")
    if isinstance(records, dict):
        records = records.get(expected_file, 0)
    if isinstance(expected, dict):
        expected = expected.get(expected_file)
    if isinstance(n_tasks, dict):
        n_tasks = n_tasks.get(expected_file)
    route = e.get("route")
    st = e.get("status")
    if st == "done":
        text = "complete"
    elif st in ("partial", "failed", "stopped", "skipped"):
        text = f"partial, {records} of {expected} records"
    else:
        text = str(st)
    return {"status": st, "records": records, "expected": expected, "n_tasks": n_tasks, "route": route, "text": text}


# ---------------------------------------------------------------- formatting


def fmt(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "n/a"
    s = f"{x:.{digits}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]      # no "-0.00"
    return s


def ci(c: list | None, digits: int = 2) -> str:
    if not c:
        return ""
    return f"[{fmt(c[0], digits)}, {fmt(c[1], digits)}]"


def est(x: float | None, c: list | None, digits: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{fmt(x, digits)} {ci(c, digits)}".strip()


def half_up(x: float) -> int:
    """x rounded to a whole number, halves away from zero (62.5 gives 63, the way a reader rounds; Python's round and
    its %.0f format give 62). scripts/check_paper_numbers.py rounds the same way."""
    return int(math.floor(abs(x) + 0.5)) * (-1 if x < 0 else 1)


def pct(x: float | None) -> str:
    """A rate as a whole-number percentage (79%)."""
    if x is None:
        return "n/a"
    return f"{half_up(x * 100):d}\\%"


def pct_plain(x: float | None) -> str:
    return "n/a" if x is None else f"{half_up(x * 100):d}%"


def p_text(p: float | None, is_floor: bool, floor: float, bound: bool = False) -> str:
    if p is None:
        return "n/a"
    if is_floor:
        return f"$<$ {fmt(floor, 3)}"
    return ("$<$ " if bound else "") + fmt(p, 3)


def tex_escape(text: str) -> str:
    return (text.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#").replace("$", "\\$"))


def kappa_value(s: dict) -> float:
    m = re.search(r"kappa ([0-9.]+)", s["kappa"][0]["kappa_text"])
    if not m:
        raise ValueError("kappa not found in summary.kappa")
    return float(m.group(1))

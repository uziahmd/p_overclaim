"""Shared paths, names and readers of the sprint's control run (declared on 2026-11-13; sprint_control/README.md).

Nothing here calls a model, and nothing here writes outside sprint_control/ and results/sprint_control/. The frozen
sets, the main run's records and the tables of reanalysis/ are read only.

Message text (the agent's messages, the edited messages, the judges' reasons) is untrusted data: it is compared,
sliced and written back out as text, and never executed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "sprint_control"
OUT = REPO / "results" / "sprint_control"
MAIN_RUN = REPO / "results" / "full_run"
SETS_FILE = REPO / "data" / "processed" / "coding" / "matched_sets_v4_full_run.jsonl"
TASKS_DOC = REPO / "data" / "processed" / "full_run_tasks.json"
MAIN_CONFIG = REPO / "configs" / "full_run.yaml"
MODELS_CONFIG = REPO / "configs" / "models.yaml"
MAIN_CACHE = REPO / "cache"                     # the main run's disk cache: looked up, never written by this run
CONFIG = HERE / "config.yaml"
MESSAGES_FILE = HERE / "messages.jsonl"         # the two new messages of every set (build_messages.py)
LEFT_OUT_FILE = HERE / "left_out.json"          # the sets that could not be built under the rules, with the reason
EXAMPLES_FILE = HERE / "examples.md"
ESTIMATE_FILE = HERE / "cost_estimate.md"
CACHE_DIR = OUT / "cache"                       # this run's own disk cache (ignored by git: the pattern "cache/")
POINTWISE_FILE = OUT / "pointwise_calls.jsonl"
RM_SCORES_FILE = OUT / "rm_scores.jsonl"
RM_STATUS_FILE = OUT / "rm_status.json"
RUNS_STATUS_FILE = OUT / "runs_status.json"
VERTEX_CHECK_FILE = OUT / "vertex_check.json"
PROMPT_CHECK_FILE = OUT / "prompt_check.json"

for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# reanalysis/scripts is not put on the path here: its helper module is also called "common". analyze.py adds it.
REANALYSIS_SCRIPTS = REPO / "reanalysis" / "scripts"
REANALYSIS_TABLES = REPO / "reanalysis" / "tables"

RUN_NAME = "sprint_control"
SET_NAME = "v4_full_run"                        # the set name of the main run's records, kept for the new ones
CONDITION = "hidden"                            # without the transcript, each message scored alone
# The paper's fixed hedge (overclaim.repair_v4.O_MINUS_SENTENCE; build_messages.py asserts they are equal).
HEDGE = "I expect this to resolve the issue, but I have not confirmed it."
# The versions of a message. O, H, P and O_minus were scored by the main run; C, G and the second score of P by this
# run. "P2" names the second placebo score in the analysis; in the records it is variant "P" of run sprint_control.
CLAIM_ALONE, HEDGE_ALONE, PLACEBO = "C", "G", "P"
NEW_VARIANTS = (CLAIM_ALONE, HEDGE_ALONE, PLACEBO)
KINDS = ("plain statement", "list marker", "bare")
CLAUDE_JUDGES = ("claude-sonnet-5-5", "claude-opus-5-5", "claude-sonnet-4-6", "claude-opus-4-6")
N_BOOT, SEED, ALPHA = 2000, 0, 0.05


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def frozen_sets() -> list[dict]:
    """The 60 frozen sets of the main run, in the frozen order. ValueError when the file's sha256 is not the one the
    frozen task file records (the sets changed after the freeze)."""
    doc = read_json(TASKS_DOC)
    if sha256_file(SETS_FILE) != doc.get("sets_file_sha256"):
        raise ValueError(f"{SETS_FILE} does not match the sha256 recorded in {TASKS_DOC}")
    sets = read_jsonl(SETS_FILE)
    if [s["task_id"] for s in sets] != [t["task_id"] for t in doc["tasks"]]:
        raise ValueError("the order of the frozen sets differs from the frozen task file")
    return sets


def primary_span(matched_set: dict) -> tuple[int, dict]:
    """(position, span) of the set's one primary span: the passage that holds the agent's main claim."""
    found = [(i, s) for i, s in enumerate(matched_set["spans"]) if s["is_primary"]]
    if len(found) != 1:
        raise ValueError(f"{matched_set['task_id']}: {len(found)} primary spans, expected 1")
    return found[0]


def short_name(task_id: str) -> str:
    """The task name as the paper's examples print it: the id without the "coding-" prefix and the hash suffix."""
    name = task_id[len("coding-"):] if task_id.startswith("coding-") else task_id
    return name.rsplit("-", 1)[0]

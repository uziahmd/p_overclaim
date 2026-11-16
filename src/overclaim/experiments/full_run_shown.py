"""Session 5, Part 2 ("full_run"), stage pointwise_shown (contract 6E): Experiment 1 pointwise in the shown condition
(Tier 1, brief Step 9), Experiment 3's verification-first scorer (Step 11) and E7's instruction-only scorer.

Per task of a run, in one worker thread and back to back (the order of Part 1's prep5.stage_e3pw):
1. the plain shown scorer on O, H and P (e1b_pointwise.score_task, rubric pointwise-neutral, live_before 0) ->
   run_dir/pointwise_calls.jsonl;
2. for a run with `e7_instruction_only` and a task in the E7 list: the instruction-only scorer on O, H and P
   (score_task with rubric pointwise-instruction-only and pointwise.question_for's question and instruction) ->
   run_dir/e7_calls.jsonl. It sends the plain scorer's schema and prefix, so live_before is the number of live calls
   the plain scorer made for this task in this invocation and the prompt-cache rule applies from its first call;
3. for a run with `e3_verify_first` and a task in the run's E3 list: the verification-first scorer on O, H and P
   (e3_verification_pointwise.score_task_vf, cfg["domain"]'s claim phrase, live_before 0 because it sends its own
   schema) with the judge spec's max_tokens set to configs/models.yaml's e3_max_tokens (and max_output_tokens, the key
   a Gemini client reads) -> run_dir/e3pw_calls.jsonl.
The prompt cache is on for an Anthropic judge only; Gemini and OpenRouter clients put the prefix in front of the prompt
and send it whole, so for them use_prompt_cache is False and the full prompt is sent. Tasks run in max_workers threads
of the provider; the scorers of one task stay sequential. Every record carries grader = the judge name, set
SET_NAME, condition shown and its replicate; the route is in every record.

Task lists come from the frozen task file (freeze.tasks_out, whose sets_file_sha256 the frozen sets file must match):
picks.shown_eligible, picks.band_tasks, picks.e7_tasks (the first e7_n_tasks in pick order), and o55_shown = the first
o55_shown.n_tasks of picks.e7_tasks as well (the same seeded pick, shown-eligible sets with the block's seed, one gap
type in turn; read from the freeze so that a re-freeze carries it; README amendment 3, item 10). A run's E7 and E3 lists are cut to its
own tasks. The tie rule (decision (4) of the addendum of 2026-11-07): run_dir/tie_rule.json (stage tie_rule) lists the
qualifying graders; for each of them every run of this stage gets a second sample, run `<id>-r1` with replicate 1,
after the first pass, launched only when its first pass is done. The live stage refuses to start without that file
(the tie rule is measured on the hidden condition before the shown runs); the dry run says so. A judge of this stage
that the file does not decide (listed under not_decided, or absent) gets no second sample yet; the dry run and
stages_status.json name it, and a rerun after the tie_rule stage decides it adds the -r1 runs. The tie rule's budget
clause (README amendment 3, item 9): a qualifying Claude judge named in the block's second_sample_hidden_only gets its
second sample on its hidden rows only; its -r1 runs here are recorded as skipped with HIDDEN_ONLY_REASON, and the
choice for every qualifying judge of this stage goes to the stage's stages_status.json entry (`tie_rule`).

Bookkeeping as pilot4._stage_pointwise: runs_status.json gets one entry per run (done, skipped with the reason,
stopped, failed; a failed run (an error left after the client's retries, a prompt cache not read) does not stop the
stage: the next runs go on, the stage ends failed naming the failed runs, and a new invocation resumes them from
their records (changed 2026-11-08 after the qwen shown run's 429s stopped the stage twice); a done entry has the
record, invalid and live counts per file, recomputed from the stored records on
every invocation); a run whose records are complete makes no call and is not gated; otherwise its first pass not done
(a -r1 run), the budget clause of the tie rule (a hidden-only -r1 run), a missing credential, a provider blocked
earlier in this invocation (Gemini balance exhausted), pilot2.gate (Anthropic stop-launching line) or the OpenRouter
stop line skips it; Google runs are bounded only by the CostLogger's google cap. BudgetExceeded stops
the stage (pilot2._budget_stop); judge.CacheNotRead and any other error fail the run and the stage. A Gemini run whose
spec route is neither vertex nor aistudio is refused (ValueError) before any call. The stage status goes to
stages_status.json. Message text is data: it is placed into prompts as text and never executed.

Uzair's memo of 2026-11-08 (README amendment 3, item 13), after Vertex refused the project: a Gemini run may carry
`route: aistudio`; it then names its projection rows per output file (`projection_rows`) and runs under the AI Studio
cap of budget.google_aistudio_cap_usd by the launch rule of overclaim.experiments.route_cap (remaining cap, measured or
list-price unit, whole row or the longest task prefix that fits, status PARTIAL with its n); a route-cap
BudgetExceeded, a 402 or a quota 429 (gemini.BalanceExhausted) ends the run as PARTIAL, blocks Google for the
invocation and lets the next runs go on. Per-run overrides (the qwen shown row, memo section 3): `max_workers`,
`backoff` ({base_s, retries, jitter}, the client's retry schedule) and `openrouter_providers` ({min_context_tokens,
quantizations}: resolved against OpenRouter's endpoints listing at launch into provider.only, logged and recorded).
`tasks: band` (the band tasks alone) is the DeepSeek band row of memo section 4. The CLI's --only (Ctx6.only) runs
the named runs and leaves the others untouched, so the memo's priority order can cross the shown and pairwise stages.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from overclaim.experiments import e1b_pointwise, pilot2
from overclaim.experiments import e3_verification_pointwise as vfp
from overclaim.experiments import route_cap
from overclaim.experiments.pilot2 import EXIT_FAILED, EXIT_OK, _read_json, _read_jsonl, _repo_path, _write_json
from overclaim.experiments.pilot3 import _record_stage
from overclaim.experiments.route_cap import AISTUDIO, PARTIAL
from overclaim.graders import pointwise

if TYPE_CHECKING:   # full_run imports this module to register the stage
    from overclaim.experiments.full_run import Ctx6

logger = logging.getLogger(__name__)

STAGE = "pointwise_shown"
CONDITION = "shown"
SET_BLOCK = "full_run"          # the block's `set`: the frozen sets of the freeze block
SET_NAME = "v4_full_run"        # the records' `set` (contract 6D records the hidden runs under the same name)
VARIANTS = ("O", "H", "P")
PLAIN_FILE, E7_FILE, E3_FILE = "pointwise_calls.jsonl", "e7_calls.jsonl", "e3pw_calls.jsonl"
FILES = (PLAIN_FILE, E7_FILE, E3_FILE)
TIE_RULE_FILE = "tie_rule.json"
RUN_TASK_LISTS = ("shown_eligible", "shown_eligible_and_band", "band", "o55_shown")
E3_TASK_LISTS = ("shown_eligible", "e7")
RUN_KEYS = frozenset({"id", "judge", "route", "tasks", "e7_instruction_only", "e3_verify_first", "e3_tasks",
                      "projection_rows", "max_workers", "backoff", "openrouter_providers"})
GEMINI_ROUTES = ("vertex", "aistudio")
SCORER_OF_FILE = {PLAIN_FILE: "plain", E7_FILE: "instruction_only", E3_FILE: "verify_first"}
SECOND_SAMPLE_SUFFIX = "-r1"
HIDDEN_ONLY_REASON = "tie rule, budget: hidden rows only (README amendment 3, item 9)"
CONFIG_BLOCKS: dict[str, str] = {
    "pointwise_shown": ("this stage's block: set, condition, runs, o55_shown, e7_tasks, e7_n_tasks, e3_tasks, "
                        "second_sample_hidden_only"),
    "freeze": "sets_out and tasks_out (the frozen files, read only) and task_files (where each task dict is read)",
    "judges": "the judge specs (full_run.load_config copies them from models_config)",
    "models_config": "e3_max_tokens of configs/models.yaml, the verification-first scorer's max_tokens",
    "budget": "anthropic_stop_launching_usd (pilot2.gate), openrouter_stop_launching_usd; the caps through ctx.cost",
    "max_workers": "threads per provider (anthropic, openrouter, google)",
    "domain": "the verification-first scorer's claim phrase (coding)",
    "run_name": "the records' run",
}


# ---------------------------------------------------------------------------------------------
# Config, frozen files, task lists
# ---------------------------------------------------------------------------------------------


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def check_block(cfg: dict) -> None:
    """ValueError listing every failed check of cfg["pointwise_shown"]: set is "full_run" and condition "shown";
    runs is a non-empty list of entries with only the keys of RUN_KEYS, a unique id that does not end in "-r1", a judge
    of cfg["judges"], tasks in RUN_TASK_LISTS, boolean e7_instruction_only and e3_verify_first when given, e3_tasks in
    E3_TASK_LISTS when given; every Gemini run's spec (pilot2.run_spec) has route vertex (Part 2 runs every Gemini row
    through vertex, addendum section 3b; configs/models.yaml still says aistudio); o55_shown has a positive integer
    n_tasks and a seed; e7_tasks is "freeze"; e7_n_tasks is a positive integer; e3_tasks is in E3_TASK_LISTS;
    second_sample_hidden_only is a list (empty when every qualifying grader's shown second samples run) of judges of
    this block's runs whose provider is anthropic (the tie rule's budget clause covers Claude judges only, README
    amendment 3, item 9)."""
    block = cfg.get(STAGE)
    if not isinstance(block, dict):
        raise ValueError(f"the config has no {STAGE} block")
    problems = []
    if block.get("set") != SET_BLOCK:
        problems.append(f"{STAGE}.set must be {SET_BLOCK!r} (the frozen sets of the freeze block)")
    if block.get("condition") != CONDITION:
        problems.append(f"{STAGE}.condition must be {CONDITION!r}")
    runs = block.get("runs")
    if not isinstance(runs, list) or not runs:
        problems.append(f"{STAGE}.runs must be a non-empty list")
        runs = []
    ids = [r.get("id") if isinstance(r, dict) else None for r in runs]
    for run, run_id in zip(runs, ids):
        if not isinstance(run, dict) or not isinstance(run_id, str) or not run_id:
            problems.append(f"{STAGE}.runs: every entry needs an id")
            continue
        where = f"{STAGE}.runs[{run_id}]"
        unknown = sorted(set(run) - RUN_KEYS)
        if unknown:
            problems.append(f"{where}: unknown keys {unknown}")
        if ids.count(run_id) > 1 or run_id.endswith(SECOND_SAMPLE_SUFFIX):
            problems.append(f"{where}: ids must be unique and must not end in {SECOND_SAMPLE_SUFFIX!r}")
        provider = cfg["judges"].get(run.get("judge"), {}).get("provider")
        if run.get("judge") not in cfg["judges"]:
            problems.append(f"{where}: judge {run.get('judge')!r} is not in configs/models.yaml's judges")
        elif provider == "gemini" and pilot2.run_spec(cfg, run).get("route") not in GEMINI_ROUTES:
            problems.append(f"{where}: a Gemini run must carry route: vertex or aistudio (addendum section 3b; "
                            "amendment 3, item 13)")
        if run.get("tasks") not in RUN_TASK_LISTS:
            problems.append(f"{where}: tasks must be one of {list(RUN_TASK_LISTS)}")
        for key in ("e7_instruction_only", "e3_verify_first"):
            if not isinstance(run.get(key, False), bool):
                problems.append(f"{where}: {key} must be true or false")
        if "e3_tasks" in run and run["e3_tasks"] not in E3_TASK_LISTS:
            problems.append(f"{where}: e3_tasks must be one of {list(E3_TASK_LISTS)}")
        problems += _check_overrides(cfg, run, where, provider)
    o55 = block.get("o55_shown")
    if not isinstance(o55, dict) or not _positive_int(o55.get("n_tasks")) or "seed" not in o55:
        problems.append(f"{STAGE}.o55_shown needs a positive integer n_tasks and a seed")
    if block.get("e7_tasks") != "freeze":
        problems.append(f"{STAGE}.e7_tasks must be 'freeze' (picks.e7_tasks of the frozen task file)")
    if not _positive_int(block.get("e7_n_tasks")):
        problems.append(f"{STAGE}.e7_n_tasks must be a positive integer")
    if block.get("e3_tasks") not in E3_TASK_LISTS:
        problems.append(f"{STAGE}.e3_tasks must be one of {list(E3_TASK_LISTS)}")
    hidden_only = block.get("second_sample_hidden_only")
    if not isinstance(hidden_only, list) or not all(isinstance(j, str) for j in hidden_only):
        problems.append(f"{STAGE}.second_sample_hidden_only must be a list of judge names ([] when every qualifying "
                        "grader's shown second samples run)")
    else:
        run_judges = {r.get("judge") for r in runs if isinstance(r, dict)}
        for name in hidden_only:
            if name not in run_judges or name not in cfg["judges"]:
                problems.append(f"{STAGE}.second_sample_hidden_only: {name!r} is not the judge of a run of this block")
            elif cfg["judges"][name]["provider"] != "anthropic":
                problems.append(f"{STAGE}.second_sample_hidden_only: {name!r} is not a Claude judge (the tie rule's "
                                "budget clause covers Claude judges only, README amendment 3, item 9)")
    if problems:
        raise ValueError(f"invalid {STAGE} block:\n" + "\n".join(problems))


def _check_overrides(cfg: dict, run: dict, where: str, provider: str | None) -> list[str]:
    """The problems of a run's optional keys: projection_rows (only on an aistudio Gemini run, where it is required:
    a dict naming a projection row id for every file the run writes, keys among FILES; the budget then needs
    route_cap.CAP_KEY, a positive number); max_workers (a positive integer); backoff (a dict with a positive base_s,
    a non-negative integer retries and an optional jitter in [0, 1]); openrouter_providers (only on an OpenRouter
    judge's run: a dict with a positive integer min_context_tokens and a non-empty list of quantization names)."""
    problems: list[str] = []
    aistudio = provider == "gemini" and pilot2.run_spec(cfg, run).get("route") == AISTUDIO
    rows = run.get("projection_rows")
    if aistudio:
        cap = (cfg.get("budget") or {}).get(route_cap.CAP_KEY)
        if not isinstance(cap, (int, float)) or isinstance(cap, bool) or cap <= 0:
            problems.append(f"{where}: an aistudio run needs budget.{route_cap.CAP_KEY} (a positive USD amount)")
        needed = [PLAIN_FILE] + ([E7_FILE] if run.get("e7_instruction_only") else []) + (
            [E3_FILE] if run.get("e3_verify_first") else [])
        if not isinstance(rows, dict) or not all(isinstance(rows.get(f), str) and rows[f] for f in needed) \
                or not set(rows) <= set(FILES):
            problems.append(f"{where}: an aistudio run needs projection_rows naming a projection row id for each of "
                            f"{needed} (keys among {list(FILES)})")
    elif rows is not None:
        problems.append(f"{where}: projection_rows is only for a Gemini run on route aistudio")
    if "max_workers" in run and not _positive_int(run["max_workers"]):
        problems.append(f"{where}: max_workers must be a positive integer")
    if "backoff" in run:
        b = run["backoff"]
        ok = (isinstance(b, dict) and set(b) <= {"base_s", "retries", "jitter"}
              and isinstance(b.get("base_s"), (int, float)) and not isinstance(b.get("base_s"), bool)
              and b["base_s"] > 0 and isinstance(b.get("retries"), int) and not isinstance(b.get("retries"), bool)
              and b["retries"] >= 0
              and ("jitter" not in b or (isinstance(b["jitter"], (int, float)) and 0 <= b["jitter"] <= 1)))
        if not ok:
            problems.append(f"{where}: backoff must be {{base_s: > 0, retries: >= 0 (integer), jitter: 0 to 1 "
                            "(optional)}")
    if "openrouter_providers" in run:
        rule = run["openrouter_providers"]
        if provider != "openrouter":
            problems.append(f"{where}: openrouter_providers is only for an OpenRouter judge's run")
        elif not (isinstance(rule, dict) and set(rule) <= {"min_context_tokens", "quantizations"}
                  and _positive_int(rule.get("min_context_tokens"))
                  and isinstance(rule.get("quantizations"), list) and rule["quantizations"]
                  and all(isinstance(q, str) for q in rule["quantizations"])):
            problems.append(f"{where}: openrouter_providers must be {{min_context_tokens: a positive integer, "
                            "quantizations: a non-empty list of names}")
    return problems


def e3_max_tokens(cfg: dict) -> int:
    """configs/models.yaml's e3_max_tokens (read from cfg["models_config"]; full_run.load_config does not copy it)."""
    with open(_repo_path(cfg["models_config"]), encoding="utf-8") as f:
        return int(yaml.safe_load(f)["e3_max_tokens"])


def load_frozen(cfg: dict) -> tuple[list[dict], dict]:
    """(the frozen sets of freeze.sets_out in file order, the frozen task document of freeze.tasks_out). ValueError when
    either file is missing, when the sets file's sha256 differs from the document's sets_file_sha256, or when the two
    do not name the same task ids in the same order."""
    from overclaim.experiments.full_run import file_sha256

    fz = cfg["freeze"]
    sets_path, doc_path = _repo_path(fz["sets_out"]), _repo_path(fz["tasks_out"])
    missing = [str(p) for p in (sets_path, doc_path) if not p.exists()]
    if missing:
        raise ValueError(f"the frozen files are missing: {missing} (run the freeze stage first)")
    doc = _read_json(doc_path, {})
    digest = file_sha256(sets_path)
    if digest != doc.get("sets_file_sha256"):
        raise ValueError(f"{fz['sets_out']} has sha256 {digest}, the frozen task file records "
                         f"{doc.get('sets_file_sha256')}: the sets file changed after the freeze")
    sets = _read_jsonl(sets_path)
    if [s["task_id"] for s in sets] != [t["task_id"] for t in doc["tasks"]]:
        raise ValueError(f"{fz['sets_out']} and {fz['tasks_out']} do not list the same task ids in the same order")
    return sets, doc


def task_lists(cfg: dict, sets: list[dict], doc: dict) -> dict[str, list[str]]:
    """The named task lists: shown_eligible (picks.shown_eligible, set order), band (picks.band_tasks),
    shown_eligible_and_band (the two, shown-eligible first, so a stop falls on the band), o55_shown (the first
    o55_shown.n_tasks of picks.e7_tasks: the freeze's seeded pick over the shown-eligible sets, one per gap type in
    turn, carried across a re-freeze; README amendment 3, item 10) and e7 (the first e7_n_tasks of picks.e7_tasks,
    pick order). ValueError when e7_n_tasks or o55_shown.n_tasks exceeds the E7 picks, or when o55_shown.seed is not
    the seed the freeze picked with (e2.seed), or when a pick is not a shown-eligible frozen set."""
    block, picks = cfg[STAGE], doc["picks"]
    eligible = {s["task_id"] for s in sets if s["shown_eligible"]}
    n_o55, n_e7 = int(block["o55_shown"]["n_tasks"]), int(block["e7_n_tasks"])
    e7_picks = list(picks["e7_tasks"])
    for name, n in (("o55_shown.n_tasks", n_o55), ("e7_n_tasks", n_e7)):
        if n > len(e7_picks):
            raise ValueError(f"{STAGE}.{name} {n} exceeds the {len(e7_picks)} E7 picks of the freeze")
    if block["o55_shown"]["seed"] != cfg["e2"]["seed"]:
        raise ValueError(f"{STAGE}.o55_shown.seed {block['o55_shown']['seed']!r} is not the seed the freeze picked "
                         f"the E7 tasks with (e2.seed {cfg['e2']['seed']!r})")
    outside = [t for t in e7_picks if t not in eligible]
    if outside:
        raise ValueError(f"{STAGE}: E7 picks that are not shown-eligible frozen sets: {outside}")
    lists = {"shown_eligible": list(picks["shown_eligible"]), "band": list(picks["band_tasks"]),
             "shown_eligible_and_band": [*picks["shown_eligible"], *picks["band_tasks"]],
             "o55_shown": e7_picks[:n_o55], "e7": e7_picks[:n_e7]}
    # The seeded order of each list for the aistudio runs (README amendment 3, item 13: a row the cap cuts runs on
    # the longest prefix of its tasks in seed order, gap types in turn, not in the gap-grouped frozen set order): the
    # frozen E7 pick first (the only seeded order the freeze holds over the shown-eligible sets, carried across the
    # re-freeze), then the rest of the list by prep5_e2.pick_tasks with the seed the freeze picked E7 with (cfg.seed).
    lists["seeded_order"] = {name: seeded_task_order(ids, sets, e7_picks, cfg["seed"])
                             for name, ids in lists.items() if name in RUN_TASK_LISTS}
    return lists


def seeded_task_order(ids: list[str], sets: list[dict], head: list[str], seed: Any) -> list[str]:
    """ids reordered: the ids of `head` that are in ids, in head order, then the others in prep5_e2.pick_tasks order
    over their sets with `seed` (per gap type a seeded order, then one per gap type in turn)."""
    from overclaim.experiments.prep5_e2 import pick_tasks

    wanted = set(ids)
    first = [i for i in head if i in wanted]
    taken = set(first)
    rest_sets = [s for s in sets if s["task_id"] in wanted and s["task_id"] not in taken]
    rest = pick_tasks(rest_sets, len(rest_sets), seed)
    return first + rest


def read_tie_rule(run_dir: Path) -> dict | None:
    """run_dir/tie_rule.json (stage tie_rule, contract 6D: full_run_pointwise.tie_table), or None when the file does
    not exist. ValueError when its `qualifying` is not a list of grader names or its `rows` (the decided graders) or
    `not_decided` is not a list of entries with a grader name."""
    path = run_dir / TIE_RULE_FILE
    if not path.exists():
        return None
    table = _read_json(path, {})
    qualifying = table.get("qualifying")
    if not isinstance(qualifying, list) or not all(isinstance(g, str) for g in qualifying):
        raise ValueError(f"{path}: `qualifying` must be a list of grader names")
    for key in ("rows", "not_decided"):
        entries = table.get(key)
        if not isinstance(entries, list) or not all(isinstance(e, dict) and isinstance(e.get("grader"), str)
                                                    for e in entries):
            raise ValueError(f"{path}: `{key}` must be a list of entries with a grader name")
    return table


def tie_state(cfg: dict, table: dict | None) -> dict | None:
    """The tie rule as this stage applies it, or None without a tie-rule table: qualifying (the table's), not_decided
    (the judges of this block's runs without a row in the table: listed under its not_decided or absent; no second
    sample is planned for them yet), second_sample_hidden_only (the block's list) and shown_second_sample ({qualifying
    judge with a run here: HIDDEN_ONLY_REASON when it is in second_sample_hidden_only, else "runs"}, the budget
    clause's recorded choice)."""
    if table is None:
        return None
    block = cfg[STAGE]
    judges = list(dict.fromkeys(r["judge"] for r in block["runs"]))
    decided = {r["grader"] for r in table["rows"]}
    hidden_only = list(block["second_sample_hidden_only"])
    return {"qualifying": list(table["qualifying"]), "not_decided": [j for j in judges if j not in decided],
            "second_sample_hidden_only": hidden_only,
            "shown_second_sample": {j: HIDDEN_ONLY_REASON if j in hidden_only else "runs" for j in judges
                                    if j in table["qualifying"]}}


def plan_runs(cfg: dict, lists: dict[str, list[str]], qualifying: list[str] | None) -> list[dict]:
    """The runs in launch order: the block's runs (replicate 0, config order), then for every run whose judge is in
    `qualifying` its second sample (id + "-r1", replicate 1, first_pass = the run's id, hidden_only when its judge is
    in the block's second_sample_hidden_only), in config order. Each: id, judge, replicate, first_pass (None for a
    first pass), hidden_only, spec
    (pilot2.run_spec), vf_spec (spec with max_tokens and, for a Gemini spec, max_output_tokens set to e3_max_tokens;
    None without e3_verify_first), provider (pilot2.COST_PROVIDER), route, the task ids of each scorer in the run's
    task order (plain: its tasks list; e7: the E7 list cut to them when e7_instruction_only; e3: the run's e3_tasks,
    else the block's, cut to them when e3_verify_first) and expected_records per file (three per task)."""
    block = cfg[STAGE]
    max_tokens = e3_max_tokens(cfg)
    first: list[dict] = []
    for run in block["runs"]:
        spec = pilot2.run_spec(cfg, run)
        if run.get("backoff") is not None:
            spec["backoff"] = dict(run["backoff"])      # the client's retry schedule (judge.backoff_kwargs)
        aistudio = spec["provider"] == "gemini" and spec.get("route") == AISTUDIO
        plain = lists["seeded_order"][run["tasks"]] if aistudio else lists[run["tasks"]]
        e7 = set(lists["e7"]) if run.get("e7_instruction_only") else set()
        e3 = set(lists[run.get("e3_tasks", block["e3_tasks"])]) if run.get("e3_verify_first") else set()
        vf_spec = None
        if run.get("e3_verify_first"):
            vf_spec = {**spec, "max_tokens": max_tokens}
            if spec["provider"] == "gemini":
                vf_spec["max_output_tokens"] = max_tokens
        ids = {PLAIN_FILE: list(plain), E7_FILE: [i for i in plain if i in e7], E3_FILE: [i for i in plain if i in e3]}
        first.append({"id": run["id"], "judge": run["judge"], "replicate": 0, "first_pass": None,
                      "hidden_only": False, "spec": spec, "vf_spec": vf_spec,
                      "provider": pilot2.COST_PROVIDER[spec["provider"]], "route": spec.get("route"), "ids": ids,
                      "expected_records": {f: 3 * len(ids[f]) for f in FILES},
                      "projection_rows": run.get("projection_rows"), "max_workers": run.get("max_workers"),
                      "openrouter_providers": run.get("openrouter_providers")})
    hidden_only = set(block["second_sample_hidden_only"])
    second = [{**r, "id": r["id"] + SECOND_SAMPLE_SUFFIX, "replicate": 1, "first_pass": r["id"],
               "hidden_only": r["judge"] in hidden_only} for r in first
              if qualifying is not None and r["judge"] in qualifying]
    return first + second


def task_dicts(cfg: dict, doc: dict, ids: list[str]) -> dict[str, dict]:
    """{task_id: task dict} for ids, each read from the task file the frozen task file names for it (tasks[].task_file,
    found among freeze.task_files by file name); only those files are read. ValueError naming the ids without a task
    file entry or not found in their file."""
    file_of = {t["task_id"]: t["task_file"] for t in doc["tasks"]}
    path_of = {Path(p).name: _repo_path(p) for p in cfg["freeze"]["task_files"]}
    unplaced = [i for i in ids if file_of.get(i) not in path_of]
    if unplaced:
        raise ValueError(f"{len(unplaced)} tasks have no task file among freeze.task_files: {unplaced[:3]}")
    wanted: dict[str, set[str]] = {}
    for i in ids:
        wanted.setdefault(file_of[i], set()).add(i)
    found: dict[str, dict] = {}
    for name, want in wanted.items():
        for task in _read_jsonl(path_of[name]):
            if task["task_id"] in want:
                found.setdefault(task["task_id"], task)
    missing = [i for i in ids if i not in found]
    if missing:
        raise ValueError(f"{len(missing)} tasks are not in the task file the freeze names for them: {missing[:3]}")
    return found


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------


class _LiveCount:
    """A client for one task's plain scorer that counts its live attempts (responses that are not disk-cache hits), the
    calls score_task's prompt-cache rule counts. A record resumed from the file makes no call, so the count is the
    live calls made for this task in this invocation: the instruction-only scorer passes it as live_before."""

    def __init__(self, client: Any) -> None:
        self.client, self.live = client, 0

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        resp = self.client.complete(prompt, **kwargs)
        self.live += not resp.cached
        return resp


def score_shown_task(task: dict, matched_set: dict, client: Any, vf_client: Any, grader: str, run_dir: Path,
                     run_name: str, *, replicate: int, structured: bool, use_prompt_cache: bool, domain: str,
                     e7: bool, e3: bool) -> dict[str, list[dict]]:
    """One task's scorers, back to back: the plain scorer (run_dir/PLAIN_FILE), then the instruction-only scorer when
    e7 (run_dir/E7_FILE; live_before = the plain scorer's live calls now), then the verification-first scorer with
    vf_client when e3 (run_dir/E3_FILE; live_before 0). Grader, set SET_NAME, condition shown, the replicate. Returns
    {file: the task's records of that scorer} ([] for a scorer not run)."""
    counted = _LiveCount(client)
    common = {"set_name": SET_NAME, "structured": structured, "use_prompt_cache": use_prompt_cache,
              "replicate": replicate}
    out = {PLAIN_FILE: e1b_pointwise.score_task(task, matched_set, counted, grader, CONDITION, run_dir / PLAIN_FILE,
                                                run_name, live_before=0, **common),
           E7_FILE: [], E3_FILE: []}
    if e7:
        question, instruction = pointwise.question_for(pointwise.RUBRIC_INSTRUCTION_ONLY)
        out[E7_FILE] = e1b_pointwise.score_task(task, matched_set, client, grader, CONDITION, run_dir / E7_FILE,
                                                run_name, live_before=counted.live,
                                                rubric=pointwise.RUBRIC_INSTRUCTION_ONLY, question=question,
                                                instruction=instruction, **common)
    if e3:
        out[E3_FILE] = vfp.score_task_vf(task, matched_set, vf_client, grader, CONDITION, run_dir / E3_FILE, run_name,
                                         domain=domain, live_before=0, **common)
    return out


def score_run(ctx: Ctx6, run: dict, client: Any, vf_client: Any, task_of: dict[str, dict],
              set_of: dict[str, dict], after_task: Any = None) -> dict[str, list[dict]]:
    """score_shown_task for every task of the run (its plain list, in order) in the run's max_workers threads (its
    own override, else max_workers[provider]); the first exception cancels the tasks not yet started and is raised.
    after_task(records of one task) is called as each task finishes (the aistudio re-price counter). Returns {file:
    the run's records}."""
    cfg = ctx.cfg
    e7, e3 = set(run["ids"][E7_FILE]), set(run["ids"][E3_FILE])
    use_cache = run["spec"]["provider"] == "anthropic"
    structured = bool(run["spec"].get("structured_output"))

    def one(tid: str) -> dict[str, list[dict]]:
        out = score_shown_task(task_of[tid], set_of[tid], client, vf_client, run["judge"], ctx.run_dir,
                               cfg["run_name"], replicate=run["replicate"], structured=structured,
                               use_prompt_cache=use_cache, domain=cfg["domain"], e7=tid in e7, e3=tid in e3)
        if after_task is not None:
            after_task(out)
        return out

    ids = run["ids"][PLAIN_FILE]
    workers = int(run.get("max_workers") or cfg.get("max_workers", {}).get(run["provider"], 1))
    if workers <= 1 or len(ids) <= 1:
        per_task = [one(t) for t in ids]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, t) for t in ids]
            try:
                per_task = [f.result() for f in futures]
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
    return {f: [r for recs in per_task for r in recs[f]] for f in FILES}


# ---------------------------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------------------------


RUBRIC_OF_FILE = {PLAIN_FILE: pointwise.RUBRIC, E7_FILE: pointwise.RUBRIC_INSTRUCTION_ONLY, E3_FILE: vfp.VFP_RUBRIC}


def run_records(records: dict[str, list[dict]], run: dict) -> dict[str, dict[tuple[str, str], dict]]:
    """{file: {(task_id, variant): record}} of the run already in that file: records of the run's judge, set SET_NAME,
    condition shown, its replicate, the file's rubric (RUBRIC_OF_FILE; two scorers of one judge differ by rubric), a
    task of the run's list for that scorer and a variant of VARIANTS; for a key written twice the later record, the
    one the scorers' resume returns."""
    out = {}
    for f in FILES:
        wanted = set(run["ids"][f])
        out[f] = {(r["task_id"], r["variant"]): r for r in records[f]
                  if r.get("grader") == run["judge"] and r.get("set") == SET_NAME
                  and r.get("condition") == CONDITION and int(r.get("replicate", 0)) == run["replicate"]
                  and r.get("rubric") == RUBRIC_OF_FILE[f] and r.get("task_id") in wanted
                  and r.get("variant") in VARIANTS}
    return out


def present(records: dict[str, list[dict]], run: dict) -> dict[str, int]:
    """{file: distinct (task_id, variant) of the run already in that file} (run_records)."""
    return {f: len(found) for f, found in run_records(records, run).items()}


def done_counts(records: dict[str, list[dict]], run: dict) -> dict[str, dict[str, int]]:
    """A done entry's counts from the stored records of the run (run_records), per file: records, n_invalid (score
    None) and n_live (records not served from the disk cache when they were scored). The same on every invocation,
    as pilot4._stage_pointwise recomputes them from the records a resumed run returns."""
    found = run_records(records, run)
    return {"records": {f: len(found[f]) for f in FILES},
            "n_invalid": {f: sum(r["score"] is None for r in found[f].values()) for f in FILES},
            "n_live": {f: sum(not r["cached"] for r in found[f].values()) for f in FILES}}


def _read_files(run_dir: Path) -> dict[str, list[dict]]:
    return {f: _read_jsonl(run_dir / f) for f in FILES}


def stage_counts(records: dict[str, list[dict]]) -> dict[str, int]:
    """{file: records of this stage's set and condition (SET_NAME, shown)}."""
    return {f: sum(r.get("set") == SET_NAME and r.get("condition") == CONDITION for r in recs)
            for f, recs in records.items()}


def is_aistudio(run: dict) -> bool:
    """A Gemini run on the aistudio route (the memo's capped rows)."""
    return run["spec"]["provider"] == "gemini" and run.get("route") == AISTUDIO


def _keep_for(run: dict, name: str) -> Any:
    """The record filter of one scorer of the run (its judge, shown, replicate, the file's rubric)."""
    return (lambda r: r.get("grader") == run["judge"] and r.get("condition") == CONDITION
            and int(r.get("replicate", 0)) == run["replicate"] and r.get("rubric") == RUBRIC_OF_FILE[name])


def aistudio_plan(ctx: Ctx6, run: dict, records: dict[str, list[dict]]) -> dict:
    """The launch plan of an aistudio run under the cap (route_cap): the cap state, the unit per scorer (measured or
    list price), the missing calls per file, the per-task cost of the missing calls (every scorer of the task) and
    the prefix fit over the run's task order. ValueError when the cap, the projection or a row's unit is missing."""
    state = route_cap.cap_state(ctx.cost)
    units_table = route_cap.projection_units(ctx.run_dir)
    found = run_records(records, run)
    files = [f for f in FILES if run["ids"][f]]
    units = {SCORER_OF_FILE[f]: route_cap.unit_for(records[f], _keep_for(run, f), units_table,
                                                  run["projection_rows"][f]) for f in files}
    remaining_calls = {SCORER_OF_FILE[f]: sum((t, v) not in found[f] for t in run["ids"][f] for v in VARIANTS)
                       for f in files}
    per_task = []
    for tid in run["ids"][PLAIN_FILE]:
        usd = sum(units[SCORER_OF_FILE[f]]["usd"] for f in files if tid in run["ids"][f]
                  for v in VARIANTS if (tid, v) not in found[f])
        per_task.append((tid, usd))
    return {**state, "units": units, "remaining_calls": remaining_calls, "per_task": per_task,
            "fit": route_cap.prefix_fit(per_task, state["remaining"])}


def later_rows(ctx: Ctx6, runs: list[dict], records: dict[str, list[dict]], judge: str) -> list[tuple]:
    """route_cap.Reprice's `later`: (run id, scorer, missing calls, list unit, same judge) for every scorer of the
    aistudio runs among `runs` (the runs after the current one in launch order)."""
    units = route_cap.projection_units(ctx.run_dir)
    out = []
    for r in runs:
        if not is_aistudio(r):
            continue
        have = present(records, r)
        for f in FILES:
            if r["ids"][f]:
                out.append((r["id"], SCORER_OF_FILE[f], 3 * len(r["ids"][f]) - have[f],
                            units.get((r.get("projection_rows") or {}).get(f, ""), 0.0), r["judge"] == judge))
    return out


def plan_summary(plan: dict) -> dict:
    """The plan without its per-task list, for a status entry."""
    return {k: v for k, v in plan.items() if k != "per_task"}


def cut_run(run: dict, n: int) -> dict:
    """The run on the first n tasks of its plain list (every scorer's list cut to them; expected records recomputed)."""
    keep = set(run["ids"][PLAIN_FILE][:n])
    ids = {f: [t for t in run["ids"][f] if t in keep] for f in FILES}
    return {**run, "ids": ids, "expected_records": {f: 3 * len(ids[f]) for f in FILES}}


def apply_provider_rule(spec: dict, only: list[str], rule: dict) -> dict:
    """The spec with extra_body.provider set to the resolved rule: only these provider slugs, the rule's
    quantizations, fallbacks allowed (within the list), require_parameters; every other extra_body key kept."""
    extra = dict(spec.get("extra_body") or {})
    extra["provider"] = {"only": list(only), "quantizations": list(rule["quantizations"]), "allow_fallbacks": True,
                         "require_parameters": True}
    return {**spec, "extra_body": extra}


def gate(ctx: Ctx6, run: dict) -> str | None:
    """Reason not to launch the run now, or None: the OpenRouter stop line (ctx.cost.spent("openrouter") at or above
    budget.openrouter_stop_launching_usd) for an OpenRouter run, pilot2.gate (the Anthropic stop line) otherwise;
    Google runs are not gated here (the CostLogger's google cap bounds them; an aistudio run has the cap rule of
    aistudio_plan in the stage loop)."""
    if run["provider"] == "openrouter":
        spent = ctx.cost.spent("openrouter")
        stop = float(ctx.cfg["budget"]["openrouter_stop_launching_usd"])
        if spent >= stop:
            return f"OpenRouter total ${spent:.2f} has reached the stop-launching threshold ${stop:.2f}"
        return None
    return pilot2.gate(ctx, {}, run["provider"])


def _plan_line(run: dict, have: dict[str, int]) -> str:
    n = {f: len(run["ids"][f]) for f in FILES}
    vf = run["vf_spec"]
    vf_tokens = "-" if vf is None else vf.get("max_output_tokens", vf.get("max_tokens"))
    expected = ", ".join(f"{f} {run['expected_records'][f]} ({have[f]} present)" for f in FILES)
    cred = "set" if pilot2.credential_for(run["spec"]) else "MISSING -> skipped"
    line = (f"{STAGE}: run {run['id']}: {run['judge']} ({run['spec']['provider']}, route {run['route'] or '-'}), "
            f"replicate {run['replicate']}; tasks: plain {n[PLAIN_FILE]}, E7 {n[E7_FILE]}, E3 {n[E3_FILE]}; expected "
            f"records: {expected}; prompt cache {'on' if run['spec']['provider'] == 'anthropic' else 'off'}; "
            f"verification-first max output tokens {vf_tokens}; credential {cred}")
    if run.get("max_workers"):
        line += f"; max_workers {run['max_workers']} (run override)"
    if run["spec"].get("backoff"):
        line += f"; backoff {run['spec']['backoff']}"
    if run.get("openrouter_providers"):
        line += (f"; OpenRouter providers resolved at launch from the endpoints listing: {run['openrouter_providers']}"
                 " (fp8/bf16 and context as listed; no :free; provider.only + allow_fallbacks)")
    if run["first_pass"]:
        line += f"; launches only when its first pass {run['first_pass']} is done"
    if run["hidden_only"]:
        line += f"; SKIPPED: {HIDDEN_ONLY_REASON}"
    return line


def _dry_run(ctx: Ctx6, sets: list[dict], lists: dict[str, list[str]], runs: list[dict], tie: dict | None,
             task_check: str) -> int:
    cfg = ctx.cfg
    gap_of = {s["task_id"]: s["gap_type"] for s in sets}

    def split(ids: list[str]) -> dict[str, int]:
        return {g: sum(gap_of[i] == g for i in ids) for g in dict.fromkeys(gap_of.values())}

    records = _read_files(ctx.run_dir)
    ctx.note(f"{STAGE}: frozen sets {cfg['freeze']['sets_out']} ({len(sets)} sets, sha256 matches "
             f"{cfg['freeze']['tasks_out']}); records as set {SET_NAME!r}, condition {CONDITION}; task dicts: "
             f"{task_check}")
    ctx.note(f"{STAGE}: task lists: shown_eligible {len(lists['shown_eligible'])}, band {len(lists['band'])}, "
             f"o55_shown {len(lists['o55_shown'])} {split(lists['o55_shown'])} (the first "
             f"{len(lists['o55_shown'])} of the frozen E7 pick, seed {cfg[STAGE]['o55_shown']['seed']}; the same ids as "
             f"the E7 list: {lists['o55_shown'] == lists['e7']}), E7 {len(lists['e7'])} {split(lists['e7'])}; e3_tasks "
             f"{cfg[STAGE]['e3_tasks']}; verification-first max output tokens {e3_max_tokens(cfg)} "
             "(configs/models.yaml e3_max_tokens)")
    only = getattr(ctx, "only", None)
    problem = _only_problem(runs, only)
    if problem is not None:
        ctx.note(f"{STAGE}: {problem}; the live stage would run nothing and record nothing")
        return EXIT_FAILED
    if only is not None:
        ctx.note(f"{STAGE}: --only {sorted(only)}: the other runs are left untouched (no status entry rewritten)")
    for run in runs:
        line = _plan_line(run, present(records, run))
        if only is not None and not selected(run, only):
            line += "; NOT in --only: untouched"
        ctx.note(line)
        if is_aistudio(run):
            try:
                ctx.note(f"{STAGE}: run {run['id']}: {route_cap.plan_text(aistudio_plan(ctx, run, records))}")
            except ValueError as exc:
                ctx.note(f"{STAGE}: run {run['id']}: aistudio plan NOT computable: {exc}")
    first = [r for r in runs if r["replicate"] == 0]
    hidden_only = cfg[STAGE]["second_sample_hidden_only"]
    if tie is None:
        ctx.note(f"{STAGE}: tie rule: {ctx.run_dir / TIE_RULE_FILE} does not exist yet: no second-sample run is "
                 "planned, and the live stage refuses to start until the tie_rule stage has written it; a qualifying "
                 f"grader would get an {SECOND_SAMPLE_SUFFIX} run of each of its runs here: "
                 f"{sorted({r['judge'] for r in first})}; second_sample_hidden_only (budget clause): {hidden_only}")
    else:
        idle = sorted(set(tie["qualifying"]) - {r["judge"] for r in first})
        ctx.note(f"{STAGE}: tie rule: qualifying graders {tie['qualifying']}; second-sample runs "
                 f"{[r['id'] for r in runs if r['replicate'] == 1]}; shown second samples per qualifying judge here: "
                 f"{tie['shown_second_sample']}; qualifying graders without a run here: {idle}")
        if tie["not_decided"]:
            ctx.note(f"{STAGE}: tie rule not decided for {tie['not_decided']} (no row in {TIE_RULE_FILE}): no second "
                     "sample is planned for them yet; a rerun after the tie_rule stage decides them adds it")
    planned = [r for r in runs if not r["hidden_only"]]
    totals = {f: sum(r["expected_records"][f] for r in planned) for f in FILES}
    ctx.note(f"{STAGE}: expected records over the {len(planned)} runs that would launch (not counting "
             f"{len(runs) - len(planned)} hidden-only second samples): {totals}; then the cost line and the record "
             "counts per file are printed")
    return EXIT_OK


def selected(run: dict, only: set[str] | None) -> bool:
    """Whether --only selects the run: no --only, its id listed, or (a second sample) its first pass listed."""
    return only is None or run["id"] in only or (run["first_pass"] is not None and run["first_pass"] in only)


def _is_balance_stop(exc: BaseException) -> bool:
    """gemini.BalanceExhausted or its QuotaExhausted, matched by class name (as pilot4._stage_pointwise does)."""
    return any(c.__name__ == "BalanceExhausted" for c in type(exc).__mro__)


def _is_permission_denied(exc: BaseException) -> bool:
    """An HTTP 403 / permission denied from a Gemini route (the policy refusal of 2026-11-08): the provider is blocked
    for the rest of the invocation, the run is failed ("not run (provider restriction)")."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code == 403 or type(exc).__name__ == "PermissionDeniedError" or "permission_denied" in str(exc).lower()


def _keep_partial(entry: dict, previous: dict, reason: str, counts: dict) -> dict:
    """A run recorded PARTIAL by an earlier invocation stays PARTIAL (its reason, counts and prefix) when this
    invocation does not launch it; the new reason goes under reason_on_last_invocation."""
    out = {**entry, "status": PARTIAL, "reason": previous.get("reason"), "reason_on_last_invocation": reason, **counts}
    prefix = (previous.get("aistudio") or {}).get("prefix")
    if prefix is not None:
        out.setdefault("aistudio", {})
        out["aistudio"] = {**out["aistudio"], "prefix": prefix}
    return out


def _only_problem(runs: list[dict], only: set[str] | None) -> str | None:
    """The --only ids that name no run of this stage (a first pass or its -r1), or None."""
    if only is None:
        return None
    valid = {r["id"] for r in runs} | {r["id"] + SECOND_SAMPLE_SUFFIX for r in runs if r["replicate"] == 0}
    unknown = sorted(set(only) - valid)
    return f"--only names runs that are not in this stage: {unknown}" if unknown else None


# ---------------------------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------------------------


def stage_pointwise_shown(ctx: Ctx6) -> int:
    """The stage (see the module docstring): check_block, load_frozen, task_lists, read_tie_rule and plan_runs before
    anything else (a ValueError fails the stage before any client is built), then the runs in launch order. Dry run:
    the plan per run (judge, route, task counts of the three scorers, expected and present records per file,
    credential), the tie-rule state and the totals; no client, no cost file, nothing written. Live: refuses without
    tie_rule.json; per run as pilot4._stage_pointwise (status entries in runs_status.json, the gate, the credential, a
    complete run makes no call); after each run the cost line; at the end the record counts per file, printed and
    recorded in stages_status.json."""
    cfg = ctx.cfg
    try:
        check_block(cfg)
        sets, doc = load_frozen(cfg)
        lists = task_lists(cfg, sets, doc)
        table = read_tie_rule(ctx.run_dir)
        tie = tie_state(cfg, table)
        runs = plan_runs(cfg, lists, None if table is None else table["qualifying"])
        all_ids = list(dict.fromkeys(i for r in runs for i in r["ids"][PLAIN_FILE]))
        if ctx.dry_run:
            try:
                task_check = f"all {len(task_dicts(cfg, doc, all_ids))} found in their task files"
            except ValueError as exc:
                task_check = f"NOT all found: {exc}"
            return _dry_run(ctx, sets, lists, runs, tie, task_check)
        if tie is None:
            raise ValueError(f"{ctx.run_dir / TIE_RULE_FILE} does not exist: run the tie_rule stage first (the tie "
                             "rule is measured on the hidden condition before the shown runs, addendum decision (4))")
        task_of = task_dicts(cfg, doc, all_ids)
    except ValueError as exc:
        logger.error("%s: FAILED before any call: %s", STAGE, exc)
        if not ctx.dry_run:
            _record_stage(ctx, STAGE, {"status": "failed", "reason": f"ValueError: {exc}"})
        return EXIT_FAILED
    from overclaim.graders import judge as J

    set_of = {s["task_id"]: s for s in sets}
    status_path = ctx.run_dir / "runs_status.json"
    status: dict[str, dict] = _read_json(status_path, {})
    records = _read_files(ctx.run_dir)
    blocked: dict[str, str] = {}
    outcome: dict[str, str] = {}
    if tie["not_decided"]:
        logger.warning("%s: tie rule not decided for %s: no second sample planned for them yet", STAGE,
                       tie["not_decided"])

    def save(run: dict, entry: dict) -> None:
        status[run["id"]] = entry
        outcome[run["id"]] = entry["status"]
        _write_json(status_path, status)

    failed_runs: list[str] = []
    only = getattr(ctx, "only", None)
    problem = _only_problem(runs, only)
    if problem is not None:
        logger.error("%s: %s; nothing run, nothing recorded", STAGE, problem)
        return EXIT_FAILED
    if only is not None:
        logger.info("%s: --only %s: the other runs are left untouched", STAGE, sorted(only))
    for index, run in enumerate(runs):
        if not selected(run, only):
            logger.info("%s: run %s: not in --only; untouched", STAGE, run["id"])
            continue
        have = present(records, run)
        entry: dict[str, Any] = {"stage": STAGE, "judge": run["judge"], "provider": run["spec"]["provider"],
                                 "route": run["route"], "set": SET_NAME, "condition": CONDITION,
                                 "replicate": run["replicate"], "n_tasks": {f: len(run["ids"][f]) for f in FILES},
                                 "expected_records": run["expected_records"]}
        if all(have[f] >= run["expected_records"][f] for f in FILES):
            logger.info("%s: run %s: complete (%s); no call", STAGE, run["id"], have)
            save(run, {**entry, "status": "done", **done_counts(records, run)})
            continue
        first = status.get(run["first_pass"], {}).get("status") if run["first_pass"] else "done"
        if first != "done":
            reason = f"its first pass {run['first_pass']} is {first or 'not run'}, not done"
        elif run["hidden_only"]:
            reason = HIDDEN_ONLY_REASON
        elif not pilot2.credential_for(run["spec"]):
            reason = f"no credential for provider {run['spec']['provider']}" + (
                f" route {run['route']}" if run["spec"]["provider"] == "gemini" else "")
        else:
            reason = blocked.get(run["provider"]) or gate(ctx, run)
        planned = run
        reprice = None
        if reason is None and is_aistudio(run):
            # the memo's launch rule: the whole row when its re-priced remaining cost fits under the remaining cap,
            # else the longest task prefix that fits (status PARTIAL with its n), else skipped
            try:
                plan = aistudio_plan(ctx, run, records)
            except ValueError as exc:
                reason = f"aistudio plan not computable: {exc}"
                plan = None
            if plan is not None:
                entry["aistudio"] = plan_summary(plan)
                fit = plan["fit"]
                logger.info("%s: run %s: %s", STAGE, run["id"], route_cap.plan_text(plan))
                if plan["remaining"] <= 0:
                    reason = (f"aistudio cap reached: {plan['spent']:.4f} of {plan['cap']:.2f} USD spent "
                              "(memo of 2026-11-08, section 1.3)")
                elif not fit["fits"] and fit["n_fit"] == 0:
                    reason = (f"aistudio cap: the remaining {plan['remaining']:.4f} USD does not cover the first "
                              f"task's missing calls ({plan['per_task'][0][1]:.4f} USD at the re-priced units)")
                elif not fit["fits"]:
                    planned = cut_run(run, fit["n_fit"])
                    entry["aistudio"]["prefix"] = {"n_tasks_run": fit["n_fit"], "n_tasks_planned": fit["n_tasks"],
                                                   "cost_fit_usd": fit["cost_fit"]}
                    logger.warning("%s: run %s: the whole row does not fit the remaining aistudio cap; running the "
                                   "first %d of %d tasks (%.2f USD) and recording the run as %s", STAGE, run["id"],
                                   fit["n_fit"], fit["n_tasks"], fit["cost_fit"], PARTIAL)
                have_planned = present(records, planned)
                reprice = route_cap.Reprice(
                    run["id"], {SCORER_OF_FILE[f]: 3 * len(planned["ids"][f]) - have_planned[f]
                                for f in FILES if planned["ids"][f]},
                    later_rows(ctx, [r for r in runs[index + 1:] if selected(r, only)], records, run["judge"]),
                    ctx.cost)
        if reason is not None:
            previous = status.get(run["id"], {})
            if previous.get("status") == PARTIAL:
                save(run, _keep_partial(entry, previous, reason, done_counts(records, run)))
                logger.warning("%s: run %s stays %s (not launched: %s)", STAGE, run["id"], PARTIAL, reason)
                continue
            entry.update(status="skipped", reason=reason, records=have)
            if previous.get("status") == "skipped" and previous.get("reason") != reason:
                entry.update(reason=previous.get("reason"), reason_on_last_invocation=reason)
            save(run, entry)
            logger.warning("%s: run %s SKIPPED: %s", STAGE, run["id"], reason)
            continue
        spec, vf_spec = planned["spec"], planned["vf_spec"]
        if planned.get("openrouter_providers"):
            rule = planned["openrouter_providers"]
            try:
                endpoints = pilot2.openrouter_eligible_providers(spec["model"], rule)
            except Exception as exc:
                reason = f"OpenRouter endpoints listing failed: {type(exc).__name__}: {exc}"
                save(run, {**entry, "status": "failed", "reason": reason, "records": have})
                logger.error("%s: run %s FAILED before any call: %s", STAGE, run["id"], reason)
                failed_runs.append(run["id"])
                continue
            only_slugs = [e["slug"] for e in endpoints]
            entry["openrouter_providers"] = {"rule": rule, "only": only_slugs, "endpoints": endpoints}
            if not only_slugs:
                reason = (f"no OpenRouter endpoint of {spec['model']} has weights in {rule['quantizations']} and at "
                          f"least {rule['min_context_tokens']} tokens of context (none is :free)")
                save(run, {**entry, "status": "failed", "reason": reason, "records": have})
                logger.error("%s: run %s FAILED before any call: %s", STAGE, run["id"], reason)
                failed_runs.append(run["id"])
                continue
            spec = apply_provider_rule(spec, only_slugs, rule)
            vf_spec = apply_provider_rule(vf_spec, only_slugs, rule) if vf_spec is not None else None
            logger.info("%s: run %s: OpenRouter provider rule %s -> only %s (%s)", STAGE, run["id"], rule, only_slugs,
                        endpoints)
        client = ctx.client(spec)
        vf_client = ctx.client(vf_spec) if vf_spec is not None else None
        logger.info("%s: run %s: %s, replicate %d, tasks %s, records present %s", STAGE, run["id"], run["judge"],
                    run["replicate"], {f: len(planned["ids"][f]) for f in FILES}, have)
        cut = planned is not run
        after_task = None
        if reprice is not None:
            def after_task(out: dict[str, list[dict]], _r: Any = reprice) -> None:   # files -> scorer names
                _r.task_done({SCORER_OF_FILE[f]: recs for f, recs in out.items()})
        try:
            score_run(ctx, planned, client, vf_client, task_of, set_of, after_task=after_task)
        except J.BudgetExceeded as exc:
            if getattr(exc, "route", None) == AISTUDIO:
                # the route cap: the row ends short (PARTIAL), Google is blocked for the invocation, the next runs go on
                reason = f"aistudio cap: {exc}"
                blocked[run["provider"]] = reason
                records = _read_files(ctx.run_dir)
                save(run, {**entry, "status": PARTIAL, "reason": reason, **done_counts(records, run)})
                logger.error("%s: run %s PARTIAL (the next runs go on): %s", STAGE, run["id"], reason)
                print(f"[{run['id']}] {route_cap.cap_line(ctx.cost)}", flush=True)
                continue
            save(run, {**entry, "status": "stopped", "reason": f"budget cap: {exc}"})
            _record_stage(ctx, STAGE, {"status": "stopped", "reason": f"budget cap: {exc}", "runs": outcome,
                                       "records": stage_counts(_read_files(ctx.run_dir)), "tie_rule": tie})
            return pilot2._budget_stop(ctx, STAGE, exc)
        except Exception as exc:
            if _is_balance_stop(exc):
                # HTTP 402 or a quota 429 (gemini.BalanceExhausted / QuotaExhausted): the row is PARTIAL, never
                # retried; Google is blocked for the rest of the invocation (memo of 2026-11-08, section 1.3)
                reason = f"{type(exc).__name__}: {exc}"
                blocked[run["provider"]] = f"Gemini billing or quota stop during run {run['id']}: {exc}"
                records = _read_files(ctx.run_dir)
                save(run, {**entry, "status": PARTIAL, "reason": reason, **done_counts(records, run)})
                logger.error("%s: run %s PARTIAL (billing or quota stop; the next runs go on): %s", STAGE, run["id"],
                             reason)
                if is_aistudio(run):
                    print(f"[{run['id']}] {route_cap.cap_line(ctx.cost)}", flush=True)
                continue
            # judge.CacheNotRead, a provider error left after the client's retries (2026-11-08: the qwen shown run's
            # 429s), or an unexpected error: the run is failed, the stage goes on with the next runs and ends failed;
            # a new invocation resumes the failed run from its records (before 2026-11-08 the stage stopped here)
            reason = f"{type(exc).__name__}: {exc}"
            if run["spec"]["provider"] == "gemini" and _is_permission_denied(exc):
                # the policy refusal (HTTP 403): no retry, no other route; the provider is blocked for the invocation
                blocked[run["provider"]] = f"Gemini HTTP 403 / permission denied during run {run['id']}: {exc}"
                reason = f"not run (provider restriction): {reason}"
            records = _read_files(ctx.run_dir)
            save(run, {**entry, "status": "failed", "reason": reason, "records": present(records, run)})
            logger.error("%s: run %s FAILED (the next runs go on): %s", STAGE, run["id"], reason, exc_info=exc)
            failed_runs.append(run["id"])
            continue
        records = _read_files(ctx.run_dir)
        if cut:
            fit = entry["aistudio"]["prefix"]
            save(run, {**entry, "status": PARTIAL, **done_counts(records, run),
                       "reason": (f"aistudio cap: ran the first {fit['n_tasks_run']} of {fit['n_tasks_planned']} "
                                  "tasks (the longest prefix whose re-priced cost fit under the remaining cap)")})
        else:
            save(run, {**entry, "status": "done", **done_counts(records, run)})
        print(f"[{run['id']}] {ctx.cost.totals_line()}", flush=True)
        if is_aistudio(run):
            print(f"[{run['id']}] {route_cap.cap_line(ctx.cost)}", flush=True)
    counts = stage_counts(records)
    # every run of the stage: this invocation's outcome, else the status an earlier invocation recorded (--only
    # leaves the others untouched), else "not run"; the stage is failed while any run is
    all_runs = {r["id"]: outcome.get(r["id"], status.get(r["id"], {}).get("status", "not run")) for r in runs}
    failed_all = [rid for rid, st in all_runs.items() if st == "failed"]
    stage_status: dict[str, Any] = {"status": "failed" if failed_all else "done", "runs": all_runs,
                                    "records": counts, "tie_rule": tie}
    if only is not None:
        stage_status["only"] = sorted(only)
    if failed_all:
        stage_status["reason"] = (f"failed runs {failed_all} (this invocation: {failed_runs}; the other runs went on; "
                                  "a new invocation resumes the failed ones from their records)")
    _record_stage(ctx, STAGE, stage_status)
    print(f"[{STAGE}] {ctx.cost.totals_line()}", flush=True)
    print(f"[{STAGE}] records (set {SET_NAME}, condition {CONDITION}): "
          + ", ".join(f"{f} {n}" for f, n in counts.items()), flush=True)
    return EXIT_FAILED if failed_runs else EXIT_OK

"""The runner of the sprint's control run (sprint_control/README.md). Three stages:

- plan (no call): the plan per grader, the spend so far, and the check that the prompt and the settings are those of
  the main run. For each judge and each of the 60 placebos it builds the request that the run would send and looks
  its key up in the main run's disk cache, without any way to send it; a hit proves that the prompt, the model name
  and every request-shaping setting are byte-identical to the main run's call of 2026-11-08. The same lookup is made
  for the reward models' inputs. The result goes to results/sprint_control/prompt_check.json.
- judges (paid): each judge of config.yaml scores C, G and the placebo of every built set, one call per message,
  without the transcript -> results/sprint_control/pointwise_calls.jsonl.
- rm (local GPU): the five reward models score the same messages -> results/sprint_control/rm_scores.jsonl. Since
  the amendment of 2026-11-14 (README section 11) a reward model is never skipped: for each one the stage confirms
  that the weights are the files of the main run (weights.py), scores the placebo first and compares it with the
  main run's stored scores, and scores C and G only when the largest difference is at most 0.05. A model that
  cannot be scored, or that stops at one of these checks, ends the stage with an error; the judges' stage and
  analyze.py then refuse to start, and the author is told.

What makes the calls those of the main run: the judges' entries of configs/models.yaml and its price table are read
unchanged; the clients, the prompt (overclaim.graders.pointwise), the task context (overclaim.graders.judge.
build_context) and the per-task loop (overclaim.experiments.e1b_pointwise.run_pointwise) are the main run's code,
called with its arguments (set name v4_full_run, replicate 0, no prompt cache). The number of threads per provider
is config.yaml's and shapes no request.

What differs, on purpose: the records, the cost log and the disk cache are this run's own, under
results/sprint_control/. The cache starts empty, so the second score of the placebo is a new call and never the
reply of 2026-11-08 read back; the stage stops if a cache entry is older than this run's first launch.

Rules of the brief that the runner enforces itself:
- the caps of config.yaml (12 USD Anthropic, 3 USD OpenRouter, 4 USD Google Cloud). Before a call, the cost log adds
  the price of that call's input and of the largest reply the judge may give to the money already booked
  (CostLogger.check_before) and does not make a call that would pass its limit; the run then stops. The check does
  not know the calls that other threads have in flight, so the limit given to the cost log is the cap minus a
  margin for them (config.yaml, in_flight_margin_usd): the brief's caps themselves are then never passed;
- a Gemini judge runs only with route vertex, only through application default credentials of the project of the
  check, and only when the one check of that route answered (vertex_check.py). No other route is ever tried. After
  a refusal by the route (HTTP 403, or an exhausted balance or quota) no further request is sent to it, in this
  invocation or a later one: not by a new call, and not by a retry of a call already under way;
- GOOGLE_API_KEY and the variables that would send a provider's requests to another host, or print request
  headers, must be unset;
- the paid stages need the start line (--go "GO control run"), refuse to run while a tracked file that shapes a
  call differs from the commit, refuse a second copy of themselves (analyze.py takes the same lock), and refuse to
  run once analyze.py has started on this run's records (the declaration's rule: no scoring call after the first
  analysis);
- a judge's run that stops on an error is started again by the runner itself, for its missing records only, until
  it is complete, two restarts are used, or a cap would be passed. No person chooses which judge is restarted: the
  stage always takes every judge of config.yaml in order, and there is no option to select one;
- keys are read from the environment by the clients and are never printed, logged or written.

Message text is data: it is placed into prompts as text and never executed.

Run:  uv run python sprint_control/run_control.py --stage plan
      uv run python sprint_control/run_control.py --stage judges --go "GO control run"
      uv run python sprint_control/run_control.py --stage rm --go "GO control run"
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import fcntl
import gc
import logging
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any

import yaml

import control_common as K
import weights as W

logger = logging.getLogger("sprint_control")
START_LINE = "GO control run"
EXIT_OK, EXIT_FAILED, EXIT_BUDGET = 0, 1, 3
COST_PROVIDER = {"anthropic": "anthropic", "openrouter": "openrouter", "gemini": "google"}
BRIEF_CAPS = {"anthropic_usd_cap": 12.0, "openrouter_usd_cap": 3.0, "google_usd_cap": 4.0}
# Variables that must be unset when a call is made: a key that is never used, hosts that would replace a provider's
# own, and the switch that makes the Google SDK print request headers.
FORBIDDEN_ENV = ("GOOGLE_API_KEY", "GOOGLE_VERTEX_BASE_URL", "GOOGLE_GEMINI_BASE_URL", "ANTHROPIC_BASE_URL",
                 "GOOGLE_GENAI_DEBUG", "GOOGLE_GENAI_HTTP_CLIENT", "GOOGLE_GENAI_CLIENT_MODE", "HTTPS_PROXY",
                 "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")
# What a paid stage must find unchanged against the commit: everything that shapes a call or an analysis.
COMMITTED_PATHS = ("sprint_control", "src", "configs", "scripts", "reanalysis", "data/processed", "results/full_run",
                   "pyproject.toml", "uv.lock")
MAX_LAUNCHES = 3            # a judge's run: the first launch and two restarts after an error
RESTART_WAIT_S = 30.0       # the pause before a restart
# Amendment 1 of 2026-11-14: a reward model stops before C and G when its second score of the placebo differs from
# the score the main run stored by more than this on any task (the model's own score units, as stored).
PLACEBO_MAX_ABS_DIFF = 0.05
ANALYSIS_FILE = K.OUT / "analysis.json"
# Written by analyze.py before it reads a judge's record or computes anything. The one thing it reads earlier is
# the reward models' status and records, for the check that all five are scored (amendment of 2026-11-14).
ANALYSIS_STARTED_FILE = K.OUT / "analysis_started.json"
LOCK_FILE = K.OUT / ".lock"
_status_lock = threading.Lock()


# ---------------------------------------------------------------------------------------------
# Config and inputs
# ---------------------------------------------------------------------------------------------


def load_config() -> dict:
    """config.yaml with the main run's model registry beside it: cfg["specs"] (the judges' entries of
    configs/models.yaml), cfg["prices"] and cfg["rm_specs"]. ValueError when the config asks for anything the brief
    does not allow."""
    cfg = yaml.safe_load(K.CONFIG.read_text(encoding="utf-8"))
    models = yaml.safe_load(K.MODELS_CONFIG.read_text(encoding="utf-8"))
    cfg["specs"], cfg["prices"] = models["judges"], models["prices_usd_per_mtok"]
    cfg["rm_specs"] = models[cfg["reward_models"]]
    problems = []
    if cfg.get("run_name") != K.RUN_NAME or cfg.get("condition") != K.CONDITION:
        problems.append(f"run_name must be {K.RUN_NAME} and condition {K.CONDITION}")
    if tuple(cfg.get("variants") or ()) != K.NEW_VARIANTS:
        problems.append(f"variants must be {list(K.NEW_VARIANTS)}")
    if cfg.get("budget") != BRIEF_CAPS:
        problems.append("budget must be the caps of the brief: 12 USD Anthropic, 3 USD OpenRouter, 4 USD Google")
    margin = cfg.get("in_flight_margin_usd") or {}
    if set(margin) != {"anthropic", "openrouter", "google"} or not all(
            isinstance(v, (int, float)) and 0 < v < 1 for v in margin.values()):
        problems.append("in_flight_margin_usd must give a margin between 0 and 1 USD for anthropic, openrouter and "
                        "google")
    ids = [r.get("id") for r in cfg.get("judges") or []]
    if not ids or len(ids) != len(set(ids)):
        problems.append("judges must be a non-empty list of runs with distinct ids")
    for run in cfg.get("judges") or []:
        if run.get("judge") not in cfg["specs"]:
            problems.append(f"run {run.get('id')}: unknown judge {run.get('judge')!r}")
            continue
        spec = judge_spec(cfg, run)
        if spec["provider"] == "gemini" and spec.get("route") != "vertex":
            problems.append(f"run {run['id']}: a Gemini judge must carry route: vertex (never aistudio, never "
                            "another provider)")
        if spec["provider"] not in COST_PROVIDER:
            problems.append(f"run {run['id']}: provider {spec['provider']!r} is not one of the main run's")
    if problems:
        raise ValueError("invalid sprint_control/config.yaml:\n" + "\n".join(problems))
    return cfg


def judge_spec(cfg: dict, run: dict) -> dict:
    """The judge's entry of configs/models.yaml, unchanged, with the run's route for a Gemini judge."""
    spec = copy.deepcopy(cfg["specs"][run["judge"]])
    if "route" in run:
        spec["route"] = run["route"]
    return spec


def inputs() -> tuple[list[dict], list[dict]]:
    """(tasks, sets): the frozen task dicts of the main run in the frozen order, restricted to the built sets, and
    for each a set {"task_id", "variants": {"C", "G", "P"}}. The placebo is the frozen one; ValueError when a built
    record's hashes do not match its texts or the frozen placebo."""
    from overclaim.experiments import full_run_pointwise as frp

    main_cfg = yaml.safe_load(K.MAIN_CONFIG.read_text(encoding="utf-8"))
    tasks, frozen = frp.frozen_inputs(main_cfg)
    by_task = {s["task_id"]: s for s in frozen}
    built = K.read_jsonl(K.MESSAGES_FILE)
    if not built:
        raise ValueError(f"{K.MESSAGES_FILE} is missing or empty; run build_messages.py first")
    sets = []
    for rec in built:
        placebo = by_task[rec["task_id"]]["variants"]["P"]
        texts = {"C": rec["C"], "G": rec["G"], "P": placebo}
        bad = [k for k, v in texts.items() if rec["sha256"][k] != K.sha256_text(v)]
        if bad:
            raise ValueError(f"{rec['task_id']}: the recorded sha256 of {bad} does not match the text")
        sets.append({"task_id": rec["task_id"], "variants": texts})
    keep = {s["task_id"] for s in sets}
    return [t for t in tasks if t["task_id"] in keep], sets


def vertex_answered() -> tuple[bool, str]:
    """(whether the one check of the vertex route answered, the reason when not)."""
    check = K.read_json(K.VERTEX_CHECK_FILE)
    if check is None:
        return False, "the check of the vertex route was not made (vertex_check.py)"
    if check.get("answered") is True:
        return True, ""
    return False, f"the check of the vertex route did not answer: {check.get('outcome')}"


def limits(cfg: dict) -> dict:
    """The limits given to the cost log: each cap of the brief minus the margin for calls in flight on other
    threads, so that the cap itself is never passed."""
    return {f"{p}_usd_cap": float(cfg["budget"][f"{p}_usd_cap"]) - float(cfg["in_flight_margin_usd"][p])
            for p in ("anthropic", "openrouter", "google")}


def forbidden_environment() -> list[str]:
    """The variables of FORBIDDEN_ENV that are set (must be none before any call)."""
    return [name for name in FORBIDDEN_ENV if os.environ.get(name)]


def committed_or_exit() -> str:
    """The commit a call is made from. SystemExit while a file of COMMITTED_PATHS differs from the commit (changed,
    staged or untracked)."""
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *COMMITTED_PATHS], cwd=K.REPO, text=True,
                           capture_output=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit("these files differ from the commit; commit them before any call:\n" + dirty)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=K.REPO, text=True, capture_output=True,
                          check=True).stdout.strip()


def update_status(change) -> dict:
    """Read runs_status.json, apply change(status), write it back; one thread at a time."""
    with _status_lock:
        status = K.read_json(K.RUNS_STATUS_FILE, {})
        change(status)
        K.write_json(K.RUNS_STATUS_FILE, status)
        return status


def only_copy_or_exit():
    """An exclusive lock for a paid stage (kept until the process ends). SystemExit when another copy holds it."""
    K.OUT.mkdir(parents=True, exist_ok=True)
    handle = open(LOCK_FILE, "w", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("another copy of a paid stage is running") from None
    return handle


def analysed_or_exit() -> None:
    """SystemExit once analyze.py has started on this run's records: no scoring call is made after the first
    analysis (declaration, section 5)."""
    for path in (ANALYSIS_STARTED_FILE, ANALYSIS_FILE):
        if path.exists():
            raise SystemExit(f"{path.name} exists: analyze.py has run on this run's records, and no scoring call "
                             "is made after the first analysis (declaration, section 5)")


def paid_stage_or_exit() -> str:
    """Every check a paid stage makes before its first call; returns the commit."""
    analysed_or_exit()
    commit = committed_or_exit()
    bad = forbidden_environment()
    if bad:
        raise SystemExit(f"these variables must be unset: {', '.join(bad)}")
    return commit


def started_at(commit: str) -> str:
    """The time of this run's first paid launch, written once to runs_status.json and kept."""
    def change(status: dict) -> None:
        status.setdefault("_started", {"at": datetime.now(timezone.utc).isoformat(), "commit": commit})

    return update_status(change)["_started"]["at"]


def cache_is_this_runs(start: str) -> list[str]:
    """Cache entries of this run's cache that are older than its first launch (must be none): such an entry would
    be a reply this run did not ask for."""
    import json

    old = []
    if K.CACHE_DIR.exists():
        for path in K.CACHE_DIR.rglob("*.json"):
            if path.name.startswith(".tmp-"):          # a temp file of the cache's atomic write
                continue
            created = json.loads(path.read_text(encoding="utf-8")).get("created_at", "")
            if created < start:
                old.append(str(path.relative_to(K.REPO)))
    return old


# ---------------------------------------------------------------------------------------------
# Stage plan: no call
# ---------------------------------------------------------------------------------------------


class _Capture:
    """Stands in for the disk cache during the prompt check: it records the (model, payload) that a client looks up
    and answers with a canned reply, so the client returns before it reaches its SDK. It never stores anything."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, dict]] = []

    def get(self, model: str, payload: dict) -> dict:
        from overclaim.graders.judge import LLMResponse

        self.seen.append((model, copy.deepcopy(payload)))
        return dataclasses.asdict(LLMResponse(text='{"score": 5, "reason": "canned reply of the prompt check"}',
                                              model="none", input_tokens=0, output_tokens=0, cost_usd=0.0,
                                              cached=True, stop_reason="canned", request={}))

    def set(self, model: str, payload: dict, value: dict) -> None:
        raise AssertionError("the prompt check must never store a reply")


class _NoSdk:
    """Stands in for a provider SDK during the prompt check: any use is an error, so no request can be sent."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the prompt check must not reach a provider SDK (attribute {name!r})")


def offline_client(spec: dict, prices: dict) -> tuple[Any, _Capture]:
    """The client the run would use for this judge (judge.make_client's arguments), with the capture object in the
    place of the disk cache and no SDK behind it. No credential is read and nothing can be sent."""
    from overclaim.graders import gemini, judge

    capture, sdk = _Capture(), _NoSdk()
    cost = judge.CostLogger(str(K.OUT), prices=prices, caps={})    # only read by the clients' constructors
    provider = spec["provider"]
    if provider == "anthropic":
        client: Any = judge.AnthropicClient(spec["model"], effort=spec.get("effort", "low"),
                                            max_tokens=spec.get("max_tokens", 2000), cache=capture, cost=cost,
                                            prices=prices, sdk_client=sdk, thinking=spec.get("thinking"))
    elif provider == "openrouter":
        client = judge.OpenAICompatClient("openrouter", spec["model"], judge.OPENROUTER_BASE_URL, "not-a-key",
                                          temperature=spec.get("temperature", 0.0),
                                          max_tokens=spec.get("max_tokens", 400), extra_body=spec.get("extra_body"),
                                          prompt_suffix=spec.get("prompt_suffix", ""), cache=capture, cost=cost,
                                          prices=prices, sdk_client=sdk, backoff=spec.get("backoff"))
    elif provider == "gemini":
        client = gemini.GeminiClient(spec["model"], route=spec["route"],
                                     thinking_level=spec.get("thinking_level", "low"),
                                     max_output_tokens=spec.get("max_output_tokens", 1500), cache=capture, cost=cost,
                                     prices=prices, sdk_client=sdk, route_label="vertex:adc")
    else:
        raise ValueError(f"unknown provider {provider!r}")
    return client, capture


def planned_payloads(spec: dict, prices: dict, tasks: list[dict], sets: list[dict], variant: str) -> list[dict]:
    """Per task, the cache key fields of the first request the run would send for `variant`: {"task_id", "model",
    "payload"}. Built by the run's own code path (pointwise.score_message on the judge's client)."""
    from overclaim.graders import judge, pointwise

    client, capture = offline_client(spec, prices)
    by_task = {s["task_id"]: s for s in sets}
    question, instruction = pointwise.question_for(pointwise.RUBRIC)
    out = []
    for task in tasks:
        before = len(capture.seen)
        pointwise.score_message(client, judge.build_context(task, K.CONDITION),
                                by_task[task["task_id"]]["variants"][variant],
                                structured=bool(spec.get("structured_output")), cache_prefix=False, replicate=0,
                                rubric=pointwise.RUBRIC, question=question, instruction=instruction)
        if len(capture.seen) != before + 1:
            raise AssertionError("one message must give exactly one request")
        model, payload = capture.seen[-1]
        out.append({"task_id": task["task_id"], "model": model, "payload": payload})
    return out


def main_run_replies() -> dict[tuple[str, str], dict]:
    """{(grader, task_id): the main run's record of the placebo without the transcript, scoring prompt of the paper's
    main rows, first sample}."""
    out = {}
    for r in K.read_jsonl(K.MAIN_RUN / "pointwise_calls.jsonl"):
        if (r["variant"] == "P" and r["condition"] == K.CONDITION and r["rubric"] == "pointwise-neutral"
                and int(r.get("replicate") or 0) == 0 and r["set"] == K.SET_NAME):
            out.setdefault((r["grader"], r["task_id"]), r)
    return out


def prompt_check(cfg: dict, tasks: list[dict], sets: list[dict]) -> dict:
    """For every judge and reward model: how many of the placebo requests of this run have their key in the main
    run's disk cache (all of them = the prompt and settings are the main run's, byte for byte), and for the judges
    how many of those cached replies are the reply the main run recorded (a record keeps the reply of the second
    attempt when the first could not be read, so a few differ) and on which day each reply was made. Nothing is sent
    and nothing is written to either cache."""
    import json

    from overclaim import render
    from overclaim.graders import rm as rm_mod
    from overclaim.graders.cache import DiskCache

    main_cache = DiskCache(str(K.MAIN_CACHE))
    recorded = main_run_replies()
    by_task = {s["task_id"]: s for s in sets}
    out: dict[str, Any] = {"main cache": str(K.MAIN_CACHE.relative_to(K.REPO)), "judges": {}, "reward models": {}}
    for run in cfg["judges"]:
        spec = judge_spec(cfg, run)
        planned = planned_payloads(spec, cfg["prices"], tasks, sets, K.PLACEBO)
        hits = same = 0
        missing = []
        dates: dict[str, int] = {}
        for item in planned:
            value = main_cache.get(item["model"], item["payload"])
            if value is None:
                missing.append(item["task_id"])
                continue
            hits += 1
            rec = recorded.get((run["judge"], item["task_id"]))
            same += bool(rec is not None and rec["raw_text"] == value["text"])
            entry = json.loads(main_cache.path(item["model"], item["payload"]).read_text(encoding="utf-8"))
            day = str(entry.get("created_at", ""))[:10]
            dates[day] = dates.get(day, 0) + 1
        out["judges"][run["judge"]] = {"placebo requests": len(planned), "key found in the main run's cache": hits,
                                       "cached reply equals the main run's record": same, "not found": missing,
                                       "date of the main run's reply (UTC)": dict(sorted(dates.items()))}
    for spec in cfg["rm_specs"]:
        model = rm_mod.RewardModel(spec["model"], dtype=spec.get("dtype", "bfloat16"),
                                   batch_size=spec.get("batch_size", 1), cache=None, name=spec["name"],
                                   quantization=spec.get("quantization"))
        hits, missing = 0, []
        for task in tasks:
            payload = {"context": render.hidden_context(task["repo"], task["issue_text"]),
                       "response": by_task[task["task_id"]]["variants"][K.PLACEBO]}
            if main_cache.get(model.cache_model, payload) is None:
                missing.append(task["task_id"])
            else:
                hits += 1
        out["reward models"][spec["name"]] = {"placebo inputs": len(tasks),
                                              "key found in the main run's cache": hits, "not found": missing}
    return out


def stage_plan(cfg: dict) -> int:
    from overclaim.graders import judge

    tasks, sets = inputs()
    cost = judge.CostLogger(str(K.OUT), prices=cfg["prices"], caps=limits(cfg))
    n = len(tasks) * len(cfg["variants"])
    print(f"plan: {len(tasks)} built sets x {list(cfg['variants'])} = {n} calls per grader, one message per call, "
          f"condition {K.CONDITION}, records -> {K.POINTWISE_FILE.relative_to(K.REPO)}")
    print(f"plan: caps {cfg['budget']}; the cost log stops at {limits(cfg)} (the caps minus the margin for calls in "
          f"flight); {cost.totals_line()}")
    ok, why = vertex_answered()
    have = K.read_jsonl(K.POINTWISE_FILE)
    for run in cfg["judges"]:
        spec = judge_spec(cfg, run)
        route = f", route {spec['route']}" if spec["provider"] == "gemini" else ""
        present = sum(r["grader"] == run["judge"] for r in have)
        gate = "" if spec["provider"] != "gemini" or ok else f"; NOT RUN: {why}"
        settings = {k: v for k, v in spec.items() if k not in ("provider", "model")}
        print(f"plan: run {run['id']}: {run['judge']} ({spec['provider']}{route}), model {spec['model']}, "
              f"settings {settings}, {n} records expected ({present} present), "
              f"{cfg['max_workers'][COST_PROVIDER[spec['provider']]]} workers{gate}")
    for spec in cfg["rm_specs"]:
        print(f"plan: reward model {spec['name']}: {spec['model']}, {spec.get('quantization') or spec.get('dtype')}"
              f", {n} scores on the local GPU")
    check = prompt_check(cfg, tasks, sets)
    K.write_json(K.PROMPT_CHECK_FILE, check)
    for group in ("judges", "reward models"):
        for name, row in check[group].items():
            total = row.get("placebo requests", row.get("placebo inputs"))
            same = row.get("cached reply equals the main run's record")
            extra = "" if same is None else f", {same} replies equal to the main run's records"
            found = row["key found in the main run's cache"]
            print(f"prompt check: {name}: {found} of {total} placebo requests found in the main run's cache{extra}")
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage judges: paid
# ---------------------------------------------------------------------------------------------


def _safe(exc: BaseException) -> str:
    """An error as text for the status file: its type and message, cut to 500 characters."""
    return f"{type(exc).__name__}: {str(exc)[:500]}"


class VertexBlocked(RuntimeError):
    """The vertex route refused a request earlier; no further request is sent to it."""


class _GuardedInteractions:
    """interactions.create of the Gemini SDK client behind the refusal flag: it raises VertexBlocked instead of
    sending once the flag is set, and it sets the flag when the route refuses."""

    def __init__(self, inner: Any, guard: "_StopAfterRefusal") -> None:
        self._inner, self._guard = inner, guard

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self._guard.check()
        try:
            return self._inner.create(*args, **kwargs)
        except Exception as exc:
            self._guard.notice(exc)
            raise


class _GuardedSdk:
    def __init__(self, inner: Any, guard: "_StopAfterRefusal") -> None:
        self._inner = inner
        self.interactions = _GuardedInteractions(inner.interactions, guard)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _StopAfterRefusal:
    """The Gemini client of a run, wrapped. After the route's first refusal (HTTP 403, an exhausted balance, or a
    quota that is not a per-minute limit) nothing more is sent to it: a new call raises VertexBlocked at once, a
    retry of a call already under way raises it before sending (the client's SDK is replaced by a guarded one, and
    VertexBlocked is not an error the client retries), and a resend inside the SDK is stopped by a hook on its HTTP
    client. The refusal is written to runs_status.json so that no later invocation sends to the route either."""

    def __init__(self, client: Any, run_id: str) -> None:
        self._client, self._run_id = client, run_id
        self._refused = threading.Event()
        self._why = ""
        self.route_label = client.route_label
        http = getattr(getattr(client.sdk, "_api_client", None), "_httpx_client", None)
        if http is not None:                               # the SDK's own resends go through this client

            def stop(request: Any) -> None:
                self.check()

            http.event_hooks["request"].append(stop)
        client.sdk = _GuardedSdk(client.sdk, self)

    def check(self) -> None:
        if self._refused.is_set():
            raise VertexBlocked(self._why)

    def notice(self, exc: BaseException) -> None:
        """Set the flag, once, when the error is a refusal by the route."""
        from overclaim.graders import gemini

        if isinstance(exc, VertexBlocked) or self._refused.is_set():
            return
        code = gemini.error_code(exc)
        refused = (gemini.is_permission_denied(exc) or isinstance(exc, gemini.BalanceExhausted) or code == 402
                   or (code == 429 and gemini.classify_429(exc) == "quota"))
        if refused:
            self._why = f"the vertex route refused a request during run {self._run_id}: {_safe(exc)}"
            self._refused.set()
            update_status(lambda status: status.setdefault(
                "_vertex_blocked", {"reason": self._why, "at": datetime.now(timezone.utc).isoformat()}))

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        self.check()
        try:
            return self._client.complete(*args, **kwargs)
        except Exception as exc:
            self.notice(exc)
            raise


def stage_judges(cfg: dict) -> int:
    """Every judge of config.yaml in order. A judge whose run stops on an error is started again by this loop, for
    its missing records only, until it is complete, MAX_LAUNCHES launches are used (counted over all invocations),
    or a cap would be passed. No judge's score is read here. The reward models' status and records are read once,
    before anything else, for the check that all five are scored (amendment of 2026-11-14)."""
    from overclaim.experiments import e1b_pointwise
    from overclaim.graders import gemini, judge, pointwise
    from overclaim.graders.cache import DiskCache
    from overclaim.paths import load_env

    lock = only_copy_or_exit()                         # held until the process ends
    load_env()
    commit = paid_stage_or_exit()
    # Amendment of 2026-11-14: the reward-model stage runs first. While one of the five reward models is not
    # scored, no judge is started: the run stops there and the author is told.
    unscored = reward_models_scored(cfg)
    if unscored:
        lock.close()
        raise SystemExit("no judge is started: not all five reward models are scored (amendment of 2026-11-14)\n"
                         + "\n".join(unscored))
    start = started_at(commit)
    old = cache_is_this_runs(start)
    if old:
        raise SystemExit(f"{len(old)} entries of this run's cache are older than its first launch, e.g. {old[:3]}")
    tasks, sets = inputs()
    cache = DiskCache(str(K.CACHE_DIR))
    cost = judge.CostLogger(str(K.OUT), prices=cfg["prices"], caps=limits(cfg))
    question, instruction = pointwise.question_for(pointwise.RUBRIC)
    variants = tuple(cfg["variants"])
    expected = len(tasks) * len(variants)
    failed = False

    def present(name: str) -> int:
        return sum(r["grader"] == name for r in K.read_jsonl(K.POINTWISE_FILE))

    for run in cfg["judges"]:
        spec = judge_spec(cfg, run)
        provider = COST_PROVIDER[spec["provider"]]
        entry: dict[str, Any] = {"judge": run["judge"], "provider": spec["provider"], "route": spec.get("route"),
                                 "variants": list(variants), "n_tasks": len(tasks), "expected_records": expected,
                                 "commit": commit}

        def finish(update: dict, entry: dict = entry, run: dict = run) -> None:
            def change(status: dict) -> None:
                earlier = status.get(run["id"], {})
                history = list(earlier.get("history", []))
                if update["status"] != "running":
                    history.append({**{k: update[k] for k in ("status", "reason", "records") if k in update},
                                    "at": datetime.now(timezone.utc).isoformat()})
                status[run["id"]] = {**entry, "launches": earlier.get("launches", 0), **update, "history": history,
                                     "at": datetime.now(timezone.utc).isoformat()}

            update_status(change)
            why = f" ({update['reason']})" if update.get("reason") else ""
            print(f"[{run['id']}] {update['status']}{why}; {cost.totals_line()}", flush=True)

        while True:                                    # one pass per launch of this judge
            analysed_or_exit()
            before = K.read_json(K.RUNS_STATUS_FILE, {})
            launches = int(before.get(run["id"], {}).get("launches", 0))
            have = present(run["judge"])
            if have >= expected and before.get(run["id"], {}).get("status") == "done":
                print(f"[{run['id']}] all {expected} records present; not launched again", flush=True)
                break
            if spec["provider"] == "gemini":
                ok, why = vertex_answered()
                check = K.read_json(K.VERTEX_CHECK_FILE) or {}
                project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
                if not ok:
                    finish({"status": "not run", "reason": why, "records": have})
                    break
                if "_vertex_blocked" in before:
                    finish({"status": "not run", "reason": before["_vertex_blocked"]["reason"], "records": have})
                    break
                if (gemini.route_credential("vertex") != "adc" or project != cfg["vertex_check"]["project"]
                        or project != check.get("project")):
                    finish({"status": "not run", "records": have,
                            "reason": "the vertex route would not use application default credentials of the "
                                      "project of the check"})
                    failed = True
                    break
            if launches >= MAX_LAUNCHES:
                finish({"status": "not run", "records": have,
                        "reason": f"already launched {launches} times (the first launch and two restarts)"})
                failed = True
                break
            client = judge.make_client(spec, cache, cost, cfg["prices"])
            if client is None:
                finish({"status": "not run", "reason": f"no credential for provider {spec['provider']}",
                        "records": have})
                failed = True
                break
            if spec["provider"] == "gemini":
                if client.route_label != "vertex:adc":
                    finish({"status": "not run", "reason": f"the Gemini client came up on {client.route_label}",
                            "records": have})
                    failed = True
                    break
                client = _StopAfterRefusal(client, run["id"])
            finish({"status": "running", "records": have, "launches": launches + 1})
            try:
                records = e1b_pointwise.run_pointwise(
                    tasks, sets, client, run["judge"], K.CONDITION, str(K.POINTWISE_FILE), K.RUN_NAME,
                    set_name=K.SET_NAME, structured=bool(spec.get("structured_output")),
                    max_workers=int(cfg["max_workers"][provider]), use_prompt_cache=False, replicate=0,
                    variants=variants, rubric=pointwise.RUBRIC, question=question, instruction=instruction)
            except judge.BudgetExceeded as exc:
                finish({"status": "stopped", "reason": f"cap: {exc}", "records": present(run["judge"])})
                print("a cap would be passed; the run stops here", flush=True)
                return EXIT_BUDGET
            except Exception as exc:                   # reported, never worked around: no other route is tried
                finish({"status": "failed", "reason": _safe(exc), "records": present(run["judge"])})
                if launches + 1 < MAX_LAUNCHES and "_vertex_blocked" not in K.read_json(K.RUNS_STATUS_FILE, {}):
                    print(f"[{run['id']}] restart {launches + 1} of {MAX_LAUNCHES - 1} in {RESTART_WAIT_S:.0f} s, "
                          "for the missing records only", flush=True)
                    time.sleep(RESTART_WAIT_S)
                    continue
                failed = True
                break
            finish({"status": "done", "records": len(records),
                    "n_invalid": sum(r["score"] is None for r in records),
                    "n_second_attempts": sum(r["parse_attempts"] > 1 for r in records),
                    "n_read_from_this_runs_cache": sum(bool(r["cached"]) for r in records),
                    "models_seen": sorted({str(r["model"]) for r in records}),
                    "routes_seen": sorted({str(r["route"]) for r in records if r.get("route")}),
                    "providers_seen": sorted({str(r["upstream_provider"]) for r in records
                                              if r.get("upstream_provider")})})
            break
    lock.close()
    return EXIT_FAILED if failed else EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage rm: the reward models on the local GPU
# ---------------------------------------------------------------------------------------------


def main_run_placebo(name: str) -> dict[str, float]:
    """{task: the placebo score that the main run stored for this reward model} (the first record of a task)."""
    out: dict[str, float] = {}
    for r in K.read_jsonl(K.MAIN_RUN / "rm_scores.jsonl"):
        if (r.get("grader") == name and r.get("condition") == K.CONDITION and r.get("set") == K.SET_NAME
                and r.get("variant") == K.PLACEBO and r["task_id"] not in out):
            out[r["task_id"]] = float(r["score"])
    return out


def main_run_placebo_tokens(name: str) -> dict[str, int]:
    """{task: the number of input tokens the main run recorded for this reward model's placebo score}."""
    out: dict[str, int] = {}
    for r in K.read_jsonl(K.MAIN_RUN / "rm_scores.jsonl"):
        if (r.get("grader") == name and r.get("condition") == K.CONDITION and r.get("set") == K.SET_NAME
                and r.get("variant") == K.PLACEBO and r["task_id"] not in out):
            out[r["task_id"]] = int(r["input_tokens"])
    return out


def placebo_check(records: list[dict], stored: dict[str, float], tasks: list[str],
                  stored_tokens: dict[str, int] | None = None) -> dict:
    """Amendment 1, item 2: this run's placebo scores of a reward model against the main run's stored ones, per
    task, in the model's own score units. "ok" needs a comparison on every task and a largest absolute difference
    of at most PLACEBO_MAX_ABS_DIFF. With `stored_tokens` it also counts the tasks whose input has another number
    of tokens than in the main run (for reading a stop; it does not decide)."""
    now: dict[str, Any] = {}
    tokens_now: dict[str, Any] = {}
    for r in records:
        if r.get("variant") == K.PLACEBO and r["task_id"] not in now:
            now[r["task_id"]] = r.get("score")
            tokens_now[r["task_id"]] = r.get("input_tokens")

    def number(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    not_compared = [t for t in tasks if not number(stored.get(t)) or not number(now.get(t))]
    diffs = {t: abs(float(now[t]) - float(stored[t])) for t in tasks if t not in not_compared}
    worst = max(diffs, key=lambda t: diffs[t]) if diffs else None
    largest = diffs[worst] if worst is not None else None
    return {"tasks": len(tasks), "compared": len(diffs), "not_compared": not_compared,
            "tasks_that_differ": sum(d > 0 for d in diffs.values()),
            "largest_abs_difference": largest,
            "largest_at": None if worst is None else {"task_id": worst, "main_run": stored[worst], "now": now[worst]},
            "tasks_with_another_token_count": None if stored_tokens is None else sum(
                tokens_now.get(t) != stored_tokens.get(t) for t in tasks),
            "limit": PLACEBO_MAX_ABS_DIFF,
            "ok": not not_compared and largest is not None and largest <= PLACEBO_MAX_ABS_DIFF}


def stage_rm(cfg: dict) -> int:
    """The main run's reward-model stage (full_run_pointwise.stage_rm) on C, G and the placebo, with this run's
    empty cache, so that every score is computed again and none is read back from 2026-11-08.

    Amendment 1 of 2026-11-14 (README section 11). No model is skipped: a model that does not fit, does not load,
    whose weights are not confirmed, or whose placebo scores differ is recorded as "stopped" (a check said no) or
    "failed" (an error while loading, checking or scoring), and the stage ends with an error after it has gone
    through the other models. Weights that are not on disk are downloaded from the Hugging Face Hub by the load,
    as in the main run. Per model, in this order: the load; the weights are confirmed as the files of the main run
    (weights.identity against the committed trace) before anything is scored; the loading check of the main run
    (two scores that are not records); the placebo is scored and compared with the main run's stored scores
    (placebo_check); only then are C and G scored. A placebo stop stands: a later invocation finds it in
    rm_status.json, and without that file it makes the comparison again from the stored records; C and G are not
    scored."""
    from overclaim import render
    from overclaim.graders import rm as rm_mod
    from overclaim.graders.cache import DiskCache
    from overclaim.paths import load_env

    lock = only_copy_or_exit()                         # held until the process ends
    load_env()
    commit = paid_stage_or_exit()
    start = started_at(commit)
    old = cache_is_this_runs(start)
    if old:
        raise SystemExit(f"{len(old)} entries of this run's cache are older than its first launch, e.g. {old[:3]}")
    tasks, sets = inputs()
    trace = W.load_trace()
    cache = DiskCache(str(K.CACHE_DIR))
    variants = tuple(cfg["variants"])
    expected = len(tasks) * len(variants)
    task_ids = [t["task_id"] for t in tasks]
    first = tasks[0]
    status: dict[str, dict] = K.read_json(K.RM_STATUS_FILE, {})
    failed = []                                        # stopped or failed: either way the stage does not end OK

    def free_gpu() -> None:
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for spec in cfg["rm_specs"]:
        name = spec["name"]
        have = sum(r["grader"] == name for r in K.read_jsonl(K.RM_SCORES_FILE))
        earlier = status.get(name, {})
        if (have >= expected and earlier.get("status") == "done" and (earlier.get("weights") or {}).get("confirmed")
                and (earlier.get("placebo_check") or {}).get("ok")):
            print(f"[rm] {name}: all {expected} scores present; not loaded again", flush=True)
            continue
        if (earlier.get("placebo_check") or {}).get("ok") is False:
            # A placebo stop stands: nothing is loaded and nothing is scored until the author decides.
            print(f"[rm] {name}: stopped earlier at the placebo check ({earlier.get('reason')}); not started again",
                  flush=True)
            failed.append(name)
            continue
        entry: dict[str, Any] = {"model": spec["model"], "dtype": spec.get("dtype", "bfloat16"),
                                 "quantization": spec.get("quantization"), "variants": list(variants),
                                 "expected_records": expected, "commit": commit}

        def record(update: dict) -> None:
            status[name] = {**entry, **update, "at": datetime.now(timezone.utc).isoformat()}
            K.write_json(K.RM_STATUS_FILE, status)
            print(f"[rm] {name}: {update['status']}" + (f": {update['reason']}" if update.get("reason") else ""),
                  flush=True)

        def present() -> int:
            return sum(r["grader"] == name for r in K.read_jsonl(K.RM_SCORES_FILE))

        try:
            model = rm_mod.RewardModel(spec["model"], dtype=spec.get("dtype", "bfloat16"),
                                       batch_size=spec.get("batch_size", 1), cache=cache, name=name,
                                       quantization=spec.get("quantization"))
            fits, why = model.check_fits()
            entry["fit"] = why
            disk_fits, disk_why = (model.check_disk_fits() if fits else (False, ""))
        except Exception as exc:
            record({"status": "failed", "reason": _safe(exc), "records": have})
            failed.append(name)
            continue
        if not fits:
            record({"status": "stopped", "reason": why, "records": have})
            failed.append(name)
            continue
        entry["weights_bytes"] = round(model.weights_gib * rm_mod.GIB)
        entry["disk"] = disk_why
        if not disk_fits:
            record({"status": "stopped", "reason": disk_why, "records": have})
            failed.append(name)
            del model
            continue
        rm_mod.reset_peak_memory()
        t0 = time.monotonic()
        try:
            model.load()                               # downloads the weights from the Hub when they are not on disk
            entry["load_seconds"] = round(time.monotonic() - t0, 1)
            # Amendment 1, item 1: the weights are the files of the main run, or nothing is scored.
            entry["weights"] = W.identity(spec["model"], trace[name], entry["weights_bytes"])
        except Exception as exc:
            record({"status": "failed", "reason": _safe(exc), "records": have})
            failed.append(name)
            del model
            free_gpu()
            continue
        if not entry["weights"]["confirmed"]:
            record({"status": "stopped", "records": have,
                    "reason": "the weights are not confirmed as the files of the main run: "
                              + entry["weights"]["reason"]})
            failed.append(name)
            del model
            free_gpu()
            continue
        records: list[dict] = []
        try:
            model.sanity_check(render.hidden_context(first["repo"], first["issue_text"]),
                               first["final_message_raw"])
            # Amendment 1, item 2: the placebo first, against the main run's stored scores.
            t0 = time.monotonic()
            placebo = rm_mod.run_scores(tasks, sets, model, K.CONDITION, str(K.RM_SCORES_FILE), K.RUN_NAME,
                                        set_name=K.SET_NAME, variants=(K.PLACEBO,))
            check = placebo_check(placebo, main_run_placebo(name), task_ids, main_run_placebo_tokens(name))
            entry["placebo_check"] = check
            if check["ok"]:
                print(f"[rm] {name}: placebo check passed: largest absolute difference "
                      f"{check['largest_abs_difference']}, {check['tasks_that_differ']} of {check['compared']} "
                      "tasks differ", flush=True)
                records = rm_mod.run_scores(tasks, sets, model, K.CONDITION, str(K.RM_SCORES_FILE), K.RUN_NAME,
                                            set_name=K.SET_NAME, variants=variants)
        except Exception as exc:                       # an error while checking or scoring: failed, never skipped
            record({"status": "failed", "reason": _safe(exc), "records": present()})
            failed.append(name)
            del model
            free_gpu()
            continue
        if not check["ok"]:
            record({"status": "stopped", "records": present(),
                    "reason": f"the placebo's scores differ from the main run's: largest absolute difference "
                              f"{check['largest_abs_difference']} on {check['compared']} of {check['tasks']} "
                              f"tasks ({check['tasks_that_differ']} differ, "
                              f"{len(check['not_compared'])} could not be compared; the limit is "
                              f"{check['limit']}); C and G are not scored"})
            failed.append(name)
            del model
            free_gpu()
            continue
        record({"status": "done", "score_seconds": round(time.monotonic() - t0, 1),
                "peak_memory_gib": rm_mod.peak_memory_gib(), "records": len(records),
                "n_read_from_this_runs_cache": sum(bool(r["cached"]) for r in records)})
        del model
        free_gpu()
    lock.close()
    if failed:
        print(f"[rm] not scored: {', '.join(failed)} (stopped or failed; see rm_status.json). No model is skipped: "
              "no judge is started and the analysis does not begin until all five are scored; the author is told "
              "(amendment of 2026-11-14).", flush=True)
    return EXIT_FAILED if failed else EXIT_OK


def judges_stage_ended(cfg: dict) -> list[str]:
    """The reasons why the analysis may not begin yet on the judges' side ([] when the judges' stage has ended):
    one per judge of the config whose run has no final entry in runs_status.json. A run is final when it is done,
    was not run (the Gemini judges after the refused check), failed after its last launch, or when the stage
    stopped at a cap. The analysis writes a marker after which no scoring call is made, so it must not start
    before the judges' stage has run to its end."""
    status = K.read_json(K.RUNS_STATUS_FILE, {})
    if any(isinstance(e, dict) and e.get("status") == "stopped" for e in status.values()):
        return []                                          # the stage stopped at a cap: nothing more is scored
    problems = []
    for run in cfg["judges"]:
        entry = status.get(run["id"])
        if entry is None:
            problems.append(f"{run['id']}: not started")
        elif entry.get("status") == "running":
            problems.append(f"{run['id']}: its run did not end (start the judges' stage again)")
        elif (entry.get("status") == "failed" and int(entry.get("launches", 0)) < MAX_LAUNCHES
              and "_vertex_blocked" not in status):
            problems.append(f"{run['id']}: failed after {entry.get('launches')} of {MAX_LAUNCHES} launches (start "
                            "the judges' stage again)")
    return problems


def reward_models_scored(cfg: dict) -> list[str]:
    """Amendment 1: the reasons why the analysis may not begin, one per reward model that is not fully scored
    ([] when all five are). A model is scored when its entry of rm_status.json is "done", its weights were
    confirmed, its placebo check passed there and passes again on the records, and every record is present."""
    status = K.read_json(K.RM_STATUS_FILE, {})
    records = K.read_jsonl(K.RM_SCORES_FILE)
    tasks = [t["task_id"] for t in inputs()[0]]            # the tasks the stage scores: every built set
    expected = len(tasks) * len(cfg["variants"])
    problems = []
    five = sorted(W.load_trace())                          # the five names of the committed trace
    if sorted(spec["name"] for spec in cfg["rm_specs"]) != five:
        problems.append(f"the reward models of the config are not the five of the committed trace: {five}")
    for spec in cfg["rm_specs"]:
        name = spec["name"]
        entry = status.get(name)
        mine = [r for r in records if r.get("grader") == name and r.get("run") == K.RUN_NAME]
        if entry is None:
            problems.append(f"{name}: not run")
        elif entry.get("status") != "done":
            problems.append(f"{name}: {entry.get('status')}: {entry.get('reason')}")
        elif not (entry.get("weights") or {}).get("confirmed"):
            problems.append(f"{name}: the weights were not confirmed as the files of the main run")
        elif not (entry.get("placebo_check") or {}).get("ok"):
            problems.append(f"{name}: the placebo check did not pass")
        elif len({(r["task_id"], r["variant"]) for r in mine}) != expected:
            problems.append(f"{name}: {len(mine)} of {expected} records")
        elif not placebo_check(mine, main_run_placebo(name), tasks)["ok"]:
            problems.append(f"{name}: the placebo's records differ from the main run's by more than "
                            f"{PLACEBO_MAX_ABS_DIFF}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", required=True, choices=("plan", "judges", "rm"))
    parser.add_argument("--go", default="", help=f'the start line, "{START_LINE}", for the paid stages')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config()
    if args.stage == "plan":
        return stage_plan(cfg)
    if args.go != START_LINE:
        raise SystemExit(f'stage {args.stage} runs only with --go "{START_LINE}" (the author\'s start line)')
    K.OUT.mkdir(parents=True, exist_ok=True)
    return stage_judges(cfg) if args.stage == "judges" else stage_rm(cfg)


if __name__ == "__main__":
    sys.exit(main())

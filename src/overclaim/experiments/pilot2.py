"""Pilot 2 stages of Experiment 1 (config schema_version 2): data -> variants -> judge -> rm -> reasons -> analysis.

Entry point: `python -m overclaim.experiments.e1_preference --config configs/e1_pilot2_coding.yaml` (that module
dispatches here when the config has `schema_version: 2`).

What is different from the session-1 runner (e1_preference.py):
- 30 pilot tasks: session 1's 20 committed tasks plus 10 `tests_narrowed` tasks sampled here;
- two matched-set designs: v1 (appended sentence, revised) and v2 (in-place edits of the agent's own message);
- a list of judge runs (judge x set x rubric x condition) executed in priority order, each resume-safe;
- budget gates: no new Anthropic run is launched once the Anthropic total has reached
  budget.anthropic_stop_launching_usd; the client-level caps (budget.*_usd_cap) stop any single call that would exceed them;
- the shown condition sends the task context + transcript as a prompt-cached prefix and fails loudly when a later call
  of a task reads nothing from the cache.

Every stage is idempotent. Trajectory content is data: it is passed between modules and never executed.
Exit codes: 0 ok; 1 a stage failed or an input is missing; 2 a required credential is missing; 3 a budget cap was hit.
"""

from __future__ import annotations

import copy
import json
import logging
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from overclaim.paths import CACHE_DIR, REPO_ROOT

log = logging.getLogger(__name__)

STAGES = ("data", "variants", "judge", "rm", "reasons", "analysis")
EXIT_OK, EXIT_FAILED, EXIT_NO_CREDENTIALS, EXIT_BUDGET = 0, 1, 2, 3
CALLS_PER_TASK = 6
# Judge provider (config) -> provider name in the cost log and in budget / max_workers keys.
COST_PROVIDER = {"anthropic": "anthropic", "openrouter": "openrouter", "gemini": "google"}
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


# ---------------------------------------------------------------------------------------------
# Config and small helpers
# ---------------------------------------------------------------------------------------------


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path) -> dict:
    """The pilot-2 config with configs/models.yaml merged in (judges, editor, classifier, reward_models, prices)
    and the rubrics loaded under cfg["rubrics"]; paths stay relative to the repo root."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(_repo_path(cfg["models_config"]), encoding="utf-8") as f:
        models = yaml.safe_load(f)
    cfg["judges"] = models["judges"]
    cfg["editor"] = models["editor"]
    cfg["classifier"] = models["classifier"]
    cfg["reward_models"] = models["reward_models"]
    cfg["prices_usd_per_mtok"] = models["prices_usd_per_mtok"]
    from overclaim.graders import judge as judge_mod

    cfg["rubrics"] = judge_mod.load_rubrics(_repo_path(cfg["rubrics_config"]))
    for run in cfg["runs"]:
        if run["judge"] not in cfg["judges"]:
            raise ValueError(f"run {run['id']}: unknown judge {run['judge']!r}")
        if run["set"] not in cfg["sets"]:
            raise ValueError(f"run {run['id']}: unknown set {run['set']!r}")
        if run["rubric"] not in cfg["rubrics"]["rubrics"]:
            raise ValueError(f"run {run['id']}: unknown rubric {run['rubric']!r}")
    ids = [run["id"] for run in cfg["runs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("run ids must be unique")
    return cfg


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class Ctx:
    """State of one invocation: paths, the disk cache and the cost logger (created on first use)."""

    def __init__(self, cfg: dict, cfg_path: Path, run_dir: Path, dry_run: bool) -> None:
        self.cfg, self.cfg_path, self.run_dir, self.dry_run = cfg, cfg_path, run_dir, dry_run
        self.processed = _repo_path(cfg["data"]["processed_dir"])
        self.funnel_path = run_dir / "funnel.json"
        self._cache = None
        self._cost = None
        self.plan: list[str] = []

    @property
    def cache(self) -> Any:
        if self._cache is None:
            from overclaim.graders.cache import DiskCache

            self._cache = DiskCache(str(CACHE_DIR))
        return self._cache

    @property
    def cost(self) -> Any:
        if self._cost is None:
            from overclaim.graders.judge import CostLogger

            self._cost = CostLogger(str(self.run_dir), prices=self.cfg["prices_usd_per_mtok"], caps=self.cfg["budget"])
        return self._cost

    def note(self, message: str) -> None:
        log.info(message)
        if self.dry_run:
            self.plan.append(message)

    def client(self, spec: dict) -> Any:
        from overclaim.graders.judge import make_client

        return make_client(spec, self.cache, self.cost, self.cfg["prices_usd_per_mtok"])

    def set_path(self, set_name: str) -> Path:
        return self.processed / self.cfg["sets"][set_name]["file"]

    def pilot_tasks(self) -> list[dict]:
        """Session 1's committed tasks followed by the added tests_narrowed tasks."""
        pilot = self.cfg["pilot"]
        return _read_jsonl(self.processed / pilot["base_tasks_file"]) + _read_jsonl(self.processed / pilot["added_tasks_file"])


def _budget_stop(ctx: Ctx, stage: str, exc: Exception) -> int:
    message = f"BUDGET CAP REACHED during the {stage} stage: {exc}"
    log.error("=" * 80)
    log.error(message)
    log.error("=" * 80)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (ctx.run_dir / "BUDGET_STOP.txt").write_text(f"{stamp}\n{message}\n", encoding="utf-8")
    return EXIT_BUDGET


# ---------------------------------------------------------------------------------------------
# Stage: data
# ---------------------------------------------------------------------------------------------

RECONCILIATION_NOTE = (
    "Session 1's chat report gave 'finished (finish call, exit_status submit): 14,834' next to gap counts that sum to "
    "29,638. 14,834 is exit_status_submit of session 1's FIRST attempt on four row groups "
    "(results/pilot_coding/attempt1_4rg/funnel.json: rows_read 16,384, has_finish 14,838, exit_status_submit 14,834); "
    "the report paired that stale figure with the eight-row-group gap counts. The eight-row-group funnel "
    "(results/pilot_coding/funnel.json) has has_finish 29,644 and exit_status_submit 29,638, and its four gap labels "
    "(100 + 1,305 + 25,868 + 2,365) sum to exactly 29,638: the labels partition the rows that have a finish call and "
    "exit_status == submit. no_final_message (3,124) counts the rows WITHOUT a finish call and is not part of that "
    "partition (3,124 + 29,644 = 32,768 rows read). Session 1's NOTES.md says no_tests_run is '0.7%' of finished "
    "trajectories: that is 100 / 14,834, the same mix-up; 100 / 29,638 is 0.34%."
)


def stage_data(ctx: Ctx) -> int:
    """Build the three-gap-type task pool, the label counts over every downloaded row, the rendered-length
    distribution, and the tests_narrowed pilot candidates. Skips work whose outputs exist."""
    cfg, data, pilot = ctx.cfg, ctx.cfg["data"], ctx.cfg["pilot"]
    tasks_path = ctx.processed / data["tasks_file"]
    cand_path = ctx.processed / pilot["candidates_file"]
    funnel = _read_json(ctx.funnel_path, {})
    todo = []
    if not (tasks_path.exists() and "tasks_written" in funnel):
        todo.append("build tasks")
    if "label_counts_all_downloaded_rows" not in funnel:
        todo.append("count labels over stats_row_groups")
    if not (cand_path.exists() and "pilot_candidates_narrowed" in funnel):
        todo.append("sample tests_narrowed candidates")
    if ctx.dry_run:
        ctx.note(f"data: would do: {todo or 'nothing (outputs present)'}")
        return EXIT_OK
    from overclaim.data import coding

    all_groups = sorted(set(data["row_groups"]) | set(data["stats_row_groups"]))
    coding.download_raw(data["raw_dir"], all_groups, data.get("source", coding.SOURCE))
    if "build tasks" in todo:
        n = coding.build_tasks(cfg, str(tasks_path), funnel)
        log.info("data: wrote %d tasks to %s", n, tasks_path)
        _write_json(ctx.funnel_path, funnel)
    if "label_counts_all_downloaded_rows" not in funnel:
        funnel["label_counts_all_downloaded_rows"] = {
            "row_groups": data["stats_row_groups"], **coding.count_labels(cfg, data["stats_row_groups"])}
        _write_json(ctx.funnel_path, funnel)
    funnel["transcript_length_distribution"] = {
        "what": "rendered transcript tokens (o200k_base, tool outputs over 2,000 tokens truncated, finish step not "
                "rendered) of the no_tests_run and tests_failed tasks written under the 60,000-token pool cap",
        **coding.transcript_length_distribution(str(tasks_path)),
    }
    funnel["transcript_length_distribution_tests_narrowed"] = {
        "what": "the same measure for the tests_narrowed tasks written under the same cap",
        **coding.transcript_length_distribution(str(tasks_path), gap_types=("tests_narrowed",)),
    }
    funnel["reconciliation"] = {"session1_report_finished": 14834, "note": RECONCILIATION_NOTE}
    # Session 1's 20 pilot tasks must come out of the rebuilt pool unchanged.
    base = _read_jsonl(ctx.processed / pilot["base_tasks_file"])
    wanted = {t["task_id"]: t for t in base}
    found: dict[str, bool] = {}
    with tasks_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            old = wanted.get(t["task_id"])
            if old is not None:
                found[t["task_id"]] = all(t[k] == old[k] for k in ("gap_type", "final_message_raw", "transcript_text"))
    funnel["base_task_check"] = {
        "what": "session 1's 20 pilot tasks looked up in the rebuilt pool: same gap_type, final_message_raw and transcript_text",
        "n_base_tasks": len(base), "n_found": len(found), "n_identical": sum(found.values()),
    }
    if len(found) != len(base) or not all(found.values()):
        _write_json(ctx.funnel_path, funnel)
        log.error("data: session 1's pilot tasks are not reproduced by the rebuilt pool: %s", funnel["base_task_check"])
        return EXIT_FAILED
    if "sample tests_narrowed candidates" in todo:
        funnel["pilot_candidates_narrowed"] = coding.sample_candidates(
            str(tasks_path), str(cand_path), pilot["added_gap_type"], int(pilot["n_added"]),
            int(pilot["oversample_factor"]), int(cfg["seed"]), pilot.get("prefer_resolved"),
            {t["instance_id"] for t in base})
    _write_json(ctx.funnel_path, funnel)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage: variants (v1 revised, v2 in place)
# ---------------------------------------------------------------------------------------------


def _filled_judgements(path: Path) -> int:
    """Rows of a human-check CSV that carry a human judgement or note (the columns after the variant texts)."""
    import csv

    if not path.exists():
        return 0
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    judgement_cols = [c for c in (rows[0].keys() if rows else []) if c.startswith(("is_", "only_")) or c == "notes"]
    return sum(1 for r in rows if any((r[c] or "").strip() for c in judgement_cols))


def stage_variants(ctx: Ctx) -> int:
    """Build matched_sets_v1.jsonl (+ the added pilot tasks, human_check.csv) and matched_sets_v2.jsonl
    (+ human_check_v2.csv, run_dir/v2_spans.jsonl) unless they exist.

    Every Opus call is cached on disk, so a rerun after a crash repeats no paid call. The funnel is written right after
    the set files; each human-check CSV is written when the funnel does not yet record it, and never over a file in
    which a human has filled in a judgement.
    """
    cfg, pilot = ctx.cfg, ctx.cfg["pilot"]
    v1_path, v2_path = ctx.set_path("v1"), ctx.set_path("v2")
    added_path = ctx.processed / pilot["added_tasks_file"]
    v1_check = ctx.processed / cfg["sets"]["v1"]["human_check"]
    v2_check = ctx.processed / cfg["sets"]["v2"]["human_check"]
    funnel = _read_json(ctx.funnel_path, {})
    v1_built = v1_path.exists() and added_path.exists() and "v1" in funnel
    v2_built = v2_path.exists() and "v2" in funnel
    if ctx.dry_run:
        ctx.note(f"variants: v1 {'present' if v1_built else 'would be built'}; "
                 f"v2 {'present' if v2_built else 'would be built'} (Opus 5.5 edits)")
        return EXIT_OK
    from overclaim import variants, variants_inplace
    from overclaim.graders.judge import BudgetExceeded

    vcfg = variants.load_variants_config(_repo_path(cfg["variants_config"]))

    def human_check(path: Path, key: str, write: Any) -> bool:
        """Write one human-check CSV unless funnel[key] already records it. False = refused (judgements present)."""
        if "human_check_rows" in funnel[key] and path.exists():
            return True
        filled = _filled_judgements(path)
        if filled:
            log.error("variants: %s has %d rows with human judgements; not overwriting it", path, filled)
            return False
        funnel[key]["human_check_rows"] = write()
        funnel[key]["human_check_file"] = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
        _write_json(ctx.funnel_path, funnel)
        return True

    try:
        base = _read_jsonl(ctx.processed / pilot["base_tasks_file"])
        if not v1_built:
            llm = ctx.client(cfg["claim_removal"])
            if llm is None:
                log.error("variants: ANTHROPIC_API_KEY is not set; the matched sets cannot be built")
                return EXIT_NO_CREDENTIALS
            candidates = _read_jsonl(ctx.processed / pilot["candidates_file"])
            if not candidates:
                log.error("variants: no tests_narrowed candidates; run the data stage first")
                return EXIT_FAILED
            sets, added = variants.build_pilot2_v1(base, candidates, int(pilot["n_added"]), vcfg, llm, funnel)
            _write_jsonl(added_path, added)
            _write_jsonl(v1_path, sets)
            _write_json(ctx.funnel_path, funnel)
            log.info("variants: v1: %d sets, %d added tasks; %s", len(sets), len(added), ctx.cost.totals_line())
        tasks = ctx.pilot_tasks()
        v1_sets = _read_jsonl(v1_path)
        if not human_check(v1_check, "v1", lambda: variants.write_human_check(v1_check, tasks, v1_sets)):
            return EXIT_FAILED
        if not v2_built:
            llm = ctx.client(cfg["editor"])
            if llm is None:
                log.error("variants: ANTHROPIC_API_KEY is not set; the v2 sets cannot be built")
                return EXIT_NO_CREDENTIALS
            sets, infos = variants_inplace.build_all_v2(tasks, vcfg, llm, funnel)
            _write_jsonl(ctx.run_dir / "v2_spans.jsonl", infos)
            _write_jsonl(v2_path, sets)
            _write_json(ctx.funnel_path, funnel)
            log.info("variants: v2: %d sets of %d tasks; %s", len(sets), len(tasks), ctx.cost.totals_line())
        v2_sets = _read_jsonl(v2_path)
        if not human_check(v2_check, "v2", lambda: variants_inplace.write_human_check_v2(v2_check, tasks, v2_sets)):
            return EXIT_FAILED
    except BudgetExceeded as exc:
        _write_json(ctx.funnel_path, funnel)
        return _budget_stop(ctx, "variants", exc)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage: judge (the run plan)
# ---------------------------------------------------------------------------------------------


def verify_openrouter_slugs(slugs: list[str], fetch: Any = None) -> None:
    """Fail loudly unless every slug is listed by GET https://openrouter.ai/api/v1/models (no key needed)."""
    if fetch is None:
        def fetch() -> dict:
            with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=60) as resp:  # noqa: S310 (fixed https URL)
                return json.loads(resp.read().decode("utf-8"))
    listed = {m["id"] for m in fetch()["data"]}
    missing = [s for s in slugs if s not in listed]
    if missing:
        raise RuntimeError(f"OpenRouter does not list these model slugs: {missing}")
    if any(s.endswith(":free") for s in slugs):
        raise RuntimeError("':free' model variants are not allowed")


OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{slug}/endpoints"


def openrouter_eligible_providers(slug: str, rule: dict, fetch: Any = None) -> list[dict]:
    """The endpoints of an OpenRouter model that satisfy a provider rule (Uzair's memo of 2026-11-08, section 3):
    quantization in rule["quantizations"], context_length at least rule["min_context_tokens"], not a free endpoint
    (a tag or name with "free", or a zero prompt and completion price). GET OPENROUTER_ENDPOINTS_URL (no key). Each
    entry: slug (the provider slug OpenRouter's provider.only takes: the tag before "/", e.g. "siliconflow" of
    "siliconflow/fp8"), provider_name, tag, quantization, context_length. Order: the listing's. Raises on a network
    error or an unexpected shape (the caller fails the run before any call)."""
    url = OPENROUTER_ENDPOINTS_URL.format(slug=slug)
    if fetch is None:
        def fetch() -> dict:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed https URL)
                return json.loads(resp.read().decode("utf-8"))
    data = fetch()["data"]
    endpoints = data["endpoints"] if isinstance(data, dict) else data
    wanted = {str(q).lower() for q in rule["quantizations"]}
    min_context = int(rule["min_context_tokens"])
    out = []
    for e in endpoints:
        tag = str(e.get("tag") or "")
        name = str(e.get("provider_name") or "")
        pricing = e.get("pricing") or {}
        free = ("free" in tag.lower() or "free" in name.lower()
                or (float(pricing.get("prompt") or 0) == 0 and float(pricing.get("completion") or 0) == 0))
        quant = str(e.get("quantization") or "").lower()
        context = int(e.get("context_length") or 0)
        if free or quant not in wanted or context < min_context:
            continue
        provider_slug = tag.split("/", 1)[0] if tag else name.lower()
        out.append({"slug": provider_slug, "provider_name": name, "tag": tag, "quantization": quant,
                    "context_length": context})
    return out


def run_spec(cfg: dict, run: dict) -> dict:
    """The judge spec of a run: the registry entry, with the run's route override for Gemini."""
    spec = copy.deepcopy(cfg["judges"][run["judge"]])
    if "route" in run:
        spec["route"] = run["route"]
    return spec


def credential_for(spec: dict) -> bool:
    """Whether the spec's provider (and, for Gemini, its route) has a credential in the environment."""
    from overclaim.graders.judge import _credential

    if spec["provider"] == "anthropic":
        return _credential("ANTHROPIC_API_KEY") is not None
    if spec["provider"] == "openrouter":
        return _credential("OPENROUTER_API_KEY") is not None
    if spec["provider"] == "gemini":
        from overclaim.graders.gemini import route_credential

        return route_credential(spec.get("route", "aistudio")) is not None
    return False


def run_tasks(ctx: Ctx, run: dict, tasks: list[dict], sets: list[dict]) -> list[dict]:
    """The tasks a run judges: those with a matched set; shown runs keep rendered transcripts at or under the cap;
    a run with `task_ids` (pilot 3) keeps only those tasks; then `max_tasks` cuts the list."""
    with_set = {ms["task_id"] for ms in sets}
    todo = [t for t in tasks if t["task_id"] in with_set]
    if run["condition"] == "shown":
        cap = int(ctx.cfg["shown"]["max_transcript_tokens"])
        todo = [t for t in todo if t["token_counts"]["transcript"] <= cap]
    if run.get("task_ids") is not None:
        keep = set(run["task_ids"])
        todo = [t for t in todo if t["task_id"] in keep]
    if run.get("max_tasks") is not None:
        todo = todo[: int(run["max_tasks"])]
    return todo


def gate(ctx: Ctx, run: dict, provider: str) -> str | None:
    """Reason not to launch the run now, or None. Only Anthropic runs are gated by the running total."""
    if provider != "anthropic":
        return None
    spent = ctx.cost.spent("anthropic")
    stop = float(ctx.cfg["budget"]["anthropic_stop_launching_usd"])
    if spent >= stop:
        return f"Anthropic total ${spent:.2f} has reached the stop-launching threshold ${stop:.2f}"
    limit = run.get("only_if_anthropic_below_usd")
    if limit is not None and spent >= float(limit):
        return f"Anthropic total ${spent:.2f} is not under ${float(limit):.2f} (condition of this run)"
    return None


def stage_judge(ctx: Ctx, only: list[str] | None = None) -> int:
    """Execute cfg["runs"] in order (or only the run ids in `only`). Writes judge_calls.jsonl and runs_status.json."""
    cfg = ctx.cfg
    runs = [r for r in cfg["runs"] if only is None or r["id"] in only]
    if only is not None:
        unknown = sorted(set(only) - {r["id"] for r in cfg["runs"]})
        if unknown:
            log.error("judge: unknown run ids %s", unknown)
            return EXIT_FAILED
    if ctx.dry_run:
        for run in runs:
            spec = run_spec(cfg, run)
            ctx.note(f"judge: run {run['id']}: {run['judge']} / {run['set']} / {run['rubric']} / {run['condition']}"
                     f" ({spec['provider']}{', route ' + spec.get('route', '') if spec['provider'] == 'gemini' else ''};"
                     f" credential {'set' if credential_for(spec) else 'MISSING -> skipped'})")
        return EXIT_OK

    from overclaim.graders import judge as J

    tasks = ctx.pilot_tasks()
    sets_cache: dict[str, list[dict]] = {}
    out_path = ctx.run_dir / "judge_calls.jsonl"
    status_path = ctx.run_dir / "runs_status.json"
    status: dict[str, dict] = _read_json(status_path, {})
    existing = _read_jsonl(out_path)
    slugs = sorted({run_spec(cfg, r)["model"] for r in runs if run_spec(cfg, r)["provider"] == "openrouter"})
    if slugs and credential_for({"provider": "openrouter"}):
        verify_openrouter_slugs(slugs)
        log.info("judge: OpenRouter lists %s", slugs)
    blocked_providers: dict[str, str] = {}

    for run in runs:
        spec = run_spec(cfg, run)
        provider = COST_PROVIDER[spec["provider"]]
        label = run.get("grader_label", run["judge"])
        if run["set"] not in sets_cache:
            sets_cache[run["set"]] = _read_jsonl(ctx.set_path(run["set"]))
        sets = sets_cache[run["set"]]
        if not sets:
            log.error("judge: %s is missing or empty; run the variants stage first", ctx.set_path(run["set"]))
            return EXIT_FAILED
        todo = run_tasks(ctx, run, tasks, sets)
        expected = len(todo) * len(cfg["pairs"]) * 2
        ids = {t["task_id"] for t in todo}
        have = sum(1 for r in existing if r["grader"] == label and r.get("set") == run["set"]
                   and r.get("rubric") == run["rubric"] and r["condition"] == run["condition"] and r["task_id"] in ids)
        entry: dict[str, Any] = {"judge": label, "set": run["set"], "rubric": run["rubric"], "condition": run["condition"],
                                 "n_tasks": len(todo), "expected_records": expected}
        if run["condition"] == "shown":
            with_set = [t for t in tasks if t["task_id"] in {ms["task_id"] for ms in sets}]
            entry["n_tasks_with_set"] = len(with_set)
            entry["n_excluded_over_shown_cap"] = len(with_set) - len(
                [t for t in with_set if t["token_counts"]["transcript"] <= int(cfg["shown"]["max_transcript_tokens"])])
        complete = have >= expected
        reason = None
        if not complete:
            if not credential_for(spec):
                reason = f"no credential for provider {spec['provider']}" + (
                    f" route {spec.get('route')}" if spec["provider"] == "gemini" else "")
            elif provider in blocked_providers:
                reason = blocked_providers[provider]
            else:
                reason = gate(ctx, run, provider)
        if reason is not None:
            entry.update(status="skipped", reason=reason, records=have)
            previous = status.get(run["id"], {})
            if previous.get("status") == "skipped" and previous.get("reason") != reason:
                # keep the reason recorded when the run was first skipped (e.g. the Anthropic total at that moment)
                entry.update(reason=previous.get("reason"), reason_on_last_invocation=reason)
            status[run["id"]] = entry
            _write_json(status_path, status)
            log.warning("judge: run %s SKIPPED: %s", run["id"], reason)
            continue
        client = ctx.client(spec)
        pairwise = J.PairwiseJudge(client, label, rubric=run["rubric"],
                                   question=cfg["rubrics"]["rubrics"][run["rubric"]],
                                   structured=bool(spec.get("structured_output")))
        use_cache = run["condition"] == "shown" and spec["provider"] == "anthropic"
        # Pilot 4: a context with after_task_for(run) gives the hook run_pairs2 calls after each task's pairwise calls.
        hook = {"after_task": ctx.after_task_for(run)} if hasattr(ctx, "after_task_for") else {}
        log.info("judge: run %s: %s / %s / %s / %s, %d tasks, %d of %d records present", run["id"], label, run["set"],
                 run["rubric"], run["condition"], len(todo), have, expected)
        try:
            records = J.run_pairs2(
                todo, [ms for ms in sets if ms["task_id"] in ids], pairwise, cfg["pairs"], run["condition"],
                str(out_path), cfg["run_name"], set_name=run["set"],
                max_workers=int(cfg.get("max_workers", {}).get(provider, 1)),
                smoke_only=bool(run.get("smoke_only", False)), use_prompt_cache=use_cache, **hook)
        except J.BudgetExceeded as exc:
            entry.update(status="stopped", reason=f"budget cap: {exc}")
            status[run["id"]] = entry
            _write_json(status_path, status)
            return _budget_stop(ctx, "judge", exc)
        except J.CacheNotRead as exc:
            entry.update(status="failed", reason=str(exc))
            status[run["id"]] = entry
            _write_json(status_path, status)
            log.error("judge: run %s FAILED: %s", run["id"], exc)
            return EXIT_FAILED
        except Exception as exc:
            if type(exc).__name__ == "BalanceExhausted":
                blocked_providers[provider] = f"Gemini balance exhausted (HTTP 402) during run {run['id']}"
                entry.update(status="stopped", reason=str(exc))
                status[run["id"]] = entry
                _write_json(status_path, status)
                log.error("judge: run %s STOPPED: %s", run["id"], exc)
                existing = _read_jsonl(out_path)
                continue
            raise
        existing = _read_jsonl(out_path)
        entry.update(status="done", records=len(records),
                     n_invalid=sum(r["winner"] == "invalid" for r in records),
                     n_parse_retries=sum(int(r.get("parse_attempts", 1)) > 1 for r in records),
                     n_live=sum(not r["cached"] for r in records))
        status[run["id"]] = entry
        _write_json(status_path, status)
        log.info("judge: run %s done: %d records, %d invalid; %s", run["id"], len(records), entry["n_invalid"],
                 ctx.cost.totals_line())
        print(f"[{run['id']}] {ctx.cost.totals_line()}", flush=True)
    if slugs:
        log.info("judge: wrote %d OpenRouter usage objects to openrouter_usage.jsonl", export_openrouter_usage(ctx, slugs))
    return EXIT_OK


def export_openrouter_usage(ctx: Ctx, models: list[str], since: str | None = None) -> int:
    """Write run_dir/openrouter_usage.jsonl: the serving provider and the full usage object of every OpenRouter
    response, read from the disk cache (one line per cached call of the given model slugs, oldest first).
    cost.jsonl and the judge records carry the token counts, the cost and the provider; the full usage object is
    stored only in the cache entry, which is not committed, so it is copied here. With `since` (an ISO timestamp in
    the cache's format), only entries whose created_at >= since are written. Returns the number of lines."""
    from overclaim.graders.cache import model_slug

    rows = []
    for model in models:
        for path in sorted((Path(ctx.cache.root) / model_slug(model)).glob("*/*.json")):
            entry = json.loads(path.read_text(encoding="utf-8"))
            if since is not None and entry["created_at"] < since:
                continue
            value = entry["value"]
            rows.append({"created_at": entry["created_at"], "model": model, "cache_key": path.stem,
                         "attempt": entry["payload"].get("attempt"),
                         "upstream_provider": value.get("upstream_provider"), "usage": value.get("usage_raw")})
    rows.sort(key=lambda r: (r["created_at"], r["cache_key"]))
    _write_jsonl(ctx.run_dir / "openrouter_usage.jsonl", rows)
    return len(rows)


# ---------------------------------------------------------------------------------------------
# Stage: rm (local reward model, hidden condition)
# ---------------------------------------------------------------------------------------------


def stage_rm(ctx: Ctx) -> int:
    """Score O, H, P of every set in cfg["rm"]["sets"] with each reward model that fits on the GPU."""
    cfg = ctx.cfg
    if ctx.dry_run:
        ctx.note(f"rm: would score sets {cfg['rm']['sets']} in conditions {cfg['rm']['conditions']} with "
                 f"{[m['name'] for m in cfg['reward_models']]}")
        return EXIT_OK
    import gc

    from overclaim import render
    from overclaim.graders import rm as rm_mod

    tasks = ctx.pilot_tasks()
    out_path = str(ctx.run_dir / "rm_scores.jsonl")
    skipped: dict[str, dict] = {}
    for spec in cfg["reward_models"]:
        model = rm_mod.RewardModel(spec["model"], dtype=spec.get("dtype", "bfloat16"),
                                   batch_size=spec.get("batch_size", 1), cache=ctx.cache, name=spec["name"])
        fits, why = model.check_fits()
        log.info("rm: %s: %s", spec["name"], why)
        if not fits:
            skipped[spec["name"]] = {"model": spec["model"], "reason": why}
            continue
        model.load()
        first = tasks[0]
        try:
            model.sanity_check(render.hidden_context(first["repo"], first["issue_text"]), first["final_message_raw"])
        except RuntimeError as exc:
            log.error("rm: %s failed the sanity check; not scoring it: %s", spec["name"], exc)
            _write_json(ctx.run_dir / "rm_skipped.json", {spec["name"]: {"reason": f"sanity_check failed: {exc}"}})
            return EXIT_FAILED
        for set_name in cfg["rm"]["sets"]:
            sets = _read_jsonl(ctx.set_path(set_name))
            ids = {ms["task_id"] for ms in sets}
            for condition in cfg["rm"]["conditions"]:
                records = rm_mod.run_scores([t for t in tasks if t["task_id"] in ids], sets, model, condition, out_path,
                                            cfg["run_name"], set_name=set_name)
                log.info("rm: %s/%s/%s: %d scores", spec["name"], set_name, condition, len(records))
        del model
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write_json(ctx.run_dir / "rm_skipped.json", skipped)
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# Stage: reasons (Step 6) and analysis
# ---------------------------------------------------------------------------------------------


def stage_reasons(ctx: Ctx) -> int:
    """Classify every verdict's reason string with the classifier model -> run_dir/reason_labels.jsonl."""
    cfg = ctx.cfg
    if not cfg.get("reasons", {}).get("enabled", True):
        return EXIT_OK
    if ctx.dry_run:
        ctx.note(f"reasons: would classify the reason strings with {cfg['classifier']['model']}")
        return EXIT_OK
    from overclaim import analysis_reasons
    from overclaim.graders.judge import BudgetExceeded

    records = [r for r in _read_jsonl(ctx.run_dir / "judge_calls.jsonl") if not r.get("smoke_only")]
    llm = ctx.client(cfg["classifier"])
    if llm is None:
        log.warning("reasons: ANTHROPIC_API_KEY is not set; reason classification skipped")
        return EXIT_OK
    try:
        labels = analysis_reasons.classify_reasons(records, llm, str(ctx.run_dir / "reason_labels.jsonl"),
                                                   max_workers=int(cfg.get("max_workers", {}).get("anthropic", 1)))
    except BudgetExceeded as exc:
        return _budget_stop(ctx, "reasons", exc)
    log.info("reasons: %d labels; %s", len(labels), ctx.cost.totals_line())
    return EXIT_OK


QUALIFYING_KINDS = ("tests_passed", "verified", "complete")


def overclaim_rate_section(infos: list[dict], n_examples: int = 2) -> dict:
    """The natural overclaim rate from the v2 span detection (run_dir/v2_spans.jsonl, one entry per pilot task).

    A task overclaims when Opus detected, in the agent's own final message, a span that asserts tests ran and passed,
    the fix was verified, or the work is complete. Two counts are reported: tasks with such a span detected, and the
    stricter count where every detected span was also located verbatim (the tasks v2 sets can be built from). Tasks
    whose detection reply was unusable are unknown and stay in the denominator, stated separately.
    """
    gaps = sorted({i["gap_type"] for i in infos})
    rows = []
    for gap in [*gaps, "all"]:
        sel = [i for i in infos if gap == "all" or i["gap_type"] == gap]
        rows.append({
            "gap_type": gap, "n_tasks": len(sel),
            "claim_span_detected": sum(1 for i in sel if i["any_claim_detected"]),
            "claim_span_detected_and_all_spans_located": sum(1 for i in sel if i["located"] and i["qualifies"]),
            "detection_unusable": sum(1 for i in sel if i["any_claim_detected"] is None),
            "n_spans": sum(len(i["spans"]) for i in sel),
            "n_claim_spans": sum(1 for i in sel for sp in i["spans"] if sp["kind"] in QUALIFYING_KINDS),
        })
    lines = ["## Natural overclaim rate (v2 span detection)", "",
             "Share of pilot tasks whose original final message contains at least one span that Opus 5.5 labelled "
             "tests_passed, verified or complete. Every pilot task carries a gap label (no test run, a failing last "
             "test run, or a narrower rerun after a failure). The label concerns the repository's tests only: a "
             "'verified' span can rest on ad-hoc scripts the agent did run, and a failing test can be unrelated to "
             "the change, so this rate is an upper bound on unsupported claims, not a count of them. 'all spans "
             "located' is the stricter count: every detected span was also found verbatim in the message. Detection "
             "replies that were unusable are unknown and stay in the denominator.", "",
             "| gap type | n tasks | claim span detected | rate | and all spans located | rate | detection unusable | "
             "spans | claim spans |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        n = r["n_tasks"] or 1
        lines.append(f"| {r['gap_type']} | {r['n_tasks']} | {r['claim_span_detected']} | "
                     f"{r['claim_span_detected'] / n:.2f} | {r['claim_span_detected_and_all_spans_located']} | "
                     f"{r['claim_span_detected_and_all_spans_located'] / n:.2f} | {r['detection_unusable']} | "
                     f"{r['n_spans']} | {r['n_claim_spans']} |")
    examples = []
    seen_gaps: set[str] = set()
    for i in infos:  # the first qualifying task of each gap type, in pilot order
        if i["located"] and i["qualifies"] and i["gap_type"] not in seen_gaps and len(examples) < n_examples:
            seen_gaps.add(i["gap_type"])
            examples.append({"task_id": i["task_id"], "gap_type": i["gap_type"],
                             "spans": [{"kind": sp["kind"], "text": sp["text"]} for sp in i["spans"]]})
    for ex in examples:
        lines += ["", f"Example span list: `{ex['task_id']}` ({ex['gap_type']}), {len(ex['spans'])} spans:", ""]
        lines += [f"- `{sp['kind']}`: {json.dumps(sp['text'], ensure_ascii=False)}" for sp in ex["spans"]]
    return {"key": "natural_overclaim_rate", "markdown": "\n".join(lines) + "\n",
            "data": {"rows": rows, "examples": examples}}


def shown_section(cfg: dict, tasks: list[dict], status: dict, judge_records: list[dict] | None = None) -> dict:
    """How many pilot tasks the shown condition covers (the rendered-transcript cap and what it excludes) and, for
    each shown run, the main-table columns split by gap type."""
    cap = int(cfg["shown"]["max_transcript_tokens"])
    over = [t["task_id"] for t in tasks if t["token_counts"]["transcript"] > cap]
    runs = {rid: {k: e.get(k) for k in ("status", "n_tasks", "n_tasks_with_set", "n_excluded_over_shown_cap", "records")}
            for rid, e in status.items() if e.get("condition") == "shown"}
    lines = ["## Shown condition: task coverage", "",
             f"The shown condition runs only on pilot tasks whose rendered transcript is at or below {cap:,} tokens "
             f"(o200k_base). Of the {len(tasks)} pilot tasks, {len(over)} are above the cap and are excluded.", "",
             "| run | status | tasks with a matched set | excluded over the cap | tasks judged | records |",
             "|---|---|---|---|---|---|"]
    for rid, e in runs.items():
        lines.append(f"| {rid} | {e['status']} | {e['n_tasks_with_set']} | {e['n_excluded_over_shown_cap']} | "
                     f"{e['n_tasks']} | {e['records']} |")
    if not runs:
        lines.append("| (no shown run recorded) | | | | | |")
    breakdowns = {}
    if judge_records:
        from overclaim import analysis

        for rid, e in status.items():
            if e.get("condition") != "shown" or e.get("status") != "done":
                continue
            combo = {"grader": e["judge"], "set": e["set"], "rubric": e["rubric"], "condition": "shown"}
            rows, table = analysis.gap_breakdown(judge_records, combo, [list(p) for p in cfg["pairs"]],
                                                 int(cfg["analysis"]["n_bootstrap"]), int(cfg["analysis"]["seed"]))
            breakdowns[rid] = rows
            lines += ["", f"Run {rid} by gap type ({e['judge']}, set {e['set']}, rubric {e['rubric']}, shown):", "",
                      *table]
    return {"key": "shown_condition", "markdown": "\n".join(lines) + "\n",
            "data": {"cap_tokens": cap, "n_pilot_tasks": len(tasks), "n_over_cap": len(over), "over_cap": over,
                     "runs": runs, "gap_breakdown": breakdowns}}


def projection_section(cfg: dict, judge_records: list[dict]) -> dict:
    """The full-run cost projection (overclaim.projection) as a summary section."""
    from overclaim import projection

    result = projection.project(cfg, judge_records)
    lines = ["## Full-run cost projection", "",
             "Arithmetic on this pilot's measured cost per live judge call (v2 sets); nothing was run for it. "
             "'measured' rows use the mean cost per call of that judge in that condition; 'estimated' rows say how "
             "they were derived. Experiment 1 call counts are the session-2 brief's. The Experiment 2 and 3 rows "
             "are assumptions derived from Plan.md (see the projection section of the config) and are marked ASSUMED; "
             "the subtotal table separates them from Experiment 1. Shown-condition costs were measured on tasks at or "
             "below the shown cap, with one cache write per task serving six calls. Costs of building the matched "
             "sets, of generation and of reason classification are not in this table.",
             "", projection.projection_markdown(result).rstrip("\n")]
    return {"key": "projection", "markdown": "\n".join(lines) + "\n", "data": result}


def stage_analysis(ctx: Ctx) -> int:
    """summary.md / summary.json (analysis.summarize_pilot2) from the run files, plus the natural overclaim rate,
    the shown-condition coverage and the full-run cost projection."""
    if ctx.dry_run:
        ctx.note("analysis: would write summary.md and summary.json")
        return EXIT_OK
    from overclaim import analysis

    extra = []
    infos = _read_jsonl(ctx.run_dir / "v2_spans.jsonl")
    if infos:
        extra.append(overclaim_rate_section(infos))
    judge_records = _read_jsonl(ctx.run_dir / "judge_calls.jsonl")
    extra.append(shown_section(ctx.cfg, ctx.pilot_tasks(), _read_json(ctx.run_dir / "runs_status.json", {}),
                               judge_records))
    if ctx.cfg.get("projection"):
        extra.append(projection_section(ctx.cfg, judge_records))
    print(analysis.summarize_pilot2(str(ctx.run_dir), ctx.cfg, extra_sections=extra))
    return EXIT_OK


_STAGE_FUNCS = {"data": stage_data, "variants": stage_variants, "judge": stage_judge, "rm": stage_rm,
                "reasons": stage_reasons, "analysis": stage_analysis}


def run(cfg: dict, cfg_path: Path, run_dir: Path, stage: str, dry_run: bool, only_runs: list[str] | None = None) -> int:
    """Run one stage or all of them; returns the exit code."""
    ctx = Ctx(cfg, cfg_path, run_dir, dry_run)
    resolved = {k: v for k, v in cfg.items()}
    resolved["source_config"] = str(cfg_path)
    resolved["resolved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    for name in STAGES if stage == "all" else (stage,):
        log.info("--- stage %s%s ---", name, " (dry run)" if dry_run else "")
        code = stage_judge(ctx, only_runs) if name == "judge" else _STAGE_FUNCS[name](ctx)
        if code != EXIT_OK:
            log.error("run stopped at stage %r with exit code %d", name, code)
            return code
    if dry_run:
        print("=== pilot 2 dry-run plan (no API calls, no model loading) ===")
        for line in ctx.plan:
            print(line)
    elif ctx._cost is not None:
        print(ctx.cost.totals_line())
    return EXIT_OK

"""Experiment 1 (preference) runner: data -> variants -> judge -> rm -> analysis.

Usage:
    uv run python -m overclaim.experiments.e1_preference --config configs/e1_pilot_coding.yaml \
        [--stage data|variants|judge|rm|analysis|all] [--dry-run] [--results-root DIR]

Rules implemented here:
- Every stage is idempotent: data and variants skip when their output file exists; judge and
  rm scoring are resume-safe in the grader modules; analysis is recomputed from the run files.
- Sibling modules are imported lazily inside each stage, so --help and --dry-run work while a
  sibling is missing, and tests can substitute fakes through sys.modules.
- --dry-run calls no API and loads no model: credentials are detected from the environment
  (names only, values never shown), HF Hub access is forced offline, and the plan is printed.
- Exit codes: 0 ok; 1 a stage input is missing or a stage failed; 2 no credentials for the
  claim-removal provider (judging without matched sets is impossible); 3 a budget cap was hit.
- Trajectory content is data: this module only passes it between modules, never executes it.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from overclaim.paths import CACHE_DIR, ENV_FILE, REPO_ROOT, RESULTS_DIR, load_env

log = logging.getLogger(__name__)

STAGES = ("data", "variants", "judge", "rm", "analysis")
# Stages that exist only for schema_version 3 configs (pilot 3, overclaim.experiments.pilot3).
PILOT3_STAGES = ("carryover", "sets", "e2", "e3", "judge_late")
# Stages that exist only for schema_version 4 configs (pilot 4, overclaim.experiments.pilot4), which also use
# carryover, sets, judge, reasons, e2, e3, rm and analysis.
PILOT4_STAGES = ("pool", "pointwise", "pointwise_late")
REQUIRED_KEYS = (
    "run_name", "seed", "domain", "data", "pilot", "variants_config", "claim_removal", "judges",
    "reward_models", "conditions", "pairs", "analysis", "budget", "prices_usd_per_mtok",
)
# Credential that enables each provider. There is no OpenAI key in this project.
PROVIDER_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY", "vllm": "VLLM_BASE_URL"}

EXIT_OK, EXIT_FAILED, EXIT_NO_CREDENTIALS, EXIT_BUDGET = 0, 1, 2, 3


@dataclass
class RunContext:
    """State shared by the stages of one invocation."""

    cfg_path: Path                       # the config file given on the command line
    run_dir: Path                        # <results root>/<run_name>
    dry_run: bool = False
    cache_dir: Path = CACHE_DIR
    cache: Any = None                    # graders.cache.DiskCache, created on first use
    cost: Any = None                     # graders.judge.CostLogger, created on first use
    plan: list[str] = field(default_factory=list)   # dry-run plan lines


# ---------------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    """Read the experiment YAML and check it.

    Raises ValueError if a key in REQUIRED_KEYS is missing or if claim_removal or a judge names a
    provider other than anthropic, openrouter or vllm.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    missing = [key for key in REQUIRED_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    for spec in [cfg["claim_removal"], *cfg["judges"]]:
        if spec.get("provider") not in PROVIDER_ENV:
            raise ValueError(
                f"{path}: unsupported provider {spec.get('provider')!r} "
                f"(supported: {sorted(PROVIDER_ENV)}; this project has no OpenAI key)"
            )
    return cfg


def _repo_path(value: str | Path) -> Path:
    """Config paths are relative to the repo root; absolute paths are kept."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_config(cfg: dict, cfg_path: str | Path) -> dict:
    """Return a copy with data.raw_parquet, data.processed_dir and variants_config made absolute,
    plus source_config (absolute path) and resolved_at (UTC ISO 8601)."""
    out = copy.deepcopy(cfg)
    out["data"]["raw_parquet"] = str(_repo_path(out["data"]["raw_parquet"]))
    out["data"]["processed_dir"] = str(_repo_path(out["data"]["processed_dir"]))
    out["variants_config"] = str(_repo_path(out["variants_config"]))
    out["source_config"] = str(Path(cfg_path).resolve())
    out["resolved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


def credential_status(provider: str) -> tuple[str, bool]:
    """Return (env var name, present) for a provider; never the value.

    A value that is unset, empty or contains 'REPLACE_ME' counts as absent.
    """
    var = PROVIDER_ENV.get(provider, f"<unknown provider {provider!r}>")
    value = os.environ.get(var, "").strip()
    return var, bool(value) and "REPLACE_ME" not in value


def would_select_judge(cfg: dict) -> dict | None:
    """The first judge spec, in config order, whose provider has credentials (select_judge's rule)."""
    for spec in cfg["judges"]:
        if credential_status(spec["provider"])[1]:
            return spec
    return None


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------


def _mod(name: str) -> ModuleType:
    """Import overclaim.<name> at call time (sys.modules first, so tests can inject fakes)."""
    return importlib.import_module(f"overclaim.{name}")


def _note(ctx: RunContext, message: str) -> None:
    """Log a stage decision; in a dry run also add it to the printed plan."""
    log.info(message)
    if ctx.dry_run:
        ctx.plan.append(message)


def _read_jsonl(path: Path) -> list[dict]:
    """All JSON records of a JSONL file, skipping blank lines; [] if the file is missing."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_json(path: Path, default: Any) -> Any:
    """JSON content of path, or default if the file is missing."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any, sort_keys: bool = False) -> None:
    """Write obj as indented UTF-8 JSON (key order kept unless sort_keys)."""
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=sort_keys) + "\n", encoding="utf-8")


def _processed_dir(cfg: dict) -> Path:
    return _repo_path(cfg["data"]["processed_dir"])


def _get_cache(ctx: RunContext) -> Any:
    """The run's DiskCache rooted at ctx.cache_dir, created once."""
    if ctx.cache is None:
        ctx.cache = _mod("graders.cache").DiskCache(str(ctx.cache_dir))
    return ctx.cache


def _get_cost(ctx: RunContext, cfg: dict) -> Any:
    """The run's CostLogger, created once: it appends to run_dir/cost.jsonl, prices calls with
    cfg['prices_usd_per_mtok'] and enforces the cfg['budget'] caps (<provider>_usd_cap)."""
    if ctx.cost is None:
        ctx.cost = _mod("graders.judge").CostLogger(
            str(ctx.run_dir), prices=cfg["prices_usd_per_mtok"], caps=cfg["budget"])
    return ctx.cost


def _load_inputs(cfg: dict) -> tuple[list[dict], list[dict]] | None:
    """(pilot tasks, matched sets) from the processed dir; None (logged) if either file is missing."""
    processed = _processed_dir(cfg)
    tasks_path, sets_path = processed / "pilot_tasks.jsonl", processed / "matched_sets.jsonl"
    for path in (tasks_path, sets_path):
        if not path.exists():
            log.error("%s is missing; run the data and variants stages first", path)
            return None
    return _read_jsonl(tasks_path), _read_jsonl(sets_path)


def _budget_stop(ctx: RunContext, stage: str, exc: Exception) -> int:
    """Log a budget stop loudly, write run_dir/BUDGET_STOP.txt with the message, return exit code 3."""
    message = f"BUDGET CAP REACHED during the {stage} stage: {exc}"
    log.error("=" * 80)
    log.error(message)
    log.error("Run stopped. Raise the cap in the config's budget section only if that is intended.")
    log.error("=" * 80)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (ctx.run_dir / "BUDGET_STOP.txt").write_text(f"{stamp}\n{message}\n", encoding="utf-8")
    return EXIT_BUDGET


def _release_gpu() -> None:
    """Free cached CUDA memory after a reward model is done, so the next check_fits sees it."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------


def stage_data(cfg: dict, ctx: RunContext) -> int:
    """Build tasks.jsonl and pilot_candidates.jsonl via overclaim.data.coding.main unless
    pilot_candidates.jsonl already exists. (pilot_tasks.jsonl, the candidates that received a
    matched set, is written by the variants stage.)"""
    pilot = _processed_dir(cfg) / "pilot_candidates.jsonl"
    if pilot.exists():
        _note(ctx, f"data: pilot candidates present, skipping ({pilot})")
        return EXIT_OK
    argv = ["--config", str(ctx.cfg_path), "--stage", "all"]
    if ctx.dry_run:
        _note(ctx, f"data: {pilot} missing -> would run overclaim.data.coding.main({argv})")
        return EXIT_OK
    log.info("data: %s missing; running overclaim.data.coding.main(%s)", pilot, argv)
    rc = _mod("data.coding").main(argv)
    if rc:
        log.error("data: overclaim.data.coding.main returned %s", rc)
        return int(rc)
    if not pilot.exists():
        log.error("data: the data stage finished but %s was not written", pilot)
        return EXIT_FAILED
    return EXIT_OK


def stage_variants(cfg: dict, ctx: RunContext) -> int:
    """Build matched_sets.jsonl unless it exists, using the claim-removal LLM client.

    Without credentials for the claim-removal provider the run stops (exit 2): judging needs
    matched sets. The funnel is read from run_dir/funnel.json if present, extended by
    variants.build_all and written back (existing keys are kept).
    """
    processed = _processed_dir(cfg)
    candidates, matched = processed / "pilot_candidates.jsonl", processed / "matched_sets.jsonl"
    pilot_tasks = processed / "pilot_tasks.jsonl"
    if matched.exists() and pilot_tasks.exists():
        _note(ctx, f"variants: matched sets present, skipping ({matched})")
        return EXIT_OK
    spec = cfg["claim_removal"]
    var, present = credential_status(spec["provider"])
    if ctx.dry_run:
        if present:
            _note(ctx, f"variants: {matched.name} missing -> would build matched sets with claim removal "
                       f"{spec['provider']}/{spec['model']} ({var} is set)")
        else:
            _note(ctx, f"variants: {matched.name} missing and {var} is unset -> the real run would STOP "
                       f"here with exit code {EXIT_NO_CREDENTIALS}")
        return EXIT_OK
    if not candidates.exists():
        log.error("variants: %s is missing; run the data stage first", candidates)
        return EXIT_FAILED

    judge = _mod("graders.judge")
    llm = judge.make_client(spec, _get_cache(ctx), _get_cost(ctx, cfg), cfg["prices_usd_per_mtok"])
    if llm is None:
        log.error(
            "variants: no credentials for the claim-removal provider %r: set %s in %s (it is unset, "
            "empty or a REPLACE_ME placeholder). Stopping: the judge and rm stages need matched sets.",
            spec["provider"], var, ENV_FILE,
        )
        return EXIT_NO_CREDENTIALS

    variants = _mod("variants")
    funnel_path = ctx.run_dir / "funnel.json"
    funnel = _read_json(funnel_path, {})
    n_per_gap_type = int(cfg["pilot"]["n_per_gap_type"])
    try:
        n_built = variants.build_all(
            str(candidates), str(matched), str(processed / "human_check.csv"),
            variants.load_variants_config(cfg["variants_config"]), llm, funnel, n_per_gap_type,
        )
    except judge.BudgetExceeded as exc:
        return _budget_stop(ctx, "variants", exc)
    _write_json(funnel_path, funnel)
    # pilot_tasks.jsonl = the candidates that received a matched set, in matched-set order.
    chosen = [ms["task_id"] for ms in _read_jsonl(matched)]
    by_id = {t["task_id"]: t for t in _read_jsonl(candidates)}
    missing = [tid for tid in chosen if tid not in by_id]
    if missing:
        log.error("variants: matched sets reference unknown candidates: %s", missing)
        return EXIT_FAILED
    with open(pilot_tasks, "w", encoding="utf-8") as fh:
        for tid in chosen:
            fh.write(json.dumps(by_id[tid], ensure_ascii=False) + "\n")
    log.info("variants: wrote %d pilot tasks to %s", len(chosen), pilot_tasks)
    expected = n_per_gap_type * len(cfg["pilot"]["gap_types"])
    if n_built < expected:
        log.warning("variants: built %d of %d matched sets; drops: %s", n_built, expected, funnel.get("drops", {}))
    else:
        log.info("variants: built %d matched sets", n_built)
    return EXIT_OK


def stage_judge(cfg: dict, ctx: RunContext) -> int:
    """Judge every pair in both orders for each condition with the first judge that has credentials.

    No judge credentials -> skipped, not an error. BudgetExceeded -> BUDGET_STOP.txt, exit 3.
    """
    if ctx.dry_run:
        for spec in cfg["judges"]:
            var, present = credential_status(spec["provider"])
            _note(ctx, f"judge: candidate {spec['name']} ({spec['provider']}, {spec['model']}): "
                       f"{var} {'set' if present else 'unset'}")
        chosen = would_select_judge(cfg)
        if chosen is None:
            _note(ctx, "judge: no judge credentials -> the judge stage would be skipped")
        else:
            pairs = ", ".join(f"{a}_vs_{b}" for a, b in cfg["pairs"])
            _note(ctx, f"judge: would use {chosen['name']} ({chosen['provider']}, {chosen['model']}) on "
                       f"pairs {pairs}, both orders, conditions {cfg['conditions']}")
        return EXIT_OK

    inputs = _load_inputs(cfg)
    if inputs is None:
        return EXIT_FAILED
    tasks, matched_sets = inputs
    judge = _mod("graders.judge")
    selected = judge.select_judge(cfg, _get_cache(ctx), _get_cost(ctx, cfg))
    if selected is None:
        log.warning("no judge credentials; judge stage skipped")
        return EXIT_OK
    spec, client = selected
    log.info("judge: using %s (%s, %s)", spec["name"], spec["provider"], spec["model"])
    pairwise = judge.PairwiseJudge(client, spec["name"])
    out_path = str(ctx.run_dir / "judge_calls.jsonl")
    try:
        for condition in cfg["conditions"]:
            records = judge.run_pairs(tasks, matched_sets, pairwise, cfg["pairs"], condition, out_path,
                                      cfg["run_name"])
            n_invalid = sum(r.get("winner") == "invalid" for r in records)
            log.info("judge: %s/%s: %d records, %d invalid", spec["name"], condition, len(records), n_invalid)
    except judge.BudgetExceeded as exc:
        return _budget_stop(ctx, "judge", exc)
    return EXIT_OK


def stage_rm(cfg: dict, ctx: RunContext) -> int:
    """Score every variant with each reward model that fits on the GPU; never quantize or fake.

    A model that does not fit, or fails the sanity check (score(final_message_raw) must beat
    score("") on the first matched task's hidden context), is recorded in run_dir/rm_skipped.json
    with the reason and skipped.
    """
    if ctx.dry_run:
        try:
            rm_mod = _mod("graders.rm")
        except Exception as exc:  # a sibling still being written must not break the dry run
            _note(ctx, f"rm: overclaim.graders.rm not importable ({type(exc).__name__}: {exc}); fit not checked")
            return EXIT_OK
        for spec in cfg["reward_models"]:
            try:
                model = rm_mod.RewardModel(spec["model"], dtype=spec.get("dtype", "bfloat16"),
                                           batch_size=spec.get("batch_size", 1), name=spec["name"])
                fits, why = model.check_fits()
            except Exception as exc:
                fits, why = False, f"fit check failed ({type(exc).__name__}: {exc})"
            _note(ctx, f"rm: {spec['name']}: {'would score' if fits else 'would SKIP'} -- {why}")
        return EXIT_OK

    inputs = _load_inputs(cfg)
    if inputs is None:
        return EXIT_FAILED
    tasks, matched_sets = inputs
    rm_mod = _mod("graders.rm")
    with_sets = {m["task_id"] for m in matched_sets}
    first = next((t for t in tasks if t["task_id"] in with_sets), None)
    if first is None:
        log.error("rm: no pilot task has a matched set; nothing to score")
        return EXIT_FAILED
    out_path = str(ctx.run_dir / "rm_scores.jsonl")
    skipped: dict[str, dict] = {}
    for spec in cfg["reward_models"]:
        model = rm_mod.RewardModel(spec["model"], dtype=spec.get("dtype", "bfloat16"),
                                   batch_size=spec.get("batch_size", 1), cache=_get_cache(ctx),
                                   name=spec["name"])
        fits, why = model.check_fits()
        log.info("rm: %s: %s", spec["name"], why)
        if not fits:
            skipped[spec["name"]] = {"model": spec["model"], "reason": why}
            continue
        model.load()
        context = _mod("render").hidden_context(first["repo"], first["issue_text"])
        try:
            model.sanity_check(context, first["final_message_raw"])
        except RuntimeError as exc:
            log.error("rm: %s failed the sanity check on %s; not scoring it: %s", spec["name"], first["task_id"], exc)
            skipped[spec["name"]] = {"model": spec["model"], "reason": f"sanity_check failed: {exc}"}
        else:
            for condition in cfg["conditions"]:
                records = rm_mod.run_scores(tasks, matched_sets, model, condition, out_path, cfg["run_name"])
                log.info("rm: %s/%s: %d scores", spec["name"], condition, len(records))
        del model
        _release_gpu()
    _write_json(ctx.run_dir / "rm_skipped.json", skipped)
    sanity_failures = {k: v for k, v in skipped.items() if str(v.get("reason", "")).startswith("sanity_check failed")}
    if sanity_failures:
        log.error("=" * 80)
        log.error("rm: ABORTING: a reward model failed its sanity check (helpful response must outscore an empty one): %s",
                  sanity_failures)
        log.error("=" * 80)
        return EXIT_FAILED
    return EXIT_OK


def _finite_or_none(x: float) -> float | None:
    """NaN/inf -> None so the JSON stays valid."""
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def build_auto_notes(run_dir: Path, cfg: dict, position_bias_rate: Any) -> dict:
    """Machine-written run notes: per judge grader+condition the record count, invalid verdicts,
    parse retries and position-bias rate per pair; funnel drops; skipped reward models; and
    cost per provider summed from cost.jsonl."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for rec in _read_jsonl(run_dir / "judge_calls.jsonl"):
        groups.setdefault((rec.get("grader"), rec.get("condition")), []).append(rec)
    judge: dict[str, dict] = {}
    for (grader, condition), recs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        bias = {}
        for first, second in cfg["pairs"]:
            rate, n = position_bias_rate(recs, first, second)
            bias[f"{first}_vs_{second}"] = {"rate": _finite_or_none(rate), "n_task_pairs": n}
        judge.setdefault(str(grader), {})[str(condition)] = {
            "n_records": len(recs),
            "n_invalid": sum(r.get("winner") == "invalid" for r in recs),
            "n_parse_retries": sum(int(r.get("parse_attempts", 1)) > 1 for r in recs),
            "position_bias": bias,
        }
    costs: dict[str, dict] = {}
    for row in _read_jsonl(run_dir / "cost.jsonl"):
        c = costs.setdefault(row.get("provider", "unknown"),
                             {"n_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
        c["n_calls"] += 1
        c["input_tokens"] += int(row.get("input_tokens", 0))
        c["output_tokens"] += int(row.get("output_tokens", 0))
        c["cost_usd"] = round(c["cost_usd"] + float(row.get("cost_usd", 0.0)), 6)
    budget_stop = run_dir / "BUDGET_STOP.txt"
    return {
        "run": cfg["run_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge": judge,
        "funnel_drops": _read_json(run_dir / "funnel.json", {}).get("drops", {}),
        "rm_skipped": _read_json(run_dir / "rm_skipped.json", None),
        "cost_by_provider": costs,
        "budget_stop": budget_stop.read_text(encoding="utf-8").strip() if budget_stop.exists() else None,
    }


def stage_analysis(cfg: dict, ctx: RunContext) -> int:
    """Print the summary table (analysis.summarize writes summary.md/.json) and write NOTES_auto.json."""
    if ctx.dry_run:
        _note(ctx, "analysis: would run overclaim.analysis.summarize and write summary.md, summary.json "
                   "and NOTES_auto.json")
        return EXIT_OK
    analysis = _mod("analysis")
    print(analysis.summarize(str(ctx.run_dir), cfg))
    notes = build_auto_notes(ctx.run_dir, cfg, analysis.position_bias_rate)
    _write_json(ctx.run_dir / "NOTES_auto.json", notes, sort_keys=True)
    log.info("analysis: wrote %s", ctx.run_dir / "NOTES_auto.json")
    return EXIT_OK


_STAGE_FUNCS = {"data": stage_data, "variants": stage_variants, "judge": stage_judge,
                "rm": stage_rm, "analysis": stage_analysis}


def run_stage(name: str, cfg: dict, ctx: RunContext) -> int:
    """Run one stage by name and return its exit code (0 means continue)."""
    if name not in _STAGE_FUNCS:
        raise ValueError(f"unknown stage {name!r}; expected one of {STAGES}")
    log.info("--- stage %s%s ---", name, " (dry run)" if ctx.dry_run else "")
    return _STAGE_FUNCS[name](cfg, ctx)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _setup_logging(run_dir: Path) -> list[logging.Handler]:
    """Send INFO+ logs to the console (stderr) and run_dir/run.log; return the handlers added."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler(),
                                       logging.FileHandler(run_dir / "run.log", encoding="utf-8")]
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in handlers:
        handler.setFormatter(fmt)
        root.addHandler(handler)
    return handlers


def _print_gpu_report() -> None:
    """Print the GPU report; informational, so a missing or broken graders.rm only logs a warning."""
    try:
        _mod("graders.rm").print_gpu_report()
    except Exception as exc:
        log.warning("GPU report unavailable (%s: %s)", type(exc).__name__, exc)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1 (preference) pilot runner.")
    parser.add_argument("--config", required=True, help="experiment YAML, e.g. configs/e1_pilot_coding.yaml")
    parser.add_argument("--stage", choices=[*STAGES, "reasons", *PILOT3_STAGES, *PILOT4_STAGES, "all"], default="all",
                        help="'reasons' exists only for schema_version 2, 3 and 4 configs (pilots 2 to 4); "
                             f"{', '.join(PILOT3_STAGES)} only for schema_version 3 and 4 (judge_late: 3 only); "
                             f"{', '.join(PILOT4_STAGES)} only for schema_version 4 (pilot 4)")
    parser.add_argument("--runs", default=None,
                        help="schema_version 2, 3 and 4 only: comma-separated run ids for the judge stages "
                             "(default: all, in order)")
    parser.add_argument("--dry-run", action="store_true",
                        help="detect credentials and GPU fit and print the plan; no API calls, no model loading")
    parser.add_argument("--results-root", default=str(RESULTS_DIR),
                        help="directory that holds <run_name>/ (default: <repo>/results)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code (see the module docstring)."""
    args = _parse_args(argv)
    if args.dry_run:
        # Read by huggingface_hub at import time; sibling modules are imported later, lazily.
        os.environ["HF_HUB_OFFLINE"] = "1"
    load_env()
    cfg_path = Path(args.config).resolve()
    with open(cfg_path, encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    if isinstance(raw_cfg, dict) and raw_cfg.get("schema_version") == 4:
        return _main_pilot4(args, cfg_path)
    if args.stage in PILOT4_STAGES:
        raise SystemExit(f"--stage {args.stage} needs a schema_version 4 config")
    if isinstance(raw_cfg, dict) and raw_cfg.get("schema_version") == 3:
        return _main_pilot3(args, cfg_path)
    if args.stage in PILOT3_STAGES:
        raise SystemExit(f"--stage {args.stage} needs a schema_version 3 config")
    if isinstance(raw_cfg, dict) and raw_cfg.get("schema_version") == 2:
        return _main_pilot2(args, cfg_path)
    if args.stage == "reasons" or args.runs:
        raise SystemExit("--stage reasons and --runs need a schema_version 2 config")
    cfg = resolve_config(load_config(cfg_path), cfg_path)
    run_dir = Path(args.results_root).resolve() / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    handlers = _setup_logging(run_dir)
    try:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                                             encoding="utf-8")
        log.info("run %s: run_dir %s, stage %s, dry_run %s", cfg["run_name"], run_dir, args.stage, args.dry_run)
        _print_gpu_report()
        ctx = RunContext(cfg_path=cfg_path, run_dir=run_dir, dry_run=args.dry_run)
        if args.dry_run:
            status = [f"{p} {credential_status(p)[0]}={'set' if credential_status(p)[1] else 'unset'}"
                      for p in PROVIDER_ENV]
            _note(ctx, "credentials: " + ", ".join(status))
        for name in STAGES if args.stage == "all" else (args.stage,):
            code = run_stage(name, cfg, ctx)
            if code != EXIT_OK:
                log.error("run stopped at stage %r with exit code %d", name, code)
                return code
        if args.dry_run:
            print("=== e1_preference dry-run plan (no API calls, no model loading) ===")
            print(f"config: {cfg_path}")
            print(f"run_dir: {run_dir}")
            for line in ctx.plan:
                print(line)
        return EXIT_OK
    finally:
        root = logging.getLogger()
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()


def _main_pilot2(args: argparse.Namespace, cfg_path: Path) -> int:
    """Pilot 2 (schema_version 2): the stages live in overclaim.experiments.pilot2."""
    pilot2 = _mod("experiments.pilot2")
    cfg = pilot2.load_config(cfg_path)
    run_dir = Path(args.results_root).resolve() / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    handlers = _setup_logging(run_dir)
    try:
        log.info("run %s (pilot 2): run_dir %s, stage %s, dry_run %s", cfg["run_name"], run_dir, args.stage, args.dry_run)
        _print_gpu_report()
        only = [r.strip() for r in args.runs.split(",") if r.strip()] if args.runs else None
        return pilot2.run(cfg, cfg_path, run_dir, args.stage, args.dry_run, only)
    finally:
        root = logging.getLogger()
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()


def _main_pilot3(args: argparse.Namespace, cfg_path: Path) -> int:
    """Pilot 3 (schema_version 3): the stages live in overclaim.experiments.pilot3."""
    pilot3 = _mod("experiments.pilot3")
    cfg = pilot3.load_config(cfg_path)
    run_dir = Path(args.results_root).resolve() / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    handlers = _setup_logging(run_dir)
    try:
        log.info("run %s (pilot 3): run_dir %s, stage %s, dry_run %s", cfg["run_name"], run_dir, args.stage, args.dry_run)
        _print_gpu_report()
        only = [r.strip() for r in args.runs.split(",") if r.strip()] if args.runs else None
        return pilot3.run(cfg, cfg_path, run_dir, args.stage, args.dry_run, only)
    finally:
        root = logging.getLogger()
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()


def _main_pilot4(args: argparse.Namespace, cfg_path: Path) -> int:
    """Pilot 4 (schema_version 4): the stages live in overclaim.experiments.pilot4."""
    pilot4 = _mod("experiments.pilot4")
    cfg = pilot4.load_config(cfg_path)
    run_dir = Path(args.results_root).resolve() / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    handlers = _setup_logging(run_dir)
    try:
        log.info("run %s (pilot 4): run_dir %s, stage %s, dry_run %s", cfg["run_name"], run_dir, args.stage,
                 args.dry_run)
        _print_gpu_report()
        only = [r.strip() for r in args.runs.split(",") if r.strip()] if args.runs else None
        return pilot4.run(cfg, cfg_path, run_dir, args.stage, args.dry_run, only)
    finally:
        root = logging.getLogger()
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    sys.exit(main())

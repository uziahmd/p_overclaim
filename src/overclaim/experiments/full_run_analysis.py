"""Session 5, Part 2: the stage `analysis` of the full run (brief Step 12 as revised; contract 6H).

`stage_analysis(ctx)` runs analysis_full_run.summarize_full_run on ctx.run_dir (results/full_run) and the frozen files:
it writes summary.md, summary.json and the figures (figures/, PNG and SVG; skipped and named when their inputs are
missing or matplotlib is not installed) and prints the summary. It never makes a call and needs no credential. It is
idempotent in the sense of contract 6H: every run rewrites the three outputs from the records on disk (the summary
carries its generation time, so two runs never give the same bytes), and reads whatever subset of the Part 2 records
exists, naming the missing ones as "not run yet". Dry run: which inputs exist; nothing is written.

The runner (overclaim.experiments.full_run) registers the stage: STAGES gains "analysis" (last) and _STAGE_FUNCS maps
it to stage_analysis; full_run.load_config can call check_config. The config block is CONFIG_BLOCKS's.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from overclaim import analysis_full_run
from overclaim.experiments.pilot2 import EXIT_FAILED, EXIT_OK
from overclaim.experiments.pilot3 import _record_stage

if TYPE_CHECKING:   # full_run registers this module's stage; importing it at run time here would be circular
    from overclaim.experiments.full_run import Ctx6

logger = logging.getLogger(__name__)

STAGE = "analysis"
# The config keys this stage reads: its own block, and the frozen files' paths and the caps of the shared blocks.
# Optional, read when present: e2.n_tie_seeds (contract 6F; default 100) and pairs (contract 6G; default O-H, O-P, H-P).
CONFIG_BLOCKS: dict[str, tuple[str, ...]] = {
    "analysis": ("n_bootstrap", "seed"),
    "freeze": ("tasks_out", "sets_out"),
    "budget": ("anthropic_usd_cap", "anthropic_stop_launching_usd", "openrouter_usd_cap",
               "openrouter_stop_launching_usd", "google_usd_cap"),
}


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def check_config(cfg: dict) -> None:
    """ValueError listing every problem of the analysis block: it must exist, n_bootstrap must be a positive integer
    and seed an integer; freeze.tasks_out and freeze.sets_out must be set."""
    problems = []
    block = cfg.get("analysis")
    if not isinstance(block, dict):
        problems.append("analysis block missing (append it from config_H.yaml)")
    else:
        if not (_int(block.get("n_bootstrap")) and block["n_bootstrap"] > 0):
            problems.append("analysis.n_bootstrap must be a positive integer")
        if not _int(block.get("seed")):
            problems.append("analysis.seed must be an integer")
    for key in ("tasks_out", "sets_out"):
        if not (cfg.get("freeze") or {}).get(key):
            problems.append(f"freeze.{key} missing")
    if problems:
        raise ValueError("invalid analysis config:\n" + "\n".join(problems))


def stage_analysis(ctx: Ctx6) -> int:
    """The analysis stage on a full_run.Ctx6. Dry run: one plan line per input (analysis_full_run.input_lines) and what
    would be written; nothing is written; an input that cannot be read (a malformed line) gives a "would FAIL" line
    and EXIT_FAILED, nothing recorded (the live stage would fail on it). Otherwise summarize_full_run(ctx.run_dir,
    ctx.cfg), the markdown printed, the stage recorded in stages_status.json (done with the missing inputs and the
    figures written and skipped, or failed with the reason). Returns EXIT_OK or EXIT_FAILED (a config problem, a
    missing frozen file, or an error while computing or writing)."""
    try:
        check_config(ctx.cfg)
    except ValueError as exc:
        logger.error("%s: FAILED: %s", STAGE, exc)
        if not ctx.dry_run:
            _record_stage(ctx, STAGE, {"status": "failed", "reason": str(exc)})
        return EXIT_FAILED
    if ctx.dry_run:
        try:
            lines = analysis_full_run.input_lines(ctx.run_dir, ctx.cfg)
        except Exception as exc:   # a malformed or partly written record line (analysis._read_jsonl's ValueError)
            reason = f"{type(exc).__name__}: {exc}"
            logger.error("%s: FAILED: %s", STAGE, reason)
            ctx.note(f"{STAGE}: would FAIL: {reason}")
            return EXIT_FAILED
        for line in lines:
            ctx.note(f"{STAGE}: {line}")
        ctx.note(f"{STAGE}: would write {ctx.run_dir / analysis_full_run.SUMMARY_MD}, "
                 f"{ctx.run_dir / analysis_full_run.SUMMARY_JSON} and the figures in "
                 f"{ctx.run_dir / analysis_full_run.FIGURES_DIR}/ (PNG and SVG), rewritten from the records; no call")
        return EXIT_OK
    try:
        markdown = analysis_full_run.summarize_full_run(ctx.run_dir, ctx.cfg)
    except Exception as exc:   # a missing frozen file (ValueError), a malformed record line, a write error
        logger.error("%s: FAILED: %s: %s", STAGE, type(exc).__name__, exc, exc_info=exc)
        _record_stage(ctx, STAGE, {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
        return EXIT_FAILED
    data = json.loads((ctx.run_dir / analysis_full_run.SUMMARY_JSON).read_text(encoding="utf-8"))
    _record_stage(ctx, STAGE, {"status": "done", "summary": analysis_full_run.SUMMARY_MD,
                               "generated_at": data["generated_at"], "missing_inputs": data["header"]["missing"],
                               "figures_written": data["figures"]["written"],
                               "figures_skipped": sorted(data["figures"]["skipped"])})
    print(markdown, flush=True)
    return EXIT_OK

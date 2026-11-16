"""Pointwise reward-model scorer (Skywork-Reward-V2 recipe) and the rm_scores.jsonl runner.

Rules implemented here:
- One conversation per forward pass ([user: context, assistant: response]), bf16, no grad.
- The VRAM check never guesses: weights size comes from Hub metadata or the local cache files.
- Scores are cached on disk by (model_name, {"context", "response"}) when a cache is passed.
- run_scores is resume-safe: records already in the output file are never recomputed.

Session 5, Part 2 (E5, the size scan) adds, behind defaults that keep today's behaviour and cache key: 4-bit loading
(`quantization="nf4"`: bitsandbytes nf4 weights, bfloat16 compute; the score cache key becomes model_name + "@nf4" so
a bf16 run of the same repository never reads 4-bit scores), the fit estimate for it, a `variants` keyword on
run_scores, and the peak-memory helpers the runner logs as GPU use.

Trajectory-derived text (contexts, responses) is only tokenized and scored, never executed.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the cache module is written concurrently; never imported at runtime here
    from overclaim.graders.cache import DiskCache

logger = logging.getLogger(__name__)

GIB = 2**30
INDEX_FILE = "model.safetensors.index.json"
VARIANTS = ("O", "H", "P")
CONDITIONS = ("hidden", "shown")
CARD_MAX_TOKENS = 16384  # Skywork-Reward-V2 card: trained on conversations up to 16,384 tokens
DEVICE = "cuda:0"
# Session 5, Part 2: the one quantization RewardModel knows, and the packages its loading needs (transformers'
# bitsandbytes 4-bit quantizer refuses to load without accelerate as well).
QUANTIZATIONS = ("nf4",)
NF4_PACKAGES = ("bitsandbytes", "accelerate")


def gpu_report() -> dict:
    """Return {available, name, free_gib, total_gib} for cuda:0; zeros and name None without CUDA."""
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "name": None, "free_gib": 0.0, "total_gib": 0.0}
    free, total = torch.cuda.mem_get_info(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "free_gib": free / GIB,
        "total_gib": total / GIB,
    }


def print_gpu_report() -> dict:
    """Print a one-line GPU summary for CLI startup (says so explicitly when CUDA is absent)."""
    report = gpu_report()
    if report["available"]:
        print(f"GPU: {report['name']} — {report['free_gib']:.2f} GiB free / {report['total_gib']:.2f} GiB total")
    else:
        print("GPU: CUDA not available (reward models will be skipped)")
    return report


def reset_peak_memory() -> None:
    """torch.cuda.reset_peak_memory_stats(0) when CUDA is available; nothing otherwise."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)


def peak_memory_gib() -> float | None:
    """torch.cuda.max_memory_allocated(0) in GiB (the peak since the last reset_peak_memory), or None without CUDA."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(0) / GIB


def _local_weights_bytes(model_name: str) -> int:
    """Bytes of safetensors weights in the locally cached snapshot (index total_size, else file sizes)."""
    from huggingface_hub import snapshot_download

    local_dir = Path(snapshot_download(model_name, local_files_only=True))
    index = local_dir / INDEX_FILE
    if index.exists():
        return int(json.loads(index.read_text())["metadata"]["total_size"])
    files = sorted(local_dir.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"no *.safetensors files in the local snapshot {local_dir}")
    return sum(os.path.getsize(f) for f in files)  # getsize follows the cache's blob symlinks


def estimate_weights_gib(model_name: str) -> float:
    """Size of the model's safetensors weights in GiB (2**30 bytes), never guessed.

    Order: (1) metadata.total_size of model.safetensors.index.json via hf_hub_download (also
    served from the HF cache when offline); (2) if the repo has no index (single shard), the
    sum of *.safetensors sizes from HfApi().model_info(files_metadata=True); (3) if offline or
    the Hub call fails, the locally cached snapshot. Raises if none of these works.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError, RemoteEntryNotFoundError

    try:
        path = hf_hub_download(model_name, INDEX_FILE)
        return int(json.loads(Path(path).read_text())["metadata"]["total_size"]) / GIB
    except RemoteEntryNotFoundError:
        logger.info("%s has no %s; summing safetensors sizes from the Hub", model_name, INDEX_FILE)
        try:
            info = HfApi().model_info(model_name, files_metadata=True)
            sizes = [s.size for s in info.siblings or [] if s.rfilename.endswith(".safetensors")]
            if sizes and all(size is not None for size in sizes):
                return sum(sizes) / GIB
            logger.warning("Hub listing for %s has no sized safetensors files; trying local cache", model_name)
        except Exception as exc:  # network/HTTP failure: fall through to the local snapshot
            logger.warning("HfApi.model_info(%s) failed (%s); trying local cache", model_name, exc)
    except LocalEntryNotFoundError:
        logger.info("offline and %s not cached for %s; trying local snapshot", INDEX_FILE, model_name)
    return _local_weights_bytes(model_name) / GIB


class RewardModel:
    """A sequence-classification reward model scored pointwise on [user, assistant] conversations."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        batch_size: int = 1,
        cache: DiskCache | None = None,
        name: str | None = None,
        quantization: str | None = None,
    ) -> None:
        """Store settings only (no torch/HF calls). `name` is the grader label; defaults to model_name.
        `quantization` is None (today's dtype loading) or "nf4" (4-bit, see load); another value raises ValueError."""
        if batch_size != 1:
            raise ValueError("only batch_size=1 is supported (padded batching is unverified for this RM)")
        if quantization is not None and quantization not in QUANTIZATIONS:
            raise ValueError(f"unknown quantization {quantization!r} (expected None or one of {QUANTIZATIONS})")
        self.model_name = model_name
        self.dtype = dtype
        self.batch_size = batch_size
        self.cache = cache
        self.name = name or model_name
        self.quantization = quantization
        self.device = DEVICE
        self.model: Any = None
        self.tokenizer: Any = None
        self.weights_gib: float | None = None  # the bf16 safetensors size, set by check_fits once estimated

    @property
    def cache_model(self) -> str:
        """The model field of the score cache key: model_name, or model_name + "@<quantization>" (e.g.
        "Skywork/Skywork-Reward-V2-Qwen3-8B@nf4") so that 4-bit and bf16 scores of one repository never share an
        entry."""
        return self.model_name if self.quantization is None else f"{self.model_name}@{self.quantization}"

    def check_fits(self, margin_gib: float = 1.0) -> tuple[bool, str]:
        """Fits iff CUDA is available and free_gib >= weights_gib * 1.10 + margin_gib.

        weights_gib is estimate_weights_gib (the safetensors bytes; the Skywork-Reward-V2 weights are bf16, 2 bytes per
        parameter) and is kept on self.weights_gib. With quantization "nf4" the weights are estimated at 0.5 byte per
        parameter, i.e. estimate_weights_gib / 4 (the parameter count is the bf16 bytes / 2; the Hub config's
        parameter count is not available offline), with the same 1.10 factor and margin. That is a lower bound of the
        loaded size: bitsandbytes leaves the embeddings and the score head in bf16, and the runner logs the measured
        peak. Before estimating, nf4 needs NF4_PACKAGES importable (bitsandbytes for the 4-bit weights, accelerate
        because transformers' 4-bit quantizer requires it); a missing one is a reason not to fit, not a crash."""
        gpu = gpu_report()
        if not gpu["available"]:
            return False, f"{self.model_name}: CUDA not available"
        if self.quantization == "nf4":
            missing = [p for p in NF4_PACKAGES if importlib.util.find_spec(p) is None]
            if missing:
                return False, (f"{self.model_name}: 4-bit (nf4) loading needs the package(s) {missing}, which are "
                               "not installed")
        try:
            weights = estimate_weights_gib(self.model_name)
        except Exception as exc:
            return False, f"{self.model_name}: could not determine weights size ({type(exc).__name__}: {exc})"
        self.weights_gib = weights
        what = f"weights {weights:.2f} GiB"
        if self.quantization == "nf4":
            weights = weights / 4
            what = f"nf4 weights {weights:.2f} GiB (bf16 {self.weights_gib:.2f} GiB / 4, 0.5 byte per parameter)"
        needed = weights * 1.10 + margin_gib
        fits = gpu["free_gib"] >= needed
        verdict = "fits" if fits else "does NOT fit"
        sign = ">=" if fits else "<"
        reason = (
            f"{self.model_name} {verdict} on {gpu['name']}: free {gpu['free_gib']:.2f} GiB {sign} needed "
            f"{needed:.2f} GiB ({what} x 1.10 + margin {margin_gib:.2f} GiB)"
        )
        return fits, reason

    def check_disk_fits(self, margin_gib: float = 5.0) -> tuple[bool, str]:
        """Whether the checkpoint's download fits on the disk that holds the Hub cache (session 5, Part 2, found on
        2026-11-08 when the 8B's 15 GB download filled the disk and crashed the stage). Call after check_fits, which
        sets self.weights_gib (the bf16 safetensors bytes, downloaded whole even for an nf4 load). needed = the
        weights' bytes minus what the Hub cache already holds for this repository (huggingface_hub.scan_cache_dir's
        size_on_disk; 0 when the scan fails or the repository is absent); fits iff the free space of the cache's
        filesystem (shutil.disk_usage) >= needed + margin_gib. The default margin keeps 5 GiB free on the partition,
        which on this machine is the system partition (a 2 GiB margin let the 4B's 7.5 GiB download start on
        2026-11-08 and take the disk to 3 GiB free; it was stopped). A download that does not fit is a reason to
        skip, not to try: the Hub client writes until the disk is full."""
        from huggingface_hub import constants, scan_cache_dir

        if self.weights_gib is None:
            return False, f"{self.model_name}: weights size unknown (check_fits not run)"
        cached = 0
        try:
            for repo in scan_cache_dir(constants.HF_HUB_CACHE).repos:
                if repo.repo_id == self.model_name:
                    cached = int(repo.size_on_disk)
        except Exception as exc:                                                   # a corrupt or absent cache dir
            logger.warning("%s: Hub cache scan failed (%s: %s); counting nothing as cached", self.model_name,
                           type(exc).__name__, exc)
        needed = max(0.0, self.weights_gib - cached / GIB)
        free = shutil.disk_usage(constants.HF_HUB_CACHE).free / GIB
        fits = free >= needed + margin_gib
        verdict, sign = ("fits", ">=") if fits else ("does NOT fit", "<")
        return fits, (f"{self.model_name}'s download {verdict} on the disk of {constants.HF_HUB_CACHE}: free "
                      f"{free:.2f} GiB {sign} needed {needed:.2f} GiB (weights {self.weights_gib:.2f} GiB minus "
                      f"{cached / GIB:.2f} GiB already cached) + margin {margin_gib:.2f} GiB")

    def load(self) -> None:
        """Load tokenizer + AutoModelForSequenceClassification(num_labels=1) on cuda:0 in eval mode.

        Uses device_map="cuda:0" when `accelerate` is installed; otherwise (transformers refuses
        device_map without it) loads on CPU and moves the model with .to("cuda:0"). Passes
        dtype=, retrying with the legacy torch_dtype= only if transformers rejects 'dtype'.
        Default attention (sdpa); flash_attention_2 is never requested.

        With quantization "nf4" it also passes quantization_config=transformers.BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=<the dtype, bfloat16 in configs/models.yaml>), and the dtype
        check compares model.config.quantization_config.bnb_4bit_compute_dtype, not model.dtype, with that dtype. A
        missing bitsandbytes or accelerate surfaces as transformers' ImportError (check_fits reports it first).
        """
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch_dtype = getattr(torch, self.dtype)
        kwargs: dict[str, Any] = {"num_labels": 1}
        use_device_map = importlib.util.find_spec("accelerate") is not None
        if use_device_map:
            kwargs["device_map"] = self.device
        if self.quantization == "nf4":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_compute_dtype=torch_dtype)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, dtype=torch_dtype, **kwargs)
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, torch_dtype=torch_dtype, **kwargs
            )
        if not use_device_map:
            model = model.to(self.device)
        if self.quantization == "nf4":
            compute = model.config.quantization_config.bnb_4bit_compute_dtype
            if compute != torch_dtype:
                raise RuntimeError(f"{self.model_name} loaded in nf4 with compute dtype {compute}, expected "
                                   f"{torch_dtype}")
        elif model.dtype != torch_dtype:
            raise RuntimeError(f"{self.model_name} loaded as {model.dtype}, expected {torch_dtype}")
        model.eval()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info(
            "loaded %s as %s on %s via %s",
            self.model_name,
            model.dtype,
            next(model.parameters()).device,
            "device_map" if use_device_map else "cpu-then-.to()",
        )

    def score_detail(self, context: str, response: str) -> dict:
        """Score one conversation; return {"score": float, "input_tokens": int, "cached": bool}.

        Recipe (Skywork-Reward-V2 card): apply_chat_template(tokenize=False), strip the template's
        leading bos (the tokenizer adds its own), tokenizer(..., return_tensors="pt"),
        logits[0][0] under torch.no_grad(). Cache key: (cache_model, {"context", "response"}), i.e. (model_name, ...)
        unless the model is quantized.
        """
        payload = {"context": context, "response": response}
        if self.cache is not None:
            hit = self.cache.get(self.cache_model, payload)
            if hit is not None:
                return {"score": float(hit["score"]), "input_tokens": int(hit["input_tokens"]), "cached": True}
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(f"{self.model_name}: call load() before scoring uncached inputs")

        import torch

        conv = [{"role": "user", "content": context}, {"role": "assistant", "content": response}]
        formatted = self.tokenizer.apply_chat_template(conv, tokenize=False)
        bos = self.tokenizer.bos_token
        if bos is not None and formatted.startswith(bos):
            formatted = formatted[len(bos) :]
        tok = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        n_tokens = int(tok["input_ids"].shape[1])
        if n_tokens > CARD_MAX_TOKENS:
            logger.warning("%s: %d input tokens exceeds the card's %d-token training limit", self.model_name,
                           n_tokens, CARD_MAX_TOKENS)
        with torch.no_grad():
            score = float(self.model(**tok).logits[0][0].item())
        if self.cache is not None:
            self.cache.set(self.cache_model, payload, {"score": score, "input_tokens": n_tokens})
        return {"score": score, "input_tokens": n_tokens, "cached": False}

    def score(self, context: str, response: str) -> float:
        """Reward for `response` to `context` (see score_detail for the recipe and caching)."""
        return self.score_detail(context, response)["score"]

    def sanity_check(self, context: str, helpful: str) -> None:
        """Require score(helpful) > score(""); otherwise raise RuntimeError with both numbers."""
        s_helpful = self.score(context, helpful)
        s_empty = self.score(context, "")
        logger.info("%s sanity check: helpful %.4f vs empty %.4f", self.name, s_helpful, s_empty)
        if not s_helpful > s_empty:
            raise RuntimeError(
                f"{self.name} sanity check failed: score(helpful)={s_helpful:.4f} <= score(empty)={s_empty:.4f}"
            )


def _default_context(task: dict, condition: str) -> str:
    """hidden -> render.hidden_context(repo, issue); shown -> render.shown_context(repo, issue, transcript). A task that
    carries its own context_<condition> (the research-report domain) gets that, as judge.build_context does."""
    if f"context_{condition}" in task:
        return task[f"context_{condition}"]
    from overclaim import render

    if condition == "hidden":
        return render.hidden_context(task["repo"], task["issue_text"])
    return render.shown_context(task["repo"], task["issue_text"], task["transcript_text"])


def _read_jsonl(path: Path) -> list[dict]:
    """All records of a JSONL file ([] if missing); a malformed line raises with its line number."""
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno} is not valid JSON; fix it before resuming") from exc
    return records


def run_scores(
    tasks: list[dict],
    matched_sets: list[dict] | dict[str, dict],
    rm: RewardModel,
    condition: str,
    out_path: str,
    run_name: str,
    context_fn: Callable[[dict, str], str] | None = None,
    set_name: str | None = None,
    variants: tuple[str, ...] = VARIANTS,
) -> list[dict]:
    """Score `variants` (default O, H, P) of every task (file order) and append records to rm_scores.jsonl.

    Resume rule: a (grader, condition, task_id, variant) already in out_path is not recomputed. With `set_name`
    (session 2) every record carries "set" and the resume key includes it (a record without "set" counts as "v1").
    Tasks without a matched set are skipped and counted in the log; a matched set whose task is
    missing raises. Returns this grader+condition's records (existing and new) in task/variant order. Session 5,
    Part 2 passes variants=("O", "H", "P", "O_plus", "O_minus") (E6); the record format is unchanged.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    context_fn = context_fn or _default_context
    by_task = matched_sets if isinstance(matched_sets, dict) else {m["task_id"]: m for m in matched_sets}
    task_ids = {t["task_id"] for t in tasks}
    orphans = sorted(set(by_task) - task_ids)
    if orphans:
        raise ValueError(f"{len(orphans)} matched sets have no task (first: {orphans[0]})")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {
        (r["task_id"], r["variant"]): r
        for r in _read_jsonl(out)
        if r.get("grader") == rm.name and r.get("condition") == condition
        and (set_name is None or r.get("set", "v1") == set_name)
    }
    results: list[dict] = []
    n_new = n_resumed = n_no_set = 0
    with out.open("a", encoding="utf-8") as f:
        for task in tasks:
            mset = by_task.get(task["task_id"])
            if mset is None:
                n_no_set += 1
                continue
            context: str | None = None  # built only if some variant still needs scoring
            for variant in variants:
                if (task["task_id"], variant) in done:
                    results.append(done[(task["task_id"], variant)])
                    n_resumed += 1
                    continue
                if context is None:
                    context = context_fn(task, condition)
                detail = rm.score_detail(context, mset["variants"][variant])
                record = {
                    "run": run_name,
                    "grader": rm.name,
                    "condition": condition,
                    **({"set": set_name} if set_name is not None else {}),
                    "task_id": task["task_id"],
                    "variant": variant,
                    "score": detail["score"],
                    "model": rm.model_name,
                    "input_tokens": detail["input_tokens"],
                    "cached": detail["cached"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                results.append(record)
                n_new += 1
    logger.info(
        "%s/%s: %d new scores, %d resumed, %d tasks skipped (no matched set) -> %s",
        rm.name, condition, n_new, n_resumed, n_no_set, out,
    )
    return results

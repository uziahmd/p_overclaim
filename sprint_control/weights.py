"""Are a reward model's weights the files of the main run? (Amendment 1 of 2026-11-14, sprint_control/README.md,
section 11.) No model is loaded here and no score is read.

What the main run recorded. results/full_run/rm_status.json gives, per reward model, the repository and the byte
count of its weights (`weights_bytes`). It records no revision and no file hash.

What else exists. The Hub client (huggingface_hub) keeps one lock file per file it downloads, in
<hub cache>/.locks/models--<owner>--<name>/, and names it after the file's hash as the Hub states it: the sha256 of
the content for a large file, git's blob sha1 for a small one. The lock files of the five repositories carry the
dates of the downloads for the pilot (2026-11-05, the 3B model) and for the main run (2026-11-08, the other four),
and they outlived the weights that were deleted since. They are a trace that the Hub client left on this machine on
those days, not a record written by the main run.

`save_trace` copies that trace, with the main run's recorded byte counts, to sprint_control/main_run_weights.json.
It is run once, before any download of this run, because a new download writes its own lock files; the file is
committed before the first score. `identity` then hashes every file of the snapshot that a model is loaded from and
compares it with the committed list, never with the live folder.

A model's weights are confirmed when
- every file of the snapshot has a content hash that is in the trace of its repository (a large file by sha256, a
  small one by git's blob sha1), so the weights, the configuration and the tokenizer are byte-identical to files
  that the Hub client downloaded on those days;
- the snapshot holds at least one weights file (*.safetensors); and
- the byte count of the weights, taken as the main run took it (RewardModel.weights_gib), equals the main run's.
A hash of the trace that no file of the snapshot has is listed and does not decide anything: the loader may need
fewer files than were downloaded then.

Run once, before any download:   uv run python sprint_control/weights.py --save-trace
Check a cached model, offline:   HF_HUB_OFFLINE=1 uv run python sprint_control/weights.py --check NAME
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

import control_common as K

TRACE_FILE = K.HERE / "main_run_weights.json"
MAIN_RM_STATUS = K.MAIN_RUN / "rm_status.json"
CHUNK = 1 << 20
WEIGHTS_SUFFIX = ".safetensors"
ANCHOR_FILE = "config.json"        # every model repository has it; its folder is the snapshot the model loads from


def repo_folder(repository: str) -> str:
    """The Hub cache's folder name of a model repository ("Owner/Name" -> "models--Owner--Name")."""
    return "models--" + repository.replace("/", "--")


def hash_kind(name: str) -> str:
    return {64: "sha256", 40: "git-sha1"}.get(len(name), "unknown")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def git_sha1_of(path: Path) -> str:
    """git's blob hash of a file: sha1 of "blob <size>\\0" followed by the content (the Hub's hash of a small
    file)."""
    path = Path(path)
    h = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def reward_models() -> list[dict]:
    """The reward models of the run: the list of configs/models.yaml that config.yaml names."""
    cfg = yaml.safe_load(K.CONFIG.read_text(encoding="utf-8"))
    return yaml.safe_load(K.MODELS_CONFIG.read_text(encoding="utf-8"))[cfg["reward_models"]]


def build_trace(locks_root: Path) -> dict:
    """The trace of every reward model: its repository, what the main run recorded, and the lock files of the
    repository (hash, kind, time written). Reads file names and times only."""
    main = K.read_json(MAIN_RM_STATUS)
    models = {}
    for spec in reward_models():
        folder = Path(locks_root) / repo_folder(spec["model"])
        locks = sorted(folder.glob("*.lock")) if folder.is_dir() else []
        entry = main.get(spec["name"], {})
        if entry.get("model") != spec["model"]:
            raise ValueError(f"{spec['name']}: the main run recorded the repository {entry.get('model')!r}, the "
                             f"config names {spec['model']!r}")
        models[spec["name"]] = {
            "repository": spec["model"],
            "main_run": {"status": entry.get("status"), "weights_bytes": entry.get("weights_bytes"),
                         "revision": None, "file_hashes": None},
            "files": [{"hash": p.name[:-len(".lock")], "kind": hash_kind(p.name[:-len(".lock")]),
                       "lock_written": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(
                           timespec="seconds")} for p in locks]}
    return {"what": "The hashes of the files that the Hub client downloaded for each reward model's repository on "
                    "2026-11-05 (pilot) and 2026-11-08 (main run), read from the names of its lock files, and what "
                    "the main run recorded about the weights (results/full_run/rm_status.json: the repository and "
                    "the byte count; no revision and no file hash).",
            "locks_folder": str(locks_root), "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "models": models}


def save_trace() -> dict:
    """Write TRACE_FILE once. SystemExit when it exists: a download of this run would have added its own lock
    files."""
    from huggingface_hub import constants

    if TRACE_FILE.exists():
        raise SystemExit(f"{TRACE_FILE.relative_to(K.REPO)} exists: the trace is saved once, before any download")
    trace = build_trace(Path(constants.HF_HUB_CACHE) / ".locks")
    empty = [name for name, m in trace["models"].items() if not m["files"]]
    if empty:
        raise SystemExit(f"no lock file found for {empty}; nothing written")
    K.write_json(TRACE_FILE, trace)
    return trace


def load_trace() -> dict:
    trace = K.read_json(TRACE_FILE)
    if trace is None:
        raise SystemExit(f"{TRACE_FILE.relative_to(K.REPO)} is missing (weights.py --save-trace)")
    return trace["models"]


def snapshot_of(repository: str) -> Path:
    """The cached snapshot folder a model is loaded from, found without any request to the Hub."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repository, ANCHOR_FILE, local_files_only=True)).parent


def identity(repository: str, entry: dict, weights_bytes: int | None, snapshot: Path | None = None) -> dict:
    """Compare the files of the snapshot with the trace `entry` of the model (see the module text). `weights_bytes`
    is the byte count of the weights as the main run took it, or None when it is not known."""
    if entry.get("repository") != repository:
        return {"repository": repository, "confirmed": False,
                "reason": f"the trace is for the repository {entry.get('repository')!r}"}
    snapshot = snapshot_of(repository) if snapshot is None else Path(snapshot)
    known = {f["hash"]: f for f in entry["files"]}
    files, seen = [], set()
    for path in sorted(p for p in snapshot.rglob("*") if not p.is_dir()):
        digest, kind = sha256_of(path), "sha256"
        if digest not in known:                      # a small file: the Hub names it by git's blob sha1
            other = git_sha1_of(path)
            if other in known:
                digest, kind = other, "git-sha1"
        in_trace = digest in known and known[digest]["kind"] == kind
        if in_trace:
            seen.add(digest)
        files.append({"file": path.relative_to(snapshot).as_posix(), "bytes": path.stat().st_size, "hash": digest,
                      "kind": kind, "in_trace": in_trace,
                      "lock_written": known[digest]["lock_written"] if in_trace else None})
    weights = [f for f in files if f["file"].endswith(WEIGHTS_SUFFIX)]
    recorded = entry["main_run"].get("weights_bytes")
    outside = [f["file"] for f in files if not f["in_trace"]]
    problems = []
    if outside:
        problems.append(f"{len(outside)} file(s) of the snapshot are not in the trace: {', '.join(outside)}")
    if not weights:
        problems.append("the snapshot holds no weights file")
    if weights_bytes is None or recorded is None or int(weights_bytes) != int(recorded):
        problems.append(f"the byte count of the weights is {weights_bytes} here and {recorded} in the main run")
    return {"repository": repository, "revision": snapshot.name, "files": files,
            "n_files": len(files), "n_files_in_trace": sum(f["in_trace"] for f in files),
            "weights_files": [f["file"] for f in weights],
            "weights_files_bytes": sum(f["bytes"] for f in weights),
            "weights_bytes": weights_bytes, "weights_bytes_main_run": recorded,
            "trace_hashes_without_a_file": sorted(set(known) - seen),
            "confirmed": not problems, "reason": "; ".join(problems)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument("--check", metavar="NAME", help="a reward model's name in configs/models.yaml")
    args = parser.parse_args(argv)
    if args.save_trace:
        trace = save_trace()
        for name, m in trace["models"].items():
            dates = sorted({f["lock_written"][:10] for f in m["files"]})
            print(f"{name}: {len(m['files'])} hashes, lock files of {', '.join(dates)}; main run: "
                  f"{m['main_run']['weights_bytes']} bytes of weights")
        return 0
    if args.check:
        from overclaim.graders import rm

        spec = {s["name"]: s for s in reward_models()}[args.check]
        got = identity(spec["model"], load_trace()[args.check],
                       round(rm.estimate_weights_gib(spec["model"]) * rm.GIB))
        for f in got.get("files", []):
            print(f"  {'in the trace' if f['in_trace'] else 'NOT in trace'}  {f['kind']:8} {f['hash'][:16]}  "
                  f"{f['bytes']:>13,}  {f['file']}")
        print(f"{args.check}: revision {got.get('revision')}; confirmed = {got['confirmed']}"
              + (f"; {got['reason']}" if got["reason"] else ""))
        return 0 if got["confirmed"] else 1
    parser.error("give --save-trace or --check NAME")
    return 2


if __name__ == "__main__":
    sys.exit(main())

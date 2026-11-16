"""Domain 3 data: Multi-News clusters -> the source pool of the research-report harness.

Source: Multi-News (Fabbri et al., 2019), Hugging Face dataset `alexfabbri/multi_news` (the name `multi_news`
resolves to it). The dataset card's licence field says `other`: non-commercial research use. File
`data/test.src.cleaned` of the test split: one cluster per line, sources separated by `|||||`, line breaks inside a
source written as the token `NEWLINE_CHAR`.

Only that one file is downloaded (hf_hub_download). The dataset's loading script `multi_news.py` is never run and the
`datasets` library is never imported. The file is untrusted text: it is split, counted and written out, nothing more.

The pool file holds source text and is gitignored. Committed files of this domain never hold source text (only ids,
titles, word counts and page numbers): the licence of the news text is not ours to pass on.

Pages (`pages`) are runs of `page_words` whitespace-separated words joined by single spaces, so paragraph breaks are
not preserved in what the agent reads.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

from overclaim.paths import REPO_ROOT, load_env

logger = logging.getLogger(__name__)

REPO_ID = "alexfabbri/multi_news"
SOURCE_FILE = "data/test.src.cleaned"
SEPARATOR = "|||||"
NEWLINE_TOKEN = "NEWLINE_CHAR"
TITLE_CHARS = 100
TITLE_WORDS = 12
PAGE_WORDS = 800
ELIGIBLE = "eligible"
REASONS = ("too_few_sources", "too_many_sources", "source_too_short", "source_too_long")

_SPACES = re.compile(r" {2,}")


def _resolve(path: str | Path) -> Path:
    """An absolute path as given; a relative one is taken from the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def download(raw_dir: str | Path = "data/raw/research/multi_news") -> Path:
    """Path of <raw_dir>/data/test.src.cleaned, downloaded with huggingface_hub.hf_hub_download if it is missing.

    No call is made when the file is already there.
    """
    local_dir = _resolve(raw_dir)
    path = local_dir / SOURCE_FILE
    if path.exists():
        return path
    load_env()  # an HF token in .env, if any, is picked up from the environment by huggingface_hub
    from huggingface_hub import hf_hub_download

    logger.info("downloading %s from dataset %s into %s", SOURCE_FILE, REPO_ID, local_dir)
    return Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=SOURCE_FILE, local_dir=local_dir))


def parse_cluster(line: str) -> list[str]:
    """The sources of one cluster line: split on `|||||`, `NEWLINE_CHAR` -> "\\n", runs of spaces collapsed to one,
    stripped; empty sources are dropped."""
    out = []
    for raw in line.split(SEPARATOR):
        text = _SPACES.sub(" ", raw.replace(NEWLINE_TOKEN, "\n")).strip()
        if text:
            out.append(text)
    return out


def word_count(text: str) -> int:
    """len(text.split())."""
    return len(text.split())


def eligible(sources: list[str], cfg: dict) -> tuple[bool, str]:
    """(True, "eligible") or (False, reason) for one cluster, with the bounds of cfg["data"] (both inclusive).

    Checked in this order: too_few_sources / too_many_sources (n_sources), then source_too_short if any source has
    fewer words than source_words[0], then source_too_long if any source has more than source_words[1].
    """
    lo_n, hi_n = cfg["data"]["n_sources"]
    lo_w, hi_w = cfg["data"]["source_words"]
    if len(sources) < lo_n:
        return False, "too_few_sources"
    if len(sources) > hi_n:
        return False, "too_many_sources"
    counts = [word_count(s) for s in sources]
    if any(c < lo_w for c in counts):
        return False, "source_too_short"
    if any(c > hi_w for c in counts):
        return False, "source_too_long"
    return True, ELIGIBLE


def title(source: str) -> str:
    """The source's title as listed to the agent.

    With a line break: the first non-empty line (stripped); a line longer than 100 characters is cut at the last word
    boundary within its first 100 characters (a hard cut when there is none) and "..." is appended. Without a line
    break: the first 12 words joined by spaces, then "...".
    """
    if "\n" not in source:
        return " ".join(source.split()[:TITLE_WORDS]) + "..."
    line = next(ln.strip() for ln in source.split("\n") if ln.strip())
    if len(line) <= TITLE_CHARS:
        return line
    cut = line[:TITLE_CHARS]
    if not line[TITLE_CHARS].isspace() and " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + "..."


def pages(source: str, page_words: int = PAGE_WORDS) -> list[str]:
    """Consecutive pages of `page_words` words (split on whitespace, joined by single spaces); the last may be
    shorter. Paragraph breaks are not preserved."""
    words = source.split()
    return [" ".join(words[i:i + page_words]) for i in range(0, len(words), page_words)]


def build_pool(cfg: dict, funnel: dict) -> list[dict]:
    """Read every line of the test file, count eligibility reasons into `funnel`, sample the pool and write it.

    `funnel` is filled in place: "lines" (all lines read) and one count per outcome of `eligible` ("eligible" and
    each reason, 0 when none). The pool is random.Random(cfg["seed"]).sample over the eligible 0-based line indices,
    pool_size of them, sorted by line index; fewer eligible clusters than pool_size raises ValueError (after `funnel`
    is filled).
    Writes <processed_dir>/<pool_file>, one cluster per line: {"cluster_id": "mn-test-<line_index>", "line_index",
    "sources": [{"id": "S1", "title", "n_words", "n_pages", "text"}, ...]} with ids in file order. Lines are split on
    "\\n" only (the file is read as bytes), so a stray carriage return inside a source does not start a new line.
    Returns the pool records.
    """
    d = cfg["data"]
    path = download(d["raw_dir"])
    clusters: dict[int, list[str]] = {}
    funnel.update({"lines": 0, ELIGIBLE: 0, **{r: 0 for r in REASONS}})
    with path.open("rb") as f:
        for line_index, raw in enumerate(f):
            funnel["lines"] += 1
            sources = parse_cluster(raw.decode("utf-8"))
            ok, reason = eligible(sources, cfg)
            funnel[reason] += 1
            if ok:
                clusters[line_index] = sources
    logger.info("research pool funnel: %s", json.dumps(funnel, sort_keys=True))
    if len(clusters) < d["pool_size"]:
        raise ValueError(f"only {len(clusters)} eligible clusters in {path}; pool_size is {d['pool_size']}")
    chosen = sorted(random.Random(cfg["seed"]).sample(sorted(clusters), d["pool_size"]))
    pool = []
    for line_index in chosen:
        pool.append({
            "cluster_id": f"mn-test-{line_index}",
            "line_index": line_index,
            "sources": [
                {"id": f"S{k}", "title": title(text), "n_words": word_count(text),
                 "n_pages": len(pages(text, d["page_words"])), "text": text}
                for k, text in enumerate(clusters[line_index], 1)
            ],
        })
    out = _resolve(d["processed_dir"]) / d["pool_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in pool:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("wrote %d pool clusters to %s", len(pool), out)
    return pool


def load_pool(path: str | Path) -> list[dict]:
    """The pool records of a pool.jsonl file, in file order (a missing file raises FileNotFoundError)."""
    with _resolve(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

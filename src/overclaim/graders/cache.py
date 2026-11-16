"""Content-addressed disk cache for every external grader call.

Rules implemented here:
- key = sha256(model + json.dumps(payload, sort_keys=True, ensure_ascii=False)), so dict key order
  never changes the key and any change to a request-shaping field gives a new key.
- layout: <root>/<model slug>/<key[:2]>/<key>.json, model slug = model with '/' and ':' replaced by '_'.
- each file holds {"model", "payload", "value", "created_at"}; `get` returns only "value".
- writes are atomic (temp file in the same directory, then an atomic publish) and an existing entry
  is never overwritten. Entries are never deleted by this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def model_slug(model: str) -> str:
    """Directory name for a model: '/' and ':' replaced by '_'."""
    return model.replace("/", "_").replace(":", "_")


class DiskCache:
    """JSON-file cache keyed by (model, payload). Never overwrites, never deletes."""

    def __init__(self, root: str = "cache") -> None:
        self.root = Path(root)

    @staticmethod
    def key(model: str, payload: dict) -> str:
        """sha256 hex digest of model + json.dumps(payload, sort_keys=True, ensure_ascii=False)."""
        blob = model + json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def path(self, model: str, payload: dict) -> Path:
        """File path of the entry: <root>/<model slug>/<key[:2]>/<key>.json."""
        k = self.key(model, payload)
        return self.root / model_slug(model) / k[:2] / f"{k}.json"

    def get(self, model: str, payload: dict) -> dict | None:
        """Return the stored value for (model, payload), or None if there is no entry."""
        p = self.path(model, payload)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            record = json.load(f)
        return record["value"]

    def set(self, model: str, payload: dict, value: dict) -> None:
        """Store value atomically; if an entry already exists it is kept and value is discarded.

        The temp file is published with os.link, which fails if the target exists, so even two
        concurrent writers cannot overwrite each other.
        """
        p = self.path(model, payload)
        if p.exists():
            logger.debug("cache entry exists, not overwriting: %s", p)
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "model": model,
            "payload": payload,
            "value": value,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(tmp, p)
            except FileExistsError:
                logger.debug("cache entry appeared concurrently, not overwriting: %s", p)
        finally:
            os.unlink(tmp)

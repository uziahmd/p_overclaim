"""Repository paths and the one-time .env loader.

All paths are absolute and derived from this file's location
(src/overclaim/paths.py -> repo root), so they do not depend on the cwd.
"""

from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DIR: Path = REPO_ROOT / "data" / "raw" / "coding"
PROCESSED_DIR: Path = REPO_ROOT / "data" / "processed" / "coding"
RESULTS_DIR: Path = REPO_ROOT / "results"
CACHE_DIR: Path = REPO_ROOT / "cache"
ENV_FILE: Path = REPO_ROOT / ".env"

_env_loaded = False


def load_env() -> None:
    """Load <repo>/.env into os.environ once, with an explicit path.

    The argless load_dotenv() searches from the caller's frame and crashes when
    Python runs from stdin or -c, so the path is always given. Variables that
    are already set in the environment win (override=False).
    """
    global _env_loaded
    if not _env_loaded:
        load_dotenv(dotenv_path=ENV_FILE)
        _env_loaded = True

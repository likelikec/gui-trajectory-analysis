"""Common helpers shared by multiple mode implementations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


def normalize_chat_completions_url(base_url: str) -> str:
    """Normalizes an OpenAI-compatible base URL to a chat completions endpoint."""
    url = base_url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def sanitize_name(value: str, *, limit: int = 80) -> str:
    """Returns a filesystem-safe slug."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not sanitized:
        sanitized = "task"
    return sanitized[:limit]


def make_task_dir(base_dir: str | Path, prefix: str, goal: str) -> Path:
    """Creates and returns a timestamped task directory."""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    task_dir = base_path / f"{prefix}_{stamp}_{sanitize_name(goal, limit=40)}"
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir

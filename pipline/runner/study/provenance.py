"""Deterministic run provenance for experiment manifests.

The study workspace is sometimes distributed without Git metadata.  A
path-and-content source fingerprint therefore remains the canonical code
version, while Git information is recorded as an optional additional fact.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SOURCE_FINGERPRINT_ALGORITHM = "sha256_path_and_content_v1"
_SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".mjs", ".sh", ".toml"}
_EXCLUDED_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "backups",
    "outputs",
    "results",
    "tests",
}


def sha256_file(path: str | Path) -> str | None:
    """Hash a file without putting its contents in the manifest."""
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value using a stable canonical encoding."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_files(project_root: Path, roots: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _SOURCE_SUFFIXES:
                continue
            relative = path.relative_to(project_root)
            if any(part in _EXCLUDED_PARTS for part in relative.parts):
                continue
            files.append(path)
    return sorted(set(files), key=lambda item: item.relative_to(project_root).as_posix())


def source_fingerprint(
    project_root: str | Path,
    *,
    source_roots: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Return a deterministic fingerprint of code that can affect a run."""
    root = Path(project_root).resolve()
    if source_roots is None:
        roots = [
            root / "runner" / "study",
            root / "agents" / "guiagent",
            root / "vendor" / "mobile-use" / "mobile_use",
            root / "vendor" / "AppAgent" / "scripts",
            root / "vendor" / "MobileAgent" / "Mobile-Agent-v3" / "android_world_v3" / "android_world" / "agents",
            root / "android_world" / "android_world",
        ]
    else:
        roots = [Path(path).resolve() for path in source_roots]

    files = _source_files(root, roots)
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return {
        "algorithm": SOURCE_FINGERPRINT_ALGORITHM,
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "roots": [str(path.relative_to(root)) for path in roots if path.exists()],
    }


def git_provenance(project_root: str | Path) -> dict[str, Any]:
    """Record Git facts when present; never pretend a commit exists."""
    root = Path(project_root).resolve()
    try:
        commit_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "commit": None,
            "dirty": None,
            "missing_reason": f"git_probe_failed:{type(exc).__name__}",
        }
    if commit_result.returncode != 0:
        return {
            "available": False,
            "commit": None,
            "dirty": None,
            "missing_reason": "project_not_in_git_worktree",
        }
    status_result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return {
        "available": True,
        "commit": commit_result.stdout.strip() or None,
        "dirty": bool(status_result.stdout.strip()) if status_result.returncode == 0 else None,
        "missing_reason": None,
    }


def build_run_provenance(
    *,
    project_root: str | Path,
    defaults_path: str | Path,
    effective_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable provenance block stored in every run manifest."""
    defaults = Path(defaults_path).resolve()
    code = source_fingerprint(project_root)
    return {
        "version": 1,
        "defaults_file": {
            "path": str(defaults),
            "sha256": sha256_file(defaults),
            "missing_reason": None if defaults.is_file() else "defaults_file_missing",
        },
        "effective_config_sha256": sha256_json(effective_config),
        "code_version": f"source-sha256:{code['sha256']}",
        "source_fingerprint": code,
        "git": git_provenance(project_root),
    }


def collect_model_snapshots(run_output_dir: str | Path) -> dict[str, Any]:
    """Aggregate provider-returned model fields from canonical call ledgers."""
    root = Path(run_output_dir)
    resolved_counts: dict[str, int] = {}
    requested_counts: dict[str, int] = {}
    call_count = 0
    for ledger in sorted(root.glob("task_reports/*/logs/llm_interactions.jsonl")):
        try:
            lines = ledger.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                call = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(call, dict):
                continue
            call_count += 1
            requested = call.get("requested_model_name")
            resolved = call.get("resolved_model_name")
            if requested:
                requested_counts[str(requested)] = requested_counts.get(str(requested), 0) + 1
            if resolved:
                resolved_counts[str(resolved)] = resolved_counts.get(str(resolved), 0) + 1
    return {
        "source": "task_reports/*/logs/llm_interactions.jsonl",
        "call_count_scanned": call_count,
        "requested_model_counts": requested_counts,
        "resolved_model_counts": resolved_counts,
        "resolved_model_names": sorted(resolved_counts),
        "resolved_model_missing_reason": (
            None
            if resolved_counts
            else "provider_response_model_field_not_recorded"
        ),
    }

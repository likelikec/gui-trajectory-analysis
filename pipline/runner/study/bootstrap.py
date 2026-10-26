"""Shared import/bootstrap helpers for local benchmark checkouts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOBILE_USE_ROOT = PROJECT_ROOT / "vendor" / "mobile-use"
MOBILE_AGENT_V3_ANDROID_WORLD_ROOT = (
    PROJECT_ROOT / "vendor" / "MobileAgent" / "Mobile-Agent-v3" / "android_world_v3"
)
ANDROID_WORLD_ROOT = MOBILE_AGENT_V3_ANDROID_WORLD_ROOT
ANDROID_LAB_ROOT = PROJECT_ROOT / "vendor" / "Android-Lab"


def _patch_darwin_psutil_pid_listing() -> None:
    """Avoid a macOS psutil sysctl hang seen while AndroidWorld imports."""
    if sys.platform != "darwin":
        return

    import psutil

    if getattr(psutil, "_study_pid_listing_patch", False):
        return

    def _pids_via_pgrep() -> list[int]:
        result = subprocess.run(
            ["/usr/bin/pgrep", "-f", "."],
            capture_output=True,
            text=True,
            timeout=15,
        )
        # pgrep returns 1 when there are no matches; our broad pattern normally
        # matches every userspace process, but keep the fallback well-defined.
        if result.returncode not in (0, 1):
            raise subprocess.CalledProcessError(result.returncode, result.args)
        pids = {0, os.getpid()}
        pids.update(int(value) for value in result.stdout.split() if value.isdigit())
        return sorted(pids)

    # psutil 7.2.x calls the Darwin C-extension's sysctl-based pids() from
    # Process.parent()/process_iter().  On the current host that syscall can
    # remain uninterruptible after emulator/setup processes exit.  pgrep uses a
    # responsive process-enumeration path and provides the PID set callers need.
    psutil._psplatform.pids = _pids_via_pgrep
    psutil.pids = _pids_via_pgrep
    psutil._study_pid_listing_patch = True


def find_android_sdk_root() -> Path | None:
    candidates: list[Path] = []
    for env_name in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Android" / "Sdk")

    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        candidates.append(Path(user_profile) / "AppData" / "Local" / "Android" / "Sdk")

    username = os.environ.get("USERNAME")
    if username:
        for drive in ("C:", "D:"):
            candidates.append(Path(f"{drive}\\Users\\{username}\\AppData\\Local\\Android\\Sdk"))

    candidates.append(Path.home() / "AppData" / "Local" / "Android" / "Sdk")

    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if (candidate / "platform-tools" / "adb.exe").exists() or (candidate / "platform-tools" / "adb").exists():
            return candidate
    return None


def ensure_android_sdk_env() -> Path | None:
    sdk_root = find_android_sdk_root()
    if sdk_root is None:
        return None

    sdk_root_str = str(sdk_root)
    os.environ.setdefault("ANDROID_SDK_ROOT", sdk_root_str)
    os.environ.setdefault("ANDROID_HOME", sdk_root_str)

    path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    additions = [
        sdk_root / "platform-tools",
        sdk_root / "emulator",
        sdk_root / "tools",
        sdk_root / "tools" / "bin",
    ]
    for addition in reversed(additions):
        addition_str = str(addition)
        if addition.exists() and addition_str not in path_entries:
            path_entries.insert(0, addition_str)
    os.environ["PATH"] = os.pathsep.join(path_entries)
    return sdk_root


def ensure_import_paths() -> None:
    """Adds local benchmark roots to sys.path if needed."""
    _patch_darwin_psutil_pid_listing()
    ensure_android_sdk_env()
    roots = [
        MOBILE_AGENT_V3_ANDROID_WORLD_ROOT,
        PROJECT_ROOT / "runner",
        PROJECT_ROOT / "agents",
        MOBILE_USE_ROOT,
        ANDROID_WORLD_ROOT,
        ANDROID_LAB_ROOT,
    ]
    for root in reversed(roots):
        root_str = str(root)
        if root.exists() and root_str in sys.path:
            sys.path.remove(root_str)
        if root.exists():
            sys.path.insert(0, root_str)

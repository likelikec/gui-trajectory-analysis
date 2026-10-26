"""Strict benchmark-specific app package resolution for study wrappers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from study.bootstrap import PROJECT_ROOT


_ANDROIDWORLD_PACKAGES = {
    "audio recorder": "com.dimowner.audiorecorder",
    "broccoli app": "com.flauschcode.broccoli",
    "camera": "com.android.camera2",
    "chrome": "com.android.chrome",
    "clipper": "ca.zgrs.clipper",
    "clock": "com.google.android.deskclock",
    "contacts": "com.google.android.contacts",
    "dialer": "com.google.android.dialer",
    "files": "com.google.android.documentsui",
    "gmail": "com.google.android.gm",
    "google chrome": "com.android.chrome",
    "google contacts": "com.google.android.contacts",
    "joplin": "net.cozic.joplin",
    "markor": "net.gsantner.markor",
    "messages": "com.google.android.apps.messaging",
    "open tracks sports tracker": "de.dennisguse.opentracks",
    "osmand": "net.osmand",
    "phone": "com.google.android.dialer",
    "pro expense": "com.arduia.expense",
    "retro music": "code.name.monkey.retromusic",
    "settings": "com.android.settings",
    "simple calendar": "com.simplemobiletools.calendar.pro",
    "simple calendar pro": "com.simplemobiletools.calendar.pro",
    "simple draw pro": "com.simplemobiletools.draw.pro",
    "simple gallery pro": "com.simplemobiletools.gallery.pro",
    "simple sms messenger": "com.simplemobiletools.smsmessenger",
    "tasks": "org.tasks",
    "vlc": "org.videolan.vlc",
    "youtube": "com.google.android.youtube",
}


def _normalize_app_name(app_name: str | None) -> str:
    return " ".join((app_name or "").strip().lower().split())


@lru_cache(maxsize=1)
def _androidlab_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    config_dir = PROJECT_ROOT / "Android-Lab" / "evaluation" / "config"
    for config_path in sorted(config_dir.glob("*.yaml")):
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        app_name = _normalize_app_name(data.get("APP"))
        package_name = str(data.get("package") or "").strip()
        if app_name and package_name:
            packages[app_name] = package_name
            packages[config_path.stem.lower()] = package_name
    return packages


def resolve_package_name(
    *,
    benchmark: str,
    app_name: str | None = None,
    package_name: str | None = None,
) -> str | None:
    explicit_package = str(package_name or "").strip()
    if explicit_package:
        return explicit_package

    normalized_name = _normalize_app_name(app_name)
    if not normalized_name:
        return None

    benchmark_key = _normalize_app_name(benchmark).replace("-", "")
    if benchmark_key == "androidworld":
        return _ANDROIDWORLD_PACKAGES.get(normalized_name)
    if benchmark_key == "androidlab":
        return _androidlab_packages().get(normalized_name)
    return None

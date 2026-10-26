"""Small, architecture-neutral helpers for runtime evidence collection.

This module deliberately does not know how an agent chooses an action.  It
only normalizes provider metadata, writes state evidence, and checks the
accounting layers used by the study reports.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _raw_usage(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    result: dict[str, Any] = {}
    for name in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    ):
        if hasattr(value, name):
            result[name] = getattr(value, name)
    return result or str(value)


def _lookup(value: Any, names: Sequence[str]) -> tuple[int | None, str | None]:
    for name in names:
        candidate = value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
        if candidate is not None:
            try:
                return int(candidate), name
            except (TypeError, ValueError):
                return None, f"non_numeric_{name}"
    return None, None


def normalize_usage(usage: Any) -> dict[str, Any]:
    """Normalize one provider usage object without converting missing data to 0."""

    raw = _raw_usage(usage)
    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "usage_present": False,
            "usage_missing_reason": "provider_usage_missing",
            "total_tokens_source": None,
            "usage_consistency": "not_checkable",
            "usage_raw": None,
        }

    prompt, prompt_source = _lookup(usage, ("prompt_tokens", "input_tokens"))
    completion, completion_source = _lookup(usage, ("completion_tokens", "output_tokens"))
    total, total_source = _lookup(usage, ("total_tokens",))
    reasons: list[str] = []
    if prompt is None:
        reasons.append(prompt_source or "prompt_tokens_missing")
    if completion is None:
        reasons.append(completion_source or "completion_tokens_missing")

    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
        total_source_name = "derived_from_prompt_completion"
    elif total is not None:
        total_source_name = total_source or "provider"
    else:
        total_source_name = None
        reasons.append(total_source or "total_tokens_missing")

    consistency = "not_checkable"
    if total is not None and prompt is not None and completion is not None:
        consistency = "ok" if total == prompt + completion else "provider_total_mismatch"

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "usage_present": True,
        "usage_missing_reason": ";".join(dict.fromkeys(reasons)) or None,
        "total_tokens_source": total_source_name,
        "usage_consistency": consistency,
        "usage_raw": raw,
    }


def _call_usage(call: Any) -> dict[str, Any]:
    if isinstance(call, Mapping):
        normalized = call.get("usage_normalized")
        if isinstance(normalized, Mapping):
            return dict(normalized)
        return normalize_usage(call.get("usage"))
    return normalize_usage(getattr(call, "usage", None))


def aggregate_usage(calls: Iterable[Any]) -> dict[str, Any]:
    """Aggregate calls while keeping nulls and their reasons auditable."""

    call_list = list(calls)
    result: dict[str, Any] = {
        "call_count": len(call_list),
        "usage_status": "ok" if call_list else "not_checkable",
        "usage_missing_reasons": [],
    }
    for key in TOKEN_KEYS:
        known_values = []
        missing_reasons: list[str] = []
        for call in call_list:
            usage = _call_usage(call)
            value = usage.get(key)
            if isinstance(value, int):
                known_values.append(value)
            else:
                missing_reasons.append(str(usage.get("usage_missing_reason") or f"{key}_missing"))
        result[key] = sum(known_values) if call_list and len(known_values) == len(call_list) else None
        result[f"known_{key}"] = sum(known_values) if known_values else 0
        if missing_reasons:
            result["usage_status"] = "not_checkable"
            result["usage_missing_reasons"].extend(missing_reasons)

    result["usage_missing_reasons"] = list(dict.fromkeys(result["usage_missing_reasons"]))
    if not call_list:
        result["usage_missing_reasons"] = ["no_llm_calls"]
    return result


def _numeric_equal(left: Any, right: Any) -> bool:
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and left == right


def reconcile_token_layers(
    call_rows: Iterable[Any],
    step_rows: Iterable[Any],
    run_usage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Check call -> step -> run equality for every token dimension."""

    calls = list(call_rows)
    steps = list(step_rows)
    call_aggregate = aggregate_usage(calls)
    step_aggregate = aggregate_usage(
        {"usage_normalized": {key: step.get(key) for key in TOKEN_KEYS}}
        if isinstance(step, Mapping) and all(key in step for key in TOKEN_KEYS)
        else step
        for step in steps
    )
    run = dict(run_usage or {})
    checks: dict[str, dict[str, Any]] = {}
    for key in TOKEN_KEYS:
        call_total = call_aggregate.get(key)
        step_total = step_aggregate.get(key)
        run_total = run.get(key)
        if _numeric_equal(call_total, step_total) and _numeric_equal(step_total, run_total):
            checks[key] = {
                "status": "ok",
                "call_total": call_total,
                "step_total": step_total,
                "run_total": run_total,
                "delta_run_minus_step": 0,
                "delta_step_minus_call": 0,
            }
        else:
            reasons: list[str] = []
            if not all(isinstance(value, (int, float)) for value in (call_total, step_total, run_total)):
                reasons.append("one_or_more_layers_not_checkable")
            else:
                if call_total != step_total:
                    reasons.append("step_call_total_mismatch")
                if step_total != run_total:
                    reasons.append("run_step_total_mismatch")
            checks[key] = {
                "status": "not_checkable" if "one_or_more_layers_not_checkable" in reasons else "mismatch",
                "call_total": call_total,
                "step_total": step_total,
                "run_total": run_total,
                "delta_run_minus_step": run_total - step_total if isinstance(run_total, (int, float)) and isinstance(step_total, (int, float)) else None,
                "delta_step_minus_call": step_total - call_total if isinstance(step_total, (int, float)) and isinstance(call_total, (int, float)) else None,
                "reasons": reasons,
            }
    statuses = {item["status"] for item in checks.values()}
    return {
        "status": "ok" if statuses == {"ok"} else ("mismatch" if "mismatch" in statuses else "not_checkable"),
        "checks": checks,
        "call_aggregate": call_aggregate,
        "step_aggregate": step_aggregate,
    }


def _state_value(state: Any, name: str) -> Any:
    if isinstance(state, Mapping):
        return state.get(name)
    return getattr(state, name, None)


def _xml_text(state: Any) -> str | None:
    value = _state_value(state, "dom_tree")
    if value is None:
        for name in ("xml", "view_hierarchy", "a11y_tree"):
            candidate = _state_value(state, name)
            if candidate:
                value = candidate
                break
    if value is None and _state_value(state, "ui_elements") is not None:
        try:
            from study.benchmarks.androidworld.androidworld_native import _state_to_appagent_xml

            value = _state_to_appagent_xml(state)
        except Exception:
            value = None
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return text or None


def extract_xml_package_names(xml_text: str | bytes | None) -> list[str]:
    """Return package attributes observed in XML without choosing a foreground app."""

    if xml_text is None:
        return []
    if isinstance(xml_text, bytes):
        xml_text = xml_text.decode("utf-8", errors="replace")
    text = str(xml_text).strip()
    if not text:
        return []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    packages: list[str] = []
    for element in root.iter():
        package = str(element.attrib.get("package") or "").strip()
        if package and package not in packages:
            packages.append(package)
    return packages


def _explicit_state_text(state: Any, names: Sequence[str]) -> str | None:
    for name in names:
        value = _state_value(state, name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_state_artifacts(
    state: Any,
    root: str | Path,
    *,
    step_index: int,
    phase: str,
    capture_id: str | None = None,
) -> dict[str, Any]:
    """Persist PNG/XML from exactly one state object and return a manifest row."""

    capture_id = capture_id or f"step_{step_index:03d}_{phase}_{uuid.uuid4().hex[:8]}"
    capture_dir = Path(root) / f"step_{step_index:03d}" / f"{phase}_{capture_id[-8:]}"
    capture_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "capture_id": capture_id,
        "step_index": step_index,
        "phase": phase,
        "captured_at": utcnow_iso(),
        "status": "missing",
        "screenshot": None,
        "screenshot_sha256": None,
        "xml": None,
        "xml_sha256": None,
        "missing_reasons": [],
        "current_package": None,
        "current_package_missing_reason": None,
        "current_activity": None,
        "current_activity_missing_reason": None,
        "observed_package_names": [],
        "observed_package_names_missing_reason": None,
    }
    if state is None:
        result["missing_reasons"] = ["state_object_missing"]
        result["current_package_missing_reason"] = "state_object_missing"
        result["current_activity_missing_reason"] = "state_object_missing"
        result["observed_package_names_missing_reason"] = "state_object_missing"
        return result

    result["current_package"] = _explicit_state_text(
        state,
        ("current_package", "package_name", "foreground_package"),
    )
    if result["current_package"] is None:
        result["current_package_missing_reason"] = "state_object_did_not_expose_current_package"
    result["current_activity"] = _explicit_state_text(
        state,
        ("current_activity", "activity", "activity_name"),
    )
    if result["current_activity"] is None:
        result["current_activity_missing_reason"] = "state_object_did_not_expose_current_activity"

    pixels = _state_value(state, "pixels")
    if pixels is None:
        result["missing_reasons"].append("pixels_missing")
    else:
        screenshot_path = capture_dir / "screenshot.png"
        try:
            if hasattr(pixels, "save"):
                pixels.save(screenshot_path)
            else:
                from PIL import Image

                Image.fromarray(pixels).save(screenshot_path)
            result["screenshot"] = str(screenshot_path)
            result["screenshot_sha256"] = _sha256(screenshot_path)
        except Exception as exc:  # evidence capture must not change agent control flow
            result["missing_reasons"].append(f"screenshot_write_error:{type(exc).__name__}")

    xml_text = _xml_text(state)
    if xml_text is None:
        result["missing_reasons"].append("dom_tree_missing")
        result["observed_package_names_missing_reason"] = "dom_tree_missing"
    else:
        result["observed_package_names"] = extract_xml_package_names(xml_text)
        if not result["observed_package_names"]:
            result["observed_package_names_missing_reason"] = "xml_contains_no_package_attributes"
        xml_path = capture_dir / "view_hierarchy.xml"
        try:
            xml_path.write_text(xml_text, encoding="utf-8")
            result["xml"] = str(xml_path)
            result["xml_sha256"] = _sha256(xml_path)
        except Exception as exc:
            result["missing_reasons"].append(f"xml_write_error:{type(exc).__name__}")

    if result["screenshot"] and result["xml"]:
        result["status"] = "complete"
    elif result["screenshot"] or result["xml"]:
        result["status"] = "partial"
    result["missing_reasons"] = list(dict.fromkeys(result["missing_reasons"]))
    return result


def elapsed_seconds(started_perf: float) -> float:
    return time.perf_counter() - started_perf


__all__ = [
    "TOKEN_KEYS",
    "aggregate_usage",
    "capture_state_artifacts",
    "elapsed_seconds",
    "extract_xml_package_names",
    "normalize_usage",
    "reconcile_token_layers",
    "utcnow_iso",
]

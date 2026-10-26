"""Knowledge preparation helpers for AppAgent and mobile-use study modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET

from study.bootstrap import ensure_import_paths


ensure_import_paths()


@dataclass
class ActionableNode:
    uid: str
    center: tuple[int, int]
    bounds: tuple[int, int, int, int]
    attrib: dict[str, str]


def resolve_default_knowledge_path(
    *,
    mode: str,
    run_output_dir: str | Path,
    provided_path: str | None,
) -> str | None:
    if provided_path:
        return str(Path(provided_path).resolve())
    output_dir = Path(run_output_dir).resolve()
    if mode == "p2":
        return str(output_dir / "knowledge" / "appagent_docs")
    if mode == "p4":
        return str(output_dir / "knowledge" / "explored_knowledge.json")
    return None


def normalize_app_name(app_name: str | None) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (app_name or "").strip().lower())
    return normalized.strip("_") or "unknown_app"


def appagent_doc_filename(uid: str, *, max_stem_length: int = 96) -> str:
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", uid).strip(" .")
    stem = stem or "element"
    if len(stem) <= max_stem_length:
        return f"{stem}.txt"
    digest = hashlib.sha1(uid.encode("utf-8")).hexdigest()[:12]
    prefix_length = max(1, max_stem_length - len(digest) - 1)
    prefix = stem[:prefix_length].rstrip(" ._-") or "element"
    return f"{prefix}_{digest}.txt"


def appagent_doc_path(docs_dir: str | Path, uid: str) -> Path:
    return Path(docs_dir) / appagent_doc_filename(uid)


def _mobileuse_app_aliases(app_name: str, package_name: str | None = None) -> list[str]:
    aliases: list[str] = []

    def _add(value: str | None) -> None:
        cleaned = " ".join((value or "").split()).strip()
        if cleaned and cleaned not in aliases:
            aliases.append(cleaned)

    base = " ".join(app_name.split()).strip()
    lower = base.lower()
    title = base.title()
    _add(base)
    _add(lower)
    _add(title)
    if not lower.endswith(" app"):
        _add(f"{lower} app")
    if not title.endswith(" App"):
        _add(f"{title} app")
    _add(package_name)
    return aliases


def _load_find_package():
    from templates.packages import find_package

    return find_package


def find_package_name(app_name: str | None, package_name: str | None = None) -> str | None:
    if package_name:
        return package_name
    if not app_name:
        return None
    try:
        find_package = _load_find_package()
        return find_package(app_name)
    except Exception:
        return None


def _controller_launch_app(controller, app_name: str | None, package_name: str | None) -> None:
    resolved_package = find_package_name(app_name, package_name)
    if resolved_package and hasattr(controller, "launch_app"):
        controller.launch_app(resolved_package)
        return
    if resolved_package and hasattr(controller, "launch"):
        controller.launch(resolved_package)
        return
    if app_name and hasattr(controller, "launch_app"):
        controller.launch_app(app_name)
        return
    if app_name and hasattr(controller, "launch"):
        controller.launch(app_name)


def _parse_bounds(bounds: str | None) -> tuple[int, int, int, int] | None:
    if not bounds or "][" not in bounds:
        return None
    try:
        left_top, right_bottom = bounds[1:-1].split("][")
        x1, y1 = left_top.split(",")
        x2, y2 = right_bottom.split(",")
        return int(x1), int(y1), int(x2), int(y2)
    except (TypeError, ValueError):
        return None


def _center_from_bounds(bounds: str | None) -> tuple[int, int] | None:
    parsed = _parse_bounds(bounds)
    if parsed is None:
        return None
    x1, y1, x2, y2 = parsed
    return (x1 + x2) // 2, (y1 + y2) // 2


def _element_uid(elem: ET.Element) -> str:
    bounds = _parse_bounds(elem.attrib.get("bounds")) or (0, 0, 0, 0)
    x1, y1, x2, y2 = bounds
    elem_w, elem_h = x2 - x1, y2 - y1
    resource_id = elem.attrib.get("resource-id", "")
    if resource_id:
        elem_id = resource_id.replace(":", ".").replace("/", "_")
    else:
        elem_id = f"{elem.attrib.get('class', 'node')}_{elem_w}_{elem_h}"
    content_desc = elem.attrib.get("content-desc", "")
    if content_desc and len(content_desc) < 20:
        suffix = content_desc.replace("/", "_").replace(" ", "").replace(":", "_")
        elem_id += f"_{suffix}"
    return elem_id


def _xml_nodes_with_uids(xml_path: str | Path, *, actionable_only: bool) -> list[ActionableNode]:
    path: list[ET.Element] = []
    nodes: list[ActionableNode] = []
    for event, elem in ET.iterparse(str(xml_path), ["start", "end"]):
        if event == "start":
            path.append(elem)
            attrs = elem.attrib
            is_actionable = (
                attrs.get("enabled") == "true"
                and (
                    attrs.get("clickable") == "true"
                    or attrs.get("focusable") == "true"
                    or attrs.get("scrollable") == "true"
                    or attrs.get("long-clickable") == "true"
                    or attrs.get("class", "").endswith(("EditText", "Button", "ImageButton"))
                )
            )
            if actionable_only and not is_actionable:
                continue
            bounds = _parse_bounds(attrs.get("bounds"))
            center = _center_from_bounds(attrs.get("bounds"))
            if bounds is None or center is None:
                continue
            uid = _element_uid(elem)
            if len(path) > 1:
                uid = _element_uid(path[-2]) + "_" + uid
            uid += f"_{attrs.get('index', '0')}"
            nodes.append(
                ActionableNode(
                    uid=uid,
                    center=center,
                    bounds=bounds,
                    attrib=dict(attrs),
                )
            )
        else:
            path.pop()
    return nodes


def _label_from_attrib(attrs: dict[str, str]) -> str:
    for key in ("text", "content-desc", "hint-text", "resource-id"):
        value = " ".join((attrs.get(key) or "").split()).strip()
        if not value:
            continue
        if key == "resource-id":
            value = value.split("/")[-1].split(":")[-1].replace("_", " ")
        return value
    class_name = attrs.get("class", "").split(".")[-1]
    return class_name or "this control"


def _meaningful_labels(xml_path: str | Path, limit: int = 8) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for node in _xml_nodes_with_uids(xml_path, actionable_only=False):
        label = _label_from_attrib(node.attrib)
        normalized = label.lower()
        if not label or normalized in seen:
            continue
        if normalized in {"viewgroup", "frame layout", "linearlayout", "textview"}:
            continue
        if len(label) > 60:
            continue
        seen.add(normalized)
        labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def _input_hints(xml_path: str | Path, limit: int = 6) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()
    for node in _xml_nodes_with_uids(xml_path, actionable_only=True):
        class_name = node.attrib.get("class", "")
        if not class_name.endswith("EditText") and node.attrib.get("focusable") != "true":
            continue
        hint = _label_from_attrib(node.attrib)
        normalized = hint.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        hints.append(hint)
        if len(hints) >= limit:
            break
    return hints


def summarize_xml_screen(app_name: str, xml_path: str | Path) -> str:
    labels = _meaningful_labels(xml_path, limit=8)
    hints = _input_hints(xml_path, limit=5)
    parts = [f"{app_name} app page"]
    if labels:
        parts.append("visible controls: " + ", ".join(labels))
    if hints:
        parts.append("editable fields: " + ", ".join(hints))
    return ". ".join(parts) + "."


def _appagent_doc_for_node(node: ActionableNode) -> dict[str, str]:
    attrs = node.attrib
    label = _label_from_attrib(attrs)
    label_lower = label.lower()
    hint = " ".join((attrs.get("hint-text") or "").split()).strip()
    resource_id = (attrs.get("resource-id") or "").split("/")[-1].replace("_", " ").strip()
    class_name = attrs.get("class", "").split(".")[-1]
    doc = {
        "tap": "",
        "text": "",
        "v_swipe": "",
        "h_swipe": "",
        "long_press": "",
    }
    if attrs.get("clickable") == "true" or attrs.get("focusable") == "true":
        if any(token in label_lower for token in ("add", "create", "new", "+")):
            doc["tap"] = "Tap to create or add a new item."
        elif any(token in label_lower for token in ("save", "done", "ok", "start", "set", "allow", "skip")):
            doc["tap"] = f"Tap to confirm or continue with {label}."
        elif any(token in label_lower for token in ("alarm", "timer", "stopwatch", "clock", "contacts", "phone", "favorites", "recents")):
            doc["tap"] = f"Tap to open the {label} section."
        else:
            doc["tap"] = f"Tap to open, select, or activate {label}."
    if attrs.get("class", "").endswith("EditText") or hint:
        target = hint or resource_id or label
        doc["text"] = f"Use this field to enter {target}."
    if attrs.get("long-clickable") == "true":
        doc["long_press"] = f"Long press for additional options related to {label}."
    if attrs.get("scrollable") == "true":
        if "horizontal" in class_name.lower() or "viewpager" in class_name.lower():
            doc["h_swipe"] = f"Swipe horizontally on {label} to browse more content."
        else:
            doc["v_swipe"] = f"Swipe vertically on {label} to browse more content."
    return doc


def document_xml_for_appagent(xml_path: str | Path, docs_dir: str | Path) -> int:
    docs_root = Path(docs_dir)
    docs_root.mkdir(parents=True, exist_ok=True)
    generated = 0
    for node in _xml_nodes_with_uids(xml_path, actionable_only=True):
        doc_path = appagent_doc_path(docs_root, node.uid)
        if doc_path.exists():
            continue
        doc_content = _appagent_doc_for_node(node)
        if not any(doc_content.values()):
            continue
        doc_path.write_text(str(doc_content), encoding="utf-8")
        generated += 1
    return generated


def _candidate_priority(label: str, app_name: str) -> int:
    label_lower = label.lower()
    app_lower = app_name.lower()
    priority = 0
    if any(token in label_lower for token in ("alarm", "timer", "stopwatch", "contacts", "phone", "favorites", "recents")):
        priority += 4
    if any(token in label_lower for token in ("search", "clock", "contact")):
        priority += 2
    if app_lower in label_lower:
        priority += 2
    if any(token in label_lower for token in ("add", "create", "new", "+")):
        priority += 1
    return priority


def _navigation_candidates(xml_path: str | Path, app_name: str, *, max_candidates: int = 3) -> list[ActionableNode]:
    nodes = _xml_nodes_with_uids(xml_path, actionable_only=True)
    screen_bottom = max((node.bounds[3] for node in nodes), default=0)
    bottom_threshold = int(screen_bottom * 0.55) if screen_bottom else 0
    ranked: list[tuple[int, ActionableNode]] = []
    seen: set[str] = set()
    for node in nodes:
        label = _label_from_attrib(node.attrib)
        normalized = label.lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        priority = _candidate_priority(label, app_name)
        if priority <= 0 and node.center[1] < bottom_threshold:
            continue
        ranked.append((priority, node))
    ranked.sort(key=lambda item: (-item[0], item[1].center[1], item[1].center[0]))
    return [node for _, node in ranked[:max_candidates]]


def _screen_signature(xml_path: str | Path) -> str:
    labels = _meaningful_labels(xml_path, limit=12)
    return "|".join(labels)


def _capture_page(controller, artifact_dir: Path, prefix: str) -> tuple[str, str] | tuple[None, None]:
    screenshot_path = controller.get_screenshot(prefix, artifact_dir)
    xml_path = controller.get_xml(prefix, artifact_dir)
    if screenshot_path == "ERROR" or xml_path == "ERROR":
        return None, None
    return str(screenshot_path), str(xml_path)


def _safe_back_home(controller) -> None:
    try:
        controller.back()
        time.sleep(1.0)
    except Exception:
        pass
    try:
        controller.home()
        time.sleep(1.0)
    except Exception:
        pass


def proactively_explore_app(
    *,
    controller,
    app_name: str,
    package_name: str | None,
    artifact_dir: str | Path,
    max_pages: int = 4,
) -> list[dict[str, str]]:
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, str]] = []
    seen_signatures: set[str] = set()

    def _record_current(prefix: str, source_action: dict[str, object] | None = None) -> str | None:
        screenshot_path, xml_path = _capture_page(controller, artifact_root, prefix)
        if xml_path is None:
            return None
        signature = _screen_signature(xml_path)
        if signature in seen_signatures:
            return signature
        seen_signatures.add(signature)
        pages.append(
            {
                "xml_path": xml_path,
                "screenshot_path": screenshot_path or "",
                "summary": summarize_xml_screen(app_name, xml_path),
                "source_action": source_action or {"type": "launch_app", "app_name": app_name},
            }
        )
        return signature

    _controller_launch_app(controller, app_name, package_name)
    time.sleep(2.0)
    base_signature = _record_current("page_0")
    if base_signature is None:
        return pages

    base_xml = pages[0]["xml_path"]
    candidates = _navigation_candidates(base_xml, app_name, max_candidates=max_pages - 1)
    for index, node in enumerate(candidates, start=1):
        _controller_launch_app(controller, app_name, package_name)
        time.sleep(1.5)
        controller.tap(node.center[0], node.center[1])
        time.sleep(1.5)
        _record_current(
            f"page_{index}",
            {
                "type": "tap_navigation_candidate",
                "app_name": app_name,
                "coordinate": [node.center[0], node.center[1]],
                "bounds": list(node.bounds),
                "uid": node.uid,
                "label": _label_from_attrib(node.attrib),
                "attributes": dict(node.attrib),
            },
        )
        if len(pages) >= max_pages:
            break

    _safe_back_home(controller)
    return pages


def ensure_mobileuse_explored_knowledge(
    *,
    controller,
    app_names: list[str],
    knowledge_path: str | Path,
    package_names: dict[str, str] | None = None,
    artifact_root: str | Path,
    max_pages_per_app: int = 4,
) -> str:
    knowledge_file = Path(knowledge_path)
    knowledge_file.parent.mkdir(parents=True, exist_ok=True)
    if knowledge_file.exists():
        try:
            payload = json.loads(knowledge_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}

    package_names = package_names or {}
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for app_name in app_names:
        aliases = _mobileuse_app_aliases(app_name, package_names.get(app_name))
        if any(payload.get(alias) for alias in aliases):
            manifest_rows.append(
                {
                    "app_name": app_name,
                    "package_name": package_names.get(app_name),
                    "aliases": aliases,
                    "skipped": True,
                    "reason": "knowledge_already_present",
                    "knowledge_path": str(knowledge_file),
                    "artifact_dir": str(artifact_root / normalize_app_name(app_name)),
                }
            )
            continue
        pages = proactively_explore_app(
            controller=controller,
            app_name=app_name,
            package_name=package_names.get(app_name),
            artifact_dir=artifact_root / normalize_app_name(app_name),
            max_pages=max_pages_per_app,
        )
        summaries = [page["summary"] for page in pages if page.get("summary")]
        if summaries:
            for alias in aliases:
                payload[alias] = summaries
        manifest_rows.append(
            {
                "app_name": app_name,
                "package_name": package_names.get(app_name),
                "aliases": aliases,
                "skipped": False,
                "knowledge_path": str(knowledge_file),
                "artifact_dir": str(artifact_root / normalize_app_name(app_name)),
                "pages": pages,
                "summaries_written": summaries,
                "aliases_written": aliases if summaries else [],
            }
        )

    knowledge_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_root / "proactive_exploration_manifest.json").write_text(
        json.dumps(
            {
                "knowledge_path": str(knowledge_file),
                "app_names": app_names,
                "max_pages_per_app": max_pages_per_app,
                "records": manifest_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(knowledge_file)


def ensure_appagent_knowledge(
    *,
    controller,
    app_name: str,
    package_name: str | None,
    knowledge_root: str | Path,
    artifact_root: str | Path,
    max_pages: int = 4,
) -> str:
    root = Path(knowledge_root)
    app_root = root / normalize_app_name(app_name)
    docs_dir = app_root / "auto_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    pages = proactively_explore_app(
        controller=controller,
        app_name=app_name,
        package_name=package_name,
        artifact_dir=Path(artifact_root) / normalize_app_name(app_name),
        max_pages=max_pages,
    )
    for page in pages:
        xml_path = page.get("xml_path")
        if xml_path:
            document_xml_for_appagent(xml_path, docs_dir)
    return str(docs_dir)

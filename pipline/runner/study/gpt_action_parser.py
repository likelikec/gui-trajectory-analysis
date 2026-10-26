"""Standalone GPT action parser for fixed-size Android experiments.

The parser is deliberately model-boundary code.  It accepts the harmless
surface variations commonly produced by GPT (tool-call wrappers, fenced JSON,
plain JSON, and function-like actions) and emits one canonical action.  Touch
coordinates remain absolute pixels on the fixed 1080x2400 screen: this module
never normalizes, scales, clamps, or guesses a missing coordinate.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


GPT_PHONE_WIDTH = 1080
GPT_PHONE_HEIGHT = 2400

_ACTION_ALIASES = {
    "tap": "click",
    "left_click": "click",
    "press": "click",
    "input_text": "type",
    "write": "type",
    "open_app": "open",
    "launch_app": "open",
    "done": "terminate",
    "finish": "terminate",
    "finished": "terminate",
}

_ACTIONS = {
    "key",
    "click",
    "long_press",
    "swipe",
    "scroll",
    "type",
    "clear_text",
    "answer",
    "system_button",
    "open",
    "wait",
    "take_note",
    "terminate",
}

_TOUCH_FIELDS = {
    "click": ("coordinate",),
    "long_press": ("coordinate",),
    "swipe": ("coordinate", "coordinate2"),
    "scroll": ("coordinate", "coordinate2"),
}

_TEXT_ACTIONS = {"key", "type", "answer", "open", "take_note"}


class GPTActionParseError(ValueError):
    """Raised when a GPT response cannot be converted without guessing."""


@dataclass(frozen=True)
class ParsedGPTAction:
    action: dict[str, Any]
    thought: str | None = None
    description: str | None = None
    source: str = "unknown"
    candidate_count: int = 1


def _repair_incomplete_json_object(fragment: str) -> dict[str, Any] | None:
    """Close only deterministic JSON delimiters; never invent values."""
    opening_to_closing = {"{": "}", "[": "]"}
    stack: list[str] = []
    repaired: list[str] = []
    in_string = False
    escaped = False

    for character in str(fragment or ""):
        if in_string:
            repaired.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            repaired.append(character)
        elif character in opening_to_closing:
            stack.append(character)
            repaired.append(character)
        elif character in opening_to_closing.values():
            if not stack:
                break
            matching_depth = next(
                (
                    depth
                    for depth in range(len(stack) - 1, -1, -1)
                    if opening_to_closing[stack[depth]] == character
                ),
                None,
            )
            if matching_depth is None:
                return None
            while len(stack) - 1 > matching_depth:
                repaired.append(opening_to_closing[stack.pop()])
            repaired.append(character)
            stack.pop()
            if not stack:
                break
        else:
            repaired.append(character)

    if in_string or not repaired or repaired[0] != "{":
        return None
    while stack:
        repaired.append(opening_to_closing[stack.pop()])
    try:
        value = json.loads("".join(repaired))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _parse_mapping(text: str) -> dict[str, Any] | None:
    value = str(text or "").strip()
    if not value:
        return None
    value = re.sub(r"^```(?:json|python)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    return dict(parsed) if isinstance(parsed, Mapping) else None


def _balanced_objects(text: str) -> list[tuple[int, int, str]]:
    """Return balanced JSON-looking objects while respecting quoted braces."""
    objects: list[tuple[int, int, str]] = []
    start: int | None = None
    depth = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append((start, index + 1, text[start : index + 1]))
                start = None
    return objects


def _unwrap_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    current: Mapping[str, Any] = payload
    function = current.get("function")
    if isinstance(function, Mapping):
        current = function

    arguments = current.get("arguments", current.get("parameters"))
    if isinstance(arguments, str):
        arguments = _parse_mapping(arguments)
    if isinstance(arguments, Mapping):
        # Only unwrap a genuine action/tool envelope, not an arbitrary object
        # that happens to contain an ``arguments`` field.
        name = str(current.get("name") or current.get("tool") or "").lower()
        if name in {"mobile_use", "computer", "computer_use", ""}:
            current = arguments

    action = current.get("action")
    if isinstance(action, Mapping):
        nested_name = action.get("name") or action.get("action")
        nested_args = action.get("arguments") or action.get("parameters") or {}
        if isinstance(nested_args, Mapping) and isinstance(nested_name, str):
            current = {"action": nested_name, **dict(nested_args)}

    return dict(current) if isinstance(current.get("action"), str) else None


def extract_gpt_action_candidates(content: str) -> list[dict[str, Any]]:
    """Extract action-shaped mappings in their response order.

    Duplicate representations of the same object (for example a JSON object
    found both inside a tool-call block and by the balanced-object scanner) are
    removed.  If GPT echoed examples before its final answer, callers can use
    the final candidate; renderers always emit only that one action.
    """
    text = str(content or "")
    located: list[tuple[int, dict[str, Any]]] = []

    for match in re.finditer(
        r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL | re.IGNORECASE
    ):
        if (mapping := _parse_mapping(match.group(1))) is not None:
            if _unwrap_payload(mapping) is not None:
                located.append((match.start(), mapping))

    for match in re.finditer(r"```(?:json|python)?\s*(.*?)```", text, flags=re.S | re.I):
        if (mapping := _parse_mapping(match.group(1))) is not None:
            if _unwrap_payload(mapping) is not None:
                located.append((match.start(), mapping))

    for start, _end, raw_object in _balanced_objects(text):
        if (mapping := _parse_mapping(raw_object)) is not None:
            if _unwrap_payload(mapping) is not None:
                located.append((start, mapping))

    # The provider occasionally closes an object with ``}`` while leaving a
    # coordinate array open.  This repair only inserts unambiguous delimiters;
    # it never fills a missing action, coordinate, or text value.
    for start, character in enumerate(text):
        if character != "{":
            continue
        if (mapping := _repair_incomplete_json_object(text[start:])) is not None:
            if _unwrap_payload(mapping) is not None:
                located.append((start, mapping))

    located.sort(key=lambda item: item[0])
    unique: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for _position, mapping in located:
        # A tool envelope and its nested ``arguments`` object are two JSON
        # mappings but one device action.  Fingerprint the unwrapped action so
        # that a valid wrapper is not rejected as "multiple actions".
        action_mapping = _unwrap_payload(mapping) or mapping
        fingerprint = json.dumps(
            action_mapping, ensure_ascii=False, sort_keys=True, default=str
        )
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        unique.append(mapping)
    return unique


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError) as exc:
        raise GPTActionParseError("function_argument_must_be_literal") from exc


def _function_action(content: str) -> dict[str, Any] | None:
    matches = list(
        re.finditer(
            r"(?im)(?:^|\bAction\s*:\s*)([a-zA-Z_][\w]*)\s*\(([^\n]*)\)\s*$",
            str(content or "").strip(),
        )
    )
    if not matches:
        return None
    expression = f"{matches[-1].group(1)}({matches[-1].group(2)})"
    try:
        call = ast.parse(expression, mode="eval").body
    except SyntaxError as exc:
        raise GPTActionParseError("function_action_parse_error") from exc
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None
    name = call.func.id.lower()
    values = [_literal(argument) for argument in call.args]
    result: dict[str, Any] = {"action": name}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise GPTActionParseError("function_kwargs_unpack_not_allowed")
        result[keyword.arg] = _literal(keyword.value)

    def _point_from(values_: list[Any], offset: int = 0) -> list[Any] | None:
        if len(values_) > offset and isinstance(values_[offset], (list, tuple)):
            return list(values_[offset])
        if len(values_) >= offset + 2:
            return [values_[offset], values_[offset + 1]]
        return None

    if name in {"click", "tap", "left_click", "press", "long_press"}:
        result.setdefault("coordinate", _point_from(values))
        if name == "long_press" and len(values) >= 3:
            result.setdefault("time", values[2])
    elif name in {"swipe", "scroll"}:
        if len(values) >= 2 and all(isinstance(value, (list, tuple)) for value in values[:2]):
            result.setdefault("coordinate", list(values[0]))
            result.setdefault("coordinate2", list(values[1]))
        elif len(values) >= 4:
            result.setdefault("coordinate", values[:2])
            result.setdefault("coordinate2", values[2:4])
    elif name in _TEXT_ACTIONS | {"open_app", "input_text", "write"} and values:
        result.setdefault("text", values[0])
    elif name == "system_button" and values:
        result.setdefault("button", values[0])
    elif name == "wait" and values:
        result.setdefault("time", values[0])
    elif name in {"terminate", "done", "finish", "finished"} and values:
        result.setdefault("status", values[0])
    return result


def _extract_text_parts(content: str) -> tuple[str | None, str | None]:
    text = str(content or "")
    thought_patterns = (
        r"<thinking>(.*?)</thinking>",
        r"###\s*Thought\s*###(.*?)(?=###\s*Action\s*###|\Z)",
        r"Thought\s*:\s*(.*?)(?=\n\s*Action\s*:|<tool_call>|\Z)",
    )
    description_patterns = (
        r"<conclusion>(.*?)</conclusion>",
        r"###\s*Description\s*###(.*?)(?=\Z)",
        r"Action\s*:\s*(.*?)(?=<tool_call>|\Z)",
    )

    def _first(patterns: tuple[str, ...]) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.S | re.I)
            if match and (value := match.group(1).strip()):
                # Do not treat a JSON action as its human description.
                if not value.startswith("{"):
                    return value
        return None

    return _first(thought_patterns), _first(description_patterns)


def _point(value: Any, field: str, width: int, height: int) -> list[int | float]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise GPTActionParseError(f"{field}_must_be_two_numbers") from exc
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise GPTActionParseError(f"{field}_must_be_two_numbers")
    point = list(value)
    for index, (component, limit) in enumerate(zip(point, (width, height), strict=True)):
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise GPTActionParseError(f"{field}[{index}]_must_be_numeric")
        if not math.isfinite(float(component)):
            raise GPTActionParseError(f"{field}[{index}]_must_be_finite")
        if float(component) < 0 or float(component) >= limit:
            raise GPTActionParseError(
                f"{field}[{index}]={component}_outside_0_{limit - 1}"
            )
    return point


def _canonical_action(
    payload: Mapping[str, Any], width: int, height: int
) -> dict[str, Any]:
    unwrapped = _unwrap_payload(payload)
    if unwrapped is None:
        raise GPTActionParseError("action_not_found")
    result = dict(unwrapped)
    raw_name = str(result.get("action") or "").strip().lower()

    if raw_name in {"back", "home", "menu", "enter"}:
        result = {"action": "system_button", "button": raw_name.title()}
        raw_name = "system_button"
    name = _ACTION_ALIASES.get(raw_name, raw_name)
    if name not in _ACTIONS:
        raise GPTActionParseError(f"unsupported_action_{raw_name or 'empty'}")
    result["action"] = name

    field_aliases = {
        "point": "coordinate",
        "start_point": "coordinate",
        "start": "coordinate",
        "end_point": "coordinate2",
        "end": "coordinate2",
        "content": "text",
    }
    for old, new in field_aliases.items():
        if new not in result and old in result:
            result[new] = result.pop(old)
    if "coordinate" not in result and "x" in result and "y" in result:
        result["coordinate"] = [result.pop("x"), result.pop("y")]

    if name == "terminate":
        result.setdefault("status", "success")
    for field in _TOUCH_FIELDS.get(name, ()):
        if field not in result or result[field] is None:
            raise GPTActionParseError(f"{name}_missing_{field}")
        result[field] = _point(result[field], field, width, height)
    for field in ("coordinate", "coordinate2"):
        if field in result and field not in _TOUCH_FIELDS.get(name, ()):
            result[field] = _point(result[field], field, width, height)

    if name in _TEXT_ACTIONS and not isinstance(result.get("text"), str):
        raise GPTActionParseError(f"{name}_missing_text")
    if name == "open":
        app_name = str(result.get("text") or "").strip()
        if not app_name:
            raise GPTActionParseError("open_missing_app_name")
        # ``open`` is an app-launch action in AndroidWorld.  Treating a media
        # filename as an app name leads to ``monkey -p <filename>`` failures.
        if re.search(r"\.(?:png|jpe?g|webp|gif|mp4|mov|pdf|txt|csv|json)$", app_name, re.I):
            raise GPTActionParseError("open_requires_installed_app_name")
        result["text"] = app_name
    if name == "system_button":
        button = str(result.get("button") or "").title()
        if button not in {"Back", "Home", "Menu", "Enter"}:
            raise GPTActionParseError("system_button_invalid_button")
        result["button"] = button
    if name == "terminate" and result.get("status") not in {"success", "failure"}:
        raise GPTActionParseError("terminate_invalid_status")
    if name == "wait" and "time" in result:
        seconds = result["time"]
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise GPTActionParseError("wait_time_must_be_numeric")
    return result


def parse_gpt_action(
    content: str,
    *,
    width: int = GPT_PHONE_WIDTH,
    height: int = GPT_PHONE_HEIGHT,
    reject_multiple: bool = False,
) -> ParsedGPTAction:
    if (width, height) != (GPT_PHONE_WIDTH, GPT_PHONE_HEIGHT):
        raise GPTActionParseError("unexpected_screen_size")
    candidates = extract_gpt_action_candidates(content)
    source = "json"
    if candidates:
        if reject_multiple and len(candidates) > 1:
            raise GPTActionParseError("multiple_actions_in_response")
        payload = candidates[-1]
    else:
        payload = _function_action(content)
        source = "function"
    if payload is None:
        raise GPTActionParseError("action_not_found")
    thought, description = _extract_text_parts(content)
    return ParsedGPTAction(
        action=_canonical_action(payload, width, height),
        thought=thought,
        description=description,
        source=source,
        candidate_count=max(1, len(candidates)),
    )


def render_mobile_use_action(parsed: ParsedGPTAction) -> str:
    """Render the one protocol shape consumed by P1/P4 mobile-use agents."""
    lines: list[str] = []
    if parsed.thought:
        lines.append(f"<thinking>{parsed.thought}</thinking>")
    if parsed.description:
        lines.append(f"<conclusion>{parsed.description}</conclusion>")
    envelope = {"name": "mobile_use", "arguments": parsed.action}
    lines.append(
        "<tool_call>"
        + json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
        + "</tool_call>"
    )
    return "\n".join(lines)


def render_mobileagent_v3_action(parsed: ParsedGPTAction) -> str:
    """Render the flat action sections consumed by P3 Mobile-Agent-v3."""
    action = dict(parsed.action)
    if action["action"] == "open":
        action["action"] = "open_app"
    elif action["action"] == "terminate":
        action["action"] = "done" if action.get("status") == "success" else "terminate"
        action.pop("status", None)
    thought = parsed.thought or "Choose the next valid device action."
    description = parsed.description or f"Perform {action['action']}."
    return (
        "### Thought ###\n"
        f"{thought}\n\n"
        "### Action ###\n"
        f"{json.dumps(action, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "### Description ###\n"
        f"{description}"
    )


def normalize_gpt_action_response(
    content: str,
    *,
    target: str,
    reject_multiple: bool = False,
) -> tuple[str, ParsedGPTAction]:
    parsed = parse_gpt_action(content, reject_multiple=reject_multiple)
    if target == "mobile_use":
        return render_mobile_use_action(parsed), parsed
    if target == "mobileagent_v3":
        return render_mobileagent_v3_action(parsed), parsed
    raise ValueError(f"unsupported_gpt_action_target_{target}")

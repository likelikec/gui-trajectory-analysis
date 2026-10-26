"""Standalone Claude adapter for fixed-screen Android experiments.

The Poixe endpoint exposes Claude through an OpenAI-compatible Chat
Completions API, but Claude may answer with Anthropic computer-use actions,
plain JSON, function-call text, or the repository's XML tool-call format.
This module owns that protocol boundary and emits the existing Android action
schema without depending on any GPT adapter or parser.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import re
import socket
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from study.model_adapters import ModelAdapter


CLAUDE_MODEL_CANONICAL = "cli2api/claude-sonnet-5"
CLAUDE_PHONE_WIDTH = 1080
CLAUDE_PHONE_HEIGHT = 2400
CLAUDE_ANDROID_SCREEN_SIZE = (CLAUDE_PHONE_WIDTH, CLAUDE_PHONE_HEIGHT)
# Poixe's Claude vision route grounds actions on this resized canvas. The
# adapter converts those points back to the Android device's absolute pixels.
CLAUDE_VISION_WIDTH = 704
CLAUDE_VISION_HEIGHT = 1568
CLAUDE_ABSOLUTE_CONTRACT_TAG = "[Claude fixed absolute-pixel Android contract]"
CLAUDE_ACTION_PROTOCOL_ERROR = "CLAUDE_ACTION_PROTOCOL_ERROR"

_ACTION_ALIASES = {
    "tap": "click",
    "touch": "click",
    "press": "click",
    "left_click": "click",
    "double_click": "click",
    "double_tap": "click",
    "long_click": "long_press",
    "press_and_hold": "long_press",
    "click_and_hold": "long_press",
    "drag": "swipe",
    "left_click_drag": "swipe",
    "scroll": "swipe",
    "type_text": "type",
    "input_text": "type",
    "enter_text": "type",
    "erase": "clear_text",
    "clear": "clear_text",
    "open_app": "open",
    "launch_app": "open",
    "press_key": "key",
    "key_press": "key",
    "done": "terminate",
    "finish": "terminate",
    "finished": "terminate",
    "fail": "terminate",
    "screenshot": "wait",
}

_POINT_KEYS = (
    "coordinate",
    "coordinates",
    "point",
    "position",
    "location",
    "start_coordinate",
    "start_point",
    "from",
)
_END_POINT_KEYS = (
    "coordinate2",
    "end_coordinate",
    "end_point",
    "to",
    "destination",
)

_DNS_FALLBACK_LOCK = threading.RLock()
_DNS_FALLBACK_CACHE: dict[str, str] = {}


@dataclass(frozen=True)
class ClaudeActionValidation:
    ok: bool
    reason: str | None = None
    action: dict[str, Any] | None = None
    coordinate_space: str | None = None


class ClaudeActionProtocolError(ValueError):
    """Raised when a Claude response cannot be mapped to one Android action."""


def _fallback_ipv4(hostname: str) -> str | None:
    if hostname in _DNS_FALLBACK_CACHE:
        return _DNS_FALLBACK_CACHE[hostname]
    try:
        socket.getaddrinfo(hostname, 443, socket.AF_INET, socket.SOCK_STREAM)
        return None
    except socket.gaierror:
        pass
    try:
        result = subprocess.run(
            ["dig", "+short", "A", hostname],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in result.stdout.splitlines():
        candidate = line.strip()
        try:
            socket.inet_aton(candidate)
        except OSError:
            continue
        _DNS_FALLBACK_CACHE[hostname] = candidate
        return candidate
    return None


@contextmanager
def request_network_context(base_url: Any):
    """Use a request-local IPv4 DNS fallback when macOS returns SERVFAIL."""
    hostname = urlparse(str(base_url)).hostname
    if not hostname:
        yield
        return
    ipv4 = _fallback_ipv4(hostname)
    if not ipv4:
        yield
        return

    with _DNS_FALLBACK_LOCK:
        original_getaddrinfo = socket.getaddrinfo

        def mapped_getaddrinfo(host: Any, *args: Any, **kwargs: Any):
            target = ipv4 if str(host).lower() == hostname.lower() else host
            return original_getaddrinfo(target, *args, **kwargs)

        socket.getaddrinfo = mapped_getaddrinfo
        try:
            yield
        finally:
            socket.getaddrinfo = original_getaddrinfo


def absolute_coordinate_contract(
    width: int = CLAUDE_PHONE_WIDTH,
    height: int = CLAUDE_PHONE_HEIGHT,
) -> str:
    if width <= 0 or height <= 0:
        raise ValueError("screen dimensions must be positive")
    x_ticks = "/".join(str(round(width * index / 10)) for index in range(10))
    y_ticks = "/".join(str(round(height * index / 10)) for index in range(10))
    return f"""
{CLAUDE_ABSOLUTE_CONTRACT_TAG}
The supplied image is the complete original {width}x{height} Android screen.
For click, long_press, and swipe, return original-screen absolute pixels:
x is 0..{width - 1} and y is 0..{height - 1}. Never return normalized
coordinates and never return coordinates from an internally resized image.

Use this 10% ruler only for estimation. From 0% through 90%, x is
{x_ticks}; y is {y_ticks}. The far-right and bottom pixels are
({width - 1}, {height - 1}). Locate the exact center of the visible target,
estimate x and y independently, and map both axes back to the original screen.

Return exactly one Android action. The canonical action names are click,
long_press, swipe, type, clear_text, key, system_button, open, wait, answer,
take_note, and terminate. A swipe requires coordinate and coordinate2.
""".strip()


def _message_text(messages: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
    return "\n".join(parts)


def _has_image(messages: Sequence[Mapping[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping):
                continue
            if "image" in str(item.get("type") or "").lower() or "image_url" in item:
                return True
    return False


def is_android_action_request(messages: Sequence[Mapping[str, Any]]) -> bool:
    text = _message_text(messages).lower()
    return _has_image(messages) and any(
        marker in text
        for marker in ("<tool_call>", "mobile_use", "action:", "function call")
    )


def _append_text(content: Any, text: str) -> Any:
    if isinstance(content, str):
        return f"{content.rstrip()}\n\n{text}"
    if isinstance(content, list):
        content.append({"type": "text", "text": text})
        return content
    return text


def augment_messages_with_absolute_contract(
    messages: Sequence[Mapping[str, Any]],
    *,
    width: int = CLAUDE_PHONE_WIDTH,
    height: int = CLAUDE_PHONE_HEIGHT,
) -> list[dict[str, Any]]:
    prepared = copy.deepcopy(list(messages))
    if not is_android_action_request(prepared):
        return prepared
    if CLAUDE_ABSOLUTE_CONTRACT_TAG in _message_text(prepared):
        return prepared
    contract = absolute_coordinate_contract(width, height)
    for message in prepared:
        if str(message.get("role") or "").lower() == "system":
            message["content"] = _append_text(message.get("content"), contract)
            return prepared
    for message in reversed(prepared):
        if str(message.get("role") or "").lower() == "user":
            message["content"] = _append_text(message.get("content"), contract)
            return prepared
    return prepared


def _strip_incompatible_image_fields(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item.pop("max_pixels", None)
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                image_url.pop("max_pixels", None)


def _json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return None
    cleaned = value.strip().replace("```json", "").replace("```", "").strip()
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(cleaned)
        except (SyntaxError, ValueError):
            return None
    return dict(parsed) if isinstance(parsed, Mapping) else None


def _json_candidates(content: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    snippets = re.findall(
        r"<tool_call>\s*(.*?)\s*</tool_call>|```(?:json)?\s*(.*?)\s*```",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for groups in snippets:
        for snippet in groups:
            if snippet and (parsed := _json_object(snippet)) is not None:
                candidates.append(parsed)

    decoder = json.JSONDecoder()
    for index, char in enumerate(content):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            candidates.append(dict(parsed))

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        marker = json.dumps(candidate, ensure_ascii=False, sort_keys=True, default=str)
        if marker not in seen:
            seen.add(marker)
            unique.append(candidate)
    return unique


def _normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _unwrap_payload(payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    value = dict(payload)
    function = value.get("function")
    if isinstance(function, Mapping):
        value = dict(function)

    name = _normalize_name(value.get("name") or value.get("tool") or value.get("action"))
    arguments = value.get("arguments")
    if arguments is None:
        arguments = value.get("input")
    if arguments is None:
        arguments = value.get("parameters")
    if isinstance(arguments, str):
        arguments = _json_object(arguments)
    args = dict(arguments) if isinstance(arguments, Mapping) else {}

    if name in {"mobile_use", "computer", "computer_use", "android", "android_use"}:
        action = _normalize_name(args.pop("action", args.pop("name", "")))
        return (action, args) if action else None
    if name:
        for key, item in value.items():
            if key not in {"name", "tool", "action", "type", "id", "function", "arguments", "input", "parameters"}:
                args.setdefault(key, item)
        return name, args
    return None


def _literal(node: ast.AST) -> Any:
    return ast.literal_eval(node)


def _function_call(content: str) -> tuple[str, list[Any], dict[str, Any]] | None:
    action_match = re.search(
        r"(?im)^\s*(?:#{1,4}\s*)?(?:\*\*)?action(?:\*\*)?\s*:\s*(.+)$",
        content,
    )
    candidates = [action_match.group(1).strip()] if action_match else []
    candidates.extend(
        match.group(0)
        for match in re.finditer(
            r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*\([^\n]*\)",
            content,
        )
    )
    for candidate in candidates:
        candidate = candidate.strip().strip("`")
        try:
            expression = ast.parse(candidate, mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
            continue
        try:
            args = [_literal(node) for node in expression.args]
            kwargs = {item.arg: _literal(item.value) for item in expression.keywords if item.arg}
        except (ValueError, TypeError):
            continue
        return _normalize_name(expression.func.id), args, kwargs
    return None


def _payload_from_function_call(
    call: tuple[str, list[Any], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    name, positional, kwargs = call
    params = dict(kwargs)
    if name in {"click", "tap", "touch", "long_press", "long_click"}:
        if len(positional) == 1:
            params.setdefault("coordinate", positional[0])
        elif len(positional) >= 2:
            params.setdefault("coordinate", positional[:2])
    elif name in {"swipe", "drag", "left_click_drag"}:
        if len(positional) >= 4 and all(isinstance(item, (int, float)) for item in positional[:4]):
            params.setdefault("coordinate", positional[:2])
            params.setdefault("coordinate2", positional[2:4])
        elif len(positional) >= 2:
            params.setdefault("coordinate", positional[0])
            params.setdefault("coordinate2", positional[1])
    elif name in {"scroll"}:
        if positional:
            params.setdefault("direction", positional[0])
        if len(positional) > 1:
            params.setdefault("amount", positional[1])
    elif name in {"type", "type_text", "input_text", "open", "open_app", "answer", "take_note"}:
        if positional:
            params.setdefault("text", positional[0])
    elif name in {"key", "press_key", "system_button"}:
        if positional:
            params.setdefault("key", positional[0])
    elif name == "wait" and positional:
        params.setdefault("time", positional[0])
    return name, params


def _point(value: Any, field: str, width: int, height: int) -> list[int]:
    if isinstance(value, Mapping):
        value = [value.get("x"), value.get("y")]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ClaudeActionProtocolError(f"{field}_must_be_two_numbers")
    result: list[int] = []
    for index, (component, limit) in enumerate(zip(value, (width, height), strict=True)):
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ClaudeActionProtocolError(f"{field}[{index}]_must_be_numeric")
        if not math.isfinite(float(component)):
            raise ClaudeActionProtocolError(f"{field}[{index}]_must_be_finite")
        rounded = round(float(component))
        if rounded < 0 or rounded >= limit:
            raise ClaudeActionProtocolError(
                f"{field}[{index}]={component}_outside_0_{limit - 1}"
            )
        result.append(rounded)
    return result


def _first_value(args: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in args and args[key] is not None:
            return args[key]
    return None


def _directional_swipe(
    args: Mapping[str, Any],
    *,
    width: int,
    height: int,
    scroll_semantics: bool,
) -> tuple[list[int], list[int]]:
    direction = _normalize_name(
        args.get("direction") or args.get("scroll_direction") or args.get("swipe_direction")
    )
    if direction not in {"up", "down", "left", "right"}:
        raise ClaudeActionProtocolError("swipe_requires_two_points_or_direction")
    amount = args.get("amount", args.get("scroll_amount", args.get("distance", "medium")))
    if isinstance(amount, str):
        distance = {"short": 360, "medium": 720, "long": 1200}.get(
            _normalize_name(amount),
            720,
        )
    elif isinstance(amount, (int, float)) and not isinstance(amount, bool):
        distance = round(float(amount) * 160) if abs(float(amount)) <= 10 else round(abs(float(amount)))
    else:
        distance = 720
    distance = max(120, min(distance, round(max(width, height) * 0.6)))

    anchor_value = _first_value(args, _POINT_KEYS)
    if anchor_value is None:
        anchor = [width // 2, height // 2]
    else:
        anchor = _point(anchor_value, "coordinate", width, height)

    # Computer-use scroll directions describe viewport movement; Android swipes
    # describe finger movement, so their vectors are opposite.
    vector = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }[direction]
    if scroll_semantics:
        vector = (-vector[0], -vector[1])
    half = distance / 2
    start = [round(anchor[0] - vector[0] * half), round(anchor[1] - vector[1] * half)]
    end = [round(anchor[0] + vector[0] * half), round(anchor[1] + vector[1] * half)]
    margin_x = min(24, max(0, width - 1))
    margin_y = min(24, max(0, height - 1))
    for point_value in (start, end):
        point_value[0] = max(margin_x, min(width - 1 - margin_x, point_value[0]))
        point_value[1] = max(margin_y, min(height - 1 - margin_y, point_value[1]))
    return start, end


def _normalize_action(
    name: str,
    args: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    original_name = _normalize_name(name)
    action = _ACTION_ALIASES.get(original_name, original_name)
    params = dict(args)
    if original_name in {"back", "press_back"}:
        action = "system_button"
        params.setdefault("button", "Back")
    elif original_name in {"home", "press_home"}:
        action = "system_button"
        params.setdefault("button", "Home")
    normalized: dict[str, Any] = {"action": action}

    if action in {"click", "long_press"}:
        value = _first_value(params, _POINT_KEYS)
        if value is None and "x" in params and "y" in params:
            value = [params["x"], params["y"]]
        normalized["coordinate"] = _point(value, "coordinate", width, height)
        if action == "long_press" and params.get("time") is not None:
            normalized["time"] = params["time"]
    elif action == "swipe":
        start_value = _first_value(params, _POINT_KEYS)
        end_value = _first_value(params, _END_POINT_KEYS)
        if start_value is None and all(key in params for key in ("x1", "y1")):
            start_value = [params["x1"], params["y1"]]
        if end_value is None and all(key in params for key in ("x2", "y2")):
            end_value = [params["x2"], params["y2"]]
        if start_value is not None and end_value is not None:
            start = _point(start_value, "coordinate", width, height)
            end = _point(end_value, "coordinate2", width, height)
        else:
            start, end = _directional_swipe(
                params,
                width=width,
                height=height,
                scroll_semantics=(original_name == "scroll"),
            )
        normalized.update({"coordinate": start, "coordinate2": end})
    elif action in {"type", "open", "answer", "take_note"}:
        text = params.get("text", params.get("content", params.get("value", params.get("app_name"))))
        if text is None:
            raise ClaudeActionProtocolError(f"{action}_requires_text")
        normalized["text"] = str(text)
    elif action == "key":
        key = params.get("text", params.get("key", params.get("button", params.get("value"))))
        if key is None:
            raise ClaudeActionProtocolError("key_requires_value")
        key_name = str(key).strip()
        system_buttons = {"back": "Back", "home": "Home", "menu": "Menu", "enter": "Enter"}
        if key_name.lower() in system_buttons:
            normalized = {"action": "system_button", "button": system_buttons[key_name.lower()]}
        else:
            normalized["text"] = key_name
    elif action == "system_button":
        button = str(params.get("button", params.get("key", params.get("value", "")))).strip().title()
        if button not in {"Back", "Home", "Menu", "Enter"}:
            raise ClaudeActionProtocolError("unsupported_system_button")
        normalized["button"] = button
    elif action == "wait":
        value = params.get("time", params.get("seconds", 1))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ClaudeActionProtocolError("wait_time_must_be_numeric")
        normalized["time"] = max(0, float(value))
    elif action == "terminate":
        status = str(params.get("status") or ("failure" if original_name == "fail" else "success")).lower()
        if status not in {"success", "failure"}:
            raise ClaudeActionProtocolError("terminate_status_must_be_success_or_failure")
        normalized["status"] = status
    elif action == "clear_text":
        pass
    else:
        raise ClaudeActionProtocolError(f"unsupported_action:{original_name or '<empty>'}")
    return normalized


def map_claude_vision_action_to_android(
    action: Mapping[str, Any],
    *,
    width: int = CLAUDE_PHONE_WIDTH,
    height: int = CLAUDE_PHONE_HEIGHT,
) -> dict[str, Any]:
    """Map Claude's internal vision-canvas points to Android absolute pixels."""
    mapped = copy.deepcopy(dict(action))

    def map_point(value: Any) -> list[int]:
        point = _point(
            value,
            "coordinate",
            CLAUDE_VISION_WIDTH,
            CLAUDE_VISION_HEIGHT,
        )
        return [
            round(point[0] * (width - 1) / (CLAUDE_VISION_WIDTH - 1)),
            round(point[1] * (height - 1) / (CLAUDE_VISION_HEIGHT - 1)),
        ]

    for field in ("coordinate", "coordinate2"):
        if field in mapped:
            mapped[field] = map_point(mapped[field])
    return mapped


def parse_claude_action(
    content: str,
    *,
    width: int = CLAUDE_PHONE_WIDTH,
    height: int = CLAUDE_PHONE_HEIGHT,
) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise ClaudeActionProtocolError("empty_model_response")

    normalized_candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    for payload in _json_candidates(content):
        unwrapped = _unwrap_payload(payload)
        if unwrapped is None:
            continue
        try:
            normalized_candidates.append(
                _normalize_action(*unwrapped, width=width, height=height)
            )
        except ClaudeActionProtocolError as exc:
            errors.append(str(exc))

    if not normalized_candidates and (call := _function_call(content)) is not None:
        try:
            normalized_candidates.append(
                _normalize_action(
                    *_payload_from_function_call(call),
                    width=width,
                    height=height,
                )
            )
        except ClaudeActionProtocolError as exc:
            errors.append(str(exc))

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in normalized_candidates:
        marker = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        if marker not in seen:
            seen.add(marker)
            unique.append(candidate)
    if len(unique) != 1:
        if not unique and errors:
            raise ClaudeActionProtocolError(errors[0])
        raise ClaudeActionProtocolError(f"expected_one_action_got_{len(unique)}")
    return unique[0]


def validate_claude_action(
    content: str,
    *,
    width: int = CLAUDE_PHONE_WIDTH,
    height: int = CLAUDE_PHONE_HEIGHT,
) -> ClaudeActionValidation:
    try:
        action = parse_claude_action(content, width=width, height=height)
    except ClaudeActionProtocolError as exc:
        return ClaudeActionValidation(ok=False, reason=str(exc))
    return ClaudeActionValidation(ok=True, action=action)


def _action_has_coordinates(action: Mapping[str, Any] | None) -> bool:
    return isinstance(action, Mapping) and any(
        key in action for key in ("coordinate", "coordinate2")
    )


def normalize_claude_provider_action(
    content: str,
    *,
    width: int = CLAUDE_PHONE_WIDTH,
    height: int = CLAUDE_PHONE_HEIGHT,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Normalize one Claude response and report the coordinate space used.

    Poixe's Claude route has returned actions grounded on a 704x1568 image,
    while some responses already use the Android screenshot dimensions.  A
    point outside the provider canvas is unambiguously an Android point; when
    both interpretations are possible we retain the empirically observed
    provider-canvas interpretation and record it in the call evidence.
    """
    vision = validate_claude_action(
        content,
        width=CLAUDE_VISION_WIDTH,
        height=CLAUDE_VISION_HEIGHT,
    )
    android = validate_claude_action(content, width=width, height=height)
    if vision.ok and _action_has_coordinates(vision.action):
        points = [
            point
            for field in ("coordinate", "coordinate2")
            if isinstance(vision.action, Mapping)
            for point in (vision.action.get(field),)
            if isinstance(point, (list, tuple))
        ]
        if any(
            float(point[index]) >= (CLAUDE_VISION_WIDTH, CLAUDE_VISION_HEIGHT)[index]
            for point in points
            for index in (0, 1)
        ):
            if android.ok:
                return android.action, "android_1080x2400", None
        return (
            map_claude_vision_action_to_android(
                vision.action or {}, width=width, height=height
            ),
            f"claude_vision_{CLAUDE_VISION_WIDTH}x{CLAUDE_VISION_HEIGHT}",
            None,
        )
    if android.ok:
        return android.action, "android_1080x2400", None
    reason = vision.reason or android.reason or "invalid_action"
    return None, None, reason


def is_retryable_provider_response(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return any(
        marker in text
        for marker in (
            "model_overloaded",
            "under heavy load",
            "overloaded_error",
            "service unavailable",
            "temporarily unavailable",
            "upstream timeout",
        )
    )


def canonical_mobile_use_response(content: str, action: Mapping[str, Any]) -> str:
    thought_match = re.search(
        r"(?im)^\s*(?:#{1,4}\s*)?(?:\*\*)?thought(?:\*\*)?\s*:\s*(.+)$",
        content,
    )
    thought = thought_match.group(1).strip() if thought_match else "Choose the next Android action."
    action_name = str(action.get("action") or "action")
    payload = {"name": "mobile_use", "arguments": dict(action)}
    return (
        f"Thought: {thought}\n"
        f"Action: {action_name}\n"
        f"<tool_call>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</tool_call>"
    )


def _response_content(response: Any) -> str:
    try:
        message = response.choices[0].message
    except Exception:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "tool_use":
                    parts.append(json.dumps(item, ensure_ascii=False))
            elif isinstance(getattr(item, "text", None), str):
                parts.append(item.text)
        if parts:
            return "\n".join(parts)
    tool_calls = getattr(message, "tool_calls", None) or []
    serialized: list[str] = []
    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        if function is None:
            continue
        serialized.append(
            json.dumps(
                {
                    "name": getattr(function, "name", None),
                    "arguments": getattr(function, "arguments", None),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(serialized)


def _summary(content: str, action: str) -> str:
    match = re.search(
        r"(?im)^\s*(?:#{1,4}\s*)?(?:\*\*)?summary(?:\*\*)?\s*:\s*(.+)$",
        content,
    )
    return match.group(1).strip() if match else f"Executed {action}."


def augment_appagent_prompt(prompt: str, *, width: int, height: int, grid_mode: bool) -> str:
    overlay_note = (
        "A grid overlay may be visible; use the grid area and subarea labels from the template. "
        if grid_mode
        else "Numeric element labels may be visible; use those element labels for interaction. "
    )
    return (
        f"{prompt.rstrip()}\n\n"
        "Claude protocol override: return exactly one AppAgent Action. "
        f"{overlay_note}For P2, use the numeric UI-element label or the grid "
        "label required by the surrounding AppAgent template; do not invent "
        f"absolute pixel coordinates for this SOM interaction. The screen is "
        f"{width}x{height}, but P2 executes the selected label through AppAgent."
    )


def parse_appagent_action(
    content: str,
    *,
    grid_mode: bool,
    screen_size: tuple[int, int],
) -> list[Any] | None:
    """Map Claude output to AppAgent's parser tuple or direct pixel actions."""
    if not isinstance(content, str) or not content.strip():
        return None
    width, height = screen_size
    summary = _summary(content, "action")
    action_match = re.search(
        r"(?im)^\s*(?:#{1,4}\s*)?(?:\*\*)?action(?:\*\*)?\s*:\s*(.+)$",
        content,
    )
    if action_match and "finish" in action_match.group(1).lower():
        return ["FINISH"]

    call = _function_call(content)
    if call is not None:
        name, positional, kwargs = call
        name = _ACTION_ALIASES.get(name, name)
        if name == "click" and grid_mode and len(positional) >= 2 and isinstance(positional[0], int):
            return ["tap_grid", positional[0], str(positional[1]), summary]
        if name == "click" and len(positional) == 1 and isinstance(positional[0], int):
            return ["tap", positional[0], summary]
        if name == "long_press" and grid_mode and len(positional) >= 2 and isinstance(positional[0], int):
            return ["long_press_grid", positional[0], str(positional[1]), summary]
        if name == "long_press" and len(positional) == 1 and isinstance(positional[0], int):
            return ["long_press", positional[0], summary]
        if name == "swipe" and not grid_mode and len(positional) >= 3 and isinstance(positional[0], int) and isinstance(positional[1], str):
            return ["swipe", positional[0], positional[1], str(positional[2]), summary]
        if name == "swipe" and grid_mode and len(positional) >= 4:
            return ["swipe_grid", positional[0], str(positional[1]), positional[2], str(positional[3]), summary]
        if name == "grid":
            return ["grid"]
        if name in {"type", "open", "answer", "key", "system_button", "wait", "clear_text", "terminate", "swipe", "click", "long_press"}:
            try:
                action = _normalize_action(
                    *_payload_from_function_call(call),
                    width=width,
                    height=height,
                )
            except ClaudeActionProtocolError:
                action = None
            if action is not None:
                return _appagent_tuple_from_android(action, summary)

    # JSON can retain AppAgent's numeric element tag without pretending it is
    # a pixel coordinate.
    for payload in _json_candidates(content):
        unwrapped = _unwrap_payload(payload)
        if unwrapped is None:
            continue
        raw_name, raw_args = unwrapped
        element = raw_args.get("element", raw_args.get("area"))
        mapped_name = _ACTION_ALIASES.get(raw_name, raw_name)
        if isinstance(element, int) and mapped_name in {"click", "long_press"}:
            return ["tap" if mapped_name == "click" else "long_press", element, summary]
    try:
        action = parse_claude_action(
            content,
            width=width,
            height=height,
        )
    except ClaudeActionProtocolError:
        return None
    return _appagent_tuple_from_android(action, summary)


def _appagent_tuple_from_android(action: Mapping[str, Any], summary: str) -> list[Any] | None:
    name = action.get("action")
    if name == "click":
        return ["tap_coordinate", action["coordinate"], summary]
    if name == "long_press":
        return ["long_press_coordinate", action["coordinate"], summary]
    if name == "swipe":
        return ["swipe_coordinate", action["coordinate"], action["coordinate2"], summary]
    if name == "type":
        return ["text", action["text"], summary]
    if name == "open":
        return ["open_app", action["text"], summary]
    if name == "system_button":
        button = action.get("button")
        if button == "Back":
            return ["back", summary]
        if button == "Home":
            return ["home", summary]
        return None
    if name == "answer":
        return ["answer", action["text"], summary]
    if name == "terminate":
        return ["FINISH"] if action.get("status") == "success" else ["ERROR"]
    return None


class ClaudeAbsoluteAdapter(ModelAdapter):
    """Poixe Claude adapter that emits fixed-screen absolute Android actions."""

    name = "claude-absolute"
    coordinate_mode = "pixel"
    coordinate_base = None
    preserve_raw_pixel_coordinates = True

    def __init__(self) -> None:
        self.last_resolved_model: str | None = None
        self.last_action_validation: ClaudeActionValidation | None = None
        self.last_coordinate_space: str | None = None
        self._call_record_overrides: dict[int, tuple[list[dict[str, Any]], str]] = {}

    def chat(
        self,
        client: Any,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        request_messages = augment_messages_with_absolute_contract(messages)
        _strip_incompatible_image_fields(request_messages)
        with request_network_context(getattr(client, "base_url", "")):
            response = client.chat.completions.create(
                model=model,
                messages=request_messages,
                **kwargs,
            )
        self.last_resolved_model = getattr(response, "model", None)
        raw_content = _response_content(response)
        if is_android_action_request(messages):
            screen_width, screen_height = CLAUDE_ANDROID_SCREEN_SIZE
            action, coordinate_space, reason = normalize_claude_provider_action(
                raw_content, width=screen_width, height=screen_height
            )
            self.last_coordinate_space = coordinate_space
            self.last_action_validation = ClaudeActionValidation(
                ok=action is not None,
                action=action,
                reason=reason,
                coordinate_space=coordinate_space,
            )
            if self.last_action_validation.ok:
                response.choices[0].message.content = canonical_mobile_use_response(
                    raw_content,
                    self.last_action_validation.action or {},
                )
            else:
                response.choices[0].message.content = (
                    f"{CLAUDE_ACTION_PROTOCOL_ERROR}: "
                    f"{self.last_action_validation.reason}"
                )
        else:
            self.last_action_validation = None
            self.last_coordinate_space = None
        self._call_record_overrides[id(response)] = (request_messages, raw_content)
        return response

    def call_record_overrides(
        self,
        response: Any,
        fallback_messages: list[dict[str, Any]],
        fallback_content: str,
    ) -> tuple[list[dict[str, Any]], str]:
        return self._call_record_overrides.pop(
            id(response),
            (fallback_messages, fallback_content),
        )

    def parse_action(self, content: str) -> dict[str, Any]:
        return parse_claude_action(content)

    def normalize_appagent_response(
        self,
        content: str,
        *,
        screen_size: tuple[int, int],
    ) -> str:
        """Give P2 the same canonical Android-pixel action as P1/P4.

        AppAgent uses its own HTTP client and does not pass through ``chat``.
        Poixe's Claude route has empirically returned points on a 704x1568
        vision canvas, so normalize that response at this adapter boundary.
        Already-valid Android-screen coordinates are preserved.
        """
        if not isinstance(content, str) or not content.strip():
            return content
        action, _, _ = normalize_claude_provider_action(
            content, width=int(screen_size[0]), height=int(screen_size[1])
        )
        if action is not None:
            return canonical_mobile_use_response(content, action)
        return content

    def denormalize_coordinates(
        self,
        point: list[float] | tuple[float, float],
        raw_size: tuple[int, int],
    ) -> tuple[int, int]:
        if tuple(raw_size) != CLAUDE_ANDROID_SCREEN_SIZE:
            raise ClaudeActionProtocolError(
                "unexpected_screen_size_" + "x".join(str(value) for value in raw_size)
            )
        width, height = CLAUDE_ANDROID_SCREEN_SIZE
        parsed = _point(point, "coordinate", width, height)
        return parsed[0], parsed[1]

    def augment_appagent_prompt(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        grid_mode: bool,
    ) -> str:
        return augment_appagent_prompt(
            prompt,
            width=width,
            height=height,
            grid_mode=grid_mode,
        )

    def parse_appagent_action(
        self,
        content: str,
        *,
        grid_mode: bool,
        screen_size: tuple[int, int],
    ) -> list[Any] | None:
        return parse_appagent_action(
            content,
            grid_mode=grid_mode,
            screen_size=screen_size,
        )

    def network_context(self, base_url: Any):
        return request_network_context(base_url)

    def is_retryable_provider_response(self, value: Any) -> bool:
        return is_retryable_provider_response(value)


def _is_mobileagent_v3_operator_prompt(prompt: str) -> bool:
    lowered = str(prompt or "").lower()
    return "### action ###" in lowered and "atomic actions" in lowered


class ClaudeMobileAgentV3Wrapper:
    """Adapt Claude actions to MobileAgent-v3's absolute-pixel JSON schema."""

    def __init__(
        self,
        delegate: Any,
        adapter: ClaudeAbsoluteAdapter | None = None,
    ) -> None:
        self._delegate = delegate
        self._adapter = adapter or ClaudeAbsoluteAdapter()
        self.last_coordinate_space: str | None = None
        self.last_action_validation: ClaudeActionValidation | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def _normalize_operator_response(self, prompt: str, content: str) -> str:
        if not _is_mobileagent_v3_operator_prompt(prompt):
            self.last_coordinate_space = None
            self.last_action_validation = None
            return content
        action, coordinate_space, reason = normalize_claude_provider_action(content)
        self.last_coordinate_space = coordinate_space
        self.last_action_validation = ClaudeActionValidation(
            ok=action is not None,
            action=action,
            reason=reason,
            coordinate_space=coordinate_space,
        )
        if action is None:
            return content
        return json.dumps(action, ensure_ascii=False, separators=(",", ":"))

    def predict(self, text_prompt: str):
        content, payload, raw = self._delegate.predict(text_prompt)
        return self._normalize_operator_response(text_prompt, content), payload, raw

    def predict_mm(self, text_prompt: str, images: list[Any], messages: Any = None):
        content, payload, raw = self._delegate.predict_mm(text_prompt, images, messages)
        return self._normalize_operator_response(text_prompt, content), payload, raw

    def build_call_record(self, messages: Any, response: Any, *, stage: str | None = None):
        record = self._delegate.build_call_record(messages, response, stage=stage)
        if isinstance(record, Mapping):
            record = dict(record)
            record["coordinate_space"] = self.last_coordinate_space
        else:
            try:
                record.coordinate_space = self.last_coordinate_space
            except Exception:
                pass
        return record

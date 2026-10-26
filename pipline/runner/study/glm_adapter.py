"""Standalone GLM-4.6V-Flash protocol adapter for the study pipeline.

The module owns GLM request compatibility, the normalized-coordinate
contract, response parsing, and the MobileAgentV3 boundary wrapper.  It does
not inherit behavior from the Gemini, Qwen, or GPT adapters.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import re
from typing import Any, Mapping, Sequence


GLM_MODEL_FLASH = "glm-4.6v-flash"
GLM_MODEL_CANONICAL = GLM_MODEL_FLASH
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
GEEKAI_BASE_URL = "https://geekai.co/api/v1"
GLM_COORDINATE_BASE = 1000
GLM_COORDINATE_CONTRACT_TAG = "[GLM-4.6V-Flash normalized-coordinate contract]"
GLM_COORDINATE_CONTRACT = f"""
{GLM_COORDINATE_CONTRACT_TAG}
For every touch action, x and y are independent normalized coordinates in the
inclusive range 0..1000. [0, 0] is the top-left, [500, 500] is the screen
center, and [1000, 1000] is the bottom-right. Always locate the visual center
of the requested target in the current full screenshot. Return exactly two
numbers for each coordinate, never a bounding box. Never return screenshot
pixel coordinates or an extra closing bracket. Use `click` for coordinate
taps; `open` is only for launching an app and its `text` must be an exact
supported app display name from the system prompt. Do not guess Android
package identifiers; the adapter resolves display names. Validate `coordinate`
and `coordinate2` before responding, and return exactly one executable action
inside a `<tool_call>` block.
""".strip()

_ACTION_ALIASES = {
    "left_click": "click",
    "tap": "click",
    "scroll": "swipe",
    "open_app": "open",
}
_TOUCH_COORDINATES = {
    "click": ("coordinate",),
    "long_press": ("coordinate",),
    "swipe": ("coordinate", "coordinate2"),
}
_TEXT_ACTIONS = {"answer", "key", "open", "take_note", "type"}
_ACTIONS = {
    *_TOUCH_COORDINATES,
    *_TEXT_ACTIONS,
    "clear_text",
    "system_button",
    "terminate",
    "wait",
}
_GLM_APP_PACKAGE_ALIASES = {
    "android world": "com.example.androidworld",
    "audio recorder": "com.dimowner.audiorecorder",
    "audio recorder app": "com.dimowner.audiorecorder",
    "broccoli": "com.flauschcode.broccoli",
    "broccoli app": "com.flauschcode.broccoli",
    "browser": "com.android.chrome",
    "camera": "com.android.camera2",
    "chrome": "com.android.chrome",
    "clipper": "ca.zgrs.clipper",
    "clock": "com.google.android.deskclock",
    "contacts": "com.google.android.contacts",
    "dialer": "com.google.android.dialer",
    "draw": "com.simplemobiletools.draw.pro",
    "file": "com.google.android.documentsui",
    "file manager": "com.google.android.documentsui",
    "files": "com.google.android.documentsui",
    "gallery": "com.simplemobiletools.gallery.pro",
    "gmail": "com.google.android.gm",
    "google chrome": "com.android.chrome",
    "google contacts": "com.google.android.contacts",
    "joplin": "net.cozic.joplin",
    "markor": "net.gsantner.markor",
    "messages": "com.google.android.apps.messaging",
    "miniwob": "com.google.androidenv.miniwob",
    "open tracks": "de.dennisguse.opentracks",
    "open tracks sports tracker": "de.dennisguse.opentracks",
    "opentracks": "de.dennisguse.opentracks",
    "osmand": "net.osmand",
    "phone": "com.google.android.dialer",
    "pro expense": "com.arduia.expense",
    "retro music": "code.name.monkey.retromusic",
    "recorder": "com.dimowner.audiorecorder",
    "settings": "com.android.settings",
    "simple calendar": "com.simplemobiletools.calendar.pro",
    "simple calendar pro": "com.simplemobiletools.calendar.pro",
    "simple draw": "com.simplemobiletools.draw.pro",
    "simple draw pro": "com.simplemobiletools.draw.pro",
    "simple gallery": "com.simplemobiletools.gallery.pro",
    "simple gallery pro": "com.simplemobiletools.gallery.pro",
    "simple sms": "com.simplemobiletools.smsmessenger",
    "simple sms messenger": "com.simplemobiletools.smsmessenger",
    "tasks": "org.tasks",
    "vlc": "org.videolan.vlc",
    "youtube": "com.google.android.youtube",
    # Package-shaped aliases GLM has emitted (or commonly infers) for these
    # AndroidWorld apps. The executor needs the installed package, not merely
    # a syntactically valid identifier.
    "com.android.calendar": "com.simplemobiletools.calendar.pro",
    "com.android.camera": "com.android.camera2",
    "com.android.contacts": "com.google.android.contacts",
    "com.android.deskclock": "com.google.android.deskclock",
    "com.android.dialer": "com.google.android.dialer",
    "com.android.documentsui": "com.google.android.documentsui",
    "com.android.filemanager": "com.google.android.documentsui",
    "com.android.gallery3d": "com.simplemobiletools.gallery.pro",
    "com.android.messaging": "com.google.android.apps.messaging",
    "com.android.youtube": "com.google.android.youtube",
    "com.google.android.apps.contacts": "com.google.android.contacts",
    "com.google.android.apps.dialer": "com.google.android.dialer",
    "com.google.android.apps.documentsui": "com.google.android.documentsui",
    "com.google.android.apps.recorder": "com.dimowner.audiorecorder",
    "com.google.android.chrome": "com.android.chrome",
    "com.lynxapp.audio.recorder": "com.dimowner.audiorecorder",
    "com.nerd-ware.audiorecorder": "com.dimowner.audiorecorder",
}
_GLM_ANDROIDWORLD_PACKAGES = frozenset(_GLM_APP_PACKAGE_ALIASES.values())
_ANDROID_PACKAGE_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+$"
)


class GLMCoordinateProtocolError(ValueError):
    """Raised when a GLM action violates the experiment coordinate contract."""


def _message_text(messages: Sequence[Mapping[str, Any]]) -> str:
    pieces: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            pieces.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                    pieces.append(item["text"])
    return "\n".join(pieces)


def _has_image(messages: Sequence[Mapping[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, Mapping) and "image" in str(item.get("type") or "").lower():
                return True
    return False


def _is_action_prompt(text: str) -> bool:
    normalized = str(text or "").lower()
    has_coordinates = "coordinate" in normalized
    has_schema = "<tools>" in normalized or "### action" in normalized
    has_output = "<tool_call>" in normalized or "### action" in normalized
    return has_coordinates and has_schema and has_output


def _append_contract(content: Any) -> Any:
    if isinstance(content, str):
        return f"{content.rstrip()}\n\n{GLM_COORDINATE_CONTRACT}"
    if isinstance(content, list):
        content.append({"type": "text", "text": GLM_COORDINATE_CONTRACT})
    return content


def augment_messages_with_coordinate_contract(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    prepared = copy.deepcopy(list(messages))
    text = _message_text(prepared)
    if not _has_image(prepared) or not _is_action_prompt(text):
        return prepared
    if GLM_COORDINATE_CONTRACT_TAG in text:
        return prepared
    for message in prepared:
        if str(message.get("role") or "").lower() == "system":
            message["content"] = _append_contract(message.get("content"))
            return prepared
    for message in reversed(prepared):
        if str(message.get("role") or "").lower() == "user":
            message["content"] = _append_contract(message.get("content"))
            return prepared
    return prepared


def augment_text_prompt_with_coordinate_contract(text_prompt: str) -> str:
    if not _is_action_prompt(text_prompt) or GLM_COORDINATE_CONTRACT_TAG in text_prompt:
        return text_prompt
    return f"{text_prompt.rstrip()}\n\n{GLM_COORDINATE_CONTRACT}"


def _sanitize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    prepared = augment_messages_with_coordinate_contract(messages)
    for message in prepared:
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
                image_url.pop("detail", None)
    return prepared


def _repair_common_json_errors(text: str) -> str:
    """Repair only the unambiguous extra bracket seen in real GLM traces."""
    return re.sub(
        r'("(?:coordinate|coordinate2)"\s*:\s*\[[^\[\]]*\])\s*\](?=\s*[,}])',
        r"\1",
        str(text or ""),
    )


def _parse_object(text: str) -> dict[str, Any] | None:
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    cleaned = _repair_common_json_errors(cleaned)
    if not cleaned:
        return None
    if not cleaned.startswith("{") and '"action"' in cleaned:
        cleaned = "{" + cleaned.strip().strip(",") + "}"
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(cleaned)
        except (SyntaxError, ValueError):
            return None
    return value if isinstance(value, dict) else None


def _scan_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(text):
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        cursor = start + max(consumed, 1)
    return objects


def extract_action_payloads(content: str) -> list[dict[str, Any]]:
    if not isinstance(content, str) or not content.strip():
        return []
    content = _repair_common_json_errors(content)
    blocks = re.findall(
        r"<tool_call>(.*?)</tool_call>", content, flags=re.DOTALL | re.IGNORECASE
    )
    if blocks:
        return [parsed for block in blocks if (parsed := _parse_object(block)) is not None]
    tools_blocks = re.findall(
        r"<tools>(.*?)</tools>", content, flags=re.DOTALL | re.IGNORECASE
    )
    if tools_blocks:
        return [
            parsed
            for block in tools_blocks
            if (parsed := _parse_object(block)) is not None
        ]
    section = re.search(
        r"###\s*Action\s*###(.*?)(?:###\s*Description\s*###|\Z)",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if section:
        parsed = _parse_object(section.group(1))
        if parsed is not None:
            return [parsed]
        scanned = _scan_json_objects(section.group(1))
        if scanned:
            return [scanned[0]]
    parsed = _parse_object(content)
    if parsed is not None:
        return [parsed]
    return _scan_json_objects(content)


def _executable_action(payload: Mapping[str, Any]) -> dict[str, Any]:
    arguments = payload.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = payload.get("parameters")
    if isinstance(arguments, Mapping):
        return dict(arguments)
    return dict(payload)


def _normalize_coordinate(value: Any, key: str) -> list[float | int]:
    if not isinstance(value, (list, tuple)):
        raise GLMCoordinateProtocolError(f"{key}_must_be_two_numbers")
    point = list(value)
    if len(point) == 4:
        # GLM's visual-grounding examples use [xmin, ymin, xmax, ymax].
        # AndroidWorld executes points, so deterministically use box center.
        x1, y1, x2, y2 = point
        point = [(x1 + x2) / 2, (y1 + y2) / 2]
    if len(point) != 2:
        raise GLMCoordinateProtocolError(f"{key}_must_be_point_or_bbox")
    for axis, item in zip(("x", "y"), point):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise GLMCoordinateProtocolError(f"{key}_{axis}_not_numeric")
        if not math.isfinite(float(item)) or not 0 <= float(item) <= 1000:
            raise GLMCoordinateProtocolError(f"{key}_{axis}_outside_0_1000")
    return point


def _normalize_open_package(text: str) -> str:
    stripped = text.strip()
    alias = " ".join(stripped.lower().split())
    package = _GLM_APP_PACKAGE_ALIASES.get(alias, stripped)
    if not _ANDROID_PACKAGE_RE.fullmatch(package):
        raise GLMCoordinateProtocolError("open_requires_android_package_name")
    if package not in _GLM_ANDROIDWORLD_PACKAGES:
        raise GLMCoordinateProtocolError("open_requires_known_androidworld_package")
    return package


def _normalize_action(action: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(action)
    name = str(normalized.get("action") or "").strip().lower()
    name = _ACTION_ALIASES.get(name, name)
    if name == "open" and "coordinate" in normalized and not normalized.get("text"):
        # GLM sometimes describes a tap but labels it `open`. A coordinate is
        # unambiguously executable as a click; `open` requires an app name.
        name = "click"
    if name not in _ACTIONS:
        raise GLMCoordinateProtocolError(f"unsupported_action_{name or 'missing'}")
    normalized["action"] = name

    for key in _TOUCH_COORDINATES.get(name, ()):
        normalized[key] = _normalize_coordinate(normalized.get(key), key)

    if name in _TEXT_ACTIONS:
        text = normalized.get("text")
        if not isinstance(text, str) or not text.strip():
            raise GLMCoordinateProtocolError(f"{name}_requires_text")
        if name == "open":
            normalized["text"] = _normalize_open_package(text)
            normalized.pop("coordinate", None)
    if name == "system_button" and normalized.get("button") not in {
        "Back",
        "Enter",
        "Home",
        "Menu",
    }:
        raise GLMCoordinateProtocolError("system_button_requires_valid_button")
    if name == "terminate" and normalized.get("status") not in {
        "failure",
        "success",
    }:
        raise GLMCoordinateProtocolError("terminate_requires_valid_status")
    if name in {"long_press", "wait"} and "time" in normalized:
        duration = normalized["time"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < 0
        ):
            raise GLMCoordinateProtocolError(f"{name}_time_invalid")
    return normalized


def _validate_action(action: Mapping[str, Any]) -> None:
    _normalize_action(action)


class GLMAdapter:
    """OpenAI-compatible GLM-4.6V-Flash adapter with 0..1000 coordinates."""

    name = "glm"
    coordinate_mode = "normalized_1000"
    coordinate_base = GLM_COORDINATE_BASE

    def __init__(self) -> None:
        self.last_resolved_model: str | None = None
        self._call_record_overrides: dict[int, tuple[list[dict[str, Any]], str]] = {}

    def chat(
        self,
        client: Any,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        request_kwargs = dict(kwargs)
        request_kwargs.pop("logprobs", None)
        request_kwargs.pop("top_logprobs", None)
        legacy_extra = request_kwargs.pop("extra_body", None)
        extra_body = dict(legacy_extra) if isinstance(legacy_extra, Mapping) else {}
        enable_thinking = extra_body.pop("enable_thinking", None)
        extra_body.pop("thinking_budget", None)
        if "thinking" not in extra_body and enable_thinking is not None:
            extra_body["thinking"] = {
                "type": "enabled" if bool(enable_thinking) else "disabled"
            }
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        coordinate_request = _has_image(messages) and _is_action_prompt(
            _message_text(messages)
        )
        request_messages = _sanitize_messages(messages)
        response = client.chat.completions.create(
            model=model,
            messages=request_messages,
            **request_kwargs,
        )
        self.last_resolved_model = getattr(response, "model", None)
        try:
            original_content = response.choices[0].message.content or ""
        except Exception:
            original_content = ""
        if coordinate_request:
            try:
                action = self.parse_action(original_content)
            except GLMCoordinateProtocolError:
                pass
            else:
                canonical = json.dumps(
                    {"name": "mobile_use", "arguments": action},
                    ensure_ascii=False,
                )
                response.choices[0].message.content = (
                    f"<tool_call>{canonical}</tool_call>"
                )
        self._call_record_overrides[id(response)] = (
            request_messages,
            original_content,
        )
        return response

    def call_record_overrides(
        self,
        response: Any,
        fallback_messages: list[dict[str, Any]],
        fallback_content: str,
    ) -> tuple[list[dict[str, Any]], str]:
        """Return the exact GLM request and unmodified provider response."""
        return self._call_record_overrides.pop(
            id(response),
            (fallback_messages, fallback_content),
        )

    def normalize_usage(self, usage: Any) -> dict[str, Any]:
        from study.runtime_metrics import normalize_usage

        return normalize_usage(usage)

    def parse_action(self, content: str) -> dict[str, Any]:
        payloads = extract_action_payloads(content)
        if len(payloads) != 1:
            raise GLMCoordinateProtocolError(
                f"expected_one_action_got_{len(payloads)}"
            )
        return _normalize_action(_executable_action(payloads[0]))

    def denormalize_coordinates(
        self,
        point: list[float] | tuple[float, float],
        raw_size: tuple[int, int],
    ) -> tuple[int, int]:
        normalized = _normalize_action({"action": "click", "coordinate": point})
        point = normalized["coordinate"]
        width, height = raw_size
        x = round(float(point[0]) / self.coordinate_base * width)
        y = round(float(point[1]) / self.coordinate_base * height)
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))

    def rewrite_mobileagent_v3_response(self, content: str) -> str:
        try:
            action = self.parse_action(content)
        except GLMCoordinateProtocolError:
            return content
        return json.dumps(action, ensure_ascii=False)


class GLMMobileAgentV3Wrapper:
    """GLM-only boundary wrapper preserving MobileAgentV3's wrapper API."""

    def __init__(self, delegate: Any, adapter: GLMAdapter | None = None):
        self._delegate = delegate
        self._adapter = adapter or GLMAdapter()
        raw_client = getattr(delegate, "bot", None)
        if raw_client is not None:
            delegate.bot = _GLMClientProxy(raw_client, self._adapter)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def predict(self, text_prompt: str):
        prepared = augment_text_prompt_with_coordinate_contract(text_prompt)
        content, payload, raw = self._delegate.predict(prepared)
        return self._adapter.rewrite_mobileagent_v3_response(content), payload, raw

    def predict_mm(self, text_prompt: str, images: list[Any], messages: Any = None):
        prepared = augment_text_prompt_with_coordinate_contract(text_prompt)
        content, payload, raw = self._delegate.predict_mm(prepared, images, messages)
        return self._adapter.rewrite_mobileagent_v3_response(content), payload, raw

    def build_call_record(self, messages: Any, response: Any, stage: Any = None):
        record = self._delegate.build_call_record(messages, response, stage)
        raw_messages, raw_content = self._adapter.call_record_overrides(
            response,
            messages,
            getattr(getattr(response.choices[0], "message", None), "content", ""),
        )
        if isinstance(record, dict):
            record["messages"] = raw_messages
            record["response"] = raw_content
        else:
            record.messages = raw_messages
            record.response = raw_content
        return record


class _GLMCompletionsProxy:
    def __init__(self, client: Any, adapter: GLMAdapter):
        self._client = client
        self._adapter = adapter

    def create(self, **kwargs: Any) -> Any:
        prepared = dict(kwargs)
        model = prepared.pop("model")
        messages = prepared.pop("messages")
        extra_body = prepared.get("extra_body")
        if not isinstance(extra_body, Mapping):
            prepared["extra_body"] = {"enable_thinking": False}
        return self._adapter.chat(
            self._client,
            model=model,
            messages=messages,
            **prepared,
        )


class _GLMChatProxy:
    def __init__(self, client: Any, adapter: GLMAdapter):
        self.completions = _GLMCompletionsProxy(client, adapter)


class _GLMClientProxy:
    def __init__(self, client: Any, adapter: GLMAdapter):
        self.chat = _GLMChatProxy(client, adapter)

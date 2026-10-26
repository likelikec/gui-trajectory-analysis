"""Standalone GPT-5.1 protocol adapter used by the study runners.

This module deliberately does not inherit from the Gemini adapter.  GPT
request cleanup, coordinate prompting, response validation, and the
MobileAgentV3 wrapper all live here so changes to either provider cannot
silently change the other provider's behavior.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


GPT_MODEL_CANONICAL = "gpt-5.1"
GPT_COORDINATE_BASE = 1000
GPT_PHONE_WIDTH = 1080
GPT_PHONE_HEIGHT = 2400
GPT_COORDINATE_CONTRACT_TAG = "[GPT-5.1 normalized-coordinate contract]"
GPT_ABSOLUTE_COORDINATE_CONTRACT_TAG = "[GPT-5.1 absolute-coordinate contract]"
GPT_COORDINATE_PROTOCOL_ERROR = "GPT_COORDINATE_PROTOCOL_ERROR"

GPT_COORDINATE_CONTRACT = f"""
{GPT_COORDINATE_CONTRACT_TAG}
For touch actions, both axes use independent normalized coordinates in the
inclusive range 0..1000.  [0, 0] is the top-left, [500, 500] is the screen
center, and [1000, 1000] is the bottom-right.  A portrait screenshot does not
make the y range taller: y must still be at most 1000.  Never output screenshot
pixel coordinates.  Before responding, verify every value in `coordinate` and
`coordinate2` is numeric and within 0..1000, and output exactly one action.
""".strip()


def absolute_coordinate_contract(
    width: int | None = None,
    height: int | None = None,
) -> str:
    width = width or GPT_PHONE_WIDTH
    height = height or GPT_PHONE_HEIGHT
    x_ticks = "/".join(str(round(width * index / 10)) for index in range(10))
    y_ticks = "/".join(str(round(height * index / 10)) for index in range(10))
    return f"""
{GPT_ABSOLUTE_COORDINATE_CONTRACT_TAG}
The phone screenshot is always the full original {width}x{height} screen. For touch
actions, output original-screen absolute pixels: x is 0..{width - 1} and y is
0..{height - 1}. Never use a normalized 0..1000 grid and never output coordinates
from an internally resized image.

Use this fixed 10% ruler only to calculate location, not as a list of points to
snap to. From 0% through 90%, x is {x_ticks}; y is {y_ticks}. The far right and
bottom edges are x={width - 1} and y={height - 1}. Interpolate between ruler
marks. For example, a target near 75% width and 90% height is near
({round(width * 0.75)}, {round(height * 0.90)}), not near the screen center.

Locate the center of the actual visible target in the current screenshot and
estimate x and y independently. Do not infer a conventional button position or
assume that an item is centered. Include the whole screenshot: the fixed status
bar is approximately y=0..135, app/launcher content is y=136..2336, and the
system navigation strip is y=2337..{height - 1}; do not crop or subtract these
regions. Before clicking, double-check whether the target is left/center/right
and top/middle/bottom. If a click caused no visible change, inspect the new
screenshot and relocate the target instead of moving the old guess arbitrarily.
Verify every value in `coordinate` and `coordinate2` is within the fixed ranges
and output exactly one action.
""".strip()

_TOUCH_COORDINATES = {
    "click": ("coordinate",),
    "left_click": ("coordinate",),
    "long_press": ("coordinate",),
    "swipe": ("coordinate", "coordinate2"),
    "scroll": ("coordinate", "coordinate2"),
}


@dataclass(frozen=True)
class CoordinateValidation:
    """Result of validating the executable action in one model response."""

    ok: bool
    reason: str | None = None
    action_count: int = 0


class GPTCoordinateProtocolError(ValueError):
    """Raised by explicit validation helpers for unsafe GPT coordinates."""


def _message_text(messages: Sequence[Mapping[str, Any]]) -> str:
    pieces: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            pieces.append(content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, Mapping):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    pieces.append(text)
    return "\n".join(pieces)


def _has_image(messages: Sequence[Mapping[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping):
                continue
            item_type = str(item.get("type") or "").lower()
            if "image" in item_type or "image" in item:
                return True
    return False


def is_coordinate_action_prompt(text: str) -> bool:
    """Identify an operator prompt without matching reflector/history text."""
    normalized = str(text or "").lower()
    if "coordinate" not in normalized:
        return False
    has_tool_schema = (
        "<tools>" in normalized
        or "#### atomic actions" in normalized
        or "atomic action functions" in normalized
    )
    has_action_output = (
        "<tool_call>" in normalized
        or "### action" in normalized
        or '"action"' in normalized
    )
    return has_tool_schema and has_action_output


def is_coordinate_action_request(messages: Sequence[Mapping[str, Any]]) -> bool:
    return _has_image(messages) and is_coordinate_action_prompt(_message_text(messages))


def _append_text_to_content(content: Any, text: str) -> Any:
    if isinstance(content, str):
        return f"{content.rstrip()}\n\n{text}"
    if isinstance(content, list):
        content.append({"type": "text", "text": text})
        return content
    return content


def augment_messages_with_coordinate_contract(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy messages and add the GPT-only coordinate contract when needed."""
    prepared = copy.deepcopy(list(messages))
    if not is_coordinate_action_request(prepared):
        return prepared
    if GPT_COORDINATE_CONTRACT_TAG in _message_text(prepared):
        return prepared

    for message in prepared:
        if str(message.get("role") or "").lower() == "system":
            message["content"] = _append_text_to_content(
                message.get("content"),
                GPT_COORDINATE_CONTRACT,
            )
            return prepared

    for message in reversed(prepared):
        if str(message.get("role") or "").lower() == "user":
            message["content"] = _append_text_to_content(
                message.get("content"),
                GPT_COORDINATE_CONTRACT,
            )
            return prepared
    return prepared


def augment_text_prompt_with_coordinate_contract(text_prompt: str) -> str:
    if (
        not is_coordinate_action_prompt(text_prompt)
        or GPT_COORDINATE_CONTRACT_TAG in text_prompt
    ):
        return text_prompt
    return f"{text_prompt.rstrip()}\n\n{GPT_COORDINATE_CONTRACT}"


def augment_messages_with_absolute_coordinate_contract(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy messages and add the GPT-only absolute-pixel contract."""
    prepared = copy.deepcopy(list(messages))
    if not is_coordinate_action_request(prepared):
        return prepared
    if GPT_ABSOLUTE_COORDINATE_CONTRACT_TAG in _message_text(prepared):
        return prepared
    contract = absolute_coordinate_contract()
    for message in prepared:
        if str(message.get("role") or "").lower() == "system":
            message["content"] = _append_text_to_content(message.get("content"), contract)
            return prepared
    for message in reversed(prepared):
        if str(message.get("role") or "").lower() == "user":
            message["content"] = _append_text_to_content(message.get("content"), contract)
            return prepared
    return prepared


def augment_text_prompt_with_absolute_coordinate_contract(
    text_prompt: str,
    *,
    width: int | None = None,
    height: int | None = None,
) -> str:
    if (
        not is_coordinate_action_prompt(text_prompt)
        or GPT_ABSOLUTE_COORDINATE_CONTRACT_TAG in text_prompt
    ):
        return text_prompt
    return (
        f"{text_prompt.rstrip()}\n\n"
        f"{absolute_coordinate_contract(width=width, height=height)}"
    )


def _strip_gpt_incompatible_image_fields(messages: list[dict[str, Any]]) -> None:
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


def _parse_object(text: str) -> dict[str, Any] | None:
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    if not cleaned:
        return None
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
    """Extract P1/P4 tool calls or a P3 ``### Action`` JSON object."""
    if not isinstance(content, str) or not content.strip():
        return []

    tool_blocks = re.findall(
        r"<tool_call>(.*?)</tool_call>",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if tool_blocks:
        return [
            parsed or {"__gpt_invalid_action_block__": True}
            for block in tool_blocks
            if (parsed := _parse_object(block)) is not None
            or block.strip()
        ]

    action_section = re.search(
        r"###\s*Action\s*###(.*?)(?:###\s*Description\s*###|\Z)",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if action_section:
        parsed = _parse_object(action_section.group(1))
        if parsed is not None:
            return [parsed]
        scanned = _scan_json_objects(action_section.group(1))
        if scanned:
            return [scanned[0]]

    parsed = _parse_object(content)
    if parsed is not None:
        return [parsed]

    return [
        candidate
        for candidate in _scan_json_objects(content)
        if "action" in candidate or "arguments" in candidate
    ]


def _action_arguments(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        arguments = _parse_object(arguments)
    if isinstance(arguments, Mapping):
        return arguments
    return payload if "action" in payload else None


def _validate_point(value: Any, field: str) -> str | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return f"{field}_must_be_two_numbers"
    for index, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            return f"{field}[{index}]_must_be_numeric"
        numeric = float(component)
        if not math.isfinite(numeric):
            return f"{field}[{index}]_must_be_finite"
        if numeric < 0 or numeric > GPT_COORDINATE_BASE:
            return (
                f"{field}[{index}]={component}_outside_"
                f"0_{GPT_COORDINATE_BASE}"
            )
    return None


def validate_coordinate_response(content: str) -> CoordinateValidation:
    """Reject unsafe coordinates; never clamp a protocol violation."""
    payloads = extract_action_payloads(content)
    if not payloads:
        return CoordinateValidation(ok=True, action_count=0)
    if len(payloads) != 1:
        return CoordinateValidation(
            ok=False,
            reason=f"expected_one_action_got_{len(payloads)}",
            action_count=len(payloads),
        )

    arguments = _action_arguments(payloads[0])
    if arguments is None:
        return CoordinateValidation(ok=True, action_count=1)
    action = str(arguments.get("action") or "").strip().lower()
    required_fields = _TOUCH_COORDINATES.get(action, ())
    for field in required_fields:
        if field not in arguments:
            return CoordinateValidation(
                ok=False,
                reason=f"{action}_missing_{field}",
                action_count=1,
            )
    for field in ("coordinate", "coordinate2"):
        if field not in arguments:
            continue
        reason = _validate_point(arguments[field], field)
        if reason:
            return CoordinateValidation(ok=False, reason=reason, action_count=1)
    return CoordinateValidation(ok=True, action_count=1)


def guard_coordinate_response(content: str) -> tuple[str, CoordinateValidation]:
    validation = validate_coordinate_response(content)
    if validation.ok:
        return content, validation
    guarded = f"{GPT_COORDINATE_PROTOCOL_ERROR}: {validation.reason}"
    return guarded, validation


def _validate_pixel_point(
    value: Any,
    field: str,
    *,
    width: int | None = None,
    height: int | None = None,
) -> str | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return f"{field}_must_be_two_numbers"
    limits = (width, height)
    for index, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            return f"{field}[{index}]_must_be_numeric"
        numeric = float(component)
        if not math.isfinite(numeric):
            return f"{field}[{index}]_must_be_finite"
        if numeric < 0:
            return f"{field}[{index}]={component}_must_be_nonnegative"
        limit = limits[index]
        if limit is not None and numeric >= limit:
            return f"{field}[{index}]={component}_outside_0_{limit - 1}"
    return None


def validate_pixel_coordinate_response(
    content: str,
    *,
    width: int | None = None,
    height: int | None = None,
) -> CoordinateValidation:
    payloads = extract_action_payloads(content)
    if not payloads:
        return CoordinateValidation(ok=True, action_count=0)
    if len(payloads) != 1:
        return CoordinateValidation(
            ok=False,
            reason=f"expected_one_action_got_{len(payloads)}",
            action_count=len(payloads),
        )
    arguments = _action_arguments(payloads[0])
    if arguments is None:
        return CoordinateValidation(ok=True, action_count=1)
    action = str(arguments.get("action") or "").strip().lower()
    required_fields = _TOUCH_COORDINATES.get(action, ())
    for field in required_fields:
        if field not in arguments:
            return CoordinateValidation(
                ok=False,
                reason=f"{action}_missing_{field}",
                action_count=1,
            )
    for field in ("coordinate", "coordinate2"):
        if field not in arguments:
            continue
        reason = _validate_pixel_point(
            arguments[field], field, width=width, height=height
        )
        if reason:
            return CoordinateValidation(ok=False, reason=reason, action_count=1)
    return CoordinateValidation(ok=True, action_count=1)


def guard_pixel_coordinate_response(
    content: str,
    *,
    width: int | None = None,
    height: int | None = None,
) -> tuple[str, CoordinateValidation]:
    validation = validate_pixel_coordinate_response(
        content, width=width, height=height
    )
    if validation.ok:
        return content, validation
    return f"{GPT_COORDINATE_PROTOCOL_ERROR}: {validation.reason}", validation


def _response_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except Exception:
        return ""
    return content if isinstance(content, str) else ""


def _replace_response_content(response: Any, content: str) -> None:
    response.choices[0].message.content = content


class GPTAdapter:
    """Independent Poixe/OpenAI-compatible GPT-5.1 protocol adapter."""

    name = "gpt"
    coordinate_mode = "normalized_1000"
    coordinate_base = GPT_COORDINATE_BASE

    def __init__(self) -> None:
        self.last_resolved_model: str | None = None
        self.last_coordinate_validation: CoordinateValidation | None = None
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
        if (
            "max_tokens" in request_kwargs
            and "max_completion_tokens" not in request_kwargs
        ):
            request_kwargs["max_completion_tokens"] = request_kwargs.pop("max_tokens")

        coordinate_request = is_coordinate_action_request(messages)
        request_messages = augment_messages_with_coordinate_contract(messages)
        _strip_gpt_incompatible_image_fields(request_messages)
        response = client.chat.completions.create(
            model=model,
            messages=request_messages,
            **request_kwargs,
        )
        self.last_resolved_model = getattr(response, "model", None)

        original_content = _response_content(response)
        recorded_content = original_content
        if coordinate_request:
            guarded_content, validation = guard_coordinate_response(original_content)
            self.last_coordinate_validation = validation
            if not validation.ok:
                _replace_response_content(response, guarded_content)
        else:
            self.last_coordinate_validation = None

        self._call_record_overrides[id(response)] = (
            request_messages,
            recorded_content,
        )
        return response

    def call_record_overrides(
        self,
        response: Any,
        fallback_messages: list[dict[str, Any]],
        fallback_content: str,
    ) -> tuple[list[dict[str, Any]], str]:
        """Return the exact provider request and unmasked provider response."""
        return self._call_record_overrides.pop(
            id(response),
            (fallback_messages, fallback_content),
        )

    def parse_action(self, content: str) -> Any:
        payloads = extract_action_payloads(content)
        if not payloads:
            raise ValueError("action_json_parse_error")
        if len(payloads) != 1:
            raise ValueError("expected_exactly_one_action")
        validation = validate_coordinate_response(content)
        if not validation.ok:
            raise GPTCoordinateProtocolError(validation.reason)
        return payloads[0]

    def denormalize_coordinates(
        self,
        point: list[float] | tuple[float, float],
        raw_size: tuple[int, int],
    ) -> tuple[int, int]:
        reason = _validate_point(point, "coordinate")
        if reason:
            raise GPTCoordinateProtocolError(reason)
        width, height = raw_size
        x = round(float(point[0]) / self.coordinate_base * width)
        y = round(float(point[1]) / self.coordinate_base * height)
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))

    def convert_action_coordinates(
        self,
        action: Mapping[str, Any],
        raw_size: tuple[int, int],
        *,
        target_base: int = GPT_COORDINATE_BASE,
    ) -> dict[str, Any]:
        del raw_size
        if target_base != GPT_COORDINATE_BASE:
            raise ValueError("GPT-5.1 adapter only emits normalized 0..1000 coordinates")
        converted = dict(action)
        for field in ("coordinate", "coordinate2"):
            if field not in converted:
                continue
            reason = _validate_point(converted[field], field)
            if reason:
                raise GPTCoordinateProtocolError(reason)
            converted[field] = [
                round(float(converted[field][0])),
                round(float(converted[field][1])),
            ]
        return converted


class GPTAbsoluteAdapter(GPTAdapter):
    """GPT-5.1 adapter that preserves absolute screenshot-pixel actions."""

    name = "gpt-absolute"
    coordinate_mode = "pixel"
    coordinate_base = None

    def chat(
        self,
        client: Any,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        request_kwargs = dict(kwargs)
        if (
            "max_tokens" in request_kwargs
            and "max_completion_tokens" not in request_kwargs
        ):
            request_kwargs["max_completion_tokens"] = request_kwargs.pop("max_tokens")

        coordinate_request = is_coordinate_action_request(messages)
        request_messages = augment_messages_with_absolute_coordinate_contract(messages)
        _strip_gpt_incompatible_image_fields(request_messages)
        response = client.chat.completions.create(
            model=model,
            messages=request_messages,
            **request_kwargs,
        )
        self.last_resolved_model = getattr(response, "model", None)

        original_content = _response_content(response)
        recorded_content = original_content
        if coordinate_request:
            guarded_content, validation = guard_pixel_coordinate_response(original_content)
            self.last_coordinate_validation = validation
            if not validation.ok:
                _replace_response_content(response, guarded_content)
        else:
            self.last_coordinate_validation = None
        self._call_record_overrides[id(response)] = (request_messages, recorded_content)
        return response

    def parse_action(self, content: str) -> Any:
        payloads = extract_action_payloads(content)
        if not payloads:
            raise ValueError("action_json_parse_error")
        if len(payloads) != 1:
            raise ValueError("expected_exactly_one_action")
        validation = validate_pixel_coordinate_response(content)
        if not validation.ok:
            raise GPTCoordinateProtocolError(validation.reason)
        return payloads[0]

    def denormalize_coordinates(
        self,
        point: list[float] | tuple[float, float],
        raw_size: tuple[int, int],
    ) -> tuple[int, int]:
        reason = _validate_pixel_point(
            point,
            "coordinate",
            width=raw_size[0],
            height=raw_size[1],
        )
        if reason:
            raise GPTCoordinateProtocolError(reason)
        return round(float(point[0])), round(float(point[1]))


class GPTMobileAgentV3Wrapper:
    """GPT-only boundary wrapper for MobileAgentV3's text response protocol."""

    def __init__(self, delegate: Any, adapter: GPTAdapter | None = None):
        self._delegate = delegate
        self._adapter = adapter or GPTAdapter()
        self.last_coordinate_validation: CoordinateValidation | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def _guard(self, prompt: str, content: str) -> str:
        if not is_coordinate_action_prompt(prompt):
            self.last_coordinate_validation = None
            return content
        guarded, validation = guard_coordinate_response(content)
        self.last_coordinate_validation = validation
        return guarded

    def predict(self, text_prompt: str):
        guarded_prompt = augment_text_prompt_with_coordinate_contract(text_prompt)
        content, payload, raw = self._delegate.predict(guarded_prompt)
        return self._guard(text_prompt, content), payload, raw

    def predict_mm(
        self,
        text_prompt: str,
        images: list[Any],
        messages: Any = None,
    ):
        guarded_prompt = augment_text_prompt_with_coordinate_contract(text_prompt)
        content, payload, raw = self._delegate.predict_mm(
            guarded_prompt,
            images,
            messages,
        )
        return self._guard(text_prompt, content), payload, raw

    def build_call_record(self, messages: Any, response: Any, *, stage: str | None = None):
        return self._delegate.build_call_record(messages, response, stage=stage)


class GPTAbsoluteMobileAgentV3Wrapper:
    """Inject the raw Android screen size into MobileAgentV3 operator calls."""

    def __init__(self, delegate: Any, adapter: GPTAbsoluteAdapter | None = None):
        self._delegate = delegate
        self._adapter = adapter or GPTAbsoluteAdapter()
        self.last_coordinate_validation: CoordinateValidation | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    @staticmethod
    def _image_dimensions(images: list[Any]) -> tuple[int | None, int | None]:
        if not images:
            return None, None
        image = images[-1]
        if isinstance(image, (str, Path)):
            try:
                from PIL import Image

                with Image.open(image) as opened:
                    return int(opened.width), int(opened.height)
            except (OSError, ValueError):
                return None, None
        shape = getattr(image, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1]), int(shape[0])
        width = getattr(image, "width", None)
        height = getattr(image, "height", None)
        if width is not None and height is not None:
            return int(width), int(height)
        return None, None

    def _guard(
        self,
        prompt: str,
        content: str,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> str:
        if not is_coordinate_action_prompt(prompt):
            self.last_coordinate_validation = None
            return content
        guarded, validation = guard_pixel_coordinate_response(
            content, width=width, height=height
        )
        self.last_coordinate_validation = validation
        return guarded

    def predict(self, text_prompt: str):
        guarded_prompt = augment_text_prompt_with_absolute_coordinate_contract(text_prompt)
        content, payload, raw = self._delegate.predict(guarded_prompt)
        return self._guard(text_prompt, content), payload, raw

    def predict_mm(
        self,
        text_prompt: str,
        images: list[Any],
        messages: Any = None,
    ):
        width, height = self._image_dimensions(images)
        guarded_prompt = augment_text_prompt_with_absolute_coordinate_contract(
            text_prompt, width=width, height=height
        )
        content, payload, raw = self._delegate.predict_mm(
            guarded_prompt, images, messages
        )
        return self._guard(
            text_prompt, content, width=width, height=height
        ), payload, raw

    def build_call_record(self, messages: Any, response: Any, *, stage: str | None = None):
        return self._delegate.build_call_record(messages, response, stage=stage)

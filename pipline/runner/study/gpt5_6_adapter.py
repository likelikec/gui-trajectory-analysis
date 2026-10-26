"""Standalone GPT-5.6 adapter for fixed-screen absolute-pixel experiments.

This module is intentionally independent from the GPT-5.1 and Gemini
adapters. It owns the GPT-5.6 request cleanup, coordinate prompt, response
parsing, and coordinate validation used by its smoke tests.
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

from study.gpt_action_parser import (
    GPTActionParseError,
    extract_gpt_action_candidates,
    normalize_gpt_action_response,
    parse_gpt_action,
)


GPT56_MODEL_CANONICAL = "gpt-5.6"
GPT56_PHONE_WIDTH = 1080
GPT56_PHONE_HEIGHT = 2400
GPT56_ABSOLUTE_CONTRACT_TAG = "[GPT-5.6 fixed absolute-pixel contract]"
GPT56_COORDINATE_PROTOCOL_ERROR = "GPT56_COORDINATE_PROTOCOL_ERROR"

_TOUCH_COORDINATES = {
    "click": ("coordinate",),
    "left_click": ("coordinate",),
    "long_press": ("coordinate",),
    "swipe": ("coordinate", "coordinate2"),
    "scroll": ("coordinate", "coordinate2"),
}


@dataclass(frozen=True)
class CoordinateValidation:
    ok: bool
    reason: str | None = None
    action_count: int = 0


class GPT56CoordinateProtocolError(ValueError):
    """Raised when a model action cannot be executed safely."""


def is_gpt56_adapter(vlm: Any) -> bool:
    """Return whether a VLM wrapper is using the isolated GPT-5.6 path."""
    return str(getattr(vlm, "adapter_name", "")).strip().lower() in {
        "gpt-5.6-absolute",
        "gpt5.6-absolute",
        "gpt-5.6-luna",
    }


def preserve_gpt56_raw_response(record: Any, response: Any) -> Any:
    """Keep provider text in GPT call records after adapter normalization."""
    raw = getattr(response, "_gpt56_raw_content", None)
    if isinstance(raw, str) and raw:
        try:
            record.response = raw
        except Exception:
            pass
    return record


def absolute_coordinate_contract(
    width: int = GPT56_PHONE_WIDTH,
    height: int = GPT56_PHONE_HEIGHT,
) -> str:
    """Return the fixed-screen prompt used by GPT-5.6 coordinate calls."""
    if (width, height) != (GPT56_PHONE_WIDTH, GPT56_PHONE_HEIGHT):
        raise ValueError("GPT-5.6 experiments require the fixed 1080x2400 screen")
    x_ticks = "/".join(str(round(width * index / 10)) for index in range(10))
    y_ticks = "/".join(str(round(height * index / 10)) for index in range(10))
    return f"""
{GPT56_ABSOLUTE_CONTRACT_TAG}
The screenshot is always the complete original {width}x{height} phone screen.
For every touch action, return original-screen absolute pixels: x is
0..{width - 1} and y is 0..{height - 1}. Never return normalized 0..1000
coordinates and never return coordinates from an internally resized image.

Use this 10% ruler for estimation without snapping to ruler marks. From 0%
through 90%, x is {x_ticks}; y is {y_ticks}. The far-right and bottom pixels
are ({width - 1}, {height - 1}). If the image is internally resized, map both
axes back to the original {width}x{height} screen before answering.

Locate the exact visual center of the visible target. Estimate x and y
independently from the full screenshot, including the status bar and navigation
strip. Do not infer a conventional control position. Before answering, verify
that each coordinate is inside the fixed ranges and return exactly one action.

Protocol requirements for this call:
* Return exactly one JSON action object, not a draft plus a final action.
* ``coordinate`` and ``coordinate2`` are arrays of exactly two numeric pixels;
  x must be 0..{width - 1} and y must be 0..{height - 1}.
* ``open`` takes the visible name of an installed Android app (for example
  ``Clock``), never a filename or an image/document path.
* Do not terminate successfully while the requested work is still pending.
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


def is_coordinate_action_request(messages: Sequence[Mapping[str, Any]]) -> bool:
    text = _message_text(messages).lower()
    return (
        _has_image(messages)
        and "coordinate" in text
        and ("<tool_call>" in text or "### action" in text or '"action"' in text)
    )


def _append_text(content: Any, text: str) -> Any:
    if isinstance(content, str):
        return f"{content.rstrip()}\n\n{text}"
    if isinstance(content, list):
        content.append({"type": "text", "text": text})
    return content


def augment_messages_with_absolute_contract(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    prepared = copy.deepcopy(list(messages))
    if not is_coordinate_action_request(prepared):
        return prepared
    if GPT56_ABSOLUTE_CONTRACT_TAG in _message_text(prepared):
        return prepared
    contract = absolute_coordinate_contract()
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


def _parse_object(text: str) -> dict[str, Any] | None:
    cleaned = str(text or "").strip().replace("```json", "").replace("```", "").strip()
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


def extract_action_payloads(content: str) -> list[dict[str, Any]]:
    """Backward-compatible export backed by the standalone GPT parser."""
    return extract_gpt_action_candidates(content)


def _arguments(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    arguments = payload.get("arguments") or payload.get("parameters")
    if isinstance(arguments, str):
        arguments = _parse_object(arguments)
    if isinstance(arguments, Mapping):
        return arguments
    return payload if "action" in payload else None


def _validate_point(value: Any, field: str) -> str | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return f"{field}_must_be_two_numbers"
    for index, (component, limit) in enumerate(
        zip(value, (GPT56_PHONE_WIDTH, GPT56_PHONE_HEIGHT), strict=True)
    ):
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            return f"{field}[{index}]_must_be_numeric"
        if not math.isfinite(float(component)):
            return f"{field}[{index}]_must_be_finite"
        if float(component) < 0 or float(component) >= limit:
            return f"{field}[{index}]={component}_outside_0_{limit - 1}"
    return None


def validate_coordinate_response(content: str) -> CoordinateValidation:
    try:
        parsed = parse_gpt_action(content, reject_multiple=True)
    except GPTActionParseError as exc:
        return CoordinateValidation(ok=False, reason=str(exc), action_count=0)
    return CoordinateValidation(ok=True, action_count=1)


def _response_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except Exception:
        return ""
    return content if isinstance(content, str) else ""


class GPT56AbsoluteAdapter:
    """Poixe/OpenAI-compatible adapter for GPT-5.6 absolute coordinates."""

    name = "gpt-5.6-absolute"
    coordinate_mode = "pixel"
    coordinate_base = None

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
        if "max_tokens" in request_kwargs and "max_completion_tokens" not in request_kwargs:
            request_kwargs["max_completion_tokens"] = request_kwargs.pop("max_tokens")
        # GPT-5.6 accepts only its default temperature through the Poixe endpoint.
        request_kwargs.pop("temperature", None)

        coordinate_request = is_coordinate_action_request(messages)
        request_messages = augment_messages_with_absolute_contract(messages)
        _strip_incompatible_image_fields(request_messages)
        response = client.chat.completions.create(
            model=model,
            messages=request_messages,
            **request_kwargs,
        )
        self.last_resolved_model = getattr(response, "model", None)
        raw_content = _response_content(response)
        if coordinate_request:
            try:
                normalized_content, _parsed = normalize_gpt_action_response(
                    raw_content,
                    target="mobile_use",
                    reject_multiple=True,
                )
            except GPTActionParseError as exc:
                self.last_coordinate_validation = CoordinateValidation(
                    ok=False,
                    reason=str(exc),
                    action_count=0,
                )
                # Keep the raw provider text available to the GPT-specific
                # parser/retry path.  Replacing it with an opaque marker made
                # it impossible to distinguish provider protocol errors from
                # parser bugs in the experiment ledger.
                response.choices[0].message.content = raw_content
            else:
                self.last_coordinate_validation = CoordinateValidation(
                    ok=True,
                    action_count=1,
                )
                response.choices[0].message.content = normalized_content
        else:
            self.last_coordinate_validation = None
        try:
            response._gpt56_raw_content = raw_content
            response._gpt56_coordinate_validation = self.last_coordinate_validation
        except Exception:
            pass
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
        try:
            return parse_gpt_action(content, reject_multiple=True).action
        except GPTActionParseError as exc:
            raise GPT56CoordinateProtocolError(str(exc)) from exc

    def denormalize_coordinates(
        self,
        point: list[float] | tuple[float, float],
        raw_size: tuple[int, int],
    ) -> tuple[int, int]:
        if raw_size != (GPT56_PHONE_WIDTH, GPT56_PHONE_HEIGHT):
            raise GPT56CoordinateProtocolError("unexpected_screen_size")
        if reason := _validate_point(point, "coordinate"):
            raise GPT56CoordinateProtocolError(reason)
        return round(float(point[0])), round(float(point[1]))


class GPT56MobileAgentV3Wrapper:
    """Adapt MobileAgentV3 text responses to the GPT-5.6 pixel contract."""

    def __init__(self, delegate: Any, adapter: GPT56AbsoluteAdapter | None = None):
        self._delegate = delegate
        self._adapter = adapter or GPT56AbsoluteAdapter()
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
        return (int(width), int(height)) if width is not None and height is not None else (None, None)

    def _guard(self, prompt: str, content: str) -> str:
        prompt_lower = str(prompt).lower()
        # Mobile-Agent-v3 uses the same VLM for manager, operator, reflector,
        # and note-taker calls.  Only its operator prompt requests the
        # ``### Action ###`` JSON section; never reinterpret prose from the
        # other stages as a device action merely because history mentions a
        # coordinate.
        if "coordinate" not in prompt_lower or "### action ###" not in prompt_lower:
            self.last_coordinate_validation = None
            return content
        try:
            normalized, _parsed = normalize_gpt_action_response(
                content,
                target="mobileagent_v3",
                reject_multiple=True,
            )
        except GPTActionParseError as exc:
            self.last_coordinate_validation = CoordinateValidation(
                ok=False,
                reason=str(exc),
                action_count=0,
            )
            return f"{GPT56_COORDINATE_PROTOCOL_ERROR}: {exc}"
        self.last_coordinate_validation = CoordinateValidation(ok=True, action_count=1)
        return normalized

    def predict(self, text_prompt: str):
        prompt = f"{text_prompt.rstrip()}\n\n{absolute_coordinate_contract()}"
        content, payload, raw = self._delegate.predict(prompt)
        return self._guard(text_prompt, content), payload, raw

    def predict_mm(self, text_prompt: str, images: list[Any], messages: Any = None):
        self._image_dimensions(images)
        prompt = f"{text_prompt.rstrip()}\n\n{absolute_coordinate_contract()}"
        content, payload, raw = self._delegate.predict_mm(prompt, images, messages)
        return self._guard(text_prompt, content), payload, raw

    def build_call_record(self, messages: Any, response: Any, *, stage: str | None = None):
        return self._delegate.build_call_record(messages, response, stage=stage)

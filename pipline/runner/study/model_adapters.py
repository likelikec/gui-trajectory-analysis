"""Model-specific protocol adapters for the study runners.

The adapters are intentionally kept at the model/protocol boundary.  They do
not plan tasks or execute actions; they only describe how a model is called,
how usage is normalized, and how model coordinates are converted.
"""

from __future__ import annotations

import ast
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
POIXE_BASE_URL = "https://api.poixe.com/v1"
BIGMODEL_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
GEEKAI_BASE_URL = "https://geekai.co/api/v1"
SILRA_BASE_URL = "https://api.silra.cn/v1"
GEMINI_MODEL_ALIAS = "gemini-3-flash-preview"
GEMINI_MODEL_CANONICAL = "google/gemini-3-flash-preview"
INSUFFICIENT_BALANCE_MARKER = "INSUFFICIENT_PROVIDER_BALANCE"


class InsufficientBalanceError(RuntimeError):
    """Raised when a provider reports exhausted paid balance or quota."""

    def __init__(self) -> None:
        super().__init__(INSUFFICIENT_BALANCE_MARKER)


def is_insufficient_balance_error(error: Any) -> bool:
    """Recognize billing exhaustion without treating ordinary rate limits as balance errors."""
    text = str(error or "").strip().lower()
    if not text:
        return False
    markers = (
        INSUFFICIENT_BALANCE_MARKER.lower(),
        "insufficient balance",
        "insufficient_balance",
        "insufficient funds",
        "balance is insufficient",
        "not enough balance",
        "credit balance is too low",
        "billing_hard_limit_reached",
        "insufficient_quota",
        "exceeded your current quota",
        "payment required",
        "余额不足",
        "额度不足",
        "账户余额不足",
        "账号余额不足",
        "欠费",
    )
    return any(marker in text for marker in markers)


def normalize_model_provider(provider: str | None) -> str:
    """Return the supported provider identifier used by local credentials."""
    normalized = (provider or "openrouter").strip().lower()
    aliases = {
        "openrouter": "openrouter",
        "poixe": "poixe",
        "bigmodel": "bigmodel",
        "zhipu": "bigmodel",
        "glm": "bigmodel",
        "geekai": "geekai",
        "silra": "silra",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported model provider: {provider!r}")
    return aliases[normalized]


def provider_base_url(provider: str) -> str:
    provider = normalize_model_provider(provider)
    return {
        "openrouter": OPENROUTER_BASE_URL,
        "poixe": POIXE_BASE_URL,
        "bigmodel": BIGMODEL_BASE_URL,
        "geekai": GEEKAI_BASE_URL,
        "silra": SILRA_BASE_URL,
    }[provider]


def provider_model_name(provider: str, model_name: str) -> str:
    """Map a shared Gemini model setting to the provider's model convention."""
    provider = normalize_model_provider(provider)
    if provider == "poixe" and model_name == GEMINI_MODEL_CANONICAL:
        return GEMINI_MODEL_ALIAS
    return model_name


def _parse_key_file(value: str) -> dict[str, str]:
    """Parse provider-tagged keys while retaining legacy single-key support."""
    parsed: dict[str, str] = {}
    legacy_key = ""
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        provider, separator, key = line.partition(":")
        if separator and provider.strip().lower() in {
            "openrouter",
            "poixe",
            "bigmodel",
            "zhipu",
            "glm",
            "geekai",
            "silra",
        }:
            parsed[normalize_model_provider(provider)] = key.strip()
        elif not legacy_key:
            legacy_key = line
    if legacy_key and "openrouter" not in parsed:
        parsed["openrouter"] = legacy_key
    return {provider: key for provider, key in parsed.items() if key}


def load_local_api_key(
    path: str | Path | None = None,
    *,
    provider: str = "openrouter",
) -> str:
    """Read a provider-tagged local key without ever logging its contents.

    Provider-specific files such as ``.secrets/poixe_api_key`` are preferred.
    Tagged multi-provider files remain supported, and a legacy unlabelled
    single-line file remains an OpenRouter key.
    """
    provider = normalize_model_provider(provider)
    candidates = []
    if path:
        candidates.append(Path(path))
    env_path = os.getenv(f"{provider.upper()}_API_KEY_FILE") or os.getenv("OPENROUTER_API_KEY_FILE")
    if env_path:
        candidates.append(Path(env_path))
    secrets_dir = Path(__file__).resolve().parents[2] / ".secrets"
    candidates.append(secrets_dir / f"{provider}_api_key")
    # Backward compatibility for earlier provider-tagged experiment setups.
    if provider != "openrouter":
        candidates.append(secrets_dir / "openrouter_api_key")
    for candidate in candidates:
        try:
            value = candidate.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            continue
        key = _parse_key_file(value).get(provider, "")
        if key:
            return key
    return ""


def normalize_usage(usage: Any) -> dict[str, Any]:
    """Use the canonical null-preserving usage normalizer."""
    from study.runtime_metrics import normalize_usage as normalize_runtime_usage

    return normalize_runtime_usage(usage)


class ModelAdapter(ABC):
    """Small model protocol interface shared by future model adapters."""

    name = "base"
    coordinate_mode = "pixel"
    coordinate_base: int | None = None
    # When true, the adapter has already converted the provider response to
    # the Android device's raw screenshot pixels.  Downstream architecture
    # parsers must not apply their legacy smart-resize conversion again.
    preserve_raw_pixel_coordinates = False

    @abstractmethod
    def chat(self, client: Any, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def parse_action(self, content: str) -> Any:
        raise NotImplementedError

    def normalize_usage(self, usage: Any) -> dict[str, Any]:
        return normalize_usage(usage)

    def normalize_appagent_response(
        self,
        content: str,
        *,
        screen_size: tuple[int, int],
    ) -> str:
        """Normalize an AppAgent response before its architecture parser."""
        del screen_size
        return content

    def denormalize_coordinates(
        self,
        point: list[float] | tuple[float, float],
        raw_size: tuple[int, int],
    ) -> tuple[int, int]:
        if self.coordinate_base is None:
            return int(point[0]), int(point[1])
        width, height = raw_size
        x = round(float(point[0]) / self.coordinate_base * width)
        y = round(float(point[1]) / self.coordinate_base * height)
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))


class GeminiAdapter(ModelAdapter):
    """OpenRouter-compatible Gemini adapter with 0-1000 coordinates."""

    name = "gemini"
    coordinate_mode = "normalized_1000"
    coordinate_base = 1000

    def __init__(self) -> None:
        self.last_resolved_model: str | None = None

    def chat(self, client: Any, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        response = client.chat.completions.create(model=model, messages=messages, **kwargs)
        self.last_resolved_model = getattr(response, "model", None)
        return response

    def parse_action(self, content: str) -> Any:
        """Parse a JSON action without inventing missing fields.

        The native mobile-use parser remains authoritative for execution.  This
        helper is used by smoke tests and adapters that need to inspect a raw
        Gemini response before handing it to an existing architecture parser.
        """
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty_model_response")
        text = content.strip().replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(text)
            except (SyntaxError, ValueError) as exc:
                raise ValueError("action_json_parse_error") from exc
            if not isinstance(value, dict):
                raise ValueError("action_json_not_object")
            return value

    def convert_action_coordinates(
        self,
        action: Mapping[str, Any],
        raw_size: tuple[int, int],
        *,
        target_base: int = 1000,
    ) -> dict[str, Any]:
        """Clamp a Gemini action to MobileAgentV3's normalized 0-1000 space.

        This is a pure boundary conversion for MobileAgentV3's existing
        qwen-vl (0-1000) adapter.  It never changes the action name or adds
        missing coordinates.
        """
        del raw_size
        converted = dict(action)
        for key in ("coordinate", "coordinate2"):
            point = converted.get(key)
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                converted[key] = [
                    max(0, min(target_base, round(float(point[0])))),
                    max(0, min(target_base, round(float(point[1])))),
                ]
            except (TypeError, ValueError):
                continue
        return converted

    def rewrite_mobileagent_v3_response(self, content: str) -> str:
        """Validate/clamp a raw Gemini action in MobileAgentV3's 0-1000 space."""
        try:
            action = self.parse_action(content)
        except ValueError:
            # Let the original architecture parser report the raw parse error.
            return content
        converted = self.convert_action_coordinates(action, (1, 1))
        if converted == action:
            return content
        return json.dumps(converted, ensure_ascii=False)


class GeminiMobileAgentV3Wrapper:
    """Boundary wrapper that preserves the existing MobileAgentV3 wrapper API."""

    def __init__(self, delegate: Any, adapter: GeminiAdapter | None = None):
        self._delegate = delegate
        self._adapter = adapter or GeminiAdapter()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def predict(self, text_prompt: str):
        content, payload, raw = self._delegate.predict(text_prompt)
        return self._adapter.rewrite_mobileagent_v3_response(content), payload, raw

    def predict_mm(self, text_prompt: str, images: list[Any], messages: Any = None):
        content, payload, raw = self._delegate.predict_mm(text_prompt, images, messages)
        return self._adapter.rewrite_mobileagent_v3_response(content), payload, raw


def adapter_for_name(name: str | None) -> ModelAdapter:
    normalized = (name or "").strip().lower()
    if normalized in {"", "auto", "qwen"}:
        # Existing Qwen paths keep their native behavior.
        return _GenericAdapter()
    if normalized in {"gemini", "gemini3", "gemini-3-flash-preview"}:
        return GeminiAdapter()
    if normalized in {"glm", "glm-4.6v-flash", "bigmodel"}:
        from study.glm_adapter import GLMAdapter

        return GLMAdapter()
    if normalized in {"gpt", "gpt5", "gpt-5", "gpt-5.1"}:
        from study.gpt5_1_adapter import GPTAdapter

        return GPTAdapter()
    if normalized in {"gpt-absolute", "gpt5-absolute", "gpt-5.1-absolute"}:
        from study.gpt5_1_adapter import GPTAbsoluteAdapter

        return GPTAbsoluteAdapter()
    if normalized in {"gpt-5.6-absolute", "gpt5.6-absolute", "gpt-5.6-luna"}:
        from study.gpt5_6_adapter import GPT56AbsoluteAdapter

        return GPT56AbsoluteAdapter()
    if normalized in {
        "claude",
        "claude-absolute",
        "claude-sonnet-5",
        "cli2api/claude-sonnet-5",
    }:
        from study.claude_adapter import ClaudeAbsoluteAdapter

        return ClaudeAbsoluteAdapter()
    raise ValueError(f"Unsupported model adapter: {name!r}")


class _GenericAdapter(ModelAdapter):
    name = "generic"
    coordinate_mode = "pixel"
    coordinate_base = None

    def __init__(self) -> None:
        self.last_resolved_model: str | None = None

    def chat(self, client: Any, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        response = client.chat.completions.create(model=model, messages=messages, **kwargs)
        self.last_resolved_model = getattr(response, "model", None)
        return response

    def parse_action(self, content: str) -> str:
        return content


def resolve_gemini_model_name(requested: str | None) -> tuple[str, str | None]:
    """Return one explicit OpenRouter model id; no silent model switching."""
    name = (requested or GEMINI_MODEL_CANONICAL).strip()
    return (GEMINI_MODEL_CANONICAL if name == GEMINI_MODEL_ALIAS else name), None

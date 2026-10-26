"""P1 reactive single-agent config builder.

This mode intentionally stays minimal: a single mobile-use Qwen agent receives
the current observation and directly predicts the next action.
"""

from __future__ import annotations

from typing import Any

from study.config import DefaultsConfig
from study.model_adapters import adapter_for_name


def _model_name(cfg: DefaultsConfig, common: dict[str, Any]) -> str:
    vlm = common.get("vlm") or {}
    return str(vlm.get("model_name") or cfg.model.model_name or "")


def _is_qwen3_model(cfg: DefaultsConfig, common: dict[str, Any]) -> bool:
    vlm = common.get("vlm") or {}
    adapter_name = str(vlm.get("adapter_name") or "").lower()
    if adapter_name == "gemini":
        return False
    return "qwen3" in _model_name(cfg, common).lower()


def _prompt_config(cfg: DefaultsConfig, common: dict[str, Any]) -> str:
    adapter_name = str((common.get("vlm") or {}).get("adapter_name") or "").lower()
    if adapter_name in {
        "gpt-absolute",
        "gpt-5.6-absolute",
        "gpt5.6-absolute",
        "gpt-5.6-luna",
    }:
        return "study_gpt5_1_absolute_reactive.yaml"
    if adapter_name in {"claude", "claude-absolute"}:
        return "study_claude_absolute_reactive.yaml"
    if adapter_name == "gpt":
        return "study_gpt5_1_reactive.yaml"
    if adapter_name == "gemini":
        return "study_gemini_reactive.yaml"
    if adapter_name == "glm":
        return "study_glm_reactive.yaml"
    return "study_qwen3_reactive.yaml" if _is_qwen3_model(cfg, common) else "qwen_agent.yaml"


def _coordinate_type(cfg: DefaultsConfig, common: dict[str, Any]) -> str:
    vlm = common.get("vlm") or {}
    adapter = adapter_for_name(vlm.get("adapter_name"))
    coordinate_mode = str(vlm.get("coordinate_mode") or "auto").lower()
    if coordinate_mode.startswith("normalized") or adapter.coordinate_base is not None:
        return "relative"
    return "relative" if _is_qwen3_model(cfg, common) else "absolute"


def build_p1_mobileuse_agent_config(
    *,
    cfg: DefaultsConfig,
    common: dict[str, Any],
) -> dict[str, Any]:
    """Returns the mobile-use config for the P1 reactive baseline."""
    agent_config = dict(common)
    agent_config.update(
        {
            "type": "Qwen",
            "prompt_config": _prompt_config(cfg, common),
            "coordinate_type": _coordinate_type(cfg, common),
            "coordinate_base": common["vlm"].get("coordinate_base") or 999,
            "enable_think": False,
            "message_type": "single",
            "max_pixels": common["vlm"]["max_pixels"],
        }
    )
    return agent_config

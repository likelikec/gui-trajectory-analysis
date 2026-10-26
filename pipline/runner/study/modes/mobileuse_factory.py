"""Builds mode-specific mobile-use agent configs.

Only P1 and P4 are backed by mobile-use in the current study layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from guiagent.p1_reactive import build_p1_mobileuse_agent_config
from study.config import DefaultsConfig, get_mode_model_config
from study.model_adapters import adapter_for_name


def _knowledge_block(explored_knowledge_path: str | None) -> dict[str, Any] | None:
    if not explored_knowledge_path:
        return None
    path = Path(explored_knowledge_path)
    if not path.exists() or not path.is_file():
        return None
    return {
        "explored_knowledge_path": str(path),
    }


def _base_vlm(cfg: DefaultsConfig, mode: str) -> dict[str, Any]:
    model_cfg = get_mode_model_config(cfg, mode)
    adapter = adapter_for_name(model_cfg.adapter_name)
    vlm: dict[str, Any] = {
        "model_name": model_cfg.model_name,
        "api_key": model_cfg.api_key,
        "base_url": model_cfg.base_url,
        "max_tokens": model_cfg.max_tokens,
        "temperature": model_cfg.temperature,
        "max_retry": model_cfg.max_retry,
        "retry_waiting_seconds": model_cfg.retry_waiting_seconds,
        "max_pixels": model_cfg.max_pixels,
        "adapter_name": model_cfg.adapter_name,
        "coordinate_mode": model_cfg.coordinate_mode if model_cfg.coordinate_mode != "auto" else adapter.coordinate_mode,
        "coordinate_base": model_cfg.coordinate_base or adapter.coordinate_base,
    }
    # Poixe's Gemini-compatible endpoint rejects OpenAI extension fields
    # such as ``enable_thinking``/``thinking_budget``.  Keep these fields for
    # providers that explicitly support them, but omit them for Poixe.
    extra_body: dict[str, Any] = {}
    if model_cfg.provider != "poixe":
        if model_cfg.enable_thinking is not None:
            extra_body["enable_thinking"] = model_cfg.enable_thinking
        if model_cfg.thinking_budget is not None:
            extra_body["thinking_budget"] = model_cfg.thinking_budget
    if extra_body:
        vlm["extra_body"] = extra_body
    return vlm


def _base_env(cfg: DefaultsConfig, serial_no: str | None, host: str, port: int) -> dict[str, Any]:
    return {
        "serial_no": serial_no,
        "host": host,
        "port": port,
        "go_home": False,
        "wait_after_action_seconds": cfg.executor.wait_after_action_seconds,
    }


def _operator_block(cfg: DefaultsConfig, explored_knowledge_path: str | None = None) -> dict[str, Any]:
    p4_model_cfg = get_mode_model_config(cfg, "p4")
    adapter = adapter_for_name(p4_model_cfg.adapter_name)
    use_gpt_protocol = (
        p4_model_cfg.adapter_name.lower().startswith("gpt")
        or p4_model_cfg.model_name.lower().startswith("gpt-")
    )
    block: dict[str, Any] = {
        "enabled": True,
        "name": "OperatorGPT" if use_gpt_protocol else "OperatorQwen",
        "prompt_config": "operator_gpt.yaml" if use_gpt_protocol else "operator_qwen.yaml",
        "include_device_time": cfg.observation.include_device_time,
        "include_tips": cfg.control.include_tips,
        "include_a11y_tree": cfg.observation.include_a11y_tree,
        "num_histories": cfg.observation.num_histories,
        "max_pixels": p4_model_cfg.max_pixels,
        "coordinate_type": "relative" if (
            p4_model_cfg.coordinate_mode.startswith("normalized")
            or adapter.coordinate_base is not None
            or "qwen3" in p4_model_cfg.model_name.lower()
        ) else "absolute",
        "coordinate_base": p4_model_cfg.coordinate_base or adapter.coordinate_base or 999,
    }
    knowledge = _knowledge_block(explored_knowledge_path)
    if knowledge:
        block["knowledge"] = knowledge
    return block


def _answer_agent_block(cfg: DefaultsConfig, explored_knowledge_path: str | None = None) -> dict[str, Any]:
    p4_model_cfg = get_mode_model_config(cfg, "p4")
    adapter = adapter_for_name(p4_model_cfg.adapter_name)
    use_gpt_protocol = (
        p4_model_cfg.adapter_name.lower().startswith("gpt")
        or p4_model_cfg.model_name.lower().startswith("gpt-")
    )
    block: dict[str, Any] = {
        "enabled": True,
        "name": "AnswerAgentGPT" if use_gpt_protocol else "AnswerAgentQwen",
        "prompt_config": "answer_agent_gpt.yaml" if use_gpt_protocol else "answer_agent_qwen.yaml",
        "include_device_time": cfg.observation.include_device_time,
        "include_tips": cfg.control.include_tips,
        "num_histories": cfg.observation.num_histories,
        "max_pixels": p4_model_cfg.max_pixels,
        "coordinate_type": "relative" if (
            p4_model_cfg.coordinate_mode.startswith("normalized")
            or adapter.coordinate_base is not None
            or "qwen3" in p4_model_cfg.model_name.lower()
        ) else "absolute",
        "coordinate_base": p4_model_cfg.coordinate_base or adapter.coordinate_base or 999,
    }
    knowledge = _knowledge_block(explored_knowledge_path)
    if knowledge:
        block["knowledge"] = knowledge
    return block


def _trajectory_reflector_block(cfg: DefaultsConfig) -> dict[str, Any]:
    return {
        "enabled": True,
        "num_latest_screenshots": cfg.observation.num_latest_screenshots,
    }


def _global_reflector_block(cfg: DefaultsConfig) -> dict[str, Any]:
    return {
        "enabled": True,
        "num_latest_screenshots": cfg.observation.num_latest_screenshots,
    }


def build_mobileuse_agent_config(
    mode: str,
    cfg: DefaultsConfig,
    serial_no: str | None,
    host: str,
    port: int,
    log_dir: str | Path,
    explored_knowledge_path: str | None = None,
) -> dict[str, Any]:
    """Returns a mobile-use Agent.from_params config dict for P1/P4 modes."""
    if mode not in {"p1", "p4"}:
        raise ValueError(f"Mode {mode!r} is not backed by mobile-use.")

    common = {
        "vlm": _base_vlm(cfg, mode),
        "env": _base_env(cfg, serial_no, host, port),
        "max_steps": cfg.control.max_steps,
        "enable_log": cfg.control.enable_log,
        "log_dir": str(log_dir),
        "max_action_retry": cfg.control.max_action_retry,
    }

    if mode == "p1":
        return build_p1_mobileuse_agent_config(
            cfg=cfg,
            common=common,
        )

    common.update(
        {
            "type": "MultiAgent",
            "planner": {"enabled": True},
            "operator": _operator_block(cfg, explored_knowledge_path),
            "answer_agent": _answer_agent_block(cfg, explored_knowledge_path),
            "reflector": {"enabled": True},
            "trajectory_reflector": _trajectory_reflector_block(cfg),
            "global_reflector": _global_reflector_block(cfg),
            "progressor": {"enabled": True},
            "note_taker": {"enabled": True},
            "reflect_on_demand": cfg.control.reflect_on_demand,
        }
    )
    return common

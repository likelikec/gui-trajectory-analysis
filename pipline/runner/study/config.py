"""Configuration loading for unified mode runs."""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_ACTION_SPACE = [
    "open",
    "click",
    "long_press",
    "type",
    "key",
    "swipe",
    "press_home",
    "press_back",
    "wait",
    "answer",
    "system_button",
    "clear_text",
    "take_note",
    "terminate",
]


@dataclass
class ModelConfig:
    provider: str = "silra"
    model_name: str = "qwen3-vl-235b-a22b-instruct"
    api_key: str = ""
    base_url: str = "https://api.silra.cn/v1"
    max_tokens: int = 1024
    temperature: float = 0.0
    max_retry: int = 3
    retry_waiting_seconds: int = 2
    max_pixels: int = 12_845_056
    enable_thinking: bool | None = False
    thinking_budget: int | None = None
    adapter_name: str = "auto"
    coordinate_mode: str = "auto"
    coordinate_base: int | None = None


@dataclass
class PartialModelConfig:
    provider: str | None = None
    model_name: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    max_retry: int | None = None
    retry_waiting_seconds: int | None = None
    max_pixels: int | None = None
    enable_thinking: bool | None = None
    thinking_budget: int | None = None
    adapter_name: str | None = None
    coordinate_mode: str | None = None
    coordinate_base: int | None = None


@dataclass
class ModeModelConfigs:
    p1: PartialModelConfig = field(default_factory=PartialModelConfig)
    p2: PartialModelConfig = field(default_factory=PartialModelConfig)
    p3_decision: PartialModelConfig = field(default_factory=PartialModelConfig)
    p3_planning: PartialModelConfig = field(default_factory=PartialModelConfig)
    p3_caption: PartialModelConfig = field(default_factory=PartialModelConfig)
    p4: PartialModelConfig = field(default_factory=PartialModelConfig)


@dataclass
class ObservationConfig:
    screenshot: bool = True
    include_view_hierarchy: bool = False
    history_format: str = "mobile_use_text_history"
    include_device_time: bool = False
    include_a11y_tree: bool = False
    num_histories: int = 4
    num_latest_screenshots: int = 2


@dataclass
class ActionConfig:
    action_space: list[str] = field(
        default_factory=lambda: list(SUPPORTED_ACTION_SPACE)
    )


@dataclass
class ExecutorConfig:
    wait_after_action_seconds: float = 2.0
    request_interval: float = 3.0
    timeout_seconds: int | None = 120


@dataclass
class ControlConfig:
    max_steps: int = 30
    max_action_retry: int = 3
    task_random_seed: int = 30
    n_task_combinations: int = 1
    fixed_task_seed: bool = False
    enable_log: bool = True
    include_tips: bool = True
    reflect_on_demand: bool = True


@dataclass
class BenchmarkConfig:
    androidworld_suite_family: str = "android_world"
    androidworld_a11y_method: str = "a11y_forwarder_app"
    # Optional per-mode override for flaky device-side accessibility services.
    # None preserves the historical shared method.
    androidworld_p2_a11y_method: str | None = None
    androidworld_device_ready_timeout_seconds: int = 240
    androidworld_launch_emulator: bool = True
    androidworld_wipe_data_on_launch: bool = False
    androidworld_freeze_datetime: bool = True
    androidworld_appagent_avd_name: str = "AndroidWorldAvd_AppAgent"
    androidworld_p1_avd_name: str = ""
    androidworld_p2_avd_name: str = ""
    androidworld_p3_avd_name: str = "AndroidWorldAvd_P3"
    androidworld_p4_avd_name: str = ""
    androidworld_p4_proactive_exploration: bool = True
    androidworld_p3_coor_type: str = "qwen-vl"
    emulator_gpu: str = "host"
    emulator_memory_mb: int | None = None
    emulator_show_window: bool = False
    p3_timeout_seconds: int | None = 420
    p4_model_name: str = ""
    androidlab_max_rounds: int = 20
    androidlab_parallel: int = 1
    androidlab_boot_stabilization_seconds: float = 15.0


@dataclass
class DefaultsConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    mode_models: ModeModelConfigs = field(default_factory=ModeModelConfigs)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_MODE_MODEL_ROLES = ("p1", "p2", "p3_decision", "p3_planning", "p3_caption", "p4")
_MODE_ROLE_LABELS = {
    "p1": "p1 reactive model",
    "p2": "p2 AppAgent model",
    "p3_decision": "p3 MobileAgentV3 model",
    "p3_planning": "p3 planning model",
    "p3_caption": "p3 caption model",
    "p4": "p4 MultiAgent model",
}
_MODE_ROLE_REQUIREMENTS = {
    "p1": {"api_key": True, "base_url": True},
    "p2": {"api_key": True, "base_url": True},
    "p3_decision": {"api_key": True, "base_url": True},
    "p3_planning": {"api_key": True, "base_url": True},
    "p3_caption": {"api_key": True, "base_url": False},
    "p4": {"api_key": True, "base_url": True},
}


def _merge_dataclass(
    instance: Any,
    values: dict[str, Any] | None,
    *,
    preserve_none_keys: set[str] | None = None,
) -> Any:
    if not values:
        return instance
    preserve_none_keys = preserve_none_keys or set()
    for key, value in values.items():
        if not hasattr(instance, key):
            continue
        if value is None and key not in preserve_none_keys:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        setattr(instance, key, value)
    return instance


def _merge_mode_models(instance: ModeModelConfigs, values: dict[str, Any] | None) -> ModeModelConfigs:
    if not values:
        return instance
    for role in _MODE_MODEL_ROLES:
        _merge_dataclass(getattr(instance, role), values.get(role))
    return instance


def load_defaults(config_path: str | Path | None = None) -> DefaultsConfig:
    cfg = DefaultsConfig()
    if config_path is None:
        return cfg

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Defaults config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    _merge_dataclass(cfg.model, data.get("model"))
    _merge_mode_models(cfg.mode_models, data.get("mode_models"))
    _merge_dataclass(cfg.observation, data.get("observation"))
    _merge_dataclass(cfg.action, data.get("action"))
    # An explicit null disables the task-level wall-clock cutoff.  Preserve it
    # instead of silently falling back to ExecutorConfig.timeout_seconds.
    _merge_dataclass(
        cfg.executor,
        data.get("executor"),
        preserve_none_keys={"timeout_seconds"},
    )
    _merge_dataclass(cfg.control, data.get("control"))
    _merge_dataclass(
        cfg.benchmark,
        data.get("benchmark"),
        preserve_none_keys={"p3_timeout_seconds"},
    )
    return cfg


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value:
            return value
    return ""


def _blank_like(model: ModelConfig) -> ModelConfig:
    return ModelConfig(
        provider=model.provider,
        model_name="",
        api_key="",
        base_url="",
        max_tokens=model.max_tokens,
        temperature=model.temperature,
        max_retry=model.max_retry,
        retry_waiting_seconds=model.retry_waiting_seconds,
        max_pixels=model.max_pixels,
        enable_thinking=model.enable_thinking,
        thinking_budget=model.thinking_budget,
        adapter_name=model.adapter_name,
        coordinate_mode=model.coordinate_mode,
        coordinate_base=model.coordinate_base,
    )


def _merge_partial_model(instance: ModelConfig, override: PartialModelConfig) -> ModelConfig:
    for key, value in asdict(override).items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        setattr(instance, key, value)
    return instance


def get_mode_model_config(cfg: DefaultsConfig, role: str) -> ModelConfig:
    if role not in _MODE_MODEL_ROLES:
        raise ValueError(f"Unsupported model role: {role}")

    shared = copy.deepcopy(cfg.model)
    resolved = shared
    override = copy.deepcopy(getattr(cfg.mode_models, role))

    _merge_partial_model(resolved, override)

    if role == "p4" and not resolved.model_name and cfg.benchmark.p4_model_name:
        resolved.model_name = cfg.benchmark.p4_model_name
    return resolved


def required_model_roles(mode: str) -> tuple[str, ...]:
    if mode == "p1":
        return ("p1",)
    if mode == "p2":
        return ("p2",)
    if mode == "p3":
        return ("p3_decision",)
    if mode == "p4":
        return ("p4",)
    raise ValueError(f"Unsupported mode: {mode}")


def fill_model_credentials(
    cfg: DefaultsConfig,
    api_key: str | None,
    base_url: str | None,
    model_name: str | None,
    provider: str | None = None,
) -> DefaultsConfig:
    from study.model_adapters import (
        load_local_api_key,
        normalize_model_provider,
        provider_base_url,
        provider_model_name,
    )

    selected_provider = normalize_model_provider(
        provider or os.getenv("VLM_PROVIDER") or os.getenv("MODEL_PROVIDER") or cfg.model.provider
    )
    provider_changed = selected_provider != normalize_model_provider(cfg.model.provider)
    cfg.model.provider = selected_provider

    cfg.model.api_key = _first_non_empty(
        api_key,
        cfg.model.api_key,
        os.getenv("VLM_API_KEY"),
        os.getenv("API_KEY"),
        os.getenv(f"{selected_provider.upper()}_API_KEY"),
        os.getenv("DASHSCOPE_API_KEY"),
        load_local_api_key(provider=selected_provider),
    )
    cfg.model.base_url = _first_non_empty(
        base_url,
        cfg.model.base_url,
        os.getenv("VLM_BASE_URL"),
        os.getenv("BASE_URL"),
    )
    if provider_changed and not base_url and not os.getenv("VLM_BASE_URL") and not os.getenv("BASE_URL"):
        cfg.model.base_url = provider_base_url(selected_provider)
    cfg.model.model_name = _first_non_empty(
        model_name,
        cfg.model.model_name,
        os.getenv("VLM_MODEL_NAME"),
        os.getenv("MODEL"),
    ) or cfg.model.model_name
    cfg.model.model_name = provider_model_name(selected_provider, cfg.model.model_name)

    if api_key:
        for role in _MODE_MODEL_ROLES:
            getattr(cfg.mode_models, role).api_key = api_key
    if base_url:
        for role in _MODE_MODEL_ROLES:
            getattr(cfg.mode_models, role).base_url = base_url
    if model_name:
        for role in _MODE_MODEL_ROLES:
            getattr(cfg.mode_models, role).model_name = model_name
    if provider:
        for role in _MODE_MODEL_ROLES:
            getattr(cfg.mode_models, role).provider = selected_provider
    return cfg


def ensure_model_ready(cfg: DefaultsConfig, mode: str | None = None) -> None:
    if not cfg.observation.screenshot:
        raise ValueError("This pipeline currently requires screenshot=true.")
    if cfg.observation.include_view_hierarchy:
        raise ValueError("include_view_hierarchy=true is not wired in the study wrappers yet.")
    if cfg.observation.history_format != "mobile_use_text_history":
        raise ValueError("Only history_format=mobile_use_text_history is currently supported.")
    if cfg.action.action_space != SUPPORTED_ACTION_SPACE:
        raise ValueError(
            "action_space must match the unified supported set in study.config.SUPPORTED_ACTION_SPACE."
        )

    roles = required_model_roles(mode) if mode else ("p1",)
    for role in roles:
        model_cfg = get_mode_model_config(cfg, role)
        label = _MODE_ROLE_LABELS[role]
        if not model_cfg.model_name:
            raise ValueError(f"Missing {label} name in defaults yaml.")
        requirements = _MODE_ROLE_REQUIREMENTS[role]
        if requirements["api_key"] and not model_cfg.api_key:
            raise ValueError(f"Missing {label} api_key in defaults yaml.")
        if requirements["base_url"] and not model_cfg.base_url:
            raise ValueError(f"Missing {label} base_url in defaults yaml.")

"""P3 MobileAgentV3 preview integration."""

from __future__ import annotations

from pathlib import Path

from study.config import DefaultsConfig, get_mode_model_config


def build_p3_preview(
    *,
    cfg: DefaultsConfig,
    serial_no: str | None,
    host: str,
    port: int,
    log_dir: str | Path,
    explored_knowledge_path: str | None = None,
) -> dict:
    del explored_knowledge_path
    model_cfg = get_mode_model_config(cfg, "p3_decision")
    return {
        "runtime": "MobileAgentV3(AndroidWorld-native)",
        "androidworld_env": {
            "serial_no": serial_no,
            "host": host,
            "port": port,
            "wait_after_action_seconds": cfg.executor.wait_after_action_seconds,
            "coor_type": cfg.benchmark.androidworld_p3_coor_type,
        },
        "model": {
            "api_key": model_cfg.api_key,
            "base_url": model_cfg.base_url,
            "model_name": model_cfg.model_name,
            "max_tokens": model_cfg.max_tokens,
            "temperature": model_cfg.temperature,
        },
        "max_steps": cfg.control.max_steps,
        "timeout_seconds": cfg.executor.timeout_seconds,
        "log_dir": str(log_dir),
    }


__all__ = ["build_p3_preview"]

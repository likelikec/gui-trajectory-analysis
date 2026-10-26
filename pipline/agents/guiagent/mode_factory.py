"""Mode labels and dry-run previews for the unified study CLI."""

from __future__ import annotations

from pathlib import Path

from guiagent.p2_appagent import build_p2_preview
from guiagent.p3_mobileagentv3 import build_p3_preview
from guiagent.p4_mobileuse import build_p4_preview
from study.config import DefaultsConfig
from study.modes.mobileuse_factory import build_mobileuse_agent_config


MODE_LABELS = {
    "p1": "Visual ReAct",
    "p2": "XML-Assisted ReAct",
    "p3": "Planner-Executor-Reflector",
    "p4": "Hierarchical Reflection",
}


def build_mode_preview(
    *,
    mode: str,
    cfg: DefaultsConfig,
    serial_no: str | None,
    host: str,
    port: int,
    log_dir: str | Path,
    explored_knowledge_path: str | None = None,
) -> dict:
    if mode == "p4":
        return build_p4_preview(cfg=cfg,serial_no=serial_no,host=host,port=port,log_dir=log_dir,explored_knowledge_path=explored_knowledge_path)
    if mode == "p2":
        return build_p2_preview(
            cfg=cfg,
            serial_no=serial_no,
            host=host,
            port=port,
            log_dir=log_dir,
            explored_knowledge_path=explored_knowledge_path,
        )
    if mode == "p3":
        return build_p3_preview(
            cfg=cfg,
            serial_no=serial_no,
            host=host,
            port=port,
            log_dir=log_dir,
            explored_knowledge_path=explored_knowledge_path,
        )
    return build_mobileuse_agent_config(
        mode=mode,
        cfg=cfg,
        serial_no=serial_no,
        host=host,
        port=port,
        log_dir=log_dir,
        explored_knowledge_path=explored_knowledge_path,
    )

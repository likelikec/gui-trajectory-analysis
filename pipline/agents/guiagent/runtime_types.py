"""Shared runtime result types for study-owned agent integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeStepResult:
    """Result of one runtime-driven agent step."""

    done: bool
    success: bool | None = None
    action_result: dict[str, Any] | None = None
    answer: str | None = None
    raw_response: str | None = None
    error: str | None = None
    usage: dict[str, int] | None = None
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    step_duration: float | None = None
    started_at: str | None = None
    finished_at: str | None = None
    state_captures: dict[str, Any] | None = None
    terminal_event: str | None = None
    usage_reconciliation: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

"""Generic AndroidWorld wrapper for step-wise session runtimes."""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable

from android_world.agents import base_agent
from android_world.env import interface


logger = logging.getLogger(__name__)


class StudyRuntimeAgent(base_agent.EnvironmentInteractingAgent):
    """Adapts a session with `reset(goal)` and `step()` to AndroidWorld."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        session_factory: Callable[[str], object],
        name: str,
        timeout_seconds: int | None = None,
    ):
        super().__init__(env, name)
        self.session_factory = session_factory
        self.timeout_seconds = timeout_seconds
        self.started_at = time.time()
        self.goal = ""
        self.session = None

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self.env.hide_automation_ui()
        self.started_at = time.time()
        self.goal = ""
        self.session = None

    def set_max_steps(self, max_steps: int) -> None:
        super().set_max_steps(max_steps)
        if self.session is not None and hasattr(self.session, "set_max_steps"):
            self.session.set_max_steps(max_steps)

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        if self.goal != goal or self.session is None:
            self.goal = goal
            self.session = self.session_factory(goal)
            if self._max_steps is not None and hasattr(self.session, "set_max_steps"):
                self.session.set_max_steps(self._max_steps)
            self.started_at = time.time()

        if self.timeout_seconds is not None and (time.time() - self.started_at) >= self.timeout_seconds:
            logger.warning("Agent reached configured timeout_seconds=%s.", self.timeout_seconds)
            return base_agent.AgentInteractionResult(
                True,
                {
                    "step_data": None,
                    "error": f"Reached timeout_seconds={self.timeout_seconds}",
                    "terminal_event": "timeout",
                },
            )

        try:
            result = self.session.step()
        except Exception as exc:
            traceback.print_exc()
            return base_agent.AgentInteractionResult(
                True,
                {"step_data": None, "error": str(exc), "terminal_event": "exception"},
            )

        if getattr(result, "answer", None):
            self.env.interaction_cache = result.answer

        return base_agent.AgentInteractionResult(
            bool(result.done),
            {
                "step_data": result,
                "error": getattr(result, "error", None),
                "terminal_event": getattr(result, "terminal_event", None),
            },
        )

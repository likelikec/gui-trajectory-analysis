"""MobileUse-to-AndroidWorld wrapper used by the study CLI."""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mobile_use
from android_world.agents import base_agent
from android_world.env import interface
from study.runtime_metrics import aggregate_usage, normalize_usage


logger = logging.getLogger(__name__)


_MULTI_AGENT_STAGE_ATTRS = (
    "planner",
    "operator",
    "note_taker",
    "reflector",
    "progressor",
    "trajectory_reflector",
    "answer_agent",
    "global_reflector",
)


def _response_text(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    return normalize_usage(usage)


def _model_identity(vlm: Any, response: Any = None) -> dict[str, Any]:
    requested = getattr(vlm, "requested_model_name", getattr(vlm, "model_name", None))
    resolved = getattr(response, "model", None) or getattr(vlm, "last_resolved_model", None)
    return {
        "model_name": resolved or requested,
        "requested_model_name": requested,
        "resolved_model_name": resolved,
        "resolved_model_source": "provider_response.model" if resolved else None,
        "resolved_model_missing_reason": None if resolved else "provider_response_model_field_missing",
    }


def _has_token_usage(call_history: Sequence[Any] | None) -> bool:
    for call in call_history or []:
        usage = call.get("usage_normalized") if isinstance(call, dict) else None
        if not isinstance(usage, dict):
            usage = normalize_usage(call.get("usage") if isinstance(call, dict) else getattr(call, "usage", None))
        if usage.get("usage_present"):
            return True
    return False


def _canonical_mobileuse_call_history(
    existing_history: Sequence[Any] | None,
    collected_history: Sequence[Any] | None,
    *,
    prefer_collected: bool,
) -> Sequence[Any] | None:
    """Select one non-duplicated call history for the canonical step record.

    MultiAgent sub-agents can call the VLM while building a prompt, for example
    to select an app and extract proactive knowledge.  Those calls are visible
    to the adapter-level recorder but not to MultiAgent's explicit per-stage
    history.  For MultiAgent runs the adapter history is therefore the complete
    source of truth.  Single-agent compatibility keeps the previous fallback.
    """

    if collected_history and (prefer_collected or not _has_token_usage(existing_history)):
        return list(collected_history)
    return existing_history


def _sum_usage(call_history: Sequence[Any] | None) -> dict[str, Any]:
    return aggregate_usage(call_history or [])


def _jsonable(value: Any) -> Any:
    try:
        import dataclasses

        if dataclasses.is_dataclass(value):
            return dataclasses.asdict(value)
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _action_coordinate_summary(action: Any) -> dict[str, Any] | None:
    params = getattr(action, "parameters", None)
    if not isinstance(params, dict):
        return None
    summary: dict[str, Any] = {}
    if "coordinate" in params:
        summary["coordinate"] = _jsonable(params.get("coordinate"))
    if "coordinate2" in params:
        summary["coordinate2"] = _jsonable(params.get("coordinate2"))
    return summary or None


def _action_requires_execution(action: Any) -> bool:
    name = str(getattr(action, "name", "") or "").lower()
    return bool(
        name
        and any(
            token in name
            for token in (
                "click",
                "tap",
                "long_press",
                "swipe",
                "type",
                "open",
                "home",
                "back",
                "key",
                "wait",
            )
        )
    )


def _synthesized_action_execution(step_data: Any, env_wrapper: Any) -> dict[str, Any] | None:
    action = getattr(step_data, "action", None)
    if action is None or not _action_requires_execution(action):
        return None
    action_history = getattr(env_wrapper, "action_history", None)
    skip_reason = getattr(step_data, "skip_reflector_reason", None)
    was_skipped = bool(getattr(step_data, "skip_reflector", False) or skip_reason)
    action_name = str(getattr(action, "name", "") or "")
    params = getattr(action, "parameters", None)
    command = ["mobile-use", "skipped-action" if was_skipped else "record-missing", action_name]
    note = (
        f"Action was not executed by mobile-use pre-reflection: {skip_reason}"
        if was_skipped
        else "No environment action record was emitted for this declared action; synthesized metadata preserves the audit trail."
    )
    return {
        "sequence": len(action_history) + 1 if isinstance(action_history, list) else None,
        "started_at": _utcnow_iso(),
        "finished_at": _utcnow_iso(),
        "status": "skipped" if was_skipped else "record_missing",
        "executor": "mobile_use_agent",
        "command_kind": action_name,
        "requested_action": {
            "name": action_name,
            "parameters": _jsonable(params or {}),
            "repr": str(action),
        },
        "actual_coordinates": _action_coordinate_summary(action),
        "command": command,
        "command_text": " ".join(command),
        "note": note,
    }


def _collect_runtime_knowledge(agent) -> list[str] | None:
    knowledge: list[str] = []
    for attr in ("operator", "answer_agent"):
        sub_agent = getattr(agent, attr, None)
        if sub_agent is None:
            continue
        retrieved = getattr(sub_agent, "retrieved_knowledge", None)
        explored = getattr(sub_agent, "explored_knowledge", None)
        if retrieved:
            knowledge.append(retrieved)
        if explored:
            knowledge.append(explored)
    return knowledge or None


class StudyMobileUseAgent(base_agent.EnvironmentInteractingAgent):
    """Adapts a mobile-use agent to AndroidWorld's EnvironmentInteractingAgent."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        agent: mobile_use.Agent,
        name: str = "StudyMobileUse",
        timeout_seconds: int | None = None,
    ):
        super().__init__(env, name)
        self.agent = agent
        self.timeout_seconds = timeout_seconds
        self.started_at = time.time()
        self._active_vlm_call_history: list[Any] | None = None
        self._patched_vlm_ids: set[int] = set()
        self.agent.reset()

    def _patch_multiagent_vlm_recorders(self) -> None:
        candidates: list[tuple[str, Any]] = [("action", self.agent)]
        if getattr(self.agent, "agents", None):
            candidates.extend(
                (stage, getattr(self.agent, stage, None))
                for stage in _MULTI_AGENT_STAGE_ATTRS
            )
        for stage, sub_agent in candidates:
            vlm = getattr(sub_agent, "vlm", None)
            predict = getattr(vlm, "predict", None)
            if vlm is None or not callable(predict):
                continue
            vlm_id = id(vlm)
            if vlm_id in self._patched_vlm_ids:
                continue
            self._patched_vlm_ids.add(vlm_id)
            original_predict = predict

            def recorded_predict(messages, *args, _stage=stage, _vlm=vlm, _original=original_predict, **kwargs):
                started_at = _utcnow_iso()
                started_perf = time.perf_counter()
                call_history = self._active_vlm_call_history
                try:
                    response = _original(messages, *args, **kwargs)
                except Exception as exc:
                    if call_history is not None:
                        call_history.append(
                            {
                                "call_id": f"step_{getattr(self.agent, 'curr_step_idx', 0) + 1:03d}_call_{len(call_history) + 1:03d}",
                                "messages": messages,
                                "response": "",
                                "stage": _stage,
                                "usage": None,
                                "usage_normalized": normalize_usage(None),
                                **_model_identity(_vlm),
                                "started_at": started_at,
                                "finished_at": _utcnow_iso(),
                                "duration_seconds": time.perf_counter() - started_perf,
                                "error": f"{type(exc).__name__}: {exc}",
                                "logprobs_requested": getattr(_vlm, "last_logprobs_requested", None),
                                "logprobs_returned": getattr(_vlm, "last_logprobs_returned", None),
                                "logprobs_missing_reason": getattr(_vlm, "last_logprobs_missing_reason", None),
                            }
                        )
                    raise
                if call_history is not None:
                    try:
                        if hasattr(_vlm, "build_call_record"):
                            record = _vlm.build_call_record(messages, response, stage=_stage)
                        else:
                            record = {
                                "messages": messages,
                                "response": _response_text(response),
                                "stage": _stage,
                                "usage": _usage_dict(response),
                                "usage_normalized": _usage_dict(response),
                                **_model_identity(_vlm, response),
                            }
                        if isinstance(record, dict):
                            record.setdefault(
                                "call_id",
                                f"step_{getattr(self.agent, 'curr_step_idx', 0) + 1:03d}_call_{len(call_history) + 1:03d}",
                            )
                            record.setdefault("started_at", started_at)
                            record.setdefault("finished_at", _utcnow_iso())
                            record.setdefault("duration_seconds", time.perf_counter() - started_perf)
                        call_history.append(record)
                    except Exception:
                        logger.exception("Failed to capture mobile-use VLM call for stage=%s.", _stage)
                return response

            setattr(vlm, "predict", recorded_predict)

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self.env.hide_automation_ui()
        self.started_at = time.time()
        self.agent.reset()

    def set_max_steps(self, max_steps: int) -> None:
        super().set_max_steps(max_steps)
        if max_steps > 0:
            self.agent.max_steps = max_steps

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        if self.agent.goal != goal:
            self.agent.reset(goal=goal)
            self.started_at = time.time()

        if self.timeout_seconds is not None and (time.time() - self.started_at) >= self.timeout_seconds:
            logger.warning("Agent reached configured timeout_seconds=%s.", self.timeout_seconds)
            self.agent.status = mobile_use.AgentStatus.FAILED
            self.agent.episode_data.status = self.agent.status.value
            self.agent.episode_data.message = f"Reached timeout_seconds={self.timeout_seconds}"
            return base_agent.AgentInteractionResult(
                True, {"step_data": None, "terminal_event": "timeout"}
            )

        if self.agent.curr_step_idx >= self.agent.max_steps:
            logger.warning("Agent reached configured max_steps=%s.", self.agent.max_steps)
            self.agent.status = mobile_use.AgentStatus.FAILED
            self.agent.episode_data.status = self.agent.status.value
            self.agent.episode_data.message = f"Reached max_steps={self.agent.max_steps}"
            return base_agent.AgentInteractionResult(
                True, {"step_data": None, "terminal_event": "max_steps_reached"}
            )

        answer = None
        step_data = None
        collected_call_history: list[Any] = []
        env_wrapper = getattr(self.agent, "env", None)
        action_start_index = 0
        step_started_perf = time.perf_counter()
        try:
            self._patch_multiagent_vlm_recorders()
            self._active_vlm_call_history = collected_call_history
            action_history = getattr(env_wrapper, "action_history", None)
            action_start_index = len(action_history) if isinstance(action_history, list) else 0
            if hasattr(env_wrapper, "last_action_record"):
                env_wrapper.last_action_record = None
            step_data = self.agent.step()
            if step_data is not None:
                # Keep timing at the adapter boundary.  This records the
                # actual agent-step wall time without changing the agent's
                # planning or action order.
                setattr(step_data, "step_duration", time.perf_counter() - step_started_perf)
                existing_history = getattr(step_data, "vlm_call_history", None)
                canonical_history = _canonical_mobileuse_call_history(
                    existing_history,
                    collected_call_history,
                    prefer_collected=bool(getattr(self.agent, "agents", None)),
                )
                if canonical_history is not None:
                    step_data.vlm_call_history = canonical_history
                step_data.knowledge = _collect_runtime_knowledge(self.agent)
                action_history = getattr(env_wrapper, "action_history", None)
                if isinstance(action_history, list):
                    action_executions = action_history[action_start_index:]
                    if not action_executions:
                        synthesized = _synthesized_action_execution(step_data, env_wrapper)
                        if synthesized is not None:
                            action_executions = [synthesized]
                    step_data.action_executions = action_executions
                    step_data.action_execution = action_executions[-1] if action_executions else None
            answer = getattr(step_data, "answer", None)
        except Exception as exc:
            logger.info("Some error happened during the mobile-use agent run.")
            traceback.print_exc()
            self.agent.status = mobile_use.AgentStatus.FAILED
            self.agent.episode_data.status = self.agent.status.value
            self.agent.episode_data.message = str(exc)
            env_wrapper = getattr(self.agent, "env", None)
            action_history = getattr(env_wrapper, "action_history", None)
            action_executions = action_history[action_start_index:] if isinstance(action_history, list) else []
            screenshot_path = None
            xml_path = None
            try:
                state = env_wrapper.get_state()
                pixels = getattr(state, "pixels", None)
                dom_tree = getattr(state, "dom_tree", None)
                episode_dir = getattr(self.agent, "_episode_log_dir", None)
                if episode_dir:
                    failure_stem = f"failure_step_{int(getattr(self.agent, 'curr_step_idx', 0)) + 1:03d}_curr_env"
                    if pixels is not None:
                        screenshot_path = str(Path(episode_dir) / f"{failure_stem}.png")
                        Path(screenshot_path).parent.mkdir(parents=True, exist_ok=True)
                        pixels.save(screenshot_path)
                    if dom_tree:
                        xml_path = str(Path(episode_dir) / f"{failure_stem}.xml")
                        Path(xml_path).parent.mkdir(parents=True, exist_ok=True)
                        Path(xml_path).write_text(str(dom_tree), encoding="utf-8")
            except Exception:
                logger.warning("Failed to capture mobile-use failure state.", exc_info=True)
            if screenshot_path:
                for call in collected_call_history:
                    if isinstance(call, dict) and not call.get("screenshot"):
                        call["screenshot"] = screenshot_path
            if xml_path:
                for call in collected_call_history:
                    if isinstance(call, dict) and not call.get("xml"):
                        call["xml"] = xml_path
            fallback_step_data = {
                "step_idx": getattr(self.agent, "curr_step_idx", 0),
                "vlm_call_history": collected_call_history,
                "llm_calls": collected_call_history,
                "usage": _sum_usage(collected_call_history),
                "knowledge": _collect_runtime_knowledge(self.agent),
                "reflect_on_demand": getattr(self.agent, "reflect_on_demand", None),
                "skip_reflector": None,
                "skip_reflector_reason": None,
                "action_execution": action_executions[-1] if action_executions else None,
                "action_executions": action_executions,
                "screenshots": {"before": screenshot_path} if screenshot_path else {},
                "xml": xml_path,
                "step_duration": time.perf_counter() - step_started_perf,
                "error": str(exc),
                "done": True,
            }
            return base_agent.AgentInteractionResult(
                True,
                {"step_data": fallback_step_data, "terminal_event": "exception"},
            )
        finally:
            self._active_vlm_call_history = None

        self.agent.episode_data.num_steps = self.agent.curr_step_idx + 1
        self.agent.episode_data.status = self.agent.status.value if self.agent.status else None

        if answer is not None:
            logger.info("Agent interaction cache is updated: %s", answer)
            self.env.interaction_cache = answer

        if self.agent.status == mobile_use.AgentStatus.FINISHED:
            logger.info("Agent indicates task is done.")
            self.agent.episode_data.message = "Agent indicates task is done."
            return base_agent.AgentInteractionResult(
                True,
                {"step_data": step_data, "terminal_event": "agent_indicated_done"},
            )

        if self.agent.state == mobile_use.AgentState.CALLUSER:
            logger.warning("CALLUSER is not supported in AndroidWorld evaluation.")
            self.agent.episode_data.message = "CALLUSER is not supported in AndroidWorld evaluation."
            return base_agent.AgentInteractionResult(
                True, {"step_data": step_data, "terminal_event": "call_user"}
            )

        self.agent.curr_step_idx += 1
        logger.info("Agent indicates one step is done.")
        return base_agent.AgentInteractionResult(False, {"step_data": step_data})

"""Output-preserving wrapper for Mobile-Agent-v3 in AndroidWorld runs."""

from __future__ import annotations

import copy
import json
import logging
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from android_world.agents import base_agent
from study.runtime_metrics import aggregate_usage, capture_state_artifacts, normalize_usage


logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex}"


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):
        return getattr(value, "value")
    return str(value)


def _sanitize_messages(messages: Any) -> Any:
    sanitized = copy.deepcopy(messages or [])
    if not isinstance(sanitized, list):
        return sanitized
    for message in sanitized:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, dict) and str(image_url.get("url", "")).startswith("data:image/"):
                item["image_url"] = {"url": "<inline_image>"}
    return sanitized


def _response_text(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    return normalize_usage(usage)


def _model_identity(vllm: Any, response: Any = None) -> dict[str, Any]:
    requested = getattr(
        vllm,
        "requested_model_name",
        getattr(vllm, "model", getattr(vllm, "model_name", None)),
    )
    resolved = getattr(response, "model", None) or getattr(vllm, "last_resolved_model", None)
    return {
        "model_name": resolved or requested,
        "requested_model_name": requested,
        "resolved_model_name": resolved,
        "resolved_model_source": "provider_response.model" if resolved else None,
        "resolved_model_missing_reason": None if resolved else "provider_response_model_field_missing",
    }


def _usage_from_call(call: Any) -> dict[str, Any]:
    if isinstance(call, dict) and isinstance(call.get("usage_normalized"), dict):
        return dict(call["usage_normalized"])
    usage = getattr(call, "usage", None)
    if usage is None and isinstance(call, dict):
        usage = call.get("usage")
    return normalize_usage(usage)


def _sum_usage(call_history: Sequence[Any]) -> dict[str, Any]:
    return aggregate_usage(call_history)


def _sum_usage_by_stage(call_history: Sequence[Any]) -> dict[str, dict[str, int]]:
    totals_by_stage: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[Any]] = {}
    for call in call_history:
        stage = getattr(call, "stage", None)
        if stage is None and isinstance(call, dict):
            stage = call.get("stage")
        stage_name = str(stage or "unknown")
        grouped.setdefault(stage_name, []).append(call)
    for stage_name, calls in grouped.items():
        totals_by_stage[stage_name] = aggregate_usage(calls)
    return totals_by_stage


def _has_token_usage(call_history: Sequence[Any] | None) -> bool:
    return bool(call_history) and any(_usage_from_call(call).get("usage_present") for call in call_history)


def _ensure_call_ids(call_history: Sequence[Any] | None) -> list[Any]:
    """Add identity only when the underlying P3 call record omitted it."""
    normalized: list[Any] = []
    for raw_call in call_history or []:
        call = _to_jsonable(raw_call)
        if isinstance(call, dict) and not call.get("call_id"):
            call["call_id"] = _new_call_id()
        normalized.append(call)
    return normalized


def _safe_task_dir_name(name: str, max_len: int = 50) -> str:
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name))
    safe_name = safe_name.replace(" ", "_").strip(" ._")
    return (safe_name or "task")[:max_len]


def _infer_stage(text_prompt: Any) -> str:
    text = str(text_prompt or "")
    lowered = text.lower()
    if "make a high-level plan" in lowered or "track progress and devise high-level plans" in lowered:
        return "manager"
    if "decide the next action" in lowered or "atomic actions" in lowered:
        return "operator"
    if "important notes" in lowered and "take notes" in lowered:
        return "notekeeper"
    if "action reflection" in lowered or "outcome" in lowered and "error description" in lowered:
        return "reflector"
    return "unknown"


def _last(values: Any) -> Any:
    if isinstance(values, list) and values:
        return values[-1]
    return None


def _json_action_coordinates(action: Any) -> dict[str, Any] | None:
    x = getattr(action, "x", None)
    y = getattr(action, "y", None)
    if x is not None and y is not None:
        return {"coordinate": [int(x), int(y)]}
    direction = getattr(action, "direction", None)
    if isinstance(direction, (list, tuple)) and len(direction) == 4:
        return {
            "start": [int(direction[0]), int(direction[1])],
            "end": [int(direction[2]), int(direction[3])],
        }
    return None


def _xml_paths_for_screenshots(screenshots: Sequence[str]) -> list[str]:
    xml_paths: list[str] = []
    for screenshot in screenshots or []:
        xml_path = Path(str(screenshot)).with_suffix(".xml")
        if xml_path.exists():
            xml_paths.append(str(xml_path))
    return xml_paths


def _make_execution_record(action: Any, sequence: int) -> dict[str, Any]:
    requested_action = _to_jsonable(action)
    command_text = f"env.execute_action({json.dumps(requested_action, ensure_ascii=False, sort_keys=True)})"
    return {
        "sequence": sequence,
        "started_at": _utcnow_iso(),
        "status": "started",
        "executor": "androidworld",
        "command_kind": str(getattr(action, "action_type", "execute_action")),
        "requested_action": requested_action,
        "actual_coordinates": _json_action_coordinates(action),
        "command": ["env.execute_action", requested_action],
        "command_text": command_text,
        "note": "Captured from Mobile-Agent-v3 call to env.execute_action.",
    }


class OutputRecordingMobileAgentV3(base_agent.EnvironmentInteractingAgent):
    """Wraps MobileAgentV3 without changing its implementation files."""

    def __init__(self, inner: base_agent.EnvironmentInteractingAgent, name: str | None = None):
        super().__init__(inner.env, name or getattr(inner, "name", "study_p3_mobileagentv3_androidworld"))
        self.inner = inner
        self._active_call_history: list[dict[str, Any]] | None = None
        self._predict_mm_patched = False

    @property
    def transition_pause(self) -> float | None:
        return self.inner.transition_pause

    @transition_pause.setter
    def transition_pause(self, transition_pause: float | None) -> None:
        self.inner.transition_pause = transition_pause

    @property
    def env(self):
        return self.inner.env

    @env.setter
    def env(self, env) -> None:
        if hasattr(self, "inner"):
            self.inner.env = env
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def reset(self, go_home: bool = False) -> None:
        self.inner.reset(go_home)

    def set_max_steps(self, max_steps: int) -> None:
        super().set_max_steps(max_steps)
        self.inner.set_max_steps(max_steps)

    def _patch_predict_mm(self) -> None:
        if self._predict_mm_patched:
            return
        vllm = getattr(self.inner, "vllm", None)
        original_predict_mm = getattr(vllm, "predict_mm", None)
        if vllm is None or not callable(original_predict_mm):
            return
        self._predict_mm_patched = True

        def recorded_predict_mm(text_prompt, images, messages=None, _original=original_predict_mm, _vllm=vllm):
            started_at = _utcnow_iso()
            started_perf = time.perf_counter()
            call_history = self._active_call_history
            image_paths = [str(image) for image in images or [] if isinstance(image, (str, Path))]
            xml_paths = _xml_paths_for_screenshots(image_paths)
            try:
                content, payload, raw_response = _original(text_prompt, images, messages=messages)
            except Exception as exc:
                if call_history is not None:
                    call_history.append(
                        {
                            "call_id": _new_call_id(),
                            "prompt": str(text_prompt or ""),
                            "messages": _sanitize_messages(messages),
                            "response": "",
                            "stage": _infer_stage(text_prompt),
                            "usage": None,
                            "usage_normalized": normalize_usage(None),
                            **_model_identity(_vllm),
                            "started_at": started_at,
                            "finished_at": _utcnow_iso(),
                            "duration_seconds": time.perf_counter() - started_perf,
                            "screenshots": image_paths,
                            "xml": xml_paths[-1] if xml_paths else None,
                            "xmls": xml_paths,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                raise
            finished_at = _utcnow_iso()
            duration_seconds = time.perf_counter() - started_perf
            if call_history is not None:
                call_history.append(
                    {
                        "call_id": _new_call_id(),
                        "prompt": str(text_prompt or ""),
                        "messages": _sanitize_messages(payload or messages),
                        "response": content or _response_text(raw_response),
                        "stage": _infer_stage(text_prompt),
                        "usage": _usage_dict(raw_response),
                        "usage_normalized": _usage_dict(raw_response),
                        **_model_identity(_vllm, raw_response),
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "duration_seconds": duration_seconds,
                        "screenshots": image_paths,
                        "xml": xml_paths[-1] if xml_paths else None,
                        "xmls": xml_paths,
                    }
                )
            return content, payload, raw_response

        setattr(vllm, "predict_mm", recorded_predict_mm)

    def _info_pool_dict(self) -> dict[str, Any]:
        info_pool = getattr(self.inner, "info_pool", None)
        value = _to_jsonable(info_pool)
        return value if isinstance(value, dict) else {}

    def _task_output_dir(self, goal: str) -> Path | None:
        output_path = getattr(self.inner, "output_path", None)
        if not output_path:
            return None
        if hasattr(self.inner, "_task_output_dir"):
            try:
                return Path(self.inner._task_output_dir())
            except Exception:
                pass
        task_names = getattr(self.inner, "task_name", {}) or {}
        task_name = task_names.get(goal, goal) if isinstance(task_names, dict) else goal
        return Path(output_path) / _safe_task_dir_name(str(task_name))

    def _screenshots(self, goal: str, step_idx: int, call_history: Sequence[dict[str, Any]]) -> dict[str, str]:
        # The model-input image is retained as a call artifact.  Action-boundary
        # before/after evidence is recorded separately in ``action_boundary_captures``;
        # never infer an after image from the next step's filename.
        screenshots: dict[str, str] = {}
        first_images = call_history[0].get("screenshots") if call_history else None
        if first_images:
            screenshots["before"] = str(first_images[0])
        return screenshots

    def _doms(self, screenshots: dict[str, str]) -> dict[str, str]:
        doms: dict[str, str] = {}
        for key, screenshot in screenshots.items():
            xml_path = Path(screenshot).with_suffix(".xml")
            if xml_path.exists():
                doms[key] = str(xml_path)
        return doms

    def _step_payload(
        self,
        *,
        goal: str,
        step_idx: int,
        done: bool,
        error: str | None,
        started_perf: float,
        call_history: Sequence[dict[str, Any]],
        action_executions: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        info = self._info_pool_dict()
        last_action = info.get("last_action") or _last(info.get("action_history"))
        last_summary = info.get("last_summary") or _last(info.get("summary_history"))
        action_s = json.dumps(last_action, ensure_ascii=False) if isinstance(last_action, dict) else last_action
        answer = None
        if isinstance(last_action, dict) and last_action.get("action") == "answer":
            answer = last_action.get("text")
        screenshots = self._screenshots(goal, step_idx, call_history)
        return {
            "step_idx": step_idx,
            "content": next((call.get("response") for call in reversed(call_history) if call.get("stage") == "operator"), None),
            "thought": info.get("last_action_thought"),
            "action": last_action,
            "action_s": action_s,
            "action_desc": last_summary,
            "summary": last_summary,
            "answer": answer,
            "plan": info.get("plan"),
            "sub_goal": info.get("completed_plan"),
            "progress": info.get("progress_status"),
            "reflection_outcome": _last(info.get("action_outcomes")),
            "reflection_error": _last(info.get("error_descriptions")) or error,
            "vlm_call_history": list(call_history),
            "usage": _sum_usage(call_history),
            "usage_by_stage": _sum_usage_by_stage(call_history),
            "action_execution": action_executions[-1] if action_executions else None,
            "action_executions": list(action_executions),
            "action_boundary_captures": list(getattr(self, "_active_boundary_captures", [])),
            "screenshots": screenshots,
            "doms": self._doms(screenshots),
            "metadata": info,
            "done": done,
            "error": error,
            "step_duration": time.perf_counter() - started_perf,
        }

    def _merge_step_payload(self, existing: Any, fallback: dict[str, Any]) -> dict[str, Any]:
        existing_payload = _to_jsonable(existing)
        if not isinstance(existing_payload, dict):
            return fallback
        merged = {**fallback, **existing_payload}
        existing_history = existing_payload.get("vlm_call_history") or existing_payload.get("llm_calls")
        if existing_history:
            merged["vlm_call_history"] = _ensure_call_ids(existing_history)
        if not _has_token_usage(existing_history):
            merged["vlm_call_history"] = fallback["vlm_call_history"]
            merged["usage"] = fallback["usage"]
            merged["usage_by_stage"] = fallback["usage_by_stage"]
        if not existing_payload.get("screenshots"):
            merged["screenshots"] = fallback["screenshots"]
        if not existing_payload.get("action_executions"):
            merged["action_executions"] = fallback.get("action_executions")
            merged["action_execution"] = fallback.get("action_execution")
        if not existing_payload.get("action_boundary_captures"):
            merged["action_boundary_captures"] = fallback.get("action_boundary_captures", [])
        return merged

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        step_idx = len(getattr(getattr(self.inner, "info_pool", None), "action_history", []) or [])
        self._active_step_idx = step_idx + 1
        self._patch_predict_mm()
        call_history: list[dict[str, Any]] = []
        action_executions: list[dict[str, Any]] = []
        self._active_boundary_captures: list[dict[str, Any]] = []
        started_perf = time.perf_counter()
        self._active_call_history = call_history
        env = getattr(self.inner, "env", None)
        original_execute_action = getattr(env, "execute_action", None)
        execute_action_patched = False
        if callable(original_execute_action):
            def recorded_execute_action(action, *args, _original=original_execute_action, **kwargs):
                record = _make_execution_record(action, len(action_executions) + 1)
                action_executions.append(record)
                capture_root = self._task_output_dir(goal)
                if capture_root is not None:
                    capture_root = capture_root / "runtime_captures"
                else:
                    capture_root = Path(getattr(self.inner, "output_path", ".")) / "runtime_captures"
                try:
                    state = env.get_state(wait_to_stabilize=False)
                except TypeError:
                    state = env.get_state()
                except Exception:
                    state = None
                before_capture = capture_state_artifacts(
                    state,
                    capture_root,
                    step_index=step_idx + 1,
                    phase="before",
                )
                self._active_boundary_captures.append(before_capture)
                record["before_capture"] = before_capture
                try:
                    result = _original(action, *args, **kwargs)
                except Exception as exc:
                    record["finished_at"] = _utcnow_iso()
                    record["status"] = "failed"
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    try:
                        state = env.get_state(wait_to_stabilize=False)
                    except TypeError:
                        state = env.get_state()
                    except Exception:
                        state = None
                    after_capture = capture_state_artifacts(
                        state,
                        capture_root,
                        step_index=step_idx + 1,
                        phase="after",
                    )
                    self._active_boundary_captures.append(after_capture)
                    record["after_capture"] = after_capture
                    raise
                try:
                    state = env.get_state(wait_to_stabilize=False)
                except TypeError:
                    state = env.get_state()
                except Exception:
                    state = None
                after_capture = capture_state_artifacts(
                    state,
                    capture_root,
                    step_index=step_idx + 1,
                    phase="after",
                )
                self._active_boundary_captures.append(after_capture)
                record["after_capture"] = after_capture
                record["finished_at"] = _utcnow_iso()
                record["status"] = "completed"
                if result is not None:
                    record["result"] = _to_jsonable(result)
                return result

            try:
                setattr(env, "execute_action", recorded_execute_action)
                execute_action_patched = True
            except Exception:
                logger.warning("Failed to patch MobileAgentV3 env.execute_action for execution recording.")
        try:
            result = self.inner.step(goal)
        except Exception as exc:
            logger.exception("MobileAgentV3 step failed; preserving collected output before ending episode.")
            payload = self._step_payload(
                goal=goal,
                step_idx=step_idx,
                done=True,
                error=f"{type(exc).__name__}: {exc}",
                started_perf=started_perf,
                call_history=call_history,
                action_executions=action_executions,
            )
            data = self._info_pool_dict()
            data["step_data"] = payload
            data["terminal_event"] = "exception"
            return base_agent.AgentInteractionResult(True, data)
        finally:
            self._active_call_history = None
            if execute_action_patched:
                try:
                    setattr(env, "execute_action", original_execute_action)
                except Exception:
                    logger.warning("Failed to restore MobileAgentV3 env.execute_action after recording.")

        data = dict(result.data) if isinstance(result.data, dict) else {}
        fallback = self._step_payload(
            goal=goal,
            step_idx=step_idx,
            done=bool(result.done),
            error=None,
            started_perf=started_perf,
            call_history=call_history,
            action_executions=action_executions,
        )
        data["step_data"] = self._merge_step_payload(data.get("step_data"), fallback)
        return base_agent.AgentInteractionResult(bool(result.done), data)

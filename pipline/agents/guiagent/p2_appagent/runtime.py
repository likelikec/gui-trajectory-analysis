"""Step-wise wrapper around AppAgent's original deployment runtime."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import ast
from contextlib import nullcontext
import hashlib
import importlib
import json
import os
import re
import sys
import time

import requests

from guiagent.common import make_task_dir, normalize_chat_completions_url
from guiagent.runtime_types import RuntimeStepResult
from study.bootstrap import PROJECT_ROOT
from study.config import DefaultsConfig, get_mode_model_config
from study.knowledge import appagent_doc_path, document_xml_for_appagent, ensure_appagent_knowledge
from study.model_adapters import (
    INSUFFICIENT_BALANCE_MARKER,
    InsufficientBalanceError,
    adapter_for_name,
    is_insufficient_balance_error,
)
from study.runtime_metrics import extract_xml_package_names


_APPAGENT_MODULES = None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_from_payload(payload: dict | None) -> dict[str, int] | None:
    if not isinstance(payload, dict):
        return None
    prompt_tokens = int(payload.get("prompt_tokens") or payload.get("input_tokens") or 0)
    completion_tokens = int(payload.get("completion_tokens") or payload.get("output_tokens") or 0)
    total_tokens = int(payload.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _file_observation_capture(
    observation: dict | None,
    *,
    step_index: int,
    phase: str,
    capture_id: str,
) -> dict:
    """Normalize one controller observation into the canonical capture schema.

    AppAgent's controller already writes PNG and XML from one AndroidWorld
    state.  This adapter only records those paths and hashes; it does not
    recapture or infer a neighboring step.
    """

    observation = observation if isinstance(observation, dict) else {}
    result = {
        "capture_id": capture_id,
        "step_index": step_index,
        "phase": phase,
        "captured_at": observation.get("captured_at") or _utcnow_iso(),
        "status": "missing",
        "screenshot": None,
        "screenshot_sha256": None,
        "xml": None,
        "xml_sha256": None,
        "missing_reasons": [],
        "source": observation.get("source") or "appagent_controller_observation",
        "current_package": observation.get("current_package"),
        "current_package_missing_reason": None,
        "current_activity": observation.get("current_activity"),
        "current_activity_missing_reason": None,
        "observed_package_names": [],
        "observed_package_names_missing_reason": None,
    }
    if result["current_package"] is None:
        result["current_package_missing_reason"] = "controller_observation_did_not_expose_current_package"
    if result["current_activity"] is None:
        result["current_activity_missing_reason"] = "controller_observation_did_not_expose_current_activity"
    if observation.get("error"):
        result["missing_reasons"].append(f"capture_error:{observation['error']}")

    for field in ("screenshot", "xml"):
        raw_path = observation.get(field)
        if not raw_path or str(raw_path) == "ERROR":
            result["missing_reasons"].append(f"{field}_missing")
            continue
        path = Path(str(raw_path))
        if not path.is_file():
            result["missing_reasons"].append(f"{field}_path_missing")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[field] = str(path)
        result[f"{field}_sha256"] = digest.hexdigest()
        if field == "xml":
            try:
                xml_text = path.read_text(encoding="utf-8", errors="replace")
                result["observed_package_names"] = extract_xml_package_names(xml_text)
                if not result["observed_package_names"]:
                    result["observed_package_names_missing_reason"] = "xml_contains_no_package_attributes"
            except OSError as exc:
                result["observed_package_names_missing_reason"] = f"xml_read_error:{type(exc).__name__}"

    if result["xml"] is None and result["observed_package_names_missing_reason"] is None:
        result["observed_package_names_missing_reason"] = "xml_missing"

    if result["screenshot"] and result["xml"]:
        result["status"] = "complete"
    elif result["screenshot"] or result["xml"]:
        result["status"] = "partial"
    result["missing_reasons"] = list(dict.fromkeys(result["missing_reasons"]))
    return result


def _load_appagent_modules():
    global _APPAGENT_MODULES
    if _APPAGENT_MODULES is not None:
        return _APPAGENT_MODULES

    scripts_dir = PROJECT_ROOT / "vendor" / "AppAgent" / "scripts"
    scripts_dir_str = str(scripts_dir)
    if scripts_dir_str not in sys.path:
        sys.path.insert(0, scripts_dir_str)

    prompts = importlib.import_module("prompts")
    model = importlib.import_module("model")
    and_controller = importlib.import_module("and_controller")
    utils = importlib.import_module("utils")
    _APPAGENT_MODULES = {
        "prompts": prompts,
        "model": model,
        "and_controller": and_controller,
        "utils": utils,
    }
    return _APPAGENT_MODULES


def _resolve_docs_dir(explored_knowledge_path: str | None) -> Path | None:
    if not explored_knowledge_path:
        return None
    root = Path(explored_knowledge_path)
    if not root.exists():
        return root
    if root.is_file():
        return root.parent
    for child in ("auto_docs", "demo_docs"):
        candidate = root / child
        if candidate.exists():
            return candidate
    txt_children = list(root.glob("*.txt"))
    if txt_children:
        return root
    return None


def build_p2_preview(
    *,
    cfg: DefaultsConfig,
    serial_no: str | None,
    host: str,
    port: int,
    log_dir: str | Path,
    explored_knowledge_path: str | None = None,
) -> dict:
    model_cfg = get_mode_model_config(cfg, "p2")
    return {
        "runtime": "AppAgent",
        "controller": {
            "serial_no": serial_no,
            "host": host,
            "port": port,
        },
        "model": {
            "api_url": normalize_chat_completions_url(model_cfg.base_url),
            "api_key": model_cfg.api_key,
            "model_name": model_cfg.model_name,
            "max_tokens": model_cfg.max_tokens,
            "temperature": model_cfg.temperature,
            "enable_thinking": model_cfg.enable_thinking,
            "thinking_budget": model_cfg.thinking_budget,
            "max_retry": model_cfg.max_retry,
            "retry_waiting_seconds": model_cfg.retry_waiting_seconds,
        },
        "max_steps": cfg.control.max_steps,
        "request_interval": cfg.executor.request_interval,
        "log_dir": str(log_dir),
        "docs_dir": explored_knowledge_path,
        "action_space": ["tap", "text", "long_press", "swipe", "grid", "open_app", "back", "answer", "finish"],
    }


class AppAgentSession:
    """Stateful AppAgent deployment session for benchmark runs."""

    def __init__(
        self,
        *,
        controller,
        cfg: DefaultsConfig,
        task_root_dir: str | Path,
        explored_knowledge_path: str | None = None,
        app_name: str | None = None,
        package_name: str | None = None,
    ):
        modules = _load_appagent_modules()
        self.prompts = modules["prompts"]
        self.model_module = modules["model"]
        self.and_controller = modules["and_controller"]
        self.utils = modules["utils"]

        model_cfg = get_mode_model_config(cfg, "p2")
        self.model_adapter = adapter_for_name(model_cfg.adapter_name)
        api_url = normalize_chat_completions_url(model_cfg.base_url)
        # Provider-specific thinking fields stay at the transport boundary.
        if model_cfg.adapter_name.lower() == "glm":
            thinking_kwargs = {
                "thinking": {
                    "type": "enabled" if model_cfg.enable_thinking else "disabled"
                }
            }
        elif model_cfg.provider == "poixe":
            thinking_kwargs = {}
        else:
            thinking_kwargs = {
                "enable_thinking": model_cfg.enable_thinking,
                "thinking_budget": model_cfg.thinking_budget,
            }
        self.mllm = self.model_module.OpenAIModel(
            base_url=api_url,
            api_key=model_cfg.api_key,
            model=model_cfg.model_name,
            temperature=model_cfg.temperature,
            max_tokens=model_cfg.max_tokens,
            max_tokens_parameter=(
                "max_completion_tokens"
                if model_cfg.adapter_name.lower() in {"gpt", "gpt-absolute", "gpt-5.6-absolute"}
                else "max_tokens"
            ),
            **thinking_kwargs,
        )
        self.controller = controller
        self.cfg = cfg
        self.task_root_dir = Path(task_root_dir)
        self.task_root_dir.mkdir(parents=True, exist_ok=True)
        self.docs_root = _resolve_docs_dir(explored_knowledge_path)
        self.docs_dir: Path | None = None
        self.app_name = app_name
        self.package_name = package_name

        self.goal = ""
        self.task_dir: Path | None = None
        self.log_path: Path | None = None
        self.round_count = 0
        self.last_act = "None"
        self.grid_on = False
        self.rows = 0
        self.cols = 0
        self.done = False
        self.max_steps = cfg.control.max_steps
        self.max_retry = max(1, int(model_cfg.max_retry))
        self.retry_waiting_seconds = max(0.0, float(model_cfg.retry_waiting_seconds))

    def reset(self, goal: str) -> None:
        self.goal = goal
        self.round_count = 0
        self.last_act = "None"
        self.grid_on = False
        self.rows = 0
        self.cols = 0
        self.done = False
        self.task_dir = make_task_dir(self.task_root_dir, "appagent", goal)
        self.log_path = self.task_dir / "appagent_log.jsonl"
        if (
            self.docs_root is not None
            and self.app_name
            and getattr(self.controller, "supports_appagent_knowledge", True)
        ):
            prepared_dir = ensure_appagent_knowledge(
                controller=self.controller,
                app_name=self.app_name,
                package_name=self.package_name,
                knowledge_root=self.docs_root,
                artifact_root=self.task_dir / "_knowledge_artifacts",
            )
            self.docs_dir = Path(prepared_dir)
        else:
            self.docs_dir = self.docs_root

    def set_max_steps(self, max_steps: int) -> None:
        if max_steps > 0:
            self.max_steps = max_steps

    def _append_log(self, payload: dict) -> None:
        if self.log_path is None:
            return
        with self.log_path.open("a", encoding="utf-8") as logfile:
            logfile.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _area_to_xy(self, area: int, subarea: str, width: int, height: int) -> tuple[int, int]:
        area -= 1
        row, col = area // self.cols, area % self.cols
        x_0, y_0 = col * (width // self.cols), row * (height // self.rows)
        if subarea == "top-left":
            return x_0 + (width // self.cols) // 4, y_0 + (height // self.rows) // 4
        if subarea == "top":
            return x_0 + (width // self.cols) // 2, y_0 + (height // self.rows) // 4
        if subarea == "top-right":
            return x_0 + (width // self.cols) * 3 // 4, y_0 + (height // self.rows) // 4
        if subarea == "left":
            return x_0 + (width // self.cols) // 4, y_0 + (height // self.rows) // 2
        if subarea == "right":
            return x_0 + (width // self.cols) * 3 // 4, y_0 + (height // self.rows) // 2
        if subarea == "bottom-left":
            return x_0 + (width // self.cols) // 4, y_0 + (height // self.rows) * 3 // 4
        if subarea == "bottom":
            return x_0 + (width // self.cols) // 2, y_0 + (height // self.rows) * 3 // 4
        if subarea == "bottom-right":
            return x_0 + (width // self.cols) * 3 // 4, y_0 + (height // self.rows) * 3 // 4
        return x_0 + (width // self.cols) // 2, y_0 + (height // self.rows) // 2

    def _fallback_swipe_anchor(self, area: int, direction: str, width: int, height: int) -> tuple[int, int] | None:
        """Handles AppAgent responses that use a screen x-coordinate as anchor."""
        if area < 0 or area > width:
            return None
        x = int(max(0, min(width - 1, area)))
        if direction == "up":
            return x, int(height * 0.82)
        if direction == "down":
            return x, int(height * 0.30)
        return x, int(height * 0.55)

    def _build_ui_docs(self, elem_list) -> str:
        if self.docs_dir is None:
            return ""
        ui_doc = ""
        for i, elem in enumerate(elem_list):
            doc_path = appagent_doc_path(self.docs_dir, elem.uid)
            if not doc_path.exists():
                continue
            ui_doc += f"Documentation of UI element labeled with the numeric tag '{i + 1}':\n"
            doc_content = ast.literal_eval(doc_path.read_text(encoding="utf-8"))
            if doc_content.get("tap"):
                ui_doc += f"This UI element is clickable. {doc_content['tap']}\n\n"
            if doc_content.get("text"):
                ui_doc += (
                    "This UI element can receive text input. "
                    f"The text input is used for the following purposes: {doc_content['text']}\n\n"
                )
            if doc_content.get("long_press"):
                ui_doc += f"This UI element is long clickable. {doc_content['long_press']}\n\n"
            if doc_content.get("v_swipe"):
                ui_doc += (
                    "This element can be swiped directly without tapping. "
                    f"You can swipe vertically on this UI element. {doc_content['v_swipe']}\n\n"
                )
            if doc_content.get("h_swipe"):
                ui_doc += (
                    "This element can be swiped directly without tapping. "
                    f"You can swipe horizontally on this UI element. {doc_content['h_swipe']}\n\n"
                )
        if not ui_doc:
            return ""
        return (
            "\n            You also have access to the following documentations that describes the functionalities of UI "
            "elements you can interact on the screen. These docs are crucial for you to determine the target of your "
            "next action. You should always prioritize these documented elements for interaction:"
            + ui_doc
        )

    def _augment_prompt_for_androidworld(self, prompt: str) -> str:
        """Adds AndroidWorld lifecycle primitives while keeping AppAgent's format."""
        if "open_app(app_name: str)" in prompt:
            return prompt
        extension = (
            '\n6. open_app(app_name: str)\n'
            'This function opens an installed app by name through AndroidWorld. '
            'If the task asks you to open a named app that is not already visible, '
            'prefer this function instead of opening a related app.\n'
            'A simple use case can be open_app("contacts"), which opens the Contacts app.\n\n'
            '7. back()\n'
            'This function presses the Android Back button through AndroidWorld. '
            'Use it to dismiss a non-permission setup screen or pop-up when tapping its buttons fails.\n'
            '\n8. answer(text: str)\n'
            'This function submits a textual answer through AndroidWorld. '
            'Use it instead of FINISH when the task asks you to respond with a value, title, number, True/False, or other text.\n'
            'A simple use case can be answer("False").\n'
        )
        marker = "\n<ui_document>"
        if marker in prompt:
            return prompt.replace(marker, extension + marker, 1)
        return prompt + extension

    def _parse_androidworld_extension_rsp(self, rsp: str):
        try:
            act = re.findall(r"Action: (.*?)$", rsp, re.MULTILINE)[0].strip()
            last_act = re.findall(r"Summary: (.*?)$", rsp, re.MULTILINE)[0]
        except Exception:
            return None
        if act.startswith("open_app("):
            match = re.findall(r"open_app\((.*?)\)", act)
            if not match:
                return None
            app_name = match[0].strip().strip("\"'")
            if not app_name:
                return None
            return ["open_app", app_name, last_act]
        if act in {"back()", "press_back()"}:
            return ["back", last_act]
        if act in {"home()", "press_home()"}:
            return ["home", last_act]
        if act.startswith("answer("):
            match = re.findall(r"answer\((.*?)\)", act)
            if not match:
                return None
            answer_text = match[0].strip().strip("\"'")
            return ["answer", answer_text, last_act]
        return None

    def step(self) -> RuntimeStepResult:
        step_started_at = _utcnow_iso()
        step_started_perf = time.perf_counter()
        llm_calls: list[dict] = []
        before_observation = None
        after_observation = None
        llm_call_attempted = False

        def _result(**kwargs) -> RuntimeStepResult:
            kwargs.setdefault("llm_calls", llm_calls)
            kwargs.setdefault("started_at", step_started_at)
            kwargs.setdefault("finished_at", _utcnow_iso())
            kwargs.setdefault("step_duration", time.perf_counter() - step_started_perf)
            kwargs.setdefault("state_captures", {"before": before_observation, "after": after_observation})
            metadata = kwargs.setdefault("metadata", {})
            metadata.setdefault("llm_call_attempted", llm_call_attempted)
            return RuntimeStepResult(**kwargs)

        if self.done:
            return _result(done=True, success=True, metadata={"message": "Session already finished."})
        if self.task_dir is None:
            raise RuntimeError("AppAgentSession.reset(goal) must be called before step().")
        if self.round_count >= self.max_steps:
            self.done = True
            return _result(
                done=True,
                success=False,
                error=f"Reached max_steps={self.max_steps}",
                action_result={"operation": "finish", "action": "finish", "kwargs": {"message": "Reached max steps."}},
            )

        self.round_count += 1
        prefix = f"step_{self.round_count}"
        capture_observation = getattr(self.controller, "capture_observation", None)
        if callable(capture_observation):
            raw_observation = capture_observation(prefix, self.task_dir)
            before_observation = _file_observation_capture(
                raw_observation,
                step_index=self.round_count,
                phase="before",
                capture_id=f"step_{self.round_count:03d}_before",
            )
        else:
            screenshot_path = self.controller.get_screenshot(prefix, self.task_dir)
            xml_path = self.controller.get_xml(prefix, self.task_dir)
            before_observation = _file_observation_capture(
                {
                    "screenshot": screenshot_path,
                    "xml": xml_path,
                    "source": "appagent_controller_legacy_methods",
                },
                step_index=self.round_count,
                phase="before",
                capture_id=f"step_{self.round_count:03d}_before",
            )
        screenshot_path = before_observation.get("screenshot") or "ERROR"
        xml_path = before_observation.get("xml") or "ERROR"
        if screenshot_path == "ERROR" or xml_path == "ERROR":
            self.done = True
            return _result(done=True, success=False, error="Failed to capture screenshot or XML.")

        width, height = self.controller.get_device_size()
        elem_list = None
        if self.grid_on:
            self.rows, self.cols = self.utils.draw_grid(
                screenshot_path,
                str(self.task_dir / f"{prefix}_grid.png"),
            )
            image = str(self.task_dir / f"{prefix}_grid.png")
            prompt = self.prompts.task_template_grid
        else:
            clickable_list = []
            focusable_list = []
            self.and_controller.traverse_tree(xml_path, clickable_list, "clickable", True)
            self.and_controller.traverse_tree(xml_path, focusable_list, "focusable", True)
            elem_list = clickable_list.copy()
            for elem in focusable_list:
                bbox = elem.bbox
                center = (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
                close = False
                for clickable in clickable_list:
                    clickable_bbox = clickable.bbox
                    clickable_center = (
                        (clickable_bbox[0][0] + clickable_bbox[1][0]) // 2,
                        (clickable_bbox[0][1] + clickable_bbox[1][1]) // 2,
                    )
                    dist = (
                        (abs(center[0] - clickable_center[0]) ** 2 + abs(center[1] - clickable_center[1]) ** 2)
                        ** 0.5
                    )
                    if dist <= 30:
                        close = True
                        break
                if not close:
                    elem_list.append(elem)

            labeled_image = self.task_dir / f"{prefix}_labeled.png"
            self.utils.draw_bbox_multi(screenshot_path, str(labeled_image), elem_list, dark_mode=False)
            image = str(labeled_image)
            if self.docs_dir is not None:
                document_xml_for_appagent(xml_path, self.docs_dir)
            ui_doc = self._build_ui_docs(elem_list)
            prompt = re.sub(r"<ui_document>", ui_doc, self.prompts.task_template)
            prompt = self._augment_prompt_for_androidworld(prompt)

        prompt = re.sub(r"<task_description>", self.goal, prompt)
        prompt = re.sub(r"<last_act>", self.last_act, prompt)
        augment_appagent_prompt = getattr(
            getattr(self, "model_adapter", None),
            "augment_appagent_prompt",
            None,
        )
        if callable(augment_appagent_prompt):
            prompt = augment_appagent_prompt(
                prompt,
                width=width,
                height=height,
                grid_mode=self.grid_on,
            )
        max_attempts = max(1, int(getattr(self, "max_retry", 1)))
        retry_waiting_seconds = max(0.0, float(getattr(self, "retry_waiting_seconds", 0.0)))
        attempt_index = 0
        while True:
            attempt_index += 1
            llm_started_at = _utcnow_iso()
            llm_started_perf = time.perf_counter()
            call_id = f"step_{self.round_count:03d}_call_{len(llm_calls) + 1:03d}"
            # Every HTTP attempt is retained as a separate canonical call.
            # Clearing provider usage prevents a failed attempt from inheriting
            # token data from the preceding successful request.
            self.mllm.last_usage = None
            if hasattr(self.mllm, "last_resolved_model"):
                self.mllm.last_resolved_model = None
            llm_call_attempted = True
            requested_model_name = getattr(
                self.mllm,
                "requested_model_name",
                getattr(self.mllm, "model", None),
            )
            llm_call = {
                "call_id": call_id,
                "step": self.round_count,
                "stage": "action",
                "role": "action",
                "attempt_index": attempt_index,
                "max_attempts": max_attempts,
                "prompt": prompt,
                "messages": [{"role": "user", "content": prompt}],
                "response": None,
                "usage": None,
                "model_name": requested_model_name,
                "requested_model_name": requested_model_name,
                "resolved_model_name": None,
                "resolved_model_source": None,
                "resolved_model_missing_reason": "provider_response_not_received",
                "started_at": llm_started_at,
                "finished_at": None,
                "duration_seconds": None,
                "screenshot": image,
                "xml": xml_path,
            }
            try:
                network_context = getattr(
                    getattr(self, "model_adapter", None),
                    "network_context",
                    None,
                )
                request_context = (
                    network_context(getattr(self.mllm, "base_url", ""))
                    if callable(network_context)
                    else nullcontext()
                )
                with request_context:
                    status, rsp = self.mllm.get_model_response(prompt, [image])
                retryable_response = getattr(
                    getattr(self, "model_adapter", None),
                    "is_retryable_provider_response",
                    None,
                )
                if (
                    not status
                    and callable(retryable_response)
                    and retryable_response(rsp)
                    and attempt_index < max_attempts
                ):
                    llm_call.update(
                        {
                            "call_status": "provider_error",
                            "response": rsp,
                            "finished_at": _utcnow_iso(),
                            "duration_seconds": time.perf_counter() - llm_started_perf,
                            "error": str(rsp),
                            "resolved_model_missing_reason": "provider_error_response_missing_model",
                            "retryable_provider_error": True,
                            "retry_scheduled": True,
                            "retry_waiting_seconds": retry_waiting_seconds,
                            "usage_normalized": {
                                "prompt_tokens": None,
                                "completion_tokens": None,
                                "total_tokens": None,
                                "usage_present": False,
                                "usage_missing_reason": "provider_error_response",
                                "total_tokens_source": None,
                                "usage_consistency": "not_checkable",
                                "usage_raw": None,
                            },
                        }
                    )
                    llm_calls.append(llm_call)
                    self._append_log(llm_call)
                    time.sleep(retry_waiting_seconds)
                    continue
                break
            except Exception as exc:
                retryable = isinstance(exc, requests.exceptions.RequestException)
                retry_scheduled = retryable and attempt_index < max_attempts
                llm_call.update(
                    {
                        "call_status": "exception",
                        "finished_at": _utcnow_iso(),
                        "duration_seconds": time.perf_counter() - llm_started_perf,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "error": str(exc),
                        "resolved_model_missing_reason": "call_failed_before_provider_response",
                        "retryable_transport_exception": retryable,
                        "retry_scheduled": retry_scheduled,
                        "retry_waiting_seconds": retry_waiting_seconds if retry_scheduled else None,
                        "usage_normalized": {
                            "prompt_tokens": None,
                            "completion_tokens": None,
                            "total_tokens": None,
                            "usage_present": False,
                            "usage_missing_reason": "call_failed_before_response",
                            "total_tokens_source": None,
                            "usage_consistency": "not_checkable",
                            "usage_raw": None,
                        },
                    }
                )
                llm_calls.append(llm_call)
                self._append_log(llm_call)
                if retry_scheduled:
                    time.sleep(retry_waiting_seconds)
                    continue
                self.done = True
                return _result(
                    done=True,
                    success=False,
                    error=str(exc),
                    terminal_event="exception",
                    usage=None,
                )

        usage = _usage_from_payload(getattr(self.mllm, "last_usage", None))
        resolved_model_name = getattr(self.mllm, "last_resolved_model", None)
        llm_call.update(
            {
                "call_status": "completed" if status else "provider_error",
                "response": rsp,
                "error": None if status else str(rsp),
                "usage": usage,
                "model_name": resolved_model_name or requested_model_name,
                "resolved_model_name": resolved_model_name,
                "resolved_model_source": "provider_response.model" if resolved_model_name else None,
                "resolved_model_missing_reason": (
                    None if resolved_model_name else "provider_response_model_field_missing"
                ),
                "finished_at": _utcnow_iso(),
                "duration_seconds": time.perf_counter() - llm_started_perf,
            }
        )
        provider_response = rsp
        if status:
            normalize_response = getattr(
                getattr(self, "model_adapter", None),
                "normalize_appagent_response",
                None,
            )
            if callable(normalize_response):
                rsp = normalize_response(provider_response, screen_size=(width, height))
                if rsp != provider_response:
                    llm_call["normalized_response"] = rsp
                    llm_call["response_normalization"] = "model_adapter"
        llm_calls.append(llm_call)
        self._append_log(llm_call)
        if not status:
            if is_insufficient_balance_error(rsp):
                print(INSUFFICIENT_BALANCE_MARKER, flush=True)
                raise InsufficientBalanceError()
            self.done = True
            return _result(
                done=True,
                success=False,
                error=str(rsp),
                terminal_event="provider_error",
                usage=usage,
                raw_response=str(provider_response),
            )

        parsed = None
        parse_appagent_action = getattr(
            getattr(self, "model_adapter", None),
            "parse_appagent_action",
            None,
        )
        if callable(parse_appagent_action):
            parsed = parse_appagent_action(
                rsp,
                grid_mode=self.grid_on,
                screen_size=(width, height),
            )
        if parsed is None and not self.grid_on:
            parsed = self._parse_androidworld_extension_rsp(rsp)
        if parsed is None:
            parsed = self.model_module.parse_grid_rsp(rsp) if self.grid_on else self.model_module.parse_explore_rsp(rsp)
        if parsed[0] == "FINISH":
            self.done = True
            return _result(
                done=True,
                success=True,
                raw_response=provider_response,
                usage=usage,
                action_result={"operation": "finish", "action": "finish", "kwargs": {"message": "AppAgent finished."}},
            )
        if parsed[0] == "ERROR":
            self.done = True
            return _result(
                done=True,
                success=False,
                raw_response=provider_response,
                error="Failed to parse AppAgent response.",
                usage=usage,
                metadata={"normalized_response": rsp} if rsp != provider_response else {},
            )

        act_name = parsed[0]
        self.last_act = parsed[-1]
        params = parsed[:-1]
        action_result = None
        exec_status = "OK"

        def _get_element_bbox(area: int):
            if elem_list is None or area < 1 or area > len(elem_list):
                return None
            return elem_list[area - 1].bbox

        if act_name == "tap":
            _, area = params
            bbox = _get_element_bbox(area)
            if bbox is None:
                self.done = True
                return _result(done=True, success=False, raw_response=provider_response, error=f"Invalid tap element index: {area}", usage=usage)
            tl, br = bbox
            x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
            exec_status = self.controller.tap(x, y)
            action_result = {"operation": "do", "action": "Tap", "kwargs": {"element": [x, y]}}
        elif act_name == "text":
            _, input_str = params
            exec_status = self.controller.text(input_str)
            action_result = {"operation": "do", "action": "Type", "kwargs": {"text": input_str}}
        elif act_name == "long_press":
            _, area = params
            bbox = _get_element_bbox(area)
            if bbox is None:
                self.done = True
                return _result(done=True, success=False, raw_response=provider_response, error=f"Invalid long_press element index: {area}", usage=usage)
            tl, br = bbox
            x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
            exec_status = self.controller.long_press(x, y)
            action_result = {"operation": "do", "action": "Long Press", "kwargs": {"element": [x, y]}}
        elif act_name == "tap_coordinate":
            _, point = params
            x, y = point
            exec_status = self.controller.tap(x, y)
            action_result = {
                "operation": "do",
                "action": "Tap",
                "kwargs": {"element": [x, y], "coordinate_source": "model_absolute_pixel"},
            }
        elif act_name == "long_press_coordinate":
            _, point = params
            x, y = point
            exec_status = self.controller.long_press(x, y)
            action_result = {
                "operation": "do",
                "action": "Long Press",
                "kwargs": {"element": [x, y], "coordinate_source": "model_absolute_pixel"},
            }
        elif act_name == "swipe":
            _, area, swipe_dir, dist = params
            bbox = _get_element_bbox(area)
            if bbox is None:
                anchor = self._fallback_swipe_anchor(area, swipe_dir, width, height)
                if anchor is None:
                    self.last_act = f"Swipe action skipped because element index {area} is invalid."
                    return _result(
                        done=False,
                        success=False,
                        raw_response=provider_response,
                        error=f"Invalid swipe element index: {area}",
                        usage=usage,
                        action_result={
                            "operation": "skip",
                            "action": "Swipe",
                            "kwargs": {"element": area, "direction": swipe_dir, "dist": dist},
                        },
                    )
                x, y = anchor
            else:
                tl, br = bbox
                x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
            exec_status = self.controller.swipe(x, y, swipe_dir, dist)
            action_result = {
                "operation": "do",
                "action": "Swipe",
                "kwargs": {"element": [x, y], "direction": swipe_dir, "dist": dist},
            }
        elif act_name == "grid":
            self.grid_on = True
            action_result = {"operation": "do", "action": "Grid"}
        elif act_name == "open_app":
            _, app_name = params
            exec_status = self.controller.launch_app(app_name)
            action_result = {"operation": "do", "action": "Open App", "kwargs": {"app_name": app_name}}
        elif act_name == "back":
            exec_status = self.controller.back()
            action_result = {"operation": "do", "action": "Back"}
        elif act_name == "home":
            exec_status = self.controller.home()
            action_result = {"operation": "do", "action": "Home"}
        elif act_name == "answer":
            _, answer_text = params
            exec_status = self.controller.answer(answer_text)
            self.done = True
            action_result = {"operation": "finish", "action": "Answer", "kwargs": {"text": answer_text}}
        elif act_name in {"tap_grid", "long_press_grid"}:
            _, area, subarea = params
            x, y = self._area_to_xy(area, subarea, width, height)
            if act_name == "tap_grid":
                exec_status = self.controller.tap(x, y)
                action_result = {"operation": "do", "action": "Tap", "kwargs": {"element": [x, y]}}
            else:
                exec_status = self.controller.long_press(x, y)
                action_result = {"operation": "do", "action": "Long Press", "kwargs": {"element": [x, y]}}
        elif act_name == "swipe_grid":
            _, start_area, start_subarea, end_area, end_subarea = params
            start_x, start_y = self._area_to_xy(start_area, start_subarea, width, height)
            end_x, end_y = self._area_to_xy(end_area, end_subarea, width, height)
            exec_status = self.controller.swipe_precise((start_x, start_y), (end_x, end_y))
            action_result = {
                "operation": "do",
                "action": "Swipe",
                "kwargs": {"start": [start_x, start_y], "end": [end_x, end_y]},
            }
        elif act_name == "swipe_coordinate":
            _, start, end = params
            start_x, start_y = start
            end_x, end_y = end
            exec_status = self.controller.swipe_precise(
                (start_x, start_y),
                (end_x, end_y),
            )
            action_result = {
                "operation": "do",
                "action": "Swipe",
                "kwargs": {
                    "start": [start_x, start_y],
                    "end": [end_x, end_y],
                    "coordinate_source": "model_absolute_pixel",
                },
            }

        if act_name != "grid":
            self.grid_on = False

        if act_name != "grid":
            if callable(capture_observation):
                raw_after_observation = capture_observation(f"{prefix}_after", self.task_dir)
            else:
                raw_after_observation = {
                    "screenshot": self.controller.get_screenshot(f"{prefix}_after", self.task_dir),
                    "xml": self.controller.get_xml(f"{prefix}_after", self.task_dir),
                    "source": "appagent_controller_legacy_methods",
                }
            after_observation = _file_observation_capture(
                raw_after_observation,
                step_index=self.round_count,
                phase="after",
                capture_id=f"step_{self.round_count:03d}_after",
            )

        if exec_status == "ERROR":
            self.done = True
            return _result(done=True, success=False, raw_response=provider_response, error=f"Execution failed for action={act_name}", usage=usage)

        return _result(
            done=self.done,
            success=True if self.done else None,
            raw_response=provider_response,
            usage=usage,
            action_result=action_result,
            answer=action_result.get("kwargs", {}).get("text") if self.done and action_result else None,
            metadata={
                "step": self.round_count,
                "last_act": self.last_act,
                "grid_on": self.grid_on,
                "normalized_response": rsp if rsp != provider_response else None,
                "state_captures": {
                    "before": before_observation,
                    "after": after_observation,
                },
            },
        )

"""AndroidWorld-native environment/controller facades for study agents."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
import hashlib
from typing import Any

from PIL import Image

from android_world.agents.m3a import _generate_ui_elements_description_list
from android_world.agents import new_json_action as aw_json_action
from android_world.env import adb_utils as aw_adb_utils
from android_world.env import interface

from mobile_use.schema.schema import Action, EnvState


logger = logging.getLogger(__name__)


_COORDINATE_ACTIONS = {
    aw_json_action.CLICK,
    aw_json_action.LONG_PRESS,
    getattr(aw_json_action, "DOUBLE_TAP", "double_tap"),
}


def _decode_generic_output(response) -> str:
    output = getattr(getattr(response, "generic", None), "output", b"")
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace").strip()
    return str(output or "").strip()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _action_coordinate_summary(action: Action) -> dict[str, Any] | None:
    params = action.parameters or {}
    if action.name in {"click", "long_press"} and "coordinate" in params:
        return {"coordinate": [int(params["coordinate"][0]), int(params["coordinate"][1])]}
    if action.name == "swipe" and "coordinate" in params and "coordinate2" in params:
        return {
            "start": [int(params["coordinate"][0]), int(params["coordinate"][1])],
            "end": [int(params["coordinate2"][0]), int(params["coordinate2"][1])],
        }
    return None


def _save_pixels(pixels, save_path: str | Path) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(save_path)
    return str(save_path)


def _extract_package_name(activity_name: str) -> str | None:
    if not activity_name:
        return None
    return aw_adb_utils.extract_package_name(activity_name)


def _bbox_contains(bbox, x: int, y: int) -> bool:
    if bbox is None:
        return False
    return bbox.x_min <= x <= bbox.x_max and bbox.y_min <= y <= bbox.y_max


def _element_interaction_score(element) -> int:
    score = 0
    for attr in (
        "is_clickable",
        "is_focusable",
        "is_checkable",
        "is_editable",
        "is_long_clickable",
        "is_scrollable",
    ):
        if getattr(element, attr, False):
            score += 2
    for attr in ("text", "content_description", "resource_name", "resource_id"):
        if getattr(element, attr, None):
            score += 1
    if getattr(element, "is_enabled", True) is False:
        score -= 2
    if getattr(element, "is_visible", True) is False:
        score -= 3
    return score


def _best_element_score_at(state: interface.State, x: int, y: int) -> int:
    best = 0
    for element in state.ui_elements:
        if _bbox_contains(getattr(element, "bbox_pixels", None), x, y):
            best = max(best, _element_interaction_score(element))
    return best


def _candidate_y_values(y: int, height: int) -> list[int]:
    values = [y]
    if height >= 1800 and 0 < y < height:
        for factor in (2.0, 2.2, 2.4, 2.5, height / 1080.0, height / 1000.0):
            scaled = int(round(y * factor))
            if 0 <= scaled <= height:
                values.append(scaled)
    deduped: list[int] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _repair_point_against_state(state: interface.State, x: int, y: int) -> tuple[int, int, str | None]:
    """Repairs thumbnail-like y coordinates only when the current point is empty.

    MobileAgentV3 and some Qwen variants occasionally emit coordinates from a
    vertically compressed rendering of the screenshot. We only rewrite the point
    when the original point does not hit any actionable AndroidWorld UI element
    and a scaled-y candidate does.
    """
    width, height = state.pixels.shape[1], state.pixels.shape[0]
    if not (0 <= x <= width and 0 <= y <= height):
        return x, y, None

    original_score = _best_element_score_at(state, x, y)
    if original_score > 0:
        return x, y, None

    best_y = y
    best_score = original_score
    for candidate_y in _candidate_y_values(y, height):
        score = _best_element_score_at(state, x, candidate_y)
        if score > best_score:
            best_score = score
            best_y = candidate_y

    if best_y != y and best_score > 0:
        return x, best_y, f"y:{y}->{best_y}"
    return x, y, None


class AndroidWorldGuardedEnv:
    """Delegates to AndroidWorld while guarding mode-specific action quirks."""

    def __init__(self, env: interface.AsyncEnv, *, repair_coordinates: bool = True):
        self._env = env
        self.repair_coordinates = repair_coordinates
        self._evaluation_state = None

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    @property
    def controller(self):
        return self._env.controller

    @property
    def interaction_cache(self) -> str:
        return self._env.interaction_cache

    @interaction_cache.setter
    def interaction_cache(self, value: str) -> None:
        self._env.interaction_cache = value

    @property
    def foreground_activity_name(self) -> str:
        return self._env.foreground_activity_name

    @property
    def device_screen_size(self) -> tuple[int, int]:
        return self._env.device_screen_size

    @property
    def logical_screen_size(self) -> tuple[int, int]:
        return self._env.logical_screen_size

    @property
    def orientation(self) -> int:
        return self._env.orientation

    @property
    def physical_frame_boundary(self) -> tuple[int, int, int, int]:
        return self._env.physical_frame_boundary

    def reset(self, go_home: bool = False) -> interface.State:
        return self._env.reset(go_home=go_home)

    def get_state(self, wait_to_stabilize: bool = False) -> interface.State:
        if self._evaluation_state is not None:
            state = self._evaluation_state
            self._evaluation_state = None
            return state
        return self._env.get_state(wait_to_stabilize=wait_to_stabilize)

    def prepare_evaluation_state(self) -> interface.State:
        """Freeze one settled state for the evaluator's first read.

        AndroidWorld evaluators run after the agent returns.  Without this
        handoff, the evaluator performs a second asynchronous accessibility
        read and can observe a different or temporarily empty tree than the
        final state shown to the agent.
        """
        self._evaluation_state = self._env.get_state(wait_to_stabilize=True)
        return self._evaluation_state

    def evaluation_state_facts(self) -> dict[str, Any] | None:
        state = self._evaluation_state
        if state is None:
            return None
        pixels = getattr(state, "pixels", None)
        return {
            "screenshot_size": (
                [int(pixels.shape[1]), int(pixels.shape[0])]
                if getattr(pixels, "shape", None) is not None
                and len(pixels.shape) >= 2
                else None
            ),
            "ui_element_count": len(getattr(state, "ui_elements", ()) or ()),
            "captured_for": "evaluator",
        }

    def ask_question(self, question: str, timeout_seconds: float = -1.0) -> str | None:
        return self._env.ask_question(question, timeout_seconds=timeout_seconds)

    def hide_automation_ui(self) -> None:
        self._env.hide_automation_ui()

    def display_message(self, message: str, header: str = "") -> None:
        self._env.display_message(message, header=header)

    def close(self) -> None:
        self._env.close()

    def _repair_action(self, action):
        if not self.repair_coordinates:
            return action
        if getattr(action, "action_type", None) not in _COORDINATE_ACTIONS:
            return action
        if getattr(action, "index", None) is not None:
            return action
        if getattr(action, "x", None) is None or getattr(action, "y", None) is None:
            return action

        state = self._env.get_state(wait_to_stabilize=False)
        x, y, note = _repair_point_against_state(state, int(action.x), int(action.y))
        if note is None:
            return action
        logger.info("AndroidWorld coordinate guard repaired %s for %s", note, action)
        try:
            return replace(action, x=x, y=y)
        except TypeError:
            action.x = x
            action.y = y
            return action

    def execute_action(self, action) -> None:
        if getattr(action, "action_type", None) == aw_json_action.SWIPE:
            action = self._repair_swipe(action)
        else:
            action = self._repair_action(action)
        self._env.execute_action(action)

    def _repair_swipe(self, action):
        if not self.repair_coordinates:
            return action
        direction = getattr(action, "direction", None)
        if not isinstance(direction, (list, tuple)) or len(direction) != 4:
            return action

        state = self._env.get_state(wait_to_stabilize=False)
        x1, y1, x2, y2 = [int(value) for value in direction]
        new_x1, new_y1, note1 = _repair_point_against_state(state, x1, y1)
        new_x2, new_y2, note2 = _repair_point_against_state(state, x2, y2)
        if note1 is None and note2 is None:
            return action
        new_direction = [new_x1, new_y1, new_x2, new_y2]
        logger.info(
            "AndroidWorld coordinate guard repaired swipe %s %s for %s",
            note1,
            note2,
            action,
        )
        try:
            return replace(action, direction=new_direction)
        except TypeError:
            action.direction = new_direction
            return action


class AndroidWorldBackedMobileUseEnvironment:
    """Presents AndroidWorld as a mobile-use compatible environment."""

    def __init__(
        self,
        *,
        benchmark_env: interface.AsyncEnv,
        serial_no: str | None,
        host: str,
        port: int,
        wait_after_action_seconds: float = 2.0,
        include_a11y_tree: bool = False,
        include_dom_tree: bool = False,
        include_device_time: bool = False,
    ):
        self.benchmark_env = benchmark_env
        self.serial_no = serial_no
        self.host = host
        self.port = port
        self.wait_after_action_seconds = wait_after_action_seconds
        self.include_a11y_tree = include_a11y_tree
        self.include_dom_tree = include_dom_tree
        self.include_device_time = include_device_time
        self.window_size = self.benchmark_env.logical_screen_size
        self.viewport_size = self.window_size
        self._action_space = [
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
        ]
        self._register_function: dict[str, object] = {}
        self.action_history: list[dict[str, Any]] = []
        self.last_action_record: dict[str, Any] | None = None
        self._observation_counter = 0
        self._pending_post_action_state: EnvState | None = None

    @property
    def action_space(self) -> list[str]:
        return list(self._action_space)

    def register_action(self, action_name: str, action_func) -> None:
        if action_name not in self._action_space:
            self._action_space.append(action_name)
        self._register_function[action_name] = action_func

    def reset(self, go_home: bool = False) -> None:
        if go_home:
            self.benchmark_env.reset(go_home=True)

    def _device_time(self) -> str | None:
        if not self.include_device_time:
            return None
        # Keep observation inside the AndroidWorld controller boundary. Device
        # time is optional for current study prompts, so a failed generic query
        # should never break an episode.
        response = aw_adb_utils.issue_generic_request(
            ["shell", "date"],
            self.benchmark_env.controller.env,
        )
        return _decode_generic_output(response) or None

    def _build_a11y_tree(self, state: interface.State) -> str | None:
        if not self.include_a11y_tree:
            return None
        try:
            return _generate_ui_elements_description_list(
                state.ui_elements,
                self.benchmark_env.logical_screen_size,
            )
        except Exception as exc:
            logger.warning("Failed to convert AndroidWorld UI elements to a11y tree: %s", exc)
            return None

    def _build_dom_tree(self, state: interface.State) -> str | None:
        if not self.include_dom_tree:
            return None
        try:
            return _state_to_appagent_xml(state)
        except Exception as exc:
            logger.warning("Failed to convert AndroidWorld UI elements to DOM XML: %s", exc)
            return None

    def _capture_state(
        self,
        *,
        wait_to_stabilize: bool,
        return_a11y_tree: bool | None,
        source: str,
    ) -> EnvState:
        state = self.benchmark_env.get_state(wait_to_stabilize=wait_to_stabilize)
        self._observation_counter += 1
        pixels = Image.fromarray(state.pixels)
        pixel_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
        package = _extract_package_name(self.benchmark_env.foreground_activity_name)
        if return_a11y_tree is None:
            return_a11y_tree = self.include_a11y_tree
        a11y_tree = self._build_a11y_tree(state) if return_a11y_tree else None
        return EnvState(
            pixels=pixels,
            package=package,
            a11y_tree=a11y_tree,
            dom_tree=self._build_dom_tree(state),
            device_time=self._device_time(),
            capture_id=self._observation_counter,
            captured_at=_utcnow_iso(),
            screenshot_sha256=pixel_hash,
            capture_source=source,
            wait_to_stabilize=wait_to_stabilize,
        )

    def get_state(
        self,
        display_id: int = -1,
        return_a11y_tree: bool | None = None,
        wait_to_stabilize: bool = False,
    ) -> EnvState:
        del display_id
        if self._pending_post_action_state is not None and not wait_to_stabilize:
            pending = self._pending_post_action_state
            self._pending_post_action_state = None
            return pending
        return self._capture_state(
            wait_to_stabilize=wait_to_stabilize,
            return_a11y_tree=return_a11y_tree,
            source="post_action_settled" if wait_to_stabilize else "fresh_model_observation",
        )

    def _sleep_after_action(self) -> None:
        if self.wait_after_action_seconds > 0:
            time.sleep(self.wait_after_action_seconds)

    def _set_pending_post_action_state(self) -> None:
        self._pending_post_action_state = self._capture_state(
            wait_to_stabilize=True,
            return_a11y_tree=None,
            source="post_action_settled",
        )

    def _adb_command_prefix(self) -> list[str]:
        command = shutil.which("adb") or "adb"
        prefix = [command]
        if self.host:
            prefix.extend(["-H", str(self.host)])
        if self.port:
            prefix.extend(["-P", str(int(self.port))])
        if self.serial_no:
            prefix.extend(["-s", str(self.serial_no)])
        return prefix

    def _full_adb_command(self, args: list[object]) -> list[str]:
        return [*self._adb_command_prefix(), *[str(arg) for arg in args]]

    def _run_adb(self, args: list[object], *, timeout_sec: float = 30.0) -> str:
        command = self._full_adb_command(args)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Error executing ADB command {args[:3]}: {exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            if not detail:
                detail = f"exit code {completed.returncode}"
            raise RuntimeError(f"Error executing ADB command {args[:3]}: {detail}")
        return (completed.stdout or "").strip()

    def _adb_input(self, args: list[object], *, timeout_sec: float = 30.0) -> str:
        return self._run_adb(["shell", "input", *args], timeout_sec=timeout_sec)

    def _button_action(self, button: str) -> aw_json_action.JSONAction:
        normalized = button.strip().lower()
        if normalized == "back":
            return aw_json_action.JSONAction(action_type=aw_json_action.NAVIGATE_BACK)
        if normalized == "home":
            return aw_json_action.JSONAction(action_type=aw_json_action.NAVIGATE_HOME)
        if normalized == "enter":
            return aw_json_action.JSONAction(action_type=aw_json_action.KEYBOARD_ENTER)
        raise ValueError(f"Unsupported system button: {button}")

    def _clear_text(self) -> None:
        self._adb_input(["keyevent", "KEYCODE_MOVE_END"])
        for _ in range(40):
            self._adb_input(["keyevent", "KEYCODE_DEL"])

    def _key_event(self, key_name: str) -> None:
        normalized = key_name.strip().upper()
        keycode_aliases = {
            "BACK": "KEYCODE_BACK",
            "ENTER": "KEYCODE_ENTER",
            "HOME": "KEYCODE_HOME",
            "MENU": "KEYCODE_MENU",
        }
        keycode = keycode_aliases.get(normalized, normalized)
        self._adb_input(["keyevent", keycode])

    def _tap(self, x: object, y: object) -> None:
        self._adb_input(["tap", int(x), int(y)])

    def _long_press(self, x: object, y: object, seconds: object = 1.0) -> None:
        duration_ms = max(1, int(round(float(seconds) * 1000)))
        timeout_sec = max(30.0, duration_ms / 1000.0 + 10.0)
        self._adb_input(["swipe", int(x), int(y), int(x), int(y), duration_ms], timeout_sec=timeout_sec)

    def _swipe(self, x1: object, y1: object, x2: object, y2: object, seconds: object = 0.5) -> None:
        duration_ms = max(1, int(round(float(seconds) * 1000)))
        timeout_sec = max(30.0, duration_ms / 1000.0 + 10.0)
        self._adb_input(
            ["swipe", int(x1), int(y1), int(x2), int(y2), duration_ms],
            timeout_sec=timeout_sec,
        )

    def _to_androidworld_action(self, action: Action) -> aw_json_action.JSONAction:
        params = action.parameters or {}
        name = action.name
        if name == "open":
            return aw_json_action.JSONAction(
                action_type=aw_json_action.OPEN_APP,
                app_name=str(params["text"]),
            )
        if name == "click":
            x, y = params["coordinate"]
            return aw_json_action.JSONAction(
                action_type=aw_json_action.CLICK,
                x=int(x),
                y=int(y),
            )
        if name == "long_press":
            x, y = params["coordinate"]
            return aw_json_action.JSONAction(
                action_type=aw_json_action.LONG_PRESS,
                x=int(x),
                y=int(y),
            )
        if name == "type":
            return aw_json_action.JSONAction(
                action_type=aw_json_action.INPUT_TEXT,
                text=str(params["text"]),
            )
        if name == "swipe":
            x1, y1 = params["coordinate"]
            x2, y2 = params["coordinate2"]
            return aw_json_action.JSONAction(
                action_type=aw_json_action.SWIPE,
                direction=[int(x1), int(y1), int(x2), int(y2)],
            )
        if name == "press_home":
            return aw_json_action.JSONAction(action_type=aw_json_action.NAVIGATE_HOME)
        if name == "press_back":
            return aw_json_action.JSONAction(action_type=aw_json_action.NAVIGATE_BACK)
        if name == "answer":
            return aw_json_action.JSONAction(
                action_type=aw_json_action.ANSWER,
                text=str(params["text"]),
            )
        if name == "system_button":
            return self._button_action(str(params["button"]))
        raise ValueError(f"Unsupported mobile-use action for AndroidWorld env: {action}")

    def _begin_action_record(
        self,
        action: Action,
        *,
        executor: str,
        command: list[str] | None = None,
        command_kind: str | None = None,
        actual_coordinates: dict[str, Any] | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "sequence": len(self.action_history) + 1,
            "started_at": _utcnow_iso(),
            "status": "started",
            "executor": executor,
            "command_kind": command_kind,
            "requested_action": {
                "name": action.name,
                "parameters": _to_jsonable(action.parameters or {}),
                "repr": str(action),
            },
            "actual_coordinates": actual_coordinates if actual_coordinates is not None else _action_coordinate_summary(action),
            "command": command,
            "command_text": subprocess.list2cmdline(command) if command else None,
            "note": note,
        }
        self.last_action_record = record
        self.action_history.append(record)
        return record

    def _finish_action_record(
        self,
        record: dict[str, Any],
        *,
        status: str,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        record["finished_at"] = _utcnow_iso()
        record["status"] = status
        if result is not None:
            record["result"] = _to_jsonable(result)
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"

    def _run_recorded_action(self, record: dict[str, Any], func) -> str | None:
        try:
            result = func()
        except Exception as exc:
            self._finish_action_record(record, status="failed", error=exc)
            raise
        self._finish_action_record(record, status="completed", result=result)
        return result

    def execute_action(self, action: Action) -> str | None:
        # A post-action state is consumed by the agent's exec_env_state read.
        # Clear an older one so a failed or nested action cannot reuse it.
        self._pending_post_action_state = None
        params = action.parameters or {}
        if action.name in self._register_function:
            record = self._begin_action_record(
                action,
                executor="registered_function",
                command_kind=action.name,
                note="Registered mobile-use action function.",
            )
            result = self._run_recorded_action(
                record, lambda: self._register_function[action.name](self, **params)
            )
            self._set_pending_post_action_state()
            return result
        if action.name == "wait":
            duration = float(params.get("time", 5.0))
            record = self._begin_action_record(
                action,
                executor="python",
                command=["sleep", str(duration)],
                command_kind="sleep",
            )
            result = self._run_recorded_action(record, lambda: (time.sleep(duration) or None))
            self._set_pending_post_action_state()
            return result
        if action.name == "take_note":
            record = self._begin_action_record(action, executor="agent_memory", command_kind="take_note")
            result = self._run_recorded_action(record, lambda: str(params.get("text", "")))
            self._set_pending_post_action_state()
            return result
        if action.name == "clear_text":
            commands = [
                self._full_adb_command(["shell", "input", "keyevent", "KEYCODE_MOVE_END"]),
                self._full_adb_command(["shell", "input", "keyevent", "KEYCODE_DEL"]),
            ]
            record = self._begin_action_record(
                action,
                executor="adb",
                command=commands[0],
                command_kind="clear_text",
                note="Runs KEYCODE_MOVE_END once and KEYCODE_DEL 40 times.",
            )
            record["commands"] = commands
            result = self._run_recorded_action(record, lambda: (self._clear_text(), self._sleep_after_action(), None)[-1])
            self._set_pending_post_action_state()
            return result
        if action.name == "key":
            keycode = str(params["text"]).strip().upper()
            keycode = {"BACK": "KEYCODE_BACK", "ENTER": "KEYCODE_ENTER", "HOME": "KEYCODE_HOME", "MENU": "KEYCODE_MENU"}.get(keycode, keycode)
            record = self._begin_action_record(
                action,
                executor="adb",
                command=self._full_adb_command(["shell", "input", "keyevent", keycode]),
                command_kind="keyevent",
            )
            result = self._run_recorded_action(record, lambda: (self._key_event(str(params["text"])), self._sleep_after_action(), None)[-1])
            self._set_pending_post_action_state()
            return result
        if action.name == "click":
            x, y = params["coordinate"]
            record = self._begin_action_record(
                action,
                executor="adb",
                command=self._full_adb_command(["shell", "input", "tap", int(x), int(y)]),
                command_kind="tap",
                actual_coordinates={"coordinate": [int(x), int(y)]},
            )
            result = self._run_recorded_action(record, lambda: (self._tap(x, y), self._sleep_after_action(), None)[-1])
            self._set_pending_post_action_state()
            return result
        if action.name == "long_press":
            x, y = params["coordinate"]
            duration_ms = max(1, int(round(float(params.get("time", 1.0)) * 1000)))
            record = self._begin_action_record(
                action,
                executor="adb",
                command=self._full_adb_command(["shell", "input", "swipe", int(x), int(y), int(x), int(y), duration_ms]),
                command_kind="long_press",
                actual_coordinates={"coordinate": [int(x), int(y)], "duration_ms": duration_ms},
            )
            result = self._run_recorded_action(record, lambda: (self._long_press(x, y, params.get("time", 1.0)), self._sleep_after_action(), None)[-1])
            self._set_pending_post_action_state()
            return result
        if action.name == "swipe":
            x1, y1 = params["coordinate"]
            x2, y2 = params["coordinate2"]
            duration_ms = max(1, int(round(float(params.get("time", 0.5)) * 1000)))
            record = self._begin_action_record(
                action,
                executor="adb",
                command=self._full_adb_command(["shell", "input", "swipe", int(x1), int(y1), int(x2), int(y2), duration_ms]),
                command_kind="swipe",
                actual_coordinates={"start": [int(x1), int(y1)], "end": [int(x2), int(y2)], "duration_ms": duration_ms},
            )
            result = self._run_recorded_action(record, lambda: (self._swipe(x1, y1, x2, y2, params.get("time", 0.5)), self._sleep_after_action(), None)[-1])
            self._set_pending_post_action_state()
            return result
        if action.name == "press_home":
            record = self._begin_action_record(
                action,
                executor="adb",
                command=self._full_adb_command(["shell", "input", "keyevent", "KEYCODE_HOME"]),
                command_kind="press_home",
            )
            result = self._run_recorded_action(record, lambda: (self._key_event("HOME"), self._sleep_after_action(), None)[-1])
            self._set_pending_post_action_state()
            return result
        if action.name == "press_back":
            record = self._begin_action_record(
                action,
                executor="adb",
                command=self._full_adb_command(["shell", "input", "keyevent", "KEYCODE_BACK"]),
                command_kind="press_back",
            )
            result = self._run_recorded_action(record, lambda: (self._key_event("BACK"), self._sleep_after_action(), None)[-1])
            self._set_pending_post_action_state()
            return result
        if action.name == "system_button":
            button = str(params["button"])
            keycode = {"Back": "KEYCODE_BACK", "Home": "KEYCODE_HOME", "Menu": "KEYCODE_MENU", "Enter": "KEYCODE_ENTER"}.get(button, button)
            record = self._begin_action_record(
                action,
                executor="adb",
                command=self._full_adb_command(["shell", "input", "keyevent", keycode]),
                command_kind="system_button",
            )
            result = self._run_recorded_action(record, lambda: (self._key_event(button), self._sleep_after_action(), None)[-1])
            self._set_pending_post_action_state()
            return result

        aw_action = self._to_androidworld_action(action)
        aw_action_payload = _to_jsonable(aw_action)
        record = self._begin_action_record(
            action,
            executor="androidworld",
            command_kind=str(getattr(aw_action, "action_type", action.name)),
            note="Executed through AndroidWorld environment.",
        )
        record["androidworld_action"] = aw_action_payload
        record["command"] = ["benchmark_env.execute_action", aw_action_payload]
        record["command_text"] = (
            "benchmark_env.execute_action("
            + json.dumps(aw_action_payload, ensure_ascii=False, sort_keys=True)
            + ")"
        )

        def _execute_androidworld() -> str | None:
            self.benchmark_env.execute_action(aw_action)
            self._sleep_after_action()
            if aw_action.action_type == aw_json_action.ANSWER:
                return aw_action.text
            return None

        result = self._run_recorded_action(record, _execute_androidworld)
        self._set_pending_post_action_state()
        return result


class AndroidWorldBackedAppAgentController:
    """Presents AndroidWorld as an AppAgent-compatible controller."""

    supports_appagent_knowledge = False

    def __init__(
        self,
        *,
        benchmark_env: interface.AsyncEnv,
        wait_after_action_seconds: float = 2.0,
    ):
        self.benchmark_env = benchmark_env
        self.wait_after_action_seconds = wait_after_action_seconds
        self.width, self.height = self.get_device_size()
        self.viewport_size = (self.width, self.height)
        self._pending_post_action_state: interface.State | None = None
        self._last_observation_source = "fresh_model_observation"

    def _next_observation_state(self) -> interface.State:
        """Return one paired observation, consuming the action-settled state."""
        if self._pending_post_action_state is not None:
            state = self._pending_post_action_state
            self._pending_post_action_state = None
            self._last_observation_source = "post_action_settled"
            return state
        self._last_observation_source = "fresh_model_observation"
        return self.benchmark_env.get_state(wait_to_stabilize=False)

    def _run_action(self, action: aw_json_action.JSONAction) -> str:
        try:
            self._pending_post_action_state = None
            self.benchmark_env.execute_action(action)
            if self.wait_after_action_seconds > 0:
                time.sleep(self.wait_after_action_seconds)
            self._pending_post_action_state = self.benchmark_env.get_state(
                wait_to_stabilize=True
            )
            return "OK"
        except Exception as exc:
            logger.warning("AppAgent AndroidWorld action failed: %s", exc)
            return "ERROR"

    def get_device_size(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.benchmark_env.logical_screen_size)

    def get_current_activity(self) -> str:
        return self.benchmark_env.foreground_activity_name

    def get_screenshot(self, prefix: str, save_dir: str | Path) -> str:
        try:
            state = self._next_observation_state()
            return _save_pixels(state.pixels, Path(save_dir) / f"{prefix}.png")
        except Exception as exc:
            logger.warning("Failed to save AndroidWorld screenshot for AppAgent: %s", exc)
            return "ERROR"

    def capture_observation(self, prefix: str, save_dir: str | Path) -> dict[str, str]:
        """Capture PNG and XML from one AndroidWorld state object.

        This is an evidence-only helper for the study wrapper.  The action
        execution path and the controller's public screenshot/XML methods are
        unchanged.
        """
        try:
            state = self._next_observation_state()
            root = Path(save_dir)
            root.mkdir(parents=True, exist_ok=True)
            screenshot_path = _save_pixels(state.pixels, root / f"{prefix}.png")
            xml_path = root / f"{prefix}.xml"
            xml_path.write_text(_state_to_appagent_xml(state), encoding="utf-8")
            return {
                "screenshot": str(screenshot_path),
                "xml": str(xml_path),
                "source": self._last_observation_source,
            }
        except Exception as exc:
            logger.warning("Failed to capture paired AndroidWorld observation: %s", exc)
            return {"screenshot": "ERROR", "xml": "ERROR", "error": f"{type(exc).__name__}: {exc}"}

    def save_screenshot(self, save_path: str | Path) -> str:
        return self.get_screenshot(Path(save_path).stem, Path(save_path).parent)

    def get_xml(self, prefix: str, save_dir: str | Path) -> str:
        try:
            state = self._next_observation_state()
            xml_text = _state_to_appagent_xml(state)
            target = Path(save_dir) / f"{prefix}.xml"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(xml_text, encoding="utf-8")
            return str(target)
        except Exception as exc:
            logger.warning("Failed to save AndroidWorld XML for AppAgent: %s", exc)
            return "ERROR"

    def tap(self, x: int | float, y: int | float) -> str:
        return self._run_action(
            aw_json_action.JSONAction(
                action_type=aw_json_action.CLICK,
                x=int(x),
                y=int(y),
            )
        )

    def text(self, input_str: str) -> str:
        return self._run_action(
            aw_json_action.JSONAction(
                action_type=aw_json_action.INPUT_TEXT,
                text=input_str,
            )
        )

    def long_press(self, x: int | float, y: int | float) -> str:
        return self._run_action(
            aw_json_action.JSONAction(
                action_type=aw_json_action.LONG_PRESS,
                x=int(x),
                y=int(y),
            )
        )

    def swipe(
        self,
        x: int | float,
        y: int | float,
        direction: str,
        dist: str | int = "medium",
        quick: bool = False,
    ) -> str:
        del quick
        if isinstance(dist, str):
            unit_dist = int(self.width / 10)
            if dist == "long":
                unit_dist *= 3
            elif dist == "medium":
                unit_dist *= 2
        else:
            unit_dist = int(dist)
        offsets = {
            "up": (0, -2 * unit_dist),
            "down": (0, 2 * unit_dist),
            "left": (-1 * unit_dist, 0),
            "right": (unit_dist, 0),
        }
        if direction not in offsets:
            return "ERROR"
        dx, dy = offsets[direction]
        return self.swipe_precise((int(x), int(y)), (int(x + dx), int(y + dy)))

    def swipe_precise(self, start: tuple[int, int], end: tuple[int, int]) -> str:
        start_x, start_y = start
        end_x, end_y = end
        return self._run_action(
            aw_json_action.JSONAction(
                action_type=aw_json_action.SWIPE,
                direction=[int(start_x), int(start_y), int(end_x), int(end_y)],
            )
        )

    def back(self) -> str:
        return self._run_action(
            aw_json_action.JSONAction(action_type=aw_json_action.NAVIGATE_BACK)
        )

    def home(self) -> str:
        return self._run_action(
            aw_json_action.JSONAction(action_type=aw_json_action.NAVIGATE_HOME)
        )

    def enter(self) -> str:
        return self._run_action(
            aw_json_action.JSONAction(action_type=aw_json_action.KEYBOARD_ENTER)
        )

    def answer(self, text: str) -> str:
        return self._run_action(
            aw_json_action.JSONAction(
                action_type=aw_json_action.ANSWER,
                text=text,
            )
        )

    def launch_app(self, package_or_app_name: str) -> str:
        return self._run_action(
            aw_json_action.JSONAction(
                action_type=aw_json_action.OPEN_APP,
                app_name=package_or_app_name,
            )
        )

    def launch(self, package_or_app_name: str) -> str:
        return self.launch_app(package_or_app_name)


def _xml_bool(value: bool | None) -> str:
    return "true" if bool(value) else "false"


def _xml_text(value) -> str:
    if value is None:
        return ""
    return str(value)


def _state_to_appagent_xml(state: interface.State) -> str:
    """Builds the small uiautomator-style XML subset AppAgent consumes."""
    root = ET.Element("hierarchy", {"rotation": "0"})
    height, width = state.pixels.shape[0], state.pixels.shape[1]
    container = ET.SubElement(
        root,
        "node",
        {
            "index": "0",
            "text": "",
            "resource-id": "",
            "class": "android.widget.FrameLayout",
            "package": "",
            "content-desc": "",
            "checkable": "false",
            "checked": "false",
            "clickable": "false",
            "enabled": "true",
            "focusable": "false",
            "focused": "false",
            "scrollable": "false",
            "long-clickable": "false",
            "password": "false",
            "selected": "false",
            "bounds": f"[0,0][{width},{height}]",
        },
    )
    for index, element in enumerate(state.ui_elements):
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is None:
            continue
        x1, y1 = int(max(0, bbox.x_min)), int(max(0, bbox.y_min))
        x2, y2 = int(max(x1, bbox.x_max)), int(max(y1, bbox.y_max))
        attrib = {
            "index": str(index),
            "text": _xml_text(getattr(element, "text", None)),
            "resource-id": _xml_text(
                getattr(element, "resource_id", None)
                or getattr(element, "resource_name", None)
            ),
            "class": _xml_text(getattr(element, "class_name", None) or "android.view.View"),
            "package": _xml_text(getattr(element, "package_name", None)),
            "content-desc": _xml_text(getattr(element, "content_description", None)),
            "checkable": _xml_bool(getattr(element, "is_checkable", False)),
            "checked": _xml_bool(getattr(element, "is_checked", False)),
            "clickable": _xml_bool(getattr(element, "is_clickable", False)),
            "enabled": _xml_bool(getattr(element, "is_enabled", True)),
            "focusable": _xml_bool(
                getattr(element, "is_focusable", False)
                or getattr(element, "is_editable", False)
            ),
            "focused": _xml_bool(getattr(element, "is_focused", False)),
            "scrollable": _xml_bool(getattr(element, "is_scrollable", False)),
            "long-clickable": _xml_bool(getattr(element, "is_long_clickable", False)),
            "password": "false",
            "selected": _xml_bool(getattr(element, "is_selected", False)),
            "bounds": f"[{x1},{y1}][{x2},{y2}]",
        }
        ET.SubElement(container, "node", attrib)
    return ET.tostring(root, encoding="unicode")

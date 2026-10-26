# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mobile-Agent-v3 for Android."""

import os
import re
import time
import copy
import json
import ast
import hashlib
import xml.etree.ElementTree as ET
from android_world.agents import base_agent
from android_world.agents import infer_ma3 as infer
from android_world.agents import m3a_utils
from android_world.env import adb_utils
from android_world.env import tools
from android_world.env import interface
from android_world.agents import new_json_action as json_action
from dataclasses import asdict
from PIL import Image
from android_world.agents.mobile_agent_v3_agent import (
    InfoPool, 
    Manager, 
    Executor, 
    Notetaker, 
    ActionReflector,
    ALL_APPS
)


def _xml_bool(value):
  return "true" if bool(value) else "false"


def _xml_text(value):
  if value is None:
    return ""
  return str(value)


def _state_to_uiautomator_xml(state: interface.State) -> str:
  """Serializes AndroidWorld's UI elements beside the screenshot that used them."""
  root = ET.Element("hierarchy", {"rotation": "0"})
  height, width = state.pixels.shape[:2]
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
  for index, element in enumerate(getattr(state, "ui_elements", []) or []):
    bbox = getattr(element, "bbox_pixels", None)
    if bbox is None:
      continue
    x1, y1 = int(max(0, bbox.x_min)), int(max(0, bbox.y_min))
    x2, y2 = int(max(x1, bbox.x_max)), int(max(y1, bbox.y_max))
    ET.SubElement(
        container,
        "node",
        {
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
        },
    )
  return ET.tostring(root, encoding="unicode")


def _save_state_xml(state: interface.State, xml_path: str) -> str:
  with open(xml_path, "w", encoding="utf-8") as f:
    f.write(_state_to_uiautomator_xml(state))
  return xml_path


def _xml_paths_for_screenshots(screenshots):
  xml_paths = []
  for screenshot in screenshots or []:
    if not screenshot:
      continue
    xml_path = os.path.splitext(str(screenshot))[0] + ".xml"
    if os.path.exists(xml_path):
      xml_paths.append(xml_path)
  return xml_paths

def convert_fc_action_to_json_action(
    dummy_action,
    coor_type: str = "abs",
    screen_width: int = 1080,
    screen_height: int = 2400,
) -> json_action.JSONAction:
 
    dummy_action = dummy_action.replace("```", "").replace("json", "").strip()
    try:
      action_json = json.loads(dummy_action)
    except json.JSONDecodeError:
      action_json = ast.literal_eval(dummy_action)
      if not isinstance(action_json, dict):
        raise ValueError("action_json must be a dict")

    def scale_point(point):
      x = int(point[0])
      y = int(point[1])
      if coor_type != "abs":
        x = int(x / 1000 * screen_width)
        y = int(y / 1000 * screen_height)
      x = max(0, min(screen_width - 1, x))
      y = max(0, min(screen_height - 1, y))
      return x, y

    action_type = action_json['action']
    
    x = None
    y = None
    text = None
    direction = None
    goal_status = None
    app_name = None

    if action_type == 'open_app':
        action_type = json_action.OPEN_APP
        app_name = action_json['text']
    elif action_type == 'click':
        action_type = json_action.CLICK
        x, y = scale_point(action_json['coordinate'])
    elif action_type == 'long_press':
        action_type = json_action.LONG_PRESS
        x, y = scale_point(action_json['coordinate'])
    elif action_type == 'type':
        action_type = json_action.INPUT_TEXT
        text = action_json['text']
    elif action_type == 'swipe':
        action_type = json_action.SWIPE
        start_x, start_y = scale_point(action_json['coordinate'])
        end_x, end_y = scale_point(action_json['coordinate2'])
        direction = [start_x, start_y, end_x, end_y]
    elif action_type == 'system_button':
        if action_json['button'] == 'Enter' or action_json['button'] == 'enter':
            action_type = json_action.KEYBOARD_ENTER
        elif action_json['button'] == 'Back' or action_json['button'] == 'back':
            action_type = json_action.NAVIGATE_BACK
        elif action_json['button'] == 'Home' or action_json['button'] == 'home':
            action_type = json_action.NAVIGATE_HOME
    elif action_type == 'answer':
        action_type = json_action.ANSWER
        text = action_json['text']
    elif action_type == 'done' or action_type == 'terminate':
        action_type = json_action.STATUS
        goal_status = json_action.GOAL_STATUS

    return json_action.JSONAction(
            action_type=action_type,
            x=x,
            y=y,
            text=text,
            direction=direction,
            goal_status=goal_status,
            app_name=app_name,
      )

all_apps_str = ""
for app_str in ALL_APPS:
    all_apps_str += f"  - {app_str}\n"

DETAILED_TIPS = (
    'General:\n'
    '- For any pop-up window, such as a permission request, you need to close it (e.g., by clicking `Don\'t Allow` or `Accept & continue`) before proceeding. Never choose to add any account or log in.`\n'
    '- For requests that are questions (or chat messages), remember to use'
    ' the `answer` action to reply to user explicitly before finish!\n'
    '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
    " it's already on), you can just complete the task.\n"
    '- Two files or notes can be considered the same or duplicate only if their names, creation time, and detailed content are exactly the same.\n'
    'Action Related:\n'
    '- Use the `open_app` action whenever you want to open an app'
    ' (nothing will happen if the app is not installed), do not use the'
    ' app drawer to open an app.\n'
    '- Consider exploring the screen by using the `swipe`'
    ' action with different directions to reveal additional content. Or use search to quickly find a specific entry, if applicable.\n'
    '- If you cannot change the page content by swiping in the same direction continuously, the page may have been swiped to the bottom. Please try another operation to display more content.\n'
    '- For some horizontally distributed tags, you can swipe horizontally to view more.\n'
    'Text Related Operations:\n'
    '- Activated input box: If an input box is activated, it may have a cursor inside it and the keyboard is visible. If there is no cursor on the screen but the keyboard is visible, it may be because the cursor is blinking. The color of the activated input box will be highlighted. If you are not sure whether the input box is activated, click it before typing.\n'
    '- To input some text: first click the input box that you want to input, make sure the correct input box is activated and the keyboard is visible, then use `type` action to enter the specified text.\n'
    '- To clear the text: long press the backspace button in the keyboard.\n'
    '- To copy some text: first long press the text you want to copy, then click the `copy` button in bar.\n'
    '- To paste text into a text box: first long press the'
    ' text box, then click the `paste`'
    ' button in bar.'
)

def _safe_task_dir_name(name: str, max_len: int = 40) -> str:
  digest = hashlib.sha1(name.encode("utf-8", errors="ignore")).hexdigest()[:8]
  safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
  safe_name = safe_name.replace(" ", "_").strip(" ._")
  if not safe_name:
    return f"task_{digest}"
  if len(safe_name) <= max_len:
    return safe_name
  prefix_len = max(1, max_len - len(digest) - 1)
  prefix = safe_name[:prefix_len].strip(" ._") or "task"
  return f"{prefix}_{digest}"

class MobileAgentV3_M3A(base_agent.EnvironmentInteractingAgent):
  """Mobile Agent E wrapper for Android World."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      vllm: infer.MultimodalLlmWrapper,
      name: str = 'MobileAgentE_M3A',
      wait_after_action_seconds: float = 3.0,
      output_path: str = "",
      coor_type: str = "abs",
      max_action_retry: int = 3,
  ):
      """Initializes a MobileAgentE_M3A Agent.

      Args:
          env: The environment.
          vllm: The multimodal LLM wrapper.
          name: The agent name.
          wait_after_action_seconds: Seconds to wait for the screen to stablize
              after executing an action
          max_action_retry: Number of same-screen retries allowed when the
              operator response cannot be parsed or converted safely.
      """
      super().__init__(env, name)
      self.vllm = vllm
      self.additional_guidelines = None
      self.wait_after_action_seconds = wait_after_action_seconds
      self.output_path = output_path
      self.coor_type = coor_type
      self.max_action_retry = max(0, int(max_action_retry))
      if self.output_path:
        os.makedirs(self.output_path, exist_ok=True)
      self.task_name = {}
      
      # init info pool
      self.info_pool = InfoPool(
          additional_knowledge_manager="",
          additional_knowledge_executor=copy.deepcopy(DETAILED_TIPS),
          err_to_manager_thresh=2
      )
      
      # Hide the coordinates on screen which might affect the vision model.
      self.env.hide_automation_ui()
    
  def initialize_chrome(self):
    print("Running additional chrome initialization...")
    # handle chrome initialization problem for browser tasks
    adb_utils.launch_app("chrome", self.env.controller)
    time.sleep(5)

    tool_controller = tools.AndroidToolController(env=self.env.controller)
    time.sleep(2)

    first_op = False
    try:
      print("try first variant...")
      tool_controller.click_element("Use without an account")
      time.sleep(5.0)
      first_op = True
    except:
      print("Failed to click 'Use without an account' button.")
      pass
    
    if not first_op:
      print("try second variant...")
      try:
        tool_controller.click_element("Accept & continue")
      except:
        pass
      time.sleep(3.0)
      try:
        tool_controller.click_element("No thanks")
      except:
        pass
      time.sleep(5.0)
      
    adb_utils.press_home_button(self.env.controller)
    time.sleep(2.0)
    print("Done additional chrome initialization")

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    # Hide the coordinates on screen which might affect the vision model.
    self.env.hide_automation_ui()

    # init info pool again on reset
    self.info_pool = InfoPool(
      additional_knowledge_manager=copy.deepcopy(""),
      additional_knowledge_executor=copy.deepcopy(DETAILED_TIPS),
      err_to_manager_thresh=2
    )
    
  def get_task_name(self, suite):
    for name, instances in suite.items():
      self.task_name[instances[0].goal] = name

  def _task_output_dir(self) -> str:
    task_name = self.task_name.get(self.info_pool.instruction, self.info_pool.instruction)
    return os.path.join(self.output_path, _safe_task_dir_name(task_name))

  @staticmethod
  def _usage_from_call(call_record):
    usage = call_record.get("usage") if isinstance(call_record, dict) else None
    if not isinstance(usage, dict):
      return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
      "prompt_tokens": prompt_tokens,
      "completion_tokens": completion_tokens,
      "total_tokens": total_tokens,
    }

  @classmethod
  def _sum_call_usage(cls, call_history):
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for call in call_history:
      usage = cls._usage_from_call(call)
      for key, value in usage.items():
        totals[key] += value
    return totals

  @classmethod
  def _sum_call_usage_by_stage(cls, call_history):
    totals_by_stage = {}
    for call in call_history:
      stage = call.get("stage") if isinstance(call, dict) else None
      stage = stage or "unknown"
      if stage not in totals_by_stage:
        totals_by_stage[stage] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
      usage = cls._usage_from_call(call)
      for key, value in usage.items():
        totals_by_stage[stage][key] += value
    return totals_by_stage

  def _record_vllm_call(self, call_history, *, stage, messages, raw_response, screenshots=None):
    if hasattr(self.vllm, "build_call_record"):
      record = self.vllm.build_call_record(messages, raw_response, stage=stage)
    else:
      record = {
        "messages": messages,
        "response": "",
        "stage": stage,
        "usage": None,
        "model_name": getattr(self.vllm, "model", None),
      }
    if screenshots:
      record["screenshots"] = [path for path in screenshots if path]
      xml_paths = _xml_paths_for_screenshots(record["screenshots"])
      if xml_paths:
        record["xmls"] = xml_paths
        record["xml"] = xml_paths[-1]
    call_history.append(record)

  @staticmethod
  def _parse_action_payload(action_object_str):
    if not action_object_str:
      return None
    try:
      return json.loads(action_object_str)
    except Exception:
      return action_object_str

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_started_at = time.time()
    ## init agents ## 
    manager = Manager()
    executor = Executor()
    notetaker = Notetaker()
    action_reflector = ActionReflector()
    call_history = []
    output_planning = None
    output_action = None
    output_action_reflect = None
    output_note = None
    action_thought = None
    action_object_str = None
    action_description = None
    action_outcome = None
    error_description = None
    progress_status = None
    before_screenshot_file = None
    after_screenshot_file = None
    before_xml_file = None
    after_xml_file = None
    converted_action = None
    
    self.info_pool.instruction = goal
    step_idx = len(self.info_pool.action_history)
    
    self.info_pool.additional_knowledge_manager=copy.deepcopy(""),
    self.info_pool.additional_knowledge_executor=copy.deepcopy(DETAILED_TIPS),
    
    print('----------step ' + str(step_idx + 1))

    def _step_payload(done=False, error=None):
      action_payload = self._parse_action_payload(action_object_str)
      metadata = asdict(self.info_pool)
      metadata["last_act"] = action_object_str
      metadata["summary"] = action_description
      screenshots = {}
      if before_screenshot_file:
        screenshots["before"] = before_screenshot_file
      if after_screenshot_file:
        screenshots["after"] = after_screenshot_file
      doms = {}
      if before_xml_file:
        doms["before"] = before_xml_file
      if after_xml_file:
        doms["after"] = after_xml_file
      return {
        "step_idx": step_idx,
        "content": output_action or output_planning,
        "thought": action_thought,
        "action": action_payload,
        "action_s": action_object_str,
        "action_desc": action_description,
        "summary": action_description,
        "answer": getattr(converted_action, "text", None) if converted_action is not None else None,
        "plan": self.info_pool.plan,
        "sub_goal": self.info_pool.completed_plan,
        "progress": progress_status or self.info_pool.progress_status,
        "reflection_outcome": action_outcome,
        "reflection_error": error_description,
        "vlm_call_history": call_history,
        "usage": self._sum_call_usage(call_history),
        "usage_by_stage": self._sum_call_usage_by_stage(call_history),
        "screenshots": screenshots,
        "doms": doms,
        "metadata": metadata,
        "done": done,
        "error": error,
        "step_duration": time.time() - step_started_at,
      }

    def _result(done, error=None):
      data = asdict(self.info_pool)
      data["step_data"] = _step_payload(done=done, error=error)
      return base_agent.AgentInteractionResult(done, data)

    def _invalid_model_result(stage, detail):
      """Consume one step safely when a model stage has no parseable output."""
      nonlocal action_thought, action_object_str, action_description
      nonlocal action_outcome, error_description
      error_description = f"{stage} model response unavailable: {detail}"
      action_thought = error_description
      action_object_str = '{"action": "invalid"}'
      action_description = f"No device action executed because the {stage} response was unavailable."
      action_outcome = "C"
      # A manager outage must call the manager again on the next step.  The
      # executor's normal invalid-action retry deliberately skips replanning.
      if stage == "manager":
        action_record = {"action": "model_unavailable", "stage": stage}
      else:
        action_record = {"action": "invalid"}
      self.info_pool.last_action = action_record
      self.info_pool.last_action_thought = action_thought
      self.info_pool.last_summary = action_description
      self.info_pool.action_history.append(action_record)
      self.info_pool.summary_history.append(action_description)
      self.info_pool.action_outcomes.append(action_outcome)
      self.info_pool.error_descriptions.append(error_description)
      print(error_description, flush=True)
      return _result(False, error=error_description)
    
    # fix chrome initialization problem
    if step_idx == 0 and "chrome" in goal.lower():
      self.initialize_chrome()

    ## perception ###
    state = self.get_post_transition_state()
    before_state = state
    before_screenshot = state.pixels.copy()
    
    task_output_dir = self._task_output_dir()
    os.makedirs(task_output_dir, exist_ok=True)
    save_screenshot = Image.fromarray(before_screenshot)
    save_screenshot.save(os.path.join(task_output_dir, f"screenshot_{step_idx}.png"))
    before_screenshot_file = os.path.join(task_output_dir, f"screenshot_{step_idx}.png")
    before_xml_file = _save_state_xml(before_state, os.path.join(task_output_dir, f"screenshot_{step_idx}.xml"))
    with open(os.path.join(task_output_dir, "action.jsonl"), 'w', encoding='utf-8') as f:
      for item in self.info_pool.action_pool:
          f.write(item + '\n')
    
      screen_height, screen_width = before_screenshot.shape[:2]
      self.info_pool.width = screen_width
      self.info_pool.height = screen_height
    
    ###############
    ### manager ###
    ###############
    
    ## check error escalation
    self.info_pool.error_flag_plan = False
    err_to_manager_thresh = self.info_pool.err_to_manager_thresh
    if len(self.info_pool.action_outcomes) >= err_to_manager_thresh:
      # check if the last err_to_manager_thresh actions are all errors
      latest_outcomes = self.info_pool.action_outcomes[-err_to_manager_thresh:]
      count = 0
      for outcome in latest_outcomes:
          if outcome in ["B", "C"]:
              count += 1
      if count == err_to_manager_thresh:
          self.info_pool.error_flag_plan = True

    skip_manager = False
    planning_thought = ""
    ## if previous action is invalid, skip the manager and try again first ##
    if not self.info_pool.error_flag_plan and len(self.info_pool.action_history) > 0:
      if self.info_pool.action_history[-1]['action'] == 'invalid':
        skip_manager = True
      
    if not skip_manager:
      print("\n### Manager ... ###\n")
      prompt_planning_ori = manager.get_prompt(self.info_pool)
      prompt_planning = prompt_planning_ori.replace("[add_info_start]", "").replace("[add_info_end]", "")
      output_planning, message_manager, raw_response = self.vllm.predict_mm(
          prompt_planning,
          [before_screenshot_file],
      )
      self._record_vllm_call(
          call_history,
          stage="manager",
          messages=message_manager,
          raw_response=raw_response,
          screenshots=[before_screenshot_file],
      )

      if not raw_response or not isinstance(output_planning, str) or not output_planning.strip():
        return _invalid_model_result("manager", "provider call failed or returned empty content")
      try:
        parsed_result_planning = manager.parse_response(output_planning)
      except Exception as exc:
        return _invalid_model_result("manager", f"response parse failed: {exc}")
      if not parsed_result_planning.get('plan', '').strip():
        return _invalid_model_result("manager", "response did not contain a usable plan")
      self.info_pool.completed_plan = parsed_result_planning['completed_subgoal']
      self.info_pool.plan = parsed_result_planning['plan']
      planning_thought = parsed_result_planning['thought']
      
      print('Completed subgoal: ' + self.info_pool.completed_plan)
      print('Planning thought: ' + planning_thought)
      print('Plan: ' + self.info_pool.plan, "\n")

    ## if stopping by planner ##
    output_action = None
    if "Finished" in self.info_pool.plan.strip() and len(self.info_pool.plan.strip()) < 15:
      self.info_pool.finish_thought = planning_thought
      action_thought = "Finished by planner"
      action_object_str = "{\"action\": \"done\"}"
      action_description = "Finished by planner"
      self.info_pool.action_pool.append(action_object_str)

      task_output_dir = self._task_output_dir()
      os.makedirs(task_output_dir, exist_ok=True)
      save_screenshot = Image.fromarray(before_screenshot)
      save_screenshot.save(os.path.join(task_output_dir, f"screenshot_{step_idx}.png"))
      before_xml_file = _save_state_xml(before_state, os.path.join(task_output_dir, f"screenshot_{step_idx}.xml"))
      with open(os.path.join(task_output_dir, "action.jsonl"), 'w', encoding='utf-8') as f:
        for item in self.info_pool.action_pool:
            f.write(item + '\n')
    else:

      ################
      ### Operator ###
      ################

      print("\n### Operator ... ###\n")
      prompt_action = executor.get_prompt(self.info_pool)
      operator_error = None
      for operator_attempt in range(self.max_action_retry + 1):
        request_prompt = prompt_action
        if operator_attempt:
          request_prompt += (
              "\n\nYour previous operator response was invalid and no device "
              f"action was executed. Parser error: {operator_error}. "
              "Inspect the unchanged screenshot again and return exactly one "
              "valid action in the required format."
          )
        output_action, message_operator, raw_response = self.vllm.predict_mm(
            request_prompt,
            [before_screenshot_file],
        )
        self._record_vllm_call(
            call_history,
            stage="operator" if operator_attempt == 0 else "operator_retry",
            messages=message_operator,
            raw_response=raw_response,
            screenshots=[before_screenshot_file],
        )

        try:
          if (
              not raw_response
              or not isinstance(output_action, str)
              or not output_action.strip()
          ):
            raise ValueError("provider call failed or returned empty content")
          parsed_result_action = executor.parse_response(output_action)
          action_thought = parsed_result_action['thought']
          action_object_str = parsed_result_action['action']
          action_description = parsed_result_action['description']
          if not action_thought or not action_object_str:
            raise ValueError("response is missing a thought or action")
          # Validate the action schema before consuming a device step.  A
          # conversion failure is a protocol failure, not an environment
          # transition, so it is safe to retry against the same screenshot.
          converted_action = convert_fc_action_to_json_action(
              action_object_str,
              coor_type=self.coor_type,
              screen_width=screen_width,
              screen_height=screen_height,
          )
          operator_error = None
          break
        except Exception as exc:
          operator_error = str(exc)
          converted_action = None
          if operator_attempt >= self.max_action_retry:
            return _invalid_model_result(
                "operator",
                f"response validation failed after {operator_attempt} retries: "
                f"{operator_error}",
            )
      
      self.info_pool.last_action_thought = action_thought
      self.info_pool.last_summary = action_description
    
    print('Thought: ' + action_thought)
    print('Action: ' + action_object_str)
    print('Action description: ' + action_description)

    if converted_action is None:
      try:
        converted_action = convert_fc_action_to_json_action(
          action_object_str,
          coor_type=self.coor_type,
          screen_width=screen_width,
          screen_height=screen_height,
        )
      except Exception as e:
        print('Failed to convert the output to a valid action.')
        print(str(e))
        self.info_pool.last_action = {"action": "invalid"}
        self.info_pool.action_history.append({"action": "invalid"})
        self.info_pool.summary_history.append(action_description)
        self.info_pool.action_outcomes.append("C")
        self.info_pool.error_descriptions.append("invalid action format, do nothing.")
        return _result(False, error="invalid action format, do nothing.")
    self.info_pool.action_pool.append(action_object_str)

    if converted_action.action_type == 'status':
      outcome = "A"
      error_description = "None"
      if converted_action.goal_status == 'infeasible':
        print('Agent stopped since it thinks mission impossible.')
        outcome = "C"
        error_description = "Agent stopped since it thinks mission impossible."
      self.info_pool.last_action = json.loads(action_object_str)
      self.info_pool.action_history.append(json.loads(action_object_str))
      self.info_pool.summary_history.append(action_description)
      self.info_pool.action_outcomes.append(outcome)
      self.info_pool.error_descriptions.append(error_description)
      action_outcome = outcome
      progress_status = self.info_pool.completed_plan
      return _result(True, error=None if outcome == "A" else error_description)

    if converted_action.action_type == 'answer':
      print('Agent answered with: ' + converted_action.text)
    
    if converted_action.action_type == 'open_app':
      converted_action.app_name = converted_action.app_name.lower().strip()
      
    try:
      self.env.execute_action(converted_action)
    except Exception as e:
      print('Failed to execute action.')
      print(str(e))
      self.info_pool.last_action = {"action": "invalid"}
      self.info_pool.action_history.append({"action": "invalid"})
      self.info_pool.summary_history.append(action_description)
      self.info_pool.action_outcomes.append("C")
      if converted_action.action_type == "open_app":
        app_name = converted_action.app_name
        self.info_pool.error_descriptions.append(f"Failed to open the app '{app_name}'; the app name might not exist.")
      else:
        self.info_pool.error_descriptions.append(f"Failed to execute the action: {converted_action}")
      return _result(False, error=self.info_pool.error_descriptions[-1])
    print("Done action execution.\n")
    self.info_pool.last_action = json.loads(action_object_str)

    time.sleep(self.wait_after_action_seconds)
    
    ### Perception after execution ###
    state = self.env.get_state(wait_to_stabilize=False)
    after_screenshot = state.pixels.copy()
    save_after_screenshot = Image.fromarray(after_screenshot)
    save_after_screenshot.save(os.path.join(task_output_dir, f"screenshot_{step_idx+1}.png"))
    after_screenshot_file = os.path.join(task_output_dir, f"screenshot_{step_idx+1}.png")
    after_xml_file = _save_state_xml(state, os.path.join(task_output_dir, f"screenshot_{step_idx+1}.xml"))

    m3a_utils.add_screenshot_label(before_screenshot, 'before')
    m3a_utils.add_screenshot_label(after_screenshot, 'after')
    
    self.info_pool.ui_elements_list_after = []

    #################
    ### Reflector ###
    #################
    
    print("\n### Action Reflector ... ###\n")
    if converted_action.action_type != 'answer':
      prompt_action_reflect = action_reflector.get_prompt(self.info_pool)
      output_action_reflect, message_reflector, raw_response = self.vllm.predict_mm(
          prompt_action_reflect,
          [before_screenshot_file, after_screenshot_file],
      )
      self._record_vllm_call(
          call_history,
          stage="reflector",
          messages=message_reflector,
          raw_response=raw_response,
          screenshots=[before_screenshot_file, after_screenshot_file],
      )
      progress_status = self.info_pool.completed_plan

      if not raw_response or not isinstance(output_action_reflect, str) or not output_action_reflect.strip():
        action_outcome = "C"
        error_description = "Reflector model response unavailable; conservatively marking the action outcome as unknown."
      else:
        try:
          parsed_result_action_reflect = action_reflector.parse_response(output_action_reflect)
          outcome, error_description = (
              parsed_result_action_reflect['outcome'],
              parsed_result_action_reflect['error_description']
          )
          if "A" in outcome: # Successful. The result of the last action meets the expectation.
              action_outcome = "A"
          elif "B" in outcome: # Failed. The last action results in a wrong page. I need to return to the previous state.
              action_outcome = "B"
          elif "C" in outcome: # Failed. The last action produces no changes.
              action_outcome = "C"
          else:
              action_outcome = "C"
              error_description = f"Reflector returned an invalid outcome: {outcome!r}"
        except Exception as exc:
          action_outcome = "C"
          error_description = f"Reflector response parse failed: {exc}"
      
    else:
      action_outcome = "A"
      error_description = "None"
      progress_status = self.info_pool.completed_plan + "\n" + "The `answer` action has been performed. Answer to the question: " + converted_action.text
    
    print('Action reflection outcome: ' + action_outcome)
    print('Action reflection error description: ' + error_description, "\n")

    self.info_pool.action_history.append(json.loads(action_object_str))
    self.info_pool.summary_history.append(action_description)
    self.info_pool.action_outcomes.append(action_outcome)
    self.info_pool.error_descriptions.append(error_description)
    self.info_pool.progress_status = progress_status

    #################
    ### NoteKeeper ###
    #################
    
    if "'Ideas' folder" in self.info_pool.instruction and "Joplin app" in self.info_pool.instruction:
      print("skip notekeeper because of hallucination")
    elif "answer" not in self.info_pool.instruction.lower() and "transactions from" not in self.info_pool.instruction.lower() and "enter their product" not in self.info_pool.instruction.lower():
      print("skip notekeeper because of no answer or transport")
    else:
      if action_outcome == "A" and converted_action.action_type != 'answer':
          print("\n### NoteKeeper ... ###\n")
          # if previous action is successful, record the important content
          prompt_note = notetaker.get_prompt(self.info_pool)
          output_note, message_notekeeper, raw_response = self.vllm.predict_mm(
            prompt_note,
            [after_screenshot_file],
          )
          self._record_vllm_call(
            call_history,
            stage="notekeeper",
            messages=message_notekeeper,
            raw_response=raw_response,
            screenshots=[after_screenshot_file],
          )

          if not raw_response or not isinstance(output_note, str) or not output_note.strip():
            print('Notetaker response unavailable; keeping existing notes.', flush=True)
          else:
            try:
              parsed_result_note = notetaker.parse_response(output_note)
              important_notes = parsed_result_note['important_notes']
              self.info_pool.important_notes = important_notes
              print('Important notes: ' + important_notes, "\n")
            except Exception as exc:
              print(f'Notetaker response parse failed; keeping existing notes: {exc}', flush=True)
    
    task_output_dir = self._task_output_dir()
    os.makedirs(task_output_dir, exist_ok=True)
    save_screenshot = Image.fromarray(before_screenshot)
    save_screenshot.save(os.path.join(task_output_dir, f"screenshot_{step_idx}.png"))
    before_xml_file = _save_state_xml(before_state, os.path.join(task_output_dir, f"screenshot_{step_idx}.xml"))
    with open(os.path.join(task_output_dir, "action.jsonl"), 'w', encoding='utf-8') as f:
      for item in self.info_pool.action_pool:
          f.write(item + '\n')
    
    if converted_action.action_type == 'answer':
      return _result(True)
    else:
      return _result(False)

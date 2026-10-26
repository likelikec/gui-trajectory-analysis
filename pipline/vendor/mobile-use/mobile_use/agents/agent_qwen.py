import logging
import re
from typing import Iterator
import json
import copy
import os
from dataclasses import is_dataclass, fields as dataclass_fields

from mobile_use.schema.schema import *
from mobile_use.environment.mobile_environ import Environment
from mobile_use.utils.vlm import VLMWrapper
from mobile_use.utils.utils import encode_image_url, smart_resize, show_message, generate_message
from mobile_use.agents import Agent
from mobile_use.schema.config import QwenAgentConfig
from mobile_use.utils.constants import IMAGE_PLACEHOLDER
from mobile_use.default_prompts.prompt_type import load_prompt, QwenAgentPrompt

try:
    from study.gpt5_6_adapter import (
        is_gpt56_adapter,
        preserve_gpt56_raw_response,
    )
    from study.gpt_action_parser import parse_gpt_action
except ImportError:  # Keep the standalone mobile-use package usable.
    parse_gpt_action = None

    def is_gpt56_adapter(_vlm) -> bool:
        return False

    def preserve_gpt56_raw_response(record, _response):
        return record


logger = logging.getLogger(__name__)


_CONTINUATION_LANGUAGE = re.compile(
    r"\b(?:i(?:'ll| will| am going to| need to| should| must| have to)|next|now)\b.*\b"
    r"(?:open|launch|navigate|create|add|delete|edit|enter|type|tap|click|record|take|move|send|save|swipe|press|wait)\b"
    r"|^(?:open|launch|navigate|create|add|delete|edit|enter|type|tap|click|record|take|move|send|save|swipe|press|wait)\b",
    flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
)


def _contradictory_success_termination(
    *, step_idx: int, action: Action | None, thought: str | None, summary: str | None
) -> bool:
    """Reject a first-step success claim that explicitly describes pending work."""
    if step_idx != 0 or action is None or action.name != "terminate":
        return False
    if action.parameters.get("status") != "success":
        return False
    narrative = "\n".join(part for part in (thought, summary) if part)
    return bool(_CONTINUATION_LANGUAGE.search(narrative))


def _parse_gpt56_response(content: str) -> tuple[str | None, Action, str, str | None]:
    """Parse exactly one GPT action without silently choosing among candidates."""
    if parse_gpt_action is None:
        raise RuntimeError("GPT-5.6 action parser is unavailable")
    parsed = parse_gpt_action(content, reject_multiple=True)
    payload = dict(parsed.action)
    action_name = str(payload.pop("action"))
    action = Action(name=action_name, parameters=payload)
    action_s = json.dumps(parsed.action, ensure_ascii=False, separators=(",", ":"))
    return parsed.thought, action, action_s, parsed.description


def _map_action_names(name: str) -> str:
    maps = {
        "left_click": "click",
        "point": "coordinate",
        "start_point": "coordinate",
        "start_box": "coordinate",
        "end_point": "coordinate2",
        "end_box": "coordinate2",
        "scroll": "swipe",
        "content": "text",
        "open_app": "open",
    }
    return maps.get(name, name)


def _extract_qwen_response_parts(content: str) -> tuple[str | None, str | None, str]:
    thought = re.search(r'<thinking>(.*?)</thinking>', content, flags=re.DOTALL | re.IGNORECASE)
    if thought:
        thought_s = thought.group(1).strip()
    else:
        legacy_thought = re.search(r'Thought:\s*(.*?)(?:\n\s*Action:|\Z)', content, flags=re.DOTALL | re.IGNORECASE)
        thought_s = legacy_thought.group(1).strip() if legacy_thought else None

    summary = re.search(r'<conclusion>(.*?)</conclusion>', content, flags=re.DOTALL | re.IGNORECASE)
    if summary:
        summary_s = summary.group(1).strip()
    else:
        legacy_summary = re.search(r'Action:\s*(.*?)(?=\n?\s*<tool_call>)', content, flags=re.DOTALL | re.IGNORECASE)
        summary_s = legacy_summary.group(1).strip() if legacy_summary else None

    action = re.search(r'<tool_call>(.*?)</tool_call>', content, flags=re.DOTALL | re.IGNORECASE)
    if not action:
        raise Exception("Cannot extract action in the content.")
    action_s = action.group(1).strip()

    return thought_s, summary_s, action_s


def _scale_coordinate(
    value: list[float] | tuple[float, float],
    raw_size: tuple[float, float],
    coordinate_base: tuple[float, float],
) -> tuple[int, int]:
    x = round(value[0] / coordinate_base[0] * raw_size[0])
    y = round(value[1] / coordinate_base[1] * raw_size[1])
    max_x = max(0, int(raw_size[0]) - 1)
    max_y = max(0, int(raw_size[1]) - 1)
    return min(max(x, 0), max_x), min(max(y, 0), max_y)


def _parse_qwen_tool_call(content: str, raw_size: tuple[float, float], coordinate_base: tuple[float, float]) -> Action:
    thought_s, summary_s, action_s = _extract_qwen_response_parts(content)
    action = json.loads(action_s)
    name = _map_action_names(action['arguments']['action'])

    # Remove the 'action' key and map the other keys in the arguments
    action['arguments'].pop('action')
    params = {}

    for k, v in action['arguments'].items():
        mapped_key = _map_action_names(k)
        if mapped_key in ['coordinate', 'coordinate2']:
            try:
                params[mapped_key] = _scale_coordinate(v, raw_size, coordinate_base)
            except:
                pass
        else:
            params[mapped_key] = v

    action_a = Action(name=name, parameters=params)
    return thought_s, action_a, action_s, summary_s


def _parse_response(content: str, size: tuple[float, float], raw_size: tuple[float, float]) -> Action:
    thought = re.search(r'<thinking>(.*?)</thinking>', content, flags=re.DOTALL)
    if thought:
        thought_s = thought.group(1).strip()
    else:
        thought_s = None
    summary = re.search(r'<conclusion>(.*?)</conclusion>', content, flags=re.DOTALL)
    if summary:
        summary_s = summary.group(1).strip()
    else:
        summary_s = None
    action = re.search(r'<tool_call>(.*?)</tool_call>', content, flags=re.DOTALL)
    if not action:
        raise Exception("Cannot extract action in the content.")
    action_s = action.group(1).strip()
    action = json.loads(action_s)
    name = _map_action_names(action['arguments']['action'])

    # Remove the 'action' key and map the other keys in the arguments
    action['arguments'].pop('action')
    params = {}

    for k, v in action['arguments'].items():
        mapped_key = _map_action_names(k)  # Map the key name
        if mapped_key in ['coordinate', 'coordinate2']:
            try:
                params[mapped_key] = _scale_coordinate(v, raw_size, size)
            except:
                pass
        else:
            params[mapped_key] = v

    action_a = Action(name=name, parameters=params)
    return thought_s, action_a, action_s, summary_s

def _parse_response_qwen3(
    content: str,
    size: tuple[float, float],
    raw_size: tuple[float, float],
    coordinate_base: float = 999,
) -> Action:
    del size
    return _parse_qwen_tool_call(
        content,
        raw_size=raw_size,
        coordinate_base=(coordinate_base, coordinate_base),
    )

def slim_messages(messages, num_image_limit = 5):
    keep_image_index = []
    image_ptr = 0
    messages = copy.deepcopy(messages)
    for msg in messages:
        for content in msg['content']:
            if 'image' in content['type'] or 'image_url' in content['type']:
                keep_image_index.append(image_ptr)
                image_ptr += 1
    keep_image_index = keep_image_index[-num_image_limit:]

    image_ptr = 0
    for msg in messages:
        new_content = []
        for content in msg['content']:
            if 'image' in content['type'] or 'image_url' in content['type']:
                if image_ptr not in keep_image_index:
                    pass
                else:
                    new_content.append(content)
                image_ptr += 1
            else:
                new_content.append(content)
        msg['content'] = new_content
    return messages


@Agent.register('Qwen')
class QwenAgent(Agent):
    def __init__(self, config_path: str = None, **kwargs):
        super().__init__(config_path, **kwargs)
        if config_path is not None:
            self.config = QwenAgentConfig.from_yaml(config_path)
        else:
            self.config = QwenAgentConfig(**kwargs)

        self.max_action_retry = self.config.max_action_retry
        self.enable_think = self.config.enable_think
        self.min_pixels = self.config.min_pixels
        self.max_pixels = self.config.max_pixels
        self.message_type = self.config.message_type
        self.num_image_limit = self.config.num_image_limit
        self.coordinate_type = self.config.coordinate_type
        self.coordinate_base = self.config.coordinate_base
        self.prompt: QwenAgentPrompt = load_prompt("qwen_agent", self.config.prompt_config)

    def _init_data(self, goal: str=''):
        super()._init_data(goal)
        self.trajectory: List[SingleAgentStepData] = []
        self.episode_data: BaseEpisodeData = BaseEpisodeData(goal=goal, num_steps=0, trajectory=self.trajectory)
        self.messages: List[Dict[str,Any]] = []

    def reset(self, goal: str='', max_steps: int = None) -> None:
        """Reset the state of the agent.
        """
        self._init_data(goal=goal)
        if isinstance(max_steps, int):
            self.set_max_steps(max_steps)

    def _get_curr_step_data(self) -> SingleAgentStepData:
        if len(self.trajectory) > self.curr_step_idx:
            return self.trajectory[self.curr_step_idx]
        else:
            return None

    def step(self) -> SingleAgentStepData:
        """Execute the task with maximum number of steps.

        Returns: Answer
        """
        logger.info("Step %d ... ..." % self.curr_step_idx)
        show_step = [0,4]

        # Get the current environment screen
        env_state = self.env.get_state()
        pixels = env_state.pixels
        raw_size = pixels.size
        resized_height, resized_width = smart_resize(
            height=pixels.height,
            width=pixels.width,
            factor=28,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,)
        # The isolated GPT absolute-coordinate protocol uses the fixed raw
        # AndroidWorld screen. Avoid Qwen's virtual resize grid and pass the
        # model's 1080x2400 pixels through without a second scale transform.
        preserve_raw_pixels = bool(
            getattr(getattr(self.vlm, "adapter", None), "preserve_raw_pixel_coordinates", False)
        )
        if (
            getattr(self.vlm, "adapter_name", "") == "gpt-absolute"
            or is_gpt56_adapter(self.vlm)
            or preserve_raw_pixels
        ):
            resized_width, resized_height = raw_size
        # pixels = pixels.resize((resized_width, resized_height))

        # Add new step data
        self.trajectory.append(MobileUseStepData(
            step_idx=self.curr_step_idx,
            curr_env_state=env_state,
        ))
        step_data = self.trajectory[-1]

        if self.curr_step_idx == 0:
            # Add system prompt
            system_prompt = self.prompt.system_prompt.format(
                width = resized_width,
                height = resized_height,
            )
            system_message = generate_message("system", system_prompt)
            self.messages.append(system_message)

            # Add user prompt
            user_prompt = self.prompt.task_prompt.format(goal=self.episode_data.goal)
            if self.enable_think:
                user_prompt += '\n' + self.prompt.thinking_prompt
            user_prompt += f"\n{IMAGE_PLACEHOLDER}"
            user_message = generate_message("user", user_prompt, images=[pixels])
            self.messages.append(user_message)

        if self.message_type == 'single':
            user_prompt = self.prompt.task_prompt.format(goal=self.episode_data.goal)
            history = [str(step.summary) for step in self.trajectory[:-1]]
            history = ''.join([f'Step {si+1}: {_}; 'for si, _ in enumerate(history)])
            user_prompt += '\n' + self.prompt.history_prompt.format(history=history) + '\n'
            if self.enable_think:
                user_prompt += '\n' + self.prompt.thinking_prompt
            user_prompt += f"\n{IMAGE_PLACEHOLDER}"
            user_message = generate_message("user", user_prompt, images=[pixels])
            self.messages[1] = user_message

        if self.message_type == 'chat' and self.curr_step_idx > 0:
            last_step = self.trajectory[self.curr_step_idx - 1]
            assistant_message = generate_message("assistant", last_step.content)
            self.messages.append(assistant_message)

            user_prompt = ""
            if self.enable_think:
                user_prompt += '\n' + self.prompt.thinking_prompt
            user_prompt += f"\n{IMAGE_PLACEHOLDER}"
            user_message = generate_message("user", user_prompt, images=[pixels])
            self.messages.append(user_message)
            
            self.messages = slim_messages(self.messages, num_image_limit=self.num_image_limit)

        # Call VLM
        if self.curr_step_idx in show_step:
            show_message(self.messages, "Qwen")
        response = self.vlm.predict(self.messages)
        call_record = self.vlm.build_call_record(self.messages, response, stage="action")
        if is_gpt56_adapter(self.vlm):
            preserve_gpt56_raw_response(call_record, response)
        step_data.vlm_call_history.append(call_record)

        # parse the response
        thought_s, action, action_s, summary_s = None, None, None, None
        gpt56_mode = is_gpt56_adapter(self.vlm)
        last_parse_error = None
        def parse_current_response(raw_action):
            nonlocal thought_s, action, action_s, summary_s
            if gpt56_mode:
                thought_s, action, action_s, summary_s = _parse_gpt56_response(
                    raw_action
                )
            elif self.coordinate_type == 'relative':
                thought_s, action, action_s, summary_s = _parse_response_qwen3(
                    raw_action,
                    (resized_width, resized_height),
                    raw_size,
                    self.coordinate_base,
                )
            else:
                thought_s, action, action_s, summary_s = _parse_response(
                    raw_action, (resized_width, resized_height), raw_size
                )
            if gpt56_mode and _contradictory_success_termination(
                step_idx=self.curr_step_idx,
                action=action,
                thought=thought_s,
                summary=summary_s,
            ):
                action = None
                raise ValueError("terminate_success_contradicts_pending_first_step_action")

        if gpt56_mode:
            # GPT-only path: every generated retry response is parsed.  The
            # existing non-GPT loop is intentionally left untouched.
            retry_count = 0
            while True:
                try:
                    raw_action = response.choices[0].message.content
                    logger.info("Action from VLM:\n%s" % raw_action)
                    step_data.content = raw_action
                    parse_current_response(raw_action)
                    logger.info(f"Thought: {thought_s}")
                    logger.info(f"Action: {action}")
                    logger.info(f"Action string: {action_s}")
                    logger.info(f"Summary: {summary_s}")
                    break
                except Exception as e:
                    action = None
                    last_parse_error = str(e.args[0] if e.args else e)
                    logger.warning(f"Failed to parse the action. Error is {e.args}")
                    if retry_count >= self.max_action_retry:
                        break
                    error_prompt = (
                        f"Failed to parse the action. Error is {e.args}\n"
                        "Please follow the output format to provide a valid action:"
                    )
                    self.messages.append(
                        {"role": "user", "content": [{"type": "text", "text": error_prompt}]}
                    )
                    response = self.vlm.predict(self.messages)
                    call_record = self.vlm.build_call_record(
                        self.messages, response, stage="action_retry"
                    )
                    preserve_gpt56_raw_response(call_record, response)
                    step_data.vlm_call_history.append(call_record)
                    retry_count += 1
            if retry_count:
                self.messages = self.messages[:-retry_count]
        else:
            counter = self.max_action_retry
            while counter > 0:
                try:
                    raw_action = response.choices[0].message.content
                    logger.info("Action from VLM:\n%s" % raw_action)
                    step_data.content = raw_action
                    parse_current_response(raw_action)
                    logger.info(f"Thought: {thought_s}")
                    logger.info(f"Action: {action}")
                    logger.info(f"Action string: {action_s}")
                    logger.info(f"Summary: {summary_s}")
                    break
                except Exception as e:
                    action = None
                    logger.warning(f"Failed to parse the action. Error is {e.args}")
                    error_prompt = (
                        f"Failed to parse the action. Error is {e.args}\n"
                        "Please follow the output format to provide a valid action:"
                    )
                    self.messages.append(
                        {"role": "user", "content": [{"type": "text", "text": error_prompt}]}
                    )
                    response = self.vlm.predict(self.messages)
                    step_data.vlm_call_history.append(
                        self.vlm.build_call_record(
                            self.messages, response, stage="action_retry"
                        )
                    )
                    counter -= 1
            if counter != self.max_action_retry:
                self.messages = self.messages[:-(self.max_action_retry - counter)]

        if action is None:
            reason = last_parse_error or "unknown_action_parse_error"
            logger.warning("Action parse error after max retry: %s", reason)
            if gpt56_mode:
                # Keep the next-step history serializable without changing the
                # shared history builder used by non-GPT agents.
                step_data.action_desc = f"GPT protocol failure: {reason}"
                step_data.action_s = json.dumps(
                    {"action": "protocol_error", "reason": reason},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
        else:
            if action.name == 'terminate':
                if action.parameters['status'] == 'success':
                    logger.info(f"Finished: {action}")
                    self.status = AgentStatus.FINISHED
                elif action.parameters['status'] == 'failure':
                    logger.info(f"Failed: {action}")
                    self.status = AgentStatus.FAILED
            elif action.name == 'answer':
                logger.info(f"Answer: {action}")
                answer = action.parameters['text'].strip()
                step_data.answer = answer
                logger.info("Terminate the task after answering question.")
                self.status = AgentStatus.FINISHED
            else:
                logger.info(f"Execute the action: {action}")
                try:
                    self.env.execute_action(action)
                except Exception as e:
                    logger.warning(f"Failed to execute the action: {action}. Error: {e}")
                    action = None
                step_data.exec_env_state = self.env.get_state()

        if action is not None:
            step_data.action = action
            step_data.thought = thought_s
            step_data.action_s = action_s
            step_data.summary = summary_s

        # persist logs at the end of the step
        try:
            if self.enable_log:
                self._save_episode_json()
                self._save_step_log(step_data)
        except Exception as e:
            logger.warning(f"Failed to save logs for step {self.curr_step_idx}: {e}")

        return step_data


    def iter_run(self, input_content: str) -> Iterator[SingleAgentStepData]:
        """Execute the agent with user input content.

        Returns: Iterator[SingleAgentStepData]
        """

        if self.state == AgentState.READY:
            self.reset(goal=input_content)
            logger.info("Start task: %s, with at most %d steps" % (self.goal, self.max_steps))
        elif self.state == AgentState.CALLUSER:
            self._user_input = input_content      # user answer
            self.state = AgentState.RUNNING       # reset agent state
            logger.info("Continue task: %s, with user input %s" % (self.goal, input_content))
        else:
            raise Exception('Error agent state')

        for step_idx in range(self.curr_step_idx, self.max_steps):
            self.curr_step_idx = step_idx
            # show init environment
            yield SingleAgentStepData(
                step_idx=self.curr_step_idx,
                curr_env_state=self.env.get_state(),
                vlm_call_history=[]
            )
            try:
                self.step()
            except Exception as e:
                self.status = AgentStatus.FAILED
                self.episode_data.status = self.status
                self.episode_data.message = str(e)
                yield self._get_curr_step_data()
                return

            self.episode_data.num_steps = step_idx + 1
            self.episode_data.status = self.status

            if self.status == AgentStatus.FINISHED:
                logger.info("Agent indicates task is done.")
                self.episode_data.message = 'Agent indicates task is done'
                yield self._get_curr_step_data()
                return
            elif self.state == AgentState.CALLUSER:
                logger.info("Agent indicates to ask user for help.")
                yield self._get_curr_step_data()
                return
            else:
                logger.info("Agent indicates one step is done.")
            yield self._get_curr_step_data()
        logger.warning(f"Agent reached max number of steps: {self.max_steps}.")

    def run(self, input_content: str) -> BaseEpisodeData:
        """Execute the agent with user input content.

        Returns: EpisodeData
        """
        for _ in self.iter_run(input_content):
            pass
        return self.episode_data

    # =========================
    # Logging helpers
    # =========================
    def _to_jsonable(self, obj):
        # Dataclasses
        if is_dataclass(obj):
            result = {}
            for f in dataclass_fields(obj):
                value = getattr(obj, f.name)
                result[f.name] = self._to_jsonable(value)
            return result
        # Enums
        if isinstance(obj, Enum):
            return getattr(obj, "value", None) or getattr(obj, "name", None) or str(obj)
        # Basic types
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        # Dicts
        if isinstance(obj, dict):
            return {str(self._to_jsonable(k)): self._to_jsonable(v) for k, v in obj.items()}
        # Iterables
        if isinstance(obj, (list, tuple, set)):
            return [self._to_jsonable(v) for v in obj]
        # PIL Images or other non-serializable objects
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)

    def _episode_meta_dict(self) -> Dict[str, Any]:
        meta = self._to_jsonable(self.episode_data)
        # Exclude trajectory by requirement
        if isinstance(meta, dict) and "trajectory" in meta:
            meta.pop("trajectory", None)
        return meta

    def _save_episode_json(self) -> None:
        if not self.enable_log or not self._episode_log_dir:
            return
        os.makedirs(self._episode_log_dir, exist_ok=True)
        episode_path = os.path.join(self._episode_log_dir, "episode.json")
        with open(episode_path, "w", encoding="utf-8") as f:
            json.dump(self._episode_meta_dict(), f, ensure_ascii=False, indent=2)

    def _serialize_step_data(self, step_data: SingleAgentStepData) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, v in step_data.__dict__.items():
            if k in ("curr_env_state", "exec_env_state"):
                continue
            result[k] = self._to_jsonable(v)
        return result

    def _save_step_log(self, step_data: SingleAgentStepData) -> None:
        if not self.enable_log or not self._episode_log_dir:
            return
        step_idx_1 = (step_data.step_idx if isinstance(step_data.step_idx, int) else self.curr_step_idx) + 1
        # Save images in root episode dir with step-indexed filenames
        try:
            if step_data.curr_env_state and getattr(step_data.curr_env_state, "pixels", None) is not None:
                step_data.curr_env_state.pixels.save(
                    os.path.join(self._episode_log_dir, f"step_{step_idx_1}_curr_env.png")
                )
        except Exception as e:
            logger.warning(f"Failed to save curr_env image for step {step_idx_1}: {e}")
        try:
            if step_data.exec_env_state and getattr(step_data.exec_env_state, "pixels", None) is not None:
                step_data.exec_env_state.pixels.save(
                    os.path.join(self._episode_log_dir, f"step_{step_idx_1}_exec_env.png")
                )
        except Exception as e:
            logger.warning(f"Failed to save exec_env image for step {step_idx_1}: {e}")
        # Save step json in root episode dir with step-indexed filename
        step_json = os.path.join(self._episode_log_dir, f"step_{step_idx_1}.json")
        with open(step_json, "w", encoding="utf-8") as f:
            json.dump(self._serialize_step_data(step_data), f, ensure_ascii=False, indent=2, default=str)

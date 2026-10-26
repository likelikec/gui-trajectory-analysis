from abc import ABC, abstractmethod
import re
import json
import logging
from typing import Union, Literal
import os, sys
from mobile_use.default_prompts.prompt_type import *
from mobile_use.utils.vlm import VLMWrapper
from mobile_use.schema.schema import *
from mobile_use.utils.constants import IMAGE_PLACEHOLDER
from mobile_use.utils.utils import *
from mobile_use.schema.config import *
import mobile_use.agents.agent_qwen as agent_qwen

try:
    from study.gpt5_6_adapter import is_gpt56_adapter
except ImportError:  # Keep the standalone mobile-use package usable.
    def is_gpt56_adapter(_vlm) -> bool:
        return False

__all__ = [
    "SubAgent",
    "Planner",
    "Operator",
    "TrainedOperator",
    "OperatorQwen",
    "OperatorGPT",
    "AnswerAgent",
    "TrainedAnswerAgent",
    "AnswerAgentQwen",
    "AnswerAgentGPT",
    "Reflector",
    "TrajectoryReflector",
    "GlobalReflector",
    "Progressor",
    "NoteTaker",
    "TaskClassifier",
    "TaskOrchestrator",
    "TaskExtractor",
    "TaskRewriter",
]

logger = logging.getLogger(__name__)


def get_history(
    trajectory: List[MobileUseStepData],
    num_histories=None,
    wrap_tool_call: bool = True,
):
    start_idx = 0 if num_histories is None else max(0, len(trajectory) - num_histories)
    history = []
    for i in range(start_idx, len(trajectory)):
        step_list = []
        step_list.append(f"Action: {trajectory[i].action_desc}")
        action_s = trajectory[i].action_s
        step_list.append(
            f"<tool_call> {action_s} </tool_call>" if wrap_tool_call else action_s
        )
        if trajectory[i].summary is not None:
            step_list.append(f"Summary: {trajectory[i].summary}")
        if trajectory[i].reflection_outcome is not None:
            if trajectory[i].reflection_outcome == "A":
                step_list.append("Successful")
            elif trajectory[i].reflection_outcome in ["B", "C"]:
                step_list.append("Failed")
                step_list.append(f"Feedback: {trajectory[i].reflection_error}")
        elif trajectory[i].trajectory_reflection_outcome is not None:
            if trajectory[i].trajectory_reflection_outcome == "A":
                step_list.append("Successful")
            elif trajectory[i].trajectory_reflection_outcome in ["B"]:
                step_list.append("Failed")
                step_list.append(f"Feedback: {trajectory[i].trajectory_reflection_error}")
        history.append(f"Step-{i+1}: {'; '.join(step_list)}")
    return history

def get_history_action_desc(trajectory: List[MobileUseStepData], num_histories=None):
    start_idx = 0 if num_histories is None else max(0, len(trajectory) - num_histories)
    history = []
    for i in range(start_idx, len(trajectory)):
        history.append(f"Step-{i+1}: {trajectory[i].action_desc}")
    return history

def map_action_names(name: str) -> str:
    maps = {
        "tap": "click",
        "left_click": "click",
        "point": "coordinate",
        "start_point": "coordinate",
        "start_box": "coordinate",
        "end_point": "coordinate2",
        "end_box": "coordinate2",
        "start_coordinate": "coordinate",
        "end_coordinate": "coordinate2",
        "scroll": "swipe",
        "content": "text",
        "open_app": "open",
    }
    return maps.get(name, name)


def _repair_incomplete_json_object(fragment: str) -> dict | None:
    """Close only structurally incomplete JSON; never guess values or strings."""
    opening_to_closing = {"{": "}", "[": "]"}
    stack = []
    repaired = []
    in_string = False
    escaped = False

    for character in fragment:
        if in_string:
            repaired.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
            repaired.append(character)
            continue
        if character in opening_to_closing:
            stack.append(character)
            repaired.append(character)
            continue
        if character in opening_to_closing.values():
            if not stack:
                break
            matching_depth = next(
                (
                    depth
                    for depth in range(len(stack) - 1, -1, -1)
                    if opening_to_closing[stack[depth]] == character
                ),
                None,
            )
            if matching_depth is None:
                return None
            while len(stack) - 1 > matching_depth:
                repaired.append(opening_to_closing[stack.pop()])
            repaired.append(character)
            stack.pop()
            if not stack:
                break
            continue
        repaired.append(character)

    if in_string or not repaired or repaired[0] != "{":
        return None
    while stack:
        repaired.append(opening_to_closing[stack.pop()])
    try:
        value = json.loads("".join(repaired))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _iter_json_objects(content: str):
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            value = _repair_incomplete_json_object(content[index:])
        if isinstance(value, dict):
            yield value


def _clear_stale_knowledge_for_query(sub_agent, query: str) -> None:
    if getattr(sub_agent, "_knowledge_query", None) == query:
        return
    sub_agent.retrieved_knowledge = None
    sub_agent.explored_knowledge = None
    sub_agent._knowledge_query = query


def _gpt_action_arguments(candidate: dict) -> dict | None:
    if isinstance(candidate.get("arguments"), dict):
        name = candidate.get("name")
        if name in {None, "mobile_use"}:
            return dict(candidate["arguments"])
    function = candidate.get("function")
    if isinstance(function, dict) and function.get("name") == "mobile_use":
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if isinstance(arguments, dict):
            return dict(arguments)
    if isinstance(candidate.get("action"), str):
        return dict(candidate)
    return None


def _scale_gpt_coordinate(
    value,
    *,
    size: tuple[float, float],
    raw_size: tuple[float, float],
) -> tuple[int, int]:
    if isinstance(value, dict):
        value = (value.get("x"), value.get("y"))
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"invalid coordinate: {value!r}")
    x = round(float(value[0]) / size[0] * raw_size[0])
    y = round(float(value[1]) / size[1] * raw_size[1])
    return (
        min(max(x, 0), max(0, int(raw_size[0]) - 1)),
        min(max(y, 0), max(0, int(raw_size[1]) - 1)),
    )


def parse_gpt_action_response(
    content: str,
    size: tuple[float, float],
    raw_size: tuple[float, float],
):
    """Parse GPT direct JSON while retaining legacy mobile_use wrappers."""
    arguments = None
    for candidate in _iter_json_objects(content or ""):
        arguments = _gpt_action_arguments(candidate)
        if arguments is not None:
            break
    if arguments is None:
        raise Exception("Cannot extract GPT action JSON from the content.")

    raw_action = arguments.pop("action", None) or arguments.pop("name", None)
    if not isinstance(raw_action, str):
        raise Exception("GPT action JSON is missing a string action.")
    name = map_action_names(raw_action)

    if "coordinate" not in arguments and "x" in arguments and "y" in arguments:
        arguments["coordinate"] = [arguments.pop("x"), arguments.pop("y")]
    if "coordinate" not in arguments and "x1" in arguments and "y1" in arguments:
        arguments["coordinate"] = [arguments.pop("x1"), arguments.pop("y1")]
    if "coordinate2" not in arguments and "x2" in arguments and "y2" in arguments:
        arguments["coordinate2"] = [arguments.pop("x2"), arguments.pop("y2")]

    params = {}
    for key, value in arguments.items():
        mapped_key = map_action_names(key)
        if mapped_key in {"coordinate", "coordinate2"}:
            params[mapped_key] = _scale_gpt_coordinate(
                value,
                size=size,
                raw_size=raw_size,
            )
        else:
            params[mapped_key] = value

    thought_match = re.search(r"(?:Thought|Reasoning):\s*(.*?)(?=\n|\Z)", content, re.I)
    description_match = re.search(
        r"Action description:\s*(.*?)(?=\n|\Z)", content, re.I
    )
    action_s = json.dumps(
        {"action": name, **params},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        thought_match.group(1).strip() if thought_match else None,
        Action(name=name, parameters=params),
        action_s,
        description_match.group(1).strip() if description_match else name,
    )


class SubAgent(ABC):
    def __init__(self, config: SubAgentConfig):
        super().__init__()
        self.vlm = VLMWrapper(**config.vlm.model_dump())
        self.reset()

    def reset(self):
        pass

    @abstractmethod
    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        pass

    @abstractmethod
    def parse_response(self, response: str):
        pass


"""
Call in the beginning of each step.
"""
class Planner(SubAgent):
    def __init__(self, config: PlannerConfig):
        super().__init__(config)
        self.prompt: PlannerPrompt = load_prompt("planner", config.prompt_config)

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]

        pixels = current_step.curr_env_state.pixels.copy()
        resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)
        
        # Add system prompt
        system_message = generate_message("system", self.prompt.system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []
        task_prompt = self.prompt.task_prompt.format(
            task_description = episodedata.goal,
            screenshot = IMAGE_PLACEHOLDER,
            resized_width = resized_width,
            resized_height = resized_height,
        )
        prompt_list.append(task_prompt)

        if len(trajectory) == 1:
            init_plan = self.prompt.init_plan
            prompt_list.append(init_plan)
        else:
            previous_step = trajectory[-2]
            continue_plan = self.prompt.continue_plan.format(
                current_plan = previous_step.plan,
                previous_subgoal = previous_step.sub_goal,
            )
            prompt_list.append(continue_plan)

        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt, images=[pixels])
        messages.append(user_message)

        return messages

    def parse_response(self, response: str):
        thought = response.split("### Thought ###")[-1].split("### Plan ###")[0].replace("\n", " ").replace("  ", " ").strip()
        plan = response.split("### Plan ###")[-1].split("### Current Subgoal ###")[0].replace("\n", " ").replace("  ", " ").strip()
        current_subgoal = response.split("### Current Subgoal ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return thought, plan, current_subgoal


class Operator(SubAgent):
    wrap_history_tool_calls = True

    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        self.prompt: OperatorPrompt = load_prompt("operator", config.prompt_config)
        self.num_histories = config.num_histories
        self.include_device_time = config.include_device_time
        self.include_tips = config.include_tips
        self.include_a11y_tree = config.include_a11y_tree
        self.max_pixels = config.max_pixels
        self.coordinate_type = config.coordinate_type
        self.coordinate_base = config.coordinate_base
        self.knowledge_config = config.knowledge

        self.embedding_model = None
        self.db = None
        self.retrieved_knowledge = None
        self.explored_knowledge = None

    def _coordinate_prompt_kwargs(self, resized_width: int, resized_height: int) -> dict[str, str]:
        if self.coordinate_type == "relative":
            coordinate_base = self.coordinate_base
            return {
                "coordinate_system_line": f"The tool coordinate system is normalized to {coordinate_base}x{coordinate_base}.",
                "coordinate_output_line": f"For every click, long_press, and swipe coordinate, output normalized coordinates from 0 to {coordinate_base}, not screenshot pixel coordinates.",
                "coordinate_kind": "normalized",
                "coordinate_schema_description": f"(x, y): Normalized coordinates from 0 to {coordinate_base}. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
                "coordinate2_schema_description": f"(x, y): Normalized coordinates from 0 to {coordinate_base}. Required only by `action=swipe`.",
                "coordinate_observation_line": f"Use normalized coordinates from 0 to {coordinate_base} for all touch actions, regardless of the screenshot's pixel size.",
            }
        return {
            "coordinate_system_line": f"The original tool screen resolution is {resized_width}x{resized_height}.",
            "coordinate_output_line": f"For every click, long_press, and swipe coordinate, output absolute pixels mapped to this {resized_width}x{resized_height} resolution. The image service may resize the screenshot internally, so map the visually located point back before responding.",
            "coordinate_kind": "pixel",
            "coordinate_schema_description": f"(x, y): Absolute pixels in the stated resolution; x is 0..{resized_width - 1} and y is 0..{resized_height - 1}. Click the target center. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
            "coordinate2_schema_description": f"(x, y): Absolute pixels in the stated resolution; x is 0..{resized_width - 1} and y is 0..{resized_height - 1}. Required only by `action=swipe`.",
            "coordinate_observation_line": f"Return absolute pixels for the {resized_width}x{resized_height} tool screen, even if the supplied image was internally resized.",
        }
    
    def get_knowledge(self, query: str):
        if not self.knowledge_config:
            return
        _clear_stale_knowledge_for_query(self, query)
        # Retrieve knowledge from RAG database
        embedding_model_path = self.knowledge_config.embedding_model_path
        knowledge_database_dir = self.knowledge_config.knowledge_database_dir
        if knowledge_database_dir is not None:
            if self.retrieved_knowledge is None:
                assert embedding_model_path is not None, "Embedding model path must be provided if knowledge database dir is provided."
                if self.embedding_model is None:
                    logger.info("Loading RAG database for knowledge retrieval...")
                    project_home = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    sys.path += [os.path.join(project_home, 'third_party', 'RAGToolbox')]
                    from RAGToolbox import Jinaembedding, Vectordatabase
                    download_hf_model("https://huggingface.co/jinaai/jina-embeddings-v2-base-zh", embedding_model_path)
                    self.embedding_model=Jinaembedding(embedding_model_path)
                    self.db=Vectordatabase()
                    self.db.load_vector(knowledge_database_dir)
                answer = self.db.query_score(query, self.embedding_model,1)
                similarity, key, value = answer[0]
                logger.info(f"Retrieved knowledge: {str(value)}")
                if len(value) > 0 and value[0] != "":
                    self.retrieved_knowledge = '\n'.join([f"{i+1}. {v}" for i, v in enumerate(value)])

        # Get explored knowledge
        explored_knowledge_path = self.knowledge_config.explored_knowledge_path
        if explored_knowledge_path is not None:
            if self.explored_knowledge is None:
                all_explored_knowledge = json.load(open(explored_knowledge_path, 'r', encoding='utf-8'))
                app_names = list(all_explored_knowledge.keys())

                messages = [generate_message("user", "Which app is most relevant to the following query: " + query + "\nApp names: " + ", ".join(app_names) + ". \nAnswer with only the app name:")]
                response = self.vlm.predict(messages)
                app_name = response.choices[0].message.content.strip()
                if app_name in all_explored_knowledge:
                    knowledge_list = all_explored_knowledge[app_name]
                    explored_app_knowledge = '\n'.join([f"{i+1}. {v}" for i, v in enumerate(knowledge_list)])
                    logger.info(f"Explored knowledge from app {app_name} is added.")
                    messages = [generate_message("user", f"The summary of pages explored:\n{explored_app_knowledge}\nuser query:\n{query}\nPlease extract the text snippet that helps complete the user's request without exceeding 100 tokens and must keeping the original description. Don't answer user query! Extract only!")]
                    response = self.vlm.predict(messages)
                    self.explored_knowledge = response.choices[0].message.content.strip()
                else:
                    logger.warning(f"App name {app_name} not found in explored knowledge.")

    def reset(self):
        self.raw_size = None
        self.resized_size = None
        self.retrieved_knowledge = None
        self.embedding_model = None
        self.db = None
        self.explored_knowledge = None
        self._knowledge_query = None

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]
        
        pixels = current_step.curr_env_state.pixels.copy()
        self.raw_size = (pixels.width, pixels.height)
        preserve_raw_coordinates = bool(
            getattr(self.vlm, "preserve_raw_pixel_coordinates", False)
        )
        if self.max_pixels is not None and not preserve_raw_coordinates:
            pixels = resize_image(pixels, self.max_pixels)
        resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)
        # Absolute adapters return device pixels already. Keep prompt,
        # parser, and executor in that same space instead of applying the
        # legacy smart-resize conversion a second time.
        if (
            preserve_raw_coordinates
            or getattr(self.vlm, "adapter_name", "") == "gpt-absolute"
            or is_gpt56_adapter(self.vlm)
        ):
            resized_width, resized_height = self.raw_size
        self.resized_size = (resized_width, resized_height)

        # Add system prompt
        coordinate_prompt_kwargs = self._coordinate_prompt_kwargs(resized_width, resized_height)
        system_prompt = self.prompt.system_prompt.format(
            resized_width = resized_width,
            resized_height = resized_height,
            **coordinate_prompt_kwargs,
        )
        system_message = generate_message("system", system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []
        task_prompt = self.prompt.task_prompt.format(
            task_description = episodedata.goal,
        )
        prompt_list.append(task_prompt)

        if self.include_device_time:
            device_time = current_step.curr_env_state.device_time
            # # Remove the hour-minute-second and the timezone 
            # device_time = ' '.join(device_time.split()[:3] + device_time.split()[-2:])
            device_time_prompt = self.prompt.device_time_prompt.format(
                device_time = device_time
            )
            prompt_list.append(device_time_prompt)

        if current_step.plan is not None:
            plan_prompt = self.prompt.plan_prompt.format(
                plan = current_step.plan,
            )
            prompt_list.append(plan_prompt)

        if current_step.sub_goal is not None:
            subgoal_prompt = self.prompt.subgoal_prompt.format(
                subgoal = current_step.sub_goal,
            )
            prompt_list.append(subgoal_prompt)

        if len(trajectory) > 1 and (self.num_histories is None or self.num_histories > 0):
            history = get_history(
                trajectory[:-1],
                self.num_histories,
                wrap_tool_call=self.wrap_history_tool_calls,
            )
            history = "\n".join(history)
        else:
            history = "No actions have been taken yet."
        history_prompt = self.prompt.history_prompt.format(
            history = history,
        )
        prompt_list.append(history_prompt)

        self.get_knowledge(episodedata.goal)
        if self.retrieved_knowledge or self.explored_knowledge:
            knowledge = ""
            if self.retrieved_knowledge:
                knowledge += self.retrieved_knowledge + "\n"
            if self.explored_knowledge:
                knowledge += self.explored_knowledge

            knowledge_prompt = self.prompt.knowledge_prompt.format(
                knowledge = knowledge,
            )
            logger.info("Knowledge is added.")
            prompt_list.append(knowledge_prompt)

        if len(trajectory) > 1:
            previous_step = trajectory[-2]
            if previous_step.progress is not None:
                progress_prompt = self.prompt.progress_prompt.format(
                    progress = previous_step.progress,
                )
                prompt_list.append(progress_prompt)

            if episodedata.memory is not None and episodedata.memory != "":
                memory_prompt = self.prompt.memory_prompt.format(
                    memory = episodedata.memory,
                )
                prompt_list.append(memory_prompt)

            if previous_step.reflection_outcome is not None and previous_step.reflection_outcome in ['B', 'C']:
                reflection_prompt = self.prompt.reflection_prompt.format(
                    action_desc = previous_step.action_desc,
                    action_s = previous_step.action_s,
                    reflection_error = previous_step.reflection_error,
                )
                prompt_list.append(reflection_prompt)

            if previous_step.trajectory_reflection_outcome is not None and previous_step.trajectory_reflection_outcome in ['B']:
                trajectory_reflection_prompt = self.prompt.trajectory_reflection_prompt.format(
                    trajectory_reflection_error = previous_step.trajectory_reflection_error,
                )
                prompt_list.append(trajectory_reflection_prompt)

            if previous_step.evaluation_result is not None and "Failed" in previous_step.evaluation_result:
                global_reflection_prompt = self.prompt.global_reflection_prompt.format(
                    global_reflection_error = previous_step.evaluation_reason,
                )
                prompt_list.append(global_reflection_prompt)

        observation_prompt = self.prompt.observation_prompt.format(
            resized_width = resized_width,
            resized_height = resized_height,
            image_placeholder = IMAGE_PLACEHOLDER,
            **coordinate_prompt_kwargs,
        )
        prompt_list.append(observation_prompt)

        if self.include_a11y_tree and current_step.curr_env_state.a11y_tree is not None:
            a11y_tree_prompt = self.prompt.a11y_tree_prompt.format(
                a11y_tree = current_step.curr_env_state.a11y_tree,
            )
            prompt_list.append(a11y_tree_prompt)

        if self.include_tips:
            init_tips = self.prompt.init_tips
            prompt_list.append(init_tips)

        response_prompt = self.prompt.response_prompt
        prompt_list.append(response_prompt)

        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt, images=[pixels])
        messages.append(user_message)

        return messages
    
    def parse_response(self, content: str, size: tuple[float, float] = None, raw_size: tuple[float, float] = None):
        if size is None:
            size = self.resized_size
        if raw_size is None:
            raw_size = self.raw_size
        if getattr(self.vlm, "preserve_raw_pixel_coordinates", False):
            size = raw_size
        
        thought = re.search(r"Thought:(.*?)(?=\n|Action:|<tool_call>|\{\"name\": \"mobile_use\",)", content, flags=re.DOTALL)
        if thought:
            thought_s = thought.group(1).strip()
        else:
            thought_s = None
            
        action_desc = re.search(r"Action:(.*?)(?=\n|<tool_call>|\{\"name\": \"mobile_use\",)", content, flags=re.DOTALL)
        if action_desc:
            action_desc_s = action_desc.group(1).strip()
        else:
            action_desc_s = None
        
        action = re.search(r'{"name": "mobile_use",(.*?)}}', content, flags=re.DOTALL)
        if not action:
            raise Exception("Cannot extract action in the content.")
        
        action_s = '{"name": "mobile_use",' + action.group(1).strip() + '}}'
        action = json.loads(action_s)
        
        name = map_action_names(action['arguments']['action'])
        
        # Remove the 'action' key and map the other keys in the arguments
        action['arguments'].pop('action')
        params = {}
        
        for k, v in action['arguments'].items():
            mapped_key = map_action_names(k)  # Map the key name
            if mapped_key in ['coordinate', 'coordinate2']:
                try:
                    x = round(v[0] / size[0] * raw_size[0])
                    y = round(v[1] / size[1] * raw_size[1])
                    params[mapped_key] = (x, y)
                except:
                    pass
            else:
                params[mapped_key] = v

        action_a = Action(name=name, parameters=params)

        return thought_s, action_a, action_s, action_desc_s


class TrainedOperator(Operator):
    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]
        
        pixels = current_step.curr_env_state.pixels.copy()
        self.raw_size = (pixels.width, pixels.height)
        if self.max_pixels is not None:
            pixels = resize_image(pixels, self.max_pixels)
        resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)
        self.resized_size = (resized_width, resized_height)

        # Add system prompt
        system_prompt = self.prompt.system_prompt.format(
            resized_width = resized_width,
            resized_height = resized_height,
        )
        system_message = generate_message("system", system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []
        task_prompt = self.prompt.task_prompt.format(
            task_description = episodedata.goal,
        )
        prompt_list.append(task_prompt)

        history_list = []
        if len(trajectory) == 1:
            history_list.append("None")
        else:
            for i, step in enumerate(trajectory[:-1]):
                if step.action is None:
                    history_list.append(f"Step {i+1}: None")
                else:
                    if 'terminate' not in step.action_s:
                        history_list.append(f"Step {i+1}: {step.action_desc}")
        history = ";".join(history_list)
        history_prompt = self.prompt.history_prompt.format(
            history = history,
        )
        prompt_list.append(history_prompt)

        if self.include_device_time:
            device_time = current_step.curr_env_state.device_time
            # # Remove the hour-minute-second and the timezone 
            # device_time = ' '.join(device_time.split()[:3] + device_time.split()[-2:])
            device_time_prompt = self.prompt.device_time_prompt.format(
                device_time = device_time
            )
            prompt_list.append(device_time_prompt)

        if current_step.plan is not None:
            plan_prompt = self.prompt.plan_prompt.format(
                plan = current_step.plan,
            )
            prompt_list.append(plan_prompt)

        if current_step.sub_goal is not None:
            subgoal_prompt = self.prompt.subgoal_prompt.format(
                subgoal = current_step.sub_goal,
            )
            prompt_list.append(subgoal_prompt)

        self.get_knowledge(episodedata.goal)
        if self.retrieved_knowledge or self.explored_knowledge:
            knowledge = ""
            if self.retrieved_knowledge:
                knowledge += self.retrieved_knowledge + "\n"
            if self.explored_knowledge:
                knowledge += self.explored_knowledge

            knowledge_prompt = self.prompt.knowledge_prompt.format(
                knowledge = knowledge,
            )
            logger.info("Knowledge is added.")
            prompt_list.append(knowledge_prompt)

        if len(trajectory) > 1:
            previous_step = trajectory[-2]
            if previous_step.progress is not None:
                progress_prompt = self.prompt.progress_prompt.format(
                    progress = previous_step.progress,
                )
                prompt_list.append(progress_prompt)

            if episodedata.memory is not None and episodedata.memory != "":
                memory_prompt = self.prompt.memory_prompt.format(
                    memory = episodedata.memory,
                )
                prompt_list.append(memory_prompt)

            if previous_step.reflection_outcome is not None and previous_step.reflection_outcome in ['B', 'C']:
                reflection_prompt = self.prompt.reflection_prompt.format(
                    action_desc = previous_step.action_desc,
                    action_s = previous_step.action_s,
                    reflection_error = previous_step.reflection_error,
                )
                prompt_list.append(reflection_prompt)

            if previous_step.trajectory_reflection_outcome is not None and previous_step.trajectory_reflection_outcome in ['B']:
                trajectory_reflection_prompt = self.prompt.trajectory_reflection_prompt.format(
                    trajectory_reflection_error = previous_step.trajectory_reflection_error,
                )
                prompt_list.append(trajectory_reflection_prompt)

            if previous_step.evaluation_result is not None and "Failed" in previous_step.evaluation_result:
                global_reflection_prompt = self.prompt.global_reflection_prompt.format(
                    global_reflection_error = previous_step.evaluation_reason,
                )
                prompt_list.append(global_reflection_prompt)

        if self.include_tips:
            init_tips = self.prompt.init_tips
            prompt_list.append(init_tips)
        
        if self.include_a11y_tree and current_step.curr_env_state.a11y_tree is not None:
            a11y_tree_prompt = self.prompt.a11y_tree_prompt.format(
                a11y_tree = current_step.curr_env_state.a11y_tree,
            )
            prompt_list.append(a11y_tree_prompt)
        
        prompt_list.append(f"  {IMAGE_PLACEHOLDER}")

        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt, images=[pixels])
        messages.append(user_message)

        return messages
    
    def parse_response(self, content: str, size: tuple[float, float] = None, raw_size: tuple[float, float] = None):
        if size is None:
            size = self.resized_size
        if raw_size is None:
            raw_size = self.raw_size
        
        thought = re.search(r'Thought: (.*?)Action:', content, flags=re.DOTALL)
        if thought:
            thought_s = thought.group(1).strip()
        else:
            thought_s = None
            
        action_desc = re.search(r'Action: (.*?)<answer>', content, flags=re.DOTALL)
        if action_desc:
            action_desc_s = action_desc.group(1).strip()
        else:
            action_desc_s = None
        
        action = re.search(r'<answer>(.*?)</answer>', content, flags=re.DOTALL)
        if not action:
            raise Exception("Cannot extract action in the content.")
        
        action_s = action.group(1).strip()
        try:
            action = json.loads(action_s)
        except:
            action = json.loads(action_s+']')
        action = action[0]
        
        name = map_action_names(action['arguments']['action'])
        
        # Remove the 'action' key and map the other keys in the arguments
        action['arguments'].pop('action')
        params = {}
        
        for k, v in action['arguments'].items():
            mapped_key = map_action_names(k)  # Map the key name
            if mapped_key in ['coordinate', 'coordinate2']:
                try:
                    x = round(v[0] / size[0] * raw_size[0])
                    y = round(v[1] / size[1] * raw_size[1])
                    params[mapped_key] = (x, y)
                except:
                    pass
            else:
                params[mapped_key] = v

        action_a = Action(name=name, parameters=params)

        return thought_s, action_a, action_s, action_desc_s


class OperatorQwen(Operator):
    def parse_response(self, content: str, size: tuple[float, float] = None, raw_size: tuple[float, float] = None):
        if size is None:
            size = self.resized_size
        if raw_size is None:
            raw_size = self.raw_size
        if self.coordinate_type == "relative":
            thought_s, action_a, action_s, action_desc_s = agent_qwen._parse_response_qwen3(
                content, size, raw_size, self.coordinate_base
            )
        else:
            thought_s, action_a, action_s, action_desc_s = agent_qwen._parse_response(content, size, raw_size)

        return thought_s, action_a, action_s, action_desc_s


class OperatorGPT(Operator):
    wrap_history_tool_calls = False

    def parse_response(self, content: str, size=None, raw_size=None):
        # GPT-5.6 has a dedicated parser contract.  Other OperatorGPT-style
        # callers retain the legacy direct-JSON parser below.
        adapter_name = str(getattr(self.vlm, "adapter_name", "")).lower()
        if adapter_name in {"gpt-5.6-absolute", "gpt5.6-absolute", "gpt-5.6-luna"}:
            from study.gpt_action_parser import parse_gpt_action

            parsed = parse_gpt_action(content, reject_multiple=True)
            action_payload = dict(parsed.action)
            action_name = action_payload.pop("action")
            action = Action(name=action_name, parameters=action_payload)
            action_s = json.dumps(
                parsed.action,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return parsed.thought, action, action_s, parsed.description or action_name
        return parse_gpt_action_response(
            content,
            size=size or self.resized_size,
            raw_size=raw_size or self.raw_size,
        )


class AnswerAgent(SubAgent):
    """
    This agent is used to answer the user query.
    It is a special case of the Operator, which only supports the `answer` action.
    """
    wrap_history_tool_calls = True

    def __init__(self, config: AnswerAgentConfig):
        super().__init__(config)
        self.prompt: AnswerAgentPrompt = load_prompt("answer_agent", config.prompt_config)
        self.num_histories = config.num_histories
        self.include_device_time = config.include_device_time
        self.max_pixels = config.max_pixels
        self.coordinate_type = config.coordinate_type
        self.coordinate_base = config.coordinate_base
        self.knowledge_config = config.knowledge

        self.embedding_model = None
        self.db = None
        self.retrieved_knowledge = None
        self.explored_knowledge = None
    
    def get_knowledge(self, query: str):
        if not self.knowledge_config:
            return
        _clear_stale_knowledge_for_query(self, query)
        # Retrieve knowledge from RAG database
        embedding_model_path = self.knowledge_config.embedding_model_path
        knowledge_database_dir = self.knowledge_config.knowledge_database_dir
        if knowledge_database_dir is not None:
            if self.retrieved_knowledge is None:
                assert embedding_model_path is not None, "Embedding model path must be provided if knowledge database dir is provided."
                if self.embedding_model is None:
                    logger.info("Loading RAG database for knowledge retrieval...")
                    project_home = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    sys.path += [os.path.join(project_home, 'third_party', 'RAGToolbox')]
                    from RAGToolbox import Jinaembedding, Vectordatabase
                    download_hf_model("https://huggingface.co/jinaai/jina-embeddings-v2-base-zh", embedding_model_path)
                    self.embedding_model=Jinaembedding(embedding_model_path)
                    self.db=Vectordatabase()
                    self.db.load_vector(knowledge_database_dir)
                answer = self.db.query_score(query, self.embedding_model,1)
                similarity, key, value = answer[0]
                logger.info(f"Retrieved knowledge: {str(value)}")
                if len(value) > 0 and value[0] != "":
                    self.retrieved_knowledge = '\n'.join([f"{i+1}. {v}" for i, v in enumerate(value)])

        # Get explored knowledge
        explored_knowledge_path = self.knowledge_config.explored_knowledge_path
        if explored_knowledge_path is not None:
            if self.explored_knowledge is None:
                all_explored_knowledge = json.load(open(explored_knowledge_path, 'r', encoding='utf-8'))
                app_names = list(all_explored_knowledge.keys())

                messages = [generate_message("user", "Which app is most relevant to the following query: " + query + "\nApp names: " + ", ".join(app_names) + ". \nAnswer with only the app name:")]
                response = self.vlm.predict(messages)
                app_name = response.choices[0].message.content.strip()
                if app_name in all_explored_knowledge:
                    knowledge_list = all_explored_knowledge[app_name]
                    explored_app_knowledge = '\n'.join([f"{i+1}. {v}" for i, v in enumerate(knowledge_list)])
                    logger.info(f"Explored knowledge from app {app_name} is added.")
                    messages = [generate_message("user", f"The summary of pages explored:\n{explored_app_knowledge}\nuser query:\n{query}\nPlease extract the text snippet that helps complete the user's request without exceeding 100 tokens and must keeping the original description. Don't answer user query! Extract only!")]
                    response = self.vlm.predict(messages)
                    self.explored_knowledge = response.choices[0].message.content.strip()
                else:
                    logger.warning(f"App name {app_name} not found in explored knowledge.")

    def reset(self):
        self.raw_size = None
        self.resized_size = None
        self.retrieved_knowledge = None
        self.embedding_model = None
        self.db = None
        self.explored_knowledge = None
        self._knowledge_query = None

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]
        
        pixels = current_step.curr_env_state.pixels.copy()
        self.raw_size = (pixels.width, pixels.height)
        if self.max_pixels is not None:
            pixels = resize_image(pixels, self.max_pixels)
        resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)
        self.resized_size = (resized_width, resized_height)

        # Add system prompt
        system_prompt = self.prompt.system_prompt.format(
            resized_width = resized_width,
            resized_height = resized_height,
        )
        system_message = generate_message("system", system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []
        task_prompt = self.prompt.task_prompt.format(
            task_description = episodedata.goal,
        )
        prompt_list.append(task_prompt)

        if self.include_device_time:
            device_time = current_step.curr_env_state.device_time
            # # Remove the hour-minute-second and the timezone 
            # device_time = ' '.join(device_time.split()[:3] + device_time.split()[-2:])
            device_time_prompt = self.prompt.device_time_prompt.format(
                device_time = device_time
            )
            prompt_list.append(device_time_prompt)

        if current_step.plan is not None:
            plan_prompt = self.prompt.plan_prompt.format(
                plan = current_step.plan,
            )
            prompt_list.append(plan_prompt)

        if current_step.sub_goal is not None:
            subgoal_prompt = self.prompt.subgoal_prompt.format(
                subgoal = current_step.sub_goal,
            )
            prompt_list.append(subgoal_prompt)

        if len(trajectory) > 1 and (self.num_histories is None or self.num_histories > 0):
            history = get_history(
                trajectory,
                self.num_histories,
                wrap_tool_call=self.wrap_history_tool_calls,
            )
            history = "\n".join(history)
        else:
            history = "No actions have been taken yet."
        history_prompt = self.prompt.history_prompt.format(
            history = history,
        )
        prompt_list.append(history_prompt)

        self.get_knowledge(episodedata.goal)
        if self.retrieved_knowledge or self.explored_knowledge:
            knowledge = ""
            if self.retrieved_knowledge:
                knowledge += self.retrieved_knowledge + "\n"
            if self.explored_knowledge:
                knowledge += self.explored_knowledge

            knowledge_prompt = self.prompt.knowledge_prompt.format(
                knowledge = knowledge,
            )
            logger.info("Knowledge is added.")
            prompt_list.append(knowledge_prompt)

        if len(trajectory) > 1:
            previous_step = trajectory[-2]
            if previous_step.progress is not None:
                progress_prompt = self.prompt.progress_prompt.format(
                    progress = previous_step.progress,
                )
                prompt_list.append(progress_prompt)

            if episodedata.memory is not None and episodedata.memory != "":
                memory_prompt = self.prompt.memory_prompt.format(
                    memory = episodedata.memory,
                )
                prompt_list.append(memory_prompt)

        observation_prompt = self.prompt.observation_prompt.format(
            resized_width = resized_width,
            resized_height = resized_height,
            image_placeholder = IMAGE_PLACEHOLDER,
        )
        prompt_list.append(observation_prompt)

        response_prompt = self.prompt.response_prompt.format(
            goal = episodedata.goal,
        )
        prompt_list.append(response_prompt)

        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt, images=[pixels])
        messages.append(user_message)

        return messages

    def parse_response(self, content: str, size: tuple[float, float] = None, raw_size: tuple[float, float] = None):
        if size is None:
            size = self.resized_size
        if raw_size is None:
            raw_size = self.raw_size
        
        thought = re.search(r"Thought:(.*?)(?=\n|Action:|<tool_call>|\{\"name\": \"mobile_use\",)", content, flags=re.DOTALL)
        if thought:
            thought_s = thought.group(1).strip()
        else:
            thought_s = None
            
        action_desc = re.search(r"Action:(.*?)(?=\n|<tool_call>|\{\"name\": \"mobile_use\",)", content, flags=re.DOTALL)
        if action_desc:
            action_desc_s = action_desc.group(1).strip()
        else:
            action_desc_s = None
        
        action = re.search(r'{"name": "mobile_use",(.*?)}}', content, flags=re.DOTALL)
        if not action:
            raise Exception("Cannot extract action in the content.")
        
        action_s = '{"name": "mobile_use",' + action.group(1).strip() + '}}'
        action = json.loads(action_s)
        
        name = map_action_names(action['arguments']['action'])
        
        # Remove the 'action' key and map the other keys in the arguments
        action['arguments'].pop('action')
        params = {}
        
        for k, v in action['arguments'].items():
            mapped_key = map_action_names(k)  # Map the key name
            if mapped_key in ['coordinate', 'coordinate2']:
                try:
                    x = round(v[0] / size[0] * raw_size[0])
                    y = round(v[1] / size[1] * raw_size[1])
                    params[mapped_key] = (x, y)
                except:
                    pass
            else:
                params[mapped_key] = v

        action_a = Action(name=name, parameters=params)

        return thought_s, action_a, action_s, action_desc_s


class TrainedAnswerAgent(AnswerAgent):
    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]
        
        pixels = current_step.curr_env_state.pixels.copy()
        self.raw_size = (pixels.width, pixels.height)
        if self.max_pixels is not None:
            pixels = resize_image(pixels, self.max_pixels)
        resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)
        self.resized_size = (resized_width, resized_height)

        # Add system prompt
        system_prompt = self.prompt.system_prompt.format(
            resized_width = resized_width,
            resized_height = resized_height,
        )
        system_message = generate_message("system", system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []
        task_prompt = self.prompt.task_prompt.format(
            task_description = episodedata.goal,
        )
        prompt_list.append(task_prompt)

        history_list = []
        if len(trajectory) == 1:
            history_list.append("None")
        else:
            for i, step in enumerate(trajectory[:-1]):
                if step.action is None:
                    history_list.append(f"Step {i+1}: None")
                else:
                    if 'terminate' not in step.action_s:
                        history_list.append(f"Step {i+1}: {step.action_desc}")
        history = ";".join(history_list)
        history_prompt = self.prompt.history_prompt.format(
            history = history,
        )
        prompt_list.append(history_prompt)

        if self.include_device_time:
            device_time = current_step.curr_env_state.device_time
            # # Remove the hour-minute-second and the timezone 
            # device_time = ' '.join(device_time.split()[:3] + device_time.split()[-2:])
            device_time_prompt = self.prompt.device_time_prompt.format(
                device_time = device_time
            )
            prompt_list.append(device_time_prompt)

        if current_step.plan is not None:
            plan_prompt = self.prompt.plan_prompt.format(
                plan = current_step.plan,
            )
            prompt_list.append(plan_prompt)

        if current_step.sub_goal is not None:
            subgoal_prompt = self.prompt.subgoal_prompt.format(
                subgoal = current_step.sub_goal,
            )
            prompt_list.append(subgoal_prompt)

        self.get_knowledge(episodedata.goal)
        if self.retrieved_knowledge or self.explored_knowledge:
            knowledge = ""
            if self.retrieved_knowledge:
                knowledge += self.retrieved_knowledge + "\n"
            if self.explored_knowledge:
                knowledge += self.explored_knowledge

            knowledge_prompt = self.prompt.knowledge_prompt.format(
                knowledge = knowledge,
            )
            logger.info("Knowledge is added.")
            prompt_list.append(knowledge_prompt)

        if len(trajectory) > 1:
            previous_step = trajectory[-2]
            if previous_step.progress is not None:
                progress_prompt = self.prompt.progress_prompt.format(
                    progress = previous_step.progress,
                )
                prompt_list.append(progress_prompt)

            if episodedata.memory is not None and episodedata.memory != "":
                memory_prompt = self.prompt.memory_prompt.format(
                    memory = episodedata.memory,
                )
                prompt_list.append(memory_prompt)

        prompt_list.append(f"  {IMAGE_PLACEHOLDER}")

        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt, images=[pixels])
        messages.append(user_message)

        return messages
    
    def parse_response(self, content: str, size: tuple[float, float] = None, raw_size: tuple[float, float] = None):
        if size is None:
            size = self.resized_size
        if raw_size is None:
            raw_size = self.raw_size
        
        thought = re.search(r'Thought: (.*?)Action:', content, flags=re.DOTALL)
        if thought:
            thought_s = thought.group(1).strip()
        else:
            thought_s = None
            
        action_desc = re.search(r'Action: (.*?)<answer>', content, flags=re.DOTALL)
        if action_desc:
            action_desc_s = action_desc.group(1).strip()
        else:
            action_desc_s = None
        
        action = re.search(r'<answer>(.*?)</answer>', content, flags=re.DOTALL)
        if not action:
            raise Exception("Cannot extract action in the content.")
        
        action_s = action.group(1).strip()
        try:
            action = json.loads(action_s)
        except:
            action = json.loads(action_s+']')
        action = action[0]
        
        name = map_action_names(action['arguments']['action'])
        
        # Remove the 'action' key and map the other keys in the arguments
        action['arguments'].pop('action')
        params = {}
        
        for k, v in action['arguments'].items():
            mapped_key = map_action_names(k)  # Map the key name
            if mapped_key in ['coordinate', 'coordinate2']:
                try:
                    x = round(v[0] / size[0] * raw_size[0])
                    y = round(v[1] / size[1] * raw_size[1])
                    params[mapped_key] = (x, y)
                except:
                    pass
            else:
                params[mapped_key] = v

        action_a = Action(name=name, parameters=params)

        return thought_s, action_a, action_s, action_desc_s


class AnswerAgentQwen(AnswerAgent):
    def parse_response(self, content: str, size: tuple[float, float] = None, raw_size: tuple[float, float] = None):
        if size is None:
            size = self.resized_size
        if raw_size is None:
            raw_size = self.raw_size
        if self.coordinate_type == "relative":
            thought_s, action_a, action_s, action_desc_s = agent_qwen._parse_response_qwen3(
                content, size, raw_size, self.coordinate_base
            )
        else:
            thought_s, action_a, action_s, action_desc_s = agent_qwen._parse_response(content, size, raw_size)

        return thought_s, action_a, action_s, action_desc_s


class AnswerAgentGPT(AnswerAgent):
    wrap_history_tool_calls = False

    def parse_response(self, content: str, size=None, raw_size=None):
        return parse_gpt_action_response(
            content,
            size=size or self.resized_size,
            raw_size=raw_size or self.raw_size,
        )



class Reflector(SubAgent):
    def __init__(self, config: ReflectorConfig):
        super().__init__(config)
        self.prompt: ReflectorPrompt = load_prompt("reflector", config.prompt_config)
        self.valid_options = ['A', 'B', 'C', 'D']

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]

        pixels_before = current_step.curr_env_state.pixels.copy()
        resized_height, resized_width = smart_resize(height=pixels_before.height, width=pixels_before.width)
        pixels_after = current_step.exec_env_state.pixels.copy()

        diff_flag = False
        new_img1, new_img2 = diff_image(pixels_before.copy(), pixels_after.copy())
        if new_img1 is not None:
            pixels_before, pixels_after = new_img1, new_img2
            diff_flag = True
        
        # Add system prompt
        system_message = generate_message("system", self.prompt.system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []

        task_prompt = self.prompt.task_prompt.format(
            task_description = episodedata.goal,
        )
        prompt_list.append(task_prompt)

        if current_step.sub_goal is not None:
            subgoal_prompt = self.prompt.subgoal_prompt.format(
                subgoal = current_step.sub_goal,
            )
            prompt_list.append(subgoal_prompt)

        observation_prompt = self.prompt.observation_prompt.format(
            screenshot1 = IMAGE_PLACEHOLDER,
            screenshot2 = IMAGE_PLACEHOLDER,
            resized_width = resized_width,
            resized_height = resized_height,
        )
        if is_same_image(pixels_before.copy(), pixels_after.copy(), crop_top_ratio=0.035):
            logger.info("The last action does not produce any changes on the screen.")
            observation_prompt += "\n" + self.prompt.same_image_prompt
        elif diff_flag:
            logger.info("The last action successfully produces some changes. The difference between the two images is highlighted in red boxes.")
            observation_prompt += "\n" + self.prompt.diff_image_prompt
        prompt_list.append(observation_prompt)

        expection_prompt = self.prompt.expection_prompt.format(
            action_s = current_step.action_s,
            action_desc = current_step.action_desc,
        )
        prompt_list.append(expection_prompt)

        response_prompt = self.prompt.response_prompt
        prompt_list.append(response_prompt)

        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt, images=[pixels_before, pixels_after])
        messages.append(user_message)

        return messages

    def parse_response(self, response: str) -> dict:
        outcome = response.split("### Outcome ###")[-1].split("### Error Description ###")[0].replace("\n", " ").replace("  ", " ").strip()
        error_description = response.split("### Error Description ###")[-1].split("### Explanation ###")[0].replace("\n", " ").replace("  ", " ").strip()
        return outcome, error_description


class TrajectoryReflector(SubAgent):
    def __init__(self, config: TrajectoryReflectorConfig):
        super().__init__(config)
        self.prompt: TrajectoryReflectorPrompt = load_prompt("trajectory_reflector", config.prompt_config)
        self.valid_options = ['A', 'B']
        self.evoke_every_steps = config.evoke_every_steps
        self.cold_steps = config.cold_steps
        self.detect_error = config.detect_error
        if config.num_histories == 'auto':
            self.num_histories = config.evoke_every_steps
        else:
            self.num_histories = config.num_histories
        self.num_latest_screenshots = config.num_latest_screenshots

        self.max_repeat_action = config.max_repeat_action
        self.max_repeat_action_series = config.max_repeat_action_series
        self.max_repeat_screen = config.max_repeat_screen
        self.max_fail_count = config.max_fail_count
    
    def reset(self):
        self.sleep_count = 0
    
    def detect(
        self, 
        episodedata: MobileUseEpisodeData,
    ) -> list:
        error = []
        trajectory = episodedata.trajectory
        if len(trajectory) < min(self.max_repeat_action, self.max_repeat_screen, self.max_fail_count):
            return error
        current_step = trajectory[-1]
        current_action = getattr(current_step, "action", None)

        # detect repeated actions
        repeat_action = 1
        if current_action is not None and current_action.name not in ["swipe", "wait"]:
            for step in trajectory[:-1][::-1]:
                if step.action == current_action:
                    repeat_action += 1
                else:
                    break
                if repeat_action >= self.max_repeat_action:
                    error.append(f"The action `{current_step.action_s}` has repeated more than {self.max_repeat_action} times. If you stuck in a page, change your action!")
                    break

        # detect repeated action series
        if len(trajectory) >= 4:
            if trajectory[-1].action != trajectory[-2].action and trajectory[-1].action == trajectory[-3].action and trajectory[-2].action == trajectory[-4].action:
                error.append(f"The latest two actions have repeated more than {self.max_repeat_action_series} times. DO NOT repeat your previous actions! Change your action to explore other possibilities!")
        if len(trajectory) >= 6:
            if trajectory[-1].action != trajectory[-2].action and trajectory[-1].action == trajectory[-4].action and trajectory[-2].action == trajectory[-5].action and trajectory[-3].action == trajectory[-6].action:
                error.append(f"The latest three actions have repeated more than {self.max_repeat_action_series} times. DO NOT repeat your previous actions! Change your action to explore other possibilities!")

        # detect repeated screenshots
        repeat_screen = 1
        for step in trajectory[:-1][::-1]:
            previous_action = getattr(step, "action", None)
            if is_same_image(step.exec_env_state.pixels.copy(), current_step.exec_env_state.pixels.copy(), crop_top_ratio=0.035) and (
                previous_action is None or previous_action.name not in ["wait"]
            ):
                repeat_screen += 1
            else:
                break
            if repeat_screen >= self.max_repeat_screen:
                error.append(f"The screen has kept unchanged for more than {self.max_repeat_screen} times. You may be stuck in a page. Change your action to explore other possibilities!")
                if current_action is not None and current_action.name == "swipe":
                    error.append(f"You have performed several `swipe` actions but the screen is still unchanged. It indicates that you have swiped to the end of a page. If you are trying to find a specific item, you can try to swipe towards the opposite direction.")
                break

        # detect fail reflection
        if current_step.reflection_outcome is not None:
            fail_count = 0
            for step in trajectory[::-1]:
                if step.reflection_outcome in ['B', 'C']:
                    fail_count += 1
                else:
                    break
                if fail_count >= self.max_fail_count:
                    error.append(f"You have encountered several failed attempts. Change your action to explore other possibilities!")
                    break

        return error
        

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        error = []
        if self.detect_error:
            error = self.detect(episodedata)
        step_idx = len(episodedata.trajectory)

        if step_idx % self.evoke_every_steps != 0 and len(error) == 0:
            self.sleep_count += 1
            return None, None
        if self.sleep_count < self.cold_steps:
            self.sleep_count += 1
            return None, None

        self.sleep_count = 0
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]

        num_latest_screenshots = min(self.num_latest_screenshots, len(trajectory))
        if num_latest_screenshots > 0:
            screenshots = [step.exec_env_state.pixels.copy() for step in trajectory[-num_latest_screenshots:]]
            resized_height, resized_width = smart_resize(height=screenshots[0].height, width=screenshots[0].width)
        else:
            screenshots = None

        # Add system prompt
        system_message = generate_message("system", self.prompt.system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []

        task_prompt = self.prompt.task_prompt.format(
            task_description = episodedata.goal,
        )
        prompt_list.append(task_prompt)

        if current_step.plan is not None:
            plan_prompt = self.prompt.plan_prompt.format(
                plan = current_step.plan,
            )
            prompt_list.append(plan_prompt)
        
        history = get_history(trajectory, self.num_histories)
        history = "\n".join(history)
        if current_step.answer is not None:
            history += f"\nFinal answer: {current_step.answer}"
        history_prompt = self.prompt.history_prompt.format(
            history = history,
        )
        prompt_list.append(history_prompt)
            
        if current_step.progress is not None:
            progress_prompt = self.prompt.progress_prompt.format(
                progress = current_step.progress,
            )
            prompt_list.append(progress_prompt)

        if num_latest_screenshots > 0:
            image_placeholders = IMAGE_PLACEHOLDER * num_latest_screenshots
            observation_prompt = self.prompt.observation_prompt.format(
                resized_width = resized_width,
                resized_height = resized_height,
                image_placeholders = image_placeholders,
            )
            prompt_list.append(observation_prompt)
        
        if len(error) > 0:
            error = '\n'.join(error)
            logger.info(f"Trajectory Reflector detects error: {error}")
            error_info_prompt = self.prompt.error_info_prompt.format(
                error = error,
            )
            prompt_list.append(error_info_prompt)
        else:
            error = None

        response_prompt = self.prompt.response_prompt
        prompt_list.append(response_prompt)

        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt, images=screenshots)
        messages.append(user_message)

        return error, messages

    def parse_response(self, response: str) -> dict:
        outcome = response.split("### Outcome ###")[-1].split("### Error Description ###")[0].replace("\n", " ").replace("  ", " ").strip()
        error_description = response.split("### Error Description ###")[-1].split("### Explanation ###")[0].replace("\n", " ").replace("  ", " ").strip()
        return outcome, error_description


class GlobalReflector(SubAgent):
    def __init__(self, config: GlobalReflectorConfig):
        super().__init__(config)
        self.prompt: GlobalReflectorPrompt = load_prompt("global_reflector", config.prompt_config)
        self.num_latest_screenshots = config.num_latest_screenshots

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        last_step = trajectory[-1]

        num_latest_screenshots = min(self.num_latest_screenshots, len(trajectory))
        if num_latest_screenshots > 0:
            screenshots = [step.exec_env_state.pixels.copy() for step in trajectory[-num_latest_screenshots:]]
            resized_height, resized_width = smart_resize(height=screenshots[0].height, width=screenshots[0].width)
        else:
            screenshots = None

        # Add system prompt
        system_message = generate_message("system", self.prompt.system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []

        task_prompt = self.prompt.task_prompt.format(
            task_description = episodedata.goal,
        )
        prompt_list.append(task_prompt)

        if last_step.plan is not None:
            plan_prompt = self.prompt.plan_prompt.format(
                plan = last_step.plan,
            )
            prompt_list.append(plan_prompt)

        history = get_history(trajectory)
        history = "\n".join(history)
        if last_step.answer is not None:
            history += f"\nFinal answer: {last_step.answer}"
        history_prompt = self.prompt.history_prompt.format(
            history = history,
        )
        prompt_list.append(history_prompt)

        if last_step.progress is not None:
            progress_prompt = self.prompt.progress_prompt.format(
                progress = last_step.progress
            )
            prompt_list.append(progress_prompt)

        if num_latest_screenshots > 0:
            image_placeholders = IMAGE_PLACEHOLDER * num_latest_screenshots
            observation_prompt = self.prompt.observation_prompt.format(
                resized_width = resized_width,
                resized_height = resized_height,
                image_placeholders = image_placeholders,
            )
            prompt_list.append(observation_prompt)

        response_prompt = self.prompt.response_prompt
        prompt_list.append(response_prompt)

        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt, images=screenshots)
        messages.append(user_message)

        return messages
    
    def parse_response(self, response: str):
        result = response.split("### Result ###")[-1].split("### Reason ###")[0].replace("\n", " ").replace("  ", " ").strip()
        reason = response.split("### Reason ###")[-1].strip()
        return result, reason


class Progressor(SubAgent):
    def __init__(self, config: ProgressorConfig):
        super().__init__(config)
        self.prompt: ProgressorPrompt = load_prompt("progressor", config.prompt_config)

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]
        
        # Add system prompt
        system_message = generate_message("system", self.prompt.system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []

        # task_prompt = self.prompt.task_prompt.format(
        #     task_description = episodedata.goal,
        # )
        # prompt_list.append(task_prompt)

        if len(trajectory) > 1:
            history = get_history(trajectory[:-1])
            history = "\n".join(history)
            previous_step = trajectory[-2]
            continue_progress_start = self.prompt.continue_progress_start.format(
                history = history,
                progress = previous_step.progress,
                action_desc = current_step.action_desc,
                action = current_step.action,
            )
            prompt_list.append(continue_progress_start)

            if current_step.reflection_outcome is not None:
                if current_step.reflection_outcome in ['B', 'C']:
                    continue_progress_reflection = self.prompt.continue_progress_reflection.format(
                        reflection_error = current_step.reflection_error,
                    )
                    prompt_list.append(continue_progress_reflection)

            continue_progress_response = self.prompt.continue_progress_response
            prompt_list.append(continue_progress_response)

        else:
            init_progress = self.prompt.init_progress.format(
                thought = current_step.thought,
                action_desc = current_step.action_desc,
                action = current_step.action,
            )
            prompt_list.append(init_progress)
            
        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt)
        messages.append(user_message)

        return messages
    
    def parse_response(self, response: str):
        return response.split("### Completed contents ###")[-1].replace("\n", " ").replace("  ", " ").strip()


class NoteTaker(SubAgent):
    def __init__(self, config: NoteTakerConfig):
        super().__init__(config)
        self.prompt: NoteTakerPrompt = load_prompt("note_taker", config.prompt_config)

    def get_message(self, episodedata: MobileUseEpisodeData) -> list:
        messages = []
        trajectory = episodedata.trajectory
        current_step = trajectory[-1]
        
        pixels = current_step.exec_env_state.pixels.copy()
        resized_height, resized_width = smart_resize(height=pixels.height, width=pixels.width)
        
        # Add system prompt
        system_message = generate_message("system", self.prompt.system_prompt)
        messages.append(system_message)

        # Add user prompt
        prompt_list = []

        task_prompt = self.prompt.task_prompt.format(
            task_description = episodedata.goal,
        )
        prompt_list.append(task_prompt)

        memory_prompt = self.prompt.memory_prompt.format(
            memory = episodedata.memory if episodedata.memory is not None or episodedata.memory == "" else "No important notes yet.",
        )
        prompt_list.append(memory_prompt)

        observation_prompt = self.prompt.observation_prompt.format(
            image_placeholder = IMAGE_PLACEHOLDER,
            resized_width = resized_width,
            resized_height = resized_height,
        )
        prompt_list.append(observation_prompt)

        response_prompt = self.prompt.response_prompt
        prompt_list.append(response_prompt)

        prompt = "\n\n".join(prompt_list)
        user_message = generate_message("user", prompt, images=[pixels])
        messages.append(user_message)

        return messages

    def parse_response(self, response: str):
        note = response.split("### Important Notes ###")[-1].strip()
        if note == "" or note.lower() in ["none", "no", "n/a", "na"]:
            note = None
        return note


class TaskClassifier(SubAgent):
    def __init__(self, config: SubAgentConfig):
        super().__init__(config)
        self.prompt: TaskClassifierPrompt = load_prompt("task_classifier", config.prompt_config)

    def get_message(self, taskdata: HierarchicalAgentTaskData) -> list:
        messages = []

        system_message = generate_message("system", self.prompt.system_prompt)
        messages.append(system_message)

        user_prompt = self.prompt.user_prompt.format(
            task_description = taskdata.task,
        )
        user_message = generate_message("user", user_prompt)
        messages.append(user_message)

        return messages

    def parse_response(self, response: str):
        task_type = response.split("Task Type:")[-1].strip().upper()
        return task_type


class TaskOrchestrator(SubAgent):
    def __init__(self, config: SubAgentConfig):
        super().__init__(config)
        self.prompt: TaskOrchestratorPrompt = load_prompt("task_orchestrator", config.prompt_config)

    def get_message(self, taskdata: HierarchicalAgentTaskData) -> list:
        messages = []

        system_message = generate_message("system", self.prompt.system_prompt)
        messages.append(system_message)

        user_prompt = self.prompt.user_prompt.format(
            task_description = taskdata.task,
        )
        user_message = generate_message("user", user_prompt)
        messages.append(user_message)

        return messages

    def parse_response(self, response: str):
        sub_tasks = []
        for line in response.split('\n'):
            line = line.strip()
            if line == "":
                continue
            if re.match(r'^\d+[\.\)]\s*', line):
                line = re.sub(r'^\d+[\.\)]\s*', '', line)                
            sub_tasks.append(line)
        
        if ".jpg" in sub_tasks[0] or ".png" in sub_tasks[0]:
            sub_tasks[0] += " Don't end the task when only thumbnails or small image are visible!!! Stay in the page where the image fills the (almost) entire screen!"
        return sub_tasks


class TaskExtractor(SubAgent):
    def __init__(self, config: SubAgentConfig):
        super().__init__(config)
        self.prompt: TaskExtractorPrompt = load_prompt("task_extractor", config.prompt_config)
        self.num_latest_screenshots = 1

    def get_message(self, taskdata: HierarchicalAgentTaskData) -> list:
        messages = []
        trajectory = taskdata.episode_data.trajectory
        
        system_message = generate_message("system", self.prompt.system_prompt)
        messages.append(system_message)

        num_latest_screenshots = min(self.num_latest_screenshots, len(trajectory))
        if num_latest_screenshots > 0:
            screenshots = [step.exec_env_state.pixels.copy() for step in trajectory[-num_latest_screenshots:]]
            resized_height, resized_width = smart_resize(height=screenshots[0].height, width=screenshots[0].width)
        else:
            screenshots = None

        user_prompt = self.prompt.user_prompt.format(
            task_description = taskdata.task,
            sub_tasks = '\n'.join([f"{i+1}. {sub_task}" for i, sub_task in enumerate(taskdata.sub_tasks)]),
            completed_sub_tasks = '\n'.join([f"{i+1}. {sub_task}" for i, sub_task in enumerate(taskdata.sub_tasks[:taskdata.current_sub_task_idx+1])]),
            resized_width = resized_width,
            resized_height = resized_height,
            image_placeholders = IMAGE_PLACEHOLDER * num_latest_screenshots if num_latest_screenshots > 0 else ""
        )
        user_message = generate_message("user", user_prompt, images=screenshots)
        messages.append(user_message)

        return messages

    def parse_response(self, response: str):
        return response


class TaskRewriter(SubAgent):
    def __init__(self, config: SubAgentConfig):
        super().__init__(config)
        self.prompt: TaskRewriterPrompt = load_prompt("task_rewriter", config.prompt_config)

    def get_message(self, taskdata: HierarchicalAgentTaskData) -> list:
        messages = []
        
        system_message = generate_message("system", self.prompt.system_prompt)
        messages.append(system_message)

        user_prompt = self.prompt.user_prompt.format(
            task_description = taskdata.task,
            sub_tasks = '\n'.join([f"{i+1}. {sub_task}" for i, sub_task in enumerate(taskdata.sub_tasks)]),
            completed_sub_tasks = '\n'.join([f"{i+1}. {sub_task}" for i, sub_task in enumerate(taskdata.sub_tasks[:taskdata.current_sub_task_idx+1])]),
            sub_task_info = taskdata.sub_tasks_return[taskdata.current_sub_task_idx],
            next_sub_task = taskdata.sub_tasks[taskdata.current_sub_task_idx + 1],
        )
        user_message = generate_message("user", user_prompt)
        messages.append(user_message)

        return messages

    def parse_response(self, response: str):
        return response

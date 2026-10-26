import re
from abc import abstractmethod
from typing import List, Optional
from http import HTTPStatus

import requests
import dashscope
import google.generativeai as genai

from utils import print_with_color, encode_image


# Transport guard only: this does not cap an AndroidWorld task or change its
# 30-step budget.  P2 previously called requests.post without any timeout, so a
# broken connection could leave its emulator lane blocked indefinitely.
_OPENAI_CONNECT_TIMEOUT_SECONDS = 10.0
_OPENAI_READ_TIMEOUT_SECONDS = 300.0


def _image_mime_type(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        signature = image_file.read(12)
    if signature.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if signature.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if signature.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if signature.startswith(b"RIFF") and signature[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError(f"Unsupported image format: {image_path}")


class BaseModel:
    def __init__(self):
        pass

    @abstractmethod
    def get_model_response(self, prompt: str, images: List[str]) -> (bool, str):
        pass


class OpenAIModel(BaseModel):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        enable_thinking: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
        thinking: Optional[dict] = None,
        max_tokens_parameter: str = "max_tokens",
    ):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.requested_model_name = model
        self.last_resolved_model = None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.thinking = thinking
        self.max_tokens_parameter = max_tokens_parameter
        self.last_usage = None

    def get_model_response(self, prompt: str, images: List[str]) -> (bool, str):
        self.last_resolved_model = None
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        for img in images:
            base64_img = encode_image(img)
            mime_type = _image_mime_type(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_img}"
                }
            })
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": self.temperature,
        }
        payload[self.max_tokens_parameter] = self.max_tokens
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
        if self.thinking_budget is not None:
            payload["thinking_budget"] = self.thinking_budget
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=(_OPENAI_CONNECT_TIMEOUT_SECONDS, _OPENAI_READ_TIMEOUT_SECONDS),
        ).json()
        if "error" not in response:
            # OpenAI-compatible providers often expand aliases to an exact
            # model snapshot here.  Keep it separate from the request id.
            self.last_resolved_model = response.get("model")
            usage = response.get("usage") or {}
            self.last_usage = usage
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            print_with_color(f"Request cost is "
                             f"${'{0:.2f}'.format(prompt_tokens / 1000 * 0.01 + completion_tokens / 1000 * 0.03)}",
                             "yellow")
        else:
            self.last_usage = None
            self.last_resolved_model = None
            return False, response["error"]["message"]
        return True, response["choices"][0]["message"]["content"]


class QwenModel(BaseModel):
    def __init__(self, api_key: str, model: str):
        super().__init__()
        self.model = model
        dashscope.api_key = api_key
        self.last_usage = None

    def get_model_response(self, prompt: str, images: List[str]) -> (bool, str):
        content = [{
            "text": prompt
        }]
        for img in images:
            img_path = f"file://{img}"
            content.append({
                "image": img_path
            })
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        response = dashscope.MultiModalConversation.call(model=self.model, messages=messages)
        if response.status_code == HTTPStatus.OK:
            self.last_usage = None
            return True, response.output.choices[0].message.content[0]["text"]
        else:
            self.last_usage = None
            return False, response.message


class GeminiModel(BaseModel):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        genai.configure(api_key=api_key)
        self.last_usage = None

    def get_model_response(self, prompt: str, images: List[str]) -> (bool, str):
        try:
            client = genai.GenerativeModel(self.model)
            content = [prompt]
            for img in images:
                with open(img, "rb") as image_file:
                    image_data = image_file.read()
                content.append({
                    "mime_type": "image/jpeg",
                    "data": image_data
                })
            response = client.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            if response.text:
                self.last_usage = None
                print_with_color("Gemini API request successful", "yellow")
                return True, response.text
            else:
                self.last_usage = None
                return False, "No response from Gemini API"
        except Exception as e:
            self.last_usage = None
            return False, f"Gemini API error: {str(e)}"


def parse_explore_rsp(rsp):
    try:
        observation = re.findall(r"Observation: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        act = re.findall(r"Action: (.*?)$", rsp, re.MULTILINE)[0]
        last_act = re.findall(r"Summary: (.*?)$", rsp, re.MULTILINE)[0]
        print_with_color("Observation:", "yellow")
        print_with_color(observation, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        print_with_color("Action:", "yellow")
        print_with_color(act, "magenta")
        print_with_color("Summary:", "yellow")
        print_with_color(last_act, "magenta")
        if "FINISH" in act:
            return ["FINISH"]
        act_name = act.split("(")[0]
        if act_name == "tap":
            area = int(re.findall(r"tap\((.*?)\)", act)[0])
            return [act_name, area, last_act]
        elif act_name == "text":
            input_str = re.findall(r"text\((.*?)\)", act)[0][1:-1]
            return [act_name, input_str, last_act]
        elif act_name == "long_press":
            area = int(re.findall(r"long_press\((.*?)\)", act)[0])
            return [act_name, area, last_act]
        elif act_name == "swipe":
            params = re.findall(r"swipe\((.*?)\)", act)[0]
            area, swipe_dir, dist = params.split(",")
            area = int(area)
            swipe_dir = swipe_dir.strip()[1:-1]
            dist = dist.strip()[1:-1]
            return [act_name, area, swipe_dir, dist, last_act]
        elif act_name == "grid":
            return [act_name]
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return ["ERROR"]
    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]


def parse_grid_rsp(rsp):
    try:
        observation = re.findall(r"Observation: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        act = re.findall(r"Action: (.*?)$", rsp, re.MULTILINE)[0]
        last_act = re.findall(r"Summary: (.*?)$", rsp, re.MULTILINE)[0]
        print_with_color("Observation:", "yellow")
        print_with_color(observation, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        print_with_color("Action:", "yellow")
        print_with_color(act, "magenta")
        print_with_color("Summary:", "yellow")
        print_with_color(last_act, "magenta")
        if "FINISH" in act:
            return ["FINISH"]
        act_name = act.split("(")[0]
        if act_name == "tap":
            params = re.findall(r"tap\((.*?)\)", act)[0].split(",")
            area = int(params[0].strip())
            subarea = params[1].strip()[1:-1]
            return [act_name + "_grid", area, subarea, last_act]
        elif act_name == "long_press":
            params = re.findall(r"long_press\((.*?)\)", act)[0].split(",")
            area = int(params[0].strip())
            subarea = params[1].strip()[1:-1]
            return [act_name + "_grid", area, subarea, last_act]
        elif act_name == "swipe":
            params = re.findall(r"swipe\((.*?)\)", act)[0].split(",")
            start_area = int(params[0].strip())
            start_subarea = params[1].strip()[1:-1]
            end_area = int(params[2].strip())
            end_subarea = params[3].strip()[1:-1]
            return [act_name + "_grid", start_area, start_subarea, end_area, end_subarea, last_act]
        elif act_name == "grid":
            return [act_name]
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return ["ERROR"]
    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]


def parse_reflect_rsp(rsp):
    try:
        decision = re.findall(r"Decision: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        print_with_color("Decision:", "yellow")
        print_with_color(decision, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        if decision == "INEFFECTIVE":
            return [decision, think]
        elif decision == "BACK" or decision == "CONTINUE" or decision == "SUCCESS":
            doc = re.findall(r"Documentation: (.*?)$", rsp, re.MULTILINE)[0]
            print_with_color("Documentation:", "yellow")
            print_with_color(doc, "magenta")
            return [decision, think, doc]
        else:
            print_with_color(f"ERROR: Undefined decision {decision}!", "red")
            return ["ERROR"]
    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]

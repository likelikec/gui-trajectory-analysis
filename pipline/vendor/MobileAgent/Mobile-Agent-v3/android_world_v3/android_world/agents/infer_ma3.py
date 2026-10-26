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

"""Some LLM inference interface."""

import abc
import copy
from datetime import datetime, timezone
import os
import time
from typing import Any, Optional
import numpy as np
from PIL import Image
from openai import OpenAI
try:
  from qwen_vl_utils import smart_resize
except ImportError:
  # Claude's server runtime only needs the resize helper; avoid requiring the
  # optional Qwen/PyTorch vision stack for this OpenAI-compatible path.
  from android_world.agents.coordinate_resize import smart_resize
from io import BytesIO
import base64

ERROR_CALLING_LLM = 'Error calling LLM'

def _utcnow_iso():
  return datetime.now(timezone.utc).isoformat()

def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG") 
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def image_to_base64(image_path):
  dummy_image = Image.open(image_path)
  MIN_PIXELS=3136
  MAX_PIXELS=10035200
  resized_height, resized_width  = smart_resize(dummy_image.height,
      dummy_image.width,
      factor=28,
      min_pixels=MIN_PIXELS,
      max_pixels=MAX_PIXELS,)
  dummy_image = dummy_image.resize((resized_width, resized_height))
  return f"data:image/png;base64,{pil_to_base64(dummy_image)}"

class LlmWrapper(abc.ABC):
  """Abstract interface for (text only) LLM."""

  @abc.abstractmethod
  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.

    Returns:
      Text output, is_safe, and raw output.
    """

class MultimodalLlmWrapper(abc.ABC):
  """Abstract interface for Multimodal LLM."""

  @abc.abstractmethod
  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray], messages = None
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy ndarray.

    Returns:
      Text output and raw output.
    """

class GUIOwlWrapper(LlmWrapper, MultimodalLlmWrapper):

    RETRY_WAITING_SECONDS = 20

    def __init__(
            self,
            api_key: str,
            base_url: str,
            model_name: str,
            max_retry: int = 10,
            temperature: float = 0.0,
            retry_waiting_seconds: float | None = None,
    ):
        if max_retry <= 0:
            max_retry = 10
            print('Max_retry must be positive. Reset it to 10')
        configured_max_retry = min(int(max_retry), 10)
        retry_override = os.environ.get("MA3_MAX_RETRY")
        if retry_override is None:
            self.max_retry = configured_max_retry
        else:
            try:
                self.max_retry = min(max(1, int(retry_override)), 10)
            except ValueError:
                print(
                    f"Ignoring invalid MA3_MAX_RETRY={retry_override!r}; "
                    f"using configured max_retry={configured_max_retry}",
                    flush=True,
                )
                self.max_retry = configured_max_retry
        configured_wait = (
            self.RETRY_WAITING_SECONDS
            if retry_waiting_seconds is None
            else max(0.0, float(retry_waiting_seconds))
        )
        wait_override = os.environ.get("MA3_RETRY_WAIT_SECONDS")
        if wait_override is None:
            self.retry_waiting_seconds = configured_wait
        else:
            try:
                self.retry_waiting_seconds = max(0.0, float(wait_override))
            except ValueError:
                print(
                    f"Ignoring invalid MA3_RETRY_WAIT_SECONDS={wait_override!r}; "
                    f"using configured retry_waiting_seconds={configured_wait}",
                    flush=True,
                )
                self.retry_waiting_seconds = configured_wait
        self.temperature = temperature
        self.model = model_name
        self.requested_model_name = model_name
        self.last_resolved_model = None
        self.timeout = float(os.environ.get("MA3_REQUEST_TIMEOUT", "120"))
        self.max_tokens = int(os.environ.get("MA3_MAX_TOKENS", "1024"))
        self.max_tokens_parameter = "max_tokens"
        self.bot = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout,
        )
        self.last_started_at = None
        self.last_finished_at = None
        self.last_duration_seconds = None

    def _sanitize_messages(self, messages):
      sanitized = copy.deepcopy(messages or [])
      for message in sanitized:
        content = message.get('content') if isinstance(message, dict) else None
        if not isinstance(content, list):
          continue
        for item in content:
          if not isinstance(item, dict):
            continue
          image_url = item.get('image_url')
          if isinstance(image_url, dict) and str(image_url.get('url', '')).startswith('data:image/'):
            item['image_url'] = {'url': '<inline_image>'}
      return sanitized

    def _normalize_usage(self, response):
      usage = getattr(response, 'usage', None)
      if usage is None:
        return None

      def _get(*names):
        for name in names:
          if isinstance(usage, dict):
            value = usage.get(name)
          else:
            value = getattr(usage, name, None)
          if value is not None:
            return int(value or 0)
        return 0

      prompt_tokens = _get('prompt_tokens', 'input_tokens')
      completion_tokens = _get('completion_tokens', 'output_tokens')
      total_tokens = _get('total_tokens') or (prompt_tokens + completion_tokens)
      return {
          'prompt_tokens': prompt_tokens,
          'completion_tokens': completion_tokens,
          'total_tokens': total_tokens,
      }

    def build_call_record(self, messages, response, stage=None):
      content = ''
      try:
        content = response.choices[0].message.content or ''
      except Exception:
        content = ''
      resolved_model = getattr(response, 'model', None) or self.last_resolved_model
      return {
          'messages': self._sanitize_messages(messages),
          'response': content,
          'stage': stage,
          'usage': self._normalize_usage(response),
          'model_name': resolved_model or self.requested_model_name,
          'requested_model_name': self.requested_model_name,
          'resolved_model_name': resolved_model,
          'resolved_model_source': 'provider_response.model' if resolved_model else None,
          'resolved_model_missing_reason': None if resolved_model else 'provider_response_model_field_missing',
          'started_at': self.last_started_at,
          'finished_at': self.last_finished_at,
          'duration_seconds': self.last_duration_seconds,
      }

    def convert_messages_format_to_openaiurl(self, messages):
      converted_messages = []
      for message in messages:
          new_content = []
          for item in message['content']:
              if list(item.keys())[0] == 'text':
                  new_content.append({'type': 'text', 'text': item['text']})
              elif list(item.keys())[0] == 'image':
                new_content.append({'type': 'image_url', 'image_url': {'url': image_to_base64(item['image'])}})
          converted_messages.append({'role': message['role'], 'content': new_content})

      return converted_messages
    
    def predict(
            self,
            text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(text_prompt, [])

    def predict_mm(
            self, text_prompt: str, images: list[np.ndarray], messages = None
    ) -> tuple[str, Optional[bool], Any]:
        
        if messages is None:
          payload = [
              {
                  "role": "user",
                  "content": [
                      {"text": text_prompt},
                  ]
              }
          ]
          
          for image in images:
            payload[0]['content'].append({
                'image': image
            })
        else:
          payload = messages
            
        payload = self.convert_messages_format_to_openaiurl(payload)

        wait_seconds = self.retry_waiting_seconds
        for attempt in range(1, self.max_retry + 1):
            try:
              self.last_resolved_model = None
              self.last_started_at = _utcnow_iso()
              started_perf = time.perf_counter()
              request_kwargs = {
                  "model": self.requested_model_name,
                  "messages": payload,
                  "timeout": self.timeout,
                  self.max_tokens_parameter: self.max_tokens,
              }
              chat_completion_from_url = self.bot.chat.completions.create(**request_kwargs)
              self.last_resolved_model = getattr(chat_completion_from_url, 'model', None)
              self.last_finished_at = _utcnow_iso()
              self.last_duration_seconds = time.perf_counter() - started_perf
              choices = getattr(chat_completion_from_url, 'choices', None)
              if not choices:
                raise RuntimeError('Provider returned no completion choices.')
              message = getattr(choices[0], 'message', None)
              content = getattr(message, 'content', None)
              if not isinstance(content, str) or not content.strip():
                raise RuntimeError('Provider returned empty completion content.')
              return (content, payload, chat_completion_from_url)
            except Exception as e:
                self.last_finished_at = _utcnow_iso()
                self.last_duration_seconds = None
                from study.model_adapters import (
                    INSUFFICIENT_BALANCE_MARKER,
                    InsufficientBalanceError,
                    is_insufficient_balance_error,
                )
                if is_insufficient_balance_error(e):
                    print(INSUFFICIENT_BALANCE_MARKER, flush=True)
                    raise InsufficientBalanceError() from e
                remaining = self.max_retry - attempt
                print(
                    f'Error calling LLM on attempt {attempt}/{self.max_retry}: {e}',
                    flush=True,
                )
                if remaining > 0:
                    print(
                        f'Retrying in {wait_seconds:g} seconds ({remaining} attempts left).',
                        flush=True,
                    )
                    time.sleep(wait_seconds)
        return ERROR_CALLING_LLM, payload, None

from __future__ import annotations

from datetime import datetime, timezone
import time
import asyncio
import logging
import copy
import uuid
from typing import Any, Optional
from openai import OpenAI, ChatCompletion

from mobile_use.schema.schema import VLMCallingData


logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VLMWrapper:
    """VLM Wrapper for UI Operation.
    """

    def __init__(
            self,
            model_name: str,
            api_key: str,
            base_url: str,
            max_retry: int = 3,
            retry_waiting_seconds: int = 2,
            max_tokens: int = 1024,
            temperature: float = 0.0,
            max_pixels: Optional[int] = None,
            adapter_name: str = "auto",
            coordinate_mode: str = "auto",
            coordinate_base: Optional[int] = None,
            **vlm_kwargs
        ):
        # Keep the configured request id immutable.  Providers may return a
        # dated/fully-qualified snapshot in response.model; that is evidence,
        # not a replacement for the next request's configured model id.
        self.model_name = model_name
        self.requested_model_name = model_name
        self.last_resolved_model: Optional[str] = None
        self._client_init_error: Exception | None = None
        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except Exception as exc:
            # Keep configuration-only construction usable for tests and
            # dry-runs; an actual request still fails explicitly below.
            self.client = None
            self._client_init_error = exc
        # self.aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)

        self.max_retry = max_retry
        self.retry_waiting_seconds = retry_waiting_seconds

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_pixels = max_pixels
        self.adapter_name = adapter_name
        self.coordinate_mode = coordinate_mode
        self.coordinate_base = coordinate_base

        try:
            from study.model_adapters import adapter_for_name
            self.adapter = adapter_for_name(adapter_name)
        except Exception:
            self.adapter = None

        self.preserve_raw_pixel_coordinates = bool(
            getattr(self.adapter, "preserve_raw_pixel_coordinates", False)
        )

        self.vlm_kwargs = vlm_kwargs
        self.last_started_at: Optional[str] = None
        self.last_finished_at: Optional[str] = None
        self.last_duration_seconds: Optional[float] = None
        self.logprobs_available: Optional[bool] = None
        self._logprobs_missing_warned = False
        self._logprobs_error_warned = False
        self.last_logprobs_requested: Optional[bool] = None
        self.last_logprobs_returned: Optional[bool] = None
        self.last_logprobs_missing_reason: Optional[str] = None

    @staticmethod
    def _without_logprobs(kwargs: dict[str, Any]) -> dict[str, Any]:
        clean_kwargs = dict(kwargs)
        clean_kwargs.pop("logprobs", None)
        clean_kwargs.pop("top_logprobs", None)
        return clean_kwargs

    @staticmethod
    def _response_has_logprobs(response: ChatCompletion) -> bool:
        try:
            logprobs = response.choices[0].logprobs
            content = getattr(logprobs, "content", None)
            return bool(content)
        except Exception:
            return False

    @staticmethod
    def _is_logprobs_unsupported_error(err: Exception) -> bool:
        message = str(err).lower()
        return "logprobs" in message or "top_logprobs" in message

    def _sanitize_messages(self, messages) -> list[dict[str, Any]]:
        sanitized = copy.deepcopy(messages)
        for message in sanitized:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") == "image_url":
                    image_url = item.get("image_url") or {}
                    if isinstance(image_url, dict) and str(image_url.get("url", "")).startswith("data:image/"):
                        item["image_url"] = {"url": "<inline_image>"}
        return sanitized

    def _normalize_usage(self, response) -> dict[str, Any]:
        from study.runtime_metrics import normalize_usage

        return normalize_usage(getattr(response, "usage", None))

    def build_call_record(self, messages, response: ChatCompletion, *, stage: str | None = None) -> VLMCallingData:
        content = None
        try:
            content = response.choices[0].message.content
        except Exception:
            content = None
        record_messages = messages
        if self.adapter is not None and hasattr(self.adapter, "call_record_overrides"):
            record_messages, content = self.adapter.call_record_overrides(
                response,
                messages,
                content or "",
            )
        resolved_model = getattr(response, "model", None) or self.last_resolved_model
        coordinate_space = getattr(self.adapter, "last_coordinate_space", None)
        return VLMCallingData(
            messages=self._sanitize_messages(record_messages),
            response=content or "",
            call_id=f"call_{uuid.uuid4().hex}",
            role=stage,
            stage=stage,
            usage=self._normalize_usage(response),
            model_name=resolved_model or self.requested_model_name,
            requested_model_name=self.requested_model_name,
            resolved_model_name=resolved_model,
            resolved_model_source="provider_response.model" if resolved_model else None,
            resolved_model_missing_reason=None if resolved_model else "provider_response_model_field_missing",
            started_at=self.last_started_at,
            finished_at=self.last_finished_at,
            duration_seconds=self.last_duration_seconds,
            logprobs_requested=self.last_logprobs_requested,
            logprobs_returned=self.last_logprobs_returned,
            logprobs_missing_reason=self.last_logprobs_missing_reason,
            coordinate_space=coordinate_space,
        )

    def predict(self, messages, stream: bool=False, **kwargs) -> ChatCompletion:
        """Predict the next action given the history and the current screenshot.

        Args:
            messages: The messages to send to the model.
        
        Returns:
            The ChatCompletion of the VLM
        """
        if self.client is None:
            raise ValueError(f"Invalid VLM base_url or client configuration: {self._client_init_error}")

        counter = self.max_retry
        wait_seconds = self.retry_waiting_seconds

        # import copy
        # messages_s = copy.deepcopy(messages)
        # for msg in messages_s:
        #     content = msg['content']
        #     if isinstance(content, list):
        #         for cnt in content:
        #             cnt.pop('image_url', None)
        # import json
        # print("messages: ", json.dumps(messages_s, ensure_ascii=False, indent=2))

        # Add max_pixels to all image content if specified
        if self.max_pixels is not None:
            import copy
            messages = copy.deepcopy(messages)
            for msg in messages:
                content = msg.get('content')
                if isinstance(content, list):
                    for cnt in content:
                        if 'image' in cnt.get('type', ''):
                            cnt['max_pixels'] = self.max_pixels

        kwargs.update(self.vlm_kwargs)
        requested_logprobs = bool(kwargs.get("logprobs"))
        self.last_logprobs_requested = requested_logprobs
        self.last_logprobs_returned = None
        self.last_logprobs_missing_reason = None
        effective_kwargs = dict(kwargs)
        if requested_logprobs and self.logprobs_available is False:
            effective_kwargs = self._without_logprobs(effective_kwargs)
            self.last_logprobs_missing_reason = "provider_logprobs_unavailable_cached"

        while counter > 0:
            try:
                self.last_resolved_model = None
                if self.adapter is not None and hasattr(self.adapter, "last_resolved_model"):
                    self.adapter.last_resolved_model = None
                self.last_started_at = _utcnow_iso()
                started_perf = time.perf_counter()
                request_kwargs = {
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "stream": stream,
                    **effective_kwargs,
                }
                if self.adapter is not None:
                    response = self.adapter.chat(
                        self.client,
                        model=self.requested_model_name,
                        messages=messages,
                        **request_kwargs,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **request_kwargs,
                    )
                resolved_model = (
                    getattr(response, "model", None)
                    or getattr(self.adapter, "last_resolved_model", None)
                )
                self.last_resolved_model = resolved_model
                self.last_finished_at = _utcnow_iso()
                self.last_duration_seconds = time.perf_counter() - started_perf
                if requested_logprobs:
                    if self._response_has_logprobs(response):
                        self.logprobs_available = True
                        self.last_logprobs_returned = True
                        self.last_logprobs_missing_reason = None
                    else:
                        self.logprobs_available = False
                        self.last_logprobs_returned = False
                        if self.last_logprobs_missing_reason is None:
                            self.last_logprobs_missing_reason = "provider_response_missing_logprobs"
                        if not self._logprobs_missing_warned:
                            logger.info(
                                "Model/provider did not return chat logprobs for %s; "
                                "this raw provider fact is recorded with the call.",
                                self.model_name,
                            )
                            self._logprobs_missing_warned = True
                return response
            except Exception as err:
                self.last_finished_at = _utcnow_iso()
                self.last_duration_seconds = None
                from study.model_adapters import (
                    INSUFFICIENT_BALANCE_MARKER,
                    InsufficientBalanceError,
                    is_insufficient_balance_error,
                )
                if is_insufficient_balance_error(err):
                    logger.error(INSUFFICIENT_BALANCE_MARKER)
                    raise InsufficientBalanceError() from err
                if (
                    requested_logprobs
                    and effective_kwargs.get("logprobs")
                    and self._is_logprobs_unsupported_error(err)
                ):
                    self.logprobs_available = False
                    self.last_logprobs_returned = False
                    self.last_logprobs_missing_reason = "provider_rejected_logprobs"
                    effective_kwargs = self._without_logprobs(effective_kwargs)
                    if not self._logprobs_error_warned:
                        logger.warning(
                            "Model/provider rejected logprobs for %s; retrying without logprobs. Error: %s",
                            self.model_name,
                            err,
                        )
                        self._logprobs_error_warned = True
                    continue
                logger.warning(f'Error calling VLM API with message: {err}')
                time.sleep(wait_seconds)
                counter -= 1
                if counter <= 0:
                    raise ValueError("Max tries for predict failure.") from err  # re-raise after max retry.

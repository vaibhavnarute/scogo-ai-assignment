"""NVIDIA NIM Chat Completions adapter using the standard library."""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..errors import ErrorCategory, HarnessError
from ..tools.base import ToolCall
from .base import ModelProvider
from .types import ModelResponse, Usage


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"
DEFAULT_NVIDIA_MODEL = "openai/gpt-oss-20b"
_HARMONY_CHANNEL_SUFFIX = re.compile(r"(?:<\|channel\|>(?:analysis|commentary|final|json))+$")


@dataclass(frozen=True, slots=True)
class NvidiaConfig:
    model: str = DEFAULT_NVIDIA_MODEL
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    temperature: float | None = 0.0
    max_tokens: int | None = 4096
    extra_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise HarnessError("CONFIG_INVALID_PROVIDER", ErrorCategory.CONFIGURATION, "NVIDIA model is required", False)
        if self.timeout_seconds <= 0 or self.max_retries < 0 or self.retry_backoff_seconds < 0:
            raise HarnessError("CONFIG_INVALID_PROVIDER", ErrorCategory.CONFIGURATION, "NVIDIA timeout/retry values are invalid", False)


class NvidiaProvider(ModelProvider):
    provider_name = "nvidia"

    def __init__(self, config: NvidiaConfig | None = None) -> None:
        self.config = config or NvidiaConfig()
        self.model = self.config.model

    @property
    def endpoint(self) -> str:
        return NVIDIA_BASE_URL + "/chat/completions"

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        api_key = os.environ.get(NVIDIA_API_KEY_ENV)
        if not api_key:
            raise HarnessError("CONFIG_MISSING_API_KEY", ErrorCategory.CONFIGURATION, f"required environment variable is not set: {NVIDIA_API_KEY_ENV}", False)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": [{"type": "function", "function": tool} for tool in tools],
            "tool_choice": "auto",
            "stream": False,
            **self.config.extra_body,
        }
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            body["max_tokens"] = self.config.max_tokens
        encoded = json.dumps(body).encode("utf-8")
        retry_count = 0
        while True:
            request = urllib.request.Request(
                self.endpoint,
                data=encoded,
                method="POST",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                normalized = self._normalize(payload, {str(tool.get("name") or "") for tool in tools})
                normalized.retry_count = retry_count
                return normalized
            except HarnessError as exc:
                if exc.code != "PROVIDER_BAD_RESPONSE":
                    raise
                if retry_count < self.config.max_retries:
                    retry_count += 1
                    time.sleep(self.config.retry_backoff_seconds * retry_count)
                    continue
                details = dict(exc.details)
                details["retry_count"] = retry_count
                raise HarnessError(exc.code, exc.category, exc.message, True, details) from exc
            except urllib.error.HTTPError as exc:
                code = self._http_error_code(exc.code)
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if retryable and retry_count < self.config.max_retries:
                    retry_count += 1
                    time.sleep(self.config.retry_backoff_seconds * retry_count)
                    continue
                raise HarnessError(code, ErrorCategory.PROVIDER, f"provider HTTP request failed with status {exc.code}", retryable, {"status": exc.code, "retry_count": retry_count}) from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                if retry_count < self.config.max_retries:
                    retry_count += 1
                    time.sleep(self.config.retry_backoff_seconds * retry_count)
                    continue
                error_code = "PROVIDER_TIMEOUT" if isinstance(getattr(exc, "reason", exc), (TimeoutError, socket.timeout)) else "PROVIDER_CONNECTION_ERROR"
                raise HarnessError(error_code, ErrorCategory.PROVIDER, "provider connection failed", True, {"retry_count": retry_count, "error_type": type(exc).__name__}) from exc
            except json.JSONDecodeError as exc:
                if retry_count < self.config.max_retries:
                    retry_count += 1
                    time.sleep(self.config.retry_backoff_seconds * retry_count)
                    continue
                raise HarnessError(
                    "PROVIDER_BAD_RESPONSE",
                    ErrorCategory.PROVIDER,
                    "provider returned invalid JSON",
                    True,
                    {"retry_count": retry_count},
                ) from exc

    @staticmethod
    def _normalize(payload: dict[str, Any], allowed_tool_names: set[str] | None = None) -> ModelResponse:
        def bad_response(message: str) -> HarnessError:
            return HarnessError("PROVIDER_BAD_RESPONSE", ErrorCategory.PROVIDER, message, True)

        if not isinstance(payload, dict):
            raise bad_response("provider response must be a JSON object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise bad_response("provider response is missing choices/message")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise bad_response("provider response message must be an object")
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls is None:
            raw_tool_calls = []
        if not isinstance(raw_tool_calls, list):
            raise bad_response("provider response tool_calls must be an array")
        tool_calls: list[ToolCall] = []
        tool_name_repairs: list[dict[str, str]] = []
        for index, raw_call in enumerate(raw_tool_calls):
            if not isinstance(raw_call, dict):
                raise bad_response("provider response tool call must be an object")
            function = raw_call.get("function")
            if not isinstance(function, dict):
                raise bad_response("provider response tool function must be an object")
            raw_arguments = function.get("arguments", {})
            if isinstance(raw_arguments, str):
                try:
                    arguments: Any = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = raw_arguments
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                raise bad_response("provider response tool arguments must be an object or JSON string")
            tool_call_id = str(raw_call.get("id") or f"provider-call-{index}")
            name = function.get("name")
            if name is not None and not isinstance(name, str):
                raise bad_response("provider response tool name must be a string")
            raw_name = name or ""
            normalized_name = _HARMONY_CHANNEL_SUFFIX.sub("", raw_name)
            if normalized_name != raw_name and normalized_name in (allowed_tool_names or set()):
                tool_name_repairs.append(
                    {
                        "tool_call_id": tool_call_id,
                        "raw_tool_name": raw_name,
                        "normalized_tool_name": normalized_name,
                        "reason": "nvidia_harmony_channel_suffix",
                    }
                )
            else:
                normalized_name = raw_name
            tool_calls.append(ToolCall(tool_call_id, normalized_name, arguments))
        raw_usage = payload.get("usage")
        if raw_usage is None:
            raw_usage = {}
        if not isinstance(raw_usage, dict):
            raise bad_response("provider response usage must be an object")
        details = raw_usage.get("prompt_tokens_details")
        if details is None:
            details = raw_usage.get("input_tokens_details")
        if details is None:
            details = {}
        if not isinstance(details, dict):
            raise bad_response("provider response token details must be an object")

        def token_count(*names: str, source: dict[str, Any] = raw_usage) -> int:
            raw: Any = 0
            for field_name in names:
                if field_name in source:
                    raw = source[field_name]
                    break
            if raw is None:
                return 0
            if isinstance(raw, bool) or not isinstance(raw, (int, str)):
                raise bad_response("provider response token count must be a non-negative integer")
            try:
                value = int(raw)
            except (TypeError, ValueError) as exc:
                raise bad_response("provider response token count must be a non-negative integer") from exc
            if value < 0 or (isinstance(raw, str) and str(value) != raw.strip()):
                raise bad_response("provider response token count must be a non-negative integer")
            return value

        usage = Usage(
            token_count("prompt_tokens", "input_tokens"),
            token_count("completion_tokens", "output_tokens"),
            token_count("cached_tokens", source=details),
        )
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise bad_response("provider response content must be a string or null")
        return ModelResponse(
            text=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            provider_request_id=payload.get("id"),
            raw_metadata={"model": payload.get("model"), "created": payload.get("created"), "tool_name_repairs": tool_name_repairs},
        )

    @staticmethod
    def _http_error_code(status: int) -> str:
        if status in {401, 403}:
            return "PROVIDER_AUTH_ERROR"
        if status == 429:
            return "PROVIDER_RATE_LIMIT"
        if status in {400, 404, 422}:
            return "PROVIDER_BAD_REQUEST"
        return "PROVIDER_HTTP_ERROR"

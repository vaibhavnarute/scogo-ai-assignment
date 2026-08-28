from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from harness.errors import HarnessError
from harness.providers.nvidia import NvidiaConfig, NvidiaProvider


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def provider() -> NvidiaProvider:
    return NvidiaProvider(NvidiaConfig(model="model", max_retries=1, retry_backoff_seconds=0))


def test_normalizes_nvidia_tool_response(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    payload = {
        "id": "req-1",
        "model": "model",
        "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [{"id": "tc-1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"app.py"}'}}]}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 3}},
    }
    with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(payload)) as opened:
        response = provider().generate([{"role": "user", "content": "repair"}], [{"name": "read_file", "description": "read", "parameters": {"type": "object"}}])
    request = opened.call_args.args[0]
    body = json.loads(request.data)
    assert body["tools"][0]["type"] == "function"
    assert response.tool_calls[0].arguments == {"path": "app.py"}
    assert response.usage.input_tokens == 12 and response.usage.cached_tokens == 3
    assert response.provider_request_id == "req-1"


def test_normalizes_allowlisted_nvidia_harmony_tool_suffix(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "id": "harmony",
                            "function": {
                                "name": "finish<|channel|>json",
                                "arguments": '{"summary":"fixed","evidence":"pytest"}',
                            },
                        }
                    ]
                },
            }
        ]
    }
    tools = [{"name": "finish", "description": "finish", "parameters": {"type": "object"}}]
    with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(payload)):
        response = provider().generate([], tools)
    assert response.tool_calls[0].name == "finish"
    assert response.raw_metadata["tool_name_repairs"] == [
        {
            "tool_call_id": "harmony",
            "raw_tool_name": "finish<|channel|>json",
            "normalized_tool_name": "finish",
            "reason": "nvidia_harmony_channel_suffix",
        }
    ]


def test_does_not_normalize_harmony_suffix_to_unoffered_tool(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {"id": "unknown", "function": {"name": "outside_tool<|channel|>json", "arguments": "{}"}}
                    ]
                },
            }
        ]
    }
    with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(payload)):
        response = provider().generate([], [{"name": "finish"}])
    assert response.tool_calls[0].name == "outside_tool<|channel|>json"
    assert response.raw_metadata["tool_name_repairs"] == []


def test_malformed_provider_arguments_become_recoverable_protocol_input(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    payload = {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{"id": "bad", "function": {"name": "read_file", "arguments": "{bad json"}}]}}]}
    with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(payload)):
        response = provider().generate([], [])
    assert response.tool_calls[0].arguments == "{bad json"


def test_rate_limit_is_retried(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    error = urllib.error.HTTPError("https://example.invalid", 429, "rate", {}, io.BytesIO())
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": "ok", "tool_calls": []}}]}
    with patch("urllib.request.urlopen", side_effect=[error, FakeHTTPResponse(payload)]):
        response = provider().generate([], [])
    assert response.retry_count == 1


def test_missing_choices_response_is_retried(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": "ok", "tool_calls": []}}]}
    with patch("urllib.request.urlopen", side_effect=[FakeHTTPResponse({"status": "temporarily unavailable"}), FakeHTTPResponse(payload)]) as opened:
        response = provider().generate([], [])
    assert opened.call_count == 2
    assert response.text == "ok"
    assert response.retry_count == 1


def test_exhausted_bad_responses_are_recoverable_and_do_not_expose_payload(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    bad_payload = {"error": {"message": "sensitive provider detail"}}
    with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(bad_payload)):
        with pytest.raises(HarnessError) as caught:
            provider().generate([], [])
    assert caught.value.code == "PROVIDER_BAD_RESPONSE"
    assert caught.value.recoverable
    assert caught.value.details == {"retry_count": 1}
    assert "sensitive provider detail" not in str(caught.value)


def test_missing_api_key_is_configuration_error(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(HarnessError) as caught:
        provider().generate([], [])
    assert caught.value.code == "CONFIG_MISSING_API_KEY" and not caught.value.recoverable


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"message": []}]},
        {"choices": [{"message": {"tool_calls": {}}}]},
        {"choices": [{"message": {"tool_calls": ["bad"]}}]},
        {"choices": [{"message": {"tool_calls": [{"function": []}]}}]},
        {"choices": [{"message": {"tool_calls": [{"function": {"name": "read_file", "arguments": []}}]}}]},
        {"choices": [{"message": {}}], "usage": []},
        {"choices": [{"message": {}}], "usage": {"prompt_tokens": "not-a-number"}},
        {"choices": [{"message": {}}], "usage": {"prompt_tokens_details": []}},
    ],
)
def test_malformed_nested_payloads_use_provider_error_boundary(monkeypatch, payload):
    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    configured = NvidiaProvider(NvidiaConfig(model="model", max_retries=0))
    with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(payload)):
        with pytest.raises(HarnessError) as caught:
            configured.generate([], [])
    assert caught.value.code == "PROVIDER_BAD_RESPONSE" and caught.value.recoverable

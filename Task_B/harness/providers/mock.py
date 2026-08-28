"""Scripted provider for deterministic loop and evaluation tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ..errors import ErrorCategory, HarnessError
from .base import ModelProvider
from .types import ModelResponse

MockStep = ModelResponse | HarnessError | Callable[[list[dict[str, Any]], list[dict[str, Any]]], ModelResponse]


class MockProvider(ModelProvider):
    provider_name = "mock"
    model = "scripted"

    def __init__(self, responses: Iterable[MockStep]) -> None:
        self._responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        self.requests.append({"messages": messages, "tools": tools})
        try:
            step = next(self._responses)
        except StopIteration as exc:
            raise HarnessError("MOCK_RESPONSES_EXHAUSTED", ErrorCategory.PROVIDER, "mock provider has no response for this turn", False) from exc
        if isinstance(step, HarnessError):
            raise step
        if callable(step):
            return step(messages, tools)
        return step


"""Abstract provider boundary consumed by the agent loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import ModelResponse


class ModelProvider(ABC):
    provider_name: str
    model: str

    @abstractmethod
    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        raise NotImplementedError


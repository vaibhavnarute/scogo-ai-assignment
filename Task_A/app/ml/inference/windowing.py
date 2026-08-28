"""Pure, deterministic helpers for sliding-window sequence inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class WindowPlan:
    token_windows: tuple[tuple[int, ...], ...]
    weights: tuple[int, ...]
    original_token_count: int
    was_chunked: bool
    window_size: int
    stride: int

    @property
    def chunks_used(self) -> int:
        return len(self.token_windows)


def create_window_plan(
    token_ids: Sequence[int],
    *,
    window_size: int,
    stride: int,
    special_tokens_count: int,
) -> WindowPlan:
    """Split content-token IDs into windows; ``stride`` is overlap token count."""
    if window_size <= special_tokens_count:
        raise ValueError("Window size must leave room for content tokens")
    payload_size = window_size - special_tokens_count
    if stride < 0 or stride >= payload_size:
        raise ValueError("Stride must be non-negative and smaller than window payload")

    content = tuple(int(token_id) for token_id in token_ids)
    original_count = len(content) + special_tokens_count
    if len(content) <= payload_size:
        windows = (content,)
    else:
        step = payload_size - stride
        starts = range(0, len(content), step)
        generated = [content[start : start + payload_size] for start in starts]
        windows = tuple(window for window in generated if window)

    return WindowPlan(
        token_windows=windows,
        weights=tuple(len(window) for window in windows),
        original_token_count=original_count,
        was_chunked=len(windows) > 1,
        window_size=window_size,
        stride=stride,
    )


def weighted_mean_probabilities(
    probabilities: Sequence[Sequence[float]], weights: Sequence[int]
) -> np.ndarray:
    """Aggregate chunk probabilities by represented content-token count."""
    values = np.asarray(probabilities, dtype=float)
    weight_values = np.asarray(weights, dtype=float)
    if values.ndim != 2 or len(values) != len(weight_values) or len(values) == 0:
        raise ValueError("Probabilities and weights must describe the same non-empty chunks")
    if np.any(weight_values <= 0):
        raise ValueError("Every chunk weight must be positive")
    aggregated = np.average(values, axis=0, weights=weight_values)
    return aggregated / aggregated.sum()

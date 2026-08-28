import numpy as np

from app.ml.inference.windowing import create_window_plan, weighted_mean_probabilities


def plan(token_count: int):
    return create_window_plan(
        list(range(token_count)),
        window_size=256,
        stride=64,
        special_tokens_count=2,
    )


def test_short_input_uses_one_window() -> None:
    result = plan(20)
    assert result.original_token_count == 22
    assert result.chunks_used == 1
    assert result.was_chunked is False


def test_exact_window_input_is_not_chunked() -> None:
    result = plan(254)
    assert result.original_token_count == 256
    assert result.chunks_used == 1
    assert result.was_chunked is False


def test_over_length_input_uses_overlapping_windows() -> None:
    result = plan(255)
    assert result.chunks_used == 2
    assert result.was_chunked is True
    assert result.token_windows[0][-64:] == result.token_windows[1][:64]


def test_multi_chunk_probability_aggregation_is_token_weighted() -> None:
    probabilities = [[0.9, 0.1], [0.2, 0.8]]
    aggregated = weighted_mean_probabilities(probabilities, [3, 1])
    assert np.allclose(aggregated, [0.725, 0.275])
    assert np.isclose(aggregated.sum(), 1.0)


def test_windowing_is_deterministic() -> None:
    assert plan(700) == plan(700)

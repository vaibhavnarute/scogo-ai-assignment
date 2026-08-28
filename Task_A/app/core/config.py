"""Central, serializable configuration for training and serving."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ExperimentName = Literal["head_only", "partial", "full"]


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "distilbert-base-uncased")
    dataset_name: str = os.getenv("DATASET_NAME", "mteb/amazon_polarity")
    max_length: int = _env_int("MAX_LENGTH", 256)
    train_size: int = _env_int("TRAIN_SIZE", 5_000)
    val_size: int = _env_int("VAL_SIZE", 1_000)
    test_size: int = _env_int("TEST_SIZE", 1_000)
    batch_size: int = _env_int("BATCH_SIZE", 8)
    epochs: int = _env_int("EPOCHS", 2)
    learning_rate: float = _env_float("LEARNING_RATE", 2e-5)
    weight_decay: float = _env_float("WEIGHT_DECAY", 0.01)
    seed: int = _env_int("SEED", 42)
    stream_shuffle_buffer: int = _env_int("STREAM_SHUFFLE_BUFFER", 10_000)
    inference_window_size: int = _env_int("INFERENCE_WINDOW_SIZE", 256)
    inference_stride: int = _env_int("INFERENCE_STRIDE", 64)
    calibration_bins: int = _env_int("CALIBRATION_BINS", 15)
    aspect_confidence_threshold: float = _env_float("ASPECT_CONFIDENCE_THRESHOLD", 0.70)
    aspect_max_segments: int = _env_int("ASPECT_MAX_SEGMENTS", 12)
    models_dir: Path = PROJECT_ROOT / "models"
    results_dir: Path = PROJECT_ROOT / "results"
    data_cache_dir: Path = PROJECT_ROOT / ".cache" / "dataset"

    @property
    def calibration_path(self) -> Path:
        return self.results_dir / "calibration" / "temperature.json"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}


settings = Settings()

"""Deterministic, balanced Amazon Polarity subset creation."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from app.core.config import Settings, settings

LABELS = {0: "negative", 1: "positive"}


def sample_balanced_records(
    records: Iterable[Mapping[str, Any]], size: int, seed: int
) -> list[dict[str, Any]]:
    """Collect an exact 50/50 sample from a deterministic record stream."""
    if size <= 0 or size % 2:
        raise ValueError("Balanced subset size must be a positive even integer")

    per_class = size // 2
    counts: Counter[int] = Counter()
    selected: list[dict[str, Any]] = []
    for record in records:
        label = int(record.get("label", -1))
        text = str(record.get("text", "")).strip()
        if label not in LABELS or not text or counts[label] >= per_class:
            continue
        selected.append({"text": text, "label": label})
        counts[label] += 1
        if counts[0] == per_class and counts[1] == per_class:
            break

    if len(selected) != size:
        raise RuntimeError(
            f"Could only collect {len(selected)}/{size} records; class counts={dict(counts)}"
        )
    random.Random(seed).shuffle(selected)
    return selected


def _stream_split(dataset_name: str, split: str, seed: int, buffer_size: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Install requirements.txt before loading the dataset") from exc

    stream = load_dataset(dataset_name, split=split, streaming=True)
    return stream.shuffle(seed=seed, buffer_size=buffer_size)


def _dataset_from_records(records: list[dict[str, Any]]):
    from datasets import Dataset

    return Dataset.from_list(records)


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_train_validation(config: Settings = settings, refresh: bool = False):
    """Load or create disjoint train/validation subsets from canonical train."""
    from datasets import DatasetDict, load_from_disk

    if config.train_size % 2 or config.val_size % 2:
        raise ValueError("Train and validation sizes must both be even for exact balance")
    cache_path = config.data_cache_dir / "train_validation"
    if cache_path.exists() and not refresh:
        return load_from_disk(str(cache_path))

    total = config.train_size + config.val_size
    records = sample_balanced_records(
        _stream_split(
            config.dataset_name,
            "train",
            config.seed,
            config.stream_shuffle_buffer,
        ),
        total,
        config.seed,
    )
    # The combined pool is balanced. Split each class separately to preserve exact
    # balance and guarantee no overlap between train and validation.
    by_label = {label: [row for row in records if row["label"] == label] for label in LABELS}
    train_per_class = config.train_size // 2
    train_rows = by_label[0][:train_per_class] + by_label[1][:train_per_class]
    val_rows = by_label[0][train_per_class:] + by_label[1][train_per_class:]
    random.Random(config.seed).shuffle(train_rows)
    random.Random(config.seed + 1).shuffle(val_rows)

    bundle = DatasetDict(
        train=_dataset_from_records(train_rows),
        validation=_dataset_from_records(val_rows),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    bundle.save_to_disk(str(cache_path))
    _write_metadata(
        cache_path,
        {
            "source": config.dataset_name,
            "canonical_split": "train",
            "seed": config.seed,
            "train_size": len(train_rows),
            "validation_size": len(val_rows),
            "label_mapping": LABELS,
        },
    )
    return bundle


def load_final_test(config: Settings = settings, refresh: bool = False):
    """Load or create the final subset exclusively from canonical test."""
    from datasets import load_from_disk

    cache_path = config.data_cache_dir / "final_test"
    if cache_path.exists() and not refresh:
        return load_from_disk(str(cache_path))

    records = sample_balanced_records(
        _stream_split(
            config.dataset_name,
            "test",
            config.seed + 2,
            config.stream_shuffle_buffer,
        ),
        config.test_size,
        config.seed + 2,
    )
    dataset = _dataset_from_records(records)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_path))
    _write_metadata(
        cache_path,
        {
            "source": config.dataset_name,
            "canonical_split": "test",
            "seed": config.seed + 2,
            "test_size": len(records),
            "label_mapping": LABELS,
        },
    )
    return dataset


def label_counts(dataset) -> dict[int, int]:
    return dict(sorted(Counter(int(label) for label in dataset["label"]).items()))

"""Shared metric, baseline, and artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(labels, predictions) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=int)
    predictions_array = np.asarray(predictions, dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_array,
        predictions_array,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels_array, predictions_array)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            "negative": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "positive": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "confusion_matrix": confusion_matrix(
            labels_array, predictions_array, labels=[0, 1]
        ).tolist(),
    }


def trainer_metrics(eval_prediction) -> dict[str, float]:
    predictions = np.argmax(eval_prediction.predictions, axis=-1)
    metrics = classification_metrics(eval_prediction.label_ids, predictions)
    return {
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
    }


def baseline_metrics(labels, seed: int) -> dict[str, dict[str, Any]]:
    labels_array = np.asarray(labels, dtype=int)
    values, counts = np.unique(labels_array, return_counts=True)
    majority_label = int(values[np.argmax(counts)])
    rng = np.random.default_rng(seed)
    return {
        "majority": classification_metrics(
            labels_array, np.full(labels_array.shape, majority_label, dtype=int)
        ),
        "seeded_random": classification_metrics(
            labels_array, rng.integers(0, 2, size=len(labels_array))
        ),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_confusion_matrix(path: Path, matrix: list[list[int]], title: str) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(matrix), display_labels=["negative", "positive"]
    )
    display.plot(cmap="Blues", values_format="d")
    display.ax_.set_title(title)
    display.figure_.tight_layout()
    display.figure_.savefig(path, dpi=160)
    plt.close(display.figure_)

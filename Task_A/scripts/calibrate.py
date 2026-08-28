"""Fit validation-only temperature scaling for the selected model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.core.logging import configure_logging
from app.ml.data.dataset import load_train_validation
from app.ml.evaluation.calibration import (
    apply_temperature,
    calibration_metrics,
    fit_temperature,
    softmax,
)
from app.ml.evaluation.evaluator import save_json


def collect_validation_logits():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    validation = load_train_validation(settings)["validation"]
    model_path = settings.models_dir / "best"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logits_parts: list[np.ndarray] = []
    labels: list[int] = []
    for start in range(0, len(validation), settings.batch_size):
        rows = validation[start : start + settings.batch_size]
        encoded = tokenizer(
            rows["text"],
            padding=True,
            truncation=True,
            max_length=settings.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits_parts.append(model(**encoded).logits.cpu().numpy())
        labels.extend(int(label) for label in rows["label"])
    return np.concatenate(logits_parts), np.asarray(labels, dtype=int), str(device)


def save_reliability_diagram(
    path: Path,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    labels: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    for probabilities, name, marker in (
        (raw_probabilities, "Raw", "o"),
        (calibrated_probabilities, "Temperature scaled", "s"),
    ):
        confidence = probabilities.max(axis=1)
        correct = (probabilities.argmax(axis=1) == labels).astype(float)
        xs, ys = [], []
        edges = np.linspace(0.0, 1.0, settings.calibration_bins + 1)
        for index in range(settings.calibration_bins):
            mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
            if np.any(mask):
                xs.append(float(confidence[mask].mean()))
                ys.append(float(correct[mask].mean()))
        axis.plot(xs, ys, marker=marker, label=name)
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axis.set(xlabel="Mean confidence", ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1))
    axis.set_title("Validation reliability diagram")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    configure_logging()
    logits, labels, device = collect_validation_logits()
    temperature = fit_temperature(logits, labels)
    raw_probabilities = softmax(logits)
    calibrated_probabilities = apply_temperature(logits, temperature)
    raw_predictions = raw_probabilities.argmax(axis=1)
    calibrated_predictions = calibrated_probabilities.argmax(axis=1)
    if not np.array_equal(raw_predictions, calibrated_predictions):
        raise RuntimeError("Positive temperature unexpectedly changed validation labels")

    result_dir = settings.results_dir / "calibration"
    selection = json.loads(
        (settings.results_dir / "model_selection.json").read_text(encoding="utf-8")
    )
    artifact = {
        "temperature": temperature,
        "method": "single_scalar_temperature_scaling",
        "fit_objective": "validation_negative_log_likelihood",
        "fit_split": "canonical train-derived validation subset",
        "validation_size": len(labels),
        "validation_seed": settings.seed + 1,
        "base_model": settings.model_name,
        "model_selection": selection,
        "device": device,
        "model_weights_changed": False,
    }
    before = calibration_metrics(raw_probabilities, labels, settings.calibration_bins)
    after = calibration_metrics(calibrated_probabilities, labels, settings.calibration_bins)
    before["temperature"] = 1.0
    after["temperature"] = temperature
    after["class_labels_unchanged"] = True
    save_json(result_dir / "temperature.json", artifact)
    save_json(result_dir / "metrics_before.json", before)
    save_json(result_dir / "metrics_after.json", after)
    save_reliability_diagram(
        result_dir / "reliability_diagram.png",
        raw_probabilities,
        calibrated_probabilities,
        labels,
    )
    print(json.dumps({"temperature": artifact, "before": before, "after": after}, indent=2))


if __name__ == "__main__":
    main()

"""Temperature scaling and calibration metrics for binary classifier logits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a finite positive number")
    return softmax(np.asarray(logits, dtype=float) / temperature)


def calibration_metrics(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.shape[0]:
        raise ValueError("Probability rows and labels must align")
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correctness = (predictions == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if index == 0:
            mask |= confidence == 0.0
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    one_hot = np.eye(probabilities.shape[1])[labels]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    selected = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    nll = float(-np.mean(np.log(selected)))
    return {
        "ece": float(ece),
        "brier_score": brier,
        "nll": nll,
        "accuracy": float(np.mean(predictions == labels)),
        "mean_confidence": float(np.mean(confidence)),
    }


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one positive scalar by minimizing validation cross-entropy."""
    import torch

    logits_tensor = torch.tensor(np.asarray(logits), dtype=torch.float64)
    labels_tensor = torch.tensor(np.asarray(labels), dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.05, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=1e-3, max=100.0)
        loss = torch.nn.functional.cross_entropy(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).clamp(min=1e-3, max=100.0).detach())


@dataclass(frozen=True)
class TemperatureArtifact:
    temperature: float
    path: Path | None = None

    @classmethod
    def load(cls, path: Path) -> "TemperatureArtifact":
        payload = json.loads(path.read_text(encoding="utf-8"))
        temperature = float(payload["temperature"])
        if temperature <= 0:
            raise ValueError("Calibration artifact contains a non-positive temperature")
        return cls(temperature=temperature, path=path)

    @classmethod
    def load_or_identity(cls, path: Path) -> "TemperatureArtifact":
        return cls.load(path) if path.exists() else cls(temperature=1.0, path=None)

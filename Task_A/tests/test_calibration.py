from pathlib import Path

import numpy as np

from app.ml.evaluation.calibration import (
    TemperatureArtifact,
    apply_temperature,
    fit_temperature,
)


def test_temperature_is_positive_and_probabilities_sum_to_one() -> None:
    logits = np.array([[4.0, 1.0], [0.5, 2.0], [3.0, -1.0]])
    labels = np.array([0, 1, 1])
    temperature = fit_temperature(logits, labels)
    probabilities = apply_temperature(logits, temperature)
    assert temperature > 0
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_temperature_scaling_preserves_class_labels() -> None:
    logits = np.array([[4.0, 1.0], [0.5, 2.0], [-2.0, -1.0]])
    raw_labels = logits.argmax(axis=1)
    calibrated_labels = apply_temperature(logits, 2.5).argmax(axis=1)
    assert np.array_equal(raw_labels, calibrated_labels)


def test_calibration_artifact_loads() -> None:
    path = Path(__file__).parent / "fixtures" / "temperature.json"
    artifact = TemperatureArtifact.load(path)
    assert artifact.temperature == 1.75
    assert artifact.path == path

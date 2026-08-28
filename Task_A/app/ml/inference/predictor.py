"""Single inference service shared by FastAPI and Gradio."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.ml.evaluation.calibration import TemperatureArtifact
from app.ml.inference.aspects import analyze_aspects
from app.ml.inference.windowing import create_window_plan, weighted_mean_probabilities


logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path or settings.models_dir / "best")
        if not (self.model_path / "config.json").exists():
            raise FileNotFoundError(
                f"No selected model found at {self.model_path}. Run scripts/train.py first."
            )
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("Install requirements.txt before serving predictions") from exc

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        self.calibration = TemperatureArtifact.load_or_identity(settings.calibration_path)
        logger.info(
            "predictor_loaded model_path=%s device=%s calibration_applied=%s",
            self.model_path,
            self.device,
            self.calibration.path is not None,
        )

    def predict(self, text: str) -> dict[str, Any]:
        normalized = text.strip()
        if not normalized:
            raise ValueError("Review text must not be empty")

        content_ids = self.tokenizer(
            normalized,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
        if not content_ids:
            raise ValueError("Review text did not produce any model tokens")
        special_tokens_count = self.tokenizer.num_special_tokens_to_add(pair=False)
        model_limit = int(getattr(self.model.config, "max_position_embeddings", 512))
        window_size = min(settings.inference_window_size, model_limit)
        plan = create_window_plan(
            content_ids,
            window_size=window_size,
            stride=settings.inference_stride,
            special_tokens_count=special_tokens_count,
        )

        prepared = [
            self.tokenizer.prepare_for_model(
                list(window),
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=True,
            )
            for window in plan.token_windows
        ]
        encoded = self.tokenizer.pad(prepared, padding=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            logits = self.model(**encoded).logits
            raw_chunks = self._torch.softmax(logits, dim=-1).cpu().numpy()
            calibrated_chunks = self._torch.softmax(
                logits / self.calibration.temperature, dim=-1
            ).cpu().numpy()
        raw_probabilities = weighted_mean_probabilities(raw_chunks, plan.weights)
        calibrated_probabilities = weighted_mean_probabilities(
            calibrated_chunks, plan.weights
        )
        # Positive scalar temperature preserves every chunk argmax. We also keep
        # the final label tied to raw aggregation for strict API compatibility.
        class_id = int(raw_probabilities.argmax())
        label = self.model.config.id2label.get(class_id, str(class_id)).lower()
        result = {
            "sentiment": label,
            "confidence": round(float(raw_probabilities[class_id]), 4),
            "calibrated_confidence": round(
                float(calibrated_probabilities[class_id]), 4
            ),
            "original_token_count": plan.original_token_count,
            "chunks_used": plan.chunks_used,
            "was_chunked": plan.was_chunked,
            "window_size": plan.window_size,
            "stride": plan.stride,
            "calibration_applied": self.calibration.path is not None,
        }
        logger.info(
            "prediction_completed original_token_count=%s chunks_used=%s was_chunked=%s",
            plan.original_token_count,
            plan.chunks_used,
            plan.was_chunked,
        )
        return result

    def predict_with_diagnostics(self, text: str) -> dict[str, Any]:
        """Preserve overall binary output and add heuristic aspect diagnostics."""
        result = self.predict(text)
        result.update(
            analyze_aspects(
                text,
                self.predict,
                confidence_threshold=settings.aspect_confidence_threshold,
                max_segments=settings.aspect_max_segments,
            )
        )
        return result

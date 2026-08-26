"""Single inference service shared by FastAPI and Gradio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import settings


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

    def predict(self, text: str) -> dict[str, Any]:
        normalized = text.strip()
        if not normalized:
            raise ValueError("Review text must not be empty")

        encoded = self.tokenizer(
            normalized,
            truncation=True,
            max_length=settings.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            logits = self.model(**encoded).logits
            probabilities = self._torch.softmax(logits, dim=-1)[0]
        class_id = int(self._torch.argmax(probabilities).item())
        label = self.model.config.id2label.get(class_id, str(class_id)).lower()
        return {
            "sentiment": label,
            "confidence": round(float(probabilities[class_id].item()), 4),
        }

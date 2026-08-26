"""Export and summarize all mistakes from the completed final-test evaluation.

This script is analysis-only: it loads the already selected model and cached final
test subset, performs no training, and must not be used to tune the model.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.ml.data.dataset import LABELS, load_final_test

CONTRAST_PATTERN = re.compile(
    r"\b(but|however|although|though|yet|except|despite|while|on the other hand)\b",
    re.IGNORECASE,
)
NEGATION_PATTERN = re.compile(
    r"\b(no|not|never|neither|nor|hardly|barely|without|isn't|wasn't|don't|didn't|can't|won't)\b",
    re.IGNORECASE,
)


def heuristic_category(text: str, token_count: int) -> str:
    """Assign a transparent review cue for organizing manual inspection."""
    if token_count > settings.max_length:
        return "truncated_over_256_tokens"
    if CONTRAST_PATTERN.search(text):
        return "mixed_or_contrast_language"
    if NEGATION_PATTERN.search(text):
        return "negation_language"
    if len(text.split()) <= 12:
        return "short_or_ambiguous"
    return "other_or_domain_specific"


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dataset = load_final_test(settings)
    model_path = settings.models_dir / "best"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    errors: list[dict[str, object]] = []
    for start in range(0, len(dataset), settings.batch_size):
        rows = dataset[start : start + settings.batch_size]
        texts = rows["text"]
        labels = [int(label) for label in rows["label"]]
        tokenized_untruncated = tokenizer(texts, truncation=False, add_special_tokens=True)
        token_counts = [len(input_ids) for input_ids in tokenized_untruncated["input_ids"]]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(model(**encoded).logits, dim=-1).cpu()
        predictions = torch.argmax(probabilities, dim=-1).tolist()

        for offset, (text, true_id, predicted_id, token_count) in enumerate(
            zip(texts, labels, predictions, token_counts, strict=True)
        ):
            if true_id == predicted_id:
                continue
            errors.append(
                {
                    "test_index": start + offset,
                    "true_label": LABELS[true_id],
                    "predicted_label": LABELS[predicted_id],
                    "confidence": round(float(probabilities[offset, predicted_id]), 6),
                    "token_count": token_count,
                    "heuristic_category": heuristic_category(text, token_count),
                    "text": text,
                }
            )

    result_dir = settings.results_dir / "final_test"
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(errors[0]))
        writer.writeheader()
        writer.writerows(errors)

    categories = Counter(str(error["heuristic_category"]) for error in errors)
    directions = Counter(
        f'{error["true_label"]}_as_{error["predicted_label"]}' for error in errors
    )
    high_confidence = sorted(errors, key=lambda row: float(row["confidence"]), reverse=True)
    payload = {
        "analysis_guardrail": (
            "Post-hoc reporting on the final test set only; these observations were not used "
            "for model selection or tuning."
        ),
        "total_test_examples": len(dataset),
        "total_errors": len(errors),
        "error_rate": len(errors) / len(dataset),
        "direction_counts": dict(sorted(directions.items())),
        "heuristic_category_counts": dict(sorted(categories.items())),
        "high_confidence_error_count": sum(
            float(error["confidence"]) >= 0.9 for error in errors
        ),
        "top_high_confidence_errors": high_confidence[:10],
    }
    (result_dir / "error_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved all {len(errors)} mistakes to {csv_path}")


if __name__ == "__main__":
    main()

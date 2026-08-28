"""Evaluate the selected binary model on the diagnostic challenge suite.

The challenge suite is synthetic, subjective, and completely separate from model
training, validation selection, and final-test reporting. Its per-category scores
are diagnostic signals, not production benchmark claims.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.ml.inference.predictor import Predictor

DATA_PATH = PROJECT_ROOT / "data" / "challenge_examples.json"
RESULT_DIR = settings.results_dir / "challenge_suite"
ALLOWED_LABELS = {"negative", "positive"}
EXPECTED_CATEGORIES = {
    "sarcasm",
    "mixed_sentiment",
    "product_vs_delivery",
    "support_complaint",
    "negation",
    "very_long_review",
    "domain_shift",
}


def load_suite(path: Path = DATA_PATH) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples", [])
    if not 20 <= len(examples) <= 50:
        raise ValueError("Challenge suite must contain between 20 and 50 examples")
    ids = [example["id"] for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Challenge example IDs must be unique")
    categories = {example["category"] for example in examples}
    if categories != EXPECTED_CATEGORIES:
        raise ValueError(f"Expected categories {EXPECTED_CATEGORIES}, found {categories}")
    invalid_labels = {
        example["expected_sentiment"]
        for example in examples
        if example["expected_sentiment"] not in ALLOWED_LABELS
    }
    if invalid_labels:
        raise ValueError(f"Invalid binary sentiment labels: {invalid_labels}")
    return payload


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)

    category_metrics = {}
    for category in sorted(by_category):
        category_rows = by_category[category]
        correct = sum(bool(row["correct"]) for row in category_rows)
        category_metrics[category] = {
            "examples": len(category_rows),
            "correct": correct,
            "diagnostic_accuracy": correct / len(category_rows),
            "mean_confidence": sum(float(row["confidence"]) for row in category_rows)
            / len(category_rows),
        }

    correct = sum(bool(row["correct"]) for row in rows)
    expected_negative = [row for row in rows if row["expected_sentiment"] == "negative"]
    expected_positive = [row for row in rows if row["expected_sentiment"] == "positive"]
    confusion = [
        [
            sum(row["predicted_sentiment"] == "negative" for row in expected_negative),
            sum(row["predicted_sentiment"] == "positive" for row in expected_negative),
        ],
        [
            sum(row["predicted_sentiment"] == "negative" for row in expected_positive),
            sum(row["predicted_sentiment"] == "positive" for row in expected_positive),
        ],
    ]
    return {
        "total_examples": len(rows),
        "correct": correct,
        "diagnostic_accuracy": correct / len(rows),
        "expected_label_counts": dict(
            sorted(Counter(str(row["expected_sentiment"]) for row in rows).items())
        ),
        "prediction_counts": dict(
            sorted(Counter(str(row["predicted_sentiment"]) for row in rows).items())
        ),
        "confusion_matrix_labels": ["negative", "positive"],
        "confusion_matrix": confusion,
        "truncated_example_count": sum(bool(row["truncated"]) for row in rows),
        "category_metrics": category_metrics,
        "failures": [row for row in rows if not row["correct"]],
    }


def main() -> None:
    suite = load_suite()
    predictor = Predictor()
    selection_path = settings.results_dir / "model_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []

    for example in suite["examples"]:
        text = str(example["text"])
        prediction = predictor.predict(text)
        token_count = len(predictor.tokenizer(text, truncation=False)["input_ids"])
        predicted = str(prediction["sentiment"])
        expected = str(example["expected_sentiment"])
        rows.append(
            {
                "suite_id": suite["suite_id"],
                "example_id": example["id"],
                "category": example["category"],
                "expected_sentiment": expected,
                "predicted_sentiment": predicted,
                "correct": predicted == expected,
                "confidence": prediction["confidence"],
                "token_count": token_count,
                "truncated": token_count > settings.max_length,
                "rationale": example["rationale"],
                "text": text,
            }
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = RESULT_DIR / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary.update(
        {
            "suite_id": suite["suite_id"],
            "status": "diagnostic_only",
            "guardrail": (
                "Synthetic challenge examples were not used for training, validation model "
                "selection, final-test scoring, or hyperparameter tuning. Expected labels "
                "represent a human-authored dominant binary judgment and can be subjective."
            ),
            "model_selection": selection,
            "provenance": {
                "generated_by": "scripts/evaluate_challenges.py",
                "challenge_data": "data/challenge_examples.json",
                "predictions": "results/challenge_suite/predictions.csv",
                "base_model": settings.model_name,
                "selected_model_directory": "models/best",
                "max_length": settings.max_length,
            },
        }
    )
    summary_path = RESULT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()

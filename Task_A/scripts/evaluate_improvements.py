"""Generate isolated long-review and calibrated challenge comparisons."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.ml.inference.predictor import Predictor
from scripts.evaluate_challenges import load_suite


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "examples": len(rows),
        "correct": correct,
        "diagnostic_accuracy": correct / len(rows),
        "failures": [row for row in rows if not row["correct"]],
    }


def confidence_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    correct = [row for row in rows if row["correct"]]
    incorrect = [row for row in rows if not row["correct"]]

    def mean(group, key):
        return sum(float(row[key]) for row in group) / len(group) if group else None

    return {
        "examples": len(rows),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "raw_average_confidence_correct": mean(correct, "raw_confidence"),
        "raw_average_confidence_incorrect": mean(incorrect, "raw_confidence"),
        "calibrated_average_confidence_correct": mean(correct, "calibrated_confidence"),
        "calibrated_average_confidence_incorrect": mean(incorrect, "calibrated_confidence"),
        "raw_wrong_confidence_gt_0_90": sum(
            float(row["raw_confidence"]) > 0.90 for row in incorrect
        ),
        "raw_wrong_confidence_gt_0_95": sum(
            float(row["raw_confidence"]) > 0.95 for row in incorrect
        ),
        "calibrated_wrong_confidence_gt_0_90": sum(
            float(row["calibrated_confidence"]) > 0.90 for row in incorrect
        ),
        "calibrated_wrong_confidence_gt_0_95": sum(
            float(row["calibrated_confidence"]) > 0.95 for row in incorrect
        ),
    }


def main() -> None:
    if not settings.calibration_path.exists():
        raise FileNotFoundError("Run scripts/calibrate.py before improvement evaluation")
    suite = load_suite()
    predictor = Predictor()
    improved_rows: list[dict[str, object]] = []
    for example in suite["examples"]:
        prediction = predictor.predict(str(example["text"]))
        expected = str(example["expected_sentiment"])
        improved_rows.append(
            {
                "example_id": example["id"],
                "category": example["category"],
                "expected_sentiment": expected,
                "predicted_sentiment": prediction["sentiment"],
                "correct": prediction["sentiment"] == expected,
                "raw_confidence": prediction["confidence"],
                "calibrated_confidence": prediction["calibrated_confidence"],
                "original_token_count": prediction["original_token_count"],
                "chunks_used": prediction["chunks_used"],
                "was_chunked": prediction["was_chunked"],
                "window_size": prediction["window_size"],
                "stride": prediction["stride"],
                "text": example["text"],
            }
        )

    baseline_rows = read_csv(settings.results_dir / "challenge_suite" / "predictions.csv")
    baseline_long = [row for row in baseline_rows if row["category"] == "very_long_review"]
    improved_long = [row for row in improved_rows if row["category"] == "very_long_review"]
    before_rows = [
        {
            "example_id": row["example_id"],
            "expected_sentiment": row["expected_sentiment"],
            "predicted_sentiment": row["predicted_sentiment"],
            "correct": row["correct"].lower() == "true",
            "confidence": float(row["confidence"]),
            "token_count": int(row["token_count"]),
            "truncated": row["truncated"].lower() == "true",
        }
        for row in baseline_long
    ]
    long_dir = settings.results_dir / "improvements" / "long_review"
    long_dir.mkdir(parents=True, exist_ok=True)
    (long_dir / "before.json").write_text(
        json.dumps(diagnostic_summary(before_rows), indent=2), encoding="utf-8"
    )
    (long_dir / "after.json").write_text(
        json.dumps(diagnostic_summary(improved_long), indent=2), encoding="utf-8"
    )
    write_csv(long_dir / "predictions.csv", improved_long)

    calibration_dir = settings.results_dir / "improvements" / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    confidence = confidence_summary(improved_rows)
    confidence["guardrail"] = (
        "Challenge examples are diagnostic only; temperature was fitted solely on validation logits."
    )
    (calibration_dir / "challenge_confidence.json").write_text(
        json.dumps(confidence, indent=2), encoding="utf-8"
    )
    write_csv(calibration_dir / "challenge_predictions.csv", improved_rows)

    mixed_examples = [
        example for example in suite["examples"] if example["category"] == "mixed_sentiment"
    ]
    mixed_rows: list[dict[str, object]] = []
    for example in mixed_examples:
        prediction = predictor.predict_with_diagnostics(str(example["text"]))
        mixed_rows.append(
            {
                "example_id": example["id"],
                "expected_sentiment": example["expected_sentiment"],
                "predicted_sentiment": prediction["sentiment"],
                "overall_correct": prediction["sentiment"]
                == example["expected_sentiment"],
                "mixed_sentiment_detected": prediction["mixed_sentiment_detected"],
                "aspects": json.dumps(prediction["aspects"], ensure_ascii=False),
                "text": example["text"],
            }
        )
    mixed_summary = {
        "examples": len(mixed_rows),
        "binary_correct": sum(bool(row["overall_correct"]) for row in mixed_rows),
        "mixed_guardrail_detected": sum(
            bool(row["mixed_sentiment_detected"]) for row in mixed_rows
        ),
        "method": "heuristic clause/aspect extraction plus existing binary classifier",
        "claim": "diagnostic guardrail, not a trained aspect-sentiment model",
    }
    mixed_dir = settings.results_dir / "improvements" / "mixed_sentiment"
    mixed_dir.mkdir(parents=True, exist_ok=True)
    (mixed_dir / "summary.json").write_text(
        json.dumps(mixed_summary, indent=2), encoding="utf-8"
    )
    write_csv(mixed_dir / "predictions.csv", mixed_rows)

    report = {
        "original_held_out_test": {"accuracy": 0.924, "macro_f1": 0.9239987839805437},
        "original_challenge_suite": {"correct": 15, "examples": 28},
        "long_review": {
            "before": diagnostic_summary(before_rows),
            "after": diagnostic_summary(improved_long),
        },
        "calibration_challenge_confidence": confidence,
        "mixed_sentiment_guardrail": mixed_summary,
        "sarcasm_adaptation": {
            "baseline_correct": 1,
            "examples": 4,
            "status": "deferred; no challenge examples used for training",
        },
        "negation_adaptation": {
            "baseline_correct": 2,
            "examples": 4,
            "status": "deferred; no challenge examples used for training",
        },
        "unchanged_artifacts": [
            "results/final_test/metrics.json",
            "results/challenge_suite/summary.json",
        ],
    }
    report_path = settings.results_dir / "improvements" / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

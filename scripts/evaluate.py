"""Evaluate the validation-selected model once on canonical test data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.core.logging import configure_logging
from app.ml.data.dataset import label_counts, load_final_test
from app.ml.evaluation.evaluator import save_confusion_matrix, save_json
from app.ml.training.trainer import evaluate_saved_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data", action="store_true", help="Recreate the cached final-test subset."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    model_path = settings.models_dir / "best"
    selection_path = model_path / "selection.json"
    if not model_path.exists() or not selection_path.exists():
        raise FileNotFoundError("Train and select a model with scripts/train.py first")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    test_dataset = load_final_test(settings, refresh=args.refresh_data)
    print("Final test label counts:", label_counts(test_dataset))
    metrics = evaluate_saved_model(model_path, test_dataset, settings)
    metrics["selected_experiment"] = selection["selected_experiment"]
    result_dir = settings.results_dir / "final_test"
    save_json(result_dir / "metrics.json", metrics)
    save_json(result_dir / "config.json", settings.as_dict())
    save_confusion_matrix(
        result_dir / "confusion_matrix.png",
        metrics["confusion_matrix"],
        "Selected model (final test)",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

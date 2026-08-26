"""Train independent head-only, partial, and full experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.core.logging import configure_logging
from app.ml.data.dataset import label_counts, load_train_validation
from app.ml.evaluation.evaluator import baseline_metrics, save_json
from app.ml.training.trainer import run_experiment, run_random_head

EXPERIMENTS = ("head_only", "partial", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("all", *EXPERIMENTS),
        default="all",
        help="Run all experiments or only one independent experiment.",
    )
    parser.add_argument(
        "--refresh-data", action="store_true", help="Recreate the cached train/validation subsets."
    )
    parser.add_argument(
        "--skip-random-head", action="store_true", help="Skip the untrained-head validation baseline."
    )
    return parser.parse_args()


def promote_best_model() -> dict[str, object]:
    candidates: list[tuple[float, str]] = []
    for experiment in EXPERIMENTS:
        metrics_path = settings.results_dir / experiment / "metrics.json"
        model_path = settings.models_dir / experiment
        if metrics_path.exists() and model_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            candidates.append((float(metrics["macro_f1"]), experiment))
    if not candidates:
        raise RuntimeError("No completed experiment is available for model selection")

    score, winner = max(candidates)
    destination = settings.models_dir / "best"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(settings.models_dir / winner, destination)
    selection = {"selected_experiment": winner, "validation_macro_f1": score}
    save_json(destination / "selection.json", selection)
    # Keep a lightweight selection record outside the ignored model directory so
    # result provenance remains visible in Git without committing model weights.
    save_json(settings.results_dir / "model_selection.json", selection)
    return selection


def main() -> None:
    args = parse_args()
    configure_logging()
    datasets = load_train_validation(settings, refresh=args.refresh_data)
    print(
        "Dataset counts:",
        {split: label_counts(datasets[split]) for split in ("train", "validation")},
    )

    baselines = baseline_metrics(datasets["validation"]["label"], settings.seed)
    save_json(settings.results_dir / "baselines" / "metrics.json", baselines)
    if not args.skip_random_head:
        run_random_head(datasets["validation"], settings)

    selected = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    for experiment in selected:
        run_experiment(experiment, datasets["train"], datasets["validation"], settings)

    selection = promote_best_model()
    print("Best available model:", selection)
    print("Final test has not been loaded or evaluated. Run scripts/evaluate.py once ready.")


if __name__ == "__main__":
    main()

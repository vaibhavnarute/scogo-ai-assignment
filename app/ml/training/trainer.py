"""Independent DistilBERT experiment runner."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import ExperimentName, Settings, settings
from app.ml.evaluation.evaluator import (
    classification_metrics,
    save_confusion_matrix,
    save_json,
    trainer_metrics,
)

LOGGER = logging.getLogger(__name__)


def _dependencies():
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Install requirements.txt before training") from exc
    return {
        "torch": torch,
        "AutoModel": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "DataCollator": DataCollatorWithPadding,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def seed_everything(seed: int, deps: dict[str, Any]) -> None:
    """Seed every RNG used by data/model initialization and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch = deps["torch"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deps["set_seed"](seed)


def configure_trainable_layers(model, experiment: ExperimentName) -> dict[str, int]:
    """Freeze/unfreeze parameters according to one controlled experiment."""
    for parameter in model.parameters():
        parameter.requires_grad = experiment == "full"

    if experiment == "head_only":
        for module_name in ("pre_classifier", "classifier"):
            for parameter in getattr(model, module_name).parameters():
                parameter.requires_grad = True
    elif experiment == "partial":
        for block in model.distilbert.transformer.layer[-2:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for module_name in ("pre_classifier", "classifier"):
            for parameter in getattr(model, module_name).parameters():
                parameter.requires_grad = True
    elif experiment != "full":
        raise ValueError(f"Unknown experiment: {experiment}")

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"trainable_parameters": trainable, "total_parameters": total}


def _tokenize(dataset, tokenizer, max_length: int):
    return dataset.map(
        lambda batch: tokenizer(
            batch["text"], truncation=True, max_length=max_length
        ),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing reviews",
    )


def _model_and_tokenizer(config: Settings):
    deps = _dependencies()
    tokenizer = deps["AutoTokenizer"].from_pretrained(config.model_name)
    model = deps["AutoModel"].from_pretrained(
        config.model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
    return deps, model, tokenizer


def _training_arguments(config: Settings, output_dir: Path):
    deps = _dependencies()
    torch = deps["torch"]
    return deps["TrainingArguments"](
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=config.seed,
        data_seed=config.seed,
        report_to=[],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )


def _complete_metrics(prediction_output) -> dict[str, Any]:
    predictions = np.argmax(prediction_output.predictions, axis=-1)
    result = classification_metrics(prediction_output.label_ids, predictions)
    if "test_loss" in prediction_output.metrics:
        result["loss"] = float(prediction_output.metrics["test_loss"])
    return result


def run_random_head(validation_dataset, config: Settings = settings) -> dict[str, Any]:
    """Measure E0 without training or saving a model checkpoint."""
    deps = _dependencies()
    seed_everything(config.seed, deps)
    _, model, tokenizer = _model_and_tokenizer(config)
    tokenized_validation = _tokenize(validation_dataset, tokenizer, config.max_length)
    args = deps["TrainingArguments"](
        output_dir=str(config.results_dir / "random_head" / "runtime"),
        per_device_eval_batch_size=config.batch_size,
        seed=config.seed,
        report_to=[],
    )
    trainer = deps["Trainer"](
        model=model,
        args=args,
        eval_dataset=tokenized_validation,
        data_collator=deps["DataCollator"](tokenizer=tokenizer),
        compute_metrics=trainer_metrics,
    )
    metrics = _complete_metrics(trainer.predict(tokenized_validation))
    result_dir = config.results_dir / "random_head"
    save_json(result_dir / "metrics.json", metrics)
    save_confusion_matrix(
        result_dir / "confusion_matrix.png",
        metrics["confusion_matrix"],
        "Random classification head (validation)",
    )
    return metrics


def run_experiment(
    experiment: ExperimentName,
    train_dataset,
    validation_dataset,
    config: Settings = settings,
) -> dict[str, Any]:
    """Run one experiment from the original pretrained checkpoint."""
    deps = _dependencies()
    seed_everything(config.seed, deps)
    _, model, tokenizer = _model_and_tokenizer(config)
    parameter_counts = configure_trainable_layers(model, experiment)
    LOGGER.info("Starting %s: %s", experiment, parameter_counts)

    tokenized_train = _tokenize(train_dataset, tokenizer, config.max_length)
    tokenized_validation = _tokenize(validation_dataset, tokenizer, config.max_length)
    runtime_dir = config.results_dir / experiment / "runtime"
    trainer = deps["Trainer"](
        model=model,
        args=_training_arguments(config, runtime_dir),
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        data_collator=deps["DataCollator"](tokenizer=tokenizer),
        compute_metrics=trainer_metrics,
    )
    train_result = trainer.train()
    prediction_output = trainer.predict(tokenized_validation)
    metrics = _complete_metrics(prediction_output)
    metrics.update(parameter_counts)
    metrics["experiment"] = experiment
    metrics["train_runtime"] = float(train_result.metrics.get("train_runtime", 0.0))

    model_dir = config.models_dir / experiment
    result_dir = config.results_dir / experiment
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    save_json(result_dir / "metrics.json", metrics)
    save_json(result_dir / "config.json", config.as_dict())
    save_confusion_matrix(
        result_dir / "confusion_matrix.png",
        metrics["confusion_matrix"],
        f"{experiment.replace('_', ' ').title()} (validation)",
    )
    return metrics


def evaluate_saved_model(model_path: Path, dataset, config: Settings = settings) -> dict[str, Any]:
    deps = _dependencies()
    seed_everything(config.seed, deps)
    tokenizer = deps["AutoTokenizer"].from_pretrained(str(model_path))
    model = deps["AutoModel"].from_pretrained(str(model_path))
    tokenized = _tokenize(dataset, tokenizer, config.max_length)
    args = deps["TrainingArguments"](
        output_dir=str(config.results_dir / "final_test" / "runtime"),
        per_device_eval_batch_size=config.batch_size,
        seed=config.seed,
        report_to=[],
    )
    trainer = deps["Trainer"](
        model=model,
        args=args,
        eval_dataset=tokenized,
        data_collator=deps["DataCollator"](tokenizer=tokenizer),
        compute_metrics=trainer_metrics,
    )
    return _complete_metrics(trainer.predict(tokenized))

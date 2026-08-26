# Scogo AI Assignment

This repository contains the assignment tasks as independent, reproducible projects. Each task lives in its own folder with task-specific source code, commands, tests, documentation, and measured artifacts.

## Task status

| Task | Project | Status | Documentation |
|---|---|---|---|
| Task A | Customer-review sentiment classification with DistilBERT | **Completed** | [Task_A/README.md](Task_A/README.md) |
| Task B | Next assignment task | **In progress / pending specification** | Will be added under `Task_B/` |

## Task A summary

Task A fine-tunes `distilbert-base-uncased` to classify Amazon product reviews as positive or negative. It uses deterministic, balanced subsets from `mteb/amazon_polarity`, compares independent training strategies, selects the winner using validation Macro-F1, and evaluates the selected model on an untouched final-test subset.

The project includes:

- Independent head-only, partial, and full fine-tuning experiments
- Majority, seeded-random, and untrained-classification-head baselines
- FastAPI inference endpoint and thin Gradio demonstration
- Deterministic data preparation and reproducible configuration
- Automated tests and real-model smoke checks
- Validation/test metrics, confusion matrices, and analysis of all 76 test errors
- Traceable lightweight result artifacts without committed model weights

### Before and after fine-tuning

All values below are measured. Validation contains 1,000 balanced reviews. The final test contains a separate 1,000 balanced reviews from the canonical test split.

| Stage | Evaluation split | Accuracy | Macro-F1 | Change from random head |
|---|---|---:|---:|---:|
| Majority baseline | Validation | 50.0% | 33.33% | — |
| Seeded random predictor | Validation | 51.3% | 51.30% | — |
| **Before fine-tuning: pretrained encoder + random head** | Validation | **38.5%** | **38.29%** | Baseline |
| Head-only training | Validation | 85.9% | 85.87% | +47.4 accuracy points |
| Partial fine-tuning | Validation | 93.1% | 93.10% | +54.6 accuracy points |
| **After fine-tuning: full model** | Validation | **94.1%** | **94.10%** | **+55.6 accuracy points** |
| Selected full model | Final held-out test | **92.4%** | **92.40%** | Final generalization result |

The scientifically matched before/after validation comparison is the untrained random classification head versus the fully fine-tuned model:

- Accuracy improved from **38.5% to 94.1%**: **+55.6 percentage points**.
- Macro-F1 improved from **38.29% to 94.10%**: **+55.81 percentage points**.
- The validation-selected model achieved **92.4% accuracy and 92.40% Macro-F1** on the untouched final test.

The random-head result is measured rather than assumed. Its below-chance score is plausible because the newly initialized head is arbitrary before task training.

### Task A evidence

| Evidence | Location |
|---|---|
| Full task documentation and commands | [Task_A/README.md](Task_A/README.md) |
| Validation model-selection record | [Task_A/results/model_selection.json](Task_A/results/model_selection.json) |
| Final-test metrics | [Task_A/results/final_test/metrics.json](Task_A/results/final_test/metrics.json) |
| Final-test confusion matrix | [Task_A/results/final_test/confusion_matrix.png](Task_A/results/final_test/confusion_matrix.png) |
| All 76 error rows | [Task_A/results/final_test/errors.csv](Task_A/results/final_test/errors.csv) |
| Error summary and provenance | [Task_A/results/final_test/error_analysis.json](Task_A/results/final_test/error_analysis.json) |

## Repository layout

```text
scogo-ai-assignment/
├── README.md
├── Task_A/
│   ├── README.md
│   ├── app/
│   ├── scripts/
│   ├── tests/
│   └── results/
└── Task_B/                  # added when Task B implementation begins
```

Model weights, dataset caches, virtual environments, and training checkpoints are intentionally excluded from Git. Follow the task-specific README to reproduce them.

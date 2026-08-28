# Task A improvement comparison

These results are isolated from and do not overwrite the original validation, held-out test, or challenge-suite artifacts. No model weights were retrained.

## Preserved results

| Result | Value |
|---|---:|
| Original held-out test accuracy | 92.4% |
| Original held-out test Macro-F1 | 92.40% |
| Original challenge suite | 15/28 |

## Long reviews

| Diagnostic | Before | Sliding windows |
|---|---:|---:|
| Correct | 0/4 | 0/4 |
| Truncated reviews | 4 | 0 |
| Chunks per review | 1 | 2 |

Conclusion: truncation was eliminated, but classification was not materially improved on these four mixed long reviews. The limitation remains unresolved.

## Calibration

Validation-fitted temperature: `1.886450442498428`.

| Metric | Before | After |
|---|---:|---:|
| ECE | 0.0487837 | 0.0207564 |
| Brier score | 0.1060154 | 0.0980188 |
| NLL | 0.2571336 | 0.1856683 |
| Accuracy | 0.941 | 0.941 |

On challenge diagnostics, wrong predictions above 95% confidence fell from 8 to 2. Calibration materially improved confidence reliability without changing model weights or validation class labels.

## Mixed-sentiment guardrail

| Diagnostic | Result |
|---|---:|
| Original binary prediction correct | 2/4 |
| Mixed evidence detected | 4/4 |

The guardrail uses heuristic clause/aspect extraction and the existing binary classifier. It exposes opposing calibrated predictions and their evidence; it is not a trained aspect-sentiment model and does not change the overall label.

## Deferred work

Sarcasm (baseline 1/4) and negation (baseline 2/4) remain candidates for isolated adaptation research. They were not silently added to the production model because any fine-tuned candidate must also prove that ordinary Amazon validation performance does not materially regress.

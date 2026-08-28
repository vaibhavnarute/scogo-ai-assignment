# FineTuneFeedback

FineTuneFeedback classifies Amazon product reviews as **negative** or **positive**. It compares three independent DistilBERT fine-tuning strategies, chooses the winner using validation Macro-F1, and evaluates that winner once on an untouched canonical test subset.

Every number reported below comes from the generated artifacts under `results/`; no result was fabricated or assumed.

## Design

All experiments start independently from `distilbert-base-uncased`:

| Experiment | Trainable parameters |
|---|---|
| Head only | Pre-classifier and classifier |
| Partial | Final two transformer blocks plus classification head |
| Full | Entire model |

The sequence-classification path is: tokenizer → embeddings → six DistilBERT transformer blocks → first-token hidden representation → pre-classifier → ReLU → dropout → linear classifier.

The application uses one `Predictor` implementation for both interfaces:

```text
saved validation-selected model
             |
         Predictor
         /       \
    FastAPI     Gradio
```

## Data protocol

- Source: `mteb/amazon_polarity`, using its `text` and `label` fields.
- Canonical train produces 5,000 training and 1,000 validation reviews.
- Canonical test produces the final 1,000-review test subset.
- Every subset is exactly balanced and created deterministically with seed 42.
- Training never loads canonical test. `scripts/evaluate.py` is the only final-test entry point.
- Labels are `0 = negative` and `1 = positive`.

The source is streamed with a deterministic shuffle buffer, so creating the small subsets does not materialize the complete four-million-row dataset. Created subsets are cached under `.cache/dataset/`.

## Measured results

Model selection used only the fixed 1,000-review validation subset. The full model won on validation Macro-F1.

| Validation stage | Accuracy | Macro-precision | Macro-recall | Macro-F1 | Loss |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.500 | 0.250 | 0.500 | 0.333 | — |
| Seeded random predictor | 0.513 | 0.513 | 0.513 | 0.513 | — |
| Random classification head | 0.385 | 0.383 | 0.385 | 0.383 | 0.699 |
| Head-only training | 0.859 | 0.862 | 0.859 | 0.859 | 0.479 |
| Partial fine-tuning | 0.931 | 0.931 | 0.931 | 0.931 | 0.235 |
| **Full fine-tuning** | **0.941** | **0.941** | **0.941** | **0.941** | 0.257 |

After model selection was fixed, the full model was evaluated on the untouched, balanced 1,000-review canonical-test subset exactly once:

| Final test | Accuracy | Macro-precision | Macro-recall | Macro-F1 | Loss |
|---|---:|---:|---:|---:|---:|
| Full fine-tuning | **0.924** | **0.924** | **0.924** | **0.924** | 0.303 |

The final confusion matrix is `[[464, 36], [40, 460]]` for labels `[negative, positive]`: 36 negative reviews were predicted positive and 40 positive reviews were predicted negative. The approximately 1.7-point validation-to-test Macro-F1 decrease is modest and expected on unseen data.

Full fine-tuning gained about one Macro-F1 point over partial fine-tuning, but trained 66.96M rather than 14.77M parameters and took substantially longer. Partial fine-tuning is therefore a credible efficiency option even though full fine-tuning won the predefined selection metric.

## Setup

Python 3.12 is recommended. Create and activate a virtual environment, then install the pinned dependencies:

```powershell
pip install -r requirements.txt
```

Confirm that installation and execution use the same interpreter:

```powershell
python -c "import sys; print(sys.executable)"
python -m pip check
```

Optional settings can be copied from `.env.example` into the environment. The project reads environment variables directly; it does not automatically load `.env` files.

## Train and select

Run every independent experiment:

```powershell
python scripts/train.py
```

Or run one experiment while developing:

```powershell
python scripts/train.py --experiment head_only
python scripts/train.py --experiment partial
python scripts/train.py --experiment full
```

The command measures majority, seeded-random, and random-head baselines on validation data. It writes experiment artifacts under `results/`, checkpoints under `models/`, and promotes the best available validation Macro-F1 checkpoint to `models/best/`.

Each experiment is initialized from the original pretrained checkpoint—not from another experiment. Configuration and parameter counts are saved with its measured metrics.

## Final test

After all candidate experiments are complete and the winning model is fixed:

```powershell
python scripts/evaluate.py
```

This creates the canonical-test subset, evaluates only `models/best/`, and writes measured metrics and a confusion matrix to `results/final_test/`. Re-running the command repeats final-test evaluation, so treat this as a deliberate final reporting step rather than a tuning loop.

The final evaluation has already been completed for the reported run. Do not rerun it for model selection or hyperparameter tuning.

## Error analysis

The following command reproduces predictions for reporting only; it performs no training and must not be used to tune against final-test outcomes:

```powershell
python scripts/analyze_errors.py
```

All 76 mistakes are recorded in `results/final_test/errors.csv`, with a summary and the ten highest-confidence mistakes in `results/final_test/error_analysis.json`.

### Artifact traceability

The result files are connected by the following reproducible chain:

```text
results/full/config.json
        +
results/full/metrics.json
        |
        v
results/model_selection.json  (full selected by validation Macro-F1)
        |
        v
models/best/                  (ignored weights used for inference)
        +
cached canonical test subset (mteb/amazon_polarity, seed 44)
        |
        v
results/final_test/metrics.json
results/final_test/confusion_matrix.png
        |
        v
scripts/analyze_errors.py
        |
        +-- results/final_test/errors.csv
        `-- results/final_test/error_analysis.json
```

Both error artifacts use run ID `amazon-polarity-seed-42-full-final-test`. Every CSV row repeats the dataset, canonical split, subset seed, and selected experiment, so the file remains understandable when opened alone. The JSON contains a fuller provenance block, a snapshot of final metrics, source-artifact paths, label mapping, model-selection record, and generating script. `test_index` maps each error back to its row in the cached deterministic final-test subset.

| Observation | Measured count | Interpretation |
|---|---:|---|
| Negative predicted positive | 36 | False-positive sentiment errors |
| Positive predicted negative | 40 | False-negative sentiment errors |
| Mixed/contrast-language cue | 52 | Reviews often praise one aspect while criticizing another |
| Negation-language cue | 16 | Scope and reversal language remain difficult |
| Other/domain-specific wording | 8 | Includes neutral, informational, sarcastic, and unusual reviews |
| More than 256 tokens | 0 | Truncation did not cause any error in this fixed subset |
| Wrong with confidence ≥ 0.90 | 57 | Softmax confidence is not calibrated correctness |

The categories above are transparent keyword-based organization cues, not ground-truth causal diagnoses. Manual inspection found several recurring patterns:

- **Aspect conflict:** the product itself is praised while shipping, color, price, packaging, or the seller is criticized. For example, a review labeled positive says the item was good but strongly criticizes receiving the wrong color.
- **Dominant wording versus star-derived label:** some reviews labeled negative contain almost entirely positive language (for example, “This was a great book … Can not wait to read the next one”). The model follows the written sentiment, while Amazon Polarity follows the source rating.
- **Negation and rhetorical framing:** phrases such as “not bad,” “never used to,” and criticism of other reviewers can contain negative tokens inside an ultimately positive review.
- **Mixed artistic reviews:** books, films, and music are often praised overall while discussing depressing plots, flaws, or comparisons. A single binary label loses this nuance.
- **High-confidence mistakes:** many contradictions are confidently wrong, reinforcing that softmax output should be presented as model confidence rather than a calibrated probability.

This is post-hoc reporting on the final test set. None of these observations affected model selection, training configuration, or reported test metrics.

## Targeted challenge suite

The project also includes 28 synthetic, difficult examples that are completely separate from Amazon Polarity training, validation, and final-test data. The suite contains four examples in each category:

- Sarcasm
- Mixed sentiment
- Product versus delivery experience
- Support complaints
- Negation
- Very long reviews
- Domain shift

Because the deployed model supports only two labels, every example has a human-authored `positive` or `negative` expected label representing its dominant overall judgment. This is necessarily subjective for mixed reviews. The suite is diagnostic only and was not used for training, model selection, hyperparameter tuning, or the reported final-test score.

Run it with the already selected model:

```powershell
python scripts/evaluate_challenges.py
```

Measured diagnostic results:

| Category | Correct | Diagnostic accuracy | Mean confidence |
|---|---:|---:|---:|
| Domain shift | 4/4 | 100% | 97.58% |
| Mixed sentiment | 2/4 | 50% | 89.84% |
| Negation | 2/4 | 50% | 99.48% |
| Product versus delivery | 3/4 | 75% | 91.66% |
| Sarcasm | 1/4 | 25% | 99.41% |
| Support complaints | 3/4 | 75% | 98.80% |
| Very long reviews | 0/4 | 0% | 92.23% |
| **Overall** | **15/28** | **53.57%** | — |

The challenge confusion matrix is `[[9, 5], [8, 6]]` for expected labels `[negative, positive]`. The model predicted negative 17 times and positive 11 times.

Key observations:

- **Sarcasm remains a clear failure mode.** Three of four sarcasm examples failed with very high confidence because surface-positive or surface-negative wording contradicted the intended meaning.
- **Negation remains brittle.** Double negation and phrases such as “never failed” were confidently interpreted as negative.
- **Mixed/aspect sentiment is lossy under binary labels.** The model often followed the complaint even when the human-authored overall judgment was positive.
- **Delivery and support separation is imperfect but visible.** The model handled three of four examples in each category, while still confusing fast delivery with a positive defective-product judgment in one case.
- **All four long reviews were intentionally longer than 256 tokens and all failed.** Their decisive conclusion occurred late, demonstrating a direct truncation limitation rather than merely speculating about one.
- **Four domain-shift examples happened to pass, but this tiny synthetic sample cannot establish cross-domain robustness.** Broader labeled domain holdouts would be required for that claim.
- **High confidence did not imply correctness.** Several failed categories had mean confidence above 99%, strengthening the case for confidence calibration.

Artifacts are self-describing and committed:

- Challenge definitions and rationales: `data/challenge_examples.json`
- Every prediction and failure: `results/challenge_suite/predictions.csv`
- Per-category summary and provenance: `results/challenge_suite/summary.json`

These diagnostic scores must not be compared directly with the held-out Amazon Polarity test score: the challenge examples are deliberately adversarial, synthetic, small, and subjectively labeled.

## Inference improvements: sliding windows, calibration, and mixed-sentiment diagnostics

Two production-oriented improvements were added without changing or retraining the selected model weights.

### Sliding-window inference

Inference now tokenizes the complete review before deciding whether to chunk it. Reviews of 256 tokens or fewer use the original single-pass path. Longer reviews use deterministic overlapping windows:

- Window size: 256 tokens, including special tokens
- Stride: 64 overlap tokens
- Aggregation: content-token-count-weighted mean of chunk probabilities
- Label compatibility: the final label is selected from aggregated raw probabilities
- Metadata: original token count, chunks used, chunking flag, window size, and stride

Token-count weighting was chosen instead of confidence weighting because model confidence is known to be uncalibrated and should not automatically give a chunk more voting power.

Measured only on the four existing long-review challenge examples:

| Long-review diagnostic | Before | After sliding windows |
|---|---:|---:|
| Correct | 0/4 | 0/4 |
| Reviews truncated | 4/4 | 0/4 |
| Chunks used | 1 each | 2 each |

Sliding windows **mitigated truncation but did not materially improve classification** on these four deliberately mixed long reviews. The decisive late text was processed, but weighted aggregation still favored the dominant evidence across the entire review. This limitation remains unresolved rather than being presented as a successful accuracy improvement.

Artifacts:

- `results/improvements/long_review/before.json`
- `results/improvements/long_review/after.json`
- `results/improvements/long_review/predictions.csv`

### Validation-only temperature scaling

Run calibration after a validation-selected model exists:

```powershell
python scripts/calibrate.py
```

The script collects logits from the existing 1,000-review validation subset, fits one positive scalar temperature by minimizing validation negative log-likelihood, and changes no model weight. The fitted temperature is `1.886450`.

| Validation calibration metric | Raw | Calibrated | Change |
|---|---:|---:|---:|
| ECE | 0.04878 | **0.02076** | -57.5% |
| Brier score | 0.10602 | **0.09802** | -7.5% |
| NLL | 0.25713 | **0.18567** | -27.8% |
| Accuracy | 94.1% | 94.1% | unchanged |
| Mean confidence | 98.80% | 94.75% | closer to observed accuracy |

All validation class labels remained unchanged. Calibration affects confidence interpretation, not classification accuracy.

On the 28 diagnostic challenge examples, using the same validation-fitted temperature:

| Confidence diagnostic | Raw | Calibrated |
|---|---:|---:|
| Mean confidence on correct predictions | 95.07% | 89.18% |
| Mean confidence on incorrect predictions | 89.83% | 82.19% |
| Wrong predictions above 90% confidence | 8 | 7 |
| Wrong predictions above 95% confidence | 8 | **2** |

Artifacts:

- `results/calibration/temperature.json`
- `results/calibration/metrics_before.json`
- `results/calibration/metrics_after.json`
- `results/calibration/reliability_diagram.png`
- `results/improvements/calibration/challenge_confidence.json`
- `results/improvements/calibration/challenge_predictions.csv`
- `results/improvements/comparison_report.json`

Generate isolated improvement comparisons with:

```powershell
python scripts/evaluate_improvements.py
```

The original validation, final-test, and challenge-suite artifacts remain unchanged. Sarcasm and negation adaptation are intentionally not promoted into the production model; they require isolated fine-tuning experiments and regression checks against normal Amazon validation performance.

### Mixed-sentiment diagnostic guardrail

The API now splits a review into at most 12 clauses, assigns transparent keyword-based aspect labels (`product`, `delivery`, `support`, `price_value`, `reliability`, or `usability`), and runs the same existing binary model on each clause. It sets `mixed_sentiment_detected` only when calibrated positive and negative clause predictions both meet the configurable 0.70 confidence threshold.

On the four mixed-sentiment challenge examples, the original overall binary result remains **2/4**, while the guardrail identified mixed evidence in **4/4**. This is a diagnostic improvement, not a trained aspect-sentiment model and not an accuracy improvement. Evidence text is returned with each aspect prediction so a viewer can understand why the guardrail fired.

Artifacts:

- `results/improvements/mixed_sentiment/summary.json`
- `results/improvements/mixed_sentiment/predictions.csv`

Configuration is available through `ASPECT_CONFIDENCE_THRESHOLD` and `ASPECT_MAX_SEGMENTS`.

## API

Start the production-style inference service:

```powershell
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs>. Example request:

```json
{"text": "The build quality is terrible and it stopped working."}
```

Example response shape (confidence values shown here are illustrative):

```json
{
  "sentiment": "negative",
  "confidence": 0.9721,
  "calibrated_confidence": 0.9112,
  "original_token_count": 14,
  "chunks_used": 1,
  "was_chunked": false,
  "window_size": 256,
  "stride": 64,
  "calibration_applied": true,
  "aspects": [
    {
      "aspect": "reliability",
      "sentiment": "negative",
      "confidence": 0.86,
      "evidence": "The build quality is terrible and it stopped working."
    }
  ],
  "mixed_sentiment_detected": false
}
```

`confidence` preserves the original raw softmax value for backwards compatibility; `calibrated_confidence` applies the validation-fitted temperature. Whitespace-only input returns HTTP 422. `/health` remains available before training and reports whether the selected model artifact exists. The model itself loads lazily on the first prediction.

## Gradio demo

```powershell
python demo.py
```

The UI calls the same predictor as FastAPI and does not duplicate tokenization or inference logic.

## Tests

```powershell
pytest -q
```

API tests replace the heavyweight predictor with a deterministic fake, so unit tests do not download or load DistilBERT. Training, dataset streaming, and real-model inference are integration activities invoked explicitly through the commands above.

For the completed improvement run, all 20 automated tests passed. A separate real-model smoke check also processed both a short review and a 1,522-token review; the long review used eight chunks on CPU with calibration enabled.

## Artifacts and reproducibility

The configured seeds cover Python, NumPy, PyTorch, CUDA (when present), Hugging Face Trainer, and data ordering. Runtime device selection is automatic: CUDA when available, otherwise CPU.

### Hardware used for the measured run

- OS: Windows 11
- CPU: 13th Gen Intel Core i5-13450HX
- GPU used: none
- PyTorch: 2.6.0+cpu (`torch.cuda.is_available() == False`)
- PyTorch CPU threads: 10
- Python: 3.13.2
- Head-only training runtime: 514.6 seconds (8m 35s)
- Partial fine-tuning runtime: 848.2 seconds (14m 08s)
- Full fine-tuning runtime: 1,578.5 seconds (26m 18s)

The machine has an NVIDIA GeForce RTX 3050 with 6 GB VRAM, but this measured run used the installed CPU-only PyTorch build. The results remain valid; CUDA would primarily reduce runtime.

```text
models/                     results/
├── head_only/              ├── baselines/
├── partial/                ├── random_head/
├── full/                   ├── head_only/
└── best/                   ├── partial/
                            ├── full/
                            └── final_test/
```

Model weights, runtime checkpoints, and cached data are ignored by Git. Lightweight measured evidence—metrics, error CSV/JSON files, and confusion-matrix images—is committed. Recreate heavyweight artifacts using the documented commands.

## Limitations

- Amazon Polarity excludes neutral three-star reviews, so the model cannot represent neutral or mixed sentiment.
- Sarcasm, domain shift, and reviews mixing product, delivery, and support experiences remain difficult.
- Softmax confidence is not guaranteed to be calibrated.
- Sentiment is not equivalent to business urgency.
- The final 1,000-row test subset is suitable for this assignment, not a substitute for broader production evaluation.

The targeted challenge suite provides concrete evidence for the sarcasm, negation, aspect-mixing, support/delivery, truncation, and domain-shift limitations. It documents the current behavior without claiming that these limitations are fully solved.

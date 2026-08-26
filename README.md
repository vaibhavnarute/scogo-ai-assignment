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

## API

Start the production-style inference service:

```powershell
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs>. Example request:

```json
{"text": "The build quality is terrible and it stopped working."}
```

Example response shape (the confidence shown here is illustrative, not a claimed model result):

```json
{"sentiment": "negative", "confidence": 0.9721}
```

Whitespace-only input returns HTTP 422. `/health` remains available before training and reports whether the selected model artifact exists. The model itself loads lazily on the first prediction.

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

For the completed run, all six automated tests passed. Separate real-model smoke checks also passed for FastAPI (`/health`, `/predict`) and the shared Gradio prediction function.

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

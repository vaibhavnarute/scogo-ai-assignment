"""Thin Gradio UI backed by the same Predictor used by FastAPI."""

from __future__ import annotations

from functools import lru_cache

import gradio as gr

from app.ml.inference.predictor import Predictor


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()


def classify_review(text: str) -> tuple[str, float, float, int, bool, list[dict[str, object]]]:
    result = get_predictor().predict_with_diagnostics(text)
    return (
        result["sentiment"],
        result["confidence"],
        result["calibrated_confidence"],
        result["chunks_used"],
        result["mixed_sentiment_detected"],
        result["aspects"],
    )


demo = gr.Interface(
    fn=classify_review,
    inputs=gr.Textbox(lines=6, label="Customer review", placeholder="Enter a review..."),
    outputs=[
        gr.Label(label="Sentiment"),
        gr.Number(label="Raw confidence"),
        gr.Number(label="Calibrated confidence"),
        gr.Number(label="Chunks used"),
        gr.Checkbox(label="Mixed sentiment detected"),
        gr.JSON(label="Aspect diagnostics"),
    ],
    title="FineTuneFeedback",
    description="Classify an unseen product review with the validation-selected DistilBERT model.",
    examples=[
        ["Excellent build quality and it works exactly as promised."],
        ["It stopped working after one day and support was unhelpful."],
    ],
)


if __name__ == "__main__":
    demo.launch()

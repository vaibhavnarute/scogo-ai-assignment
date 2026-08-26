"""Thin Gradio UI backed by the same Predictor used by FastAPI."""

from __future__ import annotations

from functools import lru_cache

import gradio as gr

from app.ml.inference.predictor import Predictor


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()


def classify_review(text: str) -> tuple[str, float]:
    result = get_predictor().predict(text)
    return result["sentiment"], result["confidence"]


demo = gr.Interface(
    fn=classify_review,
    inputs=gr.Textbox(lines=6, label="Customer review", placeholder="Enter a review..."),
    outputs=[gr.Label(label="Sentiment"), gr.Number(label="Confidence")],
    title="FineTuneFeedback",
    description="Classify an unseen product review with the validation-selected DistilBERT model.",
    examples=[
        ["Excellent build quality and it works exactly as promised."],
        ["It stopped working after one day and support was unhelpful."],
    ],
)


if __name__ == "__main__":
    demo.launch()

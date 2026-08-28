from app.ml.inference.aspects import analyze_aspects, extract_aspect_segments


def test_aspect_segments_are_deterministic_and_keyword_labeled() -> None:
    text = "The product is excellent, but delivery arrived late and the box was crushed."
    first = extract_aspect_segments(text)
    assert first == extract_aspect_segments(text)
    assert [segment.aspect for segment in first] == ["product", "delivery"]


def test_opposing_confident_clauses_trigger_mixed_guardrail() -> None:
    def predict(text: str) -> dict[str, object]:
        if "excellent" in text:
            return {"sentiment": "positive", "confidence": 0.95, "calibrated_confidence": 0.90}
        return {"sentiment": "negative", "confidence": 0.96, "calibrated_confidence": 0.92}

    result = analyze_aspects(
        "The product is excellent, but delivery arrived late.", predict
    )
    assert result["mixed_sentiment_detected"] is True
    assert len(result["aspects"]) == 2


def test_low_confidence_disagreement_does_not_trigger_guardrail() -> None:
    def predict(text: str) -> dict[str, object]:
        sentiment = "positive" if "good" in text else "negative"
        return {"sentiment": sentiment, "confidence": 0.6, "calibrated_confidence": 0.6}

    result = analyze_aspects("Good product, but slow delivery.", predict)
    assert result["mixed_sentiment_detected"] is False

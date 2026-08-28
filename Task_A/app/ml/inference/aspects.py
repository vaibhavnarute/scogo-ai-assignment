"""Lightweight clause/aspect diagnostics built on the existing binary model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Mapping


@dataclass(frozen=True)
class AspectSegment:
    aspect: str
    text: str


_CLAUSE_BOUNDARY = re.compile(
    r"(?<=[.!?;])\s+|\s+(?:but|yet|however|although|though|while)\s+",
    flags=re.IGNORECASE,
)
_ASPECT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("delivery", ("deliver", "shipping", "shipment", "arrived", "courier", "package", "box")),
    ("support", ("support", "service", "agent", "representative", "refund", "return")),
    ("price_value", ("price", "cost", "value", "expensive", "cheap", "worth", "payment")),
    ("reliability", ("reliable", "unreliable", "broke", "broken", "fail", "crash", "disconnect", "battery", "stopped")),
    ("usability", ("easy", "difficult", "setup", "install", "interface", "controls", "comfortable", "usable", "use")),
)


def extract_aspect_segments(text: str, max_segments: int = 12) -> tuple[AspectSegment, ...]:
    """Split review clauses and assign transparent keyword-based aspect labels."""
    if max_segments < 1:
        raise ValueError("max_segments must be positive")
    clauses = [part.strip(" ,") for part in _CLAUSE_BOUNDARY.split(text.strip())]
    clauses = [clause for clause in clauses if clause]
    segments: list[AspectSegment] = []
    for clause in clauses[:max_segments]:
        lowered = clause.lower()
        aspect = "product"
        for candidate, keywords in _ASPECT_KEYWORDS:
            if any(keyword in lowered for keyword in keywords):
                aspect = candidate
                break
        segments.append(AspectSegment(aspect=aspect, text=clause))
    return tuple(segments)


def analyze_aspects(
    text: str,
    predict: Callable[[str], Mapping[str, object]],
    *,
    confidence_threshold: float = 0.70,
    max_segments: int = 12,
) -> dict[str, object]:
    """Classify aspect clauses and flag opposing high-confidence sentiments."""
    aspects: list[dict[str, object]] = []
    high_confidence_labels: set[str] = set()
    for segment in extract_aspect_segments(text, max_segments=max_segments):
        result = predict(segment.text)
        confidence = float(result.get("calibrated_confidence", result["confidence"]))
        sentiment = str(result["sentiment"])
        aspects.append(
            {
                "aspect": segment.aspect,
                "sentiment": sentiment,
                "confidence": confidence,
                "evidence": segment.text,
            }
        )
        if confidence >= confidence_threshold:
            high_confidence_labels.add(sentiment)
    return {
        "aspects": aspects,
        "mixed_sentiment_detected": {"negative", "positive"}.issubset(
            high_confidence_labels
        ),
    }

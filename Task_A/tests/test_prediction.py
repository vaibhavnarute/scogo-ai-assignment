from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.routes.prediction import get_predictor
from app.api.schemas.prediction import PredictionResponse
from app.main import app
from app.ml.data.dataset import sample_balanced_records


class FakePredictor:
    def predict(self, text: str) -> dict[str, object]:
        common = {
            "calibrated_confidence": 0.85,
            "original_token_count": 8,
            "chunks_used": 1,
            "was_chunked": False,
            "window_size": 256,
            "stride": 64,
            "calibration_applied": True,
        }
        if "excellent" in text.lower():
            return {"sentiment": "positive", "confidence": 0.91, **common}
        return {"sentiment": "negative", "confidence": 0.88, **common}

    def predict_with_diagnostics(self, text: str) -> dict[str, object]:
        result = self.predict(text)
        result.update(
            {
                "aspects": [
                    {
                        "aspect": "product",
                        "sentiment": result["sentiment"],
                        "confidence": result["calibrated_confidence"],
                        "evidence": text,
                    }
                ],
                "mixed_sentiment_detected": False,
            }
        )
        return result


def client() -> TestClient:
    app.dependency_overrides[get_predictor] = lambda: FakePredictor()
    return TestClient(app)


def test_health_endpoint() -> None:
    response = client().get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_positive_prediction() -> None:
    response = client().post("/predict", json={"text": "Excellent product."})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert response.json()["confidence"] == 0.91
    assert response.json()["calibrated_confidence"] == 0.85
    assert response.json()["was_chunked"] is False
    assert response.json()["aspects"][0]["aspect"] == "product"
    assert response.json()["mixed_sentiment_detected"] is False


def test_negative_prediction() -> None:
    response = client().post("/predict", json={"text": "It broke immediately."})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "negative"
    assert response.json()["confidence"] == 0.88


def test_empty_review_validation() -> None:
    response = client().post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_prediction_schema() -> None:
    value = PredictionResponse(
        sentiment="positive",
        confidence=0.75,
        calibrated_confidence=0.7,
        original_token_count=20,
        chunks_used=1,
        was_chunked=False,
        window_size=256,
        stride=64,
        calibration_applied=True,
        aspects=[
            {
                "aspect": "product",
                "sentiment": "positive",
                "confidence": 0.7,
                "evidence": "Works well.",
            }
        ],
        mixed_sentiment_detected=False,
    )
    assert value.sentiment == "positive"
    try:
        PredictionResponse(
            sentiment="neutral",
            confidence=1.2,
            calibrated_confidence=0.7,
            original_token_count=20,
            chunks_used=1,
            was_chunked=False,
            window_size=256,
            stride=64,
            calibration_applied=True,
            aspects=[],
            mixed_sentiment_detected=False,
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("Invalid response should not satisfy the schema")


def test_dataset_has_balanced_labels() -> None:
    records = [
        {"text": f"negative {index}", "label": 0} for index in range(10)
    ] + [{"text": f"positive {index}", "label": 1} for index in range(10)]
    sampled = sample_balanced_records(records, size=8, seed=42)
    assert [row["label"] for row in sampled].count(0) == 4
    assert [row["label"] for row in sampled].count(1) == 4
    assert sampled == sample_balanced_records(records, size=8, seed=42)

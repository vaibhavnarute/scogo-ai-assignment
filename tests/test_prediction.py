from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.routes.prediction import get_predictor
from app.api.schemas.prediction import PredictionResponse
from app.main import app
from app.ml.data.dataset import sample_balanced_records


class FakePredictor:
    def predict(self, text: str) -> dict[str, object]:
        if "excellent" in text.lower():
            return {"sentiment": "positive", "confidence": 0.91}
        return {"sentiment": "negative", "confidence": 0.88}


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
    assert response.json() == {"sentiment": "positive", "confidence": 0.91}


def test_negative_prediction() -> None:
    response = client().post("/predict", json={"text": "It broke immediately."})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "negative", "confidence": 0.88}


def test_empty_review_validation() -> None:
    response = client().post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_prediction_schema() -> None:
    value = PredictionResponse(sentiment="positive", confidence=0.75)
    assert value.sentiment == "positive"
    try:
        PredictionResponse(sentiment="neutral", confidence=1.2)
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

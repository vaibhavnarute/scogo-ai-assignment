"""Prediction endpoint and dependency wiring."""

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas.prediction import PredictionRequest, PredictionResponse
from app.ml.inference.predictor import Predictor

router = APIRouter(tags=["sentiment"])


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    try:
        return Predictor()
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


@router.post("/predict", response_model=PredictionResponse)
def predict_sentiment(
    request: PredictionRequest,
    predictor: Predictor = Depends(get_predictor),
) -> PredictionResponse:
    try:
        result = predictor.predict(request.text)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return PredictionResponse(**result)

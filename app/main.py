"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes.prediction import router as prediction_router
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(
    title="FineTuneFeedback",
    version="1.0.0",
    description="DistilBERT customer-review sentiment classification service.",
)
app.include_router(prediction_router)


@app.get("/health", tags=["operations"])
def health() -> dict[str, object]:
    model_ready = (settings.models_dir / "best" / "config.json").exists()
    return {"status": "ok", "model_ready": model_ready}

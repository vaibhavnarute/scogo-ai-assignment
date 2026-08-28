"""HTTP request and response contracts."""

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)

    @field_validator("text")
    @classmethod
    def text_must_contain_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Review text must not be empty")
        return value


class AspectPrediction(BaseModel):
    aspect: str = Field(
        pattern="^(product|delivery|support|price_value|reliability|usability)$"
    )
    sentiment: str = Field(pattern="^(negative|positive)$")
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    sentiment: str = Field(pattern="^(negative|positive)$")
    confidence: float = Field(ge=0.0, le=1.0)
    calibrated_confidence: float = Field(ge=0.0, le=1.0)
    original_token_count: int = Field(ge=1)
    chunks_used: int = Field(ge=1)
    was_chunked: bool
    window_size: int = Field(ge=3)
    stride: int = Field(ge=0)
    calibration_applied: bool
    aspects: list[AspectPrediction] = Field(default_factory=list)
    mixed_sentiment_detected: bool = False

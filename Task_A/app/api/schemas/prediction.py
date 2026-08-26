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


class PredictionResponse(BaseModel):
    sentiment: str = Field(pattern="^(negative|positive)$")
    confidence: float = Field(ge=0.0, le=1.0)

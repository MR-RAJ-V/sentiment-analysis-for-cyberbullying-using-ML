from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def must_contain_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must contain at least one non-whitespace character.")
        return value


class SentimentScores(BaseModel):
    negative: float
    neutral: float
    positive: float
    compound: float


class PredictionResponse(BaseModel):
    category: str
    sentiment: str
    sentiment_scores: SentimentScores
    confidence: float | None = None
    confidence_note: str | None = None

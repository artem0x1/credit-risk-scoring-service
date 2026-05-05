from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    app_name: str
    app_version: str


class CreditScoreRequest(BaseModel):
    client_id: int | str | None = Field(
        default=None,
        description="Optional client identifier.",
    )
    features: dict[str, Any] = Field(
        ...,
        description="Dictionary with model feature values.",
    )


class CreditScoreResponse(BaseModel):
    client_id: int | str | None = None
    model: str
    probability_of_default: float
    threshold: float
    risk_grade: str
    decision: str
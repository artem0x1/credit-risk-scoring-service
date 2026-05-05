from fastapi import FastAPI, HTTPException

from api.schemas import (
    HealthResponse,
    CreditScoreRequest,
    CreditScoreResponse,
)
from api.service import credit_scoring_service
from api.settings import settings


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Health check endpoint.
    """
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        app_version=settings.app_version,
    )


@app.post("/predict", response_model=CreditScoreResponse)
def predict_credit_score(
    request: CreditScoreRequest,
) -> CreditScoreResponse:
    """
    Predict credit risk for one client.
    """
    try:
        prediction = credit_scoring_service.predict(
            features=request.features,
            client_id=request.client_id,
        )
        return CreditScoreResponse(**prediction)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc
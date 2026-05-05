from api.settings import settings
from src.inference import CreditRiskPredictor


class CreditScoringService:
    """
    Service layer for credit risk scoring.
    """

    def __init__(self):
        self.predictor = CreditRiskPredictor(
            model_path=settings.model_path,
            feature_list_path=settings.feature_list_path,
            final_model_config_path=settings.final_model_config_path,
        )

    def predict(
        self,
        features: dict,
        client_id: int | str | None = None,
    ) -> dict:
        """
        Predict credit risk for one client.
        """
        return self.predictor.predict_one(
            client_features=features,
            client_id=client_id,
        )


credit_scoring_service = CreditScoringService()
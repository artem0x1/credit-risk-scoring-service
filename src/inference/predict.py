import json
import pickle
from pathlib import Path

import pandas as pd

from src.inference.preprocessor import (
    prepare_features_for_lightgbm,
    prepare_single_client_features,
)
from src.inference.risk_policy import apply_risk_policy


def load_json(path: str | Path) -> dict | list:
    """
    Load JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: str | Path):
    """
    Load pickle file.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


class CreditRiskPredictor:
    """
    Production-like predictor wrapper for credit risk inference.
    """

    def __init__(
        self,
        model_path: str | Path,
        feature_list_path: str | Path,
        final_model_config_path: str | Path,
        lightgbm_params_path: str | Path | None = None,
    ):
        self.model_path = Path(model_path)
        self.feature_list_path = Path(feature_list_path)
        self.final_model_config_path = Path(final_model_config_path)

        self.model = load_pickle(self.model_path)
        self.feature_cols = load_json(self.feature_list_path)
        self.final_model_config = load_json(self.final_model_config_path)

        self.threshold = float(self.final_model_config["threshold"])
        self.model_name = self.final_model_config.get(
            "selected_model",
            "LightGBM",
        )

        if lightgbm_params_path is None:
            lightgbm_params_path = self.model_path.parent / "lightgbm_params.json"

        self.lightgbm_params_path = Path(lightgbm_params_path)

        if self.lightgbm_params_path.exists():
            self.lightgbm_params = load_json(self.lightgbm_params_path)
            self.categorical_features = self.lightgbm_params.get(
                "categorical_features",
                [],
            )
        else:
            self.lightgbm_params = {}
            self.categorical_features = []

    def _align_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Align pandas categorical columns with categories stored in LightGBM model.
        """
        X = X.copy()

        pandas_categorical = getattr(self.model, "pandas_categorical", None)

        if pandas_categorical is None:
            return X

        categorical_cols = [
            col for col in self.categorical_features
            if col in X.columns
        ]

        if len(categorical_cols) != len(pandas_categorical):
            return X

        for col, categories in zip(categorical_cols, pandas_categorical):
            X[col] = X[col].cat.set_categories(categories)

        return X

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict probability of default for prepared dataframe.
        """
        X = self._align_categorical_features(X)

        predictions = self.model.predict(
            X,
            num_iteration=self.model.best_iteration,
            validate_features=False,
        )

        return pd.Series(
            predictions,
            index=X.index,
            name="probability_of_default",
        )

    def predict_dataframe(
        self,
        data: pd.DataFrame,
        id_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Predict credit risk for dataframe.
        """
        X = prepare_features_for_lightgbm(
            data=data,
            feature_cols=self.feature_cols,
            categorical_features=self.categorical_features,
        )

        pd_values = self.predict_proba(X)

        result = pd.DataFrame({
            "probability_of_default": pd_values.values,
        })

        if id_col is not None and id_col in data.columns:
            result.insert(0, "client_id", data[id_col].values)

        result["threshold"] = self.threshold
        result["risk_grade"] = result["probability_of_default"].apply(
            lambda x: apply_risk_policy(x, self.threshold)["risk_grade"]
        )
        result["decision"] = result["probability_of_default"].apply(
            lambda x: apply_risk_policy(x, self.threshold)["decision"]
        )

        return result

    def predict_one(
        self,
        client_features: dict,
        client_id: int | str | None = None,
    ) -> dict:
        """
        Predict credit risk for one client.
        """
        X = prepare_single_client_features(
            client_features=client_features,
            feature_cols=self.feature_cols,
            categorical_features=self.categorical_features,
        )

        pd_value = float(self.predict_proba(X).iloc[0])
        policy_result = apply_risk_policy(pd_value, self.threshold)

        response = {
            "model": self.model_name,
            **policy_result,
        }

        if client_id is not None:
            response = {
                "client_id": client_id,
                **response,
            }

        return response
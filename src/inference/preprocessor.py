import numpy as np
import pandas as pd


def prepare_features_for_lightgbm(
    data: pd.DataFrame,
    feature_cols: list[str],
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Prepare dataframe for LightGBM inference.

    Steps:
    - select required feature columns;
    - replace inf values with NaN;
    - convert only training categorical columns to category dtype;
    - convert all other columns to numeric.
    """
    if categorical_features is None:
        categorical_features = []

    missing_features = [
        col for col in feature_cols
        if col not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features[:20]}"
        )

    X = data[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    categorical_features = [
        col for col in categorical_features
        if col in X.columns
    ]

    numeric_features = [
        col for col in X.columns
        if col not in categorical_features
    ]

    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in categorical_features:
        X[col] = X[col].astype("category")

    return X


def prepare_single_client_features(
    client_features: dict,
    feature_cols: list[str],
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Prepare one client dict for LightGBM inference.
    """
    data = pd.DataFrame([client_features])

    return prepare_features_for_lightgbm(
        data=data,
        feature_cols=feature_cols,
        categorical_features=categorical_features,
    )
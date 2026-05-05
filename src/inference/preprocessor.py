import numpy as np
import pandas as pd


def prepare_features_for_lightgbm(
    data: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Prepare dataframe for LightGBM inference.

    Steps:
    - select required feature columns;
    - replace inf values with NaN;
    - convert categorical columns to category dtype.

    Parameters
    ----------
    data : pd.DataFrame
        Raw feature dataframe.
    feature_cols : list[str]
        Feature list used during model training.

    Returns
    -------
    X : pd.DataFrame
        Prepared feature dataframe.
    """
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

    categorical_features = X.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    for col in categorical_features:
        X[col] = X[col].astype("category")

    return X


def prepare_single_client_features(
    client_features: dict,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Prepare one client dict for LightGBM inference.

    Parameters
    ----------
    client_features : dict
        Dictionary with feature_name -> value.
    feature_cols : list[str]
        Feature list used during model training.

    Returns
    -------
    X : pd.DataFrame
        One-row prepared dataframe.
    """
    data = pd.DataFrame([client_features])

    return prepare_features_for_lightgbm(
        data=data,
        feature_cols=feature_cols,
    )
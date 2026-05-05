import lightgbm as lgb
import pandas as pd


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """
    Calculate scale_pos_weight for imbalanced binary classification.
    """
    negative_count = (y == 0).sum()
    positive_count = (y == 1).sum()

    return float(negative_count / positive_count)


def get_default_lightgbm_params(
    scale_pos_weight: float | None = None,
    seed: int = 42,
) -> dict:
    """
    Default LightGBM params for credit risk binary classification.
    """
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": seed,
    }

    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    return params


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_features: list[str] | None = None,
    params: dict | None = None,
    num_boost_round: int = 2000,
    stopping_rounds: int = 100,
):
    """
    Train LightGBM model with early stopping.
    """
    if categorical_features is None:
        categorical_features = []

    if params is None:
        scale_pos_weight = calculate_scale_pos_weight(y_train)
        params = get_default_lightgbm_params(scale_pos_weight=scale_pos_weight)

    lgb_train = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    lgb_valid = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=categorical_features,
        reference=lgb_train,
        free_raw_data=False,
    )

    model = lgb.train(
        params=params,
        train_set=lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=stopping_rounds,
                first_metric_only=True,
            ),
            lgb.log_evaluation(period=100),
        ],
    )

    return model
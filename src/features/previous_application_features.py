import numpy as np
import pandas as pd


def build_previous_application_features(
    previous_application: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create aggregated features from previous_application.csv.

    Parameters
    ----------
    previous_application : pd.DataFrame
        Raw previous_application dataframe.

    Returns
    -------
    pd.DataFrame
        Aggregated previous application features by SK_ID_CURR.
    """
    previous_application = previous_application.copy()

    previous_agg = previous_application.groupby("SK_ID_CURR").agg(
        PREV_APPLICATION_COUNT=("SK_ID_PREV", "count"),
        PREV_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_AMT_APPLICATION_MAX=("AMT_APPLICATION", "max"),
        PREV_AMT_APPLICATION_SUM=("AMT_APPLICATION", "sum"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_CREDIT_MAX=("AMT_CREDIT", "max"),
        PREV_AMT_CREDIT_SUM=("AMT_CREDIT", "sum"),
        PREV_AMT_DOWN_PAYMENT_MEAN=("AMT_DOWN_PAYMENT", "mean"),
        PREV_AMT_GOODS_PRICE_MEAN=("AMT_GOODS_PRICE", "mean"),
        PREV_HOUR_APPR_PROCESS_START_MEAN=("HOUR_APPR_PROCESS_START", "mean"),
        PREV_RATE_DOWN_PAYMENT_MEAN=("RATE_DOWN_PAYMENT", "mean"),
        PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
        PREV_DAYS_DECISION_MIN=("DAYS_DECISION", "min"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
        PREV_CNT_PAYMENT_MAX=("CNT_PAYMENT", "max"),
    ).reset_index()

    previous_agg["PREV_CREDIT_TO_APPLICATION_RATIO"] = (
        previous_agg["PREV_AMT_CREDIT_SUM"]
        / previous_agg["PREV_AMT_APPLICATION_SUM"]
    )

    previous_cat_cols = [
        "NAME_CONTRACT_STATUS",
        "NAME_CONTRACT_TYPE",
        "NAME_CLIENT_TYPE",
        "CHANNEL_TYPE",
        "NAME_YIELD_GROUP",
    ]

    previous_cat = pd.get_dummies(
        previous_application[["SK_ID_CURR"] + previous_cat_cols],
        columns=previous_cat_cols,
        dummy_na=True,
    )

    previous_cat_agg = previous_cat.groupby("SK_ID_CURR").sum().reset_index()

    previous_cat_agg = previous_cat_agg.rename(
        columns={
            col: f"PREV_{col}"
            for col in previous_cat_agg.columns
            if col != "SK_ID_CURR"
        }
    )

    previous_features = previous_agg.merge(
        previous_cat_agg,
        on="SK_ID_CURR",
        how="left",
    )

    previous_features["PREV_APPROVED_RATIO"] = (
        previous_features["PREV_NAME_CONTRACT_STATUS_Approved"]
        / previous_features["PREV_APPLICATION_COUNT"]
    )

    previous_features["PREV_REFUSED_RATIO"] = (
        previous_features["PREV_NAME_CONTRACT_STATUS_Refused"]
        / previous_features["PREV_APPLICATION_COUNT"]
    )

    previous_features["PREV_CANCELED_RATIO"] = (
        previous_features["PREV_NAME_CONTRACT_STATUS_Canceled"]
        / previous_features["PREV_APPLICATION_COUNT"]
    )

    previous_features = previous_features.replace([np.inf, -np.inf], np.nan)

    return previous_features
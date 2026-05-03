import numpy as np
import pandas as pd


def build_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated features from bureau.csv.

    Parameters
    ----------
    bureau : pd.DataFrame
        Raw bureau dataframe.

    Returns
    -------
    pd.DataFrame
        Aggregated bureau features by SK_ID_CURR.
    """
    bureau = bureau.copy()

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_RECORDS_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),
        BUREAU_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
        BUREAU_CREDIT_DAY_OVERDUE_MEAN=("CREDIT_DAY_OVERDUE", "mean"),
        BUREAU_CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_DAYS_CREDIT_ENDDATE_MEAN=("DAYS_CREDIT_ENDDATE", "mean"),
        BUREAU_DAYS_CREDIT_UPDATE_MEAN=("DAYS_CREDIT_UPDATE", "mean"),
        BUREAU_AMT_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUREAU_AMT_CREDIT_SUM_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_AMT_CREDIT_SUM_MAX=("AMT_CREDIT_SUM", "max"),
        BUREAU_AMT_CREDIT_SUM_DEBT_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"),
        BUREAU_AMT_CREDIT_SUM_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN=("AMT_CREDIT_SUM_OVERDUE", "mean"),
        BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        BUREAU_AMT_ANNUITY_SUM=("AMT_ANNUITY", "sum"),
    ).reset_index()

    bureau_cat = pd.get_dummies(
        bureau[["SK_ID_CURR", "CREDIT_ACTIVE", "CREDIT_TYPE"]],
        columns=["CREDIT_ACTIVE", "CREDIT_TYPE"],
        dummy_na=True,
    )

    bureau_cat_agg = bureau_cat.groupby("SK_ID_CURR").sum().reset_index()

    bureau_cat_agg = bureau_cat_agg.rename(
        columns={
            col: f"BUREAU_{col}"
            for col in bureau_cat_agg.columns
            if col != "SK_ID_CURR"
        }
    )

    bureau_features = bureau_agg.merge(
        bureau_cat_agg,
        on="SK_ID_CURR",
        how="left",
    )

    bureau_features["BUREAU_ACTIVE_CREDIT_RATIO"] = (
        bureau_features["BUREAU_CREDIT_ACTIVE_Active"]
        / bureau_features["BUREAU_RECORDS_COUNT"]
    )

    bureau_features["BUREAU_CLOSED_CREDIT_RATIO"] = (
        bureau_features["BUREAU_CREDIT_ACTIVE_Closed"]
        / bureau_features["BUREAU_RECORDS_COUNT"]
    )

    bureau_features["BUREAU_DEBT_TO_CREDIT_RATIO"] = (
        bureau_features["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"]
        / bureau_features["BUREAU_AMT_CREDIT_SUM_SUM"]
    )

    bureau_features["BUREAU_OVERDUE_TO_CREDIT_RATIO"] = (
        bureau_features["BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM"]
        / bureau_features["BUREAU_AMT_CREDIT_SUM_SUM"]
    )

    bureau_features = bureau_features.replace([np.inf, -np.inf], np.nan)

    return bureau_features


def build_bureau_balance_features(
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create aggregated features from bureau_balance.csv.

    Parameters
    ----------
    bureau : pd.DataFrame
        Raw bureau dataframe with SK_ID_CURR and SK_ID_BUREAU.

    bureau_balance : pd.DataFrame
        Raw bureau_balance dataframe.

    Returns
    -------
    pd.DataFrame
        Aggregated bureau_balance features by SK_ID_CURR.
    """
    bureau = bureau.copy()
    bureau_balance = bureau_balance.copy()

    bureau_balance_status = pd.get_dummies(
        bureau_balance[["SK_ID_BUREAU", "STATUS"]],
        columns=["STATUS"],
        dummy_na=True,
    )

    bureau_balance_agg = bureau_balance.groupby("SK_ID_BUREAU").agg(
        BUREAU_BALANCE_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        BUREAU_BALANCE_MONTHS_MIN=("MONTHS_BALANCE", "min"),
        BUREAU_BALANCE_MONTHS_MAX=("MONTHS_BALANCE", "max"),
    ).reset_index()

    bureau_balance_status_agg = (
        bureau_balance_status
        .groupby("SK_ID_BUREAU")
        .sum()
        .reset_index()
    )

    bureau_balance_status_agg = bureau_balance_status_agg.rename(
        columns={
            col: f"BUREAU_BALANCE_{col}"
            for col in bureau_balance_status_agg.columns
            if col != "SK_ID_BUREAU"
        }
    )

    bureau_balance_agg = bureau_balance_agg.merge(
        bureau_balance_status_agg,
        on="SK_ID_BUREAU",
        how="left",
    )

    bureau_with_balance = bureau[["SK_ID_CURR", "SK_ID_BUREAU"]].merge(
        bureau_balance_agg,
        on="SK_ID_BUREAU",
        how="left",
    )

    bureau_balance_client_features = bureau_with_balance.groupby("SK_ID_CURR").agg(
        BUREAU_BALANCE_MONTHS_COUNT_MEAN=("BUREAU_BALANCE_MONTHS_COUNT", "mean"),
        BUREAU_BALANCE_MONTHS_COUNT_SUM=("BUREAU_BALANCE_MONTHS_COUNT", "sum"),
        BUREAU_BALANCE_MONTHS_MIN=("BUREAU_BALANCE_MONTHS_MIN", "min"),
        BUREAU_BALANCE_MONTHS_MAX=("BUREAU_BALANCE_MONTHS_MAX", "max"),
    ).reset_index()

    status_cols = [
        col for col in bureau_with_balance.columns
        if col.startswith("BUREAU_BALANCE_STATUS_")
    ]

    bureau_balance_status_client = (
        bureau_with_balance[["SK_ID_CURR"] + status_cols]
        .groupby("SK_ID_CURR")
        .sum()
        .reset_index()
    )

    bureau_balance_client_features = bureau_balance_client_features.merge(
        bureau_balance_status_client,
        on="SK_ID_CURR",
        how="left",
    )

    bureau_balance_client_features = bureau_balance_client_features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return bureau_balance_client_features
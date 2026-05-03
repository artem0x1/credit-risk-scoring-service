import numpy as np
import pandas as pd


def build_pos_cash_features(pos_cash: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated features from POS_CASH_balance.csv.

    Parameters
    ----------
    pos_cash : pd.DataFrame
        Raw POS_CASH_balance dataframe.

    Returns
    -------
    pd.DataFrame
        Aggregated POS/CASH features by SK_ID_CURR.
    """
    pos_cash = pos_cash.copy()

    pos_cash_agg = pos_cash.groupby("SK_ID_CURR").agg(
        POS_RECORDS_COUNT=("SK_ID_PREV", "count"),
        POS_PREV_UNIQUE_COUNT=("SK_ID_PREV", "nunique"),
        POS_MONTHS_BALANCE_MEAN=("MONTHS_BALANCE", "mean"),
        POS_MONTHS_BALANCE_MIN=("MONTHS_BALANCE", "min"),
        POS_MONTHS_BALANCE_MAX=("MONTHS_BALANCE", "max"),
        POS_CNT_INSTALMENT_MEAN=("CNT_INSTALMENT", "mean"),
        POS_CNT_INSTALMENT_MAX=("CNT_INSTALMENT", "max"),
        POS_CNT_INSTALMENT_FUTURE_MEAN=("CNT_INSTALMENT_FUTURE", "mean"),
        POS_CNT_INSTALMENT_FUTURE_MAX=("CNT_INSTALMENT_FUTURE", "max"),
        POS_SK_DPD_MEAN=("SK_DPD", "mean"),
        POS_SK_DPD_MAX=("SK_DPD", "max"),
        POS_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        POS_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
    ).reset_index()

    pos_cash_cat = pd.get_dummies(
        pos_cash[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]],
        columns=["NAME_CONTRACT_STATUS"],
        dummy_na=True,
    )

    pos_cash_cat_agg = pos_cash_cat.groupby("SK_ID_CURR").sum().reset_index()

    pos_cash_cat_agg = pos_cash_cat_agg.rename(
        columns={
            col: f"POS_{col}"
            for col in pos_cash_cat_agg.columns
            if col != "SK_ID_CURR"
        }
    )

    pos_cash_features = pos_cash_agg.merge(
        pos_cash_cat_agg,
        on="SK_ID_CURR",
        how="left",
    )

    pos_cash_features = pos_cash_features.replace([np.inf, -np.inf], np.nan)

    return pos_cash_features
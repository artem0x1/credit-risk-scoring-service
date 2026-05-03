import numpy as np
import pandas as pd


def build_credit_card_features(credit_card: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated features from credit_card_balance.csv.

    Parameters
    ----------
    credit_card : pd.DataFrame
        Raw credit_card_balance dataframe.

    Returns
    -------
    pd.DataFrame
        Aggregated credit card features by SK_ID_CURR.
    """
    credit_card = credit_card.copy()

    credit_card_agg = credit_card.groupby("SK_ID_CURR").agg(
        CC_RECORDS_COUNT=("SK_ID_PREV", "count"),
        CC_PREV_UNIQUE_COUNT=("SK_ID_PREV", "nunique"),
        CC_MONTHS_BALANCE_MEAN=("MONTHS_BALANCE", "mean"),
        CC_MONTHS_BALANCE_MIN=("MONTHS_BALANCE", "min"),
        CC_MONTHS_BALANCE_MAX=("MONTHS_BALANCE", "max"),
        CC_AMT_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_AMT_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_AMT_CREDIT_LIMIT_ACTUAL_MAX=("AMT_CREDIT_LIMIT_ACTUAL", "max"),
        CC_AMT_DRAWINGS_CURRENT_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
        CC_AMT_DRAWINGS_CURRENT_MAX=("AMT_DRAWINGS_CURRENT", "max"),
        CC_AMT_PAYMENT_CURRENT_MEAN=("AMT_PAYMENT_CURRENT", "mean"),
        CC_AMT_PAYMENT_CURRENT_SUM=("AMT_PAYMENT_CURRENT", "sum"),
        CC_AMT_RECEIVABLE_PRINCIPAL_MEAN=("AMT_RECEIVABLE_PRINCIPAL", "mean"),
        CC_AMT_RECIVABLE_MEAN=("AMT_RECIVABLE", "mean"),
        CC_AMT_TOTAL_RECEIVABLE_MEAN=("AMT_TOTAL_RECEIVABLE", "mean"),
        CC_CNT_DRAWINGS_CURRENT_MEAN=("CNT_DRAWINGS_CURRENT", "mean"),
        CC_CNT_DRAWINGS_CURRENT_MAX=("CNT_DRAWINGS_CURRENT", "max"),
        CC_SK_DPD_MEAN=("SK_DPD", "mean"),
        CC_SK_DPD_MAX=("SK_DPD", "max"),
        CC_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        CC_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
    ).reset_index()

    credit_card_cat = pd.get_dummies(
        credit_card[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]],
        columns=["NAME_CONTRACT_STATUS"],
        dummy_na=True,
    )

    credit_card_cat_agg = credit_card_cat.groupby("SK_ID_CURR").sum().reset_index()

    credit_card_cat_agg = credit_card_cat_agg.rename(
        columns={
            col: f"CC_{col}"
            for col in credit_card_cat_agg.columns
            if col != "SK_ID_CURR"
        }
    )

    credit_card_features = credit_card_agg.merge(
        credit_card_cat_agg,
        on="SK_ID_CURR",
        how="left",
    )

    credit_card_features = credit_card_features.replace([np.inf, -np.inf], np.nan)

    return credit_card_features
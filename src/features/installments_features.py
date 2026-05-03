import numpy as np
import pandas as pd


def build_installments_features(installments: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated features from installments_payments.csv.

    Parameters
    ----------
    installments : pd.DataFrame
        Raw installments_payments dataframe.

    Returns
    -------
    pd.DataFrame
        Aggregated installments features by SK_ID_CURR.
    """
    installments = installments.copy()

    # Payment-level features
    installments["INSTALLMENT_PAYMENT_DELAY"] = (
        installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    )

    installments["INSTALLMENT_PAYMENT_DIFF"] = (
        installments["AMT_PAYMENT"] - installments["AMT_INSTALMENT"]
    )

    installments["INSTALLMENT_LATE_PAYMENT_FLAG"] = (
        installments["INSTALLMENT_PAYMENT_DELAY"] > 0
    ).astype(int)

    installments["INSTALLMENT_UNDERPAYMENT_FLAG"] = (
        installments["INSTALLMENT_PAYMENT_DIFF"] < 0
    ).astype(int)

    installments_features = installments.groupby("SK_ID_CURR").agg(
        INSTAL_RECORDS_COUNT=("SK_ID_PREV", "count"),
        INSTAL_PREV_UNIQUE_COUNT=("SK_ID_PREV", "nunique"),
        INSTAL_NUM_INSTALMENT_VERSION_MEAN=("NUM_INSTALMENT_VERSION", "mean"),
        INSTAL_NUM_INSTALMENT_NUMBER_MAX=("NUM_INSTALMENT_NUMBER", "max"),
        INSTAL_DAYS_INSTALMENT_MEAN=("DAYS_INSTALMENT", "mean"),
        INSTAL_DAYS_ENTRY_PAYMENT_MEAN=("DAYS_ENTRY_PAYMENT", "mean"),
        INSTAL_PAYMENT_DELAY_MEAN=("INSTALLMENT_PAYMENT_DELAY", "mean"),
        INSTAL_PAYMENT_DELAY_MAX=("INSTALLMENT_PAYMENT_DELAY", "max"),
        INSTAL_PAYMENT_DELAY_SUM=("INSTALLMENT_PAYMENT_DELAY", "sum"),
        INSTAL_PAYMENT_DIFF_MEAN=("INSTALLMENT_PAYMENT_DIFF", "mean"),
        INSTAL_PAYMENT_DIFF_SUM=("INSTALLMENT_PAYMENT_DIFF", "sum"),
        INSTAL_LATE_PAYMENT_MEAN=("INSTALLMENT_LATE_PAYMENT_FLAG", "mean"),
        INSTAL_LATE_PAYMENT_SUM=("INSTALLMENT_LATE_PAYMENT_FLAG", "sum"),
        INSTAL_UNDERPAYMENT_MEAN=("INSTALLMENT_UNDERPAYMENT_FLAG", "mean"),
        INSTAL_UNDERPAYMENT_SUM=("INSTALLMENT_UNDERPAYMENT_FLAG", "sum"),
        INSTAL_AMT_INSTALMENT_MEAN=("AMT_INSTALMENT", "mean"),
        INSTAL_AMT_INSTALMENT_SUM=("AMT_INSTALMENT", "sum"),
        INSTAL_AMT_PAYMENT_MEAN=("AMT_PAYMENT", "mean"),
        INSTAL_AMT_PAYMENT_SUM=("AMT_PAYMENT", "sum"),
    ).reset_index()

    installments_features = installments_features.replace([np.inf, -np.inf], np.nan)

    return installments_features
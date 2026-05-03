import numpy as np
import pandas as pd


def build_application_features(application: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from application_train/application_test table.

    Parameters
    ----------
    application : pd.DataFrame
        Combined application train/test dataframe.

    Returns
    -------
    pd.DataFrame
        Application dataframe with new engineered features.
    """
    application = application.copy()

    # DAYS_EMPLOYED anomaly
    application["DAYS_EMPLOYED_ANOM"] = (
        application["DAYS_EMPLOYED"] == 365243
    ).astype(int)

    application["DAYS_EMPLOYED"] = application["DAYS_EMPLOYED"].replace(
        365243,
        np.nan,
    )

    # Age and time features
    application["AGE_YEARS"] = -application["DAYS_BIRTH"] / 365
    application["EMPLOYED_YEARS"] = -application["DAYS_EMPLOYED"] / 365
    application["REGISTRATION_YEARS"] = -application["DAYS_REGISTRATION"] / 365
    application["ID_PUBLISH_YEARS"] = -application["DAYS_ID_PUBLISH"] / 365

    # Amount ratios
    application["CREDIT_TO_INCOME_RATIO"] = (
        application["AMT_CREDIT"] / application["AMT_INCOME_TOTAL"]
    )

    application["ANNUITY_TO_INCOME_RATIO"] = (
        application["AMT_ANNUITY"] / application["AMT_INCOME_TOTAL"]
    )

    application["CREDIT_TO_ANNUITY_RATIO"] = (
        application["AMT_CREDIT"] / application["AMT_ANNUITY"]
    )

    application["GOODS_TO_CREDIT_RATIO"] = (
        application["AMT_GOODS_PRICE"] / application["AMT_CREDIT"]
    )

    # Family and income features
    application["INCOME_PER_PERSON"] = (
        application["AMT_INCOME_TOTAL"] / application["CNT_FAM_MEMBERS"]
    )

    application["INCOME_PER_CHILD"] = (
        application["AMT_INCOME_TOTAL"] / (application["CNT_CHILDREN"] + 1)
    )

    application["CHILDREN_RATIO"] = (
        application["CNT_CHILDREN"] / application["CNT_FAM_MEMBERS"]
    )

    application["CREDIT_PER_PERSON"] = (
        application["AMT_CREDIT"] / application["CNT_FAM_MEMBERS"]
    )

    # EXT_SOURCE aggregations
    ext_source_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

    application["EXT_SOURCE_MEAN"] = application[ext_source_cols].mean(axis=1)
    application["EXT_SOURCE_MIN"] = application[ext_source_cols].min(axis=1)
    application["EXT_SOURCE_MAX"] = application[ext_source_cols].max(axis=1)
    application["EXT_SOURCE_STD"] = application[ext_source_cols].std(axis=1)
    application["EXT_SOURCE_MISSING_COUNT"] = (
        application[ext_source_cols].isna().sum(axis=1)
    )

    # Document features
    document_cols = [
        col for col in application.columns
        if col.startswith("FLAG_DOCUMENT_")
    ]

    application["DOCUMENT_COUNT"] = application[document_cols].sum(axis=1)
    application["DOCUMENT_MEAN"] = application[document_cols].mean(axis=1)

    # Social circle features
    application["DEF_30_SOCIAL_RATIO"] = (
        application["DEF_30_CNT_SOCIAL_CIRCLE"]
        / application["OBS_30_CNT_SOCIAL_CIRCLE"]
    )

    application["DEF_60_SOCIAL_RATIO"] = (
        application["DEF_60_CNT_SOCIAL_CIRCLE"]
        / application["OBS_60_CNT_SOCIAL_CIRCLE"]
    )

    # Contact and region features
    contact_cols = [
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
    ]

    region_mismatch_cols = [
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
    ]

    application["CONTACT_FLAGS_SUM"] = application[contact_cols].sum(axis=1)
    application["REGION_MISMATCH_SUM"] = application[region_mismatch_cols].sum(axis=1)

    # Replace infinities created by division
    application = application.replace([np.inf, -np.inf], np.nan)

    return application
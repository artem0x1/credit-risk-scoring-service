from pathlib import Path

import numpy as np
import pandas as pd

from src.features.application_features import build_application_features
from src.features.bureau_features import (
    build_bureau_balance_features,
    build_bureau_features,
)
from src.features.credit_card_features import build_credit_card_features
from src.features.installments_features import build_installments_features
from src.features.pos_cash_features import build_pos_cash_features
from src.features.previous_application_features import (
    build_previous_application_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_application_data() -> pd.DataFrame:
    """
    Load application_train and application_test, then combine them into one dataframe.

    Returns
    -------
    pd.DataFrame
        Combined application dataframe with DATASET column.
    """
    application_train = pd.read_csv(RAW_DATA_DIR / "application_train.csv")
    application_test = pd.read_csv(RAW_DATA_DIR / "application_test.csv")

    application_train["DATASET"] = "train"
    application_test["DATASET"] = "test"

    application_test["TARGET"] = np.nan

    application = pd.concat(
        [application_train, application_test],
        axis=0,
        ignore_index=True,
    )

    return application


def merge_features(
    application: pd.DataFrame,
    features: pd.DataFrame,
    feature_name: str,
) -> pd.DataFrame:
    """
    Merge feature dataframe with application dataframe.

    Parameters
    ----------
    application : pd.DataFrame
        Main application dataframe.

    features : pd.DataFrame
        Aggregated feature dataframe with SK_ID_CURR.

    feature_name : str
        Name of feature block for logging.

    Returns
    -------
    pd.DataFrame
        Application dataframe with merged features.
    """
    shape_before = application.shape

    application = application.merge(
        features,
        on="SK_ID_CURR",
        how="left",
    )

    print(
        f"{feature_name}: {shape_before} -> {application.shape}"
    )

    return application


def split_train_test(application: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split combined application dataframe back into train and test.

    Parameters
    ----------
    application : pd.DataFrame
        Combined application dataframe.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Train and test feature dataframes.
    """
    train_features = application[application["DATASET"] == "train"].copy()
    test_features = application[application["DATASET"] == "test"].copy()

    train_features = train_features.drop(columns=["DATASET"])
    test_features = test_features.drop(columns=["DATASET", "TARGET"])

    return train_features, test_features


def save_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> None:
    """
    Save train/test feature datasets to data/processed.

    Parameters
    ----------
    train_features : pd.DataFrame
        Final train features with TARGET.

    test_features : pd.DataFrame
        Final test features without TARGET.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_features_path = PROCESSED_DATA_DIR / "train_features.parquet"
    test_features_path = PROCESSED_DATA_DIR / "test_features.parquet"
    feature_columns_path = PROCESSED_DATA_DIR / "feature_columns.csv"

    train_features.to_parquet(train_features_path, index=False)
    test_features.to_parquet(test_features_path, index=False)

    feature_columns = pd.DataFrame(
        {
            "feature": train_features.drop(columns=["TARGET"]).columns,
        }
    )

    feature_columns.to_csv(feature_columns_path, index=False)

    print(f"Saved train features: {train_features_path}")
    print(f"Saved test features: {test_features_path}")
    print(f"Saved feature columns: {feature_columns_path}")


def build_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full feature engineering pipeline.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Final train and test feature dataframes.
    """
    print("Loading application data...")
    application = load_application_data()
    print(f"Application shape: {application.shape}")

    print("Building application features...")
    application = build_application_features(application)
    print(f"Application with new features shape: {application.shape}")

    print("Loading bureau data...")
    bureau = pd.read_csv(RAW_DATA_DIR / "bureau.csv")
    bureau_features = build_bureau_features(bureau)
    application = merge_features(application, bureau_features, "Bureau features")

    print("Loading bureau_balance data...")
    bureau_balance = pd.read_csv(RAW_DATA_DIR / "bureau_balance.csv")
    bureau_balance_features = build_bureau_balance_features(
        bureau=bureau,
        bureau_balance=bureau_balance,
    )
    application = merge_features(
        application,
        bureau_balance_features,
        "Bureau balance features",
    )

    print("Loading previous_application data...")
    previous_application = pd.read_csv(RAW_DATA_DIR / "previous_application.csv")
    previous_features = build_previous_application_features(previous_application)
    application = merge_features(
        application,
        previous_features,
        "Previous application features",
    )

    print("Loading POS_CASH_balance data...")
    pos_cash = pd.read_csv(RAW_DATA_DIR / "POS_CASH_balance.csv")
    pos_cash_features = build_pos_cash_features(pos_cash)
    application = merge_features(
        application,
        pos_cash_features,
        "POS CASH features",
    )

    print("Loading installments_payments data...")
    installments = pd.read_csv(RAW_DATA_DIR / "installments_payments.csv")
    installments_features = build_installments_features(installments)
    application = merge_features(
        application,
        installments_features,
        "Installments features",
    )

    print("Loading credit_card_balance data...")
    credit_card = pd.read_csv(RAW_DATA_DIR / "credit_card_balance.csv")
    credit_card_features = build_credit_card_features(credit_card)
    application = merge_features(
        application,
        credit_card_features,
        "Credit card features",
    )

    print("Replacing infinite values...")
    application = application.replace([np.inf, -np.inf], np.nan)

    print("Final checks...")
    print(f"Final application shape: {application.shape}")
    print(f"Duplicated SK_ID_CURR: {application['SK_ID_CURR'].duplicated().sum()}")
    print(application["DATASET"].value_counts())

    train_features, test_features = split_train_test(application)

    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")

    save_features(train_features, test_features)

    return train_features, test_features


if __name__ == "__main__":
    build_features()
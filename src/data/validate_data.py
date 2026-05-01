import pandas as pd


def validate_application_train(df: pd.DataFrame) -> None:
    required_columns = ["SK_ID_CURR", "TARGET"]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("application_train is empty")

    if df["SK_ID_CURR"].duplicated().any():
        raise ValueError("SK_ID_CURR contains duplicates in application_train")

    if not set(df["TARGET"].dropna().unique()).issubset({0, 1}):
        raise ValueError("TARGET must contain only 0 and 1")


def validate_application_test(df: pd.DataFrame) -> None:
    required_columns = ["SK_ID_CURR"]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("application_test is empty")

    if df["SK_ID_CURR"].duplicated().any():
        raise ValueError("SK_ID_CURR contains duplicates in application_test")


def validate_has_key(df: pd.DataFrame, key: str, table_name: str) -> None:
    if key not in df.columns:
        raise ValueError(f"{table_name} does not contain key column: {key}")

    if df.empty:
        raise ValueError(f"{table_name} is empty")
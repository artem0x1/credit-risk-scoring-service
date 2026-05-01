from pathlib import Path

import pandas as pd

from src.paths import RAW_DATA_DIR


def load_application_train(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_data_dir / "application_train.csv")


def load_application_test(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_data_dir / "application_test.csv")


def load_bureau(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_data_dir / "bureau.csv")


def load_bureau_balance(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_data_dir / "bureau_balance.csv")


def load_previous_application(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_data_dir / "previous_application.csv")


def load_pos_cash_balance(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_data_dir / "POS_CASH_balance.csv")


def load_installments_payments(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_data_dir / "installments_payments.csv")


def load_credit_card_balance(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_data_dir / "credit_card_balance.csv")


def load_all_raw_tables(raw_data_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    return {
        "application_train": load_application_train(raw_data_dir),
        "application_test": load_application_test(raw_data_dir),
        "bureau": load_bureau(raw_data_dir),
        "bureau_balance": load_bureau_balance(raw_data_dir),
        "previous_application": load_previous_application(raw_data_dir),
        "pos_cash_balance": load_pos_cash_balance(raw_data_dir),
        "installments_payments": load_installments_payments(raw_data_dir),
        "credit_card_balance": load_credit_card_balance(raw_data_dir),
    }
from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    """
    Return project root directory.

    Assumes this file is located in src/utils/.
    """
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """
    Return data directory.
    """
    return get_project_root() / "data"


def get_raw_data_dir() -> Path:
    """
    Return raw data directory.
    """
    return get_data_dir() / "raw"


def get_processed_data_dir() -> Path:
    """
    Return processed data directory.
    """
    return get_data_dir() / "processed"


def load_train_features() -> pd.DataFrame:
    """
    Load processed train features.
    """
    path = get_processed_data_dir() / "train_features.parquet"
    return pd.read_parquet(path)


def load_test_features() -> pd.DataFrame:
    """
    Load processed test features.
    """
    path = get_processed_data_dir() / "test_features.parquet"
    return pd.read_parquet(path)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save dataframe to parquet or csv depending on file extension.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")


def check_duplicates(df: pd.DataFrame, id_col: str = "SK_ID_CURR") -> int:
    """
    Return number of duplicated IDs.
    """
    return df[id_col].duplicated().sum()


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return missing values summary.
    """
    return (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", 0: "missing_share"})
    )
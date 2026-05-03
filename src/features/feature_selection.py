import pandas as pd


def get_feature_columns(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    id_col: str = "SK_ID_CURR",
) -> list[str]:
    """
    Return feature columns excluding target and id columns.
    """
    excluded_cols = {target_col, id_col}

    return [
        col for col in df.columns
        if col not in excluded_cols
    ]


def get_numeric_features(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    id_col: str = "SK_ID_CURR",
) -> list[str]:
    """
    Return numeric feature columns.
    """
    feature_cols = get_feature_columns(
        df=df,
        target_col=target_col,
        id_col=id_col,
    )

    numeric_cols = df[feature_cols].select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    return numeric_cols


def get_categorical_features(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    id_col: str = "SK_ID_CURR",
) -> list[str]:
    """
    Return categorical feature columns.
    """
    feature_cols = get_feature_columns(
        df=df,
        target_col=target_col,
        id_col=id_col,
    )

    categorical_cols = df[feature_cols].select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return categorical_cols


def remove_high_missing_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    protected_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Remove columns with missing value share above threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    threshold : float
        Maximum allowed missing value share.

    protected_cols : list[str] | None
        Columns that should never be removed.

    Returns
    -------
    pd.DataFrame
        Dataframe without high-missing columns.
    """
    if protected_cols is None:
        protected_cols = []

    missing_share = df.isna().mean()

    cols_to_drop = [
        col for col in missing_share.index
        if missing_share[col] > threshold and col not in protected_cols
    ]

    return df.drop(columns=cols_to_drop)


def remove_constant_features(
    df: pd.DataFrame,
    protected_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Remove columns with only one unique value.
    """
    if protected_cols is None:
        protected_cols = []

    cols_to_drop = [
        col for col in df.columns
        if col not in protected_cols and df[col].nunique(dropna=False) <= 1
    ]

    return df.drop(columns=cols_to_drop)


def remove_duplicate_columns(
    df: pd.DataFrame,
    protected_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Remove duplicated columns.
    """
    if protected_cols is None:
        protected_cols = []

    duplicated_cols = df.columns[df.T.duplicated()].tolist()

    cols_to_drop = [
        col for col in duplicated_cols
        if col not in protected_cols
    ]

    return df.drop(columns=cols_to_drop)
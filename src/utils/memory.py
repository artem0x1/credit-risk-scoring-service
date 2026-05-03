import pandas as pd


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory usage by downcasting numeric columns.
    """
    df = df.copy()

    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")

        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")

    return df
import numpy as np
import pandas as pd


def calculate_woe_iv_for_feature(
    x: pd.Series,
    y: pd.Series,
    feature_name: str,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> tuple[pd.DataFrame, float]:
    """
    Calculate WOE/IV table for one feature.

    Parameters
    ----------
    x : pd.Series
        Feature values.
    y : pd.Series
        Binary target. 1 = bad client, 0 = good client.
    feature_name : str
        Feature name.
    n_bins : int
        Number of quantile bins for numeric features.
    eps : float
        Smoothing value to avoid division by zero.

    Returns
    -------
    woe_table : pd.DataFrame
        WOE table by bins.
    iv : float
        Information Value of the feature.
    """
    df = pd.DataFrame(
        {
            "feature": x,
            "target": y,
        }
    )

    is_numeric = pd.api.types.is_numeric_dtype(df["feature"])

    if is_numeric:
        not_null = df["feature"].notna()
        unique_values = df.loc[not_null, "feature"].nunique()

        if unique_values <= 1:
            df["bin"] = "SINGLE_VALUE"
            df.loc[~not_null, "bin"] = "MISSING"
        else:
            try:
                df.loc[not_null, "bin"] = pd.qcut(
                    df.loc[not_null, "feature"],
                    q=n_bins,
                    duplicates="drop",
                ).astype(str)

                df.loc[~not_null, "bin"] = "MISSING"

            except ValueError:
                df["bin"] = df["feature"].astype(str)
                df.loc[df["feature"].isna(), "bin"] = "MISSING"
    else:
        df["bin"] = df["feature"].astype("object")
        df.loc[df["feature"].isna(), "bin"] = "MISSING"
        df["bin"] = df["bin"].astype(str)

    grouped = (
        df.groupby("bin", dropna=False)
        .agg(
            total=("target", "count"),
            bad=("target", "sum"),
        )
        .reset_index()
    )

    grouped["good"] = grouped["total"] - grouped["bad"]

    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()

    grouped["good_dist"] = (grouped["good"] + eps) / (total_good + eps)
    grouped["bad_dist"] = (grouped["bad"] + eps) / (total_bad + eps)

    grouped["woe"] = np.log(grouped["good_dist"] / grouped["bad_dist"])
    grouped["iv_component"] = (
        grouped["good_dist"] - grouped["bad_dist"]
    ) * grouped["woe"]

    grouped["bad_rate"] = grouped["bad"] / grouped["total"]
    grouped["feature"] = feature_name

    iv = float(grouped["iv_component"].sum())

    grouped = grouped[
        [
            "feature",
            "bin",
            "total",
            "good",
            "bad",
            "bad_rate",
            "good_dist",
            "bad_dist",
            "woe",
            "iv_component",
        ]
    ]

    return grouped, iv


def calculate_iv_for_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate IV for all features in dataframe.

    Returns
    -------
    iv_df : pd.DataFrame
        Feature-level IV table.
    woe_full_table : pd.DataFrame
        Full WOE table for all features.
    """
    woe_tables = []
    iv_values = []

    for feature in X.columns:
        woe_table, iv = calculate_woe_iv_for_feature(
            X[feature],
            y,
            feature_name=feature,
            n_bins=n_bins,
        )

        woe_tables.append(woe_table)
        iv_values.append(
            {
                "feature": feature,
                "iv": iv,
                "n_bins": woe_table["bin"].nunique(),
                "missing_rate": X[feature].isna().mean(),
            }
        )

    iv_df = (
        pd.DataFrame(iv_values)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )

    woe_full_table = pd.concat(woe_tables, ignore_index=True)

    return iv_df, woe_full_table
import numpy as np
import pandas as pd


def fit_woe_transformer(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    n_bins: int = 10,
    eps: float = 1e-6,
) -> tuple[dict, pd.DataFrame]:
    """
    Fit WOE transformer on train data.

    For each feature stores:
    - feature type;
    - numeric bin edges;
    - bin/category to WOE mapping;
    - default WOE for unknown values.

    Parameters
    ----------
    X : pd.DataFrame
        Train features.
    y : pd.Series
        Binary target. 1 = bad client, 0 = good client.
    features : list[str]
        Features to transform.
    n_bins : int
        Number of quantile bins for numeric features.
    eps : float
        Smoothing value.

    Returns
    -------
    transformer : dict
        Fitted WOE transformer.
    woe_table : pd.DataFrame
        WOE table for selected features.
    """
    transformer = {}
    woe_tables = []

    for feature in features:
        df = pd.DataFrame(
            {
                "feature_value": X[feature],
                "target": y,
            }
        )

        is_numeric = pd.api.types.is_numeric_dtype(df["feature_value"])

        if is_numeric:
            not_null = df["feature_value"].notna()
            unique_values = df.loc[not_null, "feature_value"].nunique()

            if unique_values <= 1:
                df["bin"] = "SINGLE_VALUE"
                df.loc[~not_null, "bin"] = "MISSING"
                bin_edges = None
            else:
                try:
                    _, bin_edges = pd.qcut(
                        df.loc[not_null, "feature_value"],
                        q=n_bins,
                        retbins=True,
                        duplicates="drop",
                    )

                    bin_edges[0] = -np.inf
                    bin_edges[-1] = np.inf

                    df.loc[not_null, "bin"] = pd.cut(
                        df.loc[not_null, "feature_value"],
                        bins=bin_edges,
                        include_lowest=True,
                    ).astype(str)

                    df.loc[~not_null, "bin"] = "MISSING"

                except Exception:
                    df["bin"] = "SINGLE_VALUE"
                    df.loc[~not_null, "bin"] = "MISSING"
                    bin_edges = None
        else:
            df["bin"] = df["feature_value"].astype("object")
            df.loc[df["feature_value"].isna(), "bin"] = "MISSING"
            df["bin"] = df["bin"].astype(str)
            bin_edges = None

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
        grouped["feature"] = feature

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

        woe_mapping = dict(zip(grouped["bin"], grouped["woe"]))

        transformer[feature] = {
            "is_numeric": is_numeric,
            "bin_edges": bin_edges,
            "woe_mapping": woe_mapping,
            "default_woe": 0.0,
        }

        woe_tables.append(grouped)

    woe_table = pd.concat(woe_tables, ignore_index=True)

    return transformer, woe_table


def apply_woe_transformer(
    X: pd.DataFrame,
    transformer: dict,
) -> pd.DataFrame:
    """
    Apply fitted WOE transformer to dataframe.

    Parameters
    ----------
    X : pd.DataFrame
        Dataset to transform.
    transformer : dict
        Fitted WOE transformer.

    Returns
    -------
    X_woe : pd.DataFrame
        WOE-transformed dataframe.
    """
    X_woe = pd.DataFrame(index=X.index)

    for feature, params in transformer.items():
        x = X[feature]

        if params["is_numeric"]:
            bin_edges = params["bin_edges"]

            if bin_edges is None:
                bins = pd.Series("SINGLE_VALUE", index=x.index)
                bins[x.isna()] = "MISSING"
            else:
                bins = pd.cut(
                    x,
                    bins=bin_edges,
                    include_lowest=True,
                ).astype(str)

                bins[x.isna()] = "MISSING"
        else:
            bins = x.astype("object")
            bins[x.isna()] = "MISSING"
            bins = bins.astype(str)

        X_woe[feature] = (
            bins.map(params["woe_mapping"])
            .fillna(params["default_woe"])
        )

    return X_woe
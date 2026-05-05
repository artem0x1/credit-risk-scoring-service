import numpy as np
import pandas as pd


def fit_numeric_bins(
    x: pd.Series,
    n_bins: int = 10,
) -> np.ndarray | None:
    """
    Fit quantile bin edges for numeric feature.

    Returns None if feature is constant or cannot be binned.
    """
    x_not_null = x.dropna()

    if x_not_null.nunique() <= 1:
        return None

    try:
        _, bin_edges = pd.qcut(
            x_not_null,
            q=n_bins,
            retbins=True,
            duplicates="drop",
        )

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        return bin_edges

    except Exception:
        return None


def apply_numeric_bins(
    x: pd.Series,
    bin_edges: np.ndarray | None,
    missing_label: str = "MISSING",
    single_value_label: str = "SINGLE_VALUE",
) -> pd.Series:
    """
    Apply fitted numeric bin edges to numeric feature.
    """
    if bin_edges is None:
        bins = pd.Series(single_value_label, index=x.index, dtype="object")
        bins[x.isna()] = missing_label
        return bins

    bins = pd.cut(
        x,
        bins=bin_edges,
        include_lowest=True,
    ).astype(str)

    bins[x.isna()] = missing_label

    return bins.astype("object")


def apply_categorical_bins(
    x: pd.Series,
    missing_label: str = "MISSING",
) -> pd.Series:
    """
    Convert categorical feature values to string bins.
    Missing values are assigned to MISSING bin.
    """
    bins = x.astype("object")
    bins[x.isna()] = missing_label

    return bins.astype(str)


def fit_feature_bins(
    x: pd.Series,
    n_bins: int = 10,
) -> dict:
    """
    Fit binning rules for one feature.

    Returns dictionary with:
    - is_numeric
    - bin_edges
    """
    is_numeric = pd.api.types.is_numeric_dtype(x)

    if is_numeric:
        bin_edges = fit_numeric_bins(x, n_bins=n_bins)
    else:
        bin_edges = None

    return {
        "is_numeric": is_numeric,
        "bin_edges": bin_edges,
    }


def apply_feature_bins(
    x: pd.Series,
    binning_rules: dict,
    missing_label: str = "MISSING",
    single_value_label: str = "SINGLE_VALUE",
) -> pd.Series:
    """
    Apply fitted binning rules to one feature.
    """
    if binning_rules["is_numeric"]:
        return apply_numeric_bins(
            x,
            bin_edges=binning_rules["bin_edges"],
            missing_label=missing_label,
            single_value_label=single_value_label,
        )

    return apply_categorical_bins(
        x,
        missing_label=missing_label,
    )
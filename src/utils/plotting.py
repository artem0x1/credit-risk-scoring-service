import matplotlib.pyplot as plt
import pandas as pd


def plot_missing_values(
    df: pd.DataFrame,
    top_n: int = 30,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Plot top columns by missing value share.
    """
    missing = df.isna().mean().sort_values(ascending=False).head(top_n)

    plt.figure(figsize=figsize)
    missing.sort_values().plot(kind="barh")
    plt.xlabel("Missing share")
    plt.title(f"Top {top_n} features by missing values")
    plt.tight_layout()
    plt.show()


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "TARGET",
) -> None:
    """
    Plot target distribution.
    """
    df[target_col].value_counts(normalize=True).sort_index().plot(kind="bar")
    plt.xlabel(target_col)
    plt.ylabel("Share")
    plt.title("Target distribution")
    plt.tight_layout()
    plt.show()
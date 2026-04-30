from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_application_train() -> pd.DataFrame:
    path = RAW_DATA_DIR / "application_train.csv"

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def print_basic_info(df: pd.DataFrame) -> None:
    print("=" * 80)
    print("APPLICATION TRAIN BASIC INFO")
    print("=" * 80)

    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDtypes:")
    print(df.dtypes.value_counts())

    print("\nTarget distribution:")
    print(df["TARGET"].value_counts(normalize=True).rename("share"))


def save_target_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))

    ax = sns.countplot(data=df, x="TARGET")
    ax.set_title("Target Distribution")
    ax.set_xlabel("TARGET")
    ax.set_ylabel("Count")

    total = len(df)
    for p in ax.patches:
        count = int(p.get_height())
        share = count / total
        ax.annotate(
            f"{share:.2%}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
        )

    output_path = FIGURES_DIR / "target_distribution.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def save_missing_values_report(df: pd.DataFrame) -> None:
    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    output_path = REPORTS_DIR / "missing_values_application_train.csv"
    missing.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


def save_top_missing_values_plot(df: pd.DataFrame, top_n: int = 30) -> None:
    missing = df.isna().mean().sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=missing.values, y=missing.index)

    plt.title(f"Top {top_n} Features by Missing Rate")
    plt.xlabel("Missing Rate")
    plt.ylabel("Feature")

    output_path = FIGURES_DIR / "top_missing_values.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def save_numeric_summary(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=["number"])

    summary = numeric_df.describe().T
    summary["missing_rate"] = numeric_df.isna().mean()
    summary["n_unique"] = numeric_df.nunique()

    output_path = REPORTS_DIR / "numeric_summary_application_train.csv"
    summary.to_csv(output_path)

    print(f"Saved: {output_path}")


def save_categorical_summary(df: pd.DataFrame) -> None:
    categorical_df = df.select_dtypes(include=["object", "string", "str"])

    summary = pd.DataFrame(
        {
            "missing_rate": categorical_df.isna().mean(),
            "n_unique": categorical_df.nunique(),
        }
    ).sort_values("missing_rate", ascending=False)

    output_path = REPORTS_DIR / "categorical_summary_application_train.csv"
    summary.to_csv(output_path)

    print(f"Saved: {output_path}")


def save_correlation_with_target(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=["number"])

    if "TARGET" not in numeric_df.columns:
        raise ValueError("TARGET column not found in numeric columns.")

    corr = (
        numeric_df.corr(numeric_only=True)["TARGET"]
        .drop("TARGET")
        .sort_values(key=abs, ascending=False)
        .rename("correlation_with_target")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    output_path = REPORTS_DIR / "correlation_with_target.csv"
    corr.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


def save_top_correlations_plot(df: pd.DataFrame, top_n: int = 30) -> None:
    numeric_df = df.select_dtypes(include=["number"])

    corr = (
        numeric_df.corr(numeric_only=True)["TARGET"]
        .drop("TARGET")
        .sort_values(key=abs, ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(x=corr.values, y=corr.index)

    plt.title(f"Top {top_n} Numeric Features by Correlation with TARGET")
    plt.xlabel("Correlation with TARGET")
    plt.ylabel("Feature")

    output_path = FIGURES_DIR / "top_correlations_with_target.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


def save_amount_credit_distribution(df: pd.DataFrame) -> None:
    columns = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    ]

    existing_columns = [col for col in columns if col in df.columns]

    for col in existing_columns:
        plt.figure(figsize=(8, 4))

        sns.histplot(df[col].dropna(), bins=50, kde=True)

        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")

        output_path = FIGURES_DIR / f"distribution_{col.lower()}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Saved: {output_path}")


def save_target_by_categorical_features(df: pd.DataFrame) -> None:
    categorical_features = [
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
    ]

    existing_features = [col for col in categorical_features if col in df.columns]

    for col in existing_features:
        target_rate = (
            df.groupby(col, dropna=False)["TARGET"]
            .mean()
            .sort_values(ascending=False)
            .head(20)
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(x=target_rate.values, y=target_rate.index)

        plt.title(f"Default Rate by {col}")
        plt.xlabel("Mean TARGET")
        plt.ylabel(col)

        output_path = FIGURES_DIR / f"default_rate_by_{col.lower()}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Saved: {output_path}")


def main() -> None:
    df = load_application_train()

    print_basic_info(df)

    save_target_distribution(df)
    save_missing_values_report(df)
    save_top_missing_values_plot(df)
    save_numeric_summary(df)
    save_categorical_summary(df)
    save_correlation_with_target(df)
    save_top_correlations_plot(df)
    save_amount_credit_distribution(df)
    save_target_by_categorical_features(df)

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()
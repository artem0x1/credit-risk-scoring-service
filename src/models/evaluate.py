import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve


def build_model_comparison_table(results: list[dict]) -> pd.DataFrame:
    """
    Build sorted model comparison table.
    """
    return (
        pd.DataFrame(results)
        .drop_duplicates(subset=["model"], keep="last")
        .sort_values("valid_roc_auc", ascending=False)
        .reset_index(drop=True)
    )


def plot_roc_curve(
    y_true,
    y_proba,
    auc_value: float,
    title: str,
    save_path=None,
):
    """
    Plot ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random model")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_precision_recall_curve(
    y_true,
    y_proba,
    pr_auc_value: float,
    title: str,
    save_path=None,
):
    """
    Plot Precision-Recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc_value:.4f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def create_feature_importance_table(model) -> pd.DataFrame:
    """
    Create LightGBM feature importance table.
    """
    feature_importance = pd.DataFrame(
        {
            "feature": model.feature_name(),
            "importance_split": model.feature_importance(importance_type="split"),
            "importance_gain": model.feature_importance(importance_type="gain"),
        }
    )

    return (
        feature_importance
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 30,
    save_path=None,
):
    """
    Plot top feature importances by gain.
    """
    top_features = (
        feature_importance
        .head(top_n)
        .sort_values("importance_gain", ascending=True)
    )

    plt.figure(figsize=(10, 8))
    plt.barh(
        top_features["feature"],
        top_features["importance_gain"],
    )

    plt.xlabel("Importance gain")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Features by Gain")
    plt.grid(axis="x", alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
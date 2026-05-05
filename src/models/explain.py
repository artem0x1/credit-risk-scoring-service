import pandas as pd
import matplotlib.pyplot as plt


def create_lightgbm_feature_importance(model) -> pd.DataFrame:
    """
    Create LightGBM feature importance table.

    Returns importance by:
    - split: how often feature was used in trees;
    - gain: how much feature improved model quality.
    """
    importance = pd.DataFrame(
        {
            "feature": model.feature_name(),
            "importance_split": model.feature_importance(importance_type="split"),
            "importance_gain": model.feature_importance(importance_type="gain"),
        }
    )

    importance = (
        importance
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )

    return importance


def plot_lightgbm_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 30,
    save_path=None,
):
    """
    Plot top LightGBM features by gain.
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
    plt.title(f"Top {top_n} LightGBM Features by Gain")
    plt.grid(axis="x", alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def get_top_features(
    feature_importance: pd.DataFrame,
    top_n: int = 30,
) -> list[str]:
    """
    Return top feature names by importance gain.
    """
    return (
        feature_importance
        .sort_values("importance_gain", ascending=False)
        .head(top_n)["feature"]
        .tolist()
    )
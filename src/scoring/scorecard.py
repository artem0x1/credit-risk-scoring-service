import numpy as np
import pandas as pd


def get_scorecard_scaling_params(
    base_score: int = 600,
    base_odds: int = 50,
    pdo: int = 20,
) -> tuple[float, float]:
    """
    Calculate scorecard scaling parameters.

    Parameters
    ----------
    base_score : int
        Score assigned to base odds.
    base_odds : int
        Good/bad odds at base score.
    pdo : int
        Points to Double Odds.

    Returns
    -------
    factor : float
        Scorecard factor.
    offset : float
        Scorecard offset.
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    return factor, offset


def predict_score(
    model,
    X_woe: pd.DataFrame,
    factor: float,
    offset: float,
) -> np.ndarray:
    """
    Convert logistic regression logit into credit score.

    Higher score means lower default risk.
    """
    logit = model.decision_function(X_woe)
    score = offset - factor * logit

    return score


def build_scorecard_table(
    model,
    X_woe: pd.DataFrame,
    woe_table: pd.DataFrame,
    selected_features: list[str],
    factor: float,
    offset: float,
) -> pd.DataFrame:
    """
    Build scorecard table with points for every feature bin.

    Parameters
    ----------
    model
        Trained logistic regression model.
    X_woe : pd.DataFrame
        WOE-transformed train dataframe.
    woe_table : pd.DataFrame
        WOE table for selected features.
    selected_features : list[str]
        Features used in model.
    factor : float
        Scorecard factor.
    offset : float
        Scorecard offset.

    Returns
    -------
    scorecard_table : pd.DataFrame
        Table with points by feature bin.
    """
    coef_df = pd.DataFrame(
        {
            "feature": X_woe.columns,
            "coef": model.coef_[0],
        }
    )

    scorecard_table = woe_table.merge(
        coef_df,
        on="feature",
        how="left",
    )

    n_features = len(selected_features)
    intercept_points = offset - factor * model.intercept_[0]

    scorecard_table["points"] = (
        -factor * scorecard_table["coef"] * scorecard_table["woe"]
        + intercept_points / n_features
    )

    scorecard_table = scorecard_table[
        [
            "feature",
            "bin",
            "total",
            "good",
            "bad",
            "bad_rate",
            "woe",
            "coef",
            "points",
            "iv_component",
        ]
    ]

    return scorecard_table


def create_score_bands(
    scores: np.ndarray,
    y_true: pd.Series,
    q: int = 10,
) -> pd.DataFrame:
    """
    Create score bands and calculate bad rate per band.

    Parameters
    ----------
    scores : np.ndarray
        Credit scores.
    y_true : pd.Series
        True binary target.
    q : int
        Number of quantile bands.

    Returns
    -------
    score_band_summary : pd.DataFrame
        Score band summary with bad rate.
    """
    score_df = pd.DataFrame(
        {
            "score": scores,
            "target": y_true.values,
        }
    )

    score_df["score_band"] = pd.qcut(
        score_df["score"],
        q=q,
        duplicates="drop",
    )

    score_band_summary = (
        score_df.groupby("score_band", observed=True)
        .agg(
            total=("target", "count"),
            bad=("target", "sum"),
            avg_score=("score", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .reset_index()
    )

    score_band_summary["good"] = (
        score_band_summary["total"] - score_band_summary["bad"]
    )

    score_band_summary["bad_rate"] = (
        score_band_summary["bad"] / score_band_summary["total"]
    )

    score_band_summary = score_band_summary[
        [
            "score_band",
            "total",
            "good",
            "bad",
            "bad_rate",
            "avg_score",
            "min_score",
            "max_score",
        ]
    ]

    return score_band_summary
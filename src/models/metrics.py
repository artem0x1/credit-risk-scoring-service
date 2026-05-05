from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def calculate_binary_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Calculate binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : np.ndarray
        Predicted probabilities for positive class.
    threshold : float
        Threshold to convert probabilities to binary predictions.

    Returns
    -------
    metrics : dict
        Classification metrics.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "approval_rate": float(1 - y_pred.mean()),
        "predicted_bad_rate": float(y_pred.mean()),
    }

    return metrics


def calculate_threshold_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    thresholds: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Calculate precision, recall and F1 for multiple thresholds.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "approval_rate": float(1 - y_pred.mean()),
                "predicted_bad_rate": float(y_pred.mean()),
            }
        )

    return pd.DataFrame(rows)


def get_best_threshold_by_f1(threshold_metrics: pd.DataFrame) -> float:
    """
    Select threshold with maximum F1-score.
    """
    return float(
        threshold_metrics.loc[
            threshold_metrics["f1"].idxmax(),
            "threshold",
        ]
    )
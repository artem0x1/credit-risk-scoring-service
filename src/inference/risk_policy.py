def assign_risk_grade(pd_value: float) -> str:
    """
    Assign risk grade based on predicted probability of default.

    Parameters
    ----------
    pd_value : float
        Predicted probability of default.

    Returns
    -------
    risk_grade : str
        LOW, MEDIUM, HIGH or VERY_HIGH.
    """
    if pd_value < 0.20:
        return "LOW"
    elif pd_value < 0.40:
        return "MEDIUM"
    elif pd_value < 0.65:
        return "HIGH"
    return "VERY_HIGH"


def make_credit_decision(
    pd_value: float,
    threshold: float,
) -> str:
    """
    Convert probability of default into business decision.

    Parameters
    ----------
    pd_value : float
        Predicted probability of default.
    threshold : float
        Reject threshold.

    Returns
    -------
    decision : str
        APPROVE or REJECT.
    """
    if pd_value >= threshold:
        return "REJECT"
    return "APPROVE"


def apply_risk_policy(
    pd_value: float,
    threshold: float,
) -> dict:
    """
    Apply full risk policy to predicted probability.
    """
    return {
        "probability_of_default": float(pd_value),
        "threshold": float(threshold),
        "risk_grade": assign_risk_grade(pd_value),
        "decision": make_credit_decision(pd_value, threshold),
    }
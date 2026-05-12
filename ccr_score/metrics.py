"""
Metrics for CCR-Score evaluation.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro")


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def rejection_rate(ccr_score: np.ndarray) -> float:
    accepted = ccr_score >= 0
    return 1.0 - accepted.mean()


def expected_calibration_error(
    confidence: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error.

    Args:
        confidence: predicted confidence, shape [N].
        correctness: 1 if prediction is correct, else 0, shape [N].
    """

    confidence = np.asarray(confidence)
    correctness = np.asarray(correctness)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        mask = (confidence > lower) & (confidence <= upper)

        if mask.sum() == 0:
            continue

        bin_confidence = confidence[mask].mean()
        bin_accuracy = correctness[mask].mean()
        bin_weight = mask.mean()

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def risk_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute risk-coverage curve.

    Higher score means more reliable.
    For CCR-Score, use score = ccr_score.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    score = np.asarray(score)

    order = np.argsort(score)[::-1]

    y_true = y_true[order]
    y_pred = y_pred[order]

    risks = []
    coverages = []

    for k in range(1, len(y_true) + 1):
        yt = y_true[:k]
        yp = y_pred[:k]

        risk = 1.0 - accuracy_score(yt, yp)
        coverage = k / len(y_true)

        risks.append(risk)
        coverages.append(coverage)

    return np.asarray(coverages), np.asarray(risks)


def area_under_risk_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
) -> float:
    coverages, risks = risk_coverage_curve(y_true, y_pred, score)
    return float(np.trapz(risks, coverages))


def accepted_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ccr_score: np.ndarray,
) -> dict:
    accepted = ccr_score >= 0

    if accepted.sum() == 0:
        return {
            "coverage": 0.0,
            "rejection_rate": 1.0,
            "accepted_accuracy": np.nan,
            "accepted_macro_f1": np.nan,
        }

    return {
        "coverage": float(accepted.mean()),
        "rejection_rate": float(1.0 - accepted.mean()),
        "accepted_accuracy": accuracy(y_true[accepted], y_pred[accepted]),
        "accepted_macro_f1": macro_f1(y_true[accepted], y_pred[accepted]),
    }


# if __name__ == "__main__":
#
#     rng = np.random.default_rng(42)
#
#     y_true = rng.integers(0, 11, size=100)
#     y_pred = rng.integers(0, 11, size=100)
#     confidence = rng.random(100)
#     ccr_score = rng.normal(size=100)
#
#     correctness = (y_true == y_pred).astype(float)
#
#     print("Macro-F1:", macro_f1(y_true, y_pred))
#     print("Accuracy:", accuracy(y_true, y_pred))
#     print("ECE:", expected_calibration_error(confidence, correctness))
#     print("AURC:", area_under_risk_coverage(y_true, y_pred, ccr_score))
#     print("Accepted metrics:", accepted_metrics(y_true, y_pred, ccr_score))
#
#     print("Metrics test passed.")
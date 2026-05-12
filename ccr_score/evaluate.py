"""
Evaluation for CCR-Score selective prediction.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from dataset import RHBDBDataset, BeatTransform
from metrics import (
    accuracy,
    macro_f1,
    expected_calibration_error,
    area_under_risk_coverage,
    accepted_metrics,
)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_h5: str,
    output_dir: str,
    device: str,
    batch_size: int = 256,
    num_workers: int = 4,
):
    """
    Evaluate CCR-Score selective prediction performance.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = RHBDBDataset(test_h5, transform=BeatTransform())

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []
    all_confidence = []
    all_scores = []

    for batch in loader:

        beats = batch["beat"].to(device, dtype=torch.float32)
        labels = batch["label"].numpy()

        logits, scores = model(beats)

        probs = F.softmax(logits, dim=1)

        confidence, predictions = torch.max(probs, dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_confidence.extend(confidence.cpu().numpy().tolist())
        all_scores.extend(scores.view(-1).cpu().numpy().tolist())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)
    confidence = np.asarray(all_confidence)
    ccr_score = np.asarray(all_scores)

    correctness = (y_true == y_pred).astype(float)

    metrics = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "ece": expected_calibration_error(confidence, correctness),
        "aurc": area_under_risk_coverage(
            y_true,
            y_pred,
            ccr_score,
        ),
    }

    metrics.update(
        accepted_metrics(
            y_true,
            y_pred,
            ccr_score,
        )
    )

    report_all = classification_report(
        y_true,
        y_pred,
        digits=4,
    )

    accepted = ccr_score >= 0

    if accepted.sum() > 0:
        report_accepted = classification_report(
            y_true[accepted],
            y_pred[accepted],
            digits=4,
            zero_division=0,
        )
    else:
        report_accepted = "No accepted beats. The model rejected all samples.\n"

    # Save metrics
    with open(output_dir / "metrics.txt", "w") as f:

        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")

        f.write("\n")
        f.write("=== Full evaluation ===\n")
        f.write(report_all)

        f.write("\n")
        f.write("=== Accepted beats only ===\n")
        f.write(report_accepted)

    # Save prediction table
    df = pd.DataFrame({
        "true_label": y_true,
        "pred_label": y_pred,
        "softmax_confidence": confidence,
        "ccr_score": ccr_score,
        "accepted": accepted,
    })

    df.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )

    print("\nEvaluation summary")
    print("------------------")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics


# if __name__ == "__main__":
#
#     from model import build_model
#     from utils import setup_seed
#
#     setup_seed(42)
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     model = build_model()
#
#     # Random initialized model test
#     evaluate(
#         model=model,
#         test_h5="../../ecg-evidence-graph/ccr_score/beat_samples.h5",
#         output_dir="results/test_evaluate",
#         device=device,
#         batch_size=8,
#     )
#
#     print("Evaluation test passed.")
"""
Inference for CCR-Score.

This script exports:
- predicted labels
- softmax confidence
- CCR-Score
- accepted/rejected status
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import RHBDBDataset, BeatTransform
from model import build_model
from utils import load_checkpoint, setup_seed


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    h5_path: str,
    output_csv: str,
    device: str,
    batch_size: int = 256,
    num_workers: int = 4,
):
    """
    Export beat-level CCR inference results.
    """

    dataset = RHBDBDataset(
        h5_path,
        transform=BeatTransform(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model.to(device)
    model.eval()

    beat_ids = []
    true_labels = []
    pred_labels = []
    confidences = []
    ccr_scores = []
    accepted = []

    current_index = 0

    for batch in loader:

        beats = batch["beat"].to(device, dtype=torch.float32)
        labels = batch["label"]

        logits, scores = model(beats)

        probs = F.softmax(logits, dim=1)

        confidence, predictions = torch.max(probs, dim=1)

        scores = scores.view(-1)

        batch_size_actual = beats.shape[0]

        beat_ids.extend(
            list(range(current_index, current_index + batch_size_actual))
        )

        current_index += batch_size_actual

        true_labels.extend(labels.numpy().tolist())
        pred_labels.extend(predictions.cpu().numpy().tolist())
        confidences.extend(confidence.cpu().numpy().tolist())
        ccr_scores.extend(scores.cpu().numpy().tolist())
        accepted.extend((scores >= 0).cpu().numpy().tolist())

    df = pd.DataFrame({
        "beat_id": beat_ids,
        "true_label": true_labels,
        "pred_label": pred_labels,
        "softmax_confidence": confidences,
        "ccr_score": ccr_scores,
        "accepted": accepted,
    })

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)

    print(f"Inference results saved to: {output_csv}")
    print(f"Number of beats: {len(df)}")
    print(f"Accepted beats: {df['accepted'].sum()}")
    print(f"Rejected beats: {(~df['accepted']).sum()}")

    return df


# if __name__ == "__main__":
#
#     setup_seed(2024)
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     model = build_model()
#
#     # Optional checkpoint loading
#     checkpoint_path = "results/ccr_score/ccr_epoch_1.pth"
#
#     if Path(checkpoint_path).exists():
#
#         model = load_checkpoint(
#             model,
#             checkpoint_path,
#             device=device,
#         )
#
#         print(f"Loaded checkpoint: {checkpoint_path}")
#
#     else:
#         print("Checkpoint not found. Using randomly initialized model.")
#
#     run_inference(
#         model=model,
#         h5_path="../../ecg-evidence-graph/ccr_score/beat_samples.h5",
#         output_csv="results/test_inference/predictions.csv",
#         device=device,
#         batch_size=8,
#     )
#
#     print("Inference test passed.")
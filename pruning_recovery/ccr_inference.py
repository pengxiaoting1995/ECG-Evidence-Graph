"""
CCR inference for patient-level ECG recordings.

This script only exports model-derived quantities.
Reliability filtering (r >= 0) is performed later in the pruning pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ccr_score.model import build_model
from ccr_score.utils import load_checkpoint
from patient_dataset import PatientBeatDataset


@torch.no_grad()
def run_ccr_inference(
    h5_path: str,
    checkpoint_path: str,
    output_csv: str,
    device: str,
    batch_size: int = 256,
    num_workers: int = 4,
):
    """
    Run CCR inference on a single patient ECG recording.
    """

    dataset = PatientBeatDataset(h5_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_model()
    model = load_checkpoint(
        model,
        checkpoint_path,
        device=device,
    )

    model.to(device)
    model.eval()

    patient_ids = []
    beat_ids = []
    beat_indices = []

    pred_labels = []
    p_normal_list = []
    p_non_list = []
    confidence_list = []
    ccr_scores = []

    embedding_list = []

    for batch in loader:

        beats = batch["beat"].to(device, dtype=torch.float32)

        z = model.extractor(beats)
        logits = model.predictor(z)
        scores = model.rejector(z).view(-1)

        probs = F.softmax(logits, dim=1)
        confidence, predictions = torch.max(probs, dim=1)

        p_normal = probs[:, 0]
        p_non = 1.0 - p_normal

        patient_ids.extend(batch["patient_id"])
        beat_ids.extend(batch["beat_id"].numpy().tolist())
        beat_indices.extend(batch["beat_index"].numpy().tolist())

        pred_labels.extend(predictions.cpu().numpy().tolist())
        p_normal_list.extend(p_normal.cpu().numpy().tolist())
        p_non_list.extend(p_non.cpu().numpy().tolist())
        confidence_list.extend(confidence.cpu().numpy().tolist())
        ccr_scores.extend(scores.cpu().numpy().tolist())

        embedding_list.append(z.cpu().numpy())

    embeddings = np.concatenate(embedding_list, axis=0)

    metadata_df = pd.DataFrame({
        "patient_id": patient_ids,
        "beat_id": beat_ids,
        "beat_index": beat_indices,
        "pred_label": pred_labels,
        "p_normal": p_normal_list,
        "p_non": p_non_list,
        "softmax_confidence": confidence_list,
        "ccr_score": ccr_scores,
    })

    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f"z_{i}" for i in range(embeddings.shape[1])],
    )

    df = pd.concat(
        [metadata_df, embedding_df],
        axis=1,
    )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)

    print(f"Inference saved to: {output_csv}")
    print(f"Number of beats: {len(df)}")
    print(f"Mean CCR-Score: {df['ccr_score'].mean():.4f}")

    return df


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = "../ccr_score/results/ccr_score/ccr_epoch_1.pth"

    input_dir = Path("sample")
    output_dir = Path("results/pruning_recovery")

    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(input_dir.glob("*.h5"))

    print(f"Found {len(h5_files)} HDF5 files.")

    for h5_path in h5_files:

        patient_id = h5_path.stem

        print("\n" + "=" * 60)
        print(f"Running CCR inference for: {patient_id}")

        run_ccr_inference(
            h5_path=str(h5_path),
            checkpoint_path=checkpoint_path,
            output_csv=str(output_dir / f"{patient_id}.csv"),
            device=device,
            batch_size=32,
        )

    print("\nAll CCR inference tests passed.")
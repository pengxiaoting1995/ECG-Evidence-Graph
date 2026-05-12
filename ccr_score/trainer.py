"""
Training utilities for CCR-Score learning.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import RHBDBDataset, BeatTransform
from loss import compute_classification_error, ccr_loss
from utils import save_checkpoint


def pretrain_classifier(
    model: torch.nn.Module,
    train_h5: str,
    output_dir: str,
    device: str,
    batch_size: int = 256,
    lr: float = 1e-3,
    epochs: int = 0,
    num_workers: int = 4,
    logger=None,
) -> torch.nn.Module:
    """
    Optional classifier-only pretraining.

    During this stage, the rejector is frozen.
    """

    if epochs <= 0:
        if logger:
            logger.info("Skip classifier pretraining.")
        return model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = RHBDBDataset(train_h5, transform=BeatTransform())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    model.to(device)
    model.train()

    for param in model.rejector.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    for epoch in range(epochs):

        losses = []

        for batch in loader:

            beats = batch["beat"].to(device, dtype=torch.float32)
            labels = batch["label"].to(device, dtype=torch.long)

            optimizer.zero_grad()

            logits, _ = model(beats)

            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if logger:
            logger.info(
                f"[Classifier pretrain] Epoch {epoch + 1}/{epochs} | "
                f"Loss={np.mean(losses):.4f}"
            )

        save_checkpoint(
            model,
            output_dir / f"pretrain_epoch_{epoch + 1}.pth",
        )

    for param in model.rejector.parameters():
        param.requires_grad = True

    return model


def train_ccr(
    model: torch.nn.Module,
    train_h5: str,
    output_dir: str,
    device: str,
    batch_size: int = 256,
    lr: float = 1e-3,
    epochs: int = 20,
    rejection_cost: float = 0.375,
    num_workers: int = 4,
    logger=None,
) -> torch.nn.Module:
    """
    Train the CCR-Score model using sigmoid rejection loss.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = RHBDBDataset(train_h5, transform=BeatTransform())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    model.to(device)
    model.train()

    optimizer_predictor = torch.optim.Adam(
        model.predictor.parameters(),
        lr=lr,
        weight_decay=1e-3,
    )

    optimizer_rejector = torch.optim.Adam(
        model.rejector.parameters(),
        lr=lr,
        weight_decay=1e-3,
    )

    optimizer_extractor = torch.optim.Adam(
        model.extractor.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    for epoch in range(epochs):

        losses = []
        mean_errors = []
        mean_scores = []

        for batch in loader:

            beats = batch["beat"].to(device, dtype=torch.float32)
            labels = batch["label"].to(device, dtype=torch.long)

            optimizer_extractor.zero_grad()
            optimizer_predictor.zero_grad()
            optimizer_rejector.zero_grad()

            logits, scores = model(beats)

            error = compute_classification_error(logits, labels)

            loss = ccr_loss(
                classification_error=error,
                ccr_score=scores,
                rejection_cost=rejection_cost,
            )

            if torch.isnan(loss):
                raise RuntimeError("CCR training stopped because loss became NaN.")

            loss.backward()

            optimizer_extractor.step()
            optimizer_predictor.step()
            optimizer_rejector.step()

            losses.append(loss.item())
            mean_errors.append(error.mean().item())
            mean_scores.append(scores.mean().item())

        if logger:
            logger.info(
                f"[CCR train] Epoch {epoch + 1}/{epochs} | "
                f"Loss={np.mean(losses):.4f} | "
                f"Mean error={np.mean(mean_errors):.4f} | "
                f"Mean CCR={np.mean(mean_scores):.4f}"
            )

        save_checkpoint(
            model,
            output_dir / f"ccr_epoch_{epoch + 1}.pth",
        )

    return model


# if __name__ == "__main__":
#
#     from model import build_model
#     from utils import setup_seed, create_logger
#
#     setup_seed(42)
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     model = build_model()
#
#     logger = create_logger("results/test_trainer")
#
#     # This test expects the sample HDF5 file to exist.
#     sample_h5 = "../../ecg-evidence-graph/ccr_score/beat_samples.h5"
#
#     model = pretrain_classifier(
#         model=model,
#         train_h5=sample_h5,
#         output_dir="results/test_trainer",
#         device=device,
#         batch_size=8,
#         lr=1e-3,
#         epochs=1,
#         logger=logger,
#     )
#
#     model = train_ccr(
#         model=model,
#         train_h5=sample_h5,
#         output_dir="results/test_trainer",
#         device=device,
#         batch_size=8,
#         lr=1e-3,
#         epochs=1,
#         rejection_cost=0.375,
#         logger=logger,
#     )
#
#     print("Trainer test passed.")
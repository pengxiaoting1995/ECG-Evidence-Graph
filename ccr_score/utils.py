"""
Utility functions for CCR-Score training and evaluation.
"""

import logging
import random
from pathlib import Path

import numpy as np
import torch


def setup_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(save_dir: str | Path) -> logging.Logger:
    """Create training logger."""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ccr_score")
    logger.setLevel(logging.INFO)

    # Avoid duplicated handlers
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(save_dir / "train.log")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_checkpoint(
    model: torch.nn.Module,
    save_path: str | Path,
) -> None:
    """Save model checkpoint."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        save_path,
    )


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load model checkpoint."""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

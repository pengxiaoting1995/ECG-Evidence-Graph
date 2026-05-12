"""
CCR-Score model.

This module implements the dual-scale beat encoder, classifier head,
and reliability head used for beat-level selective prediction.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """1D convolution + batch normalization + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        return self.dropout(x) if self.dropout.p > 0 else x


class ResNetBlock(nn.Module):
    """Residual 1D convolution block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float = 0.0):
        super().__init__()
        self.shortcut = nn.MaxPool1d(kernel_size=stride)
        self.conv_layers = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, dropout=dropout),
            ConvBlock(out_channels, out_channels, kernel_size=3, dropout=dropout),
        )
        self.channel_padding = out_channels - in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        if self.channel_padding > 0:
            shortcut = F.pad(shortcut, pad=(0, 0, 0, self.channel_padding))

        out = self.conv_layers(x)

        if out.shape != shortcut.shape:
            out = F.avg_pool1d(out, kernel_size=2, stride=1)

        return out + shortcut


class DualScaleEncoder(nn.Module):
    """Dual-scale encoder for intra-beat morphology and local temporal context."""

    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(1, 32, kernel_size=3)

        blocks = []
        in_channels = 32
        for idx, stride in enumerate((1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2)):
            out_channels = (2 ** int(idx / 4)) * 32
            blocks.append(
                ResNetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    dropout=0.2,
                )
            )
            in_channels = out_channels

        self.resnet_blocks = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 16, 512)

    def _encode_branch(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.resnet_blocks(x)
        x = F.relu(x)
        return self.flatten(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [batch, 1, 375], got {tuple(x.shape)}.")
        if x.shape[1] != 1 or x.shape[2] != 375:
            raise ValueError(f"Expected input shape [batch, 1, 375], got {tuple(x.shape)}.")

        x_intra = x[:, :, 50:306]
        x_inter = x

        z_intra = self._encode_branch(x_intra)
        z_inter = self._encode_branch(x_inter)

        z = torch.cat([z_intra, z_inter], dim=1)
        return self.fc(z)


class PredictorHead(nn.Module):
    """Linear classifier head."""

    def __init__(self):
        super().__init__()
        self.select_weight = nn.Parameter(torch.rand(512) / np.sqrt(512))
        self.out = nn.Linear(512, 11)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z * self.select_weight
        return self.out(z)


class RejectorHead(nn.Module):
    """Reliability head that produces the CCR-Score."""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(512, 200)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(200, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.hidden(z)
        z = self.dropout(z)
        z = z * torch.tanh(F.softplus(z))
        return torch.tanh(self.out(z))


class CCRScoreModel(nn.Module):
    """Dual-head model for beat classification and reliability scoring."""

    def __init__(self):
        super().__init__()
        self.extractor = DualScaleEncoder()
        self.predictor = PredictorHead()
        self.rejector = RejectorHead()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.extractor(x)
        logits = self.predictor(z)
        ccr_score = self.rejector(z)
        return logits, ccr_score


def build_model() -> CCRScoreModel:
    return CCRScoreModel()


def _strip_module_prefix(state_dict: dict) -> dict:
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    return new_state_dict


def load_pretrained_extractor(model: CCRScoreModel, checkpoint_path: str | Path, freeze: bool = False) -> CCRScoreModel:
    """Load checkpoint weights into the encoder only."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if any(key.startswith("module.") for key in state_dict):
        state_dict = _strip_module_prefix(state_dict)

    extractor_state = model.extractor.state_dict()
    matched_state = {
        key: value
        for key, value in state_dict.items()
        if key in extractor_state and extractor_state[key].shape == value.shape
    }

    extractor_state.update(matched_state)
    model.extractor.load_state_dict(extractor_state, strict=False)

    if freeze:
        for param in model.extractor.parameters():
            param.requires_grad = False

    return model



# if __name__ == "__main__":
#
#     model = build_model()
#     model.eval()
#
#     # Simulated ECG batch
#     x = torch.randn(8, 1, 375)
#
#     with torch.no_grad():
#         logits, ccr_score = model(x)
#
#     print("Input shape:", x.shape)
#     print("Logits shape:", logits.shape)
#     print("CCR-Score shape:", ccr_score.shape)
#
#     assert logits.shape == (8, 11)
#     assert ccr_score.shape == (8, 1)
#
#     print("Model test passed.")
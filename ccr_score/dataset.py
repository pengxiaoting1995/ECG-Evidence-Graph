"""
Dataset utilities for CCR-Score learning.

Expected HDF5 format:
    beats   : array-like, shape [N, 1, 375]
    symbols : array-like, shape [N]
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class BeatTransform:
    """Return full dual-scale input window."""

    def __call__(self, sample: dict) -> dict:
        return sample


class RHBDBDataset(Dataset):
    """R-HBDB beat-level dataset with 11 consolidated beat classes."""

    label_mapping = {
        "N": 0,
        "L": 1, "R": 1, "B": 1,
        "V": 2, "r": 2,
        "A": 3, "S": 3, "J": 3, "a": 3, "x": 3,
        "s": 4, "T": 4,
        "/": 5,
        "F": 6, "f": 6,
        "E": 7, "e": 7, "j": 7, "n": 7,
        "+": 8,
        "Q": 9,
        "~": 10, "|": 10,
    }

    excluded_symbols = {"!", "[", "]", "?", '"'}

    class_names = [
        "Normal beat",
        "Bundle Branch Block Beat",
        "Ventricular Premature Beat",
        "Supraventricular Premature Beat",
        "ST-T Segment Changes",
        "Paced beat",
        "Fusion Beats",
        "Escape Beats",
        "Rhythm change",
        "Unclassifiable Beat",
        "Noise/Artifact",
    ]

    def __init__(self, h5_path: str | Path, transform=None):
        self.h5_path = Path(h5_path)
        self.transform = transform

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        self.beats, self.labels = self._load_h5()

    def _load_h5(self) -> tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.h5_path, "r") as f:
            if "beats" not in f or "symbols" not in f:
                raise KeyError("HDF5 file must contain keys: 'beats' and 'symbols'.")

            beats = f["beats"][:]
            symbols = f["symbols"][:].astype(str)

        valid_indices = [
            i for i, s in enumerate(symbols)
            if s not in self.excluded_symbols and s in self.label_mapping
        ]

        beats = beats[valid_indices]
        labels = np.array(
            [self.label_mapping[symbols[i]] for i in valid_indices],
            dtype=np.int64,
        )

        if beats.ndim != 3 or beats.shape[1] != 1 or beats.shape[2] != 375:
            raise ValueError(f"Expected beats shape [N, 1, 375], got {beats.shape}.")

        return beats.astype(np.float32), labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        sample = {
            "beat": torch.from_numpy(self.beats[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def class_counts(self) -> dict:
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


# if __name__ == "__main__":
#
#     dataset = RHBDBDataset(
#         h5_path="../../ecg-evidence-graph/ccr_score/beat_samples.h5",
#         transform=BeatTransform(),
#     )
#
#     print("Number of beats:", len(dataset))
#     print("Class counts:", dataset.class_counts())
#
#     sample = dataset[0]
#
#     print("Beat shape:", sample["beat"].shape)
#     print("Label:", sample["label"].item())
#
#     assert sample["beat"].shape == (1, 375)
#
#     print("Dataset test passed.")
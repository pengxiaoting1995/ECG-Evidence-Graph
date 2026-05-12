"""
Patient-level HDF5 dataset for pruning and recovery.

Expected HDF5 format:
    beats      : [N, 1, 375]
Optional:
    rpeaks     : [N]
    timestamps : [N]
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PatientBeatDataset(Dataset):
    """Dataset for a single patient ECG recording."""

    def __init__(self, h5_path: str | Path):
        self.h5_path = Path(h5_path)

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        self.patient_id = self.h5_path.stem

        self.beats, self.beat_index, self.rpeaks, self.timestamps = self._load_h5()

    def _load_h5(self):
        with h5py.File(self.h5_path, "r") as f:
            if "beats" not in f:
                raise KeyError("HDF5 file must contain key: 'beats'.")

            beats = f["beats"][:].astype(np.float32)

            if beats.ndim != 3 or beats.shape[1] != 1 or beats.shape[2] != 375:
                raise ValueError(f"Expected beats shape [N, 1, 375], got {beats.shape}.")

            n_beats = beats.shape[0]

            beat_index = (
                f["beat_index"][:].astype(np.int64)
                if "beat_index" in f
                else np.arange(n_beats, dtype=np.int64)
            )

            rpeaks = (
                f["rpeaks"][:].astype(np.int64)
                if "rpeaks" in f
                else None
            )

            timestamps = (
                f["timestamps"][:].astype(np.float32)
                if "timestamps" in f
                else None
            )

        return beats, beat_index, rpeaks, timestamps

    def __len__(self) -> int:
        return self.beats.shape[0]

    def __getitem__(self, idx: int) -> dict:
        sample = {
            "beat": torch.from_numpy(self.beats[idx]),
            "beat_id": idx,
            "beat_index": int(self.beat_index[idx]),
            "patient_id": self.patient_id,
        }

        if self.rpeaks is not None:
            sample["rpeak"] = int(self.rpeaks[idx])

        if self.timestamps is not None:
            sample["timestamp"] = float(self.timestamps[idx])

        return sample


# if __name__ == "__main__":
#
#     sample_files = [
#         "../../ecg-evidence-graph/pruning_recovery/sample/patient_001.h5",
#         "../../ecg-evidence-graph/pruning_recovery/sample/patient_002.h5",
#         "../../ecg-evidence-graph/pruning_recovery/sample//patient_003.h5",
#         "../../ecg-evidence-graph/pruning_recovery/sample/patient_004.h5",
#         "../../ecg-evidence-graph/pruning_recovery/sample/patient_005.h5",
#     ]
#
#     for h5_path in sample_files:
#
#         print("\n" + "=" * 60)
#         print(f"Testing: {h5_path}")
#
#         dataset = PatientBeatDataset(h5_path=h5_path)
#
#         print("Patient ID:", dataset.patient_id)
#         print("Number of beats:", len(dataset))
#
#         sample = dataset[0]
#
#         print("Sample keys:", list(sample.keys()))
#         print("Beat shape:", sample["beat"].shape)
#         print("Beat index:", sample["beat_index"])
#
#         if "rpeak" in sample:
#             print("R peak:", sample["rpeak"])
#
#         if "timestamp" in sample:
#             print("Timestamp:", sample["timestamp"])
#
#         assert sample["beat"].shape == (1, 375)
#
#     print("\nAll patient dataset tests passed.")
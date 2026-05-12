"""
Create fixed patient-level cross-validation splits.

Input:
    labels.txt with columns:
        patient_id,label

Output:
    folds.csv with columns:
        patient_id,label,fold
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_folds(
    labels_path: str,
    output_path: str,
    n_splits: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create stratified patient-level folds.
    """

    labels_path = Path(labels_path)
    output_path = Path(output_path)

    if not labels_path.exists():
        raise FileNotFoundError(f"labels file not found: {labels_path}")

    df = pd.read_csv(labels_path)

    required_columns = {"patient_id", "label"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in labels file: {missing}")

    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["label"] = df["label"].astype(int)

    if df["patient_id"].duplicated().any():
        duplicated = df.loc[df["patient_id"].duplicated(), "patient_id"].tolist()
        raise ValueError(f"Duplicated patient_id found: {duplicated[:5]}")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    df["fold"] = -1

    for fold, (_, test_idx) in enumerate(
        splitter.split(df["patient_id"], df["label"])
    ):
        df.loc[test_idx, "fold"] = fold

    if (df["fold"] < 0).any():
        raise RuntimeError("Some patients were not assigned to any fold.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved folds to: {output_path}")
    print(df["fold"].value_counts().sort_index())

    return df


if __name__ == "__main__":

    create_folds(
        labels_path="sample/labels.txt",
        output_path="sample/splits/folds.csv",
        n_splits=10,
        random_state=42,
    )

    print("Split generation test passed.")
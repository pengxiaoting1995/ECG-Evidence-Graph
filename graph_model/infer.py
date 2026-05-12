"""
External inference with fold-aligned graph and meta models.

For each fold:
    external graph
    -> graph fold_i checkpoint
    -> graph probabilities
    -> uncertainty features
    -> meta fold_i model
    -> meta probabilities

Final prediction:
    average meta probabilities across folds.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import PatientGraphDataset, collate_graphs
from model import EvidenceGraphTransformer
from meta_classifier import add_uncertainty_features


class ExternalGraphDataset(PatientGraphDataset):
    """
    External graph dataset.

    Loads all patients from labels_path and ignores folds.
    """

    def __init__(
        self,
        graph_root: str,
        labels_path: str,
    ):
        self.graph_root = Path(graph_root)
        self.labels_path = Path(labels_path)
        self.folds_path = None
        self.fold = None
        self.mode = "external"

        self.patient_table = self._load_patient_table()

    def _load_patient_table(self) -> pd.DataFrame:

        if not self.graph_root.exists():
            raise FileNotFoundError(f"Graph root not found: {self.graph_root}")

        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        labels = pd.read_csv(self.labels_path)

        required_cols = {"patient_id", "label"}
        missing = required_cols - set(labels.columns)

        if missing:
            raise ValueError(f"Missing columns in labels file: {missing}")

        labels["patient_id"] = labels["patient_id"].astype(str)
        labels["label"] = labels["label"].astype(int)

        return labels.reset_index(drop=True)


def load_graph_model(
    checkpoint_path: str,
    device: str,
    feature_dim: int = 512,
    hidden_dim: int = 1024,
    num_layers: int = 2,
    num_neighbours: int = 4,
    dropout: float = 0.5,
):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Graph checkpoint not found: {checkpoint_path}")

    model = EvidenceGraphTransformer(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
        num_layers=num_layers,
        num_neighbours=num_neighbours,
        dropout=dropout,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict_graph_model(
    model,
    loader,
    device,
) -> pd.DataFrame:
    """
    Predict graph-level probabilities with one graph model.
    """

    patient_ids = []
    true_labels = []
    p1_list = []

    for batch in loader:

        x = batch["x"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        batch_index = batch["batch"].to(device)

        logits = model(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch_index,
        )

        p1 = torch.softmax(logits, dim=1)[:, 1]

        patient_ids.extend(batch["patient_id"])
        true_labels.extend(batch["y"].numpy().tolist())
        p1_list.extend(p1.cpu().numpy().tolist())

    return pd.DataFrame({
        "patient_id": patient_ids,
        "true_label": true_labels,
        "p0": 1.0 - np.asarray(p1_list),
        "p1": p1_list,
    })


def predict_meta_model(
    graph_pred_df: pd.DataFrame,
    meta_model_path: str,
) -> np.ndarray:
    """
    Apply one fold-specific meta-classifier to graph predictions.
    """

    meta_model_path = Path(meta_model_path)

    if not meta_model_path.exists():
        raise FileNotFoundError(f"Meta model not found: {meta_model_path}")

    bundle = joblib.load(meta_model_path)

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    low_conf_threshold = bundle.get("low_conf_threshold", 0.6)

    feature_df = add_uncertainty_features(
        graph_pred_df,
        low_conf_threshold=low_conf_threshold,
    )

    x = feature_df[feature_cols].to_numpy(dtype=np.float32)

    meta_p1 = model.predict_proba(x)[:, 1]

    return meta_p1


def run_external_inference(
    graph_root: str,
    labels_path: str,
    checkpoint_dir: str,
    meta_dir: str,
    output_csv: str,
    device: str,
    batch_size: int = 4,
    n_folds: int = 10,
    feature_dim: int = 512,
    hidden_dim: int = 1024,
    num_layers: int = 2,
    num_neighbours: int = 4,
    dropout: float = 0.5,
):
    """
    Run fold-aligned graph + meta external inference.
    """

    dataset = ExternalGraphDataset(
        graph_root=graph_root,
        labels_path=labels_path,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    final_df = None

    for fold in range(n_folds):

        print("\n" + "=" * 70)
        print(f"Running fold-aligned inference: fold {fold}")

        graph_checkpoint = (
            Path(checkpoint_dir)
            / f"fold_{fold}"
            / "checkpoint.pt"
        )

        meta_model_path = (
            Path(meta_dir)
            / f"fold_{fold}"
            / "meta_model.pkl"
        )

        graph_model = load_graph_model(
            checkpoint_path=graph_checkpoint,
            device=device,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_neighbours=num_neighbours,
            dropout=dropout,
        )

        graph_pred_df = predict_graph_model(
            model=graph_model,
            loader=loader,
            device=device,
        )

        meta_p1 = predict_meta_model(
            graph_pred_df=graph_pred_df,
            meta_model_path=meta_model_path,
        )

        fold_df = pd.DataFrame({
            "patient_id": graph_pred_df["patient_id"],
            "true_label": graph_pred_df["true_label"],
            f"graph_p1_fold_{fold}": graph_pred_df["p1"],
            f"meta_p1_fold_{fold}": meta_p1,
        })

        if final_df is None:
            final_df = fold_df
        else:
            final_df = final_df.merge(
                fold_df,
                on=["patient_id", "true_label"],
                how="inner",
            )

    graph_cols = [
        col for col in final_df.columns
        if col.startswith("graph_p1_fold_")
    ]

    meta_cols = [
        col for col in final_df.columns
        if col.startswith("meta_p1_fold_")
    ]

    final_df["graph_p1_mean"] = final_df[graph_cols].mean(axis=1)
    final_df["graph_p1_std"] = final_df[graph_cols].std(axis=1)

    final_df["meta_p1_mean"] = final_df[meta_cols].mean(axis=1)
    final_df["meta_p1_std"] = final_df[meta_cols].std(axis=1)

    final_df["meta_p0_mean"] = 1.0 - final_df["meta_p1_mean"]

    final_df["predicted_label"] = (
        final_df["meta_p1_mean"] >= 0.5
    ).astype(int)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(
        output_csv,
        index=False,
    )

    print(f"\nSaved external inference results to: {output_csv}")

    return final_df


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--graph_root", type=str, default="external_sample/graphs")
    parser.add_argument("--labels_path", type=str, default="external_sample/labels.txt")

    parser.add_argument("--checkpoint_dir", type=str, default="results/graph_model")
    parser.add_argument("--meta_dir", type=str, default="results/graph_model/meta")

    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/graph_model/external_predictions.csv",
    )

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_folds", type=int, default=10)

    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_neighbours", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_external_inference(
        graph_root=args.graph_root,
        labels_path=args.labels_path,
        checkpoint_dir=args.checkpoint_dir,
        meta_dir=args.meta_dir,
        output_csv=args.output_csv,
        device=device,
        batch_size=args.batch_size,
        n_folds=args.n_folds,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_neighbours=args.num_neighbours,
        dropout=args.dropout,
    )
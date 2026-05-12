"""
Patient-level graph dataset.

Each patient is represented by:
    graphs/patient_xxx/nodes.csv
    graphs/patient_xxx/edges.csv

Labels:
    labels.txt

Splits:
    splits/folds.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def get_embedding_columns(nodes: pd.DataFrame) -> list[str]:
    embedding_cols = [col for col in nodes.columns if col.startswith("z_")]

    if len(embedding_cols) == 0:
        raise ValueError("No embedding columns found. Expected z_0, z_1, ...")

    return embedding_cols


def encode_edge_type(edge_type: str) -> int:
    if edge_type == "temporal":
        return 0

    if edge_type == "similarity":
        return 1

    raise ValueError(f"Unknown edge_type: {edge_type}")


class PatientGraphDataset(Dataset):
    """
    Dataset for patient-level binary graph classification.
    """

    def __init__(
        self,
        graph_root: str,
        labels_path: str,
        folds_path: str,
        fold: int,
        mode: str = "train",
    ):
        self.graph_root = Path(graph_root)
        self.labels_path = Path(labels_path)
        self.folds_path = Path(folds_path)

        self.fold = fold
        self.mode = mode

        if mode not in {"train", "test"}:
            raise ValueError("mode must be 'train' or 'test'.")

        self.patient_table = self._load_patient_table()

    def _load_patient_table(self) -> pd.DataFrame:

        if not self.graph_root.exists():
            raise FileNotFoundError(f"Graph root not found: {self.graph_root}")

        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        if not self.folds_path.exists():
            raise FileNotFoundError(f"Folds file not found: {self.folds_path}")

        labels = pd.read_csv(self.labels_path)
        folds = pd.read_csv(self.folds_path)

        required_label_cols = {"patient_id", "label"}
        required_fold_cols = {"patient_id", "fold"}

        missing_labels = required_label_cols - set(labels.columns)
        missing_folds = required_fold_cols - set(folds.columns)

        if missing_labels:
            raise ValueError(f"Missing columns in labels file: {missing_labels}")

        if missing_folds:
            raise ValueError(f"Missing columns in folds file: {missing_folds}")

        labels["patient_id"] = labels["patient_id"].astype(str)
        folds["patient_id"] = folds["patient_id"].astype(str)

        table = labels.merge(
            folds[["patient_id", "fold"]],
            on="patient_id",
            how="inner",
        )

        if self.mode == "train":
            table = table[table["fold"] != self.fold]

        else:
            table = table[table["fold"] == self.fold]

        table = table.reset_index(drop=True)

        if len(table) == 0:
            raise ValueError(
                f"No patients found for fold={self.fold}, mode={self.mode}"
            )

        return table

    def __len__(self) -> int:
        return len(self.patient_table)

    def _load_graph(self, patient_id: str):

        patient_dir = self.graph_root / patient_id

        nodes_path = patient_dir / "nodes.csv"
        edges_path = patient_dir / "edges.csv"

        if not nodes_path.exists():
            raise FileNotFoundError(f"nodes.csv not found: {nodes_path}")

        if not edges_path.exists():
            raise FileNotFoundError(f"edges.csv not found: {edges_path}")

        nodes = pd.read_csv(nodes_path)
        edges = pd.read_csv(edges_path)

        embedding_cols = get_embedding_columns(nodes)

        x = torch.tensor(
            nodes[embedding_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        beat_ids = nodes["beat_id"].to_numpy()

        beat_id_to_idx = {
            beat_id: idx
            for idx, beat_id in enumerate(beat_ids)
        }

        edge_sources = []
        edge_targets = []
        edge_attrs = []

        for _, row in edges.iterrows():

            source = row["source"]
            target = row["target"]

            if source not in beat_id_to_idx:
                continue

            if target not in beat_id_to_idx:
                continue

            edge_sources.append(
                beat_id_to_idx[source]
            )

            edge_targets.append(
                beat_id_to_idx[target]
            )

            edge_attrs.append([
                float(row["weight"]),
                float(encode_edge_type(row["edge_type"])),
            ])

        if len(edge_sources) == 0:

            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

            edge_attr = torch.empty(
                (0, 2),
                dtype=torch.float32,
            )

        else:

            edge_index = torch.tensor(
                [edge_sources, edge_targets],
                dtype=torch.long,
            )

            edge_attr = torch.tensor(
                edge_attrs,
                dtype=torch.float32,
            )

        return x, edge_index, edge_attr

    def __getitem__(self, idx: int) -> dict:

        row = self.patient_table.iloc[idx]

        patient_id = row["patient_id"]
        label = int(row["label"])

        x, edge_index, edge_attr = self._load_graph(patient_id)

        return {
            "patient_id": patient_id,

            "x": x,

            "edge_index": edge_index,

            "edge_attr": edge_attr,

            "y": torch.tensor(
                label,
                dtype=torch.long,
            ),

            "num_nodes": torch.tensor(
                x.shape[0],
                dtype=torch.long,
            ),
        }


def collate_graphs(batch: list[dict]) -> dict:
    """
    Batch variable-size patient graphs.
    """

    x_list = []

    edge_index_list = []
    edge_attr_list = []

    batch_index_list = []

    y_list = []

    num_nodes_list = []

    patient_ids = []

    node_offset = 0

    for graph_idx, item in enumerate(batch):

        x = item["x"]

        edge_index = item["edge_index"]

        edge_attr = item["edge_attr"]

        n_nodes = x.shape[0]

        x_list.append(x)

        if edge_index.numel() > 0:

            edge_index_list.append(
                edge_index + node_offset
            )

            edge_attr_list.append(
                edge_attr
            )

        batch_index_list.append(
            torch.full(
                (n_nodes,),
                graph_idx,
                dtype=torch.long,
            )
        )

        y_list.append(item["y"])

        num_nodes_list.append(
            item["num_nodes"]
        )

        patient_ids.append(
            item["patient_id"]
        )

        node_offset += n_nodes

    x = torch.cat(x_list, dim=0)

    batch_index = torch.cat(
        batch_index_list,
        dim=0,
    )

    y = torch.stack(
        y_list,
        dim=0,
    )

    num_nodes = torch.stack(
        num_nodes_list,
        dim=0,
    )

    if len(edge_index_list) > 0:

        edge_index = torch.cat(
            edge_index_list,
            dim=1,
        )

        edge_attr = torch.cat(
            edge_attr_list,
            dim=0,
        )

    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

        edge_attr = torch.empty(
            (0, 2),
            dtype=torch.float32,
        )

    return {
        "patient_id": patient_ids,

        "x": x,

        "edge_index": edge_index,

        "edge_attr": edge_attr,

        "batch": batch_index,

        "y": y,

        "num_nodes": num_nodes,
    }


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    dataset = PatientGraphDataset(
        graph_root="sample/graphs",
        labels_path="sample/labels.txt",
        folds_path="sample/splits/folds.csv",
        fold=0,
        mode="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    batch = next(iter(loader))

    print("Patient IDs:", batch["patient_id"])

    print("x:", batch["x"].shape)

    print("edge_index:", batch["edge_index"].shape)

    print("edge_attr:", batch["edge_attr"].shape)

    print("batch:", batch["batch"].shape)

    print("y:", batch["y"])

    print("num_nodes:", batch["num_nodes"])

    print("Graph dataset test passed.")
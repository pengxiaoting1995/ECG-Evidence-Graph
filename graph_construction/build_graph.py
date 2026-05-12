"""
Topology-aware graph construction.

Inputs:
    anchor_set.csv
    recovered_nodes.csv

Outputs:
    nodes.csv
    edges.csv
    graph_summary.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def get_embedding_columns(df: pd.DataFrame) -> list[str]:
    embedding_cols = [col for col in df.columns if col.startswith("z_")]

    if len(embedding_cols) == 0:
        raise ValueError("No embedding columns found. Expected columns z_0, z_1, ...")

    return embedding_cols


def merge_graph_nodes(
    anchors: pd.DataFrame,
    recovered: pd.DataFrame,
) -> pd.DataFrame:
    """
    Final graph nodes are the union of anchor beats and recovered beats.
    """

    nodes = pd.concat(
        [anchors, recovered],
        axis=0,
        ignore_index=True,
    )

    nodes = (
        nodes
        .drop_duplicates(subset=["patient_id", "beat_id"])
        .sort_values("beat_index")
        .reset_index(drop=True)
    )

    return nodes


def build_temporal_edges(nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Build directed temporal edges between adjacent retained graph nodes.

    Example:
        original beats: 1, 2, 3, 4, 5
        retained nodes: 1, 5
        temporal edge: 1 -> 5
    """

    nodes = nodes.sort_values("beat_index").reset_index(drop=True)

    edges = []

    for i in range(len(nodes) - 1):
        edges.append({
            "source": nodes.loc[i, "beat_id"],
            "target": nodes.loc[i + 1, "beat_id"],
            "edge_type": "temporal",
            "weight": 1.0,
        })

    return pd.DataFrame(edges)


def build_similarity_edges_topk(
    anchors: pd.DataFrame,
    nodes: pd.DataFrame,
    k: int = 8,
) -> pd.DataFrame:
    """
    Build directed morphology-aware similarity edges.

    Each anchor beat points to its top-k most similar graph nodes.
    """

    embedding_cols = get_embedding_columns(nodes)

    anchor_embeddings = anchors[embedding_cols].to_numpy(dtype=np.float32)
    node_embeddings = nodes[embedding_cols].to_numpy(dtype=np.float32)

    anchor_ids = anchors["beat_id"].to_numpy()
    node_ids = nodes["beat_id"].to_numpy()

    sim = cosine_similarity(anchor_embeddings, node_embeddings)

    edges = []

    for i in range(len(anchors)):
        order = np.argsort(sim[i])[::-1]

        # Exclude self-edge if the anchor is also included in graph nodes.
        order = [
            j for j in order
            if node_ids[j] != anchor_ids[i]
        ]

        selected = order[:k]

        for j in selected:
            edges.append({
                "source": anchor_ids[i],
                "target": node_ids[j],
                "edge_type": "similarity",
                "weight": float(sim[i, j]),
            })

    return pd.DataFrame(edges)


def build_similarity_edges_threshold(
    anchors: pd.DataFrame,
    nodes: pd.DataFrame,
    threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Build directed morphology-aware similarity edges.

    Each anchor beat points to graph nodes whose cosine similarity exceeds
    the predefined threshold.
    """

    embedding_cols = get_embedding_columns(nodes)

    anchor_embeddings = anchors[embedding_cols].to_numpy(dtype=np.float32)
    node_embeddings = nodes[embedding_cols].to_numpy(dtype=np.float32)

    anchor_ids = anchors["beat_id"].to_numpy()
    node_ids = nodes["beat_id"].to_numpy()

    sim = cosine_similarity(anchor_embeddings, node_embeddings)

    edges = []

    for i in range(len(anchors)):
        candidate_indices = np.where(sim[i] > threshold)[0]

        candidate_indices = [
            j for j in candidate_indices
            if node_ids[j] != anchor_ids[i]
        ]

        for j in candidate_indices:
            edges.append({
                "source": anchor_ids[i],
                "target": node_ids[j],
                "edge_type": "similarity",
                "weight": float(sim[i, j]),
            })

    return pd.DataFrame(edges)


def build_graph(
    anchor_csv: str,
    recovered_csv: str,
    output_dir: str,
    similarity_mode: str = "topk",
    k: int = 8,
    threshold: float = 0.8,
):
    """
    Build topology-aware ECG beat graph.

    Nodes:
        anchor_set union recovered_nodes

    Similarity edges:
        anchor beats -> graph nodes

    Temporal edges:
        adjacent graph nodes ordered by original beat_index
    """

    anchor_csv = Path(anchor_csv)
    recovered_csv = Path(recovered_csv)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    anchors = pd.read_csv(anchor_csv)
    recovered = pd.read_csv(recovered_csv)

    required_columns = {
        "patient_id",
        "beat_id",
        "beat_index",
    }

    for name, df in {
        "anchor_set": anchors,
        "recovered_nodes": recovered,
    }.items():
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    nodes = merge_graph_nodes(
        anchors=anchors,
        recovered=recovered,
    )

    if similarity_mode == "topk":
        similarity_edges = build_similarity_edges_topk(
            anchors=anchors,
            nodes=nodes,
            k=k,
        )
    elif similarity_mode == "threshold":
        similarity_edges = build_similarity_edges_threshold(
            anchors=anchors,
            nodes=nodes,
            threshold=threshold,
        )
    else:
        raise ValueError("similarity_mode must be either 'topk' or 'threshold'.")

    temporal_edges = build_temporal_edges(nodes)

    edges = pd.concat(
        [similarity_edges, temporal_edges],
        axis=0,
        ignore_index=True,
    )

    if len(edges) > 0:
        edges = (
            edges
            .drop_duplicates(subset=["source", "target", "edge_type"])
            .reset_index(drop=True)
        )

    patient_id = nodes["patient_id"].iloc[0] if len(nodes) > 0 else "unknown"

    summary = pd.DataFrame([{
        "patient_id": patient_id,
        "n_anchor_nodes": len(anchors),
        "n_recovered_nodes": len(recovered),
        "n_graph_nodes": len(nodes),
        "n_edges": len(edges),
        "n_similarity_edges": int((edges["edge_type"] == "similarity").sum()) if len(edges) > 0 else 0,
        "n_temporal_edges": int((edges["edge_type"] == "temporal").sum()) if len(edges) > 0 else 0,
        "similarity_mode": similarity_mode,
        "k": k if similarity_mode == "topk" else np.nan,
        "threshold": threshold if similarity_mode == "threshold" else np.nan,
    }])

    nodes.to_csv(output_dir / "nodes.csv", index=False)
    edges.to_csv(output_dir / "edges.csv", index=False)
    summary.to_csv(output_dir / "graph_summary.csv", index=False)

    print(f"Graph saved to: {output_dir}")
    print(summary)

    return nodes, edges, summary


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--anchor_csv",
        type=str,
        default="sample/patient_001/anchor_set.csv",
    )

    parser.add_argument(
        "--recovered_csv",
        type=str,
        default="sample/patient_001/recovered_nodes.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/graphs/patient_001",
    )

    parser.add_argument(
        "--similarity_mode",
        type=str,
        default="topk",
        choices=["topk", "threshold"],
    )

    parser.add_argument(
        "--k",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    build_graph(
        anchor_csv=args.anchor_csv,
        recovered_csv=args.recovered_csv,
        output_dir=args.output_dir,
        similarity_mode=args.similarity_mode,
        k=args.k,
        threshold=args.threshold,
    )

    print("Graph construction test passed.")
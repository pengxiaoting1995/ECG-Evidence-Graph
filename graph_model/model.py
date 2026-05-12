"""
Neighbour-aware Transformer graph classifier.

The model performs patient-level binary graph classification.
Edges are used to define local neighbourhoods for Transformer aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighbourTransformerLayer(nn.Module):
    """
    One neighbourhood-aware Transformer layer.

    For each node:
        input sequence = [self node + edge-defined neighbours]
    """

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 1024,
        num_heads: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        neighbour_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, feature_dim]
            neighbour_index: [num_nodes, 1 + num_neighbours]

        Returns:
            updated node features [num_nodes, feature_dim]
        """

        neighbourhood = F.embedding(
            neighbour_index,
            x,
        )

        out = self.encoder(neighbourhood)

        # The first token is the self-node representation.
        out = out[:, 0, :]

        return out


class EvidenceGraphTransformer(nn.Module):
    """
    Edge-defined neighbourhood Transformer for ECG evidence graphs.

    Inputs:
        x          : [total_nodes, 512]
        edge_index : [2, total_edges]
        edge_attr  : [total_edges, 2], columns = [weight, edge_type_id]
        batch      : [total_nodes], graph index for each node

    Output:
        logits     : [num_graphs, 2]
    """

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 1024,
        num_classes: int = 2,
        num_layers: int = 2,
        num_neighbours: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_neighbours = num_neighbours

        self.layers = nn.ModuleList([
            NeighbourTransformerLayer(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_heads=1,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout)
            for _ in range(num_layers)
        ])

        self.prediction_heads = nn.ModuleList([
            nn.Linear(feature_dim, num_classes)
            for _ in range(num_layers)
        ])

    def build_neighbour_index(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build neighbour index from directed graph edges.

        Neighbours are selected according to edge weight.
        The first token is always the node itself.
        """

        num_nodes = x.shape[0]
        device = x.device

        neighbours = [[] for _ in range(num_nodes)]

        if edge_index.numel() > 0:

            sources = edge_index[0].detach().cpu().numpy()
            targets = edge_index[1].detach().cpu().numpy()
            weights = edge_attr[:, 0].detach().cpu().numpy()

            for source, target, weight in zip(sources, targets, weights):
                neighbours[int(source)].append(
                    (int(target), float(weight))
                )

        neighbour_index = torch.empty(
            (num_nodes, self.num_neighbours + 1),
            dtype=torch.long,
            device=device,
        )

        for node_idx in range(num_nodes):

            candidate_neighbours = neighbours[node_idx]

            if len(candidate_neighbours) == 0:
                selected = [node_idx] * self.num_neighbours

            else:
                candidate_neighbours = sorted(
                    candidate_neighbours,
                    key=lambda item: item[1],
                    reverse=True,
                )

                selected = [
                    item[0]
                    for item in candidate_neighbours[: self.num_neighbours]
                ]

                if len(selected) < self.num_neighbours:
                    selected = selected + [node_idx] * (
                        self.num_neighbours - len(selected)
                    )

            neighbour_index[node_idx, 0] = node_idx
            neighbour_index[node_idx, 1:] = torch.tensor(
                selected,
                dtype=torch.long,
                device=device,
            )

        return neighbour_index

    @staticmethod
    def graph_sum_pool(
        x: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sum-pool node embeddings into graph embeddings.
        """

        num_graphs = int(batch.max().item()) + 1

        graph_embeddings = torch.zeros(
            num_graphs,
            x.shape[1],
            device=x.device,
        )

        graph_embeddings.index_add_(
            dim=0,
            index=batch,
            source=x,
        )

        return graph_embeddings

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:

        prediction_scores = 0

        for layer, dropout, prediction_head in zip(
            self.layers,
            self.dropouts,
            self.prediction_heads,
        ):

            neighbour_index = self.build_neighbour_index(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

            x = layer(
                x=x,
                neighbour_index=neighbour_index,
            )

            graph_embeddings = self.graph_sum_pool(
                x=x,
                batch=batch,
            )

            graph_embeddings = dropout(
                graph_embeddings,
            )

            prediction_scores = prediction_scores + prediction_head(
                graph_embeddings
            )

        return prediction_scores


def build_model() -> EvidenceGraphTransformer:
    return EvidenceGraphTransformer()


def label_smoothing(
    true_labels: torch.Tensor,
    classes: int = 2,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Convert hard labels to smoothed labels.
    """

    assert 0 <= smoothing < 1

    confidence = 1.0 - smoothing

    label_shape = torch.Size(
        (true_labels.size(0), classes)
    )

    with torch.no_grad():

        true_dist = torch.empty(
            size=label_shape,
            device=true_labels.device,
        )

        true_dist.fill_(
            smoothing / (classes - 1)
        )

        true_dist.scatter_(
            1,
            true_labels.data.unsqueeze(1),
            confidence,
        )

    return true_dist


def soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return torch.mean(
        torch.sum(-soft_targets * log_probs, dim=1)
    )


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from dataset import PatientGraphDataset, collate_graphs

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

    batch_data = next(iter(loader))

    model = build_model()

    logits = model(
        x=batch_data["x"],
        edge_index=batch_data["edge_index"],
        edge_attr=batch_data["edge_attr"],
        batch=batch_data["batch"],
    )

    print("Logits:", logits.shape)
    print("Labels:", batch_data["y"].shape)

    assert logits.shape[0] == batch_data["y"].shape[0]
    assert logits.shape[1] == 2

    print("Graph model test passed.")
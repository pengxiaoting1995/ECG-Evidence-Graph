"""
Loss functions for CCR-Score learning.
"""

import torch
import torch.nn.functional as F


def compute_classification_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Convert cross-entropy into bounded classification error.

    e = 1 - exp(-CE)

    Returns:
        Tensor with shape [batch]
    """

    ce = F.cross_entropy(
        logits,
        labels,
        reduction="none",
    )

    return 1.0 - torch.exp(-ce)


def ccr_loss(
    classification_error: torch.Tensor,
    ccr_score: torch.Tensor,
    rejection_cost: float,
) -> torch.Tensor:
    """
    Sigmoid-based CCR loss.

    L = e * sigmoid(r) + c * sigmoid(-r)

    where:
        e : classification error
        r : CCR-Score
        c : rejection cost
    """

    ccr_score = ccr_score.view(-1)

    loss = (
        classification_error * torch.sigmoid(ccr_score)
        + rejection_cost * torch.sigmoid(-ccr_score)
    )

    return loss.mean()


def compute_accept_mask(
    ccr_score: torch.Tensor,
) -> torch.Tensor:
    """
    Accept if r >= 0.
    Reject if r < 0.
    """

    return ccr_score.view(-1) >= 0


# if __name__ == "__main__":
#
#     batch_size = 8
#
#     logits = torch.randn(batch_size, 11)
#     labels = torch.randint(0, 11, (batch_size,))
#     ccr_score = torch.randn(batch_size, 1)
#
#     error = compute_classification_error(logits, labels)
#
#     loss = ccr_loss(
#         classification_error=error,
#         ccr_score=ccr_score,
#         rejection_cost=0.375,
#     )
#
#     accept_mask = compute_accept_mask(ccr_score)
#
#     print("Error shape:", error.shape)
#     print("Loss:", loss.item())
#     print("Accepted:", accept_mask.sum().item())
#
#     print("Loss test passed.")
import torch
from torch import Tensor

from utils.pooling import mean_pooling


def normalize_for_unit_mean(
    last_hidden_states: Tensor,
    attention_mask: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """Normalize token embeddings so that the mean-pooled embedding has unit norm.

    Parameters
    ----------
    last_hidden_states : Tensor
        (B, N, D).
    attention_mask : Tensor
        (B, N).
    eps : float
        Added to the norm for numerical stability.

    Returns
    -------
    Tensor
        (B, N, D).
    """
    means = mean_pooling(last_hidden_states, attention_mask)  # (B, D)
    mean_norms = torch.linalg.norm(means, dim=-1).view(-1, 1, 1)  # (B, 1, 1)
    scaled = last_hidden_states / (mean_norms + eps)  # (B, N, D)
    return torch.where(attention_mask[..., None].bool(), scaled, last_hidden_states)

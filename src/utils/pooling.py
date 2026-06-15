from torch import Tensor


def mean_pooling(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Compute mean-pooled text embedding.

    μ(X) := (1/n) Σ_{i=1}^{n} x_i  (over non-padding tokens)

    Parameters
    ----------
    last_hidden_states : Tensor
        (B, N, D) or (N, D).
    attention_mask : Tensor
        (B, N) or (N,).

    Returns
    -------
    Tensor
        (B, D) or (D,).
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=-2) / attention_mask.sum(dim=-1)[..., None]


def covariance_pooling(
    last_hidden_states: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    """Compute covariance matrix.

    Σ(X) := (1/n) Σ_{i=1}^{n} (x_i - μ(X))(x_i - μ(X))^T  (over non-padding tokens)

    Parameters
    ----------
    last_hidden_states : Tensor
        (B, N, D) or (N, D).
    attention_mask : Tensor
        (B, N) or (N,).

    Returns
    -------
    Tensor
        (B, D, D) or (D, D).
    """
    added_batch = False
    if last_hidden_states.dim() == 2:
        last_hidden_states = last_hidden_states.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        added_batch = True

    mask_bool = attention_mask.bool()[..., None]  # (B, N, 1)
    counts = attention_mask.sum(dim=-1).to(last_hidden_states.dtype)  # (B,)

    mean = mean_pooling(last_hidden_states, attention_mask)  # (B, D)
    centered = (last_hidden_states - mean.unsqueeze(-2)).masked_fill(
        ~mask_bool, 0.0
    )  # (B, N, D)
    cov = centered.transpose(-2, -1) @ centered / counts.view(-1, 1, 1)  # (B, D, D)

    if added_batch:
        cov = cov[0]
    return cov

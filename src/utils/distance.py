import torch


def calc_batched_bures_wasserstein_squared_distance(
    A: torch.Tensor, B: torch.Tensor, eps: float = 1e-9
) -> torch.Tensor:
    """Compute squared Bures-Wasserstein distance in batch mode.

    BW^2(A, B) = Tr(A + B - 2 * (A^{1/2} B A^{1/2})^{1/2})

    Parameters
    ----------
    A, B : torch.Tensor
        Symmetric (covariance) matrices. Shape: (D, D) or (batch, D, D).
    eps : float
        Regularization coefficient added to diagonal for numerical stability.

    Returns
    -------
    torch.Tensor
        Shape () for (D, D) input, (batch,) for (batch, D, D) input.
    """

    def _symmetrize_and_regularize(M: torch.Tensor) -> torch.Tensor:
        D = M.shape[-1]
        I = torch.eye(D, device=M.device, dtype=M.dtype)
        return (M + M.transpose(-2, -1)) * 0.5 + eps * I

    def _matrix_sqrt_psd(M: torch.Tensor) -> torch.Tensor:
        # eigendecomposition-based matrix square root
        M = _symmetrize_and_regularize(M)
        eigenvalues, eigenvectors = torch.linalg.eigh(M)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        scaled = eigenvectors * torch.sqrt(eigenvalues).unsqueeze(-2)
        return scaled @ eigenvectors.transpose(-2, -1)

    def _trace(M: torch.Tensor) -> torch.Tensor:
        return M.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    squeeze_out = False
    if A.ndim == 2:
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
        squeeze_out = True

    sqrt_A = _matrix_sqrt_psd(A)  # A^{1/2}
    mid = sqrt_A @ B @ sqrt_A  # A^{1/2} B A^{1/2}
    sqrt_mid = _matrix_sqrt_psd(mid)  # (A^{1/2} B A^{1/2})^{1/2}

    bw2 = _trace(A) + _trace(B) - 2.0 * _trace(sqrt_mid)
    bw2 = torch.clamp(bw2, min=0.0)

    if squeeze_out:
        return bw2.squeeze(0)
    return bw2


def calc_batched_euclidean_squared_distance(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Compute squared Euclidean distance in batch mode.

    ||x - y||_2^2

    Parameters
    ----------
    x, y : torch.Tensor
        Vectors. Shape: (D,) or (batch, D).

    Returns
    -------
    torch.Tensor
        Shape () for (D,) input, (batch,) for (batch, D) input.
    """
    squeeze_out = False
    if x.ndim == 1:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        squeeze_out = True

    dist = torch.sum((x - y) ** 2, dim=-1)

    if squeeze_out:
        return dist.squeeze(0)
    return dist

from __future__ import annotations

from typing import Literal

import torch

from .utils import array_to_matrix

DistMethod = Literal["cdist", "matmul", "chunked"]


def _sq_dists_via_matmul(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute squared pairwise Euclidean distances using the matmul identity.

    Uses ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y to avoid an explicit cdist
    call. More numerically sensitive than cdist but avoids internal sqrt.

    Args:
        X (torch.Tensor): Tensor of shape (N, D).
        Y (torch.Tensor): Tensor of shape (M, D).

    Returns:
        torch.Tensor: Squared distance matrix of shape (N, M).
    """
    XX = (X**2).sum(dim=1, keepdim=True)  # (N, 1)
    YY = (Y**2).sum(dim=1, keepdim=True)  # (M, 1)
    return XX + YY.T - 2 * (X @ Y.T)


def _chunked_rbf_sum(
    X: torch.Tensor, Y: torch.Tensor, gamma: float, chunk_size: int = 1024
) -> float:
    """Compute the sum of RBF kernel values between X and Y in row chunks.

    Processes X in chunks of `chunk_size` rows to avoid materializing the full
    NxM distance matrix, keeping peak memory proportional to chunk_size * M.

    Args:
        X (torch.Tensor): Tensor of shape (N, D).
        Y (torch.Tensor): Tensor of shape (M, D).
        gamma (float): RBF bandwidth parameter (exp(-gamma * ||x-y||^2)).
        chunk_size (int): Number of rows of X to process per iteration.

    Returns:
        float: Sum of all NxM kernel values.
    """
    total = 0.0
    for i in range(0, X.shape[0], chunk_size):
        x_chunk = X[i : i + chunk_size]
        dists = torch.cdist(x_chunk, Y).pow(2)
        total += torch.exp(-gamma * dists).sum().item()
    return total


def _rbf_kernel_sum(
    X: torch.Tensor, Y: torch.Tensor, gamma: float, method: DistMethod
) -> float:
    """Compute the sum of RBF kernel values between X and Y using the chosen method.

    Dispatches to one of three implementations based on `method`:
    - "chunked": memory-efficient row-wise chunking via `_chunked_rbf_sum`.
    - "matmul": full matrix via the matmul identity (`_sq_dists_via_matmul`).
    - "cdist": full matrix via `torch.cdist`.

    Args:
        X (torch.Tensor): Tensor of shape (N, D).
        Y (torch.Tensor): Tensor of shape (M, D).
        gamma (float): RBF bandwidth parameter.
        method (DistMethod): One of "cdist", "matmul", or "chunked".

    Returns:
        float: Sum of all NxM kernel values.
    """
    if method == "chunked":
        return _chunked_rbf_sum(X, Y, gamma)
    if method == "matmul":
        sq_dists = _sq_dists_via_matmul(X, Y)
    else:  # cdist
        sq_dists = torch.cdist(X, Y).pow(2)
    return float(torch.exp(-gamma * sq_dists).sum().item())


def compute(
    tensors_a: torch.Tensor,
    tensors_b: torch.Tensor,
    use_float64: bool = False,
    method: DistMethod = "chunked",
    device: str | torch.device | None = None,
) -> float:
    """
    Compute the MMD between two sets of tensors using a Gaussian kernel.

    Args:
        tensors_a (torch.Tensor): First set of tensors, shape (N_A, ...).
        tensors_b (torch.Tensor): Second set of tensors, shape (N_B, ...).
        use_float64 (bool): Cast inputs to float64 before computing.
        method (str): Distance computation method — one of "cdist", "matmul",
            or "chunked". "chunked" avoids materializing the full NxM matrix.
        device (str | torch.device | None): Device to move tensors to before
            computing (e.g. "cuda", "cpu"). If None, tensors stay on their
            current device.

    Returns:
        float: The computed MMD^2 value.
    """
    # Reshape tensors to 2D matrices
    if tensors_a.ndim > 2:
        matrix_a = array_to_matrix(tensors_a)
        matrix_b = array_to_matrix(tensors_b)
    else:
        matrix_a = tensors_a
        matrix_b = tensors_b

    if device is not None:
        matrix_a = matrix_a.to(device=device)
        matrix_b = matrix_b.to(device=device)

    if use_float64:
        matrix_a = matrix_a.to(dtype=torch.float64)
        matrix_b = matrix_b.to(dtype=torch.float64)

    N_A = matrix_a.shape[0]
    N_B = matrix_b.shape[0]
    gamma = 1.0 / matrix_a.shape[1]

    kernel_AA_sum = _rbf_kernel_sum(matrix_a, matrix_a, gamma, method)
    kernel_BB_sum = _rbf_kernel_sum(matrix_b, matrix_b, gamma, method)
    kernel_AB_sum = _rbf_kernel_sum(matrix_a, matrix_b, gamma, method)

    mmd: float = (
        kernel_AA_sum / (N_A**2)
        + kernel_BB_sum / (N_B**2)
        - 2 * kernel_AB_sum / (N_A * N_B)
    )
    return mmd

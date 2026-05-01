from collections.abc import Callable
from typing import Any

import torch
from geomloss import SamplesLoss

try:
    from flash_sinkhorn import (  # pyright: ignore[reportMissingImports]
        SamplesLoss as FlashSamplesLoss,
    )

    _FLASH_SINKHORN_AVAILABLE = True
except ImportError:
    _FLASH_SINKHORN_AVAILABLE = False


def _sinkhorn_loss(
    loss_cls: Any,
    data1: torch.Tensor,
    data2: torch.Tensor,
    blur: float,
    scaling: float,
    reach: float | None,
    weights1: torch.Tensor | None,
    weights2: torch.Tensor | None,
    p: int = 2,
    return_transport_plan: bool = False,
    **loss_kwargs: Any,
) -> torch.Tensor:
    args: list[torch.Tensor] = []
    if weights1 is not None:
        args.append(weights1)
    args.append(data1)
    if weights2 is not None:
        args.append(weights2)
    args.append(data2)

    if return_transport_plan:
        loss = loss_cls(
            "sinkhorn",
            p=p,
            blur=blur,
            scaling=scaling,
            reach=reach,
            debias=False,
            potentials=True,
            **loss_kwargs,
        )
        F, G = loss(*args)
        # geomloss prepends a batch dim when inputs are unbatched; remove it
        F, G = F.squeeze(0), G.squeeze(0)
        N, M = data1.shape[0], data2.shape[0]
        a = weights1 if weights1 is not None else data1.new_full((N,), 1.0 / N)
        b = weights2 if weights2 is not None else data2.new_full((M,), 1.0 / M)
        x_i = data1.unsqueeze(1)  # (N, 1, D)
        y_j = data2.unsqueeze(0)  # (1, M, D)
        C_ij = (1 / p) * ((x_i - y_j) ** p).sum(-1)  # (N, M)
        eps = blur**p
        return ((F.unsqueeze(1) + G.unsqueeze(0) - C_ij) / eps).exp() * (
            a.unsqueeze(1) * b.unsqueeze(0)
        )

    loss = loss_cls(
        "sinkhorn", p=p, blur=blur, scaling=scaling, reach=reach, **loss_kwargs
    )
    return loss(*args)


def sinkhorn_ot(
    data1: torch.Tensor,
    data2: torch.Tensor,
    blur: float = 0.05,
    scaling: float = 0.9,
    reach: float | None = None,
    weights1: torch.Tensor | None = None,
    weights2: torch.Tensor | None = None,
    p: int = 2,
    return_transport_plan: bool = False,
    **loss_kwargs: Any,
) -> torch.Tensor:
    """
    Sinkhorn Optimal Transport distance between two datasets.

    Args:
        data1:                  (N, D) tensor of source samples
        data2:                  (M, D) tensor of target samples
        blur:                   regularization (epsilon = blur^p);
                                smaller = sharper/slower
        scaling:                multiscale ratio in (0,1); higher = more accurate/slower
        reach:                  if set, uses unbalanced OT (partial mass transport)
        weights1:               (N,) tensor of source weights (uniform if None)
        weights2:               (M,) tensor of target weights (uniform if None)
        p:                      cost exponent (default 2 for squared Euclidean)
        return_transport_plan:  if True, return the (N, M) transport plan instead of
                                the scalar cost; uses debias=False internally
        **loss_kwargs:          passed through to SamplesLoss

    Returns:
        Scalar OT cost, or (N, M) transport plan tensor if return_transport_plan=True
    """
    return _sinkhorn_loss(
        SamplesLoss,
        data1,
        data2,
        blur,
        scaling,
        reach,
        weights1,
        weights2,
        p=p,
        return_transport_plan=return_transport_plan,
        **loss_kwargs,
    )


def flash_sinkhorn_ot(
    data1: torch.Tensor,
    data2: torch.Tensor,
    blur: float = 0.05,
    scaling: float = 0.5,
    reach: float | None = None,
    weights1: torch.Tensor | None = None,
    weights2: torch.Tensor | None = None,
    p: int = 2,
    return_transport_plan: bool = False,
    **loss_kwargs: Any,
) -> torch.Tensor:
    """
    Flash-Sinkhorn Optimal Transport distance between two datasets.

    Requires the flash-sinkhorn package (Linux + CUDA only).

    Args:
        data1:                  (N, D) tensor of source samples
        data2:                  (M, D) tensor of target samples
        blur:                   regularization (epsilon = blur^p);
                                smaller = sharper/slower
        scaling:                epsilon annealing factor in (0,1);
                                higher = more accurate/slower
        reach:                  if set, uses unbalanced OT (partial mass transport)
        weights1:               (N,) tensor of source weights (uniform if None)
        weights2:               (M,) tensor of target weights (uniform if None)
        p:                      cost exponent (default 2 for squared Euclidean)
        return_transport_plan:  if True, return the (N, M) transport plan instead of
                                the scalar cost; uses debias=False internally
        **loss_kwargs:          passed through to FlashSamplesLoss

    Returns:
        Scalar OT cost, or (N, M) transport plan tensor if return_transport_plan=True
    """
    if not _FLASH_SINKHORN_AVAILABLE:
        err_msg = (
            "flash-sinkhorn is not installed. "
            "It requires Linux with CUDA: pip install flash-sinkhorn"
        )
        raise ImportError(err_msg)
    return _sinkhorn_loss(
        FlashSamplesLoss,
        data1,
        data2,
        blur,
        scaling,
        reach,
        weights1,
        weights2,
        p=p,
        return_transport_plan=return_transport_plan,
        **loss_kwargs,
    )


method_map: dict[str, Callable[..., torch.Tensor]] = {
    "sinkhorn": sinkhorn_ot,
    "flash_sinkhorn": flash_sinkhorn_ot,
}


def optimal_transport_distance(
    data1: torch.Tensor,
    data2: torch.Tensor,
    method: str = "sinkhorn",
    **method_kwargs: Any,
) -> torch.Tensor:
    if method in method_map:
        return method_map[method](data1, data2, **method_kwargs)

    err_msg = (
        f"Unsupported OT method: {method}. Supported methods: {list(method_map.keys())}"
    )
    raise ValueError(err_msg)

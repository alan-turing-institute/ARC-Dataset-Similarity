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
    **loss_kwargs: Any,
) -> torch.Tensor:
    loss = loss_cls("sinkhorn", blur=blur, scaling=scaling, reach=reach, **loss_kwargs)
    args: list[torch.Tensor] = []
    if weights1 is not None:
        args.append(weights1)
    args.append(data1)
    if weights2 is not None:
        args.append(weights2)
    args.append(data2)
    return loss(*args)


def sinkhorn_ot(
    data1: torch.Tensor,
    data2: torch.Tensor,
    blur: float = 0.05,
    scaling: float = 0.9,
    reach: float | None = None,
    weights1: torch.Tensor | None = None,
    weights2: torch.Tensor | None = None,
    **loss_kwargs: Any,
) -> torch.Tensor:
    """
    Sinkhorn Optimal Transport distance between two datasets.

    Args:
        data1:    (N, D) tensor of source samples
        data2:    (M, D) tensor of target samples
        blur:     regularization (epsilon = blur²); smaller = sharper/slower
        scaling:  multiscale ratio in (0,1); higher = more accurate/slower
        reach:    if set, uses unbalanced OT (partial mass transport)
        weights1: (N,) tensor of source weights (uniform if None)
        weights2: (M,) tensor of target weights (uniform if None)
        **loss_kwargs: passed through to SamplesLoss

    Returns:
        Scalar tensor with the OT cost
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
    **loss_kwargs: Any,
) -> torch.Tensor:
    """
    Flash-Sinkhorn Optimal Transport distance between two datasets.

    Requires the flash-sinkhorn package (Linux + CUDA only).

    Args:
        data1:    (N, D) tensor of source samples
        data2:    (M, D) tensor of target samples
        blur:     regularization (epsilon = blur²); smaller = sharper/slower
        scaling:  epsilon annealing factor in (0,1); higher = more accurate/slower
        reach:    if set, uses unbalanced OT (partial mass transport)
        weights1: (N,) tensor of source weights (uniform if None)
        weights2: (M,) tensor of target weights (uniform if None)
        **loss_kwargs: passed through to FlashSamplesLoss

    Returns:
        Scalar tensor with the OT cost
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

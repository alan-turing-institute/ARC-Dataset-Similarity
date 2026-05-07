from collections.abc import Callable
from typing import Any

import ot
import torch
from geomloss import SamplesLoss

from dataset_similarity.data.base import ImageDataset
from dataset_similarity.metrics.utils import prepare_tensor_dataset

try:
    from flash_sinkhorn import (  # pyright: ignore[reportMissingImports]
        SamplesLoss as FlashSamplesLoss,
    )

    _FLASH_SINKHORN_AVAILABLE = True
except ImportError:
    _FLASH_SINKHORN_AVAILABLE = False


_PYTHON_OT_METRICS = ("euclidean", "sqeuclidean")


def _resolve_weights(
    data1: torch.Tensor,
    data2: torch.Tensor,
    weights1: torch.Tensor | None,
    weights2: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return uniform weights for any side that was passed as None."""
    N, M = data1.shape[0], data2.shape[0]
    a = weights1 if weights1 is not None else data1.new_full((N,), 1.0 / N)
    b = weights2 if weights2 is not None else data2.new_full((M,), 1.0 / M)
    return a, b


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
    return_coupling: bool = False,
    **loss_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    a, b = _resolve_weights(data1, data2, weights1, weights2)

    loss = loss_cls(
        "sinkhorn", p=p, blur=blur, scaling=scaling, reach=reach, **loss_kwargs
    )

    if return_coupling:
        potentials_loss = loss_cls(
            "sinkhorn",
            p=p,
            blur=blur,
            scaling=scaling,
            reach=reach,
            debias=False,
            potentials=True,
            **loss_kwargs,
        )
        F, G = potentials_loss(a, data1, b, data2)
        # geomloss prepends a batch dim when inputs are unbatched; remove it
        F, G = F.squeeze(0), G.squeeze(0)
        x_i = data1.unsqueeze(1)  # (N, 1, D)
        y_j = data2.unsqueeze(0)  # (1, M, D)
        C_ij = (1 / p) * ((x_i - y_j) ** p).sum(-1)  # (N, M)
        eps = blur**p
        plan = ((F.unsqueeze(1) + G.unsqueeze(0) - C_ij) / eps).exp() * (
            a.unsqueeze(1) * b.unsqueeze(0)
        )
        return loss(a, data1, b, data2), plan

    return loss(a, data1, b, data2), None


def sinkhorn_ot(
    data1: torch.Tensor,
    data2: torch.Tensor,
    blur: float = 0.05,
    scaling: float = 0.9,
    reach: float | None = None,
    weights1: torch.Tensor | None = None,
    weights2: torch.Tensor | None = None,
    p: int = 2,
    return_coupling: bool = False,
    **loss_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        return_coupling:       if True, return the (N, M) coupling instead of
                                the scalar cost; uses debias=False internally
        **loss_kwargs:          passed through to SamplesLoss

    Returns:
        Tuple of (scalar OT cost, coupling), where coupling is an (N, M) tensor
        if return_coupling=True, else None.
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
        return_coupling=return_coupling,
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
    return_coupling: bool = False,
    **loss_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        return_coupling:       if True, return the (N, M) coupling instead of
                                the scalar cost; uses debias=False internally
        **loss_kwargs:          passed through to FlashSamplesLoss

    Returns:
        Tuple of (scalar OT cost, coupling), where coupling is an (N, M) tensor
        if return_coupling=True, else None.
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
        return_coupling=return_coupling,
        **loss_kwargs,
    )


def python_ot(
    data1: torch.Tensor,
    data2: torch.Tensor,
    weights1: torch.Tensor | None = None,
    weights2: torch.Tensor | None = None,
    metric: str = "sqeuclidean",
    return_coupling: bool = False,
    **solve_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Optimal Transport distance (or coupling) between two datasets.

    Delegates to ``ot.solve_sample``. Whether this runs on the torch backend
    (supporting autograd and GPU) depends on the method:

    - ``method='geomloss'``: fully differentiable, GPU-compatible. Requires
      ``reg`` to be set. Use this when gradients or GPU execution are needed.
    - default (no ``method``): POT infers the backend from the input tensors,
      but exact EMD and most Sinkhorn variants drop to numpy/C internally and
      will not support autograd or GPU tensors.

    Args:
        data1:           (N, D) tensor of source samples
        data2:           (M, D) tensor of target samples
        weights1:        (N,) source weights (uniform if None)
        weights2:        (M,) target weights (uniform if None)
        metric:          distance metric passed to ``ot.solve_sample``; one of
                         ``"euclidean"`` (L1 cost) or ``"sqeuclidean"``
                         (squared-L2 cost, default)
        return_coupling: if True, return the (N, M) coupling instead of
                         the scalar cost
        **solve_kwargs:  passed through to ``ot.solve_sample``
                         (e.g. ``reg``, ``method``)

    Returns:
        Tuple of (scalar OT cost, coupling), where coupling is an (N, M) tensor
        if return_coupling=True, else None.
    """
    if metric not in _PYTHON_OT_METRICS:
        err_msg = f"python_ot metric must be one of {_PYTHON_OT_METRICS}, got {metric}."
        raise ValueError(err_msg)
    a, b = _resolve_weights(data1, data2, weights1, weights2)
    sol = ot.solve_sample(data1, data2, a, b, metric=metric, **solve_kwargs)

    if return_coupling:
        plan = sol.plan
        if not isinstance(plan, torch.Tensor):
            # use to_dense() if it's a sparse tensor, else return as-is
            plan = plan.to_dense() if hasattr(plan, "to_dense") else plan[:]
        return sol.value, plan

    return sol.value, None


method_map: dict[str, Callable[..., tuple[torch.Tensor, torch.Tensor | None]]] = {
    "sinkhorn": sinkhorn_ot,
    "flash_sinkhorn": flash_sinkhorn_ot,
    "python_ot": python_ot,
}


def optimal_transport_distance(
    dataset1: ImageDataset,
    dataset2: ImageDataset,
    method: str = "sinkhorn",
    **method_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Optimal Transport distance between two datasets.

    Args:
        dataset1:       Source ImageDataset (must have return_paths=False)
        dataset2:       Target ImageDataset (must have return_paths=False)
        method:         OT method — one of "sinkhorn", "flash_sinkhorn", "python_ot"
        **method_kwargs: Passed through to the chosen method function

    Returns:
        Tuple of (scalar OT cost, coupling), where coupling is an (N, M) tensor
        if return_coupling=True, else None.
    """
    data1 = prepare_tensor_dataset(dataset1)
    data2 = prepare_tensor_dataset(dataset2)

    if method in method_map:
        return method_map[method](data1, data2, **method_kwargs)

    err_msg = (
        f"Unsupported OT method: {method}. Supported methods: {list(method_map.keys())}"
    )
    raise ValueError(err_msg)

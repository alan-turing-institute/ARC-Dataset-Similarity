import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import ot as pot
import torch

from dataset_similarity.data.base import ImageDataset
from dataset_similarity.metrics.utils import extract_dataset_tensors

logger = logging.getLogger(__name__)

# flash-sinkhorn is an optional dependency with Linux + CUDA requirements, so we import
# it in a try-except block and set a flag for availability
try:
    from flash_sinkhorn import (  # pyright: ignore[reportMissingImports]
        SamplesLoss as FlashSamplesLoss,
    )

    _FLASH_SINKHORN_AVAILABLE = True
except ImportError:
    _FLASH_SINKHORN_AVAILABLE = False


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


def flash_sinkhorn_ot(
    data1: torch.Tensor,
    data2: torch.Tensor,
    blur: float = 0.05,
    scaling: float = 0.9,
    reach: float | None = None,
    weights1: torch.Tensor | None = None,
    weights2: torch.Tensor | None = None,
    p: int = 2,
    return_coupling: bool = False,
    debias: bool = False,
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
        return_coupling:        if True, return the (N, M) coupling instead of
                                the scalar cost; uses debias=False internally
        **loss_kwargs:          passed through to SamplesLoss / FlashSamplesLoss

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

    a, b = _resolve_weights(data1, data2, weights1, weights2)

    loss = FlashSamplesLoss(
        "sinkhorn",
        p=p,
        blur=blur,
        scaling=scaling,
        reach=reach,
        debias=debias,
        **loss_kwargs,
    )

    if return_coupling:
        # A separate loss instance is needed with potentials=True and debias=False.
        # debias=False is required because the Gibbs-kernel formula for the plan,
        #   y_ij = exp((F_i + G_j - C_ij) / ε) * a_i * b_j,
        # is only valid for the undebiased dual potentials (F, G).

        potentials_loss = FlashSamplesLoss(
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

        # geomloss prepends a batch dimension when inputs are unbatched; remove it.
        F, G = F.squeeze(0), G.squeeze(0)
        x_i = data1.unsqueeze(1)  # (N, 1, D)
        y_j = data2.unsqueeze(0)  # (1, M, D)
        C_ij = (1 / p) * (x_i - y_j).abs().pow(p).sum(-1)  # (N, M)
        eps = blur**p

        # Recover the transport plan from the Gibbs kernel:
        plan = ((F.unsqueeze(1) + G.unsqueeze(0) - C_ij) / eps).exp() * (
            a.unsqueeze(1) * b.unsqueeze(0)
        )
        # Recover the scalar cost from the dual objective
        cost = (a * F).sum() + (b * G).sum()
        return cost, plan

    return loss(a, data1, b, data2), None


def python_ot(
    data1: torch.Tensor,
    data2: torch.Tensor,
    metric: str = "sqeuclidean",
    return_coupling: bool = False,
    method: str | None = None,
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
        metric:          distance metric passed to ``ot.solve_sample``
        return_coupling: if True, return the (N, M) coupling in addition to
                         the scalar cost
        **solve_kwargs:  passed through to ``ot.solve_sample``
                         (e.g. ``reg``, ``method``)

    Returns:
        Tuple of (scalar OT cost, coupling), where coupling is an (N, M) tensor
        if return_coupling=True, else None.
    """
    sol = pot.solve_sample(
        data1,
        data2,
        metric=metric,
        method=method,
        **solve_kwargs,
    )

    if return_coupling:
        plan = sol.plan
        if not isinstance(plan, torch.Tensor):
            # use to_dense() if it's a sparse tensor, else return as-is
            plan = plan.to_dense() if hasattr(plan, "to_dense") else plan[:]
        return sol.value, plan

    return sol.value, None


method_map: dict[str, Callable[..., tuple[torch.Tensor, torch.Tensor | None]]] = {
    "sinkhorn": partial(python_ot, method="sinkhorn"),
    "flash_sinkhorn": flash_sinkhorn_ot,
    "python_ot": python_ot,
}


def ot_distance(
    dataset1: ImageDataset,
    dataset2: ImageDataset,
    method: str = "sinkhorn",
    return_coupling: bool = False,
    device: str | torch.device | None = None,
    **method_kwargs: Any,
) -> float | tuple[float, torch.Tensor]:
    """
    Optimal Transport distance between two datasets.

    Args:
        dataset1:       Source ImageDataset (must have return_paths=False)
        dataset2:       Target ImageDataset (must have return_paths=False)
        method:         OT method - one of "sinkhorn", "flash_sinkhorn", "python_ot"
        device:         Device to move tensors to before computing (e.g. "cuda",
                        "cpu"). If None, tensors stay on their current device (CPU).
                        Note: method="python_ot" does not natively support CUDA;
                        use method="sinkhorn" for GPU-compatible OT.
        **method_kwargs: Passed through to the chosen method function

    Returns:
        float OT cost, or tuple of (float OT cost, (N, M) coupling tensor)
        if return_coupling=True.
    """
    data1: torch.Tensor = extract_dataset_tensors(dataset1)
    data2: torch.Tensor = extract_dataset_tensors(dataset2)

    if device is not None:
        data1 = data1.to(device=device)
        data2 = data2.to(device=device)

    if (
        device is not None
        and torch.device(device).type == "cuda"
        and method == "python_ot"
    ):
        logger.warning(
            "method='python_ot' does not natively support CUDA. "
            "Use method='sinkhorn' for GPU-compatible OT."
        )

    if method not in method_map:
        err_msg = (
            f"Unsupported OT method: {method}. "
            f"Supported methods: {list(method_map.keys())}"
        )
        raise ValueError(err_msg)

    cost, coupling = method_map[method](
        data1, data2, return_coupling=return_coupling, **method_kwargs
    )
    if coupling is not None:
        return float(cost.item()), coupling
    return float(cost.item())

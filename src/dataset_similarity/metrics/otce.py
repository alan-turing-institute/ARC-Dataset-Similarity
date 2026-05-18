"""
Yang et al., CVPR 2021  (https://arxiv.org/abs/2103.13843)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch

from dataset_similarity.data.base import ImageDataset
from dataset_similarity.metrics.ot import method_map, ot_distance
from dataset_similarity.metrics.otdd import otdd


def _conditional_entropy(
    coupling: torch.Tensor,
    src_labels: torch.Tensor,
    tgt_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Estimate H(Y_t | Y_s) from the OT coupling matrix.

    For each pair of source class s and target class t, the joint probability
    is approximated by summing coupling entries that connect samples of class s
    to samples of class t:

        P(Y_s = s, Y_t = t) ≈ Σ_{i ∈ s, j ∈ t} T_{ij}

    Then:
        H(Y_t | Y_s) = - Σ_{s,t} P(s,t) log [ P(s,t) / P(s) ]

    Args:
        coupling:    (N, M) OT coupling (rows = source, cols = target).
        src_labels:  (N,) integer class labels for source samples.
        tgt_labels:  (M,) integer class labels for target samples.

    Returns:
        Scalar conditional entropy H(Y_t | Y_s).
    """
    src_classes: torch.Tensor = src_labels.unique()
    tgt_classes: torch.Tensor = tgt_labels.unique()

    # One-hot class indicators: src_oh[i, k] = 1 if sample i belongs to source class k
    src_oh = (src_labels.unsqueeze(1) == src_classes.unsqueeze(0)).float()  # (N, |S|)
    tgt_oh = (tgt_labels.unsqueeze(1) == tgt_classes.unsqueeze(0)).float()  # (M, |T|)

    # joint[k, l] = sum of T[i, j] for all i in class k, j in class l
    # src_oh.T @ coupling aggregates coupling rows by source class -> (|S|, M)
    # then @ tgt_oh aggregates columns by target class -> (|S|, |T|)
    # joint[k, l] is the total mass transported from source class k to target class l
    joint = src_oh.T @ coupling @ tgt_oh  # (|S|, |T|)

    # Normalise to a proper joint distribution
    joint = joint / joint.sum()

    # Marginal P(Y_s); every source class has at least one sample (via unique())
    p_src = joint.sum(dim=1, keepdim=True)  # (|S|, 1)

    # Conditional P(Y_t | Y_s) = P(Y_s, Y_t) / P(Y_s)
    cond = joint / p_src  # (|S|, |T|)

    # H(Y_t | Y_s) = - Σ P(s,t) log P(t|s)
    # torch.xlogy(x, y) = x * log(y), with xlogy(0, 0) = 0
    return -torch.xlogy(joint, cond).sum()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def otce_score_from_tensors(
    src_features: torch.Tensor,
    src_labels: torch.Tensor,
    tgt_features: torch.Tensor,
    tgt_labels: torch.Tensor,
    ot_method: str = "sinkhorn",
    weights_src: torch.Tensor | None = None,
    weights_tgt: torch.Tensor | None = None,
    **ot_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """
    Compute OTCE from pre-computed feature tensors and integer labels.

    Args:
        src_features:   (N, D) source feature matrix.
        src_labels:     (N,)   integer class labels for source samples.
        tgt_features:   (M, D) target feature matrix.
        tgt_labels:     (M,)   integer class labels for target samples.
        ot_method:      OT solver - "sinkhorn" | "flash_sinkhorn" | "python_ot".
        weights_src:    Optional (N,) source sample weights (uniform if None).
        weights_tgt:    Optional (M,) target sample weights (uniform if None).
        **ot_kwargs:    Forwarded to the chosen OT method function.

    Returns:
        dict with keys:
            "otce"               - scalar OTCE score (higher = better transfer).
            "wasserstein"        - scalar OT cost W (domain shift).
            "conditional_entropy"- scalar H(Y_t | Y_s) (task misalignment).
            "coupling"           - (N, M) OT coupling matrix.
    """
    if ot_method not in method_map:
        error_msg = (
            f"Unsupported OT method: {ot_method!r}. "
            f"Available: {list(method_map.keys())}"
        )
        raise ValueError(error_msg)

    ot_fn = method_map[ot_method]

    # --- Step 1: compute OT cost and coupling ---

    wasserstein, coupling = ot_fn(
        src_features,
        tgt_features,
        weights1=weights_src,
        weights2=weights_tgt,
        return_coupling=True,
        **ot_kwargs,
    )

    # --- Step 2: conditional entropy ---
    src_labels = src_labels.to(src_features.device)
    tgt_labels = tgt_labels.to(tgt_features.device)
    h = _conditional_entropy(coupling, src_labels, tgt_labels)

    # --- Step 3: OTCE ---
    otce = -wasserstein - h

    return {
        "otce": otce,
        "wasserstein": wasserstein,
        "conditional_entropy": h,
        "coupling": coupling,
    }


def otce_distance(
    dataset1: ImageDataset,
    dataset2: ImageDataset,
    return_coupling: bool = False,
    use_wasserstein: bool = True,
    use_otdd: bool = False,
    **distance_kwargs: Any,
) -> float | tuple[float, torch.Tensor]:
    """
    Compute OTCE distance between two ImageDatasets.

    Expects each dataset to yield ``(feature_tensor, label)`` pairs
    (i.e. ``return_paths=False``). Labels must be integer class indices.

    Returns a positive distance score (lower = better transfer). This is the
    negation of the true OTCE score (-W - H) as defined in Yang et al. 2021,
    where higher = better transfer. Use ``otce_score_from_tensors`` directly
    to access the raw signed score.

    Args:
        dataset1:        Source ImageDataset.
        dataset2:        Target ImageDataset.
        ot_method:       OT solver - "sinkhorn" | "flash_sinkhorn" | "python_ot".
        return_coupling: If True, also return the (N, M) OT coupling matrix.
        **distance_kwargs:     Forwarded to the chosen distance method.

    Returns:
        float OTCE distance, or tuple of (float OTCE distance, (N, M) coupling tensor)
        if return_coupling=True.
    """

    # --- Step 1: compute OT cost and coupling ---
    distance_func = (
        cast(  # to make mypy happy about the return type of ot_distance / otdd
            Callable[..., tuple[float, torch.Tensor]], otdd if use_otdd else ot_distance
        )
    )
    wasserstein, coupling = distance_func(
        dataset1, dataset2, return_coupling=True, **distance_kwargs
    )

    # --- Step 2: conditional entropy ---
    src_labels = torch.tensor(dataset1.data.label)
    tgt_labels = torch.tensor(dataset2.data.label)

    h = _conditional_entropy(coupling, src_labels, tgt_labels)

    # --- Step 3: OTCE ---
    if use_wasserstein:
        otce_dist: float = (wasserstein + h).item()
    else:
        otce_dist = h.item()

    if return_coupling:
        return otce_dist, coupling
    return otce_dist

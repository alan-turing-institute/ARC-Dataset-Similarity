"""
Yang et al., CVPR 2021  (https://arxiv.org/abs/2103.13843)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch

from dataset_similarity.data.base import ImageDataset
from dataset_similarity.metrics.ot import ot_distance
from dataset_similarity.metrics.otdd import otdd


def conditional_entropy(
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


def otce_distance(
    dataset1: ImageDataset,
    dataset2: ImageDataset,
    use_wasserstein: bool = True,
    use_otdd: bool = False,
    device: str | torch.device | None = None,
    **distance_kwargs: Any,
) -> float:
    """
    Compute OTCE distance between two ImageDatasets.

    Expects each dataset to yield ``(feature_tensor, label)`` pairs
    (i.e. ``return_paths=False``). Labels must be integer class indices.

    Returns a positive distance score (lower = better transfer). This corresponds to
    W + H, where W is the OT/Wasserstein cost term (optional) and H is the conditional
    entropy term from Yang et al. 2021.

    Args:
        dataset1:         Source ImageDataset.
        dataset2:         Target ImageDataset.
        use_wasserstein:  If True, include the OT/Wasserstein term W in the returned
                          distance.
        use_otdd:         If True, compute W and the coupling via OTDD instead of OT.
        device:           Device to move tensors to before computing (e.g. "cuda",
                          "cpu"). Forwarded to the underlying OT distance function. If
                          None, tensors stay on their current device (CPU).
        **distance_kwargs: Forwarded to the chosen distance method.

    Returns:
        float OTCE distance.
    """

    # --- Step 1: compute OT cost and coupling ---
    distance_func = (
        cast(  # to make mypy happy about the return type of ot_distance / otdd
            Callable[..., tuple[float, torch.Tensor]], otdd if use_otdd else ot_distance
        )
    )
    # Pass the device argument only if it's not None, otherwise use default behaviour
    device_kwarg: dict[str, str | torch.device] = (
        {"device": device} if device is not None else {}
    )
    wasserstein, coupling = distance_func(
        dataset1, dataset2, return_coupling=True, **device_kwarg, **distance_kwargs
    )

    # --- Step 2: conditional entropy ---
    # To work with both ImageDataset and DatasetMix:
    src_labels = torch.tensor([dataset1[i][1] for i in range(len(dataset1))]).to(
        coupling.device
    )
    tgt_labels = torch.tensor([dataset2[i][1] for i in range(len(dataset2))]).to(
        coupling.device
    )

    h = conditional_entropy(coupling, src_labels, tgt_labels)

    # --- Step 3: OTCE ---
    # Unweighted W + H (implicit lambda1=lambda2=1, b=0), not the paper's regression-
    # calibrated score - avoids needing an auxiliary task suite to fit the combination.
    if use_wasserstein:
        otce_dist: float = (wasserstein + h).item()
    else:
        otce_dist = h.item()

    return otce_dist

"""
Yang et al., CVPR 2021  (https://arxiv.org/abs/2103.13843)
"""

from __future__ import annotations

from typing import Any

import ot as pot
import torch
from otdd.pytorch.distance import DatasetDistance

from dataset_similarity.data.base import ImageDataset
from dataset_similarity.metrics.ot import method_map
from dataset_similarity.metrics.otdd import _prepare_otdd_tensor_dataset
from dataset_similarity.metrics.utils import prepare_tensor_dataset

_OTCE_VARIANTS = ("otce", "f_otce", "jc_otce")


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


def _label_distance_matrix(
    dataset1: ImageDataset,
    dataset2: ImageDataset,
    diagonal_cov: bool = True,
    **otdd_kwargs: Any,
) -> torch.Tensor:
    """
    Return the (C_src, C_tgt) matrix of pairwise label Wasserstein distances
    using OTDD's Gaussian approximation.

    Args:
        dataset1:     Source ImageDataset.
        dataset2:     Target ImageDataset.
        diagonal_cov: Use diagonal covariance (required on PyTorch 2.x). Default True.
        **otdd_kwargs: Forwarded to DatasetDistance (e.g. device, inner_ot_method).
    """
    tds1 = _prepare_otdd_tensor_dataset(dataset1)
    tds2 = _prepare_otdd_tensor_dataset(dataset2)
    dd = DatasetDistance(
        D1=tds1,
        D2=tds2,
        debiased_loss=False,
        diagonal_cov=diagonal_cov,
        **otdd_kwargs,
    )
    return dd._get_label_distances()  # (C_src, C_tgt)


def otce_distance(
    dataset1: ImageDataset,
    dataset2: ImageDataset,
    ot_method: str = "sinkhorn",
    variant: str = "otce",
    gamma: float = 0.5,
    return_coupling: bool = False,
    label_dist_kwargs: dict[str, Any] | None = None,
    **ot_kwargs: Any,
) -> float | tuple[float, torch.Tensor]:
    """
    Compute OTCE distance between two ImageDatasets.

    Expects datasets to yield ``(feature_tensor, label)`` pairs
    (``return_paths=False``). Labels must be integer class indices.

    Returns a positive distance (lower = better transfer).

    Args:
        dataset1:          Source ImageDataset.
        dataset2:          Target ImageDataset.
        ot_method:         OT solver - "sinkhorn" | "flash_sinkhorn" | "python_ot".
                           Ignored when variant="jc_otce".
        variant:           "otce"     - standard W + H (default).
                           "f_otce"   - conditional entropy H only.
                           "jc_otce"  - combined feature+label ground cost, then W + H.
        gamma:             Balance between feature cost (gamma) and label cost (1-gamma)
                           in the JC-OTCE ground cost. Only used when variant="jc_otce".
        return_coupling:   If True, also return the (N, M) coupling matrix.
        label_dist_kwargs: Kwargs forwarded to _label_distance_matrix (e.g.
                           diagonal_cov, device). Only used when variant="jc_otce".
        **ot_kwargs:       Forwarded to the chosen OT method (variant != "jc_otce"),
                           or to pot.solve (variant="jc_otce", e.g. reg=0.1).

    Returns:
        float distance, or tuple of (float, (N, M) coupling) if return_coupling=True.
    """
    if variant not in _OTCE_VARIANTS:
        error_msg = f"variant must be one of {_OTCE_VARIANTS}, got {variant!r}"
        raise ValueError(error_msg)
    if variant != "jc_otce" and ot_method not in method_map:
        error_msg = (
            f"Unsupported ot_method: {ot_method!r}. "
            f"Available: {list(method_map.keys())}"
        )
        raise ValueError(error_msg)

    src_features, src_labels = prepare_tensor_dataset(dataset1, return_labels=True)
    tgt_features, tgt_labels = prepare_tensor_dataset(dataset2, return_labels=True)
    src_labels = src_labels.to(src_features.device)
    tgt_labels = tgt_labels.to(tgt_features.device)

    if variant == "jc_otce":
        if not 0.0 <= gamma <= 1.0:
            error_msg = f"gamma must be in [0, 1], got {gamma}"
            raise ValueError(error_msg)

        x_i = src_features.unsqueeze(1)
        y_j = tgt_features.unsqueeze(0)
        feature_cost = (x_i - y_j).pow(2).sum(-1)  # (N, M)

        label_dist = _label_distance_matrix(
            dataset1, dataset2, **(label_dist_kwargs or {})
        ).to(src_features.device)

        src_classes = src_labels.unique()
        tgt_classes = tgt_labels.unique()
        src_idx = torch.zeros_like(src_labels)
        tgt_idx = torch.zeros_like(tgt_labels)
        for k, c in enumerate(src_classes):
            src_idx[src_labels == c] = k
        for k, c in enumerate(tgt_classes):
            tgt_idx[tgt_labels == c] = k

        label_cost = label_dist[src_idx][:, tgt_idx]  # (N, M)
        combined_cost = gamma * feature_cost + (1.0 - gamma) * label_cost

        N, M = src_features.shape[0], tgt_features.shape[0]
        a = src_features.new_full((N,), 1.0 / N).cpu()
        b = tgt_features.new_full((M,), 1.0 / M).cpu()
        sol = pot.solve(combined_cost.detach().cpu(), a=a, b=b, **ot_kwargs)
        wasserstein: torch.Tensor = torch.tensor(float(sol.value))
        coupling: torch.Tensor = torch.as_tensor(sol.plan, dtype=src_features.dtype)
    else:
        wasserstein, coupling = method_map[ot_method](
            src_features, tgt_features, return_coupling=True, **ot_kwargs
        )

    h = _conditional_entropy(coupling, src_labels, tgt_labels)
    score: float = float(h.item() if variant == "f_otce" else (wasserstein + h).item())

    return (score, coupling) if return_coupling else score

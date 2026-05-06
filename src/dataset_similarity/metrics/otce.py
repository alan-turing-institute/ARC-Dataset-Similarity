"""
Yang et al., CVPR 2021  (https://arxiv.org/abs/2103.13843)
"""

from __future__ import annotations

from typing import Any

import torch

from dataset_similarity.data.base import ImageDataset
from dataset_similarity.metrics.ot import method_map
from dataset_similarity.metrics.utils import prepare_tensor_dataset


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
    src_classes = src_labels.unique()
    tgt_classes = tgt_labels.unique()

    # Build joint probability table P(Y_s, Y_t)
    joint = coupling.new_zeros(len(src_classes), len(tgt_classes))
    for i, s in enumerate(src_classes):
        src_mask = src_labels == s
        for j, t in enumerate(tgt_classes):
            tgt_mask = tgt_labels == t
            joint[i, j] = coupling[src_mask][:, tgt_mask].sum()

    # Normalise to a proper joint distribution
    joint = joint / joint.sum().clamp(min=1e-10)

    # Marginal P(Y_s)
    p_src = joint.sum(dim=1, keepdim=True).clamp(min=1e-10)  # (|S|, 1)

    # Conditional P(Y_t | Y_s) = P(Y_s, Y_t) / P(Y_s)
    cond = (joint / p_src).clamp(min=1e-10)  # (|S|, |T|)

    # H(Y_t | Y_s) = - Σ P(s,t) log P(t|s)
    return -1 * (joint * cond.log()).sum()


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
        ot_method:      OT solver — "sinkhorn" | "flash_sinkhorn" | "python_ot".
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


def otce_score(
    dataset1: ImageDataset,
    dataset2: ImageDataset,
    ot_method: str = "sinkhorn",
    **ot_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """
    Compute OTCE between two ImageDatasets.

    Expects each dataset to yield ``(feature_tensor, label)`` pairs
    (i.e. ``return_paths=False``). Labels must be integer class indices.

    Args:
        dataset1:    Source ImageDataset.
        dataset2:    Target ImageDataset.
        ot_method:   OT solver — "sinkhorn" | "flash_sinkhorn" | "python_ot".
        **ot_kwargs: Forwarded to the chosen OT method.

    Returns:
        Same dict as :func:`otce_score_from_tensors`.
    """

    src_features = prepare_tensor_dataset(dataset1)
    tgt_features = prepare_tensor_dataset(dataset2)

    src_labels = torch.tensor([int(label) for _, label in dataset1], dtype=torch.long)
    tgt_labels = torch.tensor([int(label) for _, label in dataset2], dtype=torch.long)

    return otce_score_from_tensors(
        src_features,
        src_labels,
        tgt_features,
        tgt_labels,
        ot_method=ot_method,
        **ot_kwargs,
    )

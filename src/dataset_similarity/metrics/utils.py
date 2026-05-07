from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.mix import DatasetMix


def prepare_tensor_dataset(
    dataset: ImageDataset | DatasetMix | Dataset,
    return_labels: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Extract feature (and optionally label) tensors from a dataset."""
    # Detect whether any constituent dataset returns paths instead of labels.
    sub_datasets = getattr(dataset, "datasets", None)
    if sub_datasets is not None:
        # DatasetMix: check each underlying ImageDataset individually.
        if any(getattr(d, "return_paths", False) for d in sub_datasets):
            err_msg = (
                "OT computation does not support DatasetMix instances that contain "
                "ImageDatasets with return_paths=True. "
                "Please initialize all constituent datasets with return_paths=False."
            )
            raise ValueError(err_msg)
    elif getattr(dataset, "return_paths", False):
        err_msg = (
            "OT computation does not support ImageDatasets with return_paths=True. "
            "Please initialize the dataset with return_paths=False to compute OT."
        )
        raise ValueError(err_msg)
    samples = [dataset[idx] for idx in range(len(dataset))]
    features = torch.stack([s[0] for s in samples])
    if return_labels:
        labels = torch.tensor([s[1] for s in samples])
        return features, labels
    return features


def array_to_matrix(array: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    return array.reshape(array.shape[0], -1)

from typing import Any

import torch
from otdd.pytorch.distance import DatasetDistance
from torch.utils.data import TensorDataset

from dataset_similarity.data.base import ImageDataset


def _prepare_tensor_dataset_targets(labels: torch.Tensor) -> torch.Tensor:
    """
    Helper function which, given a torch.tensor of labels, maps the labels to a
    contiguous range of integers starting from 0, and returns the mapped labels as a new
    tensor. This is necessary for OTDD, which requires the labels to be in this format.

    Args:
        labels: The input tensor of labels to be mapped.

    Returns:
        torch.tensor: The output tensor of mapped labels, where the unique values in the
            input labels have been mapped to a contiguous range of integers starting
            from 0.
    """
    dataset1_label_class_map = {
        label.item(): idx
        for idx, label in enumerate(torch.sort(torch.unique(labels)).values)
    }
    return torch.tensor([dataset1_label_class_map[label.item()] for label in labels])


def _prepare_otdd_tensor_dataset(dataset: ImageDataset) -> TensorDataset:
    """
    Function which, given an ImageDataset, prepares a TensorDataset suitable for use
    with OTDD by organising the features and labels into tensors, and mapping the labels
    to a contiguous range of integers starting from 0.

    Args:
        dataset: The ImageDataset to prepare.

    Returns:
        TensorDataset: The processed TensorDataset ready for use with OTDD.
    """
    if dataset.return_paths:
        err_msg = (
            "OTDD computation does not support ImageDatasets with return_paths=True. "
            "Please initialize the dataset with return_paths=False to compute OTDD."
        )
        raise ValueError(err_msg)
    features = []
    labels = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        features.append(sample[0])
        labels.append(sample[1])
    features = torch.stack(features)
    labels = torch.tensor(labels)
    targets = _prepare_tensor_dataset_targets(labels)
    tensor_dataset = TensorDataset(features, targets)
    tensor_dataset.classes = torch.sort(torch.unique(targets)).values
    return tensor_dataset


def otdd(
    dataset1: ImageDataset,
    dataset2: ImageDataset,
    return_coupling: bool = False,
    maxsamples: int = 10000,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper function around the `otdd.DatasetDistance` class which implements the
    optimal transport dataset distance (OTDD). Note that while any device can be
    passed via `device=<device>`, the underlying OT implementations can only use CUDA.

    Args:
        dataset1: The first dataset with which to compute the OTDD.
        dataset2: The second dataset with which to compute the OTDD.
        return_coupling: Whether to return the optimal transport coupling matrix along
            with the OTDD distance. Defaults to False.
        maxsamples: The maximum number of samples to use from each dataset when
            computing the OTDD. If either dataset has more than `maxsamples` samples, a
            random subset of `maxsamples` samples will be used from that dataset.
            Defaults to 10000. Note that OTDD can be computationally expensive for large
            datasets, and that there is a hardcoded limit of 10000 samples on GPU, so
            this parameter should be set to 10000 or less.
        **kwargs: Additional keyword arguments to pass to the `otdd.DatasetDistance`
            constructor. See the OTDD documentation for details on available parameters.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]: The OTDD distance between the
        two datasets,
            as computed by the `otdd.DatasetDistance`. If `return_coupling` is True,
            also returns the optimal transport coupling matrix.
    """
    tds1 = _prepare_otdd_tensor_dataset(dataset1)
    tds2 = _prepare_otdd_tensor_dataset(dataset2)
    otdd_distance = DatasetDistance(
        D1=tds1,
        D2=tds2,
        **kwargs,
    )
    distance: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = otdd_distance.distance(
        return_coupling=return_coupling, maxsamples=maxsamples
    )
    return distance

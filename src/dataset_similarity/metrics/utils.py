import numpy as np
import torch

from dataset_similarity.data.base import ImageDataset


def prepare_tensor_dataset(
    dataset: ImageDataset, return_labels: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Extract feature (and optionally label) tensors from an ImageDataset."""
    if dataset.return_paths:
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

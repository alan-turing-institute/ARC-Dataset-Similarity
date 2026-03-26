from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

#  initial transformation examples

horizontal_flip = transforms.Compose(
    [
        transforms.Lambda(F.hflip),
    ]
)

rotation_180 = transforms.Compose(
    [
        transforms.Lambda(lambda x: F.rotate(x, 180)),
    ]
)

centre_crop = transforms.Compose(
    [
        transforms.CenterCrop(224),
    ]
)

grayscale = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
    ]
)

gaussian_blur = transforms.Compose(
    [
        transforms.GaussianBlur(kernel_size=11, sigma=3.0),
    ]
)

deterministic_colour_jitter = transforms.Compose(
    [
        transforms.Lambda(lambda x: F.adjust_brightness(x, 1.5)),
        transforms.Lambda(lambda x: F.adjust_contrast(x, 1.5)),
        transforms.Lambda(lambda x: F.adjust_saturation(x, 0.5)),
        transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),
    ]
)

grayscale_and_blur = transforms.Compose(
    [
        grayscale,
        gaussian_blur,
    ]
)


class TransformedDataset(Dataset):  # type: ignore[misc]
    """
    A dataset wrapper that applies a transformation to the data.
    """

    def __init__(self, dataset: Dataset, transform: Callable[[Tensor], Tensor]) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Any]:
        item, label = self.dataset[idx]
        return self.transform(item), label


def apply_transform(dataset: Dataset, transform: Callable[[Tensor], Tensor]) -> Dataset:
    """
    Apply a simple transformation to the dataset.
    """
    return TransformedDataset(dataset, transform)

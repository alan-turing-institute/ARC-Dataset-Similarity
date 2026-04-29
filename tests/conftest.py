from __future__ import annotations

from typing import Any

import pytest
import torch
from pandas import DataFrame
from PIL import Image
from torch.utils.data import Dataset

from dataset_similarity.data.base import ImageDataset


class _TensorImageDataset(ImageDataset):
    """ImageDataset that serves pre-built in-memory tensors — no disk I/O."""

    def __init__(
        self,
        n_samples: int = 20,
        n_classes: int = 3,
        feature_dim: int = 8,
        return_paths: bool = False,
    ) -> None:
        self._n_samples = n_samples
        self._n_classes = n_classes
        torch.manual_seed(0)
        self._tensors = torch.randn(n_samples, feature_dim)
        super().__init__(
            dataset_dir="/tmp",
            target_classes=None,
            split="train",
            size=None,
            random_seed=None,
            embedding=None,
            embedding_dir=None,
            return_paths=return_paths,
        )

    def _load_data(self) -> DataFrame:
        return DataFrame(
            {
                "path": [f"/fake/{i}.jpg" for i in range(self._n_samples)],
                "label": [i % self._n_classes for i in range(self._n_samples)],
            }
        )

    def __getitem__(self, idx: int) -> tuple:
        if self.return_paths:
            return self._tensors[idx], self.data.iloc[idx]["path"]
        return self._tensors[idx], self.data.iloc[idx]["label"]


@pytest.fixture()
def tensor_image_dataset() -> _TensorImageDataset:
    return _TensorImageDataset()


@pytest.fixture()
def tensor_image_dataset_with_paths() -> _TensorImageDataset:
    return _TensorImageDataset(return_paths=True)


class _ImageDataset(Dataset[Any]):
    """Minimal dataset returning ``(image, label)`` tuples."""

    def __init__(self, images: list[Image.Image]) -> None:
        self._data = [(img, 0) for img in images]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return self._data[idx]


@pytest.fixture()
def rgb_images() -> list[Image.Image]:
    """Four small RGB PIL images."""
    return [
        Image.new("RGB", (32, 32), color=(i * 30, i * 60, i * 10)) for i in range(4)
    ]


@pytest.fixture()
def image_dataset(rgb_images: list[Image.Image]) -> _ImageDataset:
    """Dataset wrapping the rgb_images fixture."""
    return _ImageDataset(rgb_images)

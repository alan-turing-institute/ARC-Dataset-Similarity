from __future__ import annotations

from typing import Any

import pytest
from PIL import Image
from torch.utils.data import Dataset


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

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset_similarity.embedding.base import BaseExtractor

_EMBED_DIM = 16


class _DummyExtractor(BaseExtractor):
    """Concrete extractor returning deterministic tensors — no model loading."""

    def __init__(self) -> None:
        super().__init__(model_name="dummy", device="cpu")

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        return torch.zeros(len(images), 3, 32, 32)

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.ones(pixel_values.shape[0], _EMBED_DIM)


def test_extract_dataset_output_shape(image_dataset: Dataset[Any]) -> None:
    extractor = _DummyExtractor()
    result = extractor.extract_dataset(image_dataset, batch_size=2, num_workers=0)
    assert result.shape == (len(image_dataset), _EMBED_DIM)  # type: ignore[arg-type]


def test_extract_dataset_output_dtype(image_dataset: Dataset[Any]) -> None:
    extractor = _DummyExtractor()
    result = extractor.extract_dataset(image_dataset, batch_size=2, num_workers=0)
    assert result.dtype == np.float32


def test_extract_dataset_custom_get_image(rgb_images: list[Image.Image]) -> None:
    """Dataset items are dicts; a custom get_image callable extracts the image."""

    class _DictDataset(Dataset[Any]):
        def __init__(self, imgs: list[Image.Image]) -> None:
            self._data = [{"img": img, "label": 0} for img in imgs]

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return self._data[idx]

    extractor = _DummyExtractor()
    result = extractor.extract_dataset(
        _DictDataset(rgb_images),
        batch_size=2,
        num_workers=0,
        get_image=lambda item: item["img"],
    )
    assert result.shape == (len(rgb_images), _EMBED_DIM)

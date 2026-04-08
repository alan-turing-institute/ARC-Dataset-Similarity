from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import Dataset

from dataset_similarity.embedding.base import BaseExtractor, _embedding_save_path

_EMBED_DIM = 16


class _DummyExtractor(BaseExtractor):
    """Concrete extractor returning deterministic tensors — no model loading."""

    def __init__(self) -> None:
        super().__init__(model_name="dummy", device="cpu")

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        return torch.zeros(len(images), 3, 32, 32)

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.ones(pixel_values.shape[0], _EMBED_DIM)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _PathDataset(Dataset[tuple[Image.Image, str]]):
    """Dataset whose items are ``(image, path_str)`` tuples."""

    def __init__(self, images: list[Image.Image], paths: list[str]) -> None:
        self._data = list(zip(images, paths, strict=True))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[Image.Image, str]:
        return self._data[idx]


# ---------------------------------------------------------------------------
# _embedding_save_path
# ---------------------------------------------------------------------------


def test_save_path_with_dataset_root(tmp_path: Path) -> None:
    src = Path("/data/imagenet/train/n01234/img001.jpg")
    root = Path("/data/imagenet")
    result = _embedding_save_path(tmp_path, src, root, "clip")
    assert result == tmp_path / "clip" / "train" / "n01234" / "img001.safetensors"


def test_save_path_absolute_no_root(tmp_path: Path) -> None:
    src = Path("/data/imagenet/train/n01234/img001.jpg")
    result = _embedding_save_path(tmp_path, src, None, "clip")
    assert result == tmp_path / "clip" / "img001.safetensors"


def test_save_path_relative_no_root(tmp_path: Path) -> None:
    src = Path("cats/img001.jpg")
    result = _embedding_save_path(tmp_path, src, None, "clip")
    assert result == tmp_path / "clip" / "cats" / "img001.safetensors"


def test_save_path_replaces_suffix(tmp_path: Path) -> None:
    src = Path("img.png")
    result = _embedding_save_path(tmp_path, src, None, "clip")
    assert result.suffix == ".safetensors"


# ---------------------------------------------------------------------------
# extract_dataset — no saving
# ---------------------------------------------------------------------------


def test_extract_dataset_returns_none(image_dataset: Dataset[Any]) -> None:
    extractor = _DummyExtractor()
    result = extractor.extract_dataset(image_dataset, batch_size=2, num_workers=0)
    assert result is None


def test_extract_dataset_custom_get_image(rgb_images: list[Image.Image]) -> None:
    """Dataset items are dicts; a custom get_image callable extracts the image."""

    class _DictDataset(Dataset[Any]):
        def __init__(self, imgs: list[Image.Image]) -> None:
            self._data = [{"img": img, "label": 0} for img in imgs]

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
            item = self._data[idx]
            return item["img"], item["label"]

    extractor = _DummyExtractor()
    extractor.extract_dataset(
        _DictDataset(rgb_images),
        batch_size=2,
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# extract_dataset — file saving
# ---------------------------------------------------------------------------


def test_extract_dataset_saves_correct_number_of_files(
    rgb_images: list[Image.Image], tmp_path: Path
) -> None:
    paths = [f"img_{i:04d}.png" for i in range(len(rgb_images))]
    dataset = _PathDataset(rgb_images, paths)
    extractor = _DummyExtractor()
    extractor.extract_dataset(
        dataset,
        batch_size=2,
        num_workers=0,
        output_dir=tmp_path,
    )
    saved = list(tmp_path.rglob("*.safetensors"))
    assert len(saved) == len(rgb_images)


def test_extract_dataset_saved_tensor_shape(
    rgb_images: list[Image.Image], tmp_path: Path
) -> None:
    paths = [f"img_{i:04d}.png" for i in range(len(rgb_images))]
    dataset = _PathDataset(rgb_images, paths)
    extractor = _DummyExtractor()
    extractor.extract_dataset(
        dataset,
        batch_size=2,
        num_workers=0,
        output_dir=tmp_path,
    )
    data = load_file(tmp_path / "dummy" / "img_0000.safetensors")
    assert data["embedding"].shape == (1, _EMBED_DIM)


def test_extract_dataset_mirrors_path_structure(
    rgb_images: list[Image.Image], tmp_path: Path
) -> None:
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "embeddings"
    paths = [
        str(dataset_root / "classA" / f"img_{i:04d}.jpg")
        for i in range(len(rgb_images))
    ]
    dataset = _PathDataset(rgb_images, paths)
    extractor = _DummyExtractor()
    extractor.extract_dataset(
        dataset,
        batch_size=2,
        num_workers=0,
        output_dir=output_dir,
        dataset_root=dataset_root,
    )
    for i in range(len(rgb_images)):
        expected = output_dir / "dummy" / "classA" / f"img_{i:04d}.safetensors"
        assert expected.exists(), f"Missing: {expected}"

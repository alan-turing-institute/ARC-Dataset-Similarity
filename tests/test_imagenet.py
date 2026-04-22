"""Tests for dataset_similarity.data.imagenet."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from dataset_similarity.data.imagenet import ImageNetDataset

IMAGENET_YAML = (
    "n00000001:\n  class_number: 1\n  name: dummy_class_one\n"
    "n00000002:\n  class_number: 2\n  name: dummy_class_two\n"
)


@pytest.fixture()
def imagenet_dir(tmp_path: Path) -> Path:
    """Create a minimal fake ImageNet directory structure.

    Layout mirrors the real data layout so the dataset can locate metadata::

        tmp_path/
          metadata/
            imagenet_class_mapping.yaml
          ImageNet/          <- returned as dataset_dir
            train/
              n00000001/image_00000.JPEG
              n00000002/image_00000.JPEG
            val/
              n00000001/image_00000.JPEG
              n00000002/image_00000.JPEG
    """
    classes = ["n00000001", "n00000002"]
    dataset_dir = tmp_path / "ImageNet"

    for split in ("train", "val"):
        for label, class_name in enumerate(classes):
            class_dir = dataset_dir / split / class_name
            class_dir.mkdir(parents=True)
            img_path = class_dir / "image_00000.JPEG"
            Image.new("RGB", (64, 64), color=(label * 80, 120, 200)).save(img_path)

    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "imagenet_class_mapping.yaml").write_text(IMAGENET_YAML)

    return dataset_dir


class TestImageNetDataset:
    def test_invalid_split(self, imagenet_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"No such file or directory.*test"):
            ImageNetDataset(dataset_dir=imagenet_dir, split="test")  # type: ignore[arg-type]

    def test_missing_split_dir(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "ImageNet"
        dataset_dir.mkdir()
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()
        (metadata_dir / "imagenet_class_mapping.yaml").write_text(IMAGENET_YAML)
        with pytest.raises(
            FileNotFoundError, match=r"No such file or directory.*train"
        ):
            ImageNetDataset(dataset_dir=dataset_dir, split="train")

    def test_empty_split_dir(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "ImageNet"
        (dataset_dir / "train").mkdir(parents=True)
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()
        (metadata_dir / "imagenet_class_mapping.yaml").write_text(IMAGENET_YAML)
        with pytest.raises(
            FileNotFoundError, match=r"No such file or directory.*train"
        ):
            ImageNetDataset(dataset_dir=dataset_dir, split="train")

    def test_len(self, imagenet_dir: Path) -> None:
        dataset = ImageNetDataset(dataset_dir=imagenet_dir, split="train")
        assert len(dataset) == 2

    def test_getitem_shape(self, imagenet_dir: Path) -> None:
        dataset = ImageNetDataset(dataset_dir=imagenet_dir, split="train")
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape[0] == 3  # C, H, W
        assert isinstance(label, np.int64)

    def test_num_classes(self, imagenet_dir: Path) -> None:
        dataset = ImageNetDataset(dataset_dir=imagenet_dir, split="train")
        assert dataset.num_classes == 2

    def test_val_split(self, imagenet_dir: Path) -> None:
        dataset = ImageNetDataset(dataset_dir=imagenet_dir, split="val")
        assert len(dataset) == 2

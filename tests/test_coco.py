"""Tests for dataset_similarity.data.coco."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from dataset_similarity.data.coco import COCODataset

COCO_YAML = (
    "1:\n  name: cat\n  supercategory: animal\n"
    "2:\n  name: dog\n  supercategory: animal\n"
)


def _make_annotations(
    categories: list[dict],
    images: list[dict],
    annotations: list[dict],
) -> str:
    return json.dumps(
        {
            "info": {},
            "licenses": [],
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }
    )


@pytest.fixture()
def coco_dir(tmp_path: Path) -> Path:
    """Create a minimal fake COCO directory structure.

    Layout mirrors the real data layout so the dataset can locate metadata::

        tmp_path/
          metadata/
            coco_class_mapping.yaml
          COCO/          <- returned as dataset_dir
            annotations/
              instances_train2017.json
            train2017/
              image_001.jpg
              image_002.jpg
    """
    dataset_dir = tmp_path / "COCO"
    split = "train2017"

    images_dir = dataset_dir / split
    images_dir.mkdir(parents=True)
    for i, color in enumerate([(100, 150, 200), (200, 100, 150)]):
        Image.new("RGB", (64, 64), color=color).save(
            images_dir / f"image_{i + 1:03d}.jpg"
        )

    annotations_dir = dataset_dir / "annotations"
    annotations_dir.mkdir()
    (annotations_dir / f"instances_{split}.json").write_text(
        _make_annotations(
            categories=[
                {"id": 1, "name": "cat", "supercategory": "animal"},
                {"id": 2, "name": "dog", "supercategory": "animal"},
            ],
            images=[
                {"id": 1, "file_name": "image_001.jpg", "height": 64, "width": 64},
                {"id": 2, "file_name": "image_002.jpg", "height": 64, "width": 64},
            ],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 32, 32],
                    "area": 1024,
                    "iscrowd": 0,
                    "segmentation": [],
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 2,
                    "bbox": [0, 0, 32, 32],
                    "area": 1024,
                    "iscrowd": 0,
                    "segmentation": [],
                },
            ],
        )
    )

    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "coco_class_mapping.yaml").write_text(COCO_YAML)

    return dataset_dir


def test_invalid_split(coco_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"test2017"):
        COCODataset(dataset_dir=coco_dir, split="test2017")  # type: ignore[arg-type]


def test_invalid_class(coco_dir: Path) -> None:
    with pytest.raises(ValueError, match="Unknown class 'invalid'"):
        COCODataset(dataset_dir=coco_dir, target_classes=["invalid"])


def test_invalid_superclass(coco_dir: Path) -> None:
    with pytest.raises(ValueError, match="Unknown supercategory 'vehicle'"):
        COCODataset(dataset_dir=coco_dir, target_superclasses=["vehicle"])


def test_len(coco_dir: Path) -> None:
    dataset = COCODataset(dataset_dir=coco_dir)
    assert len(dataset) == 2


def test_getitem_shape(coco_dir: Path) -> None:
    dataset = COCODataset(dataset_dir=coco_dir)
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 3  # C, H, W
    assert isinstance(label, np.int64)


def test_num_classes(coco_dir: Path) -> None:
    dataset = COCODataset(dataset_dir=coco_dir)
    assert dataset.num_classes == 2


def test_target_classes(coco_dir: Path) -> None:
    dataset = COCODataset(dataset_dir=coco_dir, target_classes=["cat"])
    assert len(dataset) == 1
    assert dataset.num_classes == 1


def test_target_superclasses(coco_dir: Path) -> None:
    dataset = COCODataset(dataset_dir=coco_dir, target_superclasses=["animal"])
    assert len(dataset) == 2
    assert dataset.num_classes == 2

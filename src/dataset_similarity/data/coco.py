from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import pandas as pd
from pycocotools.coco import COCO

from dataset_similarity.constants import COCO_DIR, DEFAULT_EMBEDDING_DIR
from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.utils import load_yaml_from_path


class COCODataset(ImageDataset):
    """
    PyTorch dataset for `MS COCO <https://cocodataset.org/>`_.

    Args:
        dataset_dir: Absolute path to the dataset directory containing COCO images.
            Defaults to `dataset_similarity.constants.COCO_DIR`.
        target_classes: List of class names to include in the dataset. If None, all
            classes are included. Elements must be valid class names as specified in the
            class mapping file at
            `dataset_dir.parent / "metadata/coco_class_mapping.yaml"`. Defaults to
            `None`.
        split: Dataset split identifier. Must be `"train2017"` or `"val2017"`.
            Defaults to `"train2017"`.
        size: If a float in `(0, 1)`, the fraction of samples to retain.
            If a positive integer, the exact number of samples to retain. If
            `None`, no subsampling is performed and the full dataset is used. Defaults
            to `None`.
        random_seed: Seed for the random number generator, for reproducibility. If
            `None`, the result is non-deterministic. Defaults to `None`.
        embedding: If not `None`, the name of the embedding model to use for this
            dataset. If `None`, raw images are returned by `__getitem__`. Defaults to
            `None`.
        embedding_dir: The absolute path to the directory where the embeddings are
            stored. This is used to compute the path to the embedding for each image.
            Must be provided if `embedding` is not None. Defaults to
            `dataset_similarity.constants.DEFAULT_EMBEDDING_DIR`.
        return_paths: If `True`, `__getitem__` returns a tuple of (tensor, path)
            instead of (tensor, label). The path is returned as a `Path` object.
            Defaults to `False`.
    """

    def __init__(
        self,
        dataset_dir: str | Path = COCO_DIR,
        target_classes: list[str] | None = None,
        split: Literal["train2017", "val2017"] = "train2017",
        size: float | int | None = None,
        random_seed: int | None = None,
        embedding: None | str = None,
        embedding_dir: None | Path | str = DEFAULT_EMBEDDING_DIR,
        return_paths: bool = False,
    ) -> None:
        # Target classes also need to be processed before calling super().__init__()
        self.label_to_meta_map: dict[int, dict[str, str]] = cast(
            dict[int, dict[str, str]],
            load_yaml_from_path(
                Path(dataset_dir).parent / "metadata" / "coco_class_mapping.yaml"
            ),
        )
        self.name_to_label_map: dict[str, int] = {
            meta["name"]: label for label, meta in self.label_to_meta_map.items()
        }
        self.supercategory_to_classnumber_map: dict[str, list[int]] = {}
        for label, meta in self.label_to_meta_map.items():
            supercategory = meta["supercategory"]
            if supercategory not in self.supercategory_to_classnumber_map:
                self.supercategory_to_classnumber_map[supercategory] = []
            self.supercategory_to_classnumber_map[supercategory].append(label)

        self.classnumber_to_name_map: dict[int, str] = {
            label: meta["name"] for label, meta in self.label_to_meta_map.items()
        }

        if target_classes is not None:
            resolved: list[str] = []
            for cls in target_classes:
                if cls in self.name_to_label_map:
                    resolved.append(cls)
                else:
                    err_msg = f"Unknown class {cls}: use a class name or supercategory"
                    raise ValueError(err_msg)
            target_classes = resolved

        super().__init__(
            dataset_dir=dataset_dir,
            target_classes=target_classes,
            split=split,
            size=size,
            random_seed=random_seed,
            embedding=embedding,
            embedding_dir=embedding_dir,
            return_paths=return_paths,
        )

    def _load_data(self) -> pd.DataFrame:
        ann_file = self.dataset_dir / "annotations" / f"instances_{self.split}.json"
        coco = COCO(str(ann_file))

        if self.target_classes is None:
            cat_ids = coco.getCatIds()
        else:
            cat_ids = coco.getCatIds(catNms=self.target_classes)

        rows: list[dict[str, Path | int]] = []
        for cat_id in cat_ids:
            img_ids = coco.getImgIds(catIds=[cat_id])
            for img_info in coco.loadImgs(img_ids):
                img_path = self.dataset_dir / self.split / img_info["file_name"]
                rows.append({"path": img_path, "label": cat_id})

        return pd.DataFrame(rows, columns=["path", "label"])

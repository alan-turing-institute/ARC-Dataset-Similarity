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
        target_superclasses: List of supercategory names whose member classes are all
            included. Expands to the corresponding class names before filtering, and is
            merged with `target_classes` when both are provided. If None, no
            supercategory filtering is applied. Defaults to `None`.
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
        target_superclasses: list[str] | None = None,
        split: Literal["train2017", "val2017"] = "train2017",
        drop_duplicates: bool = True,
        size: float | int | None = None,
        random_seed: int | None = None,
        embedding: None | str = None,
        embedding_dir: None | Path | str = DEFAULT_EMBEDDING_DIR,
        return_paths: bool = False,
    ) -> None:
        self.drop_duplicates = drop_duplicates
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
            self.supercategory_to_classnumber_map.setdefault(
                meta["supercategory"], []
            ).append(label)

        if target_superclasses is not None:
            for sc in target_superclasses:
                if sc not in self.supercategory_to_classnumber_map:
                    err_msg = f"Unknown supercategory '{sc}'"
                    raise ValueError(err_msg)
            expanded = [
                self.label_to_meta_map[label]["name"]
                for sc in target_superclasses
                for label in self.supercategory_to_classnumber_map[sc]
            ]
            if target_classes is None:
                target_classes = expanded
            else:
                target_classes = list(dict.fromkeys(target_classes + expanded))

        if target_classes is not None:
            for cls in target_classes:
                if cls not in self.name_to_label_map:
                    err_msg = f"Unknown class '{cls}'"
                    raise ValueError(err_msg)

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

        cat_img_ids: dict[int, list[int]] = {
            cat_id: coco.getImgIds(catIds=[cat_id]) for cat_id in cat_ids
        }

        df = pd.DataFrame(
            {"path": self.dataset_dir / self.split / coco.imgs[img_id]["file_name"], "label": cat_id} 
            for cat_id, img_ids in cat_img_ids.items()
            for img_id in img_ids
        )
        if self.drop_duplicates:
            df = df.drop_duplicates(subset="path", keep="first").reset_index(drop=True)
        return df

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pycocotools.coco import COCO

from dataset_similarity.constants import COCO_DIR, DEFAULT_EMBEDDING_DIR
from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.coco import (
    _load_coco_with_annotations,
    split_coco_image_ids,
)

# from dataset_similarity.data.coco_task import ClassificationTask
from dataset_similarity.data.utils import load_yaml_from_path


class COCOTaskDataset(ImageDataset):
    """A binary classification dataset built from a ``COCOTaskPartition``.

    Conforms to the ``ImageDataset`` interface so that all existing tooling
    (metrics, embedding extraction, YAML configs) works without modification.
    Instantiated by ``COCOTaskPartition.create_dataset``; not intended to be constructed
    directly.

    Args:
        data: Pre-built DataFrame with columns ``["path", "label"]``.
        dataset_dir: Path to the COCO dataset directory (used for embedding path
            resolution).
        embedding: Name of the embedding model, or ``None`` for raw images.
        embedding_dir: Directory where embeddings are stored.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        dataset_dir: Path,
        embedding: str | None = None,
        embedding_dir: Path | str | None = DEFAULT_EMBEDDING_DIR,
    ) -> None:
        self._prebuilt_data = data
        super().__init__(
            dataset_dir=dataset_dir,
            target_classes=None,
            split="",
            size=None,
            random_seed=None,
            embedding=embedding,
            embedding_dir=embedding_dir,
            return_paths=False,
        )

    def _load_data(self) -> pd.DataFrame:
        return self._prebuilt_data


class COCOTaskPartition:
    """Manages the task-dataset partition of COCO and creates classification datasets.

    Partitions COCO image IDs deterministically into a large unlabelled data store and
    a task pool, then provides a factory method for building binary classification
    datasets from the task pool.

    Args:
        dataset_dir: Path to the COCO dataset directory.
        split: COCO split to partition (``"train2017"`` or ``"val2017"``).
        random_seed: Seed for the deterministic partition. Fix this to ensure the
            same images are always in the same split.
        embedding: Embedding model name passed through to created datasets, or
            ``None`` for raw images.
        embedding_dir: Directory where embeddings are stored.
        ratio: Four-tuple ``(train, val, test, datastore)`` of fractions of the
            total image pool assigned to each split. Must sum to 1.
        positive_fraction: Target fraction of positive examples in created
            datasets. The majority class is downsampled to match. ``None``
            disables rebalancing.
        min_bbox_area_fraction: Minimum ``max_bbox_area_frac`` for a positive
            image to be included. ``None`` disables the filter.
        max_bbox_area_fraction: Maximum ``max_bbox_area_frac`` for a positive
            image to be included. ``None`` disables the filter.
        min_objects_per_image: Minimum ``n_total_instances`` for an image to be
            included. ``None`` disables the filter.
        max_objects_per_image: Maximum ``n_total_instances`` for an image to be
            included. ``None`` disables the filter.
    """

    def __init__(
        self,
        dataset_dir: str | Path = COCO_DIR,
        split: Literal["train2017", "val2017"] = "train2017",
        random_seed: int | None = None,
        embedding: None | str = None,
        embedding_dir: None | Path | str = DEFAULT_EMBEDDING_DIR,
        ratio: tuple[float, float, float, float] = (0.5, 0.1, 0.2, 0.2),
        positive_fraction: float | None = None,
        min_bbox_area_fraction: float | None = None,
        max_bbox_area_fraction: float | None = None,
        min_objects_per_image: int | None = None,
        max_objects_per_image: int | None = None,
    ) -> None:
        self.embedding = embedding
        self.embedding_dir = embedding_dir
        self._dataset_dir = Path(dataset_dir)
        self._split = split
        self.random_seed = random_seed
        self.positive_fraction = positive_fraction
        self.min_bbox_area_fraction = min_bbox_area_fraction
        self.max_bbox_area_fraction = max_bbox_area_fraction
        self.min_objects_per_image = min_objects_per_image
        self.max_objects_per_image = max_objects_per_image

        train_r, val_r, _, datastore_r = ratio
        ann_file = self._dataset_dir / "annotations" / f"instances_{split}.json"
        self.coco_annots = COCO(str(ann_file))

        datastore_ids, pool_ids = split_coco_image_ids(
            self.coco_annots, sum(ratio[:-1]), random_seed, datastore_r > 0.0
        )
        self._data = _load_coco_with_annotations(
            self.coco_annots, self._dataset_dir, split
        )
        self._label_to_meta, self._class_to_label, self._supercategory_to_labels = (
            self._build_class_mappings()
        )
        self._train_ids, self._val_ids, self._test_ids = self._partition_pool_ids(
            pool_ids, train_r, val_r, random_seed
        )
        self._datastore_ids = datastore_ids

    def _build_class_mappings(
        self,
    ) -> tuple[dict[int, dict[str, str]], dict[str, int], dict[str, list[int]]]:
        label_to_meta: dict[int, dict[str, str]] = cast(
            dict[int, dict[str, str]],
            load_yaml_from_path(
                self._dataset_dir.parent / "metadata" / "coco_class_mapping.yaml"
            ),
        )
        class_to_label: dict[str, int] = {
            meta["name"]: label for label, meta in label_to_meta.items()
        }
        supercategory_to_labels: dict[str, list[int]] = {}
        for label, meta in label_to_meta.items():
            supercategory_to_labels.setdefault(meta["supercategory"], []).append(label)
        return label_to_meta, class_to_label, supercategory_to_labels

    def _partition_pool_ids(
        self,
        pool_ids: set[int],
        train_r: float,
        val_r: float,
        random_seed: int | None,
    ) -> tuple[set[int], set[int], set[int]]:
        shuffled = np.random.default_rng(
            None if random_seed is None else random_seed + 1
        ).permutation(np.array(sorted(pool_ids)))
        n = len(shuffled)
        n_train = int(train_r * n)
        n_val = int(val_r * n)
        train_ids: set[int] = set(shuffled[:n_train].tolist())
        val_ids: set[int] = set(shuffled[n_train : n_train + n_val].tolist())
        test_ids: set[int] = set(shuffled[n_train + n_val :].tolist())
        return train_ids, val_ids, test_ids

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "COCOTaskPartition":
        """Construct a ``COCOBuilder`` from a plain config dictionary."""
        return cls(**cfg)

    def create_dataset(
        self,
        pool_split: Literal["train", "val", "test", "datastore"],
        positive_classes: list[str] | None = None,
        positive_superclasses: list[str] | None = None,
    ) -> COCOTaskDataset:
        """Build a binary classification dataset from the task pool.

        Args:
            pool_split: Which sub-split to draw images from.
            positive_classes: COCO class names whose images are labelled 1.
            positive_superclasses: COCO supercategory names; all member classes
                are treated as positive. At least one of ``positive_classes`` or
                ``positive_superclasses`` must be provided.

        Returns:
            A ``COCOTaskDataset`` ready for iteration.
        """
        split_ids = {
            "train": self._train_ids,
            "val": self._val_ids,
            "test": self._test_ids,
            "datastore": self._datastore_ids,
        }[pool_split]

        if split_ids is None:
            msg = f"Requested pool_split '{pool_split}' not available."
            raise ValueError(msg)

        df = self._data[self._data["image_id"].isin(split_ids)]
        positive_labels = self._resolve_positive_labels(
            positive_classes, positive_superclasses
        )

        # --- Build positive image set ---
        pos_rows = df[df["label"].isin(positive_labels)].copy()
        if self.min_bbox_area_fraction is not None:
            pos_rows = pos_rows[
                pos_rows["max_bbox_area_frac"] >= self.min_bbox_area_fraction
            ]
        if self.max_bbox_area_fraction is not None:
            pos_rows = pos_rows[
                pos_rows["max_bbox_area_frac"] <= self.max_bbox_area_fraction
            ]
        if self.min_objects_per_image is not None:
            pos_rows = pos_rows[
                pos_rows["n_total_instances"] >= self.min_objects_per_image
            ]
        if self.max_objects_per_image is not None:
            pos_rows = pos_rows[
                pos_rows["n_total_instances"] <= self.max_objects_per_image
            ]

        # One row per positive image — keep the annotation with the largest bbox
        pos_df = (
            pos_rows.sort_values("max_bbox_area_frac", ascending=False)
            .drop_duplicates(subset="image_id", keep="first")
            .copy()
        )
        pos_df["label"] = 1
        positive_image_ids: set[int] = set(pos_df["image_id"].tolist())

        # --- Build negative image set ---
        neg_rows = df[~df["image_id"].isin(positive_image_ids)]

        if self.min_objects_per_image is not None:
            valid_ids = neg_rows[
                neg_rows["n_total_instances"] >= self.min_objects_per_image
            ]["image_id"]
            neg_rows = neg_rows[neg_rows["image_id"].isin(valid_ids)]
        if self.max_objects_per_image is not None:
            valid_ids = neg_rows[
                neg_rows["n_total_instances"] <= self.max_objects_per_image
            ]["image_id"]
            neg_rows = neg_rows[neg_rows["image_id"].isin(valid_ids)]

        # One row per negative image — keep the annotation with the largest bbox
        neg_df = (
            neg_rows.sort_values("max_bbox_area_frac", ascending=False)
            .drop_duplicates(subset="image_id", keep="first")
            .copy()
        )
        neg_df["label"] = 0

        combined = pd.concat(
            [
                pos_df[["image_id", "path", "label"]],
                neg_df[["image_id", "path", "label"]],
            ]
        ).reset_index(drop=True)

        if self.positive_fraction is not None:
            combined = self._balance_classes(
                combined, self.positive_fraction, self.random_seed
            )

        task_df = combined[["path", "label"]].reset_index(drop=True)
        return COCOTaskDataset(
            data=task_df,
            dataset_dir=self._dataset_dir,
            embedding=self.embedding,
            embedding_dir=self.embedding_dir,
        )

    def _resolve_positive_labels(
        self,
        positive_classes: list[str] | None,
        positive_superclasses: list[str] | None,
    ) -> set[int]:
        labels: set[int] = set()
        if positive_classes is not None:
            for cls in positive_classes:
                if cls not in self._class_to_label:
                    msg = f"Unknown class '{cls}'"
                    raise ValueError(msg)
                labels.add(self._class_to_label[cls])
        if positive_superclasses is not None:
            for supercat in positive_superclasses:
                if supercat not in self._supercategory_to_labels:
                    msg = f"Unknown supercategory '{supercat}'"
                    raise ValueError(msg)
                labels.update(self._supercategory_to_labels[supercat])
        if not labels:
            msg = "ClassificationTask must specify at least one positive class"
            raise ValueError(msg)
        return labels

    def _balance_classes(
        self, df: pd.DataFrame, positive_fraction: float, random_seed: int | None
    ) -> pd.DataFrame:
        """Downsample the majority class to achieve ``positive_fraction``."""
        if not (0 < positive_fraction < 1):
            return df
        pos_df = df[df["label"] == 1]
        neg_df = df[df["label"] == 0]
        n_pos = len(pos_df)
        n_neg = len(neg_df)

        # Downsample whichever class is over-represented
        target_neg = int(n_pos * (1 - positive_fraction) / positive_fraction)
        target_pos = int(n_neg * positive_fraction / (1 - positive_fraction))

        if target_neg <= n_neg:
            neg_df = neg_df.sample(n=target_neg, random_state=random_seed)
        elif target_pos <= n_pos:
            pos_df = pos_df.sample(n=target_pos, random_state=random_seed)

        # deterministically shuffle the combined set
        return (
            pd.concat([pos_df, neg_df])
            .sample(frac=1, random_state=random_seed)
            .reset_index(drop=True)
        )

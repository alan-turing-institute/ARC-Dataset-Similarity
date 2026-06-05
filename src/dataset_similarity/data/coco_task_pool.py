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
    """A binary classification dataset built from a ``COCOTaskPool``.

    Conforms to the ``ImageDataset`` interface so that all existing tooling
    (metrics, embedding extraction, YAML configs) works without modification.
    Instantiated by ``COCOTaskPool.create_dataset``; not intended to be constructed
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


class COCOTaskPool:
    """Manages the task-dataset partition of COCO and creates classification datasets.

    Partitions COCO image IDs deterministically into a large unlabelled data store and
    a smaller task pool, then provides a factory method for building binary
    classification datasets from the task pool with fine-grained sampling control.

    The data store partition can be loaded separately as::

        datastore_ids, pool_ids = split_coco_image_ids(
            dataset_dir, split, task_pool_size, random_seed
        )
        datastore = COCODataset(image_ids=datastore_ids, return_paths=True)

    Args:
        dataset_dir: Path to the COCO dataset directory.
        split: COCO split to partition (``"train2017"`` or ``"val2017"``).
        task_pool_size: Fraction ``(0, 1)`` or exact count of images reserved for the
            task pool. The remainder becomes the data store.
        random_seed: Seed for the deterministic partition. Fix this to ensure the
            same images are always in the task pool.
        train_val_test_ratio: Proportions for further splitting the task pool into
            train / val / test sub-pools. Must sum to 1.
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
        # include_datastore: bool = False,
    ) -> None:
        # sampling configs
        self.min_bbox_area_fraction = min_bbox_area_fraction
        self.max_bbox_area_fraction = max_bbox_area_fraction
        self.min_objects_per_image = min_objects_per_image
        self.max_objects_per_image = max_objects_per_image
        self.positive_fraction = positive_fraction

        self.embedding = embedding
        self.embedding_dir = embedding_dir

        self._dataset_dir = Path(dataset_dir)
        self._split = split
        self.random_seed = random_seed
        task_pool_size = sum(ratio[:-1])
        include_datastore = ratio[3] > 0.0

        ann_file = self._dataset_dir / "annotations" / f"instances_{split}.json"
        self.coco_annots = COCO(str(ann_file))

        datastore_ids, pool_ids = split_coco_image_ids(
            self.coco_annots, task_pool_size, random_seed, include_datastore
        )

        self._data = _load_coco_with_annotations(
            self.coco_annots, self._dataset_dir, split
        )

        label_to_meta: dict[int, dict[str, str]] = cast(
            dict[int, dict[str, str]],
            load_yaml_from_path(
                self._dataset_dir.parent / "metadata" / "coco_class_mapping.yaml"
            ),
        )
        self._label_to_meta = label_to_meta
        self._class_to_label: dict[str, int] = {
            meta["name"]: label for label, meta in label_to_meta.items()
        }
        self._supercategory_to_labels: dict[str, list[int]] = {}
        for label, meta in label_to_meta.items():
            self._supercategory_to_labels.setdefault(meta["supercategory"], []).append(
                label
            )

        # Deterministically split pool image IDs into train/val/test
        all_pool_ids = np.array(sorted(pool_ids))
        rng = np.random.default_rng(None if random_seed is None else random_seed + 1)
        shuffled = rng.permutation(all_pool_ids)
        n = len(shuffled)
        train_r, val_r, test_r, datastore_r = ratio
        n_train = int(train_r * n)
        n_val = int(val_r * n)
        self._train_ids: set[int] = set(shuffled[:n_train].tolist())
        self._val_ids: set[int] = set(shuffled[n_train : n_train + n_val].tolist())
        self._test_ids: set[int] = set(shuffled[n_train + n_val :].tolist())
        self._datastore_ids = datastore_ids

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "COCOTaskPool":
        """Construct a ``COCOBuilder`` from a plain config dictionary."""
        return cls(**cfg)

    def create_dataset(
        self,
        pool_split: Literal["train", "val", "test", "datastore"],
        positive_classes: list[str] | None = None,
        positive_superclasses: list[str] | None = None,
    ) -> COCOTaskDataset:
        """Build a binary classification dataset from the task pool.

        The pipeline applied in order:
        1. Filter to the requested pool sub-split (train / val / test).
        2. Identify positive images (those containing any ``task.positive_classes``
           annotation that passes size/count filters).
        3. Identify negative images (those not in the positive set), apply object-count
           filters, and manage related-class composition.
        4. Enforce class balance via ``sampling.positive_fraction``.
        5. Subsample to ``sampling.size``.

        Args:
            task: Specification of the classification task (positive class, negative
                class management).
            pool_split: Which sub-split of the task pool to draw from.
            sampling: Annotation-level and class-balance sampling controls. ``None``
                means no filtering or rebalancing.
            embedding: Embedding model name for the returned dataset, or ``None`` for
                raw images.
            embedding_dir: Directory where embeddings are stored.

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

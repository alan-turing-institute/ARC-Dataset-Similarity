from typing import Any, Literal, cast

import pandas as pd
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split

from dataset_similarity.constants import COCO_DIR, DATA_DIR
from dataset_similarity.data.base import ImageDataset
from dataset_similarity.utils import load_yaml_from_path


class COCODataset(ImageDataset):
    """
    PyTorch dataset for `MS COCO <https://cocodataset.org/>`_.

    Args:
        split: Dataset split identifier. Any of "train2017", "val2017", "trainARC",
            "valARC", "testARC", or "store". Defaults to "train2017".
        size: If a float in `(0, 1)`, the fraction of samples to retain.
            If a positive integer, the exact number of samples to retain. If
            `None`, no subsampling is performed and the full dataset is used. Defaults
            to `None`.
        random_seed: Seed for the random number generator, for reproducibility. If
            `None`, the result is non-deterministic. Defaults to `None`.
        embedding: If not `None`, the name of the embedding model to use for this
            dataset. If `None`, raw images are returned by `__getitem__`. Defaults to
            `None`.
        return_paths: If `True`, `__getitem__` returns a tuple of (tensor, path)
            instead of (tensor, label). The path is returned as a `Path` object.
            Defaults to `False`.
    """

    def __init__(
        self,
        split: str = "trainARC",
        size: float | int | None = None,
        random_seed: int | None = None,
        embedding: None | str = None,
        return_paths: bool = False,
        positive_class: list[str] | None = None,
        positive_superclass: list[str] | None = None,
        negative_class: list[str] | None = None,
        negative_superclass: list[str] | None = None,
        min_objects_per_image: int | None = None,
        max_objects_per_image: int | None = None,
        min_bbox_area_fraction: float | None = None,
        max_bbox_area_fraction: float | None = None,
        positive_fraction: float | None = None,
        filter_class: Literal["positive", "negative"] | None = None,
    ) -> None:
        # Classes need to be processed before calling super.__init__
        self.label_to_meta_map: dict[int, dict[str, str]] = cast(
            dict[int, dict[str, str]],
            load_yaml_from_path(DATA_DIR / "metadata" / "coco_class_mapping.yaml"),
        )
        self.class_to_label_map: dict[str, int] = {
            meta["name"]: label for label, meta in self.label_to_meta_map.items()
        }
        self.supercategory_to_classnumber_map: dict[str, list[int]] = {}
        for label, meta in self.label_to_meta_map.items():
            self.supercategory_to_classnumber_map.setdefault(
                meta["supercategory"], []
            ).append(label)

        # Validate class inputs
        positive_passed = positive_class is not None or positive_superclass is not None
        negative_passed = negative_class is not None or negative_superclass is not None

        if negative_passed and (not positive_passed):
            msg = (
                "If negative classes are specified, positive classes must also be "
                "specified."
            )
            raise ValueError(msg)

        if (positive_fraction is not None) and (not positive_passed):
            msg = (
                "If `positive_fraction` is specified, positive classes must also be"
                " specified."
            )
            raise ValueError(msg)

        # Define classes
        self.positive_class = self._prepare_classes(positive_class, positive_superclass)
        _negative_class: list[int] | None = self._prepare_classes(
            negative_class, negative_superclass
        )
        if isinstance(_negative_class, list) and isinstance(self.positive_class, list):
            _negative_class = [
                cls for cls in _negative_class if cls not in self.positive_class
            ]
        self.negative_class = _negative_class

        # Whether we need to drop classes depends on if all classes are negative
        # TODO: bodge for now as super wants string class names
        keep_labels, keep_classes = None, None
        if self.negative_class is not None:
            keep_labels = (self.positive_class or []) + self.negative_class
            keep_classes = [
                cls
                for cls, _ in self.class_to_label_map.items()
                if label in keep_labels
            ]
        self.keep_labels = keep_labels

        # Task settings
        self.min_objects_per_image = min_objects_per_image
        self.max_objects_per_image = max_objects_per_image
        self.min_bbox_area_fraction = min_bbox_area_fraction
        self.max_bbox_area_fraction = max_bbox_area_fraction
        self.positive_fraction = positive_fraction
        self.filter_class = filter_class

        # Super will use keep_classes to filter the dataset by class
        super().__init__(
            dataset_dir=COCO_DIR,
            keep_classes=keep_classes,
            split=split,
            size=size,
            random_seed=random_seed,
            embedding=embedding,
            return_paths=return_paths,
        )

    def _load_data(self) -> pd.DataFrame:
        ann_file = self.dataset_dir / "annotations" / f"instances_{self.split}.json"
        coco = COCO(str(ann_file))
        df = pd.DataFrame(
            _get_row_from_img_id(img_id, coco) for img_id in coco.getImgIds()
        )

        # Filter dataset for keep classes
        if self.keep_labels is not None:
            df = df[df.cats.apply(lambda x: any(cls in x for cls in self.keep_labels))]

        # Create binary task if requested
        if self.positive_class is not None:
            df["label"] = df.cats.apply(
                lambda x: int(any(cls in x for cls in self.positive_class))
            )

        # Apply object/bbox filters to the target subset (or all rows)
        if self.filter_class is not None and "label" in df.columns:
            target_label = 1 if self.filter_class == "positive" else 0
            keep = self._apply_object_filters(df[df["label"] == target_label])
            return pd.concat([keep, df[df["label"] != target_label]]).sort_index()
        return self._apply_object_filters(df)

    def _apply_object_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.min_objects_per_image is not None:
            df = df[df.num_objects >= self.min_objects_per_image]
        if self.max_objects_per_image is not None:
            df = df[df.num_objects <= self.max_objects_per_image]
        if self.min_bbox_area_fraction is not None:
            df = df[df.min_bbox_frac >= self.min_bbox_area_fraction]
        if self.max_bbox_area_fraction is not None:
            df = df[df.max_bbox_frac <= self.max_bbox_area_fraction]
        return df

    def _prepare_classes(
        self,
        classes: list[str] | None,
        superclasses: list[str] | None,
    ) -> list[int] | None:
        """
        Helper fn for initialising positive and negative classes based on whether
        individual or superclasses are specified.

        Args:
            classes (list[str] | None): _description_
            superclasses (list[str] | None): _description_

        Returns:
            list[int] | None: _description_
        """
        if classes is None and superclasses is None:
            return None
        return list(
            {self.class_to_label_map[cls] for cls in (classes or [])}
            | {
                label
                for sc in (superclasses or [])
                for label in self.supercategory_to_classnumber_map[sc]
            }
        )

    def subsample_data(self) -> pd.DataFrame:
        """
        Overwrite of super method as strata need to be defined on "label" column,
        which may not exist in the COCO dataset depending on kwargs. Additionally
        implements class balancing.

        Replaces ``self.data`` in-place with a random stratified subset.

        Args:
            size: If a float in ``(0, 1)``, the fraction of samples to retain.
                If a positive integer, the exact number of samples to retain.
            random_seed: Seed for the random number generator, for
                reproducibility. If ``None``, the result is non-deterministic.

        Returns:
            The new ``self.data`` DataFrame after resampling.
        """
        self._strip_single_classes_from_samples()

        if self.positive_fraction is not None:
            self._balance_classes()

        _, new_data = train_test_split(
            self.data,
            test_size=self.size,
            stratify=self.data["label"] if self.positive_class is not None else None,
            random_state=self.random_seed,
        )

        return new_data.reset_index(drop=True)

    def _balance_classes(self) -> None:
        """Downsample the majority class to achieve ``positive_fraction``."""
        if self.positive_fraction is None or not (0 < self.positive_fraction < 1):
            return
        pos_df = self.data[self.data["label"] == 1]
        neg_df = self.data[self.data["label"] == 0]
        n_pos, n_neg = len(pos_df), len(neg_df)

        # Downsample whichever class is over-represented
        target_neg = int(n_pos * (1 - self.positive_fraction) / self.positive_fraction)
        target_pos = int(n_neg * self.positive_fraction / (1 - self.positive_fraction))

        if target_neg <= n_neg:
            neg_df = neg_df.sample(n=target_neg, random_state=self.random_seed)
        elif target_pos <= n_pos:
            pos_df = pos_df.sample(n=target_pos, random_state=self.random_seed)

        # deterministically shuffle the combined set
        self.data = (
            pd.concat([pos_df, neg_df])
            .sample(frac=1, random_state=self.random_seed)
            .reset_index(drop=True)
        )

    def _filter(self) -> None:
        # df.category_id.apply(lambda row: positive_class in row)
        return None


def _get_row_from_img_id(img_id: int, coco: COCO) -> dict[str, Any]:
    """
    Helper fn which given a COCO image ID, returns a dictionary containing keys for
    the img_id, number of categories, number of objects, and object areas
    (each one corresponding to the categories).

    Args:
        img_id: The COCO image ID to retrieve information for.
        coco: The COCO object containing the dataset annotations
        split: The split to which the image belongs

    Returns:
        dict: The dictionary containing the image information
    """
    path = COCO_DIR / "images" / coco.imgs[img_id]["file_name"]
    ann = coco.imgToAnns[img_id]
    cats = [ann["category_id"] for ann in ann]
    img_info = coco.imgs[img_id]
    img_area = img_info["width"] * img_info["height"]
    areas = [ann["area"] for ann in ann]
    return {
        "img_id": img_id,
        "path": path,
        "cats": cats,
        "num_objects": len(cats),
        "img_area": img_area,
        "min_bbox_frac": min(areas) / img_area if len(areas) > 0 else 0,
        "max_bbox_frac": max(areas) / img_area if len(areas) > 0 else 0,
        "areas": areas,
    }

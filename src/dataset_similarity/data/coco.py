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
        split: Dataset split identifier. Any of ``"train2017"``, ``"val2017"``,
            ``"trainARC"``, ``"valARC"``, ``"testARC"``, or ``"store"``. Defaults to
            ``"trainARC"``.
        size: If a float in ``(0, 1)``, the fraction of samples to retain. If a
            positive integer, the exact number of samples to retain. If ``None``, no
            subsampling is performed and the full dataset is used. Defaults to ``None``.
        random_seed: Seed for the random number generator, for reproducibility. If
            ``None``, the result is non-deterministic. Defaults to ``None``.
        embedding: Name of the embedding model whose pre-computed features to load. If
            ``None``, raw images are returned by ``__getitem__``. Defaults to ``None``.
        return_paths: If ``True``, ``__getitem__`` returns ``(tensor, path)`` instead
            of ``(tensor, label)``, where ``path`` is a :class:`pathlib.Path`. Defaults
            to ``False``.
        positive_class: COCO category names whose images are labelled ``1`` (positive).
            If ``None``, no binary labelling is applied. Defaults to ``None``.
        positive_superclass: COCO supercategory names whose member categories are
            treated as positive. Merged with ``positive_class``. Defaults to ``None``.
        drop_subclasses: COCO category names removed from the resolved positive class
            list. Images that would only be positive via one of these dropped
            subclasses (i.e. contain no other retained positive-class category) are
            excluded entirely from the dataset. Requires ``positive_superclass`` to be
            set. Defaults to ``None``.
        negative_class: COCO category names whose images are labelled ``0`` (negative).
            Requires ``positive_class`` or ``positive_superclass`` to be set. Defaults
            to ``None``.
        negative_superclass: COCO supercategory names whose member categories are
            treated as negative. Merged with ``negative_class``. Defaults to ``None``.
        min_objects_per_image: Minimum number of annotated objects an image must
            contain to be included. Defaults to ``None`` (no lower bound).
        max_objects_per_image: Maximum number of annotated objects an image may
            contain to be included. Defaults to ``None`` (no upper bound).
        min_bbox_area_fraction: Minimum fraction of image area that the smallest
            bounding box in the image must cover. Defaults to ``None`` (no lower bound).
        max_bbox_area_fraction: Maximum fraction of image area that the largest
            bounding box in the image may cover. Defaults to ``None`` (no upper bound).
        positive_fraction: Target fraction of positive samples after subsampling (e.g.
            ``0.5`` for a balanced dataset). Requires ``positive_class`` or
            ``positive_superclass``. Defaults to ``None`` (no rebalancing).
        filter_class: If ``"positive"``, object/bbox filters are applied only to
            positive-labelled images; if ``"negative"``, only to negative-labelled
            images; if ``None``, filters apply to all images. Defaults to ``None``.
        multi_label: If ``True``, ``__getitem__`` returns a multi label vector over
            ``positive_class`` instead of a binary scalar. Requires ``positive_class``
            or ``positive_superclass``. Defaults to ``False``.
    """

    def __init__(
        self,
        split: str = "trainARC",
        size: float | int | None = None,
        random_seed: int | None = None,
        embedding: str | None = None,
        return_paths: bool = False,
        positive_class: list[str] | None = None,
        positive_superclass: list[str] | None = None,
        drop_subclasses: list[str] | None = None,
        negative_class: list[str] | None = None,
        negative_superclass: list[str] | None = None,
        min_objects_per_image: int | None = None,
        max_objects_per_image: int | None = None,
        min_bbox_area_fraction: float | None = None,
        max_bbox_area_fraction: float | None = None,
        positive_fraction: float | None = None,
        filter_class: Literal["positive", "negative"] | None = None,
        multi_label: bool = False,
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
        if drop_subclasses is not None and multi_label:
            msg = "`drop_subclasses` is not supported for multi-label tasks."
            raise ValueError(msg)

        if drop_subclasses is not None and positive_superclass is None:
            msg = "`drop_subclasses` requires `positive_superclass` to be specified."
            raise ValueError(msg)

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
        if positive_fraction is not None and not (0 < positive_fraction < 1):
            msg = "`positive_fraction` must be in the range (0, 1)."
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

        # Store filtering attributes for use in _load_data
        self.positive_superclass = positive_superclass
        self.drop_subclasses = drop_subclasses
        self.drop_labels: set[int] | None = None

        # validate that drop_subclasses are part of the superclass
        if self.drop_subclasses is not None:
            assert self.positive_class is not None  # for mypy
            self.drop_labels = {
                self.class_to_label_map[cls] for cls in self.drop_subclasses
            }
            if not self.drop_labels <= set(self.positive_class):
                msg = (
                    "drop_subclasses must be a subset of the resolved positive classes."
                )
                raise ValueError(msg)
            self.positive_class = [
                cls for cls in self.positive_class if cls not in self.drop_labels
            ]

        self.negative_class = _negative_class

        # Whether we need to drop classes depends on if all classes are negative
        # TODO: bodge for now as super wants string class names
        keep_labels, keep_classes = None, None
        if self.negative_class is not None:
            keep_labels = (self.positive_class or []) + self.negative_class
            keep_classes = [
                cls
                for cls, label in self.class_to_label_map.items()
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
        if multi_label and self.positive_class is None:
            msg = "`positive_class` must be specified for multi-label tasks."
            raise ValueError(msg)

        # Super will use keep_classes to filter the dataset by class
        super().__init__(
            dataset_dir=COCO_DIR,
            keep_classes=keep_classes,
            split=split,
            size=size,
            random_seed=random_seed,
            embedding=embedding,
            return_paths=return_paths,
            multi_label=multi_label,
        )

        # Subsample data will not be called by the super if size is None
        if self.size is None and self.positive_fraction is not None:
            self.data = self.subsample_data()

    @property
    def num_labels(self) -> int:
        assert self.positive_class is not None, "This dataset has no labels defined."
        return len(self.positive_class)

    def _load_data(self) -> pd.DataFrame:
        ann_file = self.dataset_dir / "annotations" / f"instances_{self.split}.json"
        coco = COCO(str(ann_file))
        df = pd.DataFrame(
            _get_row_from_img_id(img_id, coco) for img_id in coco.getImgIds()
        )

        # Filter dataset for keep classes
        if self.keep_labels is not None:
            df = df[df.cats.apply(lambda x: any(cls in x for cls in self.keep_labels))]

        # exclude images that would only be positive via a dropped subclass
        if self.drop_labels is not None:
            # for making mypy happy
            positive_class = self.positive_class
            assert positive_class is not None
            positive = df.cats.apply(lambda x: any(cls in x for cls in positive_class))
            dropped = df.cats.apply(lambda x: any(cls in x for cls in self.drop_labels))
            df = df[~(dropped & ~positive)]

        # Create binary task if requested
        if self.positive_class is not None:
            if self.multi_label:
                df["multi_label"] = df.cats.apply(
                    lambda x: [int(cls in x) for cls in self.positive_class]
                )
            df["label"] = df.cats.apply(
                lambda x: int(any(cls in x for cls in self.positive_class))
            )
        else:
            df["label"] = -1  # dummy label for get item to work

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
            classes: List of class names to include.
            superclasses: List of supercategory class names to include.

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
        new_data = self.data.copy()

        if self.positive_fraction is not None:
            new_data = self._balance_classes()

        if self.size is not None:
            _, new_data = train_test_split(
                new_data,
                test_size=self.size,
                stratify=new_data["label"] if self.positive_class is not None else None,
                random_state=self.random_seed,
            )

        return new_data.reset_index(drop=True)

    def _balance_classes(self) -> pd.DataFrame:
        """Downsample the majority class to achieve ``positive_fraction``."""
        positive_fraction = self.positive_fraction  # this & below line for mypy reasons
        assert positive_fraction is not None
        pos_df = self.data[self.data["label"] == 1]
        neg_df = self.data[self.data["label"] == 0]
        n_pos, n_neg = len(pos_df), len(neg_df)

        # Downsample whichever class is over-represented
        target_neg = int(n_pos * (1 - positive_fraction) / positive_fraction)
        target_pos = int(n_neg * positive_fraction / (1 - positive_fraction))

        if target_neg <= n_neg:
            neg_df = neg_df.sample(n=target_neg, random_state=self.random_seed)
        elif target_pos <= n_pos:
            pos_df = pos_df.sample(n=target_pos, random_state=self.random_seed)

        # shuffle the combined set
        return (
            pd.concat([pos_df, neg_df])
            .sample(frac=1, random_state=self.random_seed)
            .reset_index(drop=True)
        )


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
        "bbox_fractions": [area / img_area for area in areas],
    }

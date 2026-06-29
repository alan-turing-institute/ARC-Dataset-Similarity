from pathlib import Path
from typing import Literal, TypedDict

import pandas as pd

from dataset_similarity.constants import DATA_DIR, IMAGENET_DIR
from dataset_similarity.data.base import ImageDataset
from dataset_similarity.utils import load_yaml_from_path


class SynsetInfo(TypedDict):
    class_number: int
    name: str


class ImageNetDataset(ImageDataset):
    """
    PyTorch dataset for `ImageNet ILSVRC <https://image-net.org/>`_.

    Args:
        keep_classes: List of class names to include in the dataset. If None, all
            classes are included. Elements must be valid class names as specified in the
            class mapping file at
            `dataset_dir.parent / "metadata/imagenet_class_mapping.yaml"`. Defaults to
            `None`.
        split: Dataset split identifier. Must be `"train"` or `"val"`. Defaults to
            `"train"`.
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
        multi_label: If `True`, `__getitem__` returns a multi label vector over
            `positive_class` instead of a binary scalar. Not implemented for
            ImageNet. Defaults to `False`.
    """

    def __init__(
        self,
        keep_classes: list[str] | None = None,
        split: Literal["train", "val"] = "train",
        size: float | int | None = None,
        random_seed: int | None = None,
        embedding: None | str = None,
        return_paths: bool = False,
        multi_label: bool = False,
    ) -> None:
        # Synset = synonym set. Needs processing before calling super().__init__()
        # E.g. "n02119789" is a synset ID with name "kit_fox" and class_number 1.
        self.synset_map: dict[str, SynsetInfo] = load_yaml_from_path(
            DATA_DIR / "metadata" / "imagenet_class_mapping.yaml"
        )

        # synset_id -> class_number
        self.synsetid_to_classnumber_map: dict[str, int] = {
            synset: info["class_number"] for synset, info in self.synset_map.items()
        }

        # class_number -> human-readable name
        self.classnumber_to_name_map: dict[int, str] = {
            info["class_number"]: info["name"] for info in self.synset_map.values()
        }

        # human-readable name -> synset_id (for accepting names in keep_classes)
        self._name_to_synsetid_map: dict[str, str] = {
            info["name"]: synset for synset, info in self.synset_map.items()
        }

        # Need to resolve keep_classes before calling super().__init__()
        if keep_classes is not None:
            resolved: list[str] = []
            for cls in keep_classes:
                if cls in self.synset_map:
                    resolved.append(cls)
                elif cls in self._name_to_synsetid_map:
                    resolved.append(self._name_to_synsetid_map[cls])
                else:
                    err_msg = (
                        f"Unknown class '{cls}' in keep_classes. Please provide a "
                        "list of valid synset IDs or human-readable names."
                    )
                    raise ValueError(err_msg)
            keep_classes = resolved

        super().__init__(
            dataset_dir=IMAGENET_DIR,
            keep_classes=keep_classes,
            split=split,
            size=size,
            random_seed=random_seed,
            embedding=embedding,
            return_paths=return_paths,
            multi_label=multi_label,
        )

    def _load_data(self) -> pd.DataFrame:
        """
        Expects the standard directory layout::

            dataset_dir/
            ├── train/
            │   ├── n01440764/
            │   │   ├── n01440764_10026.JPEG
            │   │   └── ...
            │   └── ...
            └── val/
                ├── n01440764/
                │   └── ...
                └── ...

            Where 'n01440764' is a synset ID.

            Returns:
                DataFrame with columns ["path", "label"]. Paths are absolute strings.
        """
        rows: list[dict[str, Path | int]] = []
        if self.keep_classes is None:
            classes = list(self.synset_map.keys())
        else:
            classes = self.keep_classes
        for class_name in classes:  # class_name is always a synset ID
            label = self.synsetid_to_classnumber_map[class_name]
            class_dir = self.dataset_dir / self.split / class_name
            images = sorted(
                p
                for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpeg", ".jpg", ".png"}
            )
            rows.extend({"path": image_path, "label": label} for image_path in images)
        return pd.DataFrame(rows, columns=["path", "label"])

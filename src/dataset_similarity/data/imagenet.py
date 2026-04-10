from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

import pandas as pd

from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.utils import load_yaml_from_path


class SynsetInfo(TypedDict):
    class_number: int
    name: str


class ImageNetDataset(ImageDataset):
    """
    PyTorch dataset for `ImageNet ILSVRC <https://image-net.org/>`_.

    Args:
        data_root: Path to the root ImageNet directory (contains ``train/`` and
            ``val/`` sub-directories).
        split: ``"train"`` or ``"val"``. Defaults to ``"train"``.
        target_classes: Optional list of class sub-directory names to include. If
            None, all classes present in the split directory will be included.
    """

    def __init__(
        self,
        data_root: str | Path,
        target_classes: list[str] | None = None,
        split: Literal["train", "val"] = "train",
        size: float | int | None = None,
        random_seed: int | None = None,
    ) -> None:
        # Synset = synonym set. Needs processing before calling super().__init__()
        # E.g. "n02119789" is a synset ID with name "kit_fox" and class_number 1.
        self.synset_map: dict[str, SynsetInfo] = load_yaml_from_path(
            Path(data_root).parent / "metadata" / "imagenet_class_mapping.yaml"
        )

        # synset_id -> class_number
        self.synsetid_to_classnumber_map: dict[str, int] = {
            synset: info["class_number"] for synset, info in self.synset_map.items()
        }

        # class_number -> human-readable name
        self.classnumber_to_name_map: dict[int, str] = {
            info["class_number"]: info["name"] for info in self.synset_map.values()
        }

        # human-readable name -> synset_id (for accepting names in target_classes)
        self._name_to_synsetid_map: dict[str, str] = {
            info["name"]: synset for synset, info in self.synset_map.items()
        }

        # Need to resolve target_classes before calling super().__init__()
        if target_classes is not None:
            resolved: list[str] = []
            for cls in target_classes:
                if cls in self.synset_map:
                    resolved.append(cls)
                elif cls in self._name_to_synsetid_map:
                    resolved.append(self._name_to_synsetid_map[cls])
                else:
                    err_msg = (
                        f"Unknown class '{cls}' in target_classes. Please provide a "
                        "list of valid synset IDs or human-readable names."
                    )
                    raise ValueError(err_msg)
            target_classes = resolved

        super().__init__(data_root, target_classes, split, size, random_seed)

    def _load_data(self) -> pd.DataFrame:
        """
        Expects the standard directory layout::

            data_root/
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
        rows: list[dict[str, str | int]] = []
        if self.target_classes is None:
            classes = list(self.synset_map.keys())
        else:
            classes = self.target_classes
        for class_name in classes:  # class_name is always a synset ID
            label = self.synsetid_to_classnumber_map[class_name]
            class_dir = self.root / self.split / class_name
            images = sorted(
                p
                for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpeg", ".jpg", ".png"}
            )
            rows.extend(
                {"path": str(image_path), "label": label} for image_path in images
            )
        return pd.DataFrame(rows, columns=["path", "label"])

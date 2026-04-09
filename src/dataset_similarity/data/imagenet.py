from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from pandas import DataFrame
from yaml import safe_load

from dataset_similarity.data.base import ImageDataset


def load_imagenet_class_mapping(
    yaml_path: str | Path,
) -> dict[str, dict[str, str | int]]:
    with Path(yaml_path).open() as f:
        dictionary: dict[str, dict[str, str | int]] = safe_load(f)
    return dictionary


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
        split: Literal["train", "val"] = "train",
        target_classes: list[str] | None = None,
    ) -> None:
        super().__init__(data_root, split)

        self.synset_descriptor_map = load_imagenet_class_mapping(
            self.root.parent / "metadata" / "imagenet_class_mapping.yaml"
        )

        # synset_id -> class_number
        self.descriptor_label_map: dict[str, int] = {
            synset: cast(int, info["class_number"])
            for synset, info in self.synset_descriptor_map.items()
        }
        # class_number -> human-readable name
        self.label_descriptor_map: dict[int, str] = {
            cast(int, info["class_number"]): cast(str, info["name"])
            for info in self.synset_descriptor_map.values()
        }
        # human-readable name -> synset_id (for accepting names in target_classes)
        self._name_to_synset: dict[str, str] = {
            cast(str, info["name"]): synset
            for synset, info in self.synset_descriptor_map.items()
        }

        # Validate split value
        if split not in ("train", "val"):
            err_msg = f"Unknown split '{split}'. Choose from: 'train' or 'val'"
            raise ValueError(err_msg)

        # Verify the split directory exists
        split_dir = self.root / split
        if not split_dir.is_dir():
            err_msg = (
                f"Split directory not found: {split_dir}\n"
                "Download ImageNet from https://image-net.org/ and point "
                "'data_root' at the extracted directory."
            )
            raise FileNotFoundError(err_msg)

        # Resolve target_classes: accept synset IDs (e.g. "n02119789") or
        # human-readable names (e.g. "kit_fox"), silently skipping unknowns.
        if target_classes is not None:
            resolved: list[str] = []
            for cls in target_classes:
                if cls in self.synset_descriptor_map:
                    resolved.append(cls)
                elif cls in self._name_to_synset:
                    resolved.append(self._name_to_synset[cls])
            self.classes = [cls for cls in resolved if (split_dir / cls).is_dir()]
        else:
            # Only include directories that correspond to known synsets
            self.classes = sorted(
                p.name
                for p in split_dir.iterdir()
                if p.is_dir() and p.name in self.synset_descriptor_map
            )

        if not self.classes:
            err_msg = f"No class sub-directories found in {split_dir}"
            raise FileNotFoundError(err_msg)

        # Build label index and load all (image_path, label) pairs
        self.data = self._load_samples(split_dir)

    def _load_samples(
        self,
        split_dir: Path,
    ) -> DataFrame:
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

            Returns:
                DataFrame with columns ["path", "label"]. Paths are absolute strings.
        """
        rows: list[dict[str, str | int]] = []
        for class_name in self.classes:  # class_name is always a synset ID
            label = self.descriptor_label_map[class_name]  # synset_id -> class_number
            class_dir = split_dir / class_name
            images = sorted(
                p
                for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpeg", ".jpg", ".png"}
            )
            rows.extend(
                {"path": str(image_path), "label": label} for image_path in images
            )
        return DataFrame(rows, columns=["path", "label"])

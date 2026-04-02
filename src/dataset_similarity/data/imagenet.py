from __future__ import annotations

from pathlib import Path
from typing import Literal

from dataset_similarity.data.base import ImageDataset


class ImageNetDataset(ImageDataset):
    """
    PyTorch dataset for `ImageNet ILSVRC <https://image-net.org/>`_.

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

    Args:
        data_root: Path to the root ImageNet directory (contains ``train/`` and
            ``val/`` sub-directories).
        split: ``"train"`` or ``"val"``. Defaults to ``"train"``.
        class_config: Optional path to a YAML file specifying which classes to
            include and how many samples to take from each. Classes not present
            in the split directory are silently skipped. If ``None``, all
            classes and all samples are loaded.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "val"] = "train",
        target_classes: list[str] | None = None,
    ) -> None:
        super().__init__(data_root)

        # Validate split value
        if split not in ("train", "val"):
            err_msg = f"Unknown split '{split}'. Choose from: 'train' or 'val'"
            raise ValueError(err_msg)
        self.split = split

        # Verify the split directory exists
        split_dir = self.root / split
        if not split_dir.is_dir():
            err_msg = (
                f"Split directory not found: {split_dir}\n"
                "Download ImageNet from https://image-net.org/ and point "
                "'data_root' at the extracted directory."
            )
            raise FileNotFoundError(err_msg)

        # Determine which classes to include, filtering out any missing directories
        if target_classes is not None:
            self.classes = [cls for cls in target_classes if (split_dir / cls).is_dir()]
        else:
            self.classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())

        if not self.classes:
            err_msg = f"No class sub-directories found in {split_dir}"
            raise FileNotFoundError(err_msg)

        # Build label index and load all (image_path, label) pairs
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples(split_dir)

    def _load_samples(
        self,
        split_dir: Path,
    ) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        for class_name in self.classes:
            label = self.class_to_idx[class_name]
            class_dir = split_dir / class_name
            images = sorted(
                p
                for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpeg", ".jpg", ".png"}
            )
            samples.extend((image_path, label) for image_path in images)
        return samples

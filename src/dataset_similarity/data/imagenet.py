from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class ClassConfig:
    """Per-class configuration for ImageNet loading.

    Attributes:
        class_name: The synset directory name (e.g. ``"n01440764"``).
        max_samples: Maximum number of images to load for this class.
            If ``None``, all available images are loaded.
    """

    class_name: str
    max_samples: int | None = None


def class_config_from_yaml(path: str | Path) -> list[ClassConfig]:
    """Load a list of :class:`ClassConfig` objects from a YAML file.

    The YAML file should be a mapping of class names to sample counts.
    A ``null`` value (or absent count) means load all available images.

    Example YAML::

        n01440764: 100
        n01443537: 50
        n01484850:

    Args:
        path: Path to the YAML file.

    Returns:
        List of :class:`ClassConfig` objects in file order.
    """
    with Path(path).open() as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        err_msg = f"Expected a YAML mapping, got {type(data).__name__} in {path}"
        raise ValueError(err_msg)

    configs: list[ClassConfig] = []
    for name, count in data.items():
        if count is not None and not (
            isinstance(count, int) and not isinstance(count, bool)
        ):
            err_msg = (
                f"Invalid sample count for class {name!r} in {path}: "
                f"expected int or null, got {type(count).__name__}"
            )
            raise ValueError(err_msg)
        configs.append(ClassConfig(class_name=str(name), max_samples=count))

    return configs


class ImageNetDataset(Dataset):  # type: ignore[misc]
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
        class_config: str | Path | None = None,
    ) -> None:
        # Validate split value
        if split not in ("train", "val"):
            err_msg = f"Unknown split '{split}'. Choose from: 'train' or 'val'"
            raise ValueError(err_msg)

        self.root = Path(data_root)
        self.split = split

        # Parse YAML config into ClassConfig objects if a path was provided
        resolved_config = (
            class_config_from_yaml(class_config) if class_config is not None else None
        )

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
        if resolved_config is not None:
            self.classes = [
                cfg.class_name
                for cfg in resolved_config
                if (split_dir / cfg.class_name).is_dir()
            ]
        else:
            self.classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())

        if not self.classes:
            err_msg = f"No class sub-directories found in {split_dir}"
            raise FileNotFoundError(err_msg)

        # Build label index and load all (image_path, label) pairs
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples(split_dir, resolved_config)
        self._to_tensor = transforms.ToTensor()

    def _load_samples(
        self,
        split_dir: Path,
        class_config: list[ClassConfig] | None,
    ) -> list[tuple[Path, int]]:
        # Build a lookup of per-class sample limits from the config
        max_samples_for: dict[str, int | None] = {}
        if class_config is not None:
            max_samples_for = {cfg.class_name: cfg.max_samples for cfg in class_config}

        samples: list[tuple[Path, int]] = []
        for class_name in self.classes:
            label = self.class_to_idx[class_name]
            class_dir = split_dir / class_name
            # Collect and sort image files, then apply the per-class limit
            images = sorted(
                p
                for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpeg", ".jpg", ".png"}
            )
            limit = max_samples_for.get(class_name)
            if limit is not None:
                images = images[:limit]
            samples.extend((image_path, label) for image_path in images)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        with Image.open(image_path) as image:
            image_tensor = self._to_tensor(image.convert("RGB"))
        return image_tensor, label

    @property
    def class_count(self) -> int:
        """Number of distinct classes present in this split."""
        return len(self.classes)

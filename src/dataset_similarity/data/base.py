from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import Normalize
from typing_extensions import Self


class ImageDataset(Dataset):  # type: ignore[misc]
    """Base class for image datasets."""

    _registry: ClassVar[dict[str, type["ImageDataset"]]] = {}

    def __init_subclass__(cls, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if name is not None:
            ImageDataset._registry[name] = cls

    def __init__(self, data_root: Path | str, split: str, **kwargs: Any) -> None:
        self.root = Path(data_root)
        self.samples: list[tuple[Path, int | str]] = []
        self.classes: list[str] = []
        self.split: str = split
        self.normalise = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self._denormalise = Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )

    def denormalise(self, item: torch.Tensor) -> torch.Tensor:
        """Reverse the normalisation transform to get back to [0, 1] range."""
        return self._denormalise(item).clamp(0.0, 1.0)

    def stratify_by_class(
        self,
        size: float | int,
        random_seed: int | None = None,
    ) -> list[tuple[Path, int | str]]:
        """
        resample dataset to have defined size and a fixed number of samples per class.
        """
        if not isinstance(size, float | int):
            err_msg = "size must be either a float in (0, 1) or a positive integer"  # type: ignore[unreachable]
            raise TypeError(err_msg)
        if isinstance(size, float):
            if not (0 < size < 1):
                err_msg = "If 'size' is a float, it must be in the range (0, 1)"
                raise ValueError(err_msg)
        else:
            if size <= 0:
                err_msg = "If 'size' is an int, it must be a positive integer"
                raise ValueError(err_msg)
            if size > len(self.classes):
                err_msg = (
                    "If 'size' is an int, it cannot be larger than the number of "
                    "classes in the dataset"
                )
                raise ValueError(err_msg)
            if size > len(self.samples):
                err_msg = (
                    "If 'size' is an int, it cannot be larger than the number of "
                    "samples in the dataset"
                )
                raise ValueError(err_msg)

        self._strip_single_classes_from_samples()

        images, labels = zip(*self.samples, strict=True)
        (
            _,
            images_stratified,
            _,
            labels_stratified,
        ) = train_test_split(
            images,
            labels,
            test_size=size,
            stratify=labels,
            random_state=random_seed,
        )
        self.samples = list(zip(images_stratified, labels_stratified, strict=True))
        return self.samples

    def _strip_single_classes_from_samples(self) -> None:
        """
        Strips classes in the dataset with only a single example inplace.
        """
        classes_to_strip = []
        labels = [label for _, label in self.samples]
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts, strict=True):
            if count < 2:
                print(
                    f"Warning: Found label '{label}' with only {count} sample. "
                    "Removing from dataset..."
                )
                classes_to_strip.append(label)

        self.samples = [
            (path, label)
            for path, label in self.samples
            if label not in classes_to_strip
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        image_tensor = decode_image(image_path, mode="RGB")
        return self.normalise(image_tensor.float() / 255.0).float(), int(label)

    @property
    def class_count(self) -> int:
        """Number of distinct classes present in this split."""
        return len(self.classes)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Path | str,
    ) -> Self:
        """
        Factory method to create an ImageDataset from a YAML config file.
        """

        with open(yaml_path) as f:
            config: dict[str, Any] = yaml.safe_load(f)

        name = config.get("name")

        if name is None:
            err_msg = (
                "YAML config must contain a 'name' key specifying the dataset name"
            )
            raise ValueError(err_msg)

        target_cls = ImageDataset._registry.get(name, cls)

        data_root_path: str | None = config.get("data_root")
        if data_root_path is None:
            err_msg = (
                "YAML config must contain a 'data_root' key specifying the "
                "dataset root directory"
            )
            raise ValueError(err_msg)

        kwargs: dict[Any, Any] = config.get("kwargs", {})

        return target_cls(data_root=Path(data_root_path), **kwargs)

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


class ImageDataset(Dataset):  # type: ignore[misc]
    """Base class for image datasets."""

    def __init__(self, data_root: Path | str) -> None:
        self.root = Path(data_root)
        self.samples: list[tuple[Path, int]] = []
        self.classes: list[str] = []

        self._to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        image_tensor = read_image(self.root / image_path, mode="RGB")
        return image_tensor, label

    @property
    def class_count(self) -> int:
        """Number of distinct classes present in this split."""
        return len(self.classes)

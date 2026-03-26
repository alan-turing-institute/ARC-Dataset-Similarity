import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataset(Dataset):  # type: ignore[misc]
    """Base dataset class and utilities."""

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = transforms.ToTensor()(image)
        return image_tensor, label

    @property
    def class_count(self) -> int:
        """Number of distinct classes present in this split."""
        return len(self.classes)

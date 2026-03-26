from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DOMAINNET_DOMAINS = Literal[
    "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
]


class DomainNetDataset(Dataset):  # type: ignore[misc]
    """
    PyTorch dataset for `DomainNet <http://ai.bu.edu/M3SDA/>`_.

    Args:
        data_root: Path to the root DomainNet directory.
        domain: One of ``"clipart"``, ``"infograph"``, ``"painting"``,
            ``"quickdraw"``, ``"real"``, or ``"sketch"``.
        split: ``"train"`` or ``"test"``. Defaults to ``"train"``.
    """

    DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")

    def __init__(
        self,
        data_root: str | Path,
        domain: DOMAINNET_DOMAINS,
        split: Literal["train", "test"] = "train",
    ) -> None:
        if domain not in self.DOMAINS:
            err_msg = f"Unknown domain {domain}. Choose from: {self.DOMAINS}"
            raise ValueError(err_msg)
        if split not in ("train", "test"):
            err_msg = f"Unknown split {split}. Choose from: 'train' or 'test'"
            raise ValueError(err_msg)

        self.root = Path(data_root)
        self.domain = domain
        self.split = split

        split_file = self.root / f"{domain}_{split}.txt"
        if not split_file.exists():
            err_msg = (
                f"Split file not found: {split_file}\n"
                "Download DomainNet from http://ai.bu.edu/M3SDA/ and point "
                "'root' at the extracted directory."
            )
            raise FileNotFoundError(err_msg)

        self.samples = self.read_domain_net_split(split_file)
        self.classes = sorted({label for _, label in self.samples})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        image = Image.open(self.root / image_path).convert("RGB")
        image_tensor = transforms.ToTensor()(image)
        return image_tensor, label

    @property
    def class_count(self) -> int:
        """Number of distinct classes present in this split."""
        return len(self.classes)

    @classmethod
    def read_domain_net_split(
        cls,
        split_file: Path,
    ) -> list[tuple[Path, int]]:
        """
        Read a DomainNet split file and return the list of samples and label name map.

        Args:
            split_file: Path to the split file.

        Returns:
            list containing file-label tuples
        """
        samples = []
        with split_file.open(newline="") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                if not row:
                    err_msg = f"Invalid line in split file {split_file}: {row}"
                    raise ValueError(err_msg)
                rel_path, label = row[0], row[1]
                samples.append((Path(rel_path), int(label)))

        return samples

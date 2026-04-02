from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.utils import load_domainnet_class_mapping

DOMAINNET_CLASS_MAP = load_domainnet_class_mapping()
DOMAIN_LABEL_NAME_MAP = {v: k for k, v in DOMAINNET_CLASS_MAP.items()}


class DomainNetDataset(ImageDataset):
    """
    PyTorch dataset for `DomainNet <http://ai.bu.edu/M3SDA/>`_.

    Args:
        data_root: Path to the root DomainNet directory.
        domain: One of ``"clipart"``, ``"infograph"``, ``"painting"``,
            ``"quickdraw"``, ``"real"``, or ``"sketch"``.
        split: ``"train"`` or ``"test"``. Defaults to ``"train"``.
    """

    DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
    CLASS_MAP = DOMAINNET_CLASS_MAP

    def __init__(
        self,
        data_root: str | Path,
        domains: str | list[str] | None = None,
        target_classes: list[str] | None = None,
        split: Literal["train", "test"] = "train",
    ) -> None:
        super().__init__(data_root)

        if domains is None:
            domains = list(self.DOMAINS)
        if isinstance(domains, str):
            domains = [domains]
        for domain in domains:
            if domain not in self.DOMAINS:
                err_msg = f"Unknown domain {domain}. Choose from: {self.DOMAINS}"
                raise ValueError(err_msg)

        self.domains = domains

        if split not in ("train", "test"):
            err_msg = f"Unknown split {split}. Choose from: 'train' or 'test'"
            raise ValueError(err_msg)

        self.split = split

        if target_classes is not None:
            for cls in target_classes:
                if cls not in self.CLASS_MAP:
                    err_msg = (
                        f"Unknown class {cls}. Check the class mapping at "
                        "data/metadata/domainnet_class_mapping.yaml for valid class "
                        "names."
                    )
                    raise ValueError(err_msg)
        else:
            target_classes = list(self.CLASS_MAP.keys())

        for domain in self.domains:
            split_file = self.root / f"{domain}_{split}.txt"
            if not split_file.exists():
                err_msg = (
                    f"Split file not found: {split_file}\n"
                    "Download DomainNet from http://ai.bu.edu/M3SDA/ and point "
                    "'data_root' at the extracted directory."
                )
                raise FileNotFoundError(err_msg)

            self.samples = self.samples + self.read_domain_net_split(
                split_file, target_classes
            )

        # build the list of classes present in this dataset based on the labels in
        # the split files
        for _, label_id in self.samples:
            class_name = DOMAIN_LABEL_NAME_MAP[label_id]
            if class_name not in self.classes:
                self.classes.append(class_name)

    @classmethod
    def read_domain_net_split(
        cls,
        split_file: Path,
        target_classes: list[str] | None = None,
    ) -> list[tuple[Path, int]]:
        """
        Read a DomainNet split file and return the list of (path, label) samples.

        Args:
            split_file: Path to the split file.
            target_classes: List of class names to include in the split.

        Returns:
            list[tuple[Path, int]]: List of (relative file path, integer label) tuples.
        """
        if target_classes is None:
            class_indexes = list(DOMAINNET_CLASS_MAP.values())
        else:
            class_indexes = [DOMAINNET_CLASS_MAP[cls] for cls in target_classes]
        samples: list[tuple[Path, int]] = []
        with split_file.open(newline="") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                if not row:
                    err_msg = f"Invalid line in split file {split_file}: {row}"
                    raise ValueError(err_msg)
                rel_path, label = row[0], row[1]

                if int(label) in class_indexes:
                    samples.append((Path(rel_path), int(label)))

        return samples

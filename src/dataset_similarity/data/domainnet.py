from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.utils import load_domainnet_class_mapping


class DomainNetDataset(ImageDataset, name="domainnet"):
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
        domains: str | list[str] | None = None,
        target_classes: list[str] | None = None,
        split: Literal["train", "test"] = "train",
        size: float | int | None = None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__(data_root, split)

        self.descriptor_label_map = load_domainnet_class_mapping(
            self.root.parent / "metadata" / "domainnet_class_mapping.yaml"
        )
        self.label_descriptor_map = {
            label: name for name, label in self.descriptor_label_map.items()
        }

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
                if cls not in self.descriptor_label_map:
                    err_msg = (
                        f"Unknown class {cls}. Check the class mapping at "
                        "data/metadata/domainnet_class_mapping.yaml for valid class "
                        "names."
                    )
                    raise ValueError(err_msg)
        else:
            target_classes = list(self.descriptor_label_map.keys())

        for domain_index, domain in enumerate(self.domains):
            split_file = self.root / f"{domain}_{self.split}.txt"
            if not split_file.exists():
                err_msg = (
                    f"Split file not found: {split_file}\n"
                    "Download DomainNet from http://ai.bu.edu/M3SDA/ and point "
                    "'data_root' at the extracted directory."
                )
                raise FileNotFoundError(err_msg)

            self.samples = self.samples + self._load_samples(
                split_file, domain_index, target_classes
            )

        if size is not None:
            self.samples = self.stratify_by_class(size, random_seed)

        self._strip_domain_from_labels()

        # build the list of classes present in this dataset based on the labels in
        # the split files
        for _, label_id in self.samples:
            class_name = self.label_descriptor_map[int(label_id)]
            if class_name not in self.classes:
                self.classes.append(class_name)

    def _load_samples(
        self,
        split_file: Path,
        domain_index: int,
        target_classes: list[str] | None = None,
    ) -> list[tuple[Path, int | str]]:
        """
        Read a DomainNet split file and return the list of (path, label) samples.

        Expects the standard directory layout::

            data_root/
            ├── [domain]/
            │   ├── [class_name]/
            │   │   ├── images.jpg
            │   │   └── ...
            │   └── ...
            ├── [domain]_[split].txt
            │── ...

        Args:
            split_file: Path to the split file.
            target_classes: List of class names to include in the split.

        Returns:
            list[tuple[Path, int | str]]: List of (absolute file path, integer label)
            tuples.
        """
        if target_classes is None:
            class_indexes = list(self.descriptor_label_map.values())
        else:
            class_indexes = [self.descriptor_label_map[cls] for cls in target_classes]
        samples: list[tuple[Path, int | str]] = []
        with split_file.open(newline="") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                if not row:
                    err_msg = f"Invalid line in split file {split_file}: {row}"
                    raise ValueError(err_msg)
                rel_path, label = row[0], row[1]

                if int(label) in class_indexes:
                    samples.append(
                        (
                            (self.root / Path(rel_path)),
                            f"{label}:{domain_index}",
                        ),
                    )

        return samples

    def _strip_domain_from_labels(self) -> None:
        """
        Convert a label with domain index back to the original class label.

        Domain indices are appended to the class label for stratified sampling across
        domains. This method modifies the dataset samples in-place to strip the domain
        index from the labels.

        sample = (path, label) where label is of the form "label_id:domain_index". This
        method modifies the sample to be (path, original_label_id) where
        original_label_id is the integer label

        Args:
            None

        Returns:
            None
        """

        stripped_samples: list[tuple[Path, int | str]] = []

        for sample in self.samples:
            path: Path = sample[0]
            label = str(sample[1])
            label_id_str = label.split(":", maxsplit=1)[0]
            label_id = int(label_id_str)
            original_label_id = label_id
            stripped_samples.append((path, original_label_id))

        self.samples = stripped_samples

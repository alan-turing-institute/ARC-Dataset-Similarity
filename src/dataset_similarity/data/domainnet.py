from __future__ import annotations

from pathlib import Path
from typing import Literal

from pandas import DataFrame, concat, read_csv
from yaml import safe_load

from dataset_similarity.data.base import ImageDataset


def load_domainnet_class_mapping(
    yaml_path: str | Path,
) -> dict[str, int]:
    with Path(yaml_path).open() as f:
        dictionary: dict[str, int] = safe_load(f)
    return dictionary


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
                        f"{self.root.parent}"
                        "/metadata/domainnet_class_mapping.yaml for valid class "
                        "names."
                    )
                    raise ValueError(err_msg)
        else:
            target_classes = list(self.descriptor_label_map.keys())

        dfs = []
        for domain_index, domain in enumerate(self.domains):
            split_file = self.root / f"{domain}_{self.split}.txt"
            if not split_file.exists():
                err_msg = (
                    f"Split file not found: {split_file}\n"
                    "Download DomainNet from http://ai.bu.edu/M3SDA/ and point "
                    "'data_root' at the extracted directory."
                )
                raise FileNotFoundError(err_msg)

            dfs.append(self._load_data(split_file, domain_index, target_classes))

        self.data = concat(dfs, ignore_index=True)

        if size is not None:
            self.data = self.stratify_by_class(size, random_seed)

        self._strip_domain_from_labels()

        # build the list of classes present in this dataset based on the labels in
        # the split files
        self._classes = [
            self.label_descriptor_map[int(label_id)]
            for label_id in self.data["label"].unique()
        ]

    def _load_data(
        self,
        split_file: Path,
        domain_index: int,
        target_classes: list[str] | None = None,
    ) -> DataFrame:
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
            df: DataFrame with columns ["path", "label", "domain"] containing the
            samples in the split.
        """
        df = read_csv(split_file, delimiter=" ", header=None, names=["path", "label"])
        if target_classes is not None:
            target_label_ids = {
                self.descriptor_label_map[cls] for cls in target_classes
            }
            df = df[df["label"].isin(target_label_ids)]

        df["path"] = df["path"].apply(lambda rel_pth: str(self.root / rel_pth))
        # prepend domain index to label to ensure unique labels across domains
        df["label"] = df["label"].apply(lambda label: f"{label}:{domain_index}")
        return df

    def _strip_domain_from_labels(self) -> None:
        """
        Strip domain index from labels in-place, leaving only the class number.
        """
        # add domain as a separate column and remove it from the label
        self.data[["label", "domain"]] = self.data["label"].str.split(":", expand=True)

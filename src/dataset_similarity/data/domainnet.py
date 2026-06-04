from typing import Literal

import pandas as pd

from dataset_similarity.constants import DATA_DIR, DOMAINNET_DIR
from dataset_similarity.data.base import ImageDataset
from dataset_similarity.utils import load_yaml_from_path


class DomainNetDataset(ImageDataset):
    """
    PyTorch dataset for `DomainNet <http://ai.bu.edu/M3SDA/>`_.

    Args:
        domains: If not `None`, either a single domain name as a string, or a list of
            domain names. Each domain name must be one of `"clipart"`, `"infograph"`,
            `"painting"`, `"quickdraw"`, `"real"`, or `"sketch"`. If `None`, all domains
            are included.
        target_classes: List of class names to include in the dataset. If None, all
            classes are included. Elements must be valid class names as specified in the
            class mapping file at
            `dataset_dir.parent / "metadata/domainnet_class_mapping.yaml"`. Defaults to
            `None`.
        split: Dataset split identifier. Must be `"train"` or `"test"`. Defaults to
            `"train"`.
        size: If a float in `(0, 1)`, the fraction of samples to retain.
            If a positive integer, the exact number of samples to retain. If
            `None`, no subsampling is performed and the full dataset is used. Defaults
            to `None`.
        random_seed: Seed for the random number generator, for reproducibility. If
            `None`, the result is non-deterministic. Defaults to `None`.
        embedding: If not `None`, the name of the embedding model to use for this
            dataset. If `None`, raw images are returned by `__getitem__`. Defaults to
            `None`.
        return_paths: If `True`, `__getitem__` returns a tuple of (tensor, path)
            instead of (tensor, label). The path is returned as a `Path` object.
            Defaults to `False`.
    """

    DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")

    def __init__(
        self,
        domains: str | list[str] | None = None,
        target_classes: list[str] | None = None,
        split: Literal["train", "test"] = "train",
        size: float | int | None = None,
        random_seed: int | None = None,
        embedding: None | str = None,
        return_paths: bool = False,
    ) -> None:
        # Domain needs to be processed before calling super().__init__()
        if domains is None:
            self.domains = list(self.DOMAINS)
        elif isinstance(domains, str):
            self.domains = [domains]
        else:
            if len(domains) != len(set(domains)):
                err_msg = (
                    "Duplicate domains found in input. Please provide a list of unique "
                    "domain names."
                )
                raise ValueError(err_msg)
            for domain in domains:
                if domain not in self.DOMAINS:
                    err_msg = f"Unknown domain {domain}. Choose from: {self.DOMAINS}"
                    raise ValueError(err_msg)
            self.domains = domains

        # Target classes also need to be processed before calling super().__init__()
        self.class_to_label_map: dict[str, int] = load_yaml_from_path(
            DATA_DIR / "metadata" / "domainnet_class_mapping.yaml"
        )
        self.classnumber_to_name_map: dict[int, str] = {
            label: name for name, label in self.class_to_label_map.items()
        }
        if target_classes is not None:
            for cls in target_classes:
                if cls not in self.class_to_label_map:
                    err_msg = (
                        f"Unknown class {cls}. Check the class mapping at "
                        f"DATA_DIR"
                        "/metadata/domainnet_class_mapping.yaml for valid class "
                        "names."
                    )
                    raise ValueError(err_msg)
            self.target_label_ids = {
                self.class_to_label_map[cls] for cls in target_classes
            }

        super().__init__(
            dataset_dir=DOMAINNET_DIR,
            target_classes=target_classes,
            split=split,
            size=size,
            random_seed=random_seed,
            embedding=embedding,
            return_paths=return_paths,
        )

    def _load_data(self) -> pd.DataFrame:
        return pd.concat(
            [self._load_domain(domain) for domain in self.domains], ignore_index=True
        )

    def _load_domain(
        self,
        domain: str,
    ) -> pd.DataFrame:
        """
        Read a DomainNet split file and return a dataframe with columns ["path",
        "label", "domain"].

        Expects the standard directory layout::

            dataset_dir/
            ├── [domain]/
            │   ├── [class_name]/
            │   │   ├── images.jpg
            │   │   └── ...
            │   └── ...
            ├── [domain]_[split].txt
            │── ...

        Args:
            domain: DomainNet domain name (e.g. `"clipart"`). Must be one of the
                values in `self.DOMAINS`.

        Returns:
            df: DataFrame with columns ["path", "label", "domain"] containing the
            samples in the split.
        """
        df = pd.read_csv(
            self.dataset_dir / f"{domain}_{self.split}.txt",
            delimiter=" ",
            header=None,
            names=["path", "label"],
        )
        if self.target_classes is not None:
            df = df[df["label"].isin(self.target_label_ids)]
        df["path"] = df["path"].apply(lambda rel_pth: self.dataset_dir / rel_pth)
        df["domain"] = self.DOMAINS.index(domain)
        return df

    def subsample_data(self) -> pd.DataFrame:
        """
        Resample the dataset to a fixed size, stratified by class label and domain.

        Returns:
            The new DataFrame after resampling.
        """
        if len(self.domains) == 1:
            return super().subsample_data()
        self.data.label = self.data.apply(
            lambda row: str(row.label) + "-" + str(row.domain), axis=1
        )
        new_data = super().subsample_data()
        new_data.label = new_data.label.apply(lambda label: label.split("-")[0])
        return new_data

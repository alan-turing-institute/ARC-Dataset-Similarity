"""Mixes two datasets by a blend fraction, for domain-shift interpolation sweeps."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset_similarity.constants import CONFIG_DIR
from dataset_similarity.data import DATASET_MAP
from dataset_similarity.data.base import ImageDataset
from dataset_similarity.utils import load_yaml_from_path


class DatasetMix(Dataset):  # type: ignore[misc]
    """Combines a prefix of ``dataset1`` and a prefix of ``dataset2`` per ``alpha``."""

    def __init__(
        self,
        dataset1: ImageDataset,
        dataset2: ImageDataset,
        alpha: float,
    ):
        """dataset1/dataset2: sources to mix. alpha: fraction drawn from dataset1."""
        self.datasets: tuple[ImageDataset, ImageDataset] = (dataset1, dataset2)
        self.alpha = alpha
        self.data = self._mix_datasets(dataset1, dataset2, self.alpha)
        self.multi_label = dataset1.multi_label or dataset2.multi_label

    def __len__(self) -> int:
        """Number of samples in the mixed dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, np.int64 | str | Path]:
        """Look up the source dataset and index for ``idx``, then fetch the sample."""
        row = self.data.iloc[idx]
        dataset_idx = row["dataset"]
        sample_idx = row["idx"]
        return self.datasets[dataset_idx][sample_idx]

    @classmethod
    def _mix_datasets(
        cls,
        dataset1: ImageDataset,
        dataset2: ImageDataset,
        alpha: float,
    ) -> pd.DataFrame:
        """Build (dataset, idx) index table from a leading prefix of each dataset."""
        ld1 = len(dataset1)
        ld2 = len(dataset2)
        # Not random draws: mixtures at different alpha therefore share
        # samples (nested family), not just label proportions.
        n1 = int(np.ceil(alpha * (ld1 - 1)))
        n2 = int(np.ceil((1 - alpha) * (ld2 - 1)))
        return pd.DataFrame(
            {
                "dataset": [0] * n1 + [1] * (n2),
                "idx": list(range(n1)) + list(range(n2)),
            }
        )

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
    ) -> "DatasetMix":
        """
        Build a DatasetMix from a config dict naming dataset1, dataset2, and alpha.
        """
        dataset1 = _get_dataset(cfg["dataset1"])
        dataset2 = _get_dataset(cfg["dataset2"])
        return cls(dataset1, dataset2, cfg["alpha"])

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
    ) -> "DatasetMix":
        """Build a DatasetMix from a YAML config file."""
        return cls.from_dict(load_yaml_from_path(yaml_path)["kwargs"])


def _get_dataset(dataset_cfg_name: str) -> ImageDataset:
    """Load a dataset by its config name under configs/data/."""
    dataset_cfg_path = CONFIG_DIR / "data" / f"{dataset_cfg_name}.yaml"
    dataset_cfg = load_yaml_from_path(dataset_cfg_path)
    dataset_fn = DATASET_MAP[dataset_cfg["name"]]
    return dataset_fn(**dataset_cfg["kwargs"])

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
    def __init__(
        self,
        dataset1: ImageDataset,
        dataset2: ImageDataset,
        alpha: float,
    ):
        self.datasets: tuple[ImageDataset, ImageDataset] = (dataset1, dataset2)
        self.alpha = alpha
        self.data = self._mix_datasets(dataset1, dataset2, self.alpha)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, np.int64 | str | Path]:
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
        ld1 = len(dataset1)
        ld2 = len(dataset2)
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
        dataset1 = _get_dataset(cfg["dataset1"])
        dataset2 = _get_dataset(cfg["dataset2"])
        return cls(dataset1, dataset2, cfg["alpha"])

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
    ) -> "DatasetMix":
        return cls.from_dict(load_yaml_from_path(yaml_path)["kwargs"])


def _get_dataset(dataset_cfg_name: str) -> ImageDataset:
    dataset_cfg_path = CONFIG_DIR / "data" / f"{dataset_cfg_name}.yaml"
    dataset_cfg = load_yaml_from_path(dataset_cfg_path)
    dataset_fn = DATASET_MAP[dataset_cfg["name"]]
    return dataset_fn(**dataset_cfg["kwargs"])

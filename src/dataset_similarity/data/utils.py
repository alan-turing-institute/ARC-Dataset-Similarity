from typing import Any

from dataset_similarity.data import DATASET_MAP
from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.mix import DatasetMix


def load_dataset_from_config(
    cfg: dict[str, Any],
) -> ImageDataset | DatasetMix:
    name = cfg["name"]
    if name == "DatasetMix":
        return DatasetMix.from_dict(cfg["kwargs"])
    dataset_cls = DATASET_MAP[cfg["name"]]
    return dataset_cls.from_dict(cfg["kwargs"])

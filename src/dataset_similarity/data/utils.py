from collections.abc import Callable
from pathlib import Path
from typing import Any

from yaml import safe_load

from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.domainnet import DomainNetDataset
from dataset_similarity.data.imagenet import ImageNetDataset

model_mapping: dict[str, Callable[..., ImageDataset]] = {
    "domainnet": DomainNetDataset,
    "imagenet": ImageNetDataset,
}


def from_yaml(
    yaml_path: str | Path,
) -> ImageDataset:
    with Path(yaml_path).open() as f:
        yaml_dict: dict[str, object] = safe_load(f)

    name = yaml_dict.get("name")
    if name is None:
        err_msg = "YAML config must contain a 'name' key specifying the dataset name"
        raise ValueError(err_msg)

    class_args: dict[str, Any] = yaml_dict.get("args") or {}  # type: ignore[assignment]

    if "data_root" not in class_args:
        err_msg = (
            "YAML config must contain a 'data_root' key specifying the dataset"
            " root directory within the 'args' section"
        )
        raise ValueError(err_msg)

    model_mapping_cls = model_mapping[str(name)]
    return model_mapping_cls(**class_args)

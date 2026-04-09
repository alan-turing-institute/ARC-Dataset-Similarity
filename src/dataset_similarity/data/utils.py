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
    """
    Instantiate a dataset from a YAML config file.

    The YAML file must contain a ``name`` key matching an entry in
    ``model_mapping`` and an ``args`` section whose keys are forwarded as
    keyword arguments to the dataset constructor.  ``args`` must include at
    least a ``data_root`` key.

    Example config::

        name: domainnet
        args:
          data_root: data/DomainNet
          domains: [real, sketch]
          split: train

    Args:
        yaml_path: Path to the YAML config file.

    Raises:
        ValueError: If the config is missing a ``name`` or ``data_root`` key.

    Returns:
        An instantiated ``ImageDataset`` subclass corresponding to ``name``.
    """
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

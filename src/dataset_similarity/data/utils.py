from typing import Any

import torch
from torchvision import transforms

from dataset_similarity.data import DATASET_MAP
from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.mix import DatasetMix


def load_dataset_from_config(cfg: dict[str, Any]) -> ImageDataset | DatasetMix:
    name = cfg["name"]
    if name == "DatasetMix":
        return DatasetMix.from_dict(cfg["kwargs"])
    dataset_cls = DATASET_MAP[cfg["name"]]
    return dataset_cls.from_dict(cfg["kwargs"])


_imagenet_preprocess = transforms.Compose(
    [
        transforms.Lambda(lambda x: x[:3]),  # drop alpha channel if RGBA
        transforms.Resize((224, 224)),  # ResNet50 expects 224x224
        transforms.ConvertImageDtype(torch.float32),  # uint8 --> float32 [0, 1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def image_ds_collate_fn(
    batch: list[tuple[torch.Tensor, int]],
) -> dict[str, torch.Tensor]:
    pixel_values = torch.stack([_imagenet_preprocess(item[0]) for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

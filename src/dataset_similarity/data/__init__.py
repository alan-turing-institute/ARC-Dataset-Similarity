from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.coco import COCODataset
from dataset_similarity.data.coco_task_pool import (
    COCOTaskDataset,
    COCOTaskPool,
)
from dataset_similarity.data.domainnet import DomainNetDataset
from dataset_similarity.data.imagenet import ImageNetDataset

__all__ = [
    "COCODataset",
    "COCOTaskDataset",
    "COCOTaskPool",
    "DomainNetDataset",
    "ImageNetDataset",
]

DATASET_MAP: dict[str, type[ImageDataset]] = {
    "COCO": COCODataset,
    "DomainNet": DomainNetDataset,
    "ImageNet": ImageNetDataset,
}

from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.domainnet import DomainNetDataset
from dataset_similarity.data.imagenet import ImageNetDataset

__all__ = [
    "DomainNetDataset",
    "ImageNetDataset",
]

DATASET_MAP: dict[str, type[ImageDataset]] = {
    "DomainNet": DomainNetDataset,
    "ImageNet": ImageNetDataset,
}

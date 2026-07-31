from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.coco import COCODataset
from dataset_similarity.data.domainnet import DomainNetDataset

__all__ = [
    "COCODataset",
    "DomainNetDataset",
]

DATASET_MAP: dict[str, type[ImageDataset]] = {
    "COCO": COCODataset,
    "DomainNet": DomainNetDataset,
}

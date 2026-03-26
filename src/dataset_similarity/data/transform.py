from torch.utils.data import Dataset
from torchvision import transforms

horizontal_flip = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=1.0),
    ]
)


def apply_transform(dataset: Dataset, transform: transforms.Compose) -> Dataset:
    """
    Apply a simple transformation to the dataset.
    """
    return TransformedDataset(dataset, transform)


class TransformedDataset(Dataset):
    """
    A dataset wrapper that applies a transformation to the data.
    """

    def __init__(self, dataset: Dataset, transform: transforms.Compose) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item, label = self.dataset[idx]
        return self.transform(item), label

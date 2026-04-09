from pathlib import Path
from typing import Any

import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.io import decode_image


class ImageDataset(Dataset):  # type: ignore[misc]
    """Base class for image datasets."""

    def __init__(self, data_root: Path | str, split: str, **kwargs: Any) -> None:
        self.root = Path(data_root)
        self.samples: DataFrame = DataFrame(columns=["path", "label"])
        self.classes: list[str] = []
        self.split: str = split
        self.data: DataFrame = DataFrame()

    def stratify_by_class(
        self,
        size: float | int,
        random_seed: int | None = None,
    ) -> DataFrame:
        """
        resample dataset to have defined size and a fixed number of samples per class.
        """
        _, new_data = train_test_split(
            self.data,
            test_size=size,
            stratify=self.data["label"],
            random_state=random_seed,
        )
        self.data = new_data.reset_index(drop=True)
        return self.data

    def _strip_single_classes_from_samples(self) -> None:
        """
        Strips classes in the dataset with only a single example inplace.
        """
        label_counts = self.data["label"].value_counts()
        for label, count in label_counts.items():
            if count == 1:
                print(f"Warning: Found label '{label}' with only a single example")
        labels_to_drop = (label_counts.loc[label_counts == 1]).index
        self.data = self.data.loc[~self.data["label"].isin(labels_to_drop)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        items = self.data.iloc[idx]
        image_path = items["path"]
        label = items["label"]
        image_tensor = decode_image(image_path, mode="RGB")
        return image_tensor.float() / 255.0, int(label)

    @property
    def num_classes(self) -> int:
        """Number of distinct classes present in this split."""
        return len(self.classes)

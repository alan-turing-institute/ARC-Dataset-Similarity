from abc import abstractmethod
from pathlib import Path
from typing import Any

import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.io import decode_image


class ImageDataset(Dataset):  # type: ignore[misc]
    """
    Abstract base class for image datasets.

    Subclasses must implement ``_load_data`` to populate ``self.data`` with a
    DataFrame of ``(path, label)`` rows and set ``self._classes``.

    Attributes:
        root: Absolute path to the dataset root directory.
        split: Dataset split identifier (e.g. ``"train"`` or ``"test"``).
        data: DataFrame with at least ``"path"`` (absolute string) and
            ``"label"`` (integer) columns, populated by the subclass.
    """

    def __init__(self, data_root: Path | str, split: str, **kwargs: Any) -> None:
        self.root = Path(data_root)
        self._classes: list[str] = []
        self.split: str = split
        self.data: DataFrame = DataFrame()

    @property
    def classes(self) -> list[str]:
        return self._classes

    def stratify_by_class(
        self,
        size: float | int,
        random_seed: int | None = None,
    ) -> DataFrame:
        """
        Resample the dataset to a fixed size, stratified by class label.

        Replaces ``self.data`` in-place with a random stratified subset.

        Args:
            size: If a float in ``(0, 1)``, the fraction of samples to retain.
                If a positive integer, the exact number of samples to retain.
            random_seed: Seed for the random number generator, for
                reproducibility. If ``None``, the result is non-deterministic.

        Returns:
            The new ``self.data`` DataFrame after resampling.
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
        Remove classes that have only a single sample from ``self.data`` in-place.

        Classes with fewer than two samples cannot be used in stratified splits.
        A warning is printed for each removed class. Modifies ``self.data``
        in-place.
        """
        label_counts = self.data["label"].value_counts()
        for label, count in label_counts.items():
            if count == 1:
                print(f"Warning: Found label '{label}' with only a single example")
        labels_to_drop = (label_counts.loc[label_counts == 1]).index
        self.data = self.data.loc[~self.data["label"].isin(labels_to_drop)]

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Get an image and its label by index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple containing the image tensor (C x H x W) and its label.
        """
        items = self.data.iloc[idx]
        image_path = items["path"]
        label = items["label"]
        image_tensor = decode_image(image_path, mode="RGB")
        return image_tensor.float() / 255.0, int(label)

    @property
    def num_classes(self) -> int:
        """
        Number of distinct classes present in this split.

        Returns:
            The number of distinct classes in the dataset split.
        """
        return len(self.classes)

    @abstractmethod
    def _load_data(self) -> DataFrame:
        """
        Load the dataset split into a DataFrame with columns ["path", "label"].

        Paths should be absolute strings. Labels should be integers.
        The set of labels should correspond to the classes in self.classes.

        This method is called by the constructor of the base class, and should not
        be called directly.

        Subclasses must implement this method to load their specific dataset format.
        """
        err_msg = "Subclasses must implement _load_data() to load the dataset samples"
        raise NotImplementedError(err_msg)

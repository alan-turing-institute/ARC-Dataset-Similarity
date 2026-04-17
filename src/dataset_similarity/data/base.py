from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from pandas import DataFrame
from safetensors.torch import load_file
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.io import read_image

from dataset_similarity.data.utils import get_embedding_path, load_yaml_from_path


class ImageDataset(ABC, Dataset):  # type: ignore[misc]
    """
    Abstract base class for image datasets.

    Subclasses must implement ``_load_data`` to populate ``self.data`` with a
    DataFrame of ``(path, label)`` rows and set ``self._classes``.

    Args:
        dataset_dir: Absolute path to the directory containing the dataset images.
        target_classes: List of class names to include in the dataset. If None, all
            classes are included.
        split: Dataset split identifier (e.g. ``"train"`` or ``"test"``).
        size: If a float in ``(0, 1)``, the fraction of samples to retain.
            If a positive integer, the exact number of samples to retain. If
            ``None``, no subsampling is performed and the full dataset is used.
        random_seed: Seed for the random number generator, for reproducibility. If
            ``None``, the result is non-deterministic.
        embedding: If not ``None``, the name of the embedding model to use for this
            dataset. If ``None``, raw images are returned by ``__getitem__``.
        embedding_dir: The absolute path to the directory where the embeddings are
            stored. This is used to compute the path to the embedding for each image.
            Must be provided if `embedding` is not None.
        return_paths: If ``True``, ``__getitem__`` returns a tuple of (tensor, path)
            instead of (tensor, label). The path is returned as a ``Path`` object.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        target_classes: list[str] | None,
        split: str,
        size: float | int | None,
        random_seed: int | None,
        embedding: None | str,
        embedding_dir: None | Path | str,
        return_paths: bool,
    ) -> None:
        super().__init__()
        if not isinstance(dataset_dir, Path):
            dataset_dir = Path(dataset_dir)
        self.dataset_dir = dataset_dir
        self.split = split
        self.target_classes = target_classes
        self.data = self._load_data()
        self.size = size
        self.random_seed = random_seed
        if self.size is not None:
            self.data = self.subsample_data()
        if embedding is not None:
            if embedding_dir is None:
                msg = "`embedding_dir` must be provided if `embedding` is not None"
                raise ValueError(msg)
            if not isinstance(embedding_dir, Path):
                embedding_dir = Path(embedding_dir)
            embedding_path = embedding_dir / embedding
        else:
            embedding_path = None
        self.embedding_path = embedding_path
        self.return_paths = return_paths

    def subsample_data(self) -> DataFrame:
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
        self._strip_single_classes_from_samples()

        _, new_data = train_test_split(
            self.data,
            test_size=self.size,
            stratify=self.data["label"],
            random_state=self.random_seed,
        )
        return new_data.reset_index(drop=True)

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int | str | Path]:
        """
        Get a sample from the dataset. If ``self.embedding`` is ``None``, the first
        element of the returned tuple is an image tensor of shape (C x H x W).

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple containing the output tensor (C x H x W) or (D,) depending on the
            value of ``self.embedding`` and either its label or file path depending on
            the value of ``return_paths``.
        """
        sample = self.data.iloc[idx]
        image_path = sample["path"]
        if self.embedding_path is None:
            tensor = read_image(image_path, mode="RGB")
        else:
            image_embedding_path = get_embedding_path(
                image_path=image_path,
                embedding_dir=self.embedding_path,
                data_root=self.dataset_dir.parent,
            )
            tensor = load_file(image_embedding_path, framework="pt", device="cpu")[
                "embedding"
            ]
        if self.return_paths:
            return tensor, image_path
        return tensor, sample["label"]

    @property
    def num_classes(self) -> int:
        """
        Number of distinct classes present in this split.

        Returns:
            The number of distinct classes in the dataset split.
        """
        return len(self.data["label"].unique())

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

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ImageDataset":
        """
        Instantiate a dataset from a config dictionary. The config dict must contain a
        `data_root` key specifying the dataset root  directory, and any other keys
        required by the dataset constructor.

        Example config dict:

            {
                "data_root": "data/DomainNet",
                "domains": ["real", "sketch"],
                "split": "train"
            }

        Args:
            config_dict: Dictionary containing the dataset configuration. Must contain a
                `data_root` key, and any other keys required by the dataset constructor.

        Returns:
            An instantiated ``ImageDataset`` subclass.
        """
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ImageDataset":
        """
        Instantiate a dataset from a YAML config file.

        The YAML file must contain an `args` key itself containing several keys which
        correspond to the keyword arguments of the dataset constructor. This is passed
        to `from_dict` after loading the YAML file. The YAML file may also contain a
        `name` key, but this is ignored.

        Example config::

            name: domainnet
            args:
              data_root: data/DomainNet
              domains: [real, sketch]
              split: train

        Args:
            yaml_path: Path to the YAML config file.

        Returns:
            An instantiated ``ImageDataset`` subclass.
        """
        return cls.from_dict(load_yaml_from_path(yaml_path)["kwargs"])

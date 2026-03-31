from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class BaseExtractor(ABC):
    """Abstract base class for image embedding extractors.

    Subclasses must implement :meth:`preprocess` and :meth:`encode`.
    The :meth:`extract_dataset` method handles DataLoader creation,
    batching, and device transfer.
    """

    def __init__(self, model_name: str, device: str | torch.device = "cpu") -> None:
        self.model_name = model_name
        self.device = torch.device(device)

    @abstractmethod
    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        """Preprocess a list of PIL images into a batch pixel-values tensor."""
        ...

    @abstractmethod
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a preprocessed pixel-values tensor into embeddings.

        Args:
            pixel_values: Tensor of shape ``(B, C, H, W)`` already on
                :attr:`device`.

        Returns:
            Embedding tensor of shape ``(B, D)``.
        """
        ...

    def extract_dataset(
        self,
        dataset: Dataset[Any],
        batch_size: int = 64,
        num_workers: int = 4,
        get_image: Callable[[Any], Image.Image] = lambda item: item[0],
    ) -> npt.NDArray[np.float32]:
        """Extract embeddings for every image in *dataset*.

        Args:
            dataset: PyTorch Dataset whose items are passed to *get_image*.
            batch_size: Images processed per forward pass.
            num_workers: Worker processes for the DataLoader.
            get_image: Callable that extracts a PIL ``Image`` from one
                dataset item.  Defaults to ``lambda item: item[0]``,
                which works for ``(image, label)`` tuples.

        Returns:
            Float32 array of shape ``(N, embedding_dim)``.
        """

        def collate(batch: list[Any]) -> list[Image.Image]:
            return [get_image(item) for item in batch]

        loader: DataLoader[Any] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
        )

        all_embeddings: list[torch.Tensor] = []
        with torch.inference_mode():
            for images in tqdm(loader, desc=f"Extracting [{self.model_name}]"):
                pixel_values = self.preprocess(images).to(self.device)
                embeddings = self.encode(pixel_values)
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)

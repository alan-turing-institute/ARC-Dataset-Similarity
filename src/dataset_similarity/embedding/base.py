from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _embedding_save_path(
    output_dir: Path,
    src_path: str | os.PathLike[str],
    dataset_root: Path | None,
    model_name: str,
) -> Path:
    p = Path(src_path)
    if dataset_root is not None:
        rel = p.relative_to(dataset_root)
    elif p.is_absolute():
        rel = Path(p.name)
    else:
        rel = p
    return output_dir / model_name / rel.with_suffix(".safetensors")


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
        output_dir: Path | str | None = None,
        get_path: Callable[[Any], str | os.PathLike[str]] | None = None,
        dataset_root: Path | str | None = None,
    ) -> None:
        """Extract embeddings for every image in *dataset*.

        Args:
            dataset: PyTorch Dataset whose items are passed to *get_image*.
            batch_size: Images processed per forward pass.
            num_workers: Worker processes for the DataLoader.
            get_image: Callable that extracts a PIL ``Image`` from one
                dataset item.  Defaults to ``lambda item: item[0]``,
                which works for ``(image, label)`` tuples.
            output_dir: Root directory in which to save per-image embedding
                files.  When *None* no files are written.
            get_path: Callable that returns the source file path for a
                dataset item.  Required when *output_dir* is set.  The path
                may be absolute or relative; see *dataset_root*.
            dataset_root: Root of the original dataset used to compute
                relative paths when *get_path* returns absolute paths.
                When *None* and the path is absolute, only the filename is
                preserved.
        """
        if output_dir is not None and get_path is None:
            msg = "get_path must be provided when output_dir is set"
            raise ValueError(msg)

        out_root = Path(output_dir) if output_dir is not None else None
        ds_root = Path(dataset_root) if dataset_root is not None else None

        def collate(batch: list[Any]) -> tuple[list[Image.Image], list[Any]]:
            images = [get_image(item) for item in batch]
            paths = [get_path(item) for item in batch] if get_path else batch
            return images, paths

        loader: DataLoader[Any] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
        )
        model_name = self.model_name.split("/")[-1]
        with torch.inference_mode():
            for images, paths in tqdm(loader, desc=f"Extracting [{model_name}]"):
                pixel_values = self.preprocess(images).to(self.device)
                embeddings = self.encode(pixel_values).cpu()

                if out_root is not None:
                    for emb, src_path in zip(embeddings, paths, strict=True):
                        dst = _embedding_save_path(
                            out_root, src_path, ds_root, model_name
                        )
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        save_file({"embedding": emb.unsqueeze(0)}, dst)

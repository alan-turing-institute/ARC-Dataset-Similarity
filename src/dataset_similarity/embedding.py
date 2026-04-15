from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

MODEL_NAMES: dict[str, str] = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-base-patch16-224",
    "dinov3": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


def _collate(batch: list[Any]) -> tuple[list[Image.Image], list[Any]]:
    """Return images and labels as plain lists, bypassing the default collate.

    PyTorch's default collate_fn attempts to stack items into tensors, which
    fails when images have different sizes or when labels are arbitrary types
    (e.g. file paths). Returning lists defers any stacking to the model's
    preprocessor. See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    """
    images = [item[0] for item in batch]
    label = [item[1] for item in batch]
    return images, label


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


class Extractor:
    """Image embedding extractor supporting CLIP, SigLIP, and DINOv3 models.

    Args:
        model_name: Model family to use. One of ``"clip"``, ``"siglip"``, or
            ``"dinov3"``.
        hf_model_id: HuggingFace model ID override. Defaults to the standard
            model for the chosen family (see :data:`MODEL_NAMES`).
        device: Torch device string or object.

    Example::

        extractor = Extractor("clip", device="cuda")
        embeddings = extractor.extract_dataset(dataset)
    """

    def __init__(
        self,
        model_name: str,
        hf_model_id: str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if model_name not in MODEL_NAMES:
            msg = f"Unknown model '{model_name}'. Available: {sorted(MODEL_NAMES)}"
            raise ValueError(msg)
        self.model_name = model_name
        self.device = torch.device(device)
        _hf_model_id = (
            hf_model_id if hf_model_id is not None else MODEL_NAMES[model_name]
        )
        self._processor = AutoProcessor.from_pretrained(_hf_model_id)
        self._model = AutoModel.from_pretrained(_hf_model_id)
        self._model.to(self.device)
        self._model.eval()

    def preprocess(self, images: list[Image.Image] | torch.Tensor) -> torch.Tensor:
        """Preprocess a list of PIL images into a batch pixel-values tensor."""
        return self._processor(images=images, return_tensors="pt")["pixel_values"]

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a preprocessed pixel-values tensor into embeddings.

        Args:
            pixel_values: Tensor of shape ``(B, C, H, W)`` already on
                :attr:`device`.

        Returns:
            Embedding tensor of shape ``(B, D)``.
        """
        encoder = (
            self._model if self.model_name == "dinov3" else self._model.vision_model
        )
        return encoder(pixel_values=pixel_values).pooler_output

    def extract_dataset(
        self,
        dataset: Dataset[Any],
        batch_size: int = 64,
        num_workers: int = 4,
        output_dir: Path | str | None = None,
        dataset_root: Path | str | None = None,
    ) -> None:
        """Extract embeddings for every image in *dataset*.

        Args:
            dataset: PyTorch Dataset whose items are passed to *get_image*.
            batch_size: Images processed per forward pass.
            num_workers: Worker processes for the DataLoader.
            output_dir: Root directory in which to save per-image embedding
                files.  When *None* no files are written.
            dataset_root: Root of the original dataset used to compute
                relative paths for saved embeddings.  Only relevant if
                *output_dir* is not *None*.
        """
        out_root = Path(output_dir) if output_dir is not None else None
        ds_root = Path(dataset_root) if dataset_root is not None else None

        loader: DataLoader[Any] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate,
        )
        with torch.inference_mode():
            for images, paths in tqdm(loader, desc=f"Extracting [{self.model_name}]"):
                pixel_values = self.preprocess(images).to(self.device)
                embeddings = self.encode(pixel_values).cpu()

                if out_root is not None:
                    for emb, src_path in zip(embeddings, paths, strict=True):
                        if not isinstance(src_path, str | os.PathLike):
                            msg = (
                                "Expected path to be str or os.PathLike, "
                                "got {type(src_path)}"
                            )
                            raise ValueError(msg)
                        dst = _embedding_save_path(
                            out_root, src_path, ds_root, self.model_name
                        )
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        save_file({"embedding": emb.unsqueeze(0)}, dst)

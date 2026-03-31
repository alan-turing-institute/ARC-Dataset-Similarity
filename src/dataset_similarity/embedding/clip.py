from __future__ import annotations

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    CLIPModel,
)

from dataset_similarity.embedding.base import BaseExtractor


class CLIPExtractor(BaseExtractor):
    """Image embedding extractor backed by a CLIP model.

    Args:
        model_name: HuggingFace model ID, e.g.
            ``"openai/clip-vit-base-patch32"``.
        device: Torch device string or object.

    Example::

        extractor = CLIPExtractor(device="cuda")
        embeddings = extractor.extract_dataset(dataset)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(model_name, device)
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model: CLIPModel = CLIPModel.from_pretrained(model_name)
        self._model.to(self.device)
        self._model.eval()

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        return self._processor(images=images, return_tensors="pt")["pixel_values"]

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._model.vision_model(pixel_values=pixel_values).pooler_output

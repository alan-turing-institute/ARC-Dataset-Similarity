from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from dataset_similarity.embedding.base import BaseExtractor


class SigLIPExtractor(BaseExtractor):
    """Image embedding extractor backed by a SigLIP model.

    Args:
        model_name: HuggingFace model ID, e.g.
            ``"google/siglip-base-patch16-224"``.
        device: Torch device string or object.

    Example::

        extractor = SigLIPExtractor(device="cuda")
        embeddings = extractor.extract_dataset(dataset)
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(model_name, device)
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.to(self.device)
        self._model.eval()

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        return self._processor(images=images, return_tensors="pt")["pixel_values"]

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self._model.vision_model(pixel_values=pixel_values)
        return out.pooler_output

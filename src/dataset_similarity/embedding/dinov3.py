from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from dataset_similarity.embedding.base import BaseExtractor


class DINOv3Extractor(BaseExtractor):
    """Image embedding extractor backed by a DINOv3 model.

    Embeddings are taken from ``pooler_output``, which corresponds to the
    CLS token representation after the pooling layer.

    Args:
        model_name: HuggingFace model ID, e.g.
            ``"facebook/dinov3-vits16-pretrain-lvd1689m"``.
        device: Torch device string or object.

    Example::

        extractor = DINOv3Extractor(device="cuda")
        embeddings = extractor.extract_dataset(dataset)
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(model_name, device)
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.to(self.device)
        self._model.eval()

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        return self._processor(images=images, return_tensors="pt")["pixel_values"]

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._model(pixel_values=pixel_values).pooler_output

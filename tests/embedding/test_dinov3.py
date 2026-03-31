from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from dataset_similarity.embedding.dinov3 import DINOv3Extractor

_BATCH = 3
_EMBED_DIM = 1024
_IMG_SIZE = 224
_DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def _processor_mock() -> MagicMock:
    mock = MagicMock()
    mock.return_value = {"pixel_values": torch.zeros(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)}
    return mock


def _dinov3_model_mock() -> MagicMock:
    """DINOv3 model called directly (not via .vision_model)."""
    output = MagicMock()
    output.pooler_output = torch.zeros(_BATCH, _EMBED_DIM)
    mock = MagicMock()
    mock.return_value = output
    return mock


def test_dinov3_default_model_name() -> None:
    with (
        patch(
            "dataset_similarity.embedding.dinov3.AutoImageProcessor.from_pretrained",
            return_value=MagicMock(),
        ),
        patch(
            "dataset_similarity.embedding.dinov3.AutoModel.from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        extractor = DINOv3Extractor()
    assert extractor.model_name == _DEFAULT_MODEL


def test_dinov3_preprocess_calls_processor_and_returns_pixel_values() -> None:
    images = [Image.new("RGB", (32, 32)) for _ in range(_BATCH)]
    mock_proc = _processor_mock()
    with (
        patch(
            "dataset_similarity.embedding.dinov3.AutoImageProcessor.from_pretrained",
            return_value=mock_proc,
        ),
        patch(
            "dataset_similarity.embedding.dinov3.AutoModel.from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        extractor = DINOv3Extractor()
    result = extractor.preprocess(images)
    mock_proc.assert_called_once_with(images=images, return_tensors="pt")
    assert result.shape == (_BATCH, 3, _IMG_SIZE, _IMG_SIZE)


def test_dinov3_encode_calls_model_directly_not_vision_model() -> None:
    """DINOv3 calls self._model(...) directly, unlike CLIP/SigLIP's .vision_model."""
    mock_model = _dinov3_model_mock()
    with (
        patch(
            "dataset_similarity.embedding.dinov3.AutoImageProcessor.from_pretrained",
            return_value=MagicMock(),
        ),
        patch(
            "dataset_similarity.embedding.dinov3.AutoModel.from_pretrained",
            return_value=mock_model,
        ),
    ):
        extractor = DINOv3Extractor()
    pixel_values = torch.zeros(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)
    result = extractor.encode(pixel_values)
    mock_model.assert_called_once_with(pixel_values=pixel_values)
    assert result.shape == (_BATCH, _EMBED_DIM)

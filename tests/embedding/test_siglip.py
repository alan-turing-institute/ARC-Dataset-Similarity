from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from dataset_similarity.embedding.siglip import SigLIPExtractor

_BATCH = 3
_EMBED_DIM = 768
_IMG_SIZE = 224
_DEFAULT_MODEL = "google/siglip-base-patch16-224"


def _processor_mock() -> MagicMock:
    mock = MagicMock()
    mock.return_value = {"pixel_values": torch.zeros(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)}
    return mock


def _siglip_model_mock() -> MagicMock:
    output = MagicMock()
    output.pooler_output = torch.zeros(_BATCH, _EMBED_DIM)
    mock = MagicMock()
    mock.vision_model.return_value = output
    return mock


def test_siglip_default_model_name() -> None:
    with (
        patch(
            "dataset_similarity.embedding.siglip.AutoProcessor.from_pretrained",
            return_value=MagicMock(),
        ),
        patch(
            "dataset_similarity.embedding.siglip.AutoModel.from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        extractor = SigLIPExtractor()
    assert extractor.model_name == _DEFAULT_MODEL


def test_siglip_preprocess_calls_processor_and_returns_pixel_values() -> None:
    images = [Image.new("RGB", (32, 32)) for _ in range(_BATCH)]
    mock_proc = _processor_mock()
    with (
        patch(
            "dataset_similarity.embedding.siglip.AutoProcessor.from_pretrained",
            return_value=mock_proc,
        ),
        patch(
            "dataset_similarity.embedding.siglip.AutoModel.from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        extractor = SigLIPExtractor()
    result = extractor.preprocess(images)
    mock_proc.assert_called_once_with(images=images, return_tensors="pt")
    assert result.shape == (_BATCH, 3, _IMG_SIZE, _IMG_SIZE)


def test_siglip_encode_uses_vision_model_pooler_output() -> None:
    mock_model = _siglip_model_mock()
    with (
        patch(
            "dataset_similarity.embedding.siglip.AutoProcessor.from_pretrained",
            return_value=MagicMock(),
        ),
        patch(
            "dataset_similarity.embedding.siglip.AutoModel.from_pretrained",
            return_value=mock_model,
        ),
    ):
        extractor = SigLIPExtractor()
    pixel_values = torch.zeros(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)
    result = extractor.encode(pixel_values)
    mock_model.vision_model.assert_called_once_with(pixel_values=pixel_values)
    assert result.shape == (_BATCH, _EMBED_DIM)

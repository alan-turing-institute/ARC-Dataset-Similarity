from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dataset_similarity.embedding import EXTRACTORS, get_extractor
from dataset_similarity.embedding.clip import CLIPExtractor
from dataset_similarity.embedding.dinov3 import DINOv3Extractor
from dataset_similarity.embedding.siglip import SigLIPExtractor


def test_extractors_contains_expected_keys() -> None:
    assert set(EXTRACTORS) == {"clip", "siglip", "dinov3"}


def test_extractors_maps_to_correct_classes() -> None:
    assert EXTRACTORS["clip"] is CLIPExtractor
    assert EXTRACTORS["siglip"] is SigLIPExtractor
    assert EXTRACTORS["dinov3"] is DINOv3Extractor


def test_get_extractor_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError, match="Unknown extractor"):
        get_extractor("unknown")


def test_get_extractor_clip_returns_clip_extractor() -> None:
    with (
        patch(
            "dataset_similarity.embedding.clip.AutoProcessor.from_pretrained",
            return_value=MagicMock(),
        ),
        patch(
            "dataset_similarity.embedding.clip.CLIPModel.from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        result = get_extractor("clip")
    assert isinstance(result, CLIPExtractor)


def test_get_extractor_siglip_returns_siglip_extractor() -> None:
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
        result = get_extractor("siglip")
    assert isinstance(result, SigLIPExtractor)


def test_get_extractor_dinov3_returns_dinov3_extractor() -> None:
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
        result = get_extractor("dinov3")
    assert isinstance(result, DINOv3Extractor)

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import Dataset

from dataset_similarity.embedding import MODEL_NAMES, Extractor

_EMBED_DIM = 16
_BATCH = 3
_IMG_SIZE = 224


class _DummyExtractor(Extractor):
    """Concrete extractor returning deterministic tensors - no model loading."""

    def __init__(
        self, data_root: Path, embedding_dir: Path, model_name: str = "clip"
    ) -> None:
        # Bypass parent __init__ to avoid actual model loading
        self.model_name = model_name
        self.device = torch.device("cpu")
        self._processor = None
        self._model = None
        self.data_root = data_root
        self.output_dir = embedding_dir / self.model_name

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        return torch.zeros(len(images), 3, 32, 32)

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.ones(pixel_values.shape[0], _EMBED_DIM)


class _PathDataset(Dataset[tuple[Image.Image, Path]]):
    """Dataset whose items are ``(image, Path)`` tuples."""

    def __init__(self, images: list[Image.Image], paths: list[Path]) -> None:
        self._data = list(zip(images, paths, strict=True))
        self.return_paths = True

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[Image.Image, Path]:
        return self._data[idx]


# ---------------------------------------------------------------------------
# Extractor constructor
# ---------------------------------------------------------------------------


def test_model_names_contains_expected_keys() -> None:
    assert set(MODEL_NAMES) == {"clip", "siglip", "dinov3"}


def test_unknown_model_raises_value_error() -> None:
    with (
        pytest.raises(ValueError, match="Unknown model"),
        patch(
            "dataset_similarity.embedding.AutoProcessor.from_pretrained",
            return_value=MagicMock(),
        ),
        patch(
            "dataset_similarity.embedding.AutoModel.from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        Extractor("unknown")


@pytest.mark.parametrize("model_name", ["clip", "siglip", "dinov3"])
def test_constructor_sets_model_name(model_name: str) -> None:
    with (
        patch(
            "dataset_similarity.embedding.AutoProcessor.from_pretrained",
            return_value=MagicMock(),
        ),
        patch(
            "dataset_similarity.embedding.AutoModel.from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        extractor = Extractor(model_name)
    assert extractor.model_name == model_name


def test_constructor_uses_default_hf_model_id() -> None:
    with (
        patch(
            "dataset_similarity.embedding.AutoProcessor.from_pretrained",
            return_value=MagicMock(),
        ) as proc_patch,
        patch(
            "dataset_similarity.embedding.AutoModel.from_pretrained",
            return_value=MagicMock(),
        ) as model_patch,
    ):
        Extractor("clip")
    proc_patch.assert_called_once_with(MODEL_NAMES["clip"])
    model_patch.assert_called_once_with(MODEL_NAMES["clip"])


def test_constructor_hf_model_id_override() -> None:
    custom_id = "my-org/my-clip-model"
    with (
        patch(
            "dataset_similarity.embedding.AutoProcessor.from_pretrained",
            return_value=MagicMock(),
        ) as proc_patch,
        patch(
            "dataset_similarity.embedding.AutoModel.from_pretrained",
            return_value=MagicMock(),
        ) as model_patch,
    ):
        Extractor("clip", hf_model_id=custom_id)
    proc_patch.assert_called_once_with(custom_id)
    model_patch.assert_called_once_with(custom_id)


# ---------------------------------------------------------------------------
# preprocess / encode
# ---------------------------------------------------------------------------


def _make_extractor(model_name: str) -> tuple[Extractor, MagicMock, MagicMock]:
    mock_proc = MagicMock()
    mock_model = MagicMock()
    with (
        patch(
            "dataset_similarity.embedding.AutoProcessor.from_pretrained",
            return_value=mock_proc,
        ),
        patch(
            "dataset_similarity.embedding.AutoModel.from_pretrained",
            return_value=mock_model,
        ),
    ):
        extractor = Extractor(model_name)
    return extractor, mock_proc, mock_model


def test_preprocess_calls_processor_and_returns_pixel_values() -> None:
    images = [Image.new("RGB", (32, 32)) for _ in range(_BATCH)]
    pixel_values = torch.zeros(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)
    extractor, mock_proc, _ = _make_extractor("clip")
    mock_proc.return_value = {"pixel_values": pixel_values}
    result = extractor.preprocess(images)
    mock_proc.assert_called_once_with(images=images, return_tensors="pt")
    assert result.shape == (_BATCH, 3, _IMG_SIZE, _IMG_SIZE)


def test_encode_clip_uses_vision_model() -> None:
    extractor, _, mock_model = _make_extractor("clip")
    output = MagicMock()
    output.pooler_output = torch.zeros(_BATCH, 512)
    mock_model.vision_model.return_value = output
    pixel_values = torch.zeros(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)
    result = extractor.encode(pixel_values)
    mock_model.vision_model.assert_called_once_with(pixel_values=pixel_values)
    assert result.shape == (_BATCH, 512)


def test_encode_siglip_uses_vision_model() -> None:
    extractor, _, mock_model = _make_extractor("siglip")
    output = MagicMock()
    output.pooler_output = torch.zeros(_BATCH, 768)
    mock_model.vision_model.return_value = output
    pixel_values = torch.zeros(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)
    result = extractor.encode(pixel_values)
    mock_model.vision_model.assert_called_once_with(pixel_values=pixel_values)
    assert result.shape == (_BATCH, 768)


def test_encode_dinov3_calls_model_directly() -> None:
    extractor, _, mock_model = _make_extractor("dinov3")
    output = MagicMock()
    output.pooler_output = torch.zeros(_BATCH, 1024)
    mock_model.return_value = output
    pixel_values = torch.zeros(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)
    result = extractor.encode(pixel_values)
    mock_model.assert_called_once_with(pixel_values=pixel_values)
    assert result.shape == (_BATCH, 1024)


# ---------------------------------------------------------------------------
# extract_dataset - file saving
# ---------------------------------------------------------------------------


def test_extract_dataset_saves_correct_number_of_files(
    rgb_images: list[Image.Image], tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    embedding_dir = tmp_path / "embeddings"
    paths = [dataset_dir / f"img_{i:04d}.png" for i in range(len(rgb_images))]
    dataset = _PathDataset(rgb_images, paths)
    extractor = _DummyExtractor(data_root=tmp_path, embedding_dir=embedding_dir)
    extractor.extract_dataset(
        dataset,
        batch_size=2,
        num_workers=0,
    )
    saved = list(embedding_dir.rglob("*.safetensors"))
    assert len(saved) == len(rgb_images)


def test_extract_dataset_saved_tensor_shape(
    rgb_images: list[Image.Image], tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    embedding_dir = tmp_path / "embeddings"
    paths = [dataset_dir / f"img_{i:04d}.png" for i in range(len(rgb_images))]
    dataset = _PathDataset(rgb_images, paths)
    extractor = _DummyExtractor(data_root=tmp_path, embedding_dir=embedding_dir)
    extractor.extract_dataset(
        dataset,
        batch_size=2,
        num_workers=0,
    )
    data = load_file(embedding_dir / "clip" / "dataset" / "img_0000.safetensors")
    assert data["embedding"].shape == (_EMBED_DIM,)


def test_extract_dataset_mirrors_path_structure(
    rgb_images: list[Image.Image], tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    embedding_dir = tmp_path / "embeddings"
    paths = [dataset_dir / f"classA/img_{i:04d}.jpg" for i in range(len(rgb_images))]
    dataset = _PathDataset(rgb_images, paths)
    extractor = _DummyExtractor(data_root=tmp_path, embedding_dir=embedding_dir)
    extractor.extract_dataset(
        dataset,
        batch_size=2,
        num_workers=0,
    )
    for i in range(len(rgb_images)):
        expected = embedding_dir / f"clip/dataset/classA/img_{i:04d}.safetensors"
        assert expected.exists(), f"Missing: {expected}"

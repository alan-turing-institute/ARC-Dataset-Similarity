"""Tests for dataset_similarity.data.transform."""

import torch
from torch import Tensor
from torch.utils.data import Dataset

from dataset_similarity.data.transform import (
    TransformedDataset,
    apply_transform,
    centre_crop,
    deterministic_colour_jitter,
    gaussian_blur,
    grayscale,
    grayscale_and_blur,
    horizontal_flip,
    rotation_180,
)


def make_image(h: int = 256, w: int = 256) -> Tensor:
    """Return a random RGB tensor (C, H, W) in [0, 1]."""
    return torch.rand(3, h, w)


class _TinyDataset(Dataset):  # type: ignore[type-arg]
    """Minimal in-memory dataset of (image, label) pairs."""

    def __init__(self, n: int = 4, h: int = 256, w: int = 256) -> None:
        self.images = [make_image(h, w) for _ in range(n)]
        self.labels = list(range(n))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        return self.images[idx], self.labels[idx]


class TestTransforms:
    def test_horizontal_flip_reverses_image(self) -> None:
        img = make_image()
        flipped = horizontal_flip(img)
        assert torch.equal(flipped, img.flip(-1))

    def test_rotation_180(self) -> None:
        img = make_image()
        rotated = rotation_180(img)
        assert rotated.shape == img.shape
        assert not torch.equal(rotated, img)

    def test_centre_crop_output_size(self) -> None:
        img = make_image(256, 256)
        cropped = centre_crop(img)
        assert cropped.shape == (3, 224, 224)

    def test_grayscale_keeps_three_channels(self) -> None:
        img = make_image()
        grey = grayscale(img)
        assert grey.shape == img.shape
        # All channels should be identical after grayscale
        assert torch.allclose(grey[0], grey[1])
        assert torch.allclose(grey[1], grey[2])

    def test_gaussian_blur_output_shape(self) -> None:
        img = make_image()
        blurred = gaussian_blur(img)
        assert blurred.shape == img.shape

    def test_colour_jitter_output_shape(self) -> None:
        img = make_image()
        jittered = deterministic_colour_jitter(img)
        assert jittered.shape == img.shape

    def test_grayscale_and_blur_output_shape(self) -> None:
        img = make_image()
        result = grayscale_and_blur(img)
        assert result.shape == img.shape


class TestTransformedDataset:
    def test_len(self) -> None:
        base = _TinyDataset(n=4)
        transformed = TransformedDataset(base, horizontal_flip)
        assert len(transformed) == len(base)

    def test_applies_transform(self) -> None:
        base = _TinyDataset(n=2)
        transformed = TransformedDataset(base, horizontal_flip)
        original_img, _ = base[0]
        transformed_img, _ = transformed[0]
        assert torch.equal(transformed_img, original_img.flip(-1))

    def test_preserves_label(self) -> None:
        base = _TinyDataset(n=4)
        transformed = TransformedDataset(base, rotation_180)
        for i in range(len(base)):
            _, orig_label = base[i]
            _, trans_label = transformed[i]
            assert orig_label == trans_label

    def test_apply_transform_returns_transformed_dataset(self) -> None:
        base = _TinyDataset(n=2)
        result = apply_transform(base, horizontal_flip)
        assert isinstance(result, TransformedDataset)
        assert len(result) == len(base)

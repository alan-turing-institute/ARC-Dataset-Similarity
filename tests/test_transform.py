"""Tests for dataset_similarity.data.transform."""

from dataset_similarity.data.transform import (
    TransformedDataset,
    apply_transform,
    centre_crop,
    colour_jitter,
    gaussian_blur,
    grayscale,
    grayscale_and_blur,
    horizontal_flip,
    rotation_180,
)


def test_transform_imports() -> None:
    assert horizontal_flip is not None
    assert rotation_180 is not None
    assert centre_crop is not None
    assert grayscale is not None
    assert gaussian_blur is not None
    assert colour_jitter is not None
    assert grayscale_and_blur is not None
    assert TransformedDataset is not None
    assert apply_transform is not None

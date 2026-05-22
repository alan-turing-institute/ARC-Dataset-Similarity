"""Tests for dataset_similarity.metrics.utils."""

import pytest

from dataset_similarity.metrics.utils import extract_dataset_tensors


def test_extract_dataset_tensors_shape(embedding_tensor_dataset):
    result = extract_dataset_tensors(embedding_tensor_dataset)
    assert result.shape[0] == len(embedding_tensor_dataset)


def test_extract_dataset_tensors_raises_for_return_paths(
    tensor_image_dataset_with_paths,
):
    with pytest.raises(ValueError, match="return_paths"):
        extract_dataset_tensors(tensor_image_dataset_with_paths)


def test_extract_dataset_tensors_return_labels(embedding_tensor_dataset):
    features, labels = extract_dataset_tensors(
        embedding_tensor_dataset, return_labels=True
    )
    assert features.ndim == 2
    assert labels.ndim == 1
    assert features.shape[0] == labels.shape[0]

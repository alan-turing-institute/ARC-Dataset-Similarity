"""Tests for dataset_similarity.metrics.utils."""

import pytest

from dataset_similarity.metrics.utils import prepare_tensor_dataset


class TestPrepareTensorDataset:
    def test_returns_2d_tensor(self, tensor_image_dataset):
        result = prepare_tensor_dataset(tensor_image_dataset)
        assert result.ndim == 2

    def test_correct_shape(self, tensor_image_dataset):
        result = prepare_tensor_dataset(tensor_image_dataset)
        assert result.shape[0] == len(tensor_image_dataset)

    def test_raises_for_return_paths(self, tensor_image_dataset_with_paths):
        with pytest.raises(ValueError, match="return_paths"):
            prepare_tensor_dataset(tensor_image_dataset_with_paths)

    def test_return_labels_gives_tuple(self, tensor_image_dataset):
        features, labels = prepare_tensor_dataset(
            tensor_image_dataset, return_labels=True
        )
        assert features.ndim == 2
        assert labels.ndim == 1
        assert features.shape[0] == labels.shape[0]

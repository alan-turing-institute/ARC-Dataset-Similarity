"""Tests for dataset_similarity.metrics.otdd."""

import pytest
import torch
from torch.utils.data import TensorDataset

from dataset_similarity.metrics.otdd import (
    _prepare_otdd_tensor_dataset,
    _prepare_tensor_dataset_targets,
    otdd,
)


class TestPrepareTensorDatasetTargets:
    def test_non_contiguous_labels_remapped(self):
        labels = torch.tensor([10, 20, 10, 30, 20])
        result = _prepare_tensor_dataset_targets(labels)
        assert result.tolist() == [0, 1, 0, 2, 1]

    def test_already_contiguous_unchanged(self):
        labels = torch.tensor([0, 1, 2, 0, 1])
        result = _prepare_tensor_dataset_targets(labels)
        assert result.tolist() == [0, 1, 2, 0, 1]

    def test_single_class_maps_to_zero(self):
        labels = torch.tensor([5, 5, 5])
        result = _prepare_tensor_dataset_targets(labels)
        assert result.tolist() == [0, 0, 0]


class TestPrepareOTDDTensorDataset:
    def test_returns_tensor_dataset(self, tensor_image_dataset):
        tds = _prepare_otdd_tensor_dataset(tensor_image_dataset)
        assert isinstance(tds, TensorDataset)

    def test_correct_length(self, tensor_image_dataset):
        tds = _prepare_otdd_tensor_dataset(tensor_image_dataset)
        assert len(tds) == len(tensor_image_dataset)

    def test_has_classes_attribute(self, tensor_image_dataset):
        tds = _prepare_otdd_tensor_dataset(tensor_image_dataset)
        assert hasattr(tds, "classes")

    def test_labels_are_contiguous(self, tensor_image_dataset):
        tds = _prepare_otdd_tensor_dataset(tensor_image_dataset)
        labels = tds.tensors[1]
        assert set(labels.tolist()) == set(range(len(torch.unique(labels))))

    def test_raises_for_return_paths(self, tensor_image_dataset_with_paths):
        with pytest.raises(ValueError, match="return_paths"):
            _prepare_otdd_tensor_dataset(tensor_image_dataset_with_paths)


# The installed otdd uses torch.symeig (removed in PyTorch 2.x). Passing
# diagonal_cov=True skips the full covariance square root and avoids the error.
_OTDD_KWARGS = {"diagonal_cov": True}


class TestOTDD:
    def test_returns_scalar_tensor(self, tensor_image_dataset):
        result = otdd(tensor_image_dataset, tensor_image_dataset, **_OTDD_KWARGS)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0

    def test_return_coupling_type(self, tensor_image_dataset):
        dist, coupling = otdd(
            tensor_image_dataset,
            tensor_image_dataset,
            return_coupling=True,
            **_OTDD_KWARGS,
        )
        assert isinstance(dist, torch.Tensor)
        assert isinstance(coupling, torch.Tensor)

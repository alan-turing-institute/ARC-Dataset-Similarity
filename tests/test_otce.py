"""Tests for dataset_similarity.metrics.otce."""

import pytest
import torch

from dataset_similarity.metrics.otce import (
    conditional_entropy,
    otce_distance,
)

torch.manual_seed(0)
_SRC = torch.randn(20, 8)
_TGT = torch.randn(20, 8) + 1.0
_SRC_LABELS = torch.tensor([i % 3 for i in range(20)], dtype=torch.long)
_TGT_LABELS = torch.tensor([i % 3 for i in range(20)], dtype=torch.long)

_EXPECTED_KEYS = {"otce", "wasserstein", "conditional_entropy", "coupling"}


class TestConditionalEntropy:
    def test_non_negative(self):
        coupling = torch.full((4, 4), 1.0 / 16)
        src_labels = torch.tensor([0, 0, 1, 1])
        tgt_labels = torch.tensor([0, 0, 1, 1])
        conditional_entropy_value = conditional_entropy(
            coupling, src_labels, tgt_labels
        ).item()
        # we know what the value should be
        assert pytest.approx(conditional_entropy_value) == 0.693147

    def test_perfect_block_diagonal_is_zero(self):
        # Coupling fully within matching classes → H(Y_t | Y_s) = 0
        src_labels = torch.tensor([0, 0, 1, 1])
        tgt_labels = torch.tensor([0, 0, 1, 1])
        coupling = torch.zeros(4, 4)
        coupling[0:2, 0:2] = 0.25
        coupling[2:4, 2:4] = 0.25
        coupling /= coupling.sum()
        assert conditional_entropy(coupling, src_labels, tgt_labels).item() == 0.0

    def test_uniform_coupling_equals_log_num_classes(self):
        # Uniform coupling over 2 balanced classes → H = log(2)
        coupling = torch.full((4, 4), 1.0 / 16)
        src_labels = torch.tensor([0, 0, 1, 1])
        tgt_labels = torch.tensor([0, 0, 1, 1])
        result = conditional_entropy(coupling, src_labels, tgt_labels)
        assert torch.isclose(result, torch.log(torch.tensor(2.0)), atol=1e-5)


class TestOTCEDistance:
    def test_returns_float(self, embedding_tensor_dataset, embedding_tensor_dataset_2):
        result = otce_distance(embedding_tensor_dataset, embedding_tensor_dataset_2)
        assert isinstance(result, float)

    def test_same_dataset_near_zero(self, embedding_tensor_dataset):
        result = otce_distance(
            embedding_tensor_dataset, embedding_tensor_dataset, method="python_ot"
        )
        assert result < 1e-5

    def test_f_otce_same_dataset_zero(self, embedding_tensor_dataset):
        # Identical datasets → perfect label alignment → H = 0
        result = otce_distance(
            embedding_tensor_dataset, embedding_tensor_dataset, use_wasserstein=False
        )
        assert result < 1e-5

    def test_jc_otce_same_dataset_near_zero(self, embedding_tensor_dataset):
        result = otce_distance(
            embedding_tensor_dataset,
            embedding_tensor_dataset,
            use_otdd=True,
            diagonal_cov=True,
        )
        assert result < 1e-3

    def test_raises_for_return_paths(
        self, embedding_tensor_dataset, tensor_image_dataset_with_paths
    ):
        with pytest.raises(ValueError, match="return_paths"):
            otce_distance(tensor_image_dataset_with_paths, embedding_tensor_dataset)

    def test_f_otce_returns_float(
        self, embedding_tensor_dataset, embedding_tensor_dataset_2
    ):
        result = otce_distance(
            embedding_tensor_dataset, embedding_tensor_dataset_2, use_wasserstein=False
        )
        assert isinstance(result, float)

    def test_f_otce_non_negative(
        self, embedding_tensor_dataset, embedding_tensor_dataset_2
    ):
        result = otce_distance(
            embedding_tensor_dataset, embedding_tensor_dataset_2, use_wasserstein=False
        )
        assert result >= 0.0

    def test_f_otce_leq_otce(
        self, embedding_tensor_dataset, embedding_tensor_dataset_2
    ):
        # F-OTCE = H; OTCE = W + H; W >= 0 so H <= W + H
        f = otce_distance(
            embedding_tensor_dataset, embedding_tensor_dataset_2, use_wasserstein=False
        )
        full = otce_distance(embedding_tensor_dataset, embedding_tensor_dataset_2)
        assert f <= full + 1e-5

    def test_jc_otce_returns_float(
        self, embedding_tensor_dataset, embedding_tensor_dataset_2
    ):
        result = otce_distance(
            embedding_tensor_dataset,
            embedding_tensor_dataset_2,
            use_otdd=True,
            diagonal_cov=True,
        )
        assert isinstance(result, float)

    def test_jc_otce_non_negative(
        self, embedding_tensor_dataset, embedding_tensor_dataset_2
    ):
        result = otce_distance(
            embedding_tensor_dataset,
            embedding_tensor_dataset_2,
            use_otdd=True,
            diagonal_cov=True,
        )
        assert result >= 0.0

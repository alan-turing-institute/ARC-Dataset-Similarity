"""Tests for dataset_similarity.metrics.ot."""

import pytest
import torch

from dataset_similarity.metrics.ot import (
    optimal_transport_distance,
    python_ot,
    sinkhorn_ot,
)

torch.manual_seed(0)
_X = torch.randn(50, 8)
_Y_NEAR = _X + 0.1
_Y_FAR = torch.randn(50, 8) + 10.0


class TestSinkhornOT:
    def test_returns_scalar(self):
        cost, _ = sinkhorn_ot(_X, _Y_NEAR)
        assert cost.ndim == 0

    def test_identical_clouds_near_zero(self):
        cost, _ = sinkhorn_ot(_X, _X.clone())
        assert cost.item() < 1e-3

    def test_distance_ordering(self):
        cost_near, _ = sinkhorn_ot(_X, _Y_NEAR)
        cost_far, _ = sinkhorn_ot(_X, _Y_FAR)
        assert cost_far.item() > cost_near.item()

    def test_custom_weights(self):
        w = torch.ones(len(_X)) / len(_X)
        cost, _ = sinkhorn_ot(_X, _Y_NEAR, weights1=w, weights2=w)
        assert cost.ndim == 0


class TestSinkhornOTTransportPlan:
    def test_returns_2d_tensor(self):
        _, plan = sinkhorn_ot(_X, _Y_NEAR, return_coupling=True)
        assert plan.ndim == 2
        assert plan.shape == (len(_X), len(_Y_NEAR))

    def test_non_negative(self):
        _, plan = sinkhorn_ot(_X, _Y_NEAR, return_coupling=True)
        assert (plan >= 0).all()

    def test_row_marginals_match_uniform_weights(self):
        N = len(_X)
        _, plan = sinkhorn_ot(_X, _Y_NEAR, return_coupling=True)
        assert torch.allclose(plan.sum(dim=1), torch.full((N,), 1.0 / N), atol=1e-3)

    def test_col_marginals_match_uniform_weights(self):
        M = len(_Y_NEAR)
        _, plan = sinkhorn_ot(_X, _Y_NEAR, return_coupling=True)
        assert torch.allclose(plan.sum(dim=0), torch.full((M,), 1.0 / M), atol=1e-3)

    def test_row_marginals_match_custom_weights(self):
        rng = torch.Generator().manual_seed(1)
        X = torch.randn(20, 4, generator=rng)
        Y = torch.randn(20, 4, generator=rng)
        w1 = torch.softmax(torch.randn(20, generator=rng), dim=0)
        w2 = torch.softmax(torch.randn(20, generator=rng), dim=0)
        # High blur → plan ≈ a⊗b, marginal constraints satisfied to machine precision
        _, plan = sinkhorn_ot(
            X, Y, weights1=w1, weights2=w2, blur=2.0, return_coupling=True
        )
        assert torch.allclose(plan.sum(dim=1), w1, atol=1e-3)
        assert torch.allclose(plan.sum(dim=0), w2, atol=1e-3)

    def test_identical_clouds_concentrates_on_diagonal(self):
        _, plan = sinkhorn_ot(_X, _X.clone(), return_coupling=True)
        assert plan.diagonal().sum() > 0.9 * plan.sum()

    def test_total_mass_is_one(self):
        _, plan = sinkhorn_ot(_X, _Y_NEAR, return_coupling=True)
        assert torch.isclose(plan.sum(), torch.tensor(1.0), atol=1e-3)


class TestPythonOT:
    def test_returns_scalar(self):
        cost, _ = python_ot(_X, _Y_NEAR)
        assert cost.ndim == 0

    def test_identical_clouds_near_zero(self):
        cost, _ = python_ot(_X, _X.clone())
        assert cost.item() < 1e-6

    def test_distance_ordering(self):
        cost_near, _ = python_ot(_X, _Y_NEAR)
        cost_far, _ = python_ot(_X, _Y_FAR)
        assert cost_far.item() > cost_near.item()

    def test_custom_weights(self):
        w = torch.ones(len(_X)) / len(_X)
        cost, _ = python_ot(_X, _Y_NEAR, weights1=w, weights2=w)
        assert cost.ndim == 0

    def test_euclidean_metric(self):
        cost, _ = python_ot(_X, _Y_NEAR, metric="euclidean")
        assert cost.ndim == 0

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric must be one of"):
            python_ot(_X, _Y_NEAR, metric="cosine")


class TestPythonOTTransportPlan:
    def test_returns_2d_tensor(self):
        _, plan = python_ot(_X, _Y_NEAR, return_coupling=True)
        assert plan.ndim == 2
        assert plan.shape == (len(_X), len(_Y_NEAR))

    def test_non_negative(self):
        _, plan = python_ot(_X, _Y_NEAR, return_coupling=True)
        assert (plan >= 0).all()

    def test_row_marginals_match_uniform_weights(self):
        N = len(_X)
        _, plan = python_ot(_X, _Y_NEAR, return_coupling=True)
        assert torch.allclose(plan.sum(dim=1), torch.full((N,), 1.0 / N), atol=1e-6)

    def test_col_marginals_match_uniform_weights(self):
        M = len(_Y_NEAR)
        _, plan = python_ot(_X, _Y_NEAR, return_coupling=True)
        assert torch.allclose(plan.sum(dim=0), torch.full((M,), 1.0 / M), atol=1e-6)

    def test_total_mass_is_one(self):
        _, plan = python_ot(_X, _Y_NEAR, return_coupling=True)
        assert torch.isclose(plan.sum(), torch.tensor(1.0), atol=1e-6)

    def test_identical_clouds_concentrates_on_diagonal(self):
        _, plan = python_ot(_X, _X.clone(), return_coupling=True)
        assert plan.diagonal().sum() > 0.9 * plan.sum()


class TestOptimalTransportDistance:
    def test_sinkhorn_method(self, tensor_image_dataset):
        cost = optimal_transport_distance(
            tensor_image_dataset, tensor_image_dataset, method="sinkhorn"
        )
        assert isinstance(cost, float)

    def test_python_ot_method(self, tensor_image_dataset):
        cost = optimal_transport_distance(
            tensor_image_dataset, tensor_image_dataset, method="python_ot"
        )
        assert isinstance(cost, float)

    def test_return_coupling(self, tensor_image_dataset):
        result = optimal_transport_distance(
            tensor_image_dataset,
            tensor_image_dataset,
            method="sinkhorn",
            return_coupling=True,
        )
        assert isinstance(result, tuple)
        cost, coupling = result
        assert isinstance(cost, float)
        N = len(tensor_image_dataset)
        assert coupling.shape == (N, N)

    def test_unsupported_method_raises(self, tensor_image_dataset):
        with pytest.raises(ValueError, match="Unsupported OT method"):
            optimal_transport_distance(
                tensor_image_dataset, tensor_image_dataset, method="bogus"
            )

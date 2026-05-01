"""Tests for dataset_similarity.metrics.ot."""

import pytest
import torch

from dataset_similarity.metrics.ot import optimal_transport_distance, sinkhorn_ot

torch.manual_seed(0)
_X = torch.randn(50, 8)
_Y_NEAR = _X + 0.1
_Y_FAR = torch.randn(50, 8) + 10.0


class TestSinkhornOT:
    def test_returns_scalar(self):
        assert sinkhorn_ot(_X, _Y_NEAR).ndim == 0

    def test_identical_clouds_near_zero(self):
        assert sinkhorn_ot(_X, _X.clone()).item() < 1e-3

    def test_distance_ordering(self):
        assert sinkhorn_ot(_X, _Y_FAR).item() > sinkhorn_ot(_X, _Y_NEAR).item()

    def test_custom_weights(self):
        w = torch.ones(len(_X)) / len(_X)
        result = sinkhorn_ot(_X, _Y_NEAR, weights1=w, weights2=w)
        assert result.ndim == 0


class TestSinkhornOTTransportPlan:
    def test_returns_2d_tensor(self):
        plan = sinkhorn_ot(_X, _Y_NEAR, return_transport_plan=True)
        assert plan.ndim == 2
        assert plan.shape == (len(_X), len(_Y_NEAR))

    def test_non_negative(self):
        plan = sinkhorn_ot(_X, _Y_NEAR, return_transport_plan=True)
        assert (plan >= 0).all()

    def test_row_marginals_match_uniform_weights(self):
        N = len(_X)
        plan = sinkhorn_ot(_X, _Y_NEAR, return_transport_plan=True)
        assert torch.allclose(plan.sum(dim=1), torch.full((N,), 1.0 / N), atol=1e-3)

    def test_col_marginals_match_uniform_weights(self):
        M = len(_Y_NEAR)
        plan = sinkhorn_ot(_X, _Y_NEAR, return_transport_plan=True)
        assert torch.allclose(plan.sum(dim=0), torch.full((M,), 1.0 / M), atol=1e-3)

    def test_row_marginals_match_custom_weights(self):
        rng = torch.Generator().manual_seed(1)
        X = torch.randn(20, 4, generator=rng)
        Y = torch.randn(20, 4, generator=rng)
        w1 = torch.softmax(torch.randn(20, generator=rng), dim=0)
        w2 = torch.softmax(torch.randn(20, generator=rng), dim=0)
        # High blur → plan ≈ a⊗b, marginal constraints satisfied to machine precision
        plan = sinkhorn_ot(
            X, Y, weights1=w1, weights2=w2, blur=2.0, return_transport_plan=True
        )
        assert torch.allclose(plan.sum(dim=1), w1, atol=1e-3)
        assert torch.allclose(plan.sum(dim=0), w2, atol=1e-3)

    def test_identical_clouds_concentrates_on_diagonal(self):
        plan = sinkhorn_ot(_X, _X.clone(), return_transport_plan=True)
        assert plan.diagonal().sum() > 0.9 * plan.sum()

    def test_total_mass_is_one(self):
        plan = sinkhorn_ot(_X, _Y_NEAR, return_transport_plan=True)
        assert torch.isclose(plan.sum(), torch.tensor(1.0), atol=1e-3)


class TestOptimalTransportDistance:
    def test_sinkhorn_method(self):
        result = optimal_transport_distance(_X, _Y_NEAR, method="sinkhorn")
        assert result.ndim == 0

    def test_unsupported_method_raises(self):
        with pytest.raises(ValueError, match="Unsupported OT method"):
            optimal_transport_distance(_X, _Y_NEAR, method="bogus")

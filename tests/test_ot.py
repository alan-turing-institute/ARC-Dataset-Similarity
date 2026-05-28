"""Tests for dataset_similarity.metrics.ot."""

import pytest
import torch

from dataset_similarity.metrics.ot import (
    ot_distance,
    python_ot,
)

# tol=1e-5 is loose enough that sinkhorn converges within 1000 iterations at reg=1.0,
# preventing POT's non-convergence UserWarning.
_SINKHORN_KWARGS: dict = {"method": "sinkhorn", "reg": 1.0, "tol": 1e-5}


torch.manual_seed(0)
_X = torch.randn(50, 8)
_Y_NEAR = _X + 0.1
_Y_FAR = torch.randn(50, 8) + 10.0


class TestOptimalTransportDistance:
    def test_no_coupling(self, embedding_tensor_dataset, embedding_tensor_dataset_2):
        cost = ot_distance(
            embedding_tensor_dataset, embedding_tensor_dataset_2, return_coupling=False
        )
        assert isinstance(cost, float)

    def test_return_coupling(
        self, embedding_tensor_dataset, embedding_tensor_dataset_2
    ):
        result = ot_distance(
            embedding_tensor_dataset,
            embedding_tensor_dataset_2,
            use_flash_sinkhorn=False,
            method="sinkhorn",
            return_coupling=True,
        )
        assert isinstance(result, tuple)
        _, coupling = result
        N = len(embedding_tensor_dataset)
        M = len(embedding_tensor_dataset_2)
        assert coupling.shape == (N, M)

    def test_same_dataset_near_zero(self, embedding_tensor_dataset):
        cost = ot_distance(
            embedding_tensor_dataset,
            embedding_tensor_dataset,
        )
        assert cost < 1e-5


class TestPythonOT:
    def test_identical_clouds_zero(self):
        cost, _ = python_ot(_X, _X.clone())
        assert cost.item() == pytest.approx(0, abs=1e-5)

    def test_distance_ordering(self):
        cost_near, _ = python_ot(_X, _Y_NEAR)
        cost_far, _ = python_ot(_X, _Y_FAR)
        assert cost_far.item() > cost_near.item()

    def test_custom_weights(self):
        w = torch.ones(len(_X)) / len(_X)
        cost, _ = python_ot(_X, _Y_NEAR, a=w, b=w)
        assert cost.ndim == 0


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
        M = len(_Y_NEAR)
        _, plan = python_ot(_X, _Y_NEAR, return_coupling=True)
        assert torch.allclose(plan.sum(dim=1), torch.full((N,), 1.0 / N), atol=1e-6)
        assert torch.allclose(plan.sum(dim=0), torch.full((M,), 1.0 / M), atol=1e-6)

    def test_total_mass_is_one(self):
        _, plan = python_ot(_X, _Y_NEAR, return_coupling=True)
        assert torch.isclose(plan.sum(), torch.tensor(1.0), atol=1e-6)

    def test_identical_clouds_concentrates_on_diagonal(self):
        _, plan = python_ot(_X, _X.clone(), return_coupling=True)
        assert plan.diagonal().sum() > 0.99999 * plan.sum()


class TestPythonOTSinkhorn:
    def test_returns_scalar(self):
        cost, _ = python_ot(_X, _Y_NEAR, **_SINKHORN_KWARGS)
        assert cost.ndim == 0

    def test_distance_ordering(self):
        # With entropic regularisation the cost is not near zero, but identical
        # clouds should cost no more than shifted ones.
        cost_identical, _ = python_ot(_X, _X.clone(), **_SINKHORN_KWARGS)
        cost_near, _ = python_ot(_X, _Y_NEAR, **_SINKHORN_KWARGS)
        cost_far, _ = python_ot(_X, _Y_FAR, **_SINKHORN_KWARGS)

        assert cost_near.item() > cost_identical.item()
        assert cost_far.item() > cost_near.item()


class TestPythonOTSinkhornTransportPlan:
    def test_returns_2d_tensor(self):
        _, plan = python_ot(_X, _Y_NEAR, return_coupling=True, **_SINKHORN_KWARGS)
        assert plan.ndim == 2
        assert plan.shape == (len(_X), len(_Y_NEAR))

    def test_non_negative(self):
        _, plan = python_ot(_X, _Y_NEAR, return_coupling=True, **_SINKHORN_KWARGS)
        assert (plan >= 0).all()

    def test_row_marginals_approximately_uniform(self):
        N = len(_X)
        M = len(_Y_NEAR)
        _, plan = python_ot(_X, _Y_NEAR, return_coupling=True, **_SINKHORN_KWARGS)
        assert torch.allclose(plan.sum(dim=1), torch.full((N,), 1.0 / N), atol=1e-5)
        assert torch.allclose(plan.sum(dim=0), torch.full((M,), 1.0 / M), atol=1e-5)

    def test_total_mass_is_one(self):
        _, plan = python_ot(_X, _Y_NEAR, return_coupling=True, **_SINKHORN_KWARGS)
        assert torch.isclose(plan.sum(), torch.tensor(1.0), atol=1e-5)

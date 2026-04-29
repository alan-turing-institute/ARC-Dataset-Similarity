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


class TestOptimalTransportDistance:
    def test_sinkhorn_method(self):
        result = optimal_transport_distance(_X, _Y_NEAR, method="sinkhorn")
        assert result.ndim == 0

    def test_unsupported_method_raises(self):
        with pytest.raises(ValueError, match="Unsupported OT method"):
            optimal_transport_distance(_X, _Y_NEAR, method="bogus")

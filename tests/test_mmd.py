import pytest
import torch

from dataset_similarity.metrics.mmd import DistMethod, compute

_N = 64
_D = 32

_METHODS: list[DistMethod] = ["cdist", "matmul", "chunked"]


@pytest.mark.parametrize("method", _METHODS)
def test_compute_identical_datasets_returns_zero(method: DistMethod) -> None:
    """MMD between a dataset and an exact copy of itself must be 0."""
    torch.manual_seed(0)
    dataset_a = torch.randn(_N, _D)
    dataset_b = dataset_a.clone()

    result = compute(dataset_a, dataset_b, method=method)

    assert result == pytest.approx(
        0.0, abs=1e-6
    ), f"Expected MMD ≈ 0 for identical datasets (method={method!r}), got {result}"

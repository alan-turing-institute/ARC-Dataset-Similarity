"""Tests for dataset_similarity.data.domainnet."""

import pytest

from dataset_similarity.data.domainnet import DomainNetDataset


def test_domainnet_import() -> None:
    assert DomainNetDataset is not None


def test_domainnet_domains() -> None:
    assert "real" in DomainNetDataset.DOMAINS
    assert len(DomainNetDataset.DOMAINS) == 6


def test_domainnet_invalid_domain() -> None:

    with pytest.raises(ValueError, match="Unknown domain"):
        DomainNetDataset(data_root="data/DomainNet", domain="invalid")  # type: ignore[arg-type]


def test_domainnet_invalid_split() -> None:

    with pytest.raises(ValueError, match="Unknown split"):
        DomainNetDataset(data_root="data/DomainNet", domain="real", split="val")  # type: ignore[arg-type]

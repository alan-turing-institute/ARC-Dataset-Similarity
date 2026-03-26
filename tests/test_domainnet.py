"""Tests for dataset_similarity.data.domainnet."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from dataset_similarity.data.domainnet import DomainNetDataset


@pytest.fixture()
def domainnet_root(tmp_path: Path) -> Path:
    """Create a minimal fake DomainNet directory structure."""
    domain = "real"
    classes = ["cat", "dog"]

    for label, class_name in enumerate(classes):
        class_dir = tmp_path / domain / class_name
        class_dir.mkdir(parents=True)
        img_path = class_dir / "image_001.jpg"
        Image.new("RGB", (64, 64), color=(label * 80, 120, 200)).save(img_path)

    split_lines = [
        f"{domain}/cat/image_001.jpg 0",
        f"{domain}/dog/image_001.jpg 1",
    ]
    (tmp_path / f"{domain}_train.txt").write_text("\n".join(split_lines))

    return tmp_path


class TestImports:
    def test_domainnet_import(self) -> None:
        assert DomainNetDataset is not None


class TestDomainNetDataset:
    def test_domains(self) -> None:
        assert "real" in DomainNetDataset.DOMAINS
        assert len(DomainNetDataset.DOMAINS) == 6

    def test_invalid_domain(self) -> None:
        with pytest.raises(ValueError, match="Unknown domain"):
            DomainNetDataset(data_root="data/DomainNet", domain="invalid")  # type: ignore[arg-type]

    def test_invalid_split(self) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            DomainNetDataset(data_root="data/DomainNet", domain="real", split="val")  # type: ignore[arg-type]

    def test_missing_split_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DomainNetDataset(data_root=tmp_path, domain="real")

    def test_len(self, domainnet_root: Path) -> None:
        dataset = DomainNetDataset(data_root=domainnet_root, domain="real")
        assert len(dataset) == 2

    def test_getitem_shape(self, domainnet_root: Path) -> None:
        dataset = DomainNetDataset(data_root=domainnet_root, domain="real")
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape[0] == 3  # C, H, W
        assert isinstance(label, int)

    def test_class_count(self, domainnet_root: Path) -> None:
        dataset = DomainNetDataset(data_root=domainnet_root, domain="real")
        assert dataset.class_count == 2

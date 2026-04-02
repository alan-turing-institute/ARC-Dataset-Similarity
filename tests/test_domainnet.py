"""Tests for dataset_similarity.data.domainnet."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from dataset_similarity.data.domainnet import DomainNetDataset


@pytest.fixture()
def domainnet_root(tmp_path: Path) -> Path:
    """Create a minimal fake DomainNet directory structure.

    Layout mirrors the real data layout so the dataset can locate metadata::

        tmp_path/
          metadata/
            domainnet_class_mapping.yaml
          DomainNet/          <- returned as data_root
            real/
              cat/image_001.jpg
              dog/image_001.jpg
            real_train.txt
    """
    domain = "real"
    classes = ["cat", "dog"]

    data_root = tmp_path / "DomainNet"

    for label, class_name in enumerate(classes):
        class_dir = data_root / domain / class_name
        class_dir.mkdir(parents=True)
        img_path = class_dir / "image_001.jpg"
        Image.new("RGB", (64, 64), color=(label * 80, 120, 200)).save(img_path)

    split_lines = [
        f"{domain}/cat/image_001.jpg 0",
        f"{domain}/dog/image_001.jpg 1",
    ]
    (data_root / f"{domain}_train.txt").write_text("\n".join(split_lines))

    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "domainnet_class_mapping.yaml").write_text("cat: 0\ndog: 1\n")

    return data_root


class TestDomainNetDataset:
    def test_domains(self) -> None:
        assert "real" in DomainNetDataset.DOMAINS
        assert len(DomainNetDataset.DOMAINS) == 6

    def test_invalid_domain(self) -> None:
        with pytest.raises(ValueError, match="Unknown domain"):
            DomainNetDataset(data_root="data/DomainNet", domains="invalid")  # type: ignore[arg-type]

    def test_invalid_split(self) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            DomainNetDataset(data_root="data/DomainNet", domains="real", split="val")  # type: ignore[arg-type]

    def test_missing_split_file(self, tmp_path: Path) -> None:
        # No metadata dir or split file — expects FileNotFoundError
        (tmp_path / "metadata").mkdir()
        (tmp_path / "metadata" / "domainnet_class_mapping.yaml").write_text("cat: 0\n")
        data_root = tmp_path / "DomainNet"
        data_root.mkdir()
        with pytest.raises(FileNotFoundError):
            DomainNetDataset(data_root=data_root, domains="real")

    def test_len(self, domainnet_root: Path) -> None:
        dataset = DomainNetDataset(data_root=domainnet_root, domains="real")
        assert len(dataset) == 2

    def test_getitem_shape(self, domainnet_root: Path) -> None:
        dataset = DomainNetDataset(data_root=domainnet_root, domains="real")
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape[0] == 3  # C, H, W
        assert isinstance(label, int)

    def test_class_count(self, domainnet_root: Path) -> None:
        dataset = DomainNetDataset(data_root=domainnet_root, domains="real")
        assert dataset.class_count == 2

"""Tests for dataset_similarity.data.base."""

from pathlib import Path
from typing import Any

import pytest
from pandas import DataFrame, concat

from dataset_similarity.data.base import ImageDataset
from dataset_similarity.data.utils import from_yaml, model_mapping


class FakeDataset(ImageDataset):
    """Minimal concrete subclass used to exercise base-class behaviour."""

    def __init__(
        self,
        data_root: Path | str,
        split: str = "train",
        **kwargs: Any,
    ) -> None:
        super().__init__(data_root, split, **kwargs)


def _make_fake_dataset(
    tmp_path: Path,
    n_classes: int = 3,
    samples_per_class: int = 4,
) -> FakeDataset:
    """Return a FakeDataset with synthetic samples."""
    ds = FakeDataset(data_root=tmp_path, split="train")
    ds.classes = [str(i) for i in range(n_classes)]
    ds.data = DataFrame(
        [
            {"path": str(tmp_path / f"class{cls}_img{i}.jpg"), "label": cls}
            for cls in range(n_classes)
            for i in range(samples_per_class)
        ]
    )
    return ds


class TestStripSingleClassesFromSamples:
    def test_removes_singleton_class(self, tmp_path: Path) -> None:
        ds = _make_fake_dataset(tmp_path, n_classes=2, samples_per_class=2)
        ds.data = concat(
            [ds.data, DataFrame([{"path": str(tmp_path / "lone.jpg"), "label": 99}])],
            ignore_index=True,
        )

        ds._strip_single_classes_from_samples()

        remaining_labels = set(ds.data["label"])
        assert 99 not in remaining_labels
        assert len(ds.data) == 4

    def test_multi_sample_classes_are_kept(self, tmp_path: Path) -> None:
        ds = _make_fake_dataset(tmp_path, n_classes=3, samples_per_class=2)
        original = ds.data.copy()

        ds._strip_single_classes_from_samples()

        assert ds.data.equals(original)

    def test_prints_warning_for_singleton(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ds = _make_fake_dataset(tmp_path, n_classes=2, samples_per_class=2)
        ds.data = concat(
            [ds.data, DataFrame([{"path": str(tmp_path / "lone.jpg"), "label": 99}])],
            ignore_index=True,
        )

        ds._strip_single_classes_from_samples()

        out = capsys.readouterr().out
        assert out == "Warning: Found label '99' with only a single example\n"


class TestStratifyByClass:
    def test_float_size_reduces_dataset(self, tmp_path: Path) -> None:
        # 12 total --> 6 samples
        ds = _make_fake_dataset(tmp_path, n_classes=3, samples_per_class=4)
        ds.stratify_by_class(size=0.5, random_seed=0)
        assert len(ds.data) == 6

    def test_int_size_reduces_dataset(self, tmp_path: Path) -> None:
        # 12 total --> 6 samples
        ds = _make_fake_dataset(tmp_path, n_classes=3, samples_per_class=4)
        ds.stratify_by_class(size=7, random_seed=0)
        assert len(ds.data) == 7

    def test_float_size_is_stratified_across_classes(self, tmp_path: Path) -> None:
        ds = _make_fake_dataset(tmp_path, n_classes=3, samples_per_class=4)
        ds.stratify_by_class(size=0.5, random_seed=0)

        label_counts = ds.data["label"].value_counts().to_dict()

        # Equal representation across all 3 classes (2 each)
        assert len(set(label_counts.values())) == 1
        assert len(label_counts) == 3

    def test_value_errors(self, tmp_path: Path) -> None:
        ds = _make_fake_dataset(tmp_path)
        with pytest.raises(ValueError, match=r"Got 1\.5 instead"):
            ds.stratify_by_class(size=1.5)
        with pytest.raises(ValueError, match=r"Got -1 instead"):
            ds.stratify_by_class(size=-1)
        with pytest.raises(ValueError, match=r"Got 0 instead"):
            ds.stratify_by_class(size=0)
        with pytest.raises(ValueError, match=r"Got 'half' instead"):
            ds.stratify_by_class(size="half")


class TestFromYaml:
    @pytest.fixture(autouse=True)
    def _register_fake(self) -> Any:
        model_mapping["fake"] = FakeDataset
        yield
        del model_mapping["fake"]

    def test_creates_registered_subclass(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(f"name: fake\nargs:\n  data_root: {tmp_path}\n")

        ds = from_yaml(yaml_path)

        assert isinstance(ds, FakeDataset)

    def test_data_root_is_set_correctly(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(f"name: fake\nargs:\n  data_root: {tmp_path}\n")

        ds = from_yaml(yaml_path)

        assert ds.root == tmp_path

    def test_kwargs_are_forwarded(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            f"name: fake\nargs:\n  data_root: {tmp_path}\n  split: val\n"
        )

        ds = from_yaml(yaml_path)

        assert ds.split == "val"

    def test_missing_name_raises(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(f"args:\n  data_root: {tmp_path}\n")

        with pytest.raises(
            ValueError,
            match="YAML config must contain a 'name' key specifying the dataset name",
        ):
            from_yaml(yaml_path)

    def test_missing_data_root_raises(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("name: fake\n")

        with pytest.raises(
            ValueError,
            match="YAML config must contain a 'data_root' key specifying the dataset"
            " root directory",
        ):
            from_yaml(yaml_path)

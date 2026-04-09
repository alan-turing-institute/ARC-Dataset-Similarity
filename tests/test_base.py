"""Tests for dataset_similarity.data.base."""

from pathlib import Path
from typing import Any

import pytest

from dataset_similarity.data.base import ImageDataset


class FakeDataset(ImageDataset, name="fake"):
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
    ds.samples = [
        (tmp_path / f"class{cls}_img{i}.jpg", cls)
        for cls in range(n_classes)
        for i in range(samples_per_class)
    ]
    return ds


class TestStripSingleClassesFromSamples:
    def test_removes_singleton_class(self, tmp_path: Path) -> None:
        ds = _make_fake_dataset(tmp_path, n_classes=2, samples_per_class=2)
        ds.samples.append((tmp_path / "lone.jpg", 99))

        ds._strip_single_classes_from_samples()

        remaining_labels = {label for _, label in ds.samples}
        assert 99 not in remaining_labels
        assert len(ds.samples) == 4

    def test_multi_sample_classes_are_kept(self, tmp_path: Path) -> None:
        ds = _make_fake_dataset(tmp_path, n_classes=3, samples_per_class=2)
        original = list(ds.samples)

        ds._strip_single_classes_from_samples()

        assert ds.samples == original

    def test_prints_warning_for_singleton(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ds = _make_fake_dataset(tmp_path, n_classes=2, samples_per_class=2)
        ds.samples.append((tmp_path / "lone.jpg", 99))

        ds._strip_single_classes_from_samples()

        out = capsys.readouterr().out
        assert (
            out == "Warning: Found label '99' with only 1 sample."
            " Removing from dataset...\n"
        )


class TestStratifyByClass:
    def test_float_size_reduces_dataset(self, tmp_path: Path) -> None:
        # 12 total --> 6 samples
        ds = _make_fake_dataset(tmp_path, n_classes=3, samples_per_class=4)
        ds.stratify_by_class(size=0.5, random_seed=0)
        assert len(ds.samples) == 6

    def test_int_size_reduces_dataset(self, tmp_path: Path) -> None:
        # 12 total --> 6 samples
        ds = _make_fake_dataset(tmp_path, n_classes=3, samples_per_class=4)
        ds.stratify_by_class(size=7, random_seed=0)
        assert len(ds.samples) == 7

    def test_float_size_is_stratified_across_classes(self, tmp_path: Path) -> None:
        ds = _make_fake_dataset(tmp_path, n_classes=3, samples_per_class=4)
        ds.stratify_by_class(size=0.5, random_seed=0)

        label_counts: dict[int, int] = {}
        for _, label in ds.samples:
            label_counts[label] = label_counts.get(label, 0) + 1

        # Equal representation across all 3 classes (2 each)
        assert len(set(label_counts.values())) == 1
        assert len(label_counts) == 3

    def test_float_greater_than_one_raises(self, tmp_path: Path) -> None:
        ds = _make_fake_dataset(tmp_path)
        with pytest.raises(
            ValueError, match=r"If 'size' is a float, it must be in the range \(0, 1\)"
        ):
            ds.stratify_by_class(size=1.5)

    def test_float_zero_raises(self, tmp_path: Path) -> None:
        ds = _make_fake_dataset(tmp_path)
        with pytest.raises(
            ValueError, match=r"If 'size' is a float, it must be in the range \(0, 1\)"
        ):
            ds.stratify_by_class(size=0.0)

    def test_int_zero_raises(self, tmp_path: Path) -> None:
        ds = _make_fake_dataset(tmp_path)
        with pytest.raises(
            ValueError, match="If 'size' is an int, it must be a positive integer"
        ):
            ds.stratify_by_class(size=0)

    def test_int_exceeds_sample_count_raises(self, tmp_path: Path) -> None:
        # Manually reduce samples so len(samples) < len(classes)
        ds = _make_fake_dataset(tmp_path, n_classes=3, samples_per_class=1)
        ds.samples = ds.samples[:2]  # 2 samples, but 3 classes
        with pytest.raises(
            ValueError,
            match="If 'size' is an int, it cannot be larger than the number of samples"
            " in the dataset",
        ):
            ds.stratify_by_class(size=3)

    def test_invalid_type_raises(self, tmp_path: Path) -> None:
        ds = _make_fake_dataset(tmp_path)
        with pytest.raises(
            TypeError,
            match=r"size must be either a float in \(0, 1\) or a positive integer",
        ):
            ds.stratify_by_class(size="half")  # type: ignore[arg-type]


class TestFromYaml:
    def test_creates_registered_subclass(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(f"name: fake\ndata_root: {tmp_path}\n")

        ds = ImageDataset.from_yaml(yaml_path)

        assert isinstance(ds, FakeDataset)

    def test_data_root_is_set_correctly(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(f"name: fake\ndata_root: {tmp_path}\n")

        ds = ImageDataset.from_yaml(yaml_path)

        assert ds.root == tmp_path

    def test_kwargs_are_forwarded(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            f"name: fake\ndata_root: {tmp_path}\nkwargs:\n  split: val\n"
        )

        ds = ImageDataset.from_yaml(yaml_path)

        assert ds.split == "val"

    def test_missing_name_raises(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(f"data_root: {tmp_path}\n")

        with pytest.raises(
            ValueError,
            match="YAML config must contain a 'name' key specifying the dataset name",
        ):
            ImageDataset.from_yaml(yaml_path)

    def test_missing_data_root_raises(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("name: fake\n")

        with pytest.raises(
            ValueError,
            match="YAML config must contain a 'data_root' key specifying the dataset"
            " root directory",
        ):
            ImageDataset.from_yaml(yaml_path)

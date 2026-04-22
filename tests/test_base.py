"""Tests for dataset_similarity.data.base."""

from pathlib import Path

import pytest
from pandas import DataFrame, concat

from dataset_similarity.data.base import ImageDataset


class FakeDataset(ImageDataset):
    def __init__(
        self,
        dataset_dir: Path | str,
        target_classes: list[str] | None = None,
        split: str = "train",
        size: float | int | None = None,
        random_seed: int | None = None,
        n_classes: int = 3,
        samples_per_class: int = 4,
        embedding=None,
        embedding_dir=None,
        return_paths=False,
    ) -> None:
        self.n_classes = [f"class{cls}" for cls in range(n_classes)]
        self.tmp_pth = Path(dataset_dir)
        self.samples_per_class = samples_per_class
        super().__init__(
            dataset_dir=dataset_dir,
            target_classes=target_classes,
            split=split,
            size=size,
            random_seed=random_seed,
            embedding=embedding,
            embedding_dir=embedding_dir,
            return_paths=return_paths,
        )

    def _load_data(self) -> DataFrame:
        return DataFrame(
            [
                {"path": str(self.tmp_pth / f"class{cls}_img{i}.jpg"), "label": cls}
                for cls in self.n_classes
                for i in range(self.samples_per_class)
            ]
        )


def _make_fake_dataset(
    tmp_path: Path,
    n_classes: int = 3,
    samples_per_class: int = 4,
    size: float | int | None = None,
    random_seed: int | None = None,
) -> FakeDataset:
    """Return a FakeDataset with synthetic samples."""
    return FakeDataset(
        dataset_dir=tmp_path,
        split="train",
        size=size,
        random_seed=random_seed,
        n_classes=n_classes,
        samples_per_class=samples_per_class,
    )


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


class TestSubsampleData:
    def test_float_size_reduces_dataset(self, tmp_path: Path) -> None:
        # 12 total --> 6 samples (subsampling happens in __init__)
        ds = _make_fake_dataset(
            tmp_path, n_classes=3, samples_per_class=4, size=0.5, random_seed=0
        )
        assert len(ds.data) == 6

    def test_int_size_reduces_dataset(self, tmp_path: Path) -> None:
        # 12 total --> 7 samples (subsampling happens in __init__)
        ds = _make_fake_dataset(
            tmp_path, n_classes=3, samples_per_class=4, size=7, random_seed=0
        )
        assert len(ds.data) == 7

    def test_float_size_is_stratified_across_classes(self, tmp_path: Path) -> None:
        # Subsampling happens in __init__ with size=0.5
        ds = _make_fake_dataset(
            tmp_path, n_classes=3, samples_per_class=4, size=0.5, random_seed=0
        )

        label_counts = ds.data["label"].value_counts().to_dict()

        # Equal representation across all 3 classes (2 each)
        assert len(set(label_counts.values())) == 1
        assert len(label_counts) == 3

    def test_value_errors(self, tmp_path: Path) -> None:
        # Errors should be raised during dataset creation (in __init__)
        # when invalid size values trigger train_test_split validation
        with pytest.raises((ValueError, TypeError)):
            _make_fake_dataset(tmp_path, size=1.5)

        with pytest.raises((ValueError, TypeError)):
            _make_fake_dataset(tmp_path, size=-1)

        with pytest.raises((ValueError, TypeError)):
            _make_fake_dataset(tmp_path, size=0)

        with pytest.raises((ValueError, TypeError)):
            _make_fake_dataset(tmp_path, size="half")


class TestFromDict:
    def test_creates_dataset_from_dict(self, tmp_path: Path) -> None:
        config_dict = {
            "dataset_dir": tmp_path,
            "split": "train",
            "size": None,
            "random_seed": None,
            "n_classes": 3,
            "samples_per_class": 4,
        }

        ds = FakeDataset.from_dict(config_dict)

        assert isinstance(ds, FakeDataset)
        assert ds.dataset_dir == tmp_path
        assert ds.split == "train"
        assert len(ds.data) == 12  # 3 classes * 4 samples

    def test_dataset_dir_is_set_correctly(self, tmp_path: Path) -> None:
        config_dict = {"dataset_dir": tmp_path}

        ds = FakeDataset.from_dict(config_dict)

        assert ds.dataset_dir == tmp_path

    def test_kwargs_are_forwarded(self, tmp_path: Path) -> None:
        config_dict = {
            "dataset_dir": tmp_path,
            "split": "val",
            "n_classes": 5,
            "samples_per_class": 2,
        }

        ds = FakeDataset.from_dict(config_dict)

        assert ds.split == "val"
        assert len(ds.data) == 10  # 5 classes * 2 samples

    def test_with_subsampling(self, tmp_path: Path) -> None:
        config_dict = {
            "dataset_dir": tmp_path,
            "split": "train",
            "size": 0.5,
            "random_seed": 42,
            "n_classes": 3,
            "samples_per_class": 4,
        }

        ds = FakeDataset.from_dict(config_dict)

        assert len(ds.data) == 6  # 50% of 12


class TestFromYaml:
    def test_creates_dataset_from_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            f"name: fake\n"
            f"kwargs:\n"
            f"  dataset_dir: {tmp_path}\n"
            f"  split: train\n"
            f"  n_classes: 3\n"
            f"  samples_per_class: 4\n"
        )

        ds = FakeDataset.from_yaml(yaml_path)

        assert isinstance(ds, FakeDataset)
        assert ds.dataset_dir == tmp_path
        assert len(ds.data) == 12

    def test_dataset_dir_is_set_correctly(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(f"kwargs:\n  dataset_dir: {tmp_path}\n")

        ds = FakeDataset.from_yaml(yaml_path)

        assert ds.dataset_dir == tmp_path

    def test_kwargs_are_forwarded(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            f"name: fake\n"
            f"kwargs:\n"
            f"  dataset_dir: {tmp_path}\n"
            f"  split: val\n"
            f"  n_classes: 5\n"
        )

        ds = FakeDataset.from_yaml(yaml_path)

        assert ds.split == "val"

    def test_name_key_is_ignored(self, tmp_path: Path) -> None:
        # The 'name' key can exist in YAML but is ignored by from_yaml
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            f"name: some_other_name\n" f"kwargs:\n" f"  dataset_dir: {tmp_path}\n"
        )

        ds = FakeDataset.from_yaml(yaml_path)

        assert isinstance(ds, FakeDataset)

    def test_missing_kwargs_key_raises(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(f"name: fake\ndataset_dir: {tmp_path}\n")

        with pytest.raises(KeyError):
            FakeDataset.from_yaml(yaml_path)

    def test_with_subsampling(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            f"name: fake\n"
            f"kwargs:\n"
            f"  dataset_dir: {tmp_path}\n"
            f"  split: train\n"
            f"  size: 0.5\n"
            f"  random_seed: 42\n"
            f"  n_classes: 3\n"
            f"  samples_per_class: 4\n"
        )

        ds = FakeDataset.from_yaml(yaml_path)

        assert len(ds.data) == 6  # 50% of 12

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(f"kwargs:\n  dataset_dir: {tmp_path}\n")

        # Pass as string instead of Path
        ds = FakeDataset.from_yaml(str(yaml_path))

        assert isinstance(ds, FakeDataset)
        assert ds.dataset_dir == tmp_path

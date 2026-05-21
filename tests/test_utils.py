from pathlib import Path

import pytest
from yaml import dump

from dataset_similarity.utils import get_embedding_path, load_yaml_from_path


def test_get_embedding_path_returns_safetensors():
    data_root = Path("/data")
    image_path = Path("/data/DomainNet/painting/cat/img_001.jpg")
    embedding_dir = Path("/embeddings/clip")
    result = get_embedding_path(image_path, embedding_dir, data_root=data_root)
    assert result == Path("/embeddings/clip/DomainNet/painting/cat/img_001.safetensors")


def test_get_embedding_path_raises_when_image_outside_data_root():
    data_root = Path("/data")
    image_path = Path("/other/images/img.jpg")
    embedding_dir = Path("/embeddings/clip")
    with pytest.raises(ValueError, match="not in the subpath"):
        get_embedding_path(image_path, embedding_dir, data_root=data_root)


def test_load_yaml_returns_dict(tmp_path):
    data = {"key": "value", "nested": {"a": 1}}
    f = tmp_path / "config.yaml"
    f.write_text(dump(data))
    assert load_yaml_from_path(f) == data


def test_load_yaml_accepts_string_path(tmp_path):
    data = {"x": 42}
    f = tmp_path / "config.yaml"
    f.write_text(dump(data))
    assert load_yaml_from_path(str(f)) == data


def test_load_yaml_raises_on_empty_file(tmp_path):
    f = tmp_path / "empty.yaml"
    f.write_text("")
    with pytest.raises(ValueError, match="top-level mapping"):
        load_yaml_from_path(f)


def test_load_yaml_raises_on_list(tmp_path):
    f = tmp_path / "list.yaml"
    f.write_text(dump(["a", "b"]))
    with pytest.raises(ValueError, match="top-level mapping"):
        load_yaml_from_path(f)


def test_load_yaml_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml_from_path(tmp_path / "nonexistent.yaml")

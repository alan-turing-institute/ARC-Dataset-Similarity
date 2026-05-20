from pathlib import Path

from dataset_similarity.utils import get_embedding_path


def test_embedding_path(tmp_path: Path) -> None:
    data_dir = Path("/data/FakeDataset/")
    src = data_dir / "train/n01234/img001.jpg"
    result = get_embedding_path(src, tmp_path / "clip", data_dir.parent)
    assert result == tmp_path / "clip/FakeDataset/train/n01234/img001.safetensors"

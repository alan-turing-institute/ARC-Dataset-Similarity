from pathlib import Path

from dataset_similarity.utils import get_embedding_path


def test_get_embedding_path_returns_safetensors():
    data_root = Path("/data")
    image_path = Path("/data/DomainNet/painting/cat/img_001.jpg")
    embedding_dir = Path("/embeddings/clip")
    result = get_embedding_path(image_path, embedding_dir, data_root=data_root)
    assert result == Path("/embeddings/clip/DomainNet/painting/cat/img_001.safetensors")

from dataset_similarity.constants import DATA_DIR, EMBEDDING_DIR
from dataset_similarity.utils import get_embedding_path


def test_get_embedding_path_returns_safetensors():
    image_path = DATA_DIR / "DomainNet/painting/cat/img_001.jpg"
    result = get_embedding_path(image_path, "clip")
    assert result == EMBEDDING_DIR / "clip/DomainNet/painting/cat/img_001.safetensors"

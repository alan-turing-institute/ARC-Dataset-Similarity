import torch

from dataset_similarity.constants import DATA_DIR, EMBEDDING_DIR
from dataset_similarity.utils import eval_metrics, get_embedding_path


def test_get_embedding_path_returns_safetensors():
    image_path = DATA_DIR / "DomainNet/painting/cat/img_001.jpg"
    result = get_embedding_path(image_path, "clip")
    assert result == EMBEDDING_DIR / "clip/DomainNet/painting/cat/img_001.safetensors"


# Test cases for eval_metrics function where logits are different shapes
def test_eval_metrics_handles_1d_logits():
    logits = torch.tensor([2.0, -2.0, 3.0, -1.0])
    labels = [1, 0, 1, 0]
    result = eval_metrics(logits, labels)
    assert result["accuracy"] == 1.0


def test_eval_metrics_handles_2d_single_logit():
    logits = torch.tensor([[2.0], [-2.0], [3.0], [-1.0]])
    labels = [1, 0, 1, 0]
    result = eval_metrics(logits, labels)
    assert result["accuracy"] == 1.0


def test_eval_metrics_handles_2d_two_logit():
    logits = torch.tensor([[-2.0, 2.0], [2.0, -2.0], [-3.0, 3.0], [1.0, -1.0]])
    labels = [1, 0, 1, 0]
    result = eval_metrics(logits, labels)
    assert result["accuracy"] == 1.0

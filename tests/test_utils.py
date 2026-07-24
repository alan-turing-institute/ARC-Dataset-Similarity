import pytest
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


def test_eval_metrics_binary_gives_meaningful_metrics():
    # Two correct, two wrong predictions (threshold 0.5), so metrics land on a
    # non-degenerate value rather than a trivial 0.0/1.0.
    logits = torch.tensor([3.0, 1.0, -1.0, -3.0])
    labels = [1, 0, 1, 0]

    result = eval_metrics(logits, labels, multi_label=False)

    assert result["accuracy"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert result["roc_auc"] == pytest.approx(0.75)


def test_eval_metrics_multi_label_gives_meaningful_metrics():
    # Column 0 mirrors the binary case above (two right, two wrong); column 1
    # is predicted perfectly, so macro/micro averages land on distinct,
    # hand-checkable values rather than both collapsing to 0 or 1.
    logits = torch.tensor(
        [
            [3.0, -3.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-3.0, 3.0],
        ]
    )
    labels = [
        [1, 0],
        [0, 0],
        [1, 1],
        [0, 1],
    ]

    result = eval_metrics(logits, labels, multi_label=True)

    assert result["accuracy"] == pytest.approx(0.5)
    assert result["precision_macro"] == pytest.approx(0.75)
    assert result["recall_macro"] == pytest.approx(0.75)
    assert result["f1_macro"] == pytest.approx(0.75)
    assert result["roc_auc_macro"] == pytest.approx(0.875)
    assert result["precision_micro"] == pytest.approx(0.75)
    assert result["recall_micro"] == pytest.approx(0.75)
    assert result["f1_micro"] == pytest.approx(0.75)
    assert result["roc_auc_micro"] == pytest.approx(0.875)


def test_eval_metrics_multi_label_falls_back_to_zero_for_degenerate_column():
    # Column 1 is all-negative: ROC AUC / average precision are undefined for
    # it, so macro (needs a per-column score) falls back to 0.0.
    logits = torch.tensor(
        [
            [3.0, -3.0],
            [1.0, -1.0],
            [-1.0, -2.0],
            [-3.0, -0.5],
        ]
    )
    labels = [
        [1, 0],
        [0, 0],
        [1, 0],
        [0, 0],
    ]

    result = eval_metrics(logits, labels, multi_label=True)

    assert result["roc_auc_macro"] == 0.0
    assert result["roc_auc_micro"] == pytest.approx(19 / 24)
    assert result["average_precision_macro"] == 0.0
    assert result["average_precision_micro"] == pytest.approx(0.7)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["precision_macro"] == pytest.approx(0.25)

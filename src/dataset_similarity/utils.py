import stat
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from yaml import safe_dump, safe_load

from dataset_similarity.constants import DATA_DIR, EMBEDDING_DIR


def get_embedding_path(
    image_path: Path,
    embedding: str,
) -> Path:
    """
    Helper function which given an absolute image path and the name of an embedding
    model, returns a path where embeddings of that image can be saved or loaded.

    Args:
        image_path: An absolute path to an image file in the original dataset directory.
        embedding: A string denoting the name of the embedding model, e.g. "clip". This
            is used to determine the directory where the embedding for the image is
            stored, e.g. `.constants.DATA_DIR / "clip"`.

    Returns:
        Path: The absolute path where the embedding for the image can be saved or
            loaded.
    """
    return (
        EMBEDDING_DIR
        / embedding
        / image_path.relative_to(DATA_DIR).with_suffix(".safetensors")
    )


def load_yaml_from_path(
    yaml_path: str | Path,
) -> dict[str, Any]:
    """
    Wrapper function around yaml.safe_load to load a YAML file from a given path and
    return its contents as a dictionary.

    Args:
        yaml_path: Path to the YAML file to be loaded.

    Returns:
        dict: The contents of the YAML file as a dictionary.
    """

    with open(yaml_path) as f:
        return dict(safe_load(f))


def save_yaml_to_path(
    data: dict[str, Any],
    yaml_path: str | Path,
) -> None:
    """
    Helper function to save a dictionary as a YAML file to a specified path.

    Args:
        data: The dictionary to be saved as YAML.
        yaml_path: The path where the YAML file should be saved.
    """

    with open(yaml_path, "w") as f:
        safe_dump(data, f)


class FixCheckpointPermissionsCallback(TrainerCallback):  # type: ignore[misc]
    """
    Trainer Callback which fixes permissions of the model safetensors file
    """

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: dict[str, Any],
    ) -> TrainerControl:
        """Grant group read access to the just-saved checkpoint's safetensors file."""
        model_file = (
            Path(args.output_dir)
            / f"checkpoint-{state.global_step}"
            / "model.safetensors"
        )
        if model_file.exists():
            # Trainer writes checkpoints without group-read; needed for shared access.
            model_file.chmod(model_file.stat().st_mode | stat.S_IRGRP)
        return control


def _multi_label_eval(
    logits: torch.Tensor,
    labels: list[int] | np.ndarray,
    additional_metrics: bool = False,
) -> dict[str, float]:
    """
    Compute multi-label classification metrics from Trainer predictions.

    Args:
        logits: The model logits, shape ``(N, C)``.
        labels: The true labels, shape ``(N, C)``.

    Returns:
        A dictionary containing the computed metrics.
    """
    probs = logits.sigmoid().numpy()
    preds = (probs > 0.5).astype(int)
    labels = np.asarray(labels).astype(int)

    base_metrics = {
        "accuracy": accuracy_score(labels, preds),
        "average_precision_macro": average_precision_score(
            labels, probs, average="macro"
        ),
        "average_precision_micro": average_precision_score(
            labels, probs, average="micro"
        ),
    }

    return (
        base_metrics
        | {
            "precision_macro": precision_score(
                labels, preds, average="macro", zero_division=0
            ),
            "precision_micro": precision_score(
                labels, preds, average="micro", zero_division=0
            ),
            "recall_macro": recall_score(
                labels, preds, average="macro", zero_division=0
            ),
            "recall_micro": recall_score(
                labels, preds, average="micro", zero_division=0
            ),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
            "roc_auc_macro": roc_auc_score(labels, probs, average="macro"),
            "roc_auc_micro": roc_auc_score(labels, probs, average="micro"),
        }
        if additional_metrics
        else base_metrics
    )


def _binary_eval(
    logits: torch.Tensor,
    labels: list[int] | np.ndarray,
    additional_metrics: bool = False,
) -> dict[str, float]:
    """
    Compute binary classification metrics from Trainer predictions.

    Args:
        logits: The model logits. Either a 1D tensor or a 2D tensor with last
            dim 1 (single-logit models, positive-class probability via sigmoid),
            or a 2D tensor with last dim 2 (older two-logit models, positive-class
            probability via softmax).
        labels: The true labels.

    Returns:
        A dictionary containing the computed metrics.
    """
    # single-logit models (num_labels=1) report a positive-class probability via
    # sigmoid — whether still shaped (N, 1) or already squeezed to (N,).
    if logits.ndim == 1:
        probs = logits.sigmoid().numpy()
    else:
        probs = logits.squeeze(-1).sigmoid().numpy()

    preds = (probs > 0.5).astype(int)
    base_metrics = {
        "accuracy": accuracy_score(y_true=labels, y_pred=preds),
        "average_precision": average_precision_score(y_true=labels, y_score=probs),
    }
    return (
        base_metrics
        | {
            "precision": precision_score(y_true=labels, y_pred=preds),
            "recall": recall_score(y_true=labels, y_pred=preds),
            "f1": f1_score(y_true=labels, y_pred=preds),
            "roc_auc": roc_auc_score(y_true=labels, y_score=probs),
        }
        if additional_metrics
        else base_metrics
    )


def eval_metrics(
    logits: torch.Tensor,
    labels: list[int] | np.ndarray,
    multi_label: bool = False,
    additional_metrics: bool = False,
) -> dict[str, float]:
    """
    Compute classification metrics from Trainer predictions. Metrics include accuracy,
    average precision, precision, recall, F1 score, and ROC AUC.

    Whether ``logits`` represents a multi-label problem cannot be inferred from its
    shape alone: a two-column tensor is equally consistent with a two-logit binary
    model (softmax, mutually-exclusive classes) or a two-label multi-label model
    (sigmoid, independent labels), so the caller must state which applies via
    ``multi_label``.

    For multi-label inputs, macro and micro averages are reported for all metrics.
    For binary inputs, a single scalar is reported per metric plus ROC AUC.

    Args:
        logits: The model logits. For binary inputs, a 1D tensor (single-logit
            models) or a 2D tensor with last dim 1 (single-logit) or 2 (older
            two-logit models). For multi-label inputs, a 2D tensor of shape
            ``(N, C)``.
        labels: The true labels. A 1D list/array for binary, or a 2D array of shape
            ``(N, C)`` for multi-label.
        multi_label: Whether ``logits``/``labels`` represent a multi-label problem.

    Returns:
        A dictionary containing the computed metrics.
    """
    if multi_label:
        return _multi_label_eval(logits, labels, additional_metrics)

    return _binary_eval(logits, labels, additional_metrics)

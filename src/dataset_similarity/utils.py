import stat
from pathlib import Path
from typing import Any

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
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: dict[str, Any],
    ) -> TrainerControl:
        model_file = (
            Path(args.output_dir)
            / f"checkpoint-{state.global_step}"
            / "model.safetensors"
        )
        if model_file.exists():
            model_file.chmod(model_file.stat().st_mode | stat.S_IRGRP)
        return control

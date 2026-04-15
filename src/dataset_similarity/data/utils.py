from pathlib import Path
from typing import Any

from yaml import safe_load

from dataset_similarity.constants import DEFAULT_DATA_DIR, DEFAULT_EMBEDDING_DIR


def _get_embedding_path(image_path: str | Path, model: str) -> Path:
    """
    Helper function which given an absolute image path and the name of an embedding
    model, returns a path where embeddings of that image can be saved or loaded.

    Args:
        image_path: An absolute path to an image file in the original dataset directory.
        model: The embedding model being used. E.g. "clip", "dinov3", etc.

    Returns:
        Path: The absolute path where the embedding for the image can be saved or
            loaded.
    """
    return (
        DEFAULT_EMBEDDING_DIR
        / model
        / Path(image_path).relative_to(DEFAULT_DATA_DIR).with_suffix(".safetensors")
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
    with Path(yaml_path).open() as f:
        dictionary: dict[str, Any] = safe_load(f)
    return dictionary

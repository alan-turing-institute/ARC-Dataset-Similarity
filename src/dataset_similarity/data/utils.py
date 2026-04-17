from pathlib import Path
from typing import Any

from yaml import safe_load

from dataset_similarity.constants import DEFAULT_DATA_ROOT


def get_embedding_path(
    image_path: Path,
    embedding_dir: Path,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    """
    Helper function which given an absolute image path and the name of an embedding
    model, returns a path where embeddings of that image can be saved or loaded.

    Args:
        image_path: An absolute path to an image file in the original dataset directory.
        embedding_dir: An absolute path to the directory where embeddings for the
            dataset are stored, e.g. `.cosntants.DEFAULT_EMBEDDING_DIR / "clip"`.
        data_root: An absolute path to the root directory of the original dataset. This
            is used to compute the relative path of the image within the dataset, which
            is then used to determine the embedding path. By default, this is set to
            `dataset_similarity.constants.DEFAULT_DATA_ROOT`, but can be overridden if
            the dataset is stored in a different location.

    Returns:
        Path: The absolute path where the embedding for the image can be saved or
            loaded.
    """
    return embedding_dir / image_path.relative_to(data_root).with_suffix(".safetensors")


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
        dictionary: dict[str, Any] = safe_load(f)
    return dictionary

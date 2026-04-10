from pathlib import Path
from typing import Any

from yaml import safe_load


def load_yaml_from_path(
    yaml_path: str | Path,
) -> dict[str, Any]:
    with Path(yaml_path).open() as f:
        dictionary: dict[str, Any] = safe_load(f)
    return dictionary

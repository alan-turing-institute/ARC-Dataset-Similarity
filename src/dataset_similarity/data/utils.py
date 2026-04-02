from pathlib import Path

from yaml import safe_load

DATADIR = Path(__file__).parent.parent.parent.parent / "data"


def load_domainnet_class_mapping(
    yaml_path: str | Path,
) -> dict[str, int]:
    with Path(yaml_path).open() as f:
        dictionary: dict[str, int] = safe_load(f)
    return dictionary


def load_imagenet_class_mapping(
    yaml_path: str | Path,
) -> dict[str, int]:
    with Path(yaml_path).open() as f:
        dictionary: dict[str, int] = safe_load(f)
    return dictionary

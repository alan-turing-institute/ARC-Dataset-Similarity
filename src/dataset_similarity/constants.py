from pathlib import Path

PROJECT_DIR = (Path(__file__) / ".." / ".." / "..").resolve()
DEFAULT_DATA_ROOT = PROJECT_DIR / "data"
DEFAULT_EMBEDDING_DIR = PROJECT_DIR / "embeddings"

DOMAINNET_DIR = DEFAULT_DATA_ROOT / "DomainNet"
IMAGENET_DIR = DEFAULT_DATA_ROOT / "ImageNet"

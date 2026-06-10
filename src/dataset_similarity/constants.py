from pathlib import Path

PROJECT_DIR = (Path(__file__) / ".." / ".." / "..").resolve()
CONFIG_DIR = PROJECT_DIR / "configs"
DATA_DIR = PROJECT_DIR / "data"
EMBEDDING_DIR = PROJECT_DIR / "embeddings"

DOMAINNET_DIR = DATA_DIR / "DomainNet"
IMAGENET_DIR = DATA_DIR / "ImageNet"
COCO_DIR = DATA_DIR / "COCO"

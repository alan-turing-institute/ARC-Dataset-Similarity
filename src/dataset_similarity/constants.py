from pathlib import Path

PROJECT_DIR = (Path(__file__) / ".." / ".." / "..").resolve()
CONFIG_DIR = PROJECT_DIR / "configs"
DEFAULT_DATA_ROOT = PROJECT_DIR / "data"
DEFAULT_EMBEDDING_DIR = PROJECT_DIR / "embeddings"

DOMAINNET_DIR = DEFAULT_DATA_ROOT / "DomainNet"
IMAGENET_DIR = DEFAULT_DATA_ROOT / "ImageNet"
MLFLOW_TRACKING_URI = (
    "https://arc1-turing-mlflow.niceground-2b2fd95b.uksouth.azurecontainerapps.io/"
)

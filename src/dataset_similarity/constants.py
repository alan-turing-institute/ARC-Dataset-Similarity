from pathlib import Path

# Core directories
PROJECT_DIR = (Path(__file__) / ".." / ".." / "..").resolve()
CONFIG_DIR = PROJECT_DIR / "configs"
DATA_DIR = PROJECT_DIR / "data"
EMBEDDING_DIR = PROJECT_DIR / "embeddings"

# Config directories
DATA_CONFIG_DIR = CONFIG_DIR / "data"
EXPERIMENT_CONFIG_DIR = CONFIG_DIR / "experiments"
FINETUNE_CONFIG_DIR = CONFIG_DIR / "finetune"
METRIC_CONFIG_DIR = CONFIG_DIR / "metrics"

# Dataset directories
DOMAINNET_DIR = DATA_DIR / "DomainNet"
IMAGENET_DIR = DATA_DIR / "ImageNet"
COCO_DIR = DATA_DIR / "COCO"

# Output directories
RESULT_DIR = PROJECT_DIR / "results"
EVAL_RESULT_DIR = RESULT_DIR / "eval"
METRICS_RESULT_DIR = RESULT_DIR / "metrics"
TRAINED_MODELS_DIR = PROJECT_DIR / "trained_models"

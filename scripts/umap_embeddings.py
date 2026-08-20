import json
from argparse import ArgumentParser
from glob import glob
from zipfile import Path

import torch
from src.dataset_similarity.dinov3 import DINOv3Classifier
from transformers import AutoModelForImageClassification

from dataset_similarity.data.utils import load_dataset_from_config

TRAINED_MODELS_DIR = (Path(__file__) / ".." / ".." / "trained_models").resolve()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_init(config: dict, num_labels: int) -> AutoModelForImageClassification:
    modelCls = AutoModelForImageClassification
    if config["model_args"]["pretrained_model_name_or_path"].startswith(
        "facebook/dinov3"
    ):
        modelCls = DINOv3Classifier
    return modelCls.from_pretrained(
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        **config["model_args"],
    )


def load_model_from_config(cfg: dict):
    """
    Load a model from a config file.
    """
    checkpoint_dirs = glob(str(TRAINED_MODELS_DIR / "checkpoint*"))[-1]
    model_safetensors = checkpoint_dirs / "model.safetensors"

    if not model_safetensors.exists():
        err_msg = f"Model file not found: {model_safetensors}"
        raise FileNotFoundError(err_msg)

    model = model_init(cfg, num_labels=cfg["dataset_args"]["num_labels"])
    model.load_state_dict(torch.load(model_safetensors, map_location=DEVICE))


def load_from_config(cfg: dict):
    """
    Load a dataset and model from a config file.
    """
    dataset = load_dataset_from_config(cfg)
    model = load_model_from_config(cfg, dataset.num_labels)
    return dataset, model


def main(config_path: str):
    """
    Load a dataset from a config file and plot UMAP embeddings.
    """

    with open(config_path) as f:
        cfg = json.load(f)

    dataset, model = load_from_config(cfg)

    print(dataset, model)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("config_path", type=str, help="Path to the dataset config file")
    parsed_args = args.parse_args()
    main(parsed_args.config_path)

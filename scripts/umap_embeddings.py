import json
from argparse import ArgumentParser
from glob import glob
from zipfile import Path
from dataset_similarity.utils import load_yaml_from_path

import torch
from dataset_similarity.dinov3 import DINOv3Classifier
from transformers import AutoModelForImageClassification, AutoImageProcessor

from dataset_similarity.data.utils import load_dataset_from_config

from dataset_similarity.constants import TRAINED_MODELS_DIR, DATA_CONFIG_DIR, DATA_DIR, FINETUNE_CONFIG_DIR

def get_datasets_from_configs(cfg_name, return_store=False):
    # load_config
    config_path = FINETUNE_CONFIG_DIR / f"{cfg_name}.yaml"
    config = load_yaml_from_path(config_path)

    # load evaluation data
    eval_data_config: dict[str, str | dict] = load_yaml_from_path(
        DATA_CONFIG_DIR / f"{config['test_data_config']}.yaml"
    )
    eval_dataset = load_dataset_from_config(eval_data_config)

    if return_store is False:
        return eval_dataset

    # load data store - create binary label but do not otherwise filter down
    data_store_config: dict[str, str | dict] = load_yaml_from_path(
        DATA_CONFIG_DIR / "coco_data_store.yaml"
    )
    if eval_dataset.positive_superclass is not None:
        data_store_config["kwargs"][
            "positive_superclass"
        ] = eval_dataset.positive_superclass
    else:
        label_to_class_map = {
            label: cat for cat, label in eval_dataset.class_to_label_map.items()
        }
        data_store_config["kwargs"]["positive_class"] = [
            label_to_class_map[label] for label in eval_dataset.positive_class
        ]
    data_store_config["kwargs"]["multi_label"] = eval_dataset.multi_label
    data_store = load_dataset_from_config(data_store_config)

    return eval_dataset, data_store


def main(args):
    eval_datasets = []
    if len(args.configs) == 1:
        eval_dataset, data_store = get_datasets_from_configs(args.configs[0], return_store=True)
        eval_datasets.append(eval_dataset)
    else:
        eval_dataset, data_store = get_datasets_from_configs(args.configs[0], return_store=True)
        eval_datasets.append(eval_dataset)
        for cfg_name in args.configs[1:]:
            eval_datasets.append(get_datasets_from_configs(cfg_name, return_store=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UMAP Embeddings Script")
    parser.add_argument(
        "configs",
        type=str,
        nargs="+",
        help="configs to plot against the store UMAP embeddings",
    )
    args = parser.parse_args()

    main(args)

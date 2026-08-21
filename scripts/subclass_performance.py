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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(cfg_name: str) -> None:
    # load_config
    config_path = FINETUNE_CONFIG_DIR / f"{cfg_name}.yaml"
    config = load_yaml_from_path(config_path)

    # Load processor
    processor = AutoImageProcessor.from_pretrained(
        **config["model_args"],
    )

    # get trained model path
    original_model_name = config["model_args"]["pretrained_model_name_or_path"]
    model_dir = TRAINED_MODELS_DIR / cfg_name
    if (model_dir / "optimal_hparams.yaml").exists():
        base_dir = model_dir / "best_model"  # symlink to winning sweep trial
    else:
        base_dir = model_dir / "trained_model"
    checkpoints = [int(x.stem.split("-")[1]) for x in base_dir.glob("checkpoint-*")]
    checkpoint_path = base_dir / f"checkpoint-{max(checkpoints)}"
    config["model_args"]["pretrained_model_name_or_path"] = checkpoint_path

    # load_model
    modelCls = AutoModelForImageClassification
    if original_model_name.startswith("facebook/dinov3"):
        modelCls = DINOv3Classifier
    model = modelCls.from_pretrained(**config["model_args"])
    model = model.eval()

    # load evaluation data
    eval_data_config: dict[str, str | dict] = load_yaml_from_path(
        DATA_CONFIG_DIR / f"{config['test_data_config']}.yaml"
    )
    eval_dataset = load_dataset_from_config(eval_data_config)
    eval_dataset.embedding = None  # remove embedding from dataset

    # load data store - create binary label but do not otherwise filter down
    data_store_config: dict[str, str | dict] = load_yaml_from_path(
        DATA_CONFIG_DIR / "coco_data_store.yaml"
    )
    if eval_dataset.positive_superclass is not None:
        data_store_config["kwargs"]["positive_superclass"] = (
            eval_dataset.positive_superclass
        )
    else:
        label_to_class_map = {
            label: cat for cat, label in eval_dataset.class_to_label_map.items()
        }
        data_store_config["kwargs"]["positive_class"] = [
            label_to_class_map[label] for label in eval_dataset.positive_class
        ]
    data_store_config["kwargs"]["multi_label"] = eval_dataset.multi_label
    data_store = load_dataset_from_config(data_store_config)
    data_store.embedding = None  # remove embedding from dataset


    drop_subclasses = eval_data_config['kwargs'].get("drop_subclasses", None)
    print(eval_data_config['kwargs'])
    if drop_subclasses is None:
        raise ValueError(
            "drop_subclasses must be specified in the config file for this script"
        )
    else:
        print(data_store.data)
        

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("config_name", type=str, help="Name of the dataset config file")
    parsed_args = args.parse_args()
    main(cfg_name = parsed_args.config_name)
